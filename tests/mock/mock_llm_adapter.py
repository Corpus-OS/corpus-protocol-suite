# SPDX-License-Identifier: Apache-2.0
"""
Mock LLM adapter used in Corpus SDK example scripts and conformance tests.

Implements BaseLLMAdapter methods for demonstration purposes only.
Simulates latency, token counting, streaming behavior, stop sequences, max_tokens,
deadline semantics, and deterministic behavior for tests.

Key properties:
- Deterministic given the same inputs (including ctx.request_id)
- Shared planning path for complete() and stream() so final text matches
- Default failure_rate=0.0 for conformance (can be raised for demos)
- Deterministic health where ctx.attrs["health"] can force degraded/error

Notes (contract alignment):
- Deadlines are enforced via SimpleDeadline even in "thin" mode by default.
- Uses BaseLLMAdapter's normalized DeadlineExceeded semantics (no custom codes).
- Allows passing BaseLLMAdapter infra knobs via dataclass fields.
- Role validation is intentionally permissive to avoid blocking future protocol roles.
- Tool calling is supported and exercised deterministically when tools are provided.

Conformance-focused behavior:
- Tool-call usage accounting includes tool call payload (name + JSON args) so completion_tokens
  are non-zero for tool-calling turns.
- Enforces max_tool_calls_per_turn (capabilities ↔ behavior alignment).
- Optional strict_tool_choice mode to reject tool_choice names not present in tools.
- ToolCall IDs are deterministic (derived from stable hashes), preserving determinism promise.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from corpus_sdk.llm.llm_base import (
    BaseLLMAdapter,
    Cache,
    CircuitBreaker,
    DeadlinePolicy,
    LLMCapabilities,
    LLMCompletion,
    LLMChunk,
    MetricsSink,
    NoopDeadline,
    OperationContext as LLMContext,
    RateLimiter,
    SimpleDeadline,
    TokenUsage,
    ToolCall,
    ToolCallFunction,
    # normalized error taxonomy
    BadRequest,
    ResourceExhausted,
    Unavailable,
)

# Demo-only helpers (not required by tests)
try:  # guard so import doesn’t explode if examples package isn’t present
    from examples.common.ctx import make_ctx
    from examples.common.printing import print_json, print_kv, box
except Exception:  # pragma: no cover - demo convenience only
    make_ctx = None
    print_json = None
    print_kv = None
    box = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

# Permissive to avoid blocking future protocol roles (e.g., tool/function/developer).
_ALLOWED_ROLES = {"system", "user", "assistant", "tool", "function", "developer"}


def _stable_seed(*parts: str) -> int:
    """Stable 48-bit seed from a list of string parts."""
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    # use 48 bits to keep randint fast & stable
    return int(h[:12], 16)


def _tokenize(s: str) -> List[str]:
    # Ultra-simple tokenizer: whitespace split
    return s.split()


def _join_tokens(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def _approx_usage(prompt_text: str, completion_text: str) -> TokenUsage:
    p = len(_tokenize(prompt_text))
    c = len(_tokenize(completion_text))
    return TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


def _tool_calls_usage_text(tool_calls: List[ToolCall]) -> str:
    """
    Represent tool calls as a synthetic completion payload for usage accounting.

    This keeps tool-calling turns from reporting ~0 completion tokens, which is
    often unrealistic compared to real providers that "generate" tool call JSON.
    """
    parts: List[str] = []
    for tc in tool_calls or []:
        try:
            parts.append(f"[tool_call] {tc.function.name} {tc.function.arguments}")
        except Exception:
            parts.append("[tool_call]")
    return "\n".join(parts)


def _extract_tool_name(tool: Mapping[str, Any]) -> Optional[str]:
    """
    Best-effort extraction of a tool name from a tool definition.

    Supports common shapes:
      - {"type":"function","function":{"name":"foo", ...}}
      - {"name":"foo", ...}
    """
    try:
        fn = tool.get("function")
        if isinstance(fn, Mapping):
            n = fn.get("name")
            if isinstance(n, str) and n:
                return n
        n2 = tool.get("name")
        if isinstance(n2, str) and n2:
            return n2
    except Exception:
        pass
    return None


def _extract_requested_tool_name(tool_choice: Any) -> Optional[str]:
    """
    Best-effort extraction of a requested tool name from tool_choice dicts.

    Common shapes:
      - {"type":"function","function":{"name":"foo"}}
      - {"function":{"name":"foo"}}
      - {"name":"foo"}
    """
    if not isinstance(tool_choice, Mapping):
        return None
    try:
        fn = tool_choice.get("function")
        if isinstance(fn, Mapping):
            n = fn.get("name")
            if isinstance(n, str) and n:
                return n
        # Sometimes nested under {"type":"function","function":{"name":...}}
        if tool_choice.get("type") == "function":
            fn2 = tool_choice.get("function")
            if isinstance(fn2, Mapping):
                n2 = fn2.get("name")
                if isinstance(n2, str) and n2:
                    return n2
        n3 = tool_choice.get("name")
        if isinstance(n3, str) and n3:
            return n3
    except Exception:
        pass
    return None


@dataclass
class MockLLMAdapter(BaseLLMAdapter):
    """A mock LLM adapter for protocol demonstrations & conformance tests."""

    name: str = "mock-llm"
    # Default 0.0 so conformance runs are deterministic and non-flaky.
    failure_rate: float = 0.0

    # Optional strictness: if True, reject tool_choice dict names not present in tools.
    strict_tool_choice: bool = False

    # ---- BaseLLMAdapter configuration passthrough (so tests can exercise core gates) ----
    metrics: Optional[MetricsSink] = None
    mode: str = "thin"
    deadline_policy: Optional[DeadlinePolicy] = None
    breaker: Optional[CircuitBreaker] = None
    cache: Optional[Cache] = None
    limiter: Optional[RateLimiter] = None
    tag_model_in_metrics: bool = True
    cache_ttl_s: int = 60
    # For conformance, check deadlines frequently in streams.
    stream_deadline_check_every_n_chunks: int = 1

    def __post_init__(self) -> None:
        # Cache the most recent caps observed via _do_capabilities so we don't
        # re-fetch inside tool-call enforcement (Base has already fetched caps
        # before calling _do_complete/_do_stream).
        self._last_caps: Optional[LLMCapabilities] = None

        # Ensure BaseLLMAdapter infra is initialized. In "thin" mode, we still
        # enforce deadlines by default using SimpleDeadline to keep
        # capabilities.supports_deadline ↔ behavior aligned.
        if self.deadline_policy is not None:
            effective_deadline_policy: DeadlinePolicy = self.deadline_policy
        else:
            # Default: enforce deadlines. If a caller truly wants no deadlines,
            # they can pass deadline_policy=NoopDeadline().
            effective_deadline_policy = SimpleDeadline()

        super().__init__(
            metrics=self.metrics,
            mode=self.mode,
            deadline_policy=effective_deadline_policy,
            breaker=self.breaker,
            cache=self.cache,
            limiter=self.limiter,
            tag_model_in_metrics=self.tag_model_in_metrics,
            cache_ttl_s=self.cache_ttl_s,
            stream_deadline_check_every_n_chunks=self.stream_deadline_check_every_n_chunks,
        )

    # -----------------------------------------------------------------------
    # Local validation helper
    # -----------------------------------------------------------------------
    def _validate_roles(self, messages: List[Mapping[str, Any]]) -> None:
        """
        Reject unknown roles beyond the base schema checks.

        Kept permissive to avoid false negatives as the protocol evolves
        (e.g., tool/function/developer roles).
        """
        for m in messages:
            role = str(m.get("role", ""))
            if role and role not in _ALLOWED_ROLES:
                raise BadRequest(f"unknown role: {role!r}")

    # -----------------------------------------------------------------------
    # Shared planning for complete() and stream()
    # -----------------------------------------------------------------------
    def _make_rng(
        self,
        *,
        model: Optional[str],
        system_message: Optional[str],
        messages: List[Mapping[str, Any]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        stop_sequences: Optional[List[str]],
        tool_names: Sequence[str],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        ctx: Optional[LLMContext],
    ) -> random.Random:
        """Build a deterministic RNG that is identical for complete + stream."""
        seed = _stable_seed(
            str(model or "mock-model"),
            str(system_message or ""),
            repr(messages),
            str(max_tokens),
            str(temperature),
            str(top_p),
            str(frequency_penalty),
            str(presence_penalty),
            ",".join(stop_sequences or []),
            ",".join(tool_names),
            repr(tool_choice),
            str(getattr(ctx, "request_id", "")),
        )
        return random.Random(seed)

    def _build_base_tokens(self, *, last_content: str, model: Optional[str]) -> List[str]:
        """
        Core token plan before sampling / temperature effects.
        Always adds a deterministic suffix so it's not pure echo.
        """
        base = (last_content.strip() or "ok")
        words = _tokenize(base)
        suffix = ["(mock)", f"[{model or 'mock-model'}]"]
        return words + suffix

    def _apply_temperature(
        self,
        gen_tokens: List[str],
        *,
        temperature: Optional[float],
        rnd: random.Random,
    ) -> List[str]:
        """Deterministic temperature effects (dup/drop some tokens)."""
        t = float(temperature or 0.0)
        if t <= 0.0:
            return gen_tokens

        mutated: List[str] = []
        for tok in gen_tokens:
            r = rnd.random()
            dup_thresh = min(0.05 * t, 0.2)
            drop_thresh = min(0.10 * t, 0.4)
            if r < dup_thresh:
                mutated.extend([tok, tok])
            elif r < drop_thresh:
                continue
            else:
                mutated.append(tok)

        return mutated or gen_tokens

    def _apply_max_and_stops(
        self,
        gen_tokens: List[str],
        *,
        max_tokens: Optional[int],
        stop_sequences: Optional[List[str]],
    ) -> tuple[List[str], bool, bool]:
        """
        Apply max_tokens then stop sequences. Returns:
        (final_tokens, cut_by_max, cut_by_stop).

        Also ensures that final_tokens is never empty, to satisfy tests that
        aggregate text must be non-empty.
        """
        tokens = list(gen_tokens)
        cut_by_max = False
        cut_by_stop = False

        # max_tokens
        if max_tokens is not None:
            lim = max(0, int(max_tokens))
            if len(tokens) > lim:
                cut_by_max = True
            tokens = tokens[:lim]

        # stop sequences (string-based, then re-tokenize)
        if stop_sequences:
            completion_text = _join_tokens(tokens)
            cut_at: Optional[int] = None
            for s in stop_sequences:
                if not s:
                    continue
                idx = completion_text.find(s)
                if idx != -1:
                    cut_at = idx if cut_at is None else min(cut_at, idx)
            if cut_at is not None:
                completion_text = completion_text[:cut_at].rstrip()
                tokens = _tokenize(completion_text)
                cut_by_stop = True

        # Ensure non-empty final tokens to keep aggregate non-empty.
        if not tokens:
            tokens = ["(mock)"]
            cut_by_max = False
            cut_by_stop = False

        return tokens, cut_by_max, cut_by_stop

    def _should_emit_tool_call(
        self,
        *,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        last_user_content: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide whether to emit a tool call, and (optionally) which tool name.

        Rules (V1-friendly, minimal):
          - If no tools: no tool call.
          - tool_choice == "none": no tool call.
          - tool_choice == "required": emit a tool call (choose first tool).
          - tool_choice is a dict with a tool name: emit that tool if present.
              * If strict_tool_choice=True and requested name is not in tools: BadRequest.
              * Else: still emit requested name (contract-valid).
          - tool_choice in {None, "auto"}: emit only if the user message contains a trigger.
        """
        if not tools:
            return False, None

        # tool_choice string handling
        if isinstance(tool_choice, str):
            if tool_choice == "none":
                return False, None
            if tool_choice == "required":
                return True, _extract_tool_name(tools[0]) or "mock_tool"
            # "auto" -> trigger-based below

        # tool_choice dict handling (best-effort)
        requested = _extract_requested_tool_name(tool_choice)
        if requested:
            names = {_extract_tool_name(t) for t in tools}
            if requested in names:
                return True, requested
            if self.strict_tool_choice:
                raise BadRequest(f"tool_choice requested unknown tool: {requested!r}")
            return True, requested

        # Auto/None: trigger-based
        lc = (last_user_content or "").lower()
        trigger = ("call:" in lc) or ("tool:" in lc) or ("use_tool" in lc)
        if trigger:
            return True, _extract_tool_name(tools[0]) or "mock_tool"
        return False, None

    @staticmethod
    def _deterministic_tool_call_id(
        *,
        tool_name: str,
        args_json: str,
        request_id: Optional[str],
    ) -> str:
        """
        Deterministic tool call ID derived from stable inputs.

        This avoids flakiness in conformance tests and preserves the module's
        determinism promise (given the same inputs).
        """
        payload = f"{tool_name}\n{args_json}\n{request_id or ''}"
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"call_{h}"

    def _make_tool_call(
        self,
        *,
        tool_name: str,
        last_user_content: str,
        rnd: random.Random,
        ctx: Optional[LLMContext],
    ) -> ToolCall:
        """
        Construct a deterministic ToolCall object with JSON string arguments.
        """
        # Extract an "input" payload deterministically. If the message contains "call:",
        # use what follows; else use a shortened version of the message.
        extracted = last_user_content
        lc = last_user_content.lower()
        idx = lc.find("call:")
        if idx != -1:
            extracted = last_user_content[idx + len("call:") :].strip() or last_user_content

        # Add a tiny deterministic nonce to avoid identical arguments across distinct runs
        # when last_user_content is empty.
        nonce = rnd.randint(0, 1_000_000)

        args_obj = {"input": extracted, "nonce": nonce}
        args_json = json.dumps(args_obj, separators=(",", ":"), sort_keys=True)

        tc_id = self._deterministic_tool_call_id(
            tool_name=tool_name,
            args_json=args_json,
            request_id=getattr(ctx, "request_id", None) if ctx else None,
        )

        return ToolCall(
            id=tc_id,
            type="function",
            function=ToolCallFunction(name=tool_name, arguments=args_json),
        )

    def _plan_response(
        self,
        *,
        messages: List[Mapping[str, Any]],
        model: Optional[str],
        system_message: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        stop_sequences: Optional[List[str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        ctx: Optional[LLMContext],
    ) -> tuple[str, str, str, List[ToolCall], str]:
        """
        Compute prompt_text, completion_text, finish_reason, tool_calls, usage_completion_text.

        - completion_text is what we return to callers (often "" for tool_calls turns)
        - usage_completion_text is what we use to approximate completion token usage
          (tool call payload for tool_calls turns; completion_text for normal turns)
        """
        # Build prompt representation
        prompt_parts: List[str] = []
        if system_message:
            prompt_parts.append(f"[system] {system_message}")
        for m in messages:
            prompt_parts.append(f"[{m.get('role','')}] {m.get('content','')}")
        prompt_text = "\n".join(prompt_parts)

        last_content = str(messages[-1].get("content", "")) if messages else ""

        tool_names: List[str] = []
        if tools:
            for t in tools:
                n = _extract_tool_name(t)
                if n:
                    tool_names.append(n)

        rnd = self._make_rng(
            model=model,
            system_message=system_message,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            tool_names=tool_names,
            tool_choice=tool_choice,
            ctx=ctx,
        )

        # Tool call planning
        emit_tool, tool_name = self._should_emit_tool_call(
            tools=tools,
            tool_choice=tool_choice,
            last_user_content=last_content,
        )
        if emit_tool:
            chosen_name = tool_name or (tool_names[0] if tool_names else "mock_tool")
            tc = self._make_tool_call(
                tool_name=chosen_name,
                last_user_content=last_content,
                rnd=rnd,
                ctx=ctx,
            )
            tool_calls = [tc]

            completion_text = ""
            usage_completion_text = _tool_calls_usage_text(tool_calls)
            return prompt_text, completion_text, "tool_calls", tool_calls, usage_completion_text

        # Normal text planning
        base_tokens = self._build_base_tokens(last_content=last_content, model=model)
        gen_tokens = self._apply_temperature(base_tokens, temperature=temperature, rnd=rnd)

        final_tokens, cut_by_max, cut_by_stop = self._apply_max_and_stops(
            gen_tokens,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )

        completion_text = _join_tokens(final_tokens)

        finish_reason = "stop"
        if cut_by_max and not cut_by_stop:
            finish_reason = "length"

        return prompt_text, completion_text, finish_reason, [], completion_text

    def _maybe_simulate_failure(
        self,
        *,
        messages: List[Mapping[str, Any]],
        model: Optional[str],
        system_message: Optional[str],
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        stop_sequences: Optional[List[str]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        ctx: Optional[LLMContext],
    ) -> None:
        """
        Optional failure injection for demos/tests. By default failure_rate=0.0
        so conformance runs are stable.
        """
        if self.failure_rate <= 0.0:
            return

        # Keep failure injection deterministic with the same seed inputs.
        tool_names: List[str] = []
        if tools:
            for t in tools:
                n = _extract_tool_name(t)
                if n:
                    tool_names.append(n)

        rnd = self._make_rng(
            model=model,
            system_message=system_message,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            tool_names=tool_names,
            tool_choice=tool_choice,
            ctx=ctx,
        )

        last_content = str(messages[-1].get("content", "")) if messages else ""
        if rnd.random() < self.failure_rate:
            if "overload" in last_content.lower():
                raise Unavailable("Mocked service overload", retry_after_ms=2000)
            raise ResourceExhausted("Mocked rate limit", retry_after_ms=1000)

    def _enforce_max_tool_calls_per_turn(self, tool_calls: List[ToolCall]) -> None:
        """
        Enforce caps.max_tool_calls_per_turn for capability↔behavior alignment.

        Uses _last_caps populated by _do_capabilities (Base has already called
        capabilities() before invoking _do_complete/_do_stream), avoiding extra
        async calls and overhead on hot paths.
        """
        if not tool_calls:
            return

        caps = self._last_caps
        if caps is None:
            # Fallback: mock should still behave sensibly if called out-of-band.
            return

        lim = caps.max_tool_calls_per_turn
        if lim is not None and len(tool_calls) > int(lim):
            raise BadRequest("too many tool calls in one turn")

    # -----------------------------------------------------------------------
    # Completion
    # -----------------------------------------------------------------------
    async def _do_complete(
        self,
        *,
        messages: List[Mapping[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[LLMContext] = None,
    ) -> LLMCompletion:
        """
        Pretend to complete a chat turn with deterministic behavior.

        If tools are provided and tool_choice/trigger demands it, emits tool_calls
        and sets finish_reason="tool_calls".
        """
        self._validate_roles(messages)

        self._maybe_simulate_failure(
            messages=messages,
            model=model,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            ctx=ctx,
        )

        prompt_text, completion_text, finish_reason, tool_calls, usage_completion_text = (
            self._plan_response(
                messages=messages,
                model=model,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=stop_sequences,
                tools=tools,
                tool_choice=tool_choice,
                ctx=ctx,
            )
        )

        self._enforce_max_tool_calls_per_turn(tool_calls)

        await asyncio.sleep(0.03)

        usage = _approx_usage(prompt_text, usage_completion_text)

        return LLMCompletion(
            text=completion_text,
            model=model or "mock-model",
            model_family="mock",
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    # -----------------------------------------------------------------------
    # Streaming
    # -----------------------------------------------------------------------
    async def _do_stream(
        self,
        *,
        messages: List[Mapping[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        ctx: Optional[LLMContext] = None,
    ) -> AsyncIterator[LLMChunk]:
        """
        Simulate token streaming with progressive usage and final sentinel chunk.

        If a tool call is planned, emits at least one non-final chunk and then a
        final chunk with tool_calls populated (minimum viable tool-calling stream).
        """
        self._validate_roles(messages)

        self._maybe_simulate_failure(
            messages=messages,
            model=model,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            ctx=ctx,
        )

        prompt_text, completion_text, finish_reason, tool_calls, usage_completion_text = (
            self._plan_response(
                messages=messages,
                model=model,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=stop_sequences,
                tools=tools,
                tool_choice=tool_choice,
                ctx=ctx,
            )
        )

        model_name = model or "mock-model"

        if finish_reason == "tool_calls" and tool_calls:
            self._enforce_max_tool_calls_per_turn(tool_calls)

            await asyncio.sleep(0.01)
            usage_so_far = _approx_usage(prompt_text, "")
            yield LLMChunk(
                text="",
                is_final=False,
                model=model_name,
                usage_so_far=usage_so_far,
                tool_calls=[],
            )

            final_usage = _approx_usage(prompt_text, usage_completion_text)
            yield LLMChunk(
                text="",
                is_final=True,
                model=model_name,
                usage_so_far=final_usage,
                tool_calls=tool_calls,
            )
            return

        final_tokens = _tokenize(completion_text)
        if not final_tokens:
            final_tokens = ["(mock)"]

        emitted_tokens: List[str] = []
        for i, tok in enumerate(final_tokens, start=1):
            emitted_tokens.append(tok)
            partial_text = _join_tokens(emitted_tokens)
            usage_so_far = _approx_usage(prompt_text, partial_text)

            await asyncio.sleep(0.01)

            yield LLMChunk(
                text=tok + (" " if i < len(final_tokens) else ""),
                is_final=False,
                model=model_name,
                usage_so_far=usage_so_far,
            )

        final_usage = _approx_usage(prompt_text, _join_tokens(emitted_tokens))
        yield LLMChunk(
            text="",
            is_final=True,
            model=model_name,
            usage_so_far=final_usage,
        )

    # -----------------------------------------------------------------------
    # Token counting
    # -----------------------------------------------------------------------
    async def _do_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        ctx: Optional[LLMContext] = None,
    ) -> int:
        """Mock token counting with a word-based approximation (+3 overhead)."""
        if not text:
            return 0
        await asyncio.sleep(0.005)
        return len(_tokenize(text)) + 3

    # -----------------------------------------------------------------------
    # Capabilities & Health
    # -----------------------------------------------------------------------
    async def _do_capabilities(self) -> LLMCapabilities:
        """Report mock model capabilities."""
        caps = LLMCapabilities(
            server="mock",
            version="1.0.0",
            model_family="mock",
            max_context_length=4096,
            supports_streaming=True,
            supports_roles=True,
            supports_json_output=False,
            supports_tools=True,
            supports_parallel_tool_calls=False,
            supports_tool_choice=True,
            max_tool_calls_per_turn=4,
            idempotent_writes=False,
            supports_multi_tenant=True,
            supports_system_message=True,
            supports_deadline=True,
            supports_count_tokens=True,
            supported_models=("mock-model", "mock-model-pro"),
        )
        # Cache for reuse during _do_complete/_do_stream without re-fetching.
        self._last_caps = caps
        return caps

    async def _do_health(self, *, ctx: Optional[LLMContext] = None) -> Mapping[str, object]:
        """
        Mock health check.

        Deterministic shape; allow ctx.attrs["health"] to force degraded/error.
        - health="degraded" -> ok=False, status="degraded"
        - health="error"    -> ok=False, status="error"
        - anything else     -> ok=True,  status="healthy"
        """
        status_hint = (ctx and ctx.attrs.get("health")) or "ok"
        if status_hint == "degraded":
            return {"ok": False, "status": "degraded", "server": "mock", "version": "1.0.0"}
        if status_hint == "error":
            return {"ok": False, "status": "error", "server": "mock", "version": "1.0.0"}
        return {"ok": True, "status": "healthy", "server": "mock", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Demo usage (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual demo only
    async def _demo() -> None:
        if not (make_ctx and print_json and print_kv and box):
            print("Demo helpers not available; run within examples environment.")
            return

        random.seed(42)  # Deterministic for reproducible demos
        box("MockLLMAdapter Demo")
        adapter = MockLLMAdapter(failure_rate=0.3)  # Higher failure rate for demo
        ctx = make_ctx(LLMContext, tenant="demo", request_id="demo-req-1")

        print("\n=== CAPABILITIES ===")
        caps = await adapter.capabilities()
        print_json(caps.__dict__)

        print("\n=== HEALTH CHECK ===")
        health = await adapter.health(ctx=ctx)
        print_kv(health)

        print("\n=== COMPLETE ===")
        try:
            result = await adapter.complete(
                messages=[{"role": "user", "content": "hello world please show me something nice"}],
                model="mock-model",
                max_tokens=6,
                stop_sequences=["something"],
                system_message="You are helpful.",
                ctx=ctx,
            )
            print_kv({"Output": result.text, "FinishReason": result.finish_reason})
            print_json(result.usage.__dict__)
        except Exception as e:
            print_kv({"Error": str(e), "Type": type(e).__name__})

        print("\n=== TOOL CALLING ===")
        tools = [
            {"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}},
        ]
        result_tc = await adapter.complete(
            messages=[{"role": "user", "content": "call: please echo this back"}],
            tools=tools,
            tool_choice="auto",
            ctx=ctx,
        )
        print_kv({"FinishReason": result_tc.finish_reason, "ToolCalls": len(result_tc.tool_calls)})
        if result_tc.tool_calls:
            print_json(
                {
                    "tool_call": {
                        "id": result_tc.tool_calls[0].id,
                        "name": result_tc.tool_calls[0].function.name,
                        "args": result_tc.tool_calls[0].function.arguments,
                    }
                }
            )
            print_json({"usage": result_tc.usage.__dict__})

        print("\n=== STREAM ===")
        try:
            async for chunk in adapter.stream(
                messages=[{"role": "user", "content": "stream this message and then STOP please"}],
                model="mock-model-pro",
                temperature=0.7,
                stop_sequences=["STOP"],
                ctx=ctx,
            ):
                if chunk.is_final:
                    print("\n[final]", chunk.usage_so_far.__dict__)
                else:
                    print(chunk.text, end="", flush=True)
            print("\n[done]")
        except Exception as e:
            print(f"\nStream error: {e}")

        print("\n=== STREAM TOOL CALLING ===")
        try:
            async for chunk in adapter.stream(
                messages=[{"role": "user", "content": "call: stream a tool call"}],
                tools=tools,
                tool_choice="required",
                ctx=ctx,
            ):
                if chunk.is_final:
                    print("\n[final tool_calls]", [tc.function.name for tc in chunk.tool_calls])
                    print("[final usage]", chunk.usage_so_far.__dict__)
                else:
                    print(".", end="", flush=True)
            print("\n[done]")
        except Exception as e:
            print(f"\nStream tool error: {e}")

        print("\n=== TOKEN COUNTING ===")
        try:
            count = await adapter.count_tokens("This is a test sentence", ctx=ctx)
            print_kv({"Text": "This is a test sentence", "Tokens": count})
            count_empty = await adapter.count_tokens("", ctx=ctx)
            print_kv({"Text": "<empty>", "Tokens": count_empty})
        except Exception as e:
            print_kv({"Error": str(e)})

    asyncio.run(_demo())
