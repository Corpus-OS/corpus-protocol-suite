# corpus_sdk/llm/framework_adapters/common/llm_translation.py
# SPDX-License-Identifier: Apache-2.0
"""
Framework-agnostic LLM → Framework translation layer.

Purpose
-------
Provide a high-level orchestration and translation layer between:

- The Corpus LLM Protocol V1 (`LLMProtocolV1` / `BaseLLMAdapter`), and
- Framework-specific chat integrations (LangChain, LlamaIndex, SK, AutoGen, CrewAI, custom).

This module is intentionally *framework-neutral* and focuses on:

- Translating framework-level message objects into Corpus wire messages
  ({role, content}) via NormalizedMessage
- Translating `LLMCompletion` / `LLMChunk` / capabilities / health responses
  back into framework-facing shapes
- Providing sync + async APIs, including streaming via a sync bridge
- Attaching rich error context for observability while delegating all hard
  policies (deadlines, breakers, caching, rate limiting) to the adapter

Context translation
-------------------
This module does **not** parse framework configs directly. Instead:

- `corpus_sdk.core.context_translation` is responsible for taking
  framework-native contexts (LangChain RunnableConfig, LlamaIndex CallbackManager,
  etc.) and producing a core `OperationContext` type.
- Callers pass either an LLM `OperationContext` or a simple dict-like context
  into the methods here; we normalize that via `ctx_from_dict` into the core
  context, and then adapt into the LLM protocol's `OperationContext`.

Streaming
---------
For streaming completions, this module exposes:

- An async API that yields translated framework chunks, and
- A sync API that wraps the async iterator via `SyncStreamBridge`, preserving
  proper cancellation and error propagation.

Registry
--------
A comprehensive registry lets you register per-framework LLM translators:

- `register_llm_translator("my_framework", factory)`
- `get_llm_translator_factory("my_framework")`
- `create_llm_translator("my_framework", adapter, ...)`

This makes it straightforward to plug in framework-specific behaviors while
reusing the common orchestration logic here.

Post-processing overrides
-------------------------
Per-request LLM completion post-processing can be controlled via
`OperationContext.attrs` using the following keys:

- `llm_postprocess_enabled`: bool-like, enable/disable post-processing
- `llm_postprocess_safety_filter`: bool-like, enable/disable safety filtering
- `llm_postprocess_json_repair`: bool-like, enable/disable JSON repair
- `llm_postprocess_output_format`: "text" | "json" | "markdown"
- `llm_postprocess_max_length`: int-like, maximum output length in characters
"""

from __future__ import annotations

import json
import logging
import re
import time
import threading
from dataclasses import asdict, dataclass, field, replace
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from corpus_sdk.llm.llm_base import (
    BadRequest,
    LLMCapabilities,
    LLMChunk,
    LLMCompletion,
    LLMProtocolV1,
    OperationContext,
)
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    from_generic_dict,
    to_corpus,
)
from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.context_translation import from_dict as ctx_from_dict
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.core.sync_bridge import SyncStreamBridge

LOG = logging.getLogger(__name__)

R = TypeVar("R")


# =============================================================================
# Enhanced Token Counting Configuration
# =============================================================================


@dataclass(frozen=True)
class TokenCountingConfig:
    """
    Configuration for sophisticated token counting strategies.

    Different LLMs and frameworks have varying requirements for how
    messages should be formatted for accurate token counting.

    Attributes:
        format_strategy:
            How to format messages for token counting:
            - "simple": "role: content" format (default, basic but fast)
            - "openai_chatml": OpenAI ChatML template formatting
            - "anthropic": Anthropic's specific message formatting
            - "custom": Use custom_format_template

        custom_format_template:
            Template string for custom formatting. Use {role} and {content}
            placeholders. Example: "<|im_start|>{role}\\n{content}<|im_end|>\\n"

        include_system_in_messages:
            When True, the system message (if any) is embedded into the
            formatted text for token counting. When False, system messages
            are omitted from the formatted text (callers may count them
            separately if desired).

        add_special_tokens:
            Whether to account for special tokens (BOS/EOS) in count.
            Currently used to add token overhead for formatting templates.
    """

    format_strategy: str = "simple"
    custom_format_template: Optional[str] = None
    include_system_in_messages: bool = True
    add_special_tokens: bool = True

    def __post_init__(self) -> None:
        """Validate token counting configuration."""
        valid_strategies = {"simple", "openai_chatml", "anthropic", "custom"}
        if self.format_strategy not in valid_strategies:
            raise ValueError(
                f"format_strategy must be one of {valid_strategies}, got {self.format_strategy}"
            )
        if self.format_strategy == "custom" and not self.custom_format_template:
            raise ValueError(
                "custom_format_template is required when format_strategy='custom'"
            )


# =============================================================================
# Enhanced Tool Validation
# =============================================================================


class ToolValidationError(BadRequest):
    """
    Specialized BadRequest subtype for tool schema and tool_choice validation.

    This keeps external behavior compatible with BadRequest while allowing
    downstream systems to distinguish tool-related issues if desired.
    """


class ToolValidator:
    """
    Enhanced validation for tool schemas to ensure compatibility with LLM protocols.

    Validates tool definitions according to common LLM provider expectations
    and provides helpful error messages for framework integrators.
    """

    REQUIRED_FUNCTION_KEYS = {"name", "description", "parameters"}
    REQUIRED_PARAMETERS_KEYS = {"type", "properties"}

    @classmethod
    def validate_tool_schema(cls, tool: Dict[str, Any]) -> None:
        """
        Validate a single tool schema for LLM compatibility.

        Raises ToolValidationError with detailed error messages if validation fails.
        """
        if not isinstance(tool, dict):
            raise ToolValidationError(
                "Tool must be a dictionary",
                code="BAD_TOOL_SCHEMA",
                details={"received_type": type(tool).__name__},
            )

        # Check tool type
        tool_type = tool.get("type")
        if tool_type != "function":
            raise ToolValidationError(
                f"Tool type must be 'function', got {tool_type!r}",
                code="BAD_TOOL_TYPE",
                details={"supported_types": ["function"], "received_type": tool_type},
            )

        # Check function definition
        function_def = tool.get("function")
        if not isinstance(function_def, dict):
            raise ToolValidationError(
                "Tool must contain a 'function' dictionary",
                code="BAD_TOOL_FUNCTION",
                details={"received_type": type(function_def).__name__},
            )

        # Validate required function keys
        missing_keys = cls.REQUIRED_FUNCTION_KEYS - set(function_def.keys())
        if missing_keys:
            raise ToolValidationError(
                f"Tool function missing required keys: {missing_keys}",
                code="BAD_TOOL_FUNCTION_KEYS",
                details={
                    "required_keys": sorted(cls.REQUIRED_FUNCTION_KEYS),
                    "missing_keys": sorted(missing_keys),
                },
            )

        # Validate function name
        name = function_def.get("name", "")
        if not isinstance(name, str) or not name.strip():
            raise ToolValidationError(
                "Tool function name must be a non-empty string",
                code="BAD_TOOL_NAME",
                details={"received_value": name},
            )

        # Validate parameters schema
        parameters = function_def.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ToolValidationError(
                "Tool function parameters must be a dictionary",
                code="BAD_TOOL_PARAMETERS",
                details={"received_type": type(parameters).__name__},
            )

        # Validate required parameters keys
        missing_param_keys = cls.REQUIRED_PARAMETERS_KEYS - set(parameters.keys())
        if missing_param_keys:
            raise ToolValidationError(
                f"Tool parameters missing required keys: {missing_param_keys}",
                code="BAD_TOOL_PARAMETERS_KEYS",
                details={
                    "required_keys": sorted(cls.REQUIRED_PARAMETERS_KEYS),
                    "missing_keys": sorted(missing_param_keys),
                },
            )

        # Validate parameters type
        param_type = parameters.get("type")
        if param_type != "object":
            raise ToolValidationError(
                f"Parameters type must be 'object', got {param_type!r}",
                code="BAD_TOOL_PARAMETERS_TYPE",
                details={"received_type": param_type},
            )

        # Validate properties
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            raise ToolValidationError(
                "Parameters properties must be a dictionary",
                code="BAD_TOOL_PROPERTIES",
                details={"received_type": type(properties).__name__},
            )

    @classmethod
    def validate_tool_choice(cls, tool_choice: Any) -> None:
        """
        Validate tool_choice parameter for LLM compatibility.

        Raises ToolValidationError with detailed error messages if validation fails.
        """
        if tool_choice is None:
            return

        if isinstance(tool_choice, str):
            valid_choices = {"auto", "none", "required"}
            if tool_choice not in valid_choices:
                raise ToolValidationError(
                    f"tool_choice must be one of {valid_choices}, got {tool_choice!r}",
                    code="BAD_TOOL_CHOICE",
                    details={
                        "valid_choices": sorted(valid_choices),
                        "received_value": tool_choice,
                    },
                )
        elif isinstance(tool_choice, dict):
            missing = [key for key in ("type", "function") if key not in tool_choice]
            if missing:
                raise ToolValidationError(
                    "tool_choice dict must contain 'type' and 'function' keys",
                    code="BAD_TOOL_CHOICE_DICT",
                    details={"missing_keys": missing},
                )
        else:
            raise ToolValidationError(
                "tool_choice must be a string or dictionary",
                code="BAD_TOOL_CHOICE_TYPE",
                details={"received_type": type(tool_choice).__name__},
            )


# =============================================================================
# LLM Post-Processing Configuration
# =============================================================================


@dataclass(frozen=True)
class LLMPostProcessingConfig:
    """
    Configuration for LLM result post-processing (MMR-equivalent for LLMs).

    Provides hooks for safety filtering, output formatting, JSON repair,
    and other post-processing operations on LLM completions.

    Attributes:
        enabled:
            Whether to apply post-processing.

        safety_filter:
            Apply safety/content filtering to completions.

        json_repair:
            Attempt to repair malformed JSON in tool calls or content.

        output_format:
            Force specific output format: "text", "json", "markdown".

        max_length:
            Truncate completion if it exceeds this character length.

        custom_processors:
            List of custom processor functions to apply to completions.
    """

    enabled: bool = False
    safety_filter: bool = False
    json_repair: bool = False
    output_format: Optional[str] = None
    max_length: Optional[int] = None
    custom_processors: Optional[List[Callable[[LLMCompletion], LLMCompletion]]] = None

    def __post_init__(self) -> None:
        """Validate post-processing configuration."""
        if self.output_format and self.output_format not in {"text", "json", "markdown"}:
            raise ValueError(
                "output_format must be one of 'text', 'json', 'markdown', "
                f"got {self.output_format}"
            )
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")


# =============================================================================
# Enhanced Metrics Tracking
# =============================================================================


@dataclass
class TranslatorMetrics:
    """
    Metrics collected at the translator level for observability.

    Tracks framework-specific translation patterns and error rates
    for better monitoring and debugging.
    Thread-safe.
    """

    # Translation operation counts
    message_translations: int = 0
    tool_translations: int = 0
    completion_translations: int = 0
    chunk_translations: int = 0

    # Error counts
    translation_errors: int = 0
    validation_errors: int = 0
    normalization_errors: int = 0

    # Timing metrics (in seconds)
    total_translation_time: float = 0.0
    avg_translation_time: float = 0.0

    # Framework-specific usage
    framework_specific_paths_used: int = 0

    # Internal lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_translation(self, duration: float, success: bool = True) -> None:
        """Record a translation operation with timing."""
        with self._lock:
            self.total_translation_time += duration
            self.message_translations += 1
            if not success:
                self.translation_errors += 1
            if self.message_translations > 0:
                self.avg_translation_time = (
                    self.total_translation_time / self.message_translations
                )

    def record_validation_error(self) -> None:
        with self._lock:
            self.validation_errors += 1

    def record_normalization_error(self) -> None:
        with self._lock:
            self.normalization_errors += 1

    def record_framework_path_usage(self) -> None:
        with self._lock:
            self.framework_specific_paths_used += 1

    def record_tool_translation(self) -> None:
        with self._lock:
            self.tool_translations += 1

    def record_completion_translation(self) -> None:
        with self._lock:
            self.completion_translations += 1

    def record_chunk_translation(self) -> None:
        with self._lock:
            self.chunk_translations += 1


# =============================================================================
# Safety Filter Implementation
# =============================================================================


class SafetyFilter:
    """
    Content safety filtering for LLM completions.

    Provides configurable safety checks for LLM outputs including
    profanity filtering, PII detection, and content moderation.
    """

    PROFANITY_PATTERNS = [
        r"\b(asshole|bastard|bitch|damn|fuck|shit)\b",
        r"\b(cunt|dick|piss|whore)\b",
    ]

    PII_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{16}\b",  # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Phone
    ]

    _PROFANITY_REGEX = re.compile("|".join(PROFANITY_PATTERNS), re.IGNORECASE)
    _PII_REGEX = re.compile("|".join(PII_PATTERNS))

    def __init__(self, filter_profanity: bool = True, filter_pii: bool = True) -> None:
        self.filter_profanity = filter_profanity
        self.filter_pii = filter_pii

    def filter_content(self, text: str) -> Tuple[str, List[str]]:
        violations: List[str] = []
        filtered_text = text

        if self.filter_profanity:
            profanity_matches = self._PROFANITY_REGEX.findall(text)
            if profanity_matches:
                violations.append(
                    f"Profanity detected: {', '.join(sorted(set(profanity_matches)))}"
                )
                filtered_text = self._PROFANITY_REGEX.sub("[REDACTED]", filtered_text)

        if self.filter_pii:
            pii_matches = self._PII_REGEX.findall(text)
            if pii_matches:
                violations.append("PII detected")
                filtered_text = self._PII_REGEX.sub("[REDACTED]", filtered_text)

        return filtered_text, violations


# =============================================================================
# JSON Repair Implementation
# =============================================================================


class JSONRepair:
    """
    Robust JSON repair for malformed LLM outputs.

    Handles common JSON formatting issues in LLM completions including
    unclosed brackets, trailing commas, and missing quotes.
    """

    @staticmethod
    def repair_json(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text

        if "{" not in stripped and "[" not in stripped:
            return text

        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

        repaired = stripped

        # Balance braces/brackets
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")

        if open_braces > 0:
            repaired += "}" * open_braces
        elif open_braces < 0:
            repaired = "{" * (-open_braces) + repaired

        if open_brackets > 0:
            repaired += "]" * open_brackets
        elif open_brackets < 0:
            repaired = "[" * (-open_brackets) + repaired

        # Remove trailing commas
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        # Add missing quotes around keys
        repaired = re.sub(
            r"(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
            r'\1 "\2":',
            repaired,
        )

        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

        # Last resort: fix odd number of quotes
        quote_count = repaired.count('"')
        if quote_count % 2 != 0:
            def _escape_quotes(match: re.Match) -> str:
                inner_text = match.group(1)
                escaped = inner_text.replace('"', '\\"')
                return f'"{escaped}"'

            repaired = re.sub(
                r'(?<!\\)"([^"\\]*(\\.[^"\\]*)*)"',
                _escape_quotes,
                repaired,
            )
        
        return repaired

    @staticmethod
    def _extract_first_balanced(
        text: str,
        opener: str,
        closer: str,
    ) -> Optional[Any]:
        stack: List[str] = []
        start_index = -1

        for i, char in enumerate(text):
            if char == opener:
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == closer:
                if stack:
                    stack.pop()
                    if not stack and start_index != -1:
                        json_str = text[start_index : i + 1]
                        repaired = JSONRepair.repair_json(json_str)
                        try:
                            return json.loads(repaired)
                        except json.JSONDecodeError:
                            start_index = -1
                else:
                    start_index = -1
        return None

    @staticmethod
    def extract_and_repair_json(text: str) -> Optional[Any]:
        if "{" in text:
            obj = JSONRepair._extract_first_balanced(text, "{", "}")
            if obj is not None:
                return obj
        if "[" in text:
            arr = JSONRepair._extract_first_balanced(text, "[", "]")
            if arr is not None:
                return arr
        return None


# =============================================================================
# Completion Post-Processor
# =============================================================================


@dataclass
class LLMCompletionPostProcessor:
    """
    Encapsulates completion post-processing logic.

    This class is stateless with respect to individual requests; per-request
    overrides are derived from OperationContext.attrs.
    """

    base_config: LLMPostProcessingConfig
    safety_filter: SafetyFilter
    json_repair: JSONRepair

    ATTR_ENABLED = "llm_postprocess_enabled"
    ATTR_SAFETY_FILTER = "llm_postprocess_safety_filter"
    ATTR_JSON_REPAIR = "llm_postprocess_json_repair"
    ATTR_OUTPUT_FORMAT = "llm_postprocess_output_format"
    ATTR_MAX_LENGTH = "llm_postprocess_max_length"

    def apply(
        self,
        completion: LLMCompletion,
        ctx: OperationContext,
    ) -> LLMCompletion:
        config = self._resolve_config(ctx)

        if not config.enabled:
            return completion

        result = completion

        if config.safety_filter:
            result = self._apply_safety_filtering(result, ctx)

        if config.json_repair:
            result = self._apply_json_repair(result)

        if config.output_format:
            result = self._apply_output_formatting(result, config)

        if config.max_length is not None:
            result = self._apply_length_truncation(result, config.max_length)

        if config.custom_processors:
            for processor in config.custom_processors:
                try:
                    result = processor(result)
                except Exception as processor_exc:
                    LOG.debug(
                        "Custom post-processing processor failed: %s",
                        processor_exc,
                        extra={"request_id": ctx.request_id, "tenant": ctx.tenant}
                    )

        return result

    def _resolve_config(self, ctx: OperationContext) -> LLMPostProcessingConfig:
        attrs = ctx.attrs or {}
        override_keys = {
            self.ATTR_ENABLED,
            self.ATTR_SAFETY_FILTER,
            self.ATTR_JSON_REPAIR,
            self.ATTR_OUTPUT_FORMAT,
            self.ATTR_MAX_LENGTH,
        }

        if not any(key in attrs for key in override_keys):
            return self.base_config

        enabled = self._coerce_bool(attrs.get(self.ATTR_ENABLED), self.base_config.enabled)
        safety_filter = self._coerce_bool(
            attrs.get(self.ATTR_SAFETY_FILTER), self.base_config.safety_filter
        )
        json_repair = self._coerce_bool(
            attrs.get(self.ATTR_JSON_REPAIR), self.base_config.json_repair
        )
        output_format = attrs.get(self.ATTR_OUTPUT_FORMAT, self.base_config.output_format)
        max_length_raw = attrs.get(self.ATTR_MAX_LENGTH, self.base_config.max_length)
        max_length = self._coerce_int(max_length_raw)

        if max_length is None and isinstance(self.base_config.max_length, int):
            max_length = self.base_config.max_length

        return LLMPostProcessingConfig(
            enabled=enabled,
            safety_filter=safety_filter,
            json_repair=json_repair,
            output_format=output_format,
            max_length=max_length,
            custom_processors=self.base_config.custom_processors,
        )

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if value is None: return default
        if isinstance(value, bool): return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}: return True
            if lowered in {"false", "0", "no", "n", "off"}: return False
        return bool(value)

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None: return None
        if isinstance(value, int): return value
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _apply_safety_filtering(self, completion: LLMCompletion, ctx: OperationContext) -> LLMCompletion:
        filtered_text, violations = self.safety_filter.filter_content(completion.text)
        if violations:
            LOG.warning(
                "safety_filtering_applied",
                extra={
                    "violations": violations,
                    "request_id": ctx.request_id,
                    "tenant": ctx.tenant,
                }
            )
        
        # Use dataclasses.replace to preserve other fields (e.g. metadata from adapter)
        return replace(completion, text=filtered_text)

    def _apply_json_repair(self, completion: LLMCompletion) -> LLMCompletion:
        repaired_tool_calls: List[Any] = []
        for tool_call in completion.tool_calls:
            try:
                original_args = tool_call.function.arguments
                if isinstance(original_args, str):
                    repaired_args = self.json_repair.repair_json(original_args)
                    # Validate it is valid JSON
                    parsed = json.loads(repaired_args)
                    # Re-serialize minified
                    
                    # Deep copy via replace to preserve other fields
                    new_func = replace(tool_call.function, arguments=json.dumps(parsed))
                    repaired_tool_call = replace(tool_call, function=new_func)
                    
                    repaired_tool_calls.append(repaired_tool_call)
                else:
                    repaired_tool_calls.append(tool_call)
            except Exception as exc:
                LOG.debug("Failed to repair tool call JSON: %s", exc)
                repaired_tool_calls.append(tool_call)

        repaired_text = completion.text
        stripped = completion.text.strip()
        if stripped.startswith(("{", "[")):
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                repaired = self.json_repair.repair_json(stripped)
                try:
                    json.loads(repaired)
                    repaired_text = repaired
                    LOG.info("Repaired JSON in completion text")
                except json.JSONDecodeError:
                    pass

        return replace(completion, text=repaired_text, tool_calls=repaired_tool_calls)

    def _apply_output_formatting(self, completion: LLMCompletion, config: LLMPostProcessingConfig) -> LLMCompletion:
        if config.output_format == "json":
            stripped = completion.text.strip()
            if not stripped.startswith(("{", "[")):
                extracted = self.json_repair.extract_and_repair_json(stripped)
                if extracted is not None:
                    formatted = json.dumps(extracted, indent=2)
                else:
                    formatted = json.dumps({"text": completion.text}, indent=2)
                
                return replace(completion, text=formatted)

        elif config.output_format == "markdown":
            if "```" not in completion.text:
                return replace(completion, text=f"```text\n{completion.text}\n```")
        
        return completion

    @staticmethod
    def _apply_length_truncation(completion: LLMCompletion, max_length: int) -> LLMCompletion:
        if max_length <= 0 or len(completion.text) <= max_length:
            return completion

        truncated = completion.text[:max_length]
        cutoff = max(truncated.rfind("."), truncated.rfind("\n"))
        
        if cutoff > max_length * 0.7:
            truncated = truncated[:cutoff + 1] + " [truncated]"
        else:
            truncated += " [truncated]"

        return replace(completion, text=truncated, finish_reason="length")


# =============================================================================
# Helpers: OperationContext normalization
# =============================================================================


def _ensure_llm_operation_context(
    ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
) -> OperationContext:
    """Normalize various context shapes into an LLM OperationContext."""
    if ctx is None:
        core_ctx = ctx_from_dict({})
    elif isinstance(ctx, OperationContext):
        return ctx
    elif hasattr(ctx, "request_id") and hasattr(ctx, "attrs"):
        # ctx is an OperationContext from another module (e.g. core.operational_context)
        # Convert it to llm_base.OperationContext
        return OperationContext(
            request_id=getattr(ctx, "request_id", None),
            idempotency_key=getattr(ctx, "idempotency_key", None),
            deadline_ms=getattr(ctx, "deadline_ms", None),
            traceparent=getattr(ctx, "traceparent", None),
            tenant=getattr(ctx, "tenant", None),
            attrs=getattr(ctx, "attrs", None) or {},
        )
    elif isinstance(ctx, Mapping):
        core_ctx = ctx_from_dict(ctx)
    else:
        raise BadRequest(
            f"Unsupported context type: {type(ctx).__name__}",
            code="BAD_OPERATION_CONTEXT",
        )

    return OperationContext(
        request_id=getattr(core_ctx, "request_id", None),
        idempotency_key=getattr(core_ctx, "idempotency_key", None),
        deadline_ms=getattr(core_ctx, "deadline_ms", None),
        traceparent=getattr(core_ctx, "traceparent", None),
        tenant=getattr(core_ctx, "tenant", None),
        attrs=getattr(core_ctx, "attrs", None) or {},
    )


# =============================================================================
# Enhanced Framework-agnostic translator protocol
# =============================================================================


class LLMFrameworkTranslator(Protocol):
    """
    Per-framework translator contract for LLM operations.
    
    Implementations are responsible for:
        - Converting framework-level message inputs into NormalizedMessage[]
        - Deciding how system messages are handled
        - Converting framework-native tool definitions / tool_choice into
          Corpus wire-compatible shapes
        - Translating LLMCompletion / LLMChunk / capabilities / health /
          token counts back into framework-level outputs
    """

    def to_normalized_messages(self, raw_messages: Any, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> List[NormalizedMessage]: ...
    def build_system_message(self, normalized_messages: List[NormalizedMessage], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Tuple[Optional[str], List[NormalizedMessage]]: ...
    def build_tools(self, raw_tools: Optional[Any], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Optional[List[Dict[str, Any]]]: ...
    def build_tool_choice(self, raw_tool_choice: Optional[Any], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Optional[Union[str, Dict[str, Any]]]: ...
    
    def from_completion(self, completion: LLMCompletion, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any: ...
    def from_chunk(self, chunk: LLMChunk, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any: ...
    def from_count_tokens(self, token_count: int, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any: ...
    def from_health(self, health: Mapping[str, Any], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any: ...
    def from_capabilities(self, caps: LLMCapabilities, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any: ...

    def translate_tool_calls_to_framework(self, tool_calls: List[Dict[str, Any]], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any: ...
    def translate_tool_outputs_from_framework(self, tool_outputs: Any, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> List[Dict[str, Any]]: ...

    def preferred_model(self, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Optional[str]: ...
    def decorate_messages_before_send(self, messages: List[NormalizedMessage], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> List[NormalizedMessage]: ...
    def get_token_counting_config(self, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> TokenCountingConfig: ...
    def get_metrics(self) -> TranslatorMetrics: ...


# =============================================================================
# Enhanced Default generic translator implementation
# =============================================================================


class DefaultLLMFrameworkTranslator:
    """Generic, framework-neutral translator implementation."""

    def __init__(self) -> None:
        self._tool_validator = ToolValidator()
        self._metrics = TranslatorMetrics()

    def to_normalized_messages(self, raw_messages: Any, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> List[NormalizedMessage]:
        start_time = time.time()
        try:
            if isinstance(raw_messages, Mapping):
                raw_seq = [raw_messages]
            else:
                raw_seq = raw_messages

            if not isinstance(raw_seq, Iterable) or isinstance(raw_seq, (str, bytes)):
                self._metrics.record_validation_error()
                raise BadRequest("raw_messages must be a mapping or iterable", code="BAD_MESSAGES")

            messages = []
            for idx, m in enumerate(raw_seq):
                if isinstance(m, NormalizedMessage):
                    messages.append(m)
                elif isinstance(m, Mapping):
                    try:
                        messages.append(from_generic_dict(m))
                    except Exception:
                        self._metrics.record_normalization_error()
                        raise BadRequest(f"raw_messages[{idx}] malformed", code="BAD_MESSAGE_FORMAT", details={"index": idx})
                else:
                    self._metrics.record_validation_error()
                    raise BadRequest(f"raw_messages[{idx}] must be mapping or NormalizedMessage", code="BAD_MESSAGES")

            if not messages:
                self._metrics.record_validation_error()
                raise BadRequest("raw_messages must contain at least one message", code="BAD_MESSAGES")

            self._metrics.record_translation(time.time() - start_time, success=True)
            return messages
        except Exception:
            self._metrics.record_translation(time.time() - start_time, success=False)
            raise

    def build_system_message(self, normalized_messages: List[NormalizedMessage], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Tuple[Optional[str], List[NormalizedMessage]]:
        system_message = None
        remaining = []
        extracting = True
        for msg in normalized_messages:
            if extracting and msg.role == "system":
                if system_message is None: system_message = msg.content
                else: system_message += "\n" + msg.content
            else:
                extracting = False
                remaining.append(msg)
        return system_message, remaining

    def build_tools(self, raw_tools: Optional[Any], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Optional[List[Dict[str, Any]]]:
        if raw_tools is None: return None
        if isinstance(raw_tools, Mapping): raw_tools = [raw_tools]
        
        if not isinstance(raw_tools, Sequence):
            self._metrics.record_validation_error()
            raise BadRequest("tools must be a sequence or mapping", code="BAD_TOOLS")

        tools = []
        for idx, t in enumerate(raw_tools):
            if not isinstance(t, Mapping):
                self._metrics.record_validation_error()
                raise BadRequest(f"tools[{idx}] must be a mapping", code="BAD_TOOLS")
            try:
                self._tool_validator.validate_tool_schema(dict(t))
                self._metrics.record_tool_translation()
                tools.append(dict(t))
            except ToolValidationError as exc:
                self._metrics.record_validation_error()
                raise
        return tools

    def build_tool_choice(self, raw_tool_choice: Optional[Any], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Optional[Union[str, Dict[str, Any]]]:
        if raw_tool_choice is None: return None
        try:
            self._tool_validator.validate_tool_choice(raw_tool_choice)
        except ToolValidationError:
            self._metrics.record_validation_error()
            raise
        
        if isinstance(raw_tool_choice, (str, Mapping)):
            return raw_tool_choice if isinstance(raw_tool_choice, str) else dict(raw_tool_choice)
            
        self._metrics.record_validation_error()
        raise BadRequest("tool_choice must be string or mapping", code="BAD_TOOL_CHOICE")

    def from_completion(self, completion: LLMCompletion, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any:
        self._metrics.record_completion_translation()
        return {
            "text": completion.text,
            "model": completion.model,
            "model_family": completion.model_family,
            "usage": asdict(completion.usage) if completion.usage else None,
            "finish_reason": completion.finish_reason,
            "tool_calls": [asdict(tc) for tc in completion.tool_calls],
        }

    def from_chunk(self, chunk: LLMChunk, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any:
        self._metrics.record_chunk_translation()
        return {
            "text": chunk.text,
            "is_final": chunk.is_final,
            "model": chunk.model,
            "usage_so_far": asdict(chunk.usage_so_far) if chunk.usage_so_far else None,
            "tool_calls": [asdict(tc) for tc in chunk.tool_calls],
        }

    def from_count_tokens(self, token_count: int, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any:
        return int(token_count)

    def from_health(self, health: Mapping[str, Any], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any:
        return dict(health)

    def from_capabilities(self, caps: LLMCapabilities, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any:
        return asdict(caps)

    def translate_tool_calls_to_framework(self, tool_calls: List[Dict[str, Any]], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Any:
        return tool_calls

    def translate_tool_outputs_from_framework(self, tool_outputs: Any, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> List[Dict[str, Any]]:
        if tool_outputs is None: return []
        if isinstance(tool_outputs, list): return tool_outputs
        if isinstance(tool_outputs, dict): return [tool_outputs]
        raise BadRequest("tool_outputs must be list or dict", code="BAD_TOOL_OUTPUTS")

    def preferred_model(self, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> Optional[str]:
        attrs = op_ctx.attrs or {}
        candidate = attrs.get("llm_model") or attrs.get("model")
        return str(candidate).strip() if candidate else None

    def decorate_messages_before_send(self, messages: List[NormalizedMessage], *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> List[NormalizedMessage]:
        return list(messages)

    def get_token_counting_config(self, *, op_ctx: OperationContext, framework_ctx: Optional[Any] = None) -> TokenCountingConfig:
        return TokenCountingConfig()

    def get_metrics(self) -> TranslatorMetrics:
        return self._metrics


# =============================================================================
# Enhanced LLM Translator Orchestrator
# =============================================================================


class LLMTranslator:
    """
    Framework-agnostic orchestrator for LLM operations.

    This class:
        - Accepts framework-level inputs and a normalized OperationContext
        - Delegates to an LLMFrameworkTranslator to build specs and translate results
        - Calls into an LLMProtocolV1 adapter to execute operations
        - Provides sync + async variants for all core operations
        - Handles streaming via SyncStreamBridge for sync callers
        - Attaches rich error context for diagnostics

    Sync methods use AsyncBridge to call async adapters from sync contexts.
    """

    def __init__(
        self,
        *,
        adapter: LLMProtocolV1,
        framework: str = "generic",
        translator: Optional[LLMFrameworkTranslator] = None,
        post_processing_config: Optional[LLMPostProcessingConfig] = None,
        safety_filter: Optional[SafetyFilter] = None,
        json_repair: Optional[JSONRepair] = None,
    ) -> None:
        self._adapter = adapter
        self._framework = framework
        self._translator = translator or DefaultLLMFrameworkTranslator()
        self._post_processing_config = post_processing_config or LLMPostProcessingConfig()
        self._post_processor = LLMCompletionPostProcessor(
            base_config=self._post_processing_config,
            safety_filter=safety_filter or SafetyFilter(),
            json_repair=json_repair or JSONRepair(),
        )

    def __enter__(self) -> "LLMTranslator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        close = getattr(self._adapter, "close", None)
        if callable(close):
            close()

    async def __aenter__(self) -> "LLMTranslator":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        aclose = getattr(self._adapter, "aclose", None)
        if callable(aclose):
            await aclose()

    # --------------------------------------------------------------------- #
    # Internal execution helpers (Executor Pattern)
    # --------------------------------------------------------------------- #

    def _run_operation(
        self,
        *,
        op_name: str,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
        sync: bool,
        logic: Callable[[OperationContext], Awaitable[R]],
    ) -> Union[R, Awaitable[R]]:
        """
        Centralized execution for non-streaming operations.

        - Normalizes OperationContext
        - Wraps logic with consistent error-context attachment
        - Uses AsyncBridge for sync variants, returns coroutine for async variants
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        async def _wrapped() -> R:
            try:
                return await logic(ctx)
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    resource_type="llm",
                    operation=f"llm.{op_name}",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        if sync:
            timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None
            return AsyncBridge.run_async(_wrapped(), timeout=timeout)
        return _wrapped()

    def _run_stream_operation(
        self,
        *,
        op_name: str,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
        sync: bool,
        factory: Callable[[OperationContext], AsyncIterator[R]],
    ) -> Union[Iterator[R], AsyncIterator[R]]:
        """
        Centralized execution for streaming operations.
        
        Args:
            factory: A function that takes context and returns an AsyncIterator.
                     (Note: AsyncGenerator functions return AsyncIterator immediately,
                      no await required to get the iterator).
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        async def _async_gen() -> AsyncIterator[R]:
            try:
                # FIX: Calling factory(ctx) returns the async generator immediately.
                # Do NOT await factory(ctx).
                async for chunk in factory(ctx):
                    yield chunk
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    resource_type="llm",
                    operation=f"llm.{op_name}",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        if sync:
            return SyncStreamBridge(
                coro_factory=_async_gen,
                framework=self._framework,
                error_context={
                    "operation": f"llm.{op_name}",
                    "request_id": ctx.request_id,
                    "tenant": ctx.tenant,
                },
            ).run()
        return _async_gen()

    def _prepare_stream_factory(self, raw_messages, **kwargs) -> Callable[[OperationContext], AsyncIterator[Any]]:
        """Shared logic to prepare the stream generator."""
        
        async def _factory(ctx: OperationContext) -> AsyncIterator[Any]:
            normalized = self._translator.to_normalized_messages(raw_messages, op_ctx=ctx, framework_ctx=kwargs.get('framework_ctx'))
            normalized = self._translator.decorate_messages_before_send(normalized, op_ctx=ctx, framework_ctx=kwargs.get('framework_ctx'))
            
            auto_system, remaining = self._translator.build_system_message(normalized, op_ctx=ctx, framework_ctx=kwargs.get('framework_ctx'))
            effective_system = kwargs.get('system_message') if kwargs.get('system_message') is not None else auto_system
            
            tools_corpus = self._translator.build_tools(kwargs.get('tools'), op_ctx=ctx, framework_ctx=kwargs.get('framework_ctx'))
            tool_choice_corpus = self._translator.build_tool_choice(kwargs.get('tool_choice'), op_ctx=ctx, framework_ctx=kwargs.get('framework_ctx'))
            
            wire_messages = to_corpus(remaining)
            effective_model = kwargs.get('model') or self._translator.preferred_model(op_ctx=ctx, framework_ctx=kwargs.get('framework_ctx'))

            # Call stream on adapter
            agen = self._adapter.stream(
                messages=wire_messages,
                max_tokens=kwargs.get('max_tokens'),
                temperature=kwargs.get('temperature'),
                top_p=kwargs.get('top_p'),
                frequency_penalty=kwargs.get('frequency_penalty'),
                presence_penalty=kwargs.get('presence_penalty'),
                stop_sequences=kwargs.get('stop_sequences'),
                model=effective_model,
                system_message=effective_system,
                tools=tools_corpus,
                tool_choice=tool_choice_corpus,
                ctx=ctx,
            )
            
            async for chunk in agen:
                if not isinstance(chunk, LLMChunk):
                     raise BadRequest(f"adapter.stream yielded invalid type: {type(chunk)}", code="BAD_ADAPTER_RESULT")
                yield self._translator.from_chunk(chunk, op_ctx=ctx, framework_ctx=kwargs.get('framework_ctx'))
        
        return _factory

    @staticmethod
    def _coerce_token_count(result: Any) -> int:
        """Robustly coerce token count result to int."""
        try:
            if isinstance(result, (int, float)):
                return int(result)
            if isinstance(result, str):
                return int(float(str(result).strip()))
        except (TypeError, ValueError):
            pass
        
        raise BadRequest(
            "count_tokens returned non-numeric value",
            code="BAD_ADAPTER_COUNT",
            details={"received_type": type(result).__name__}
        )

    # --------------------------------------------------------------------- #
    # Enhanced Token Counting Helpers
    # --------------------------------------------------------------------- #

    def _format_messages_for_token_counting(
        self,
        normalized_messages: List[NormalizedMessage],
        system_message: Optional[str],
        token_config: TokenCountingConfig,
    ) -> str:
        """
        Format messages for token counting using sophisticated strategies.
        """
        parts: List[str] = []

        special_tokens_overhead = 0
        if token_config.add_special_tokens:
            if token_config.format_strategy == "openai_chatml":
                special_tokens_overhead = 20
            elif token_config.format_strategy == "anthropic":
                special_tokens_overhead = 10

        if token_config.format_strategy == "simple":
            if system_message and token_config.include_system_in_messages:
                parts.append(f"system:{system_message}")
            for msg in normalized_messages:
                parts.append(f"{msg.role}:{msg.content}")

        elif token_config.format_strategy == "openai_chatml":
            if system_message and token_config.include_system_in_messages:
                parts.append(f"<|im_start|>system\n{system_message}<|im_end|>")
            for msg in normalized_messages:
                parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")

        elif token_config.format_strategy == "anthropic":
            if system_message and token_config.include_system_in_messages:
                parts.append(f"System: {system_message}")
            for msg in normalized_messages:
                role_display = "Human" if msg.role == "user" else "Assistant"
                parts.append(f"\n\n{role_display}: {msg.content}")

        elif (
            token_config.format_strategy == "custom"
            and token_config.custom_format_template
        ):
            template = token_config.custom_format_template
            if system_message and token_config.include_system_in_messages:
                parts.append(template.format(role="system", content=system_message))
            for msg in normalized_messages:
                parts.append(template.format(role=msg.role, content=msg.content))

        formatted_text = "\n".join(parts)
        if token_config.add_special_tokens and special_tokens_overhead > 0:
            formatted_text += " " * special_tokens_overhead

        return formatted_text

    def count_tokens_for_messages(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Enhanced synchronous helper to count tokens for chat messages."""
        
        async def _logic(ctx):
            normalized = self._translator.to_normalized_messages(raw_messages, op_ctx=ctx, framework_ctx=framework_ctx)
            auto_system, remaining = self._translator.build_system_message(normalized, op_ctx=ctx, framework_ctx=framework_ctx)
            token_config = self._translator.get_token_counting_config(op_ctx=ctx, framework_ctx=framework_ctx)
            
            combined = self._format_messages_for_token_counting(remaining, auto_system, token_config)
            
            res = await self._adapter.count_tokens(text=combined, model=model, ctx=ctx)
            numeric = self._coerce_token_count(res)
            return self._translator.from_count_tokens(numeric, op_ctx=ctx, framework_ctx=framework_ctx)

        return self._run_operation(op_name="count_tokens_for_messages", op_ctx=op_ctx, sync=True, logic=_logic)

    async def arun_count_tokens_for_messages(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Enhanced async helper to count tokens for chat messages."""
        
        async def _logic(ctx):
            normalized = self._translator.to_normalized_messages(raw_messages, op_ctx=ctx, framework_ctx=framework_ctx)
            auto_system, remaining = self._translator.build_system_message(normalized, op_ctx=ctx, framework_ctx=framework_ctx)
            token_config = self._translator.get_token_counting_config(op_ctx=ctx, framework_ctx=framework_ctx)
            
            combined = self._format_messages_for_token_counting(remaining, auto_system, token_config)
            
            res = await self._adapter.count_tokens(text=combined, model=model, ctx=ctx)
            numeric = self._coerce_token_count(res)
            return self._translator.from_count_tokens(numeric, op_ctx=ctx, framework_ctx=framework_ctx)

        return await self._run_operation(op_name="count_tokens_for_messages", op_ctx=op_ctx, sync=False, logic=_logic)

    # --------------------------------------------------------------------- #
    # Public APIs
    # --------------------------------------------------------------------- #

    def complete(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        system_message: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous complete API.

        Uses AsyncBridge to call the async adapter from a sync context.
        Supports post-processing of results (safety, JSON repair).
        """
        async def _logic(ctx):
            # Normalize & decorate
            normalized = self._translator.to_normalized_messages(raw_messages, op_ctx=ctx, framework_ctx=framework_ctx)
            normalized = self._translator.decorate_messages_before_send(normalized, op_ctx=ctx, framework_ctx=framework_ctx)
            
            # System message & Tools
            auto_system, remaining = self._translator.build_system_message(normalized, op_ctx=ctx, framework_ctx=framework_ctx)
            effective_system = system_message if system_message is not None else auto_system
            tools_corpus = self._translator.build_tools(tools, op_ctx=ctx, framework_ctx=framework_ctx)
            tool_choice_corpus = self._translator.build_tool_choice(tool_choice, op_ctx=ctx, framework_ctx=framework_ctx)

            wire_messages = to_corpus(remaining)
            effective_model = model or self._translator.preferred_model(op_ctx=ctx, framework_ctx=framework_ctx)

            # Call Adapter
            result = await self._adapter.complete(
                messages=wire_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=stop_sequences,
                model=effective_model,
                system_message=effective_system,
                tools=tools_corpus,
                tool_choice=tool_choice_corpus,
                ctx=ctx,
            )

            if not isinstance(result, LLMCompletion):
                raise BadRequest(f"adapter.complete returned invalid type: {type(result)}", code="BAD_ADAPTER_RESULT")

            # Post-process
            result_processed = self._post_processor.apply(result, ctx)
            return self._translator.from_completion(result_processed, op_ctx=ctx, framework_ctx=framework_ctx)

        return self._run_operation(op_name="complete", op_ctx=op_ctx, sync=True, logic=_logic)

    async def arun_complete(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        system_message: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async complete API (preferred for async applications).

        Fully async flow: adapter.complete -> post-processing -> translation.
        """
        async def _logic(ctx):
            # Normalize & decorate
            normalized = self._translator.to_normalized_messages(raw_messages, op_ctx=ctx, framework_ctx=framework_ctx)
            normalized = self._translator.decorate_messages_before_send(normalized, op_ctx=ctx, framework_ctx=framework_ctx)
            
            # System message & Tools
            auto_system, remaining = self._translator.build_system_message(normalized, op_ctx=ctx, framework_ctx=framework_ctx)
            effective_system = system_message if system_message is not None else auto_system
            tools_corpus = self._translator.build_tools(tools, op_ctx=ctx, framework_ctx=framework_ctx)
            tool_choice_corpus = self._translator.build_tool_choice(tool_choice, op_ctx=ctx, framework_ctx=framework_ctx)

            wire_messages = to_corpus(remaining)
            effective_model = model or self._translator.preferred_model(op_ctx=ctx, framework_ctx=framework_ctx)

            # Call Adapter
            result = await self._adapter.complete(
                messages=wire_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=stop_sequences,
                model=effective_model,
                system_message=effective_system,
                tools=tools_corpus,
                tool_choice=tool_choice_corpus,
                ctx=ctx,
            )

            if not isinstance(result, LLMCompletion):
                raise BadRequest(f"adapter.complete returned invalid type: {type(result)}", code="BAD_ADAPTER_RESULT")

            # Post-process
            result_processed = self._post_processor.apply(result, ctx)
            return self._translator.from_completion(result_processed, op_ctx=ctx, framework_ctx=framework_ctx)

        return await self._run_operation(op_name="complete", op_ctx=op_ctx, sync=False, logic=_logic)

    def stream(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        system_message: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Iterator[Any]:
        """
        Synchronous streaming API.

        Returns a sync iterator yielding framework-level streaming chunks.
        """
        return self._run_stream_operation(
            op_name="stream",
            op_ctx=op_ctx,
            sync=True,
            factory=self._prepare_stream_factory(
                raw_messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=stop_sequences,
                tools=tools,
                tool_choice=tool_choice,
                system_message=system_message,
                framework_ctx=framework_ctx,
            )
        )

    def arun_stream(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        system_message: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> AsyncIterator[Any]:
        """
        Async streaming API.

        Returns an async iterator yielding framework-level streaming chunks.
        """
        # Returns AsyncIterator directly (not a coroutine)
        return self._run_stream_operation(
            op_name="stream",
            op_ctx=op_ctx,
            sync=False,
            factory=self._prepare_stream_factory(
                raw_messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop_sequences=stop_sequences,
                tools=tools,
                tool_choice=tool_choice,
                system_message=system_message,
                framework_ctx=framework_ctx,
            )
        )

    def count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous count_tokens wrapper."""
        async def _logic(ctx):
            res = await self._adapter.count_tokens(text=text, model=model, ctx=ctx)
            numeric = self._coerce_token_count(res)
            return self._translator.from_count_tokens(numeric, op_ctx=ctx, framework_ctx=framework_ctx)

        return self._run_operation(op_name="count_tokens", op_ctx=op_ctx, sync=True, logic=_logic)

    async def arun_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async count_tokens wrapper."""
        async def _logic(ctx):
            res = await self._adapter.count_tokens(text=text, model=model, ctx=ctx)
            numeric = self._coerce_token_count(res)
            return self._translator.from_count_tokens(numeric, op_ctx=ctx, framework_ctx=framework_ctx)

        return await self._run_operation(op_name="count_tokens", op_ctx=op_ctx, sync=False, logic=_logic)

    def health(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous health check wrapper."""
        async def _logic(ctx):
            res = await self._adapter.health(ctx=ctx)
            return self._translator.from_health(res, op_ctx=ctx, framework_ctx=framework_ctx)
        return self._run_operation(op_name="health", op_ctx=op_ctx, sync=True, logic=_logic)

    async def arun_health(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async health check wrapper."""
        async def _logic(ctx):
            res = await self._adapter.health(ctx=ctx)
            return self._translator.from_health(res, op_ctx=ctx, framework_ctx=framework_ctx)
        return await self._run_operation(op_name="health", op_ctx=op_ctx, sync=False, logic=_logic)

    def capabilities(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Synchronous capabilities wrapper."""
        async def _logic(ctx):
            res = await self._adapter.capabilities()
            return self._translator.from_capabilities(res, op_ctx=ctx, framework_ctx=framework_ctx)
        return self._run_operation(op_name="capabilities", op_ctx=op_ctx, sync=True, logic=_logic)

    async def arun_capabilities(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Async capabilities wrapper."""
        async def _logic(ctx):
            res = await self._adapter.capabilities()
            return self._translator.from_capabilities(res, op_ctx=ctx, framework_ctx=framework_ctx)
        return await self._run_operation(op_name="capabilities", op_ctx=op_ctx, sync=False, logic=_logic)

    def get_metrics(self) -> Optional[TranslatorMetrics]:
        """Get current translation metrics for observability."""
        return self._translator.get_metrics() if hasattr(self._translator, "get_metrics") else None

    def translate_tool_calls_to_framework(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """Helper to access translator's tool call translation logic."""
        ctx = _ensure_llm_operation_context(op_ctx)
        return self._translator.translate_tool_calls_to_framework(tool_calls, op_ctx=ctx, framework_ctx=framework_ctx)

    def translate_tool_outputs_from_framework(
        self,
        tool_outputs: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Helper to access translator's tool output translation logic."""
        ctx = _ensure_llm_operation_context(op_ctx)
        return self._translator.translate_tool_outputs_from_framework(tool_outputs, op_ctx=ctx, framework_ctx=framework_ctx)


# =============================================================================
# Registry
# =============================================================================

_TranslatorFactory = Callable[[LLMProtocolV1], LLMFrameworkTranslator]
_LLM_TRANSLATOR_FACTORIES: Dict[str, _TranslatorFactory] = {}
_REGISTRY_LOCK = threading.Lock()

def register_llm_translator(framework: str, factory: _TranslatorFactory) -> None:
    if not framework or not isinstance(framework, str):
        raise BadRequest("framework name must be a non-empty string", code="BAD_TRANSLATOR_REGISTRATION")
    if not callable(factory):
        raise BadRequest("translator factory must be callable", code="BAD_TRANSLATOR_REGISTRATION")
    
    with _REGISTRY_LOCK:
        _LLM_TRANSLATOR_FACTORIES[framework] = factory
    LOG.debug("Registered LLM translator factory for framework=%s", framework)

def get_llm_translator_factory(framework: str) -> Optional[_TranslatorFactory]:
    with _REGISTRY_LOCK:
        return _LLM_TRANSLATOR_FACTORIES.get(framework)

def create_llm_translator(
    *,
    adapter: LLMProtocolV1,
    framework: str = "generic",
    translator: Optional[LLMFrameworkTranslator] = None,
    post_processing_config: Optional[LLMPostProcessingConfig] = None,
    safety_filter: Optional[SafetyFilter] = None,
    json_repair: Optional[JSONRepair] = None,
) -> LLMTranslator:
    if translator is None:
        factory = get_llm_translator_factory(framework)
        if factory is not None:
            translator = factory(adapter)
        else:
            translator = DefaultLLMFrameworkTranslator()
            
    return LLMTranslator(
        adapter=adapter,
        framework=framework,
        translator=translator,
        post_processing_config=post_processing_config,
        safety_filter=safety_filter,
        json_repair=json_repair,
    )

__all__ = [
    "TokenCountingConfig",
    "ToolValidationError",
    "ToolValidator",
    "LLMPostProcessingConfig",
    "TranslatorMetrics",
    "SafetyFilter",
    "JSONRepair",
    "LLMCompletionPostProcessor",
    "LLMFrameworkTranslator",
    "DefaultLLMFrameworkTranslator",
    "LLMTranslator",
    "register_llm_translator",
    "get_llm_translator_factory",
    "create_llm_translator",
]
