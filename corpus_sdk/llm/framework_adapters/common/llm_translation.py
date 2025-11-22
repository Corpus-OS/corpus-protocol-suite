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
"""

from __future__ import annotations

import logging
import time
import json
import re
from dataclasses import asdict, dataclass
from typing import (
    Any,
    AsyncIterator,
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
    LLMProtocolV1,
    OperationContext,
    LLMCompletion,
    LLMChunk,
    LLMCapabilities,
    BadRequest,
)
from corpus_sdk.llm.framework_adapters.common.message_translation import (
    NormalizedMessage,
    to_corpus,
    from_generic_dict,
)
from corpus_sdk.core.context_translation import from_dict as ctx_from_dict
from corpus_sdk.core.sync_bridge import SyncStreamBridge
from corpus_sdk.core.async_bridge import AsyncBridge
from corpus_sdk.core.error_context import attach_context

LOG = logging.getLogger(__name__)

T = TypeVar("T")


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
            placeholders. Example: "<|im_start|>{role}\n{content}<|im_end|>\n"

        include_system_in_messages:
            Whether to include system messages in the main message count
            or handle them separately.

        add_special_tokens:
            Whether to account for special tokens (BOS/EOS) in count.
            Currently used to add token overhead for formatting templates.
    """

    format_strategy: str = "simple"
    custom_format_template: Optional[str] = None
    include_system_in_messages: bool = True
    add_special_tokens: bool = True

    def __post_init__(self):
        """Validate token counting configuration."""
        valid_strategies = {"simple", "openai_chatml", "anthropic", "custom"}
        if self.format_strategy not in valid_strategies:
            raise ValueError(
                f"format_strategy must be one of {valid_strategies}, got {self.format_strategy}"
            )
        if self.format_strategy == "custom" and not self.custom_format_template:
            raise ValueError("custom_format_template is required when format_strategy='custom'")


# =============================================================================
# Enhanced Tool Validation
# =============================================================================


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

        Raises BadRequest with detailed error messages if validation fails.
        """
        if not isinstance(tool, dict):
            raise BadRequest("Tool must be a dictionary", code="BAD_TOOL_SCHEMA")

        # Check tool type
        tool_type = tool.get("type")
        if tool_type != "function":
            raise BadRequest(
                f"Tool type must be 'function', got {tool_type!r}",
                code="BAD_TOOL_TYPE",
                details={"supported_types": ["function"]},
            )

        # Check function definition
        function_def = tool.get("function")
        if not isinstance(function_def, dict):
            raise BadRequest(
                "Tool must contain a 'function' dictionary",
                code="BAD_TOOL_FUNCTION",
            )

        # Validate required function keys
        missing_keys = cls.REQUIRED_FUNCTION_KEYS - set(function_def.keys())
        if missing_keys:
            raise BadRequest(
                f"Tool function missing required keys: {missing_keys}",
                code="BAD_TOOL_FUNCTION_KEYS",
                details={"required_keys": list(cls.REQUIRED_FUNCTION_KEYS)},
            )

        # Validate function name
        name = function_def.get("name", "")
        if not isinstance(name, str) or not name.strip():
            raise BadRequest(
                "Tool function name must be a non-empty string",
                code="BAD_TOOL_NAME",
            )

        # Validate parameters schema
        parameters = function_def.get("parameters", {})
        if not isinstance(parameters, dict):
            raise BadRequest(
                "Tool function parameters must be a dictionary",
                code="BAD_TOOL_PARAMETERS",
            )

        # Validate required parameters keys
        missing_param_keys = cls.REQUIRED_PARAMETERS_KEYS - set(parameters.keys())
        if missing_param_keys:
            raise BadRequest(
                f"Tool parameters missing required keys: {missing_param_keys}",
                code="BAD_TOOL_PARAMETERS_KEYS",
                details={"required_keys": list(cls.REQUIRED_PARAMETERS_KEYS)},
            )

        # Validate parameters type
        param_type = parameters.get("type")
        if param_type != "object":
            raise BadRequest(
                f"Parameters type must be 'object', got {param_type!r}",
                code="BAD_TOOL_PARAMETERS_TYPE",
            )

        # Validate properties
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            raise BadRequest(
                "Parameters properties must be a dictionary",
                code="BAD_TOOL_PROPERTIES",
            )

    @classmethod
    def validate_tool_choice(cls, tool_choice: Any) -> None:
        """
        Validate tool_choice parameter for LLM compatibility.
        """
        if tool_choice is None:
            return

        if isinstance(tool_choice, str):
            valid_choices = {"auto", "none", "required"}
            if tool_choice not in valid_choices:
                raise BadRequest(
                    f"tool_choice must be one of {valid_choices}, got {tool_choice!r}",
                    code="BAD_TOOL_CHOICE",
                )
        elif isinstance(tool_choice, dict):
            if "type" not in tool_choice or "function" not in tool_choice:
                raise BadRequest(
                    "tool_choice dict must contain 'type' and 'function' keys",
                    code="BAD_TOOL_CHOICE_DICT",
                )
        else:
            raise BadRequest(
                "tool_choice must be a string or dictionary",
                code="BAD_TOOL_CHOICE_TYPE",
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

    def __post_init__(self):
        """Validate post-processing configuration."""
        if self.output_format and self.output_format not in {"text", "json", "markdown"}:
            raise ValueError(
                f"output_format must be one of 'text', 'json', 'markdown', got {self.output_format}"
            )
        if self.max_length and self.max_length <= 0:
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

    def record_translation(self, duration: float, success: bool = True) -> None:
        """Record a translation operation with timing."""
        self.total_translation_time += duration
        self.message_translations += 1
        if not success:
            self.translation_errors += 1
        if self.message_translations > 0:
            self.avg_translation_time = self.total_translation_time / self.message_translations

    def record_validation_error(self) -> None:
        """Record a validation error."""
        self.validation_errors += 1

    def record_normalization_error(self) -> None:
        """Record a normalization error."""
        self.normalization_errors += 1

    def record_framework_path_usage(self) -> None:
        """Record usage of framework-specific translation path."""
        self.framework_specific_paths_used += 1

    def record_tool_translation(self) -> None:
        """Record a tool translation operation."""
        self.tool_translations += 1

    def record_completion_translation(self) -> None:
        """Record a completion translation operation."""
        self.completion_translations += 1

    def record_chunk_translation(self) -> None:
        """Record a chunk translation operation."""
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

    # Common profanity patterns (simplified for example)
    PROFANITY_PATTERNS = [
        r'\b(asshole|bastard|bitch|damn|fuck|shit)\b',
        r'\b(cunt|dick|piss|whore)\b',
    ]

    # PII patterns
    PII_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{16}\b',  # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone
    ]

    def __init__(self, filter_profanity: bool = True, filter_pii: bool = True):
        self.filter_profanity = filter_profanity
        self.filter_pii = filter_pii
        self.profanity_regex = re.compile('|'.join(self.PROFANITY_PATTERNS), re.IGNORECASE)
        self.pii_regex = re.compile('|'.join(self.PII_PATTERNS))

    def filter_content(self, text: str) -> Tuple[str, List[str]]:
        """
        Filter content for safety issues.
        
        Returns:
            Tuple of (filtered_text, list_of_violations)
        """
        violations = []
        filtered_text = text

        if self.filter_profanity:
            profanity_matches = self.profanity_regex.findall(text)
            if profanity_matches:
                violations.append(f"Profanity detected: {', '.join(set(profanity_matches))}")
                # Replace profanity with [REDACTED]
                filtered_text = self.profanity_regex.sub('[REDACTED]', filtered_text)

        if self.filter_pii:
            pii_matches = self.pii_regex.findall(text)
            if pii_matches:
                violations.append(f"PII detected: {', '.join(set(pii_matches))}")
                # Replace PII with [REDACTED]
                filtered_text = self.pii_regex.sub('[REDACTED]', filtered_text)

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
        """
        Attempt to repair common JSON formatting issues.
        
        Handles:
        - Unclosed brackets and braces
        - Trailing commas
        - Missing quotes around keys
        - Unescaped quotes in strings
        """
        if not text.strip():
            return text

        # Try parsing first - if it works, return as-is
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass  # Continue with repair attempts

        repaired = text.strip()

        # Balance brackets and braces
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')

        # Add missing closing braces
        if open_braces > 0:
            repaired += '}' * open_braces
        elif open_braces < 0:
            repaired = '{' * (-open_braces) + repaired

        # Add missing closing brackets
        if open_brackets > 0:
            repaired += ']' * open_brackets
        elif open_brackets < 0:
            repaired = '[' * (-open_brackets) + repaired

        # Remove trailing commas before closing braces/brackets
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

        # Add missing quotes around keys (simple heuristic)
        repaired = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', repaired)

        # Handle unescaped quotes in strings (basic)
        def _escape_quotes(match: re.Match) -> str:
            inner_text = match.group(1)
            # Escape unescaped quotes within the string
            escaped = inner_text.replace('"', '\\"')
            return f'"{escaped}"'
        
        repaired = re.sub(r'(?<!\\)"([^"\\]*(\\.[^"\\]*)*)"', _escape_quotes, repaired)

        return repaired

    @staticmethod
    def extract_and_repair_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text and repair it if necessary.
        
        Returns parsed JSON dict or None if repair fails.
        """
        # Use a stack-based approach to find complete JSON objects
        stack = []
        start_index = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_index != -1:
                        # Found a complete JSON object
                        json_str = text[start_index:i+1]
                        repaired = JSONRepair.repair_json(json_str)
                        try:
                            return json.loads(repaired)
                        except json.JSONDecodeError:
                            # Continue searching for other JSON objects
                            start_index = -1
                else:
                    start_index = -1
        
        return None


# =============================================================================
# Helpers: OperationContext normalization
# =============================================================================


def _ensure_llm_operation_context(
    ctx: Optional[Union[OperationContext, Mapping[str, Any]]],
) -> OperationContext:
    """
    Normalize various context shapes into an LLM OperationContext.

    Accepts:
        - None:
            Uses context_translation.from_dict({}) to construct an "empty"
            core OperationContext, then adapts it into the LLM OperationContext.
        - OperationContext:
            Returned as-is.
        - Mapping[str, Any]:
            Interpreted via context_translation.from_dict, then adapted into
            an LLM OperationContext.

    This mirrors the graph translation layer and keeps responsibilities clean:
        - Framework-native → normalized core context happens in
          corpus_sdk.core.context_translation.
        - This helper simply ensures the LLM adapter receives the right type
          and shape for its OperationContext.
    """
    if ctx is None:
        core_ctx = ctx_from_dict({})
    elif isinstance(ctx, OperationContext):
        return ctx
    elif isinstance(ctx, Mapping):
        core_ctx = ctx_from_dict(ctx)
    else:
        raise BadRequest(
            f"Unsupported context type: {type(ctx).__name__}",
            code="BAD_OPERATION_CONTEXT",
        )

    # Reconstruct as LLM OperationContext with validation and without
    # leaking any framework-native details.
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
        - Optionally decorating messages before send (guardrails, tags, etc.)
        - Optionally suggesting a preferred model when none is specified
    """

    # ---- input translation ----

    def to_normalized_messages(
        self,
        raw_messages: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> List[NormalizedMessage]:
        """
        Translate framework-native messages into a list of NormalizedMessage.

        raw_messages may be:
            - List[NormalizedMessage]
            - List[Mapping[str, Any]]
            - Single Mapping[str, Any] (treated as length-1 list)
            - Framework-specific message objects (LangChain, LlamaIndex, etc.)
        """
        ...

    def build_system_message(
        self,
        normalized_messages: List[NormalizedMessage],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Tuple[Optional[str], List[NormalizedMessage]]:
        """
        Decide how to handle system messages.

        Enhanced behavior preserves message ordering while extracting system content.
        Returns:
            (system_message_text, remaining_messages_preserving_order)
        """
        ...

    def build_tools(
        self,
        raw_tools: Optional[Any],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Translate framework-native tool definitions into Corpus tool schema.

        Returned structure should match the LLMProtocolV1 expectations:
            [ {"type": "function", "function": { ... }}, ... ] or None

        Enhanced with validation and framework-specific tool translation.
        """
        ...

    def build_tool_choice(
        self,
        raw_tool_choice: Optional[Any],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Translate framework-native tool_choice into the Corpus wire format.

        Typical values:
            - "auto", "none", "required"
            - A specific tool descriptor as a dict
            - None (adapter chooses)

        Enhanced with validation.
        """
        ...

    # ---- output translation ----

    def from_completion(
        self,
        completion: LLMCompletion,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert an LLMCompletion into a framework-level result object.
        """
        ...

    def from_chunk(
        self,
        chunk: LLMChunk,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert a streaming LLMChunk into a framework-level chunk representation.
        """
        ...

    def from_count_tokens(
        self,
        token_count: int,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert raw token count into a framework-level count response.
        """
        ...

    def from_health(
        self,
        health: Mapping[str, Any],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert adapter health mapping into a framework-facing health result.
        """
        ...

    def from_capabilities(
        self,
        caps: LLMCapabilities,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert LLMCapabilities into a framework-facing capabilities structure.
        """
        ...

    # ---- enhanced tool call translation ----

    def translate_tool_calls_to_framework(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert protocol tool calls into framework-native tool call objects.

        This enables frameworks to work with their native tool call representations
        rather than raw dictionaries.
        """
        ...

    def translate_tool_outputs_from_framework(
        self,
        tool_outputs: Any,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert framework-native tool outputs back to protocol format.

        Used for subsequent LLM calls with tool execution results.
        """
        ...

    # ---- optional hooks ----

    def preferred_model(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Optional hook for translators to derive a default model identifier.

        This can come from:
            - framework_ctx (e.g., configured model for a given index/router)
            - op_ctx.attrs (e.g., "llm_model" key)
        """
        ...

    def decorate_messages_before_send(
        self,
        messages: List[NormalizedMessage],
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> List[NormalizedMessage]:
        """
        Optional hook that can inject guardrails, additional context, or
        other framework-specific message transformations before calling
        the adapter.

        Default behavior is to return messages unchanged.
        """
        ...

    def get_token_counting_config(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,
    ) -> TokenCountingConfig:
        """
        Optional hook for framework-specific token counting strategies.

        Returns a TokenCountingConfig that defines how messages should be
        formatted for accurate token counting with specific LLM providers.
        """
        ...


# =============================================================================
# Enhanced Default generic translator implementation
# =============================================================================


class DefaultLLMFrameworkTranslator:
    """
    Generic, framework-neutral translator implementation.

    Enhanced with:
        - Sophisticated token counting strategies
        - Tool validation
        - Message ordering preservation
        - Tool call translation hooks
    """

    def __init__(self) -> None:
        self._tool_validator = ToolValidator()
        self._metrics = TranslatorMetrics()

    def to_normalized_messages(
        self,
        raw_messages: Any,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> List[NormalizedMessage]:
        start_time = time.time()
        try:
            # Accept a single mapping as a single message
            if isinstance(raw_messages, Mapping):
                raw_seq: Iterable[Any] = [raw_messages]
            else:
                raw_seq = raw_messages

            if not isinstance(raw_seq, Iterable) or isinstance(raw_seq, (str, bytes)):
                self._metrics.record_validation_error()
                raise BadRequest(
                    "raw_messages must be a mapping or iterable of messages",
                    code="BAD_MESSAGES",
                )

            messages: List[NormalizedMessage] = []

            for idx, m in enumerate(raw_seq):
                if isinstance(m, NormalizedMessage):
                    messages.append(m)
                    continue

                if isinstance(m, Mapping):
                    try:
                        messages.append(from_generic_dict(m))
                    except Exception:
                        self._metrics.record_normalization_error()
                        raise BadRequest(
                            f"raw_messages[{idx}] could not be normalized",
                            code="BAD_MESSAGE_FORMAT",
                            details={"index": idx, "type": type(m).__name__},
                        )
                    continue

                self._metrics.record_validation_error()
                raise BadRequest(
                    f"raw_messages[{idx}] must be a NormalizedMessage or mapping",
                    code="BAD_MESSAGES",
                    details={"index": idx, "type": type(m).__name__},
                )

            if not messages:
                self._metrics.record_validation_error()
                raise BadRequest(
                    "raw_messages must contain at least one message",
                    code="BAD_MESSAGES",
                )

            self._metrics.record_translation(time.time() - start_time, success=True)
            return messages
        except Exception:
            self._metrics.record_translation(time.time() - start_time, success=False)
            raise

    def build_system_message(
        self,
        normalized_messages: List[NormalizedMessage],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Tuple[Optional[str], List[NormalizedMessage]]:
        """
        Enhanced system message extraction that preserves message ordering.

        Only extracts system messages that appear at the beginning of the
        conversation, maintaining the semantic structure for models that
        care about message order.
        """
        system_message: Optional[str] = None
        remaining: List[NormalizedMessage] = []

        # Only extract system messages from the beginning of the conversation
        extracting_system = True
        for msg in normalized_messages:
            if extracting_system and msg.role == "system":
                if system_message is None:
                    system_message = msg.content
                else:
                    # Append to existing system message for multiple system messages at start
                    system_message += "\n" + msg.content
            else:
                # Once we hit a non-system message, stop extracting
                extracting_system = False
                remaining.append(msg)

        return system_message, remaining

    def build_tools(
        self,
        raw_tools: Optional[Any],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Optional[List[Dict[str, Any]]]:
        if raw_tools is None:
            return None

        if isinstance(raw_tools, Mapping):
            raw_tools = [raw_tools]

        if not isinstance(raw_tools, Sequence):
            self._metrics.record_validation_error()
            raise BadRequest(
                "tools must be a mapping or a sequence of mappings",
                code="BAD_TOOLS",
            )

        tools: List[Dict[str, Any]] = []
        for idx, t in enumerate(raw_tools):
            if not isinstance(t, Mapping):
                self._metrics.record_validation_error()
                raise BadRequest(
                    f"tools[{idx}] must be a mapping",
                    code="BAD_TOOLS",
                    details={"index": idx, "type": type(t).__name__},
                )

            # Enhanced tool validation
            try:
                self._tool_validator.validate_tool_schema(dict(t))
                self._metrics.record_tool_translation()
            except BadRequest as e:
                self._metrics.record_validation_error()
                # Add context about which tool failed
                e.details = e.details or {}
                e.details["tool_index"] = idx
                raise

            tools.append(dict(t))

        return tools

    def build_tool_choice(
        self,
        raw_tool_choice: Optional[Any],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Optional[Union[str, Dict[str, Any]]]:
        if raw_tool_choice is None:
            return None

        # Enhanced tool choice validation
        try:
            self._tool_validator.validate_tool_choice(raw_tool_choice)
        except BadRequest:
            self._metrics.record_validation_error()
            raise

        if isinstance(raw_tool_choice, str):
            return raw_tool_choice

        if isinstance(raw_tool_choice, Mapping):
            return dict(raw_tool_choice)

        self._metrics.record_validation_error()
        raise BadRequest(
            "tool_choice must be a string or mapping",
            code="BAD_TOOL_CHOICE",
            details={"type": type(raw_tool_choice).__name__},
        )

    def from_completion(
        self,
        completion: LLMCompletion,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: return a neutral dict compatible with JSON.
        """
        self._metrics.record_completion_translation()
        return {
            "text": completion.text,
            "model": completion.model,
            "model_family": completion.model_family,
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            },
            "finish_reason": completion.finish_reason,
            "tool_calls": [asdict(tc) for tc in completion.tool_calls],
        }

    def from_chunk(
        self,
        chunk: LLMChunk,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: return a neutral dict per streaming chunk.
        """
        self._metrics.record_chunk_translation()
        usage = chunk.usage_so_far
        return {
            "text": chunk.text,
            "is_final": chunk.is_final,
            "model": chunk.model,
            "usage_so_far": (
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if usage is not None
                else None
            ),
            "tool_calls": [asdict(tc) for tc in chunk.tool_calls],
        }

    def from_count_tokens(
        self,
        token_count: int,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: return the integer count directly.
        """
        return int(token_count)

    def from_health(
        self,
        health: Mapping[str, Any],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: shallow copy of health mapping.
        """
        return dict(health)

    def from_capabilities(
        self,
        caps: LLMCapabilities,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: capabilities as a plain dict via asdict().
        """
        return asdict(caps)

    def translate_tool_calls_to_framework(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Any:
        """
        Default: return tool calls as-is (list of dicts).
        Framework-specific translators can convert to native objects.
        """
        return tool_calls

    def translate_tool_outputs_from_framework(
        self,
        tool_outputs: Any,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> List[Dict[str, Any]]:
        """
        Default: assume tool_outputs is already in protocol format.
        Framework-specific translators can convert from native objects.
        """
        if tool_outputs is None:
            return []
        if isinstance(tool_outputs, list):
            return tool_outputs
        return [tool_outputs] if isinstance(tool_outputs, dict) else []

    def preferred_model(
        self,
        *,
        op_ctx: OperationContext,
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> Optional[str]:
        """
        Default: derive model from context attrs if present, else None.
        """
        attrs = op_ctx.attrs or {}
        candidate = attrs.get("llm_model") or attrs.get("model")
        if candidate is None:
            return None
        value = str(candidate).strip()
        return value or None

    def decorate_messages_before_send(
        self,
        messages: List[NormalizedMessage],
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> List[NormalizedMessage]:
        """
        Default: no-op, return messages unchanged.
        """
        return list(messages)

    def get_token_counting_config(
        self,
        *,
        op_ctx: OperationContext,  # noqa: ARG002
        framework_ctx: Optional[Any] = None,  # noqa: ARG002
    ) -> TokenCountingConfig:
        """
        Default: use simple token counting strategy.
        Framework-specific translators can override for provider-specific formats.
        """
        return TokenCountingConfig()

    def get_metrics(self) -> TranslatorMetrics:
        """
        Get current translation metrics for observability.
        """
        return self._metrics


# =============================================================================
# Enhanced LLM Translator Orchestrator
# =============================================================================


class LLMTranslator:
    """
    Framework-agnostic orchestrator for LLM operations.

    Enhanced with:
        - Sophisticated token counting with multiple strategies
        - LLM post-processing (safety filtering, JSON repair, etc.)
        - Comprehensive metrics collection
        - Enhanced tool validation and translation
        - Improved streaming ergonomics
    """

    def __init__(
        self,
        *,
        adapter: LLMProtocolV1,
        framework: str = "generic",
        translator: Optional[LLMFrameworkTranslator] = None,
        post_processing_config: Optional[LLMPostProcessingConfig] = None,
    ) -> None:
        self._adapter = adapter
        self._framework = framework
        self._translator: LLMFrameworkTranslator = translator or DefaultLLMFrameworkTranslator()
        self._post_processing_config = post_processing_config or LLMPostProcessingConfig()
        self._safety_filter = SafetyFilter()
        self._json_repair = JSONRepair()

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

        Supports multiple formatting approaches for different LLM providers.
        """
        parts: List[str] = []

        # Add token overhead for special tokens if configured
        special_tokens_overhead = 0
        if token_config.add_special_tokens:
            if token_config.format_strategy == "openai_chatml":
                special_tokens_overhead = 20  # Approximate overhead for ChatML tokens
            elif token_config.format_strategy == "anthropic":
                special_tokens_overhead = 10  # Approximate overhead for Anthropic formatting

        if token_config.format_strategy == "simple":
            # Simple "role: content" format
            if system_message and not token_config.include_system_in_messages:
                parts.append(f"system:{system_message}")
            for msg in normalized_messages:
                parts.append(f"{msg.role}:{msg.content}")

        elif token_config.format_strategy == "openai_chatml":
            # OpenAI ChatML template format
            if system_message:
                parts.append(f"<|im_start|>system\n{system_message}<|im_end|>")
            for msg in normalized_messages:
                parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")

        elif token_config.format_strategy == "anthropic":
            # Anthropic-specific formatting
            if system_message:
                parts.append(f"System: {system_message}")
            for msg in normalized_messages:
                role_display = "Human" if msg.role == "user" else "Assistant"
                parts.append(f"\n\n{role_display}: {msg.content}")

        elif token_config.format_strategy == "custom" and token_config.custom_format_template:
            # Custom template formatting
            template = token_config.custom_format_template
            if system_message and not token_config.include_system_in_messages:
                parts.append(template.format(role="system", content=system_message))
            for msg in normalized_messages:
                parts.append(template.format(role=msg.role, content=msg.content))

        else:
            # Fallback to simple format
            if system_message and not token_config.include_system_in_messages:
                parts.append(f"system:{system_message}")
            for msg in normalized_messages:
                parts.append(f"{msg.role}:{msg.content}")

        formatted_text = "\n".join(parts)
        
        # Add special tokens overhead as whitespace (simplified approach)
        if token_config.add_special_tokens and special_tokens_overhead > 0:
            formatted_text += " " * special_tokens_overhead

        return formatted_text

    def _apply_post_processing(self, completion: LLMCompletion) -> LLMCompletion:
        """
        Apply post-processing to LLM completions (safety, formatting, etc.).
        """
        if not self._post_processing_config.enabled:
            return completion

        result = completion

        # Apply safety filtering
        if self._post_processing_config.safety_filter:
            result = self._apply_safety_filtering(result)

        # Apply JSON repair for tool calls
        if self._post_processing_config.json_repair:
            result = self._apply_json_repair(result)

        # Apply output formatting
        if self._post_processing_config.output_format:
            result = self._apply_output_formatting(result)

        # Apply length truncation
        if self._post_processing_config.max_length:
            result = self._apply_length_truncation(result)

        # Apply custom processors
        if self._post_processing_config.custom_processors:
            for processor in self._post_processing_config.custom_processors:
                result = processor(result)

        return result

    def _apply_safety_filtering(self, completion: LLMCompletion) -> LLMCompletion:
        """Apply comprehensive safety/content filtering."""
        filtered_text, violations = self._safety_filter.filter_content(completion.text)
        
        # Log safety violations
        if violations:
            LOG.warning(
                "Safety filtering applied to completion: %s",
                "; ".join(violations)
            )

        # Repair tool calls if safety filtering was applied
        repaired_tool_calls = completion.tool_calls
        if violations and completion.tool_calls:
            # Re-parse tool calls if text was modified
            try:
                # This is a simplified approach - in practice you might want more sophisticated repair
                repaired_tool_calls = completion.tool_calls
            except Exception:
                LOG.warning("Failed to repair tool calls after safety filtering")
                # Clear tool calls if we can't repair them safely
                repaired_tool_calls = []

        return LLMCompletion(
            text=filtered_text,
            model=completion.model,
            model_family=completion.model_family,
            usage=completion.usage,
            finish_reason=completion.finish_reason,
            tool_calls=repaired_tool_calls,
        )

    def _apply_json_repair(self, completion: LLMCompletion) -> LLMCompletion:
        """Attempt to repair malformed JSON in tool calls and content."""
        repaired_tool_calls = []
        
        # Repair tool calls
        for tool_call in completion.tool_calls:
            try:
                # Access tool_call.function.arguments for JSON arguments
                original_args = tool_call.function.arguments
                repaired_args = self._json_repair.repair_json(original_args)
                parsed_args = json.loads(repaired_args)
                
                # Create repaired tool call
                repaired_tool_call = type(tool_call)(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=type(tool_call.function)(
                        name=tool_call.function.name,
                        arguments=json.dumps(parsed_args)
                    )
                )
                repaired_tool_calls.append(repaired_tool_call)
                
            except (json.JSONDecodeError, AttributeError) as e:
                # If repair fails, keep the original (might fail downstream)
                LOG.warning("Failed to repair JSON in tool call %s: %s", tool_call.id, str(e))
                repaired_tool_calls.append(tool_call)

        # Repair JSON in text content if it looks like JSON
        repaired_text = completion.text
        if completion.text.strip().startswith(('{', '[')):
            try:
                json.loads(completion.text)
            except json.JSONDecodeError:
                repaired_json = self._json_repair.repair_json(completion.text)
                try:
                    json.loads(repaired_json)  # Validate repair
                    repaired_text = repaired_json
                    LOG.info("Repaired JSON in completion text")
                except json.JSONDecodeError:
                    LOG.warning("Failed to repair JSON in completion text")

        return LLMCompletion(
            text=repaired_text,
            model=completion.model,
            model_family=completion.model_family,
            usage=completion.usage,
            finish_reason=completion.finish_reason,
            tool_calls=repaired_tool_calls,
        )

    def _apply_output_formatting(self, completion: LLMCompletion) -> LLMCompletion:
        """Apply output format constraints."""
        if self._post_processing_config.output_format == "json":
            # Ensure output is valid JSON
            if not completion.text.strip().startswith(('{', '[')):
                # Try to extract JSON from text
                extracted_json = self._json_repair.extract_and_repair_json(completion.text)
                if extracted_json:
                    formatted_text = json.dumps(extracted_json, indent=2)
                else:
                    # Wrap in JSON object as fallback
                    formatted_text = json.dumps({"text": completion.text}, indent=2)
                return LLMCompletion(
                    text=formatted_text,
                    model=completion.model,
                    model_family=completion.model_family,
                    usage=completion.usage,
                    finish_reason=completion.finish_reason,
                    tool_calls=completion.tool_calls,
                )
        
        elif self._post_processing_config.output_format == "markdown":
            # Ensure text has basic markdown structure if it doesn't already
            if not any(char in completion.text for char in '#*`['):
                formatted_text = f"```text\n{completion.text}\n```"
                return LLMCompletion(
                    text=formatted_text,
                    model=completion.model,
                    model_family=completion.model_family,
                    usage=completion.usage,
                    finish_reason=completion.finish_reason,
                    tool_calls=completion.tool_calls,
                )

        return completion

    def _apply_length_truncation(self, completion: LLMCompletion) -> LLMCompletion:
        """Truncate completion if it exceeds max length."""
        if self._post_processing_config.max_length and len(completion.text) > self._post_processing_config.max_length:
            truncated_text = completion.text[:self._post_processing_config.max_length]
            # Try to truncate at sentence boundary
            last_period = truncated_text.rfind('.')
            last_newline = truncated_text.rfind('\n')
            cutoff = max(last_period, last_newline)
            
            if cutoff > self._post_processing_config.max_length * 0.7:  # Only if we can preserve most content
                truncated_text = truncated_text[:cutoff + 1] + " [truncated]"
            else:
                truncated_text += " [truncated]"
                
            return LLMCompletion(
                text=truncated_text,
                model=completion.model,
                model_family=completion.model_family,
                usage=completion.usage,
                finish_reason="length",
                tool_calls=completion.tool_calls,
            )
        return completion

    # --------------------------------------------------------------------- #
    # Sync Complete (uses AsyncBridge)
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
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None

        async def _complete_coro() -> Any:
            try:
                normalized = self._translator.to_normalized_messages(
                    raw_messages,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                normalized = self._translator.decorate_messages_before_send(
                    normalized,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                auto_system, remaining = self._translator.build_system_message(
                    normalized,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                effective_system = system_message if system_message is not None else auto_system
                tools_corpus = self._translator.build_tools(
                    tools,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                tool_choice_corpus = self._translator.build_tool_choice(
                    tool_choice,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                wire_messages = to_corpus(remaining)
                effective_model = model or self._translator.preferred_model(
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

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
                    raise BadRequest(
                        f"adapter.complete returned unsupported type: {type(result).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )

                # Apply post-processing
                result_processed = self._apply_post_processing(result)

                return self._translator.from_completion(
                    result_processed,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="complete",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        return AsyncBridge.run_async(_complete_coro(), timeout=timeout)

    # --------------------------------------------------------------------- #
    # Async Complete
    # --------------------------------------------------------------------- #

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
        Async complete API.

        Preferred for async applications and services.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            normalized = self._translator.to_normalized_messages(
                raw_messages,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            normalized = self._translator.decorate_messages_before_send(
                normalized,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            auto_system, remaining = self._translator.build_system_message(
                normalized,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            effective_system = system_message if system_message is not None else auto_system
            tools_corpus = self._translator.build_tools(
                tools,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            tool_choice_corpus = self._translator.build_tool_choice(
                tool_choice,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            wire_messages = to_corpus(remaining)
            effective_model = model or self._translator.preferred_model(
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

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
                raise BadRequest(
                    f"adapter.complete returned unsupported type: {type(result).__name__}",
                    code="BAD_ADAPTER_RESULT",
                )

            # Apply post-processing
            result_processed = self._apply_post_processing(result)

            return self._translator.from_completion(
                result_processed,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="complete",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Enhanced Sync Stream (uses SyncStreamBridge)
    # --------------------------------------------------------------------- #

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
        Enhanced synchronous streaming API.

        Returns a sync iterator that yields framework-level streaming chunks
        by bridging the async adapter.stream(...) via SyncStreamBridge.
        Provides better ergonomics and error handling.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        async def _stream_factory() -> AsyncIterator[Any]:
            try:
                normalized = self._translator.to_normalized_messages(
                    raw_messages,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                normalized = self._translator.decorate_messages_before_send(
                    normalized,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                auto_system, remaining = self._translator.build_system_message(
                    normalized,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                effective_system = system_message if system_message is not None else auto_system
                tools_corpus = self._translator.build_tools(
                    tools,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                tool_choice_corpus = self._translator.build_tool_choice(
                    tool_choice,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                wire_messages = to_corpus(remaining)
                effective_model = model or self._translator.preferred_model(
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                agen = self._adapter.stream(
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

                async for chunk in agen:
                    if not isinstance(chunk, LLMChunk):
                        raise BadRequest(
                            f"adapter.stream yielded unsupported type: {type(chunk).__name__}",
                            code="BAD_ADAPTER_RESULT",
                        )
                    yield self._translator.from_chunk(
                        chunk,
                        op_ctx=ctx,
                        framework_ctx=framework_ctx,
                    )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="stream",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        bridge = SyncStreamBridge(
            coro_factory=_stream_factory,
            framework=self._framework,
            error_context={
                "operation": "llm.stream",
                "request_id": ctx.request_id,
                "tenant": ctx.tenant,
            },
        )
        return bridge.run()

    # --------------------------------------------------------------------- #
    # Async Stream
    # --------------------------------------------------------------------- #

    async def arun_stream(
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
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            normalized = self._translator.to_normalized_messages(
                raw_messages,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            normalized = self._translator.decorate_messages_before_send(
                normalized,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            auto_system, remaining = self._translator.build_system_message(
                normalized,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            effective_system = system_message if system_message is not None else auto_system
            tools_corpus = self._translator.build_tools(
                tools,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            tool_choice_corpus = self._translator.build_tool_choice(
                tool_choice,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            wire_messages = to_corpus(remaining)
            effective_model = model or self._translator.preferred_model(
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            agen = self._adapter.stream(
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

            async for chunk in agen:
                if not isinstance(chunk, LLMChunk):
                    raise BadRequest(
                        f"adapter.stream yielded unsupported type: {type(chunk).__name__}",
                        code="BAD_ADAPTER_RESULT",
                    )
                yield self._translator.from_chunk(
                    chunk,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="stream",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Enhanced Token Counting with Sophisticated Formatting
    # --------------------------------------------------------------------- #

    def count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous count_tokens wrapper around adapter.count_tokens().
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None

        async def _count_coro() -> Any:
            try:
                result = await self._adapter.count_tokens(
                    text=text,
                    model=model,
                    ctx=ctx,
                )
                return self._translator.from_count_tokens(
                    int(result),
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="count_tokens",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        return AsyncBridge.run_async(_count_coro(), timeout=timeout)

    async def arun_count_tokens(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async count_tokens wrapper.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            result = await self._adapter.count_tokens(
                text=text,
                model=model,
                ctx=ctx,
            )
            return self._translator.from_count_tokens(
                int(result),
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="count_tokens",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Enhanced Helper: count_tokens for messages with sophisticated formatting
    # --------------------------------------------------------------------- #

    def count_tokens_for_messages(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Enhanced synchronous helper to count tokens for chat messages.

        Uses sophisticated formatting strategies for accurate token counting
        across different LLM providers.
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None

        async def _count_msgs_coro() -> Any:
            try:
                normalized = self._translator.to_normalized_messages(
                    raw_messages,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
                auto_system, remaining = self._translator.build_system_message(
                    normalized,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                # Get framework-specific token counting configuration
                token_config = self._translator.get_token_counting_config(
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )

                # Format messages using sophisticated strategy
                combined = self._format_messages_for_token_counting(
                    remaining, auto_system, token_config
                )

                result = await self._adapter.count_tokens(
                    text=combined,
                    model=model,
                    ctx=ctx,
                )
                return self._translator.from_count_tokens(
                    int(result),
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="count_tokens_for_messages",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        return AsyncBridge.run_async(_count_msgs_coro(), timeout=timeout)

    async def arun_count_tokens_for_messages(
        self,
        raw_messages: Any,
        *,
        model: Optional[str] = None,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Enhanced async helper to count tokens for chat messages.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            normalized = self._translator.to_normalized_messages(
                raw_messages,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
            auto_system, remaining = self._translator.build_system_message(
                normalized,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            # Get framework-specific token counting configuration
            token_config = self._translator.get_token_counting_config(
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )

            # Format messages using sophisticated strategy
            combined = self._format_messages_for_token_counting(
                remaining, auto_system, token_config
            )

            result = await self._adapter.count_tokens(
                text=combined,
                model=model,
                ctx=ctx,
            )
            return self._translator.from_count_tokens(
                int(result),
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="count_tokens_for_messages",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Enhanced Tool Call Translation Methods
    # --------------------------------------------------------------------- #

    def translate_tool_calls_to_framework(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Convert protocol tool calls to framework-native representations.
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        return self._translator.translate_tool_calls_to_framework(
            tool_calls,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

    def translate_tool_outputs_from_framework(
        self,
        tool_outputs: Any,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert framework-native tool outputs back to protocol format.
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        return self._translator.translate_tool_outputs_from_framework(
            tool_outputs,
            op_ctx=ctx,
            framework_ctx=framework_ctx,
        )

    # --------------------------------------------------------------------- #
    # Health (sync + async)
    # --------------------------------------------------------------------- #

    def health(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous health check wrapper.
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None

        async def _health_coro() -> Any:
            try:
                h = await self._adapter.health(ctx=ctx)
                return self._translator.from_health(
                    h,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="health",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        return AsyncBridge.run_async(_health_coro(), timeout=timeout)

    async def arun_health(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async health check wrapper.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            h = await self._adapter.health(ctx=ctx)
            return self._translator.from_health(
                h,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="health",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Capabilities (sync + async)
    # --------------------------------------------------------------------- #

    def capabilities(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Synchronous capabilities wrapper.

        Returns framework-level capabilities derived from LLMCapabilities.
        """
        ctx = _ensure_llm_operation_context(op_ctx)
        timeout = ctx.deadline_ms / 1000.0 if ctx.deadline_ms else None

        async def _caps_coro() -> Any:
            try:
                caps = await self._adapter.capabilities()
                return self._translator.from_capabilities(
                    caps,
                    op_ctx=ctx,
                    framework_ctx=framework_ctx,
                )
            except Exception as exc:
                attach_context(
                    exc,
                    framework=self._framework,
                    llm_operation="capabilities",
                    request_id=ctx.request_id,
                    tenant=ctx.tenant,
                )
                raise

        return AsyncBridge.run_async(_caps_coro(), timeout=timeout)

    async def arun_capabilities(
        self,
        *,
        op_ctx: Optional[Union[OperationContext, Mapping[str, Any]]] = None,
        framework_ctx: Optional[Any] = None,
    ) -> Any:
        """
        Async capabilities wrapper.
        """
        ctx = _ensure_llm_operation_context(op_ctx)

        try:
            caps = await self._adapter.capabilities()
            return self._translator.from_capabilities(
                caps,
                op_ctx=ctx,
                framework_ctx=framework_ctx,
            )
        except Exception as exc:
            attach_context(
                exc,
                framework=self._framework,
                llm_operation="capabilities",
                request_id=ctx.request_id,
                tenant=ctx.tenant,
            )
            raise

    # --------------------------------------------------------------------- #
    # Metrics Access
    # --------------------------------------------------------------------- #

    def get_metrics(self) -> Optional[TranslatorMetrics]:
        """
        Get translation metrics for observability.

        Returns metrics if the translator supports metrics collection.
        """
        if hasattr(self._translator, 'get_metrics'):
            return self._translator.get_metrics()
        return None


# =============================================================================
# Comprehensive Registry for per-framework translators
# =============================================================================


_TranslatorFactory = Callable[[LLMProtocolV1], LLMFrameworkTranslator]
_LLM_TRANSLATOR_FACTORIES: Dict[str, _TranslatorFactory] = {}


def register_llm_translator(
    framework: str,
    factory: _TranslatorFactory,
) -> None:
    """
    Register or override an LLMFrameworkTranslator factory for a given framework.

    Example
    -------
        def make_langchain_llm_translator(adapter: LLMProtocolV1) -> LLMFrameworkTranslator:
            return LangChainLLMTranslator(adapter=adapter)

        register_llm_translator("langchain", make_langchain_llm_translator)
    """
    if not framework or not isinstance(framework, str):
        raise BadRequest(
            "framework name must be a non-empty string",
            code="BAD_TRANSLATOR_REGISTRATION",
        )
    if not callable(factory):
        raise BadRequest(
            "translator factory must be callable",
            code="BAD_TRANSLATOR_REGISTRATION",
        )
    _LLM_TRANSLATOR_FACTORIES[framework] = factory
    LOG.debug("Registered LLM translator factory for framework=%s", framework)


def get_llm_translator_factory(framework: str) -> Optional[_TranslatorFactory]:
    """
    Return a previously registered LLMFrameworkTranslator factory for a framework, if any.
    """
    return _LLM_TRANSLATOR_FACTORIES.get(framework)


def create_llm_translator(
    *,
    adapter: LLMProtocolV1,
    framework: str = "generic",
    translator: Optional[LLMFrameworkTranslator] = None,
    post_processing_config: Optional[LLMPostProcessingConfig] = None,
) -> LLMTranslator:
    """
    Convenience helper to construct an LLMTranslator for a given framework.

    Enhanced with post-processing configuration support.

    Behavior:
        - If `translator` is provided explicitly, it is used as-is.
        - Else, if a factory is registered for `framework`, it is used.
        - Else, DefaultLLMFrameworkTranslator is used.
    """
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
    )


__all__ = [
    "TokenCountingConfig",
    "ToolValidator",
    "LLMPostProcessingConfig",
    "TranslatorMetrics",
    "SafetyFilter",
    "JSONRepair",
    "LLMFrameworkTranslator",
    "DefaultLLMFrameworkTranslator",
    "LLMTranslator",
    "register_llm_translator",
    "get_llm_translator_factory",
    "create_llm_translator",
]