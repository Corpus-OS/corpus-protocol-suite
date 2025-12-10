# corpus_sdk/vector/framework_adapters/mcp_vector_service.py
# SPDX-License-Identifier: Apache-2.0

"""
MCP-facing vector service for Corpus Vector protocol.

This is a *thin* integration layer between:

- MCP tools implemented in the MCP server (typically JS), and
- Corpus `VectorProtocolV1` adapters via `VectorTranslator`.

Design
------

- Mirrors the patterns used by other MCP services (LLM, Graph, etc.).
- Async-only API for MCP server to call.
- Does *not* implement:
    * MMR
    * Embeddings
    * Business logic
    * Metadata envelopes
    * Length / shape validation

All heavy lifting is delegated to:

- The underlying `VectorProtocolV1` adapter
- The shared `VectorTranslator` (batching, limits, error handling, etc.)

This layer focuses on:

- Building `OperationContext` via `context_translation.from_mcp`
- Normalizing framework context for vector operations
- Adding rich error context via `attach_context` + `VectorCoercionErrorCodes`
- Keeping the request/response payloads as generic `dict`/`Any` so
  the translator defines the contract.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, Dict, Mapping, Optional

from corpus_sdk.core.context_translation import (
    from_mcp as ctx_from_mcp,
    from_dict as ctx_from_dict,
)
from corpus_sdk.core.error_context import attach_context
from corpus_sdk.vector.vector_base import (
    VectorProtocolV1,
    OperationContext,
)
from corpus_sdk.vector.framework_adapters.common.vector_translation import (
    DefaultVectorFrameworkTranslator,
    VectorTranslator,
)
from corpus_sdk.vector.framework_adapters.common.framework_utils import (
    VectorCoercionErrorCodes,
    VectorResourceLimits,
    VectorValidationFlags,
    TopKWarningConfig,
    normalize_vector_context,
    attach_vector_context_to_framework_ctx,
    warn_if_extreme_k,
)

logger = logging.getLogger(__name__)

_FRAMEWORK_NAME = "mcp"

VECTOR_ERROR_CODES = VectorCoercionErrorCodes(framework_label=_FRAMEWORK_NAME)
VECTOR_LIMITS = VectorResourceLimits()
VECTOR_FLAGS = VectorValidationFlags()
TOPK_WARNING_CONFIG = TopKWarningConfig(framework_label=_FRAMEWORK_NAME)


class MCPVectorService:
    """
    Thin MCP vector service over a Corpus `VectorProtocolV1`.

    The MCP server should:
    - Instantiate this once per adapter.
    - Expose its methods as MCP tools (e.g. `vector.query`, `vector.upsert`, `vector.delete`).

    This class intentionally does *not* reshape inputs/outputs; it expects
    the MCP layer to pass through "raw" vector protocol-style dicts and
    lets `VectorTranslator` define the concrete schema.
    """

    def __init__(
        self,
        *,
        corpus_adapter: VectorProtocolV1,
        namespace: Optional[str] = "default",
    ) -> None:
        self.corpus_adapter = corpus_adapter
        self.namespace = namespace
        self._caps = None  # lazily fetched capabilities

    # ------------------------------------------------------------------ #
    # Translator wiring
    # ------------------------------------------------------------------ #

    @cached_property
    def _translator(self) -> VectorTranslator:
        framework_translator = DefaultVectorFrameworkTranslator()
        return VectorTranslator(
            adapter=self.corpus_adapter,
            framework=_FRAMEWORK_NAME,
            translator=framework_translator,
        )

    # ------------------------------------------------------------------ #
    # Context / framework_ctx helpers
    # ------------------------------------------------------------------ #

    def _effective_namespace(self, overridden: Optional[str], raw_request: Dict[str, Any]) -> Optional[str]:
        """
        Resolve namespace precedence:

        1. Explicit `namespace` argument
        2. `raw_request["namespace"]`
        3. Default `self.namespace`
        """
        if overridden is not None:
            return overridden
        if "namespace" in raw_request:
            ns = raw_request.get("namespace")
            return str(ns) if ns is not None else None
        return self.namespace

    def _build_ctx(
        self,
        *,
        ctx: Optional[OperationContext] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        context_dict: Optional[Mapping[str, Any]] = None,
    ) -> Optional[OperationContext]:
        """
        Build OperationContext with precedence:

        1. Explicit `ctx` (already OperationContext)
        2. `mcp_context` via ctx_from_mcp
        3. `context_dict` via ctx_from_dict
        """
        if isinstance(ctx, OperationContext):
            return ctx

        if mcp_context is not None:
            try:
                maybe = ctx_from_mcp(mcp_context)
                if isinstance(maybe, OperationContext):
                    return maybe
            except Exception as exc:  # noqa: BLE001
                logger.debug("ctx_from_mcp failed: %s", exc)

        if isinstance(context_dict, Mapping):
            try:
                maybe = ctx_from_dict(context_dict)
                if isinstance(maybe, OperationContext):
                    return maybe
            except Exception as exc:  # noqa: BLE001
                logger.debug("ctx_from_dict failed: %s", exc)

        return None

    def _framework_ctx_for_namespace(
        self,
        namespace: Optional[str],
        extra_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a framework_ctx enriched with normalized vector context.
        """
        raw_ctx: Dict[str, Any] = {}
        if namespace is not None:
            raw_ctx["namespace"] = namespace
        if extra_context:
            raw_ctx.update(extra_context)

        vector_ctx = normalize_vector_context(
            raw_ctx,
            framework=_FRAMEWORK_NAME,
            logger=logger,
        )

        framework_ctx: Dict[str, Any] = {}
        attach_vector_context_to_framework_ctx(
            framework_ctx,
            vector_context=vector_ctx,
            limits=VECTOR_LIMITS,
            flags=VECTOR_FLAGS,
        )
        return framework_ctx

    # ------------------------------------------------------------------ #
    # Capabilities (thin wrapper)
    # ------------------------------------------------------------------ #

    async def aget_capabilities(self) -> Any:
        """
        Pass-through capabilities call.

        Returns whatever the translator/adapter returns (shape is adapter-defined).
        """
        try:
            return await self._translator.arun_capabilities()
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework=_FRAMEWORK_NAME,
                operation="vector_aget_capabilities",
                error_codes=VECTOR_ERROR_CODES,
            )
            raise

    # ------------------------------------------------------------------ #
    # Thin async ops: upsert / query / delete
    # ------------------------------------------------------------------ #

    async def aupsert(
        self,
        request: Dict[str, Any],
        *,
        namespace: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        context_dict: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """
        Thin async upsert wrapper.

        `request` should already be shaped as the vector upsert request
        the translator expects (e.g. with `vectors`, `namespace`, etc.).
        """
        raw_request = dict(request or {})
        ns = self._effective_namespace(namespace, raw_request)
        if ns is not None:
            raw_request.setdefault("namespace", ns)

        op_ctx = self._build_ctx(
            ctx=ctx,
            mcp_context=mcp_context,
            context_dict=context_dict,
        )
        framework_ctx = self._framework_ctx_for_namespace(ns)

        try:
            result = await self._translator.arun_upsert(
                raw_request,
                op_ctx=op_ctx,
                framework_ctx=framework_ctx,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework=_FRAMEWORK_NAME,
                operation="vector_aupsert",
                error_codes=VECTOR_ERROR_CODES,
                namespace=ns,
            )
            raise

    async def aquery(
        self,
        request: Dict[str, Any],
        *,
        namespace: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        context_dict: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """
        Thin async query wrapper.

        `request` should be a dict shaped for the translator's query:
            {
                "vector": [...],
                "top_k": int,
                "namespace": Optional[str],
                "filters": Optional[dict],
                "include_metadata": bool,
                "include_vectors": bool,
                ...
            }

        This method:
        - Resolves namespace
        - Normalizes vector context
        - Adds error context
        - Delegates everything else to the translator
        """
        raw_request = dict(request or {})
        ns = self._effective_namespace(namespace, raw_request)
        if ns is not None:
            raw_request.setdefault("namespace", ns)

        # Optional: warn for extreme top_k, following other adapters
        top_k = raw_request.get("top_k")
        if isinstance(top_k, int):
            warn_if_extreme_k(
                top_k,
                framework=_FRAMEWORK_NAME,
                op_name="vector_aquery",
                warning_config=TOPK_WARNING_CONFIG,
                logger=logger,
            )

        op_ctx = self._build_ctx(
            ctx=ctx,
            mcp_context=mcp_context,
            context_dict=context_dict,
        )
        framework_ctx = self._framework_ctx_for_namespace(ns)

        try:
            result = await self._translator.arun_query(
                raw_request,
                op_ctx=op_ctx,
                framework_ctx=framework_ctx,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework=_FRAMEWORK_NAME,
                operation="vector_aquery",
                error_codes=VECTOR_ERROR_CODES,
                namespace=ns,
                top_k=top_k,
            )
            raise

    async def adelete(
        self,
        request: Dict[str, Any],
        *,
        namespace: Optional[str] = None,
        ctx: Optional[OperationContext] = None,
        mcp_context: Optional[Mapping[str, Any]] = None,
        context_dict: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """
        Thin async delete wrapper.

        `request` should be shaped for the translator's delete:
            {
                "ids": Optional[list[str]],
                "namespace": Optional[str],
                "filter": Optional[dict],
                ...
            }
        """
        raw_request = dict(request or {})
        ns = self._effective_namespace(namespace, raw_request)
        if ns is not None:
            raw_request.setdefault("namespace", ns)

        op_ctx = self._build_ctx(
            ctx=ctx,
            mcp_context=mcp_context,
            context_dict=context_dict,
        )
        framework_ctx = self._framework_ctx_for_namespace(ns)

        try:
            result = await self._translator.arun_delete(
                raw_request,
                op_ctx=op_ctx,
                framework_ctx=framework_ctx,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            attach_context(
                exc,
                framework=_FRAMEWORK_NAME,
                operation="vector_adelete",
                error_codes=VECTOR_ERROR_CODES,
                namespace=ns,
            )
            raise


__all__ = [
    "MCPVectorService",
]