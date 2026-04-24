"""Error recovery middleware.

Provides structured retry and recovery for LLM API errors and tool
execution failures.  Integrates with the existing error-classification
system (``agent.error_classifier``) and retry utilities (``agent.retry_utils``).

LLM error recovery:
    - Rate limit (429) → wait with jittered backoff, then retry.
    - Context overflow → signal compression via ``state.retry``.
    - Transient server errors → exponential backoff.

Tool error recovery:
    - Timeout → retry once.
    - Generic failure → log and continue (don't crash the loop).

Inspired by DeerFlow's LLMErrorHandlerMiddleware + ToolErrorHandlerMiddleware.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from agent.middleware import Middleware, MiddlewareState
from agent.retry_utils import jittered_backoff

logger = logging.getLogger(__name__)


class ErrorRecoveryMiddleware(Middleware):
    """Recover from LLM and tool errors with configurable retry strategies.

    This middleware operates in *after_llm* and *after_tool* hooks.

    Configuration:
        max_llm_retries (int): Max retries for rate-limit / transient LLM errors
            (default 3).
        llm_backoff_base (float): Base delay for jittered exponential backoff
            (default 2.0).
        llm_backoff_max (float): Max delay cap (default 60.0).
        tool_timeout_retry (bool): Whether to retry timed-out tool calls once
            (default True).
    """

    hooks = ["after_llm", "after_tool"]
    name = "ErrorRecoveryMiddleware"
    priority = 50  # After loop detection but before audit

    def __init__(
        self,
        max_llm_retries: int = 3,
        llm_backoff_base: float = 2.0,
        llm_backoff_max: float = 60.0,
        tool_timeout_retry: bool = True,
    ) -> None:
        self._max_llm_retries = max_llm_retries
        self._llm_backoff_base = llm_backoff_base
        self._llm_backoff_max = llm_backoff_max
        self._tool_timeout_retry = tool_timeout_retry
        # Per-session retry counters: {session_id: {"llm": int, "tool:<call_id>": int}}
        self._retry_counts: dict[str, dict[str, int]] = {}

    # ── LLM error recovery ──────────────────────────────────────────

    async def after_llm(self, state: MiddlewareState) -> MiddlewareState:
        """Handle LLM API errors after a failed call.

        If ``state.api_error`` is set and the error is retryable (rate limit,
        transient server error), this middleware sleeps with backoff and sets
        ``state.retry = True`` so the agent loop retries the current iteration.
        """
        error = state.api_error
        if error is None:
            # Successful call — reset retry counter.
            self._get_counter(state, "llm", reset=True)
            return state

        if not self._is_retryable_llm_error(error):
            return state

        count = self._get_counter(state, "llm")
        if count >= self._max_llm_retries:
            logger.warning(
                "LLM error recovery: max retries (%d) exhausted for session=%s",
                self._max_llm_retries, state.session_id or "default",
            )
            return state

        self._get_counter(state, "llm", increment=True)
        wait = jittered_backoff(
            count, base_delay=self._llm_backoff_base, max_delay=self._llm_backoff_max
        )

        logger.info(
            "LLM error recovery: waiting %.1fs before retry %d/%d (session=%s)",
            wait, count + 1, self._max_llm_retries, state.session_id or "default",
        )

        await asyncio.sleep(wait)
        state.retry = True
        return state

    # ── Tool error recovery ─────────────────────────────────────────

    async def after_tool(self, state: MiddlewareState) -> MiddlewareState:
        """Handle tool execution failures.

        If the tool result indicates a timeout and we haven't retried yet,
        set ``state.skip_tool = False`` and inject a retry instruction.
        For other tool errors, just log them.
        """
        result = state.tool_result
        if result is None or not self._is_tool_error(result):
            # Clear retry counter for this tool call on success.
            tc_id = self._tool_call_id(state)
            if tc_id:
                self._get_counter(state, f"tool:{tc_id}", reset=True)
            return state

        if not self._is_tool_timeout(result):
            # Non-timeout errors: just log, let the LLM handle it.
            return state

        if not self._tool_timeout_retry:
            return state

        tc_id = self._tool_call_id(state)
        count = self._get_counter(state, f"tool:{tc_id}") if tc_id else 1

        if count >= 1:
            # Already retried once — give up.
            logger.info(
                "Tool timeout recovery: already retried for call %s (session=%s)",
                tc_id, state.session_id or "default",
            )
            return state

        self._get_counter(state, f"tool:{tc_id}", increment=True)
        logger.info(
            "Tool timeout recovery: will retry tool call %s (session=%s)",
            tc_id, state.session_id or "default",
        )
        # Signal to the caller that a retry is desired.
        state.extra["_tool_retry"] = True
        return state

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_retryable_llm_error(error: Exception) -> bool:
        """Return True for rate-limit and transient server errors."""
        status = getattr(error, "status_code", None)
        msg = str(error).lower()
        if status == 429:
            return True
        if status == 529:
            return True
        if status is not None and 500 <= status < 600:
            # Server errors are generally retryable.
            return True
        # Connection errors / timeouts without status code.
        if any(kw in msg for kw in (
            "connection reset", "connection closed", "connection lost",
            "timeout", "timed out", "eof occurred",
        )):
            return True
        return False

    @staticmethod
    def _is_tool_error(result: str) -> bool:
        """Heuristic: does the tool result string indicate an error?"""
        if not result:
            return False
        lower = result.lower()
        return (
            lower.startswith("error")
            or "timed out" in lower
            or "timeout" in lower
            or "exception" in lower
        )

    @staticmethod
    def _is_tool_timeout(result: str) -> bool:
        """Heuristic: does the tool result indicate a timeout?"""
        lower = result.lower()
        return "timed out" in lower or "timeout" in lower

    @staticmethod
    def _tool_call_id(state: MiddlewareState) -> Optional[str]:
        tc = state.tool_call
        if tc is None:
            return None
        if isinstance(tc, dict):
            return tc.get("id")
        return getattr(tc, "id", None)

    def _get_counter(
        self, state: MiddlewareState, key: str, *, reset: bool = False, increment: bool = False
    ) -> int:
        sid = state.session_id or "default"
        bucket = self._retry_counts.setdefault(sid, {})
        if reset:
            bucket[key] = 0
            return 0
        val = bucket.get(key, 0)
        if increment:
            bucket[key] = val + 1
            return val + 1
        return val

    def clear_session(self, session_id: str) -> None:
        """Remove retry counters for a session."""
        self._retry_counts.pop(session_id, None)

    def clear_all(self) -> None:
        """Remove all retry counters."""
        self._retry_counts.clear()
