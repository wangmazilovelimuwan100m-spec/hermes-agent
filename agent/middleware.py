"""Middleware pipeline framework for Hermes Agent.

Provides the base Middleware class, MiddlewareState dataclass for passing
state through the pipeline, and MiddlewarePipeline for managing middleware
execution at each lifecycle hook.

Hooks:
    before_llm  — called before each LLM API call
    after_llm   — called after each LLM API call (response received)
    before_tool — called before each tool execution
    after_tool  — called after each tool execution

Based on the DeerFlow middleware architecture design (bytedance/deer-flow).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MiddlewareState:
    """Shared state bag passed through all middleware hooks.

    Attributes:
        messages: The conversation message list (mutable in-place).
        tool_calls: Current tool calls from the LLM response (after_llm / before_tool).
        tool_call: The individual tool call being processed (before_tool / after_tool).
        tool_result: The result of the tool execution (after_tool).
        api_call_count: Current iteration number in the agent loop.
        api_response: The raw LLM response object (after_llm).
        api_error: The exception from a failed LLM call (after_llm, when retryable).
        model: Model name being used.
        session_id: Current session identifier.
        task_id: Task ID for VM/workspace isolation.
        extra: Arbitrary key-value store for middleware to pass data between hooks.
        aborted: If True, the pipeline short-circuits (e.g. loop detection injected a message).
        retry: If True, the agent loop should retry the current iteration.
        skip_tool: If True, the current tool call should be skipped.
    """

    messages: list[dict[str, Any]]
    api_call_count: int = 0
    tool_calls: list[Any] = field(default_factory=list)
    tool_call: Optional[Any] = None
    tool_result: Optional[str] = None
    api_response: Optional[Any] = None
    api_error: Optional[Exception] = None
    model: str = ""
    session_id: str = ""
    task_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    aborted: bool = False
    retry: bool = False
    skip_tool: bool = False


class Middleware:
    """Base class for all middleware.

    Subclasses declare which hooks they participate in via the ``hooks``
    class attribute, then override the corresponding methods.  Only declared
    hooks are called — un-overridden methods on the base class are no-ops.

    Example::

        class MyMiddleware(Middleware):
            hooks = ["before_llm", "after_tool"]

            async def before_llm(self, state: MiddlewareState) -> MiddlewareState:
                # inspect or modify state.messages
                return state

            async def after_tool(self, state: MiddlewareState) -> MiddlewareState:
                # inspect tool_result, set state.skip_tool, etc.
                return state
    """

    hooks: list[str] = []

    # Human-readable name for logging / diagnostics.
    name: str = ""

    # If True, an exception from this middleware will propagate up through
    # the pipeline instead of being silently swallowed.  Use for critical
    # middleware where silent failure is worse than crashing (e.g. loop
    # detection, security audit).
    fail_fast: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if not cls.name:
            cls.name = cls.__name__
        super().__init_subclass__(**kwargs)

    async def before_llm(self, state: MiddlewareState) -> MiddlewareState:
        return state

    async def after_llm(self, state: MiddlewareState) -> MiddlewareState:
        return state

    async def before_tool(self, state: MiddlewareState) -> MiddlewareState:
        return state

    async def after_tool(self, state: MiddlewareState) -> MiddlewareState:
        return state


class MiddlewarePipeline:
    """Ordered pipeline that runs middleware at each lifecycle hook.

    Middleware are sorted by priority (lower number = earlier execution)
    within each hook bucket.  Middleware without an explicit ``priority``
    attribute default to 100.
    """

    _HOOK_NAMES = ("before_llm", "after_llm", "before_tool", "after_tool")

    def __init__(self, middlewares: list[Middleware]) -> None:
        # Bucket middleware by declared hook, sorted by priority.
        self._buckets: dict[str, list[Middleware]] = {h: [] for h in self._HOOK_NAMES}
        self._all: list[Middleware] = list(middlewares)
        for m in middlewares:
            prio = getattr(m, "priority", 100)
            for hook in m.hooks:
                if hook in self._buckets:
                    self._buckets[hook].append(m)
        # Stable sort by priority
        for hook in self._buckets:
            self._buckets[hook].sort(key=lambda mw: getattr(mw, "priority", 100))

    # ── Public runner methods ─────────────────────────────────────────

    async def run_before_llm(self, state: MiddlewareState) -> MiddlewareState:
        return await self._run_chain(self._buckets["before_llm"], "before_llm", state)

    async def run_after_llm(self, state: MiddlewareState) -> MiddlewareState:
        return await self._run_chain(self._buckets["after_llm"], "after_llm", state)

    async def run_before_tool(self, state: MiddlewareState) -> MiddlewareState:
        return await self._run_chain(self._buckets["before_tool"], "before_tool", state)

    async def run_after_tool(self, state: MiddlewareState) -> MiddlewareState:
        return await self._run_chain(self._buckets["after_tool"], "after_tool", state)

    # ── Synchronous wrappers ────────────────────────────────────────
    # The agent loop in run_agent.py is synchronous, so these wrappers
    # provide a safe way to invoke async middleware from sync code.

    def run_before_llm_sync(self, state: MiddlewareState) -> MiddlewareState:
        return self._run_sync(self.run_before_llm, state)

    def run_after_llm_sync(self, state: MiddlewareState) -> MiddlewareState:
        return self._run_sync(self.run_after_llm, state)

    def run_before_tool_sync(self, state: MiddlewareState) -> MiddlewareState:
        return self._run_sync(self.run_before_tool, state)

    def run_after_tool_sync(self, state: MiddlewareState) -> MiddlewareState:
        return self._run_sync(self.run_after_tool, state)

    @staticmethod
    def _run_sync(coro_fn, state: MiddlewareState) -> MiddlewareState:
        """Run an async coroutine from synchronous code.

        If an event loop is already running (gateway context), we avoid the
        nested ``asyncio.run()`` pattern which is fragile and loses contextvars.
        Instead, since all current middleware hooks are pure computation with no
        real async I/O, we use ``loop.run_until_complete()`` on a fresh loop in
        a helper thread.  This preserves the calling loop's state while safely
        executing the coroutine.

        When no event loop is running (CLI context), ``asyncio.run()`` is used
        directly.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Gateway context — avoid asyncio.run() which would fail with a
            # nested-loop error.  Run the coroutine on a *new* event loop in
            # a separate thread (safe because middleware has no real async I/O
            # that depends on the outer loop's state).
            import concurrent.futures
            import threading

            result_box: list[MiddlewareState] = []
            error_box: list[Exception] = []

            def _thread_target():
                try:
                    result_box.append(asyncio.run(coro_fn(state)))
                except Exception as exc:
                    error_box.append(exc)

            t = threading.Thread(target=_thread_target, daemon=True)
            t.start()
            t.join(timeout=30)
            if error_box:
                raise error_box[0]
            if not result_box:
                raise TimeoutError("Middleware sync wrapper timed out after 30s")
            return result_box[0]
        else:
            return asyncio.run(coro_fn(state))

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    async def _run_chain(
        middlewares: list[Middleware], hook_name: str, state: MiddlewareState
    ) -> MiddlewareState:
        for m in middlewares:
            try:
                handler = getattr(m, hook_name)
                state = await handler(state)
            except Exception:
                if getattr(m, "fail_fast", False):
                    logger.exception(
                        "Middleware %s raised in %s — fail_fast enabled, propagating",
                        m.name, hook_name,
                    )
                    raise
                logger.exception(
                    "Middleware %s raised in %s — skipping", m.name, hook_name
                )
        return state

    def __repr__(self) -> str:
        names = [m.name for m in self._all]
        return f"MiddlewarePipeline({names})"


def build_default_pipeline() -> MiddlewarePipeline:
    """Build the default middleware pipeline with all built-in middleware.

    This is the single factory used by AIAgent.__init__ to create the
    pipeline.  Session-scoped state (e.g. loop-detection windows, retry
    counters) is managed internally by each middleware using the
    ``session_id`` from ``MiddlewareState``.
    """
    from agent.middlewares.loop_detection import LoopDetectionMiddleware
    from agent.middlewares.error_recovery import ErrorRecoveryMiddleware
    from agent.middlewares.tool_audit import ToolAuditMiddleware

    return MiddlewarePipeline([
        LoopDetectionMiddleware(),
        ErrorRecoveryMiddleware(),
        ToolAuditMiddleware(),
    ])
