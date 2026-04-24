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

        If an event loop is already running (gateway context), schedule the
        coroutine on it.  Otherwise, use ``asyncio.run()`` (CLI context).
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside an existing event loop — create a future and
            # schedule the coroutine.  This is safe because the middleware
            # hooks are fast (no blocking I/O).
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro_fn(state))
                return future.result(timeout=30)
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
                logger.exception(
                    "Middleware %s raised in %s — skipping", m.name, hook_name
                )
        return state

    def __repr__(self) -> str:
        names = [m.name for m in self._all]
        return f"MiddlewarePipeline({names})"
