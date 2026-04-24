"""Loop detection middleware.

Detects when the agent is stuck in a repetitive tool-calling loop by
tracking the signatures of recent tool calls.  When the same call signature
appears consecutively more than a configurable threshold, the middleware
injects an interruption message into the conversation to break the cycle.

Inspired by DeerFlow's LoopDetectionMiddleware.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from typing import Any, Optional

from agent.middleware import Middleware, MiddlewareState

logger = logging.getLogger(__name__)


def _compute_call_signature(tool_name: str, tool_args: Any) -> str:
    """Return a stable hash for a tool call's identity.

    Normalises ``tool_args`` to a deterministic JSON string (sorted keys)
    and hashes it together with the tool name.
    """
    try:
        if isinstance(tool_args, str):
            args_str = tool_args
        else:
            args_str = json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        args_str = str(tool_args)
    payload = f"{tool_name}:{args_str}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LoopDetectionMiddleware(Middleware):
    """Detect and break repetitive tool-call loops.

    Configuration:
        window_size (int): How many recent signatures to keep (default 10).
        threshold (int): Number of *consecutive* identical calls before
            triggering (default 3).

    When a loop is detected, a system-level interruption message is appended
    to ``state.messages`` and ``state.aborted`` is set to ``True`` so the
    caller can skip the current tool execution and let the LLM see the
    interruption on the next turn.
    """

    hooks = ["after_tool"]
    name = "LoopDetectionMiddleware"
    priority = 10  # Run early in after_tool chain

    def __init__(
        self,
        window_size: int = 10,
        threshold: int = 3,
    ) -> None:
        self._window_size = window_size
        self._threshold = threshold
        # Per-session state: {session_id: deque_of_signatures}
        self._history: dict[str, deque[str]] = {}

    # ── Middleware hook ──────────────────────────────────────────────

    async def after_tool(self, state: MiddlewareState) -> MiddlewareState:
        tool_call = state.tool_call
        if tool_call is None:
            return state

        tool_name: str = ""
        tool_args: Any = {}

        # Support both dict-style (normalised in run_agent) and object-style.
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("function", {}).get("name", "")
            raw_args = tool_call.get("function", {}).get("arguments", "{}")
            try:
                tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, TypeError):
                tool_args = {}
        else:
            fn = getattr(tool_call, "function", None)
            if fn:
                tool_name = getattr(fn, "name", "")
                raw_args = getattr(fn, "arguments", "{}")
                try:
                    tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    tool_args = {}

        if not tool_name:
            return state

        sig = _compute_call_signature(tool_name, tool_args)
        sid = state.session_id or "default"
        history = self._history.setdefault(sid, deque(maxlen=self._window_size))
        history.append(sig)

        # Count consecutive occurrences at the tail of the deque.
        consecutive = 0
        for s in reversed(history):
            if s == sig:
                consecutive += 1
            else:
                break

        if consecutive >= self._threshold:
            logger.warning(
                "Loop detected: %s called %d consecutive times (session=%s)",
                tool_name, consecutive, sid,
            )
            interrupt_msg = (
                "[System: You have been calling the same tool with the same arguments "
                f"repeatedly ({tool_name}, {consecutive} times). This indicates a loop. "
                "Please try a different approach, change the arguments, or stop if the "
                "task is already complete.]"
            )
            state.messages.append({"role": "user", "content": interrupt_msg})
            state.aborted = True
            # Clear history so the same loop isn't re-detected immediately.
            history.clear()

        return state

    # ── Housekeeping ─────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> None:
        """Remove history for a session (call on session reset/end)."""
        self._history.pop(session_id, None)

    def clear_all(self) -> None:
        """Remove all session histories."""
        self._history.clear()
