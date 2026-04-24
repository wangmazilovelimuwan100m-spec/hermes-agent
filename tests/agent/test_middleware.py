"""Tests for the DeerFlow middleware pipeline framework (TASK-009).

Covers:
    - MiddlewarePipeline: ordering, priority, short-circuit on abort, error isolation
    - LoopDetectionMiddleware: consecutive signature detection, threshold, clear
    - ErrorRecoveryMiddleware: LLM retryable errors, tool timeout recovery
    - ToolAuditMiddleware: dangerous command logging, sensitive path detection
"""

import asyncio
import json
import logging
from unittest.mock import patch

import pytest

from agent.middleware import Middleware, MiddlewarePipeline, MiddlewareState
from agent.middlewares.loop_detection import (
    LoopDetectionMiddleware,
    _compute_call_signature,
)
from agent.middlewares.error_recovery import ErrorRecoveryMiddleware
from agent.middlewares.tool_audit import ToolAuditMiddleware, _redact_args


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_state(**overrides):
    """Create a MiddlewareState with sensible defaults."""
    defaults = {
        "messages": [],
        "api_call_count": 0,
        "session_id": "test-session",
    }
    defaults.update(overrides)
    return MiddlewareState(**defaults)


def _tool_call_dict(name: str, args: dict = None, call_id: str = "call_1"):
    """Create a dict-style tool call (OpenAI format)."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args or {}),
        },
    }


class _RecordingMiddleware(Middleware):
    """Middleware that appends its name to state.extra['order'] on each hook."""

    hooks = ["before_llm", "after_llm", "before_tool", "after_tool"]

    def __init__(self, tag: str):
        self.tag = tag

    async def before_llm(self, state):
        state.extra.setdefault("order", []).append(f"before_llm:{self.tag}")
        return state

    async def after_llm(self, state):
        state.extra.setdefault("order", []).append(f"after_llm:{self.tag}")
        return state

    async def before_tool(self, state):
        state.extra.setdefault("order", []).append(f"before_tool:{self.tag}")
        return state

    async def after_tool(self, state):
        state.extra.setdefault("order", []).append(f"after_tool:{self.tag}")
        return state


class _AbortingMiddleware(Middleware):
    """Middleware that sets state.aborted = True on after_tool."""

    hooks = ["after_tool"]

    async def after_tool(self, state):
        state.aborted = True
        return state


class _FailingMiddleware(Middleware):
    """Middleware that raises on before_llm — used to test error isolation."""

    hooks = ["before_llm"]

    async def before_llm(self, state):
        raise RuntimeError("boom")


# ── Test: MiddlewareState ──────────────────────────────────────────────────


class TestMiddlewareState:
    def test_defaults(self):
        s = MiddlewareState(messages=[])
        assert s.messages == []
        assert s.tool_calls == []
        assert s.tool_call is None
        assert s.tool_result is None
        assert s.api_call_count == 0
        assert s.aborted is False
        assert s.retry is False
        assert s.skip_tool is False
        assert isinstance(s.extra, dict)

    def test_custom_values(self):
        s = _make_state(api_call_count=5, model="gpt-4")
        assert s.api_call_count == 5
        assert s.model == "gpt-4"
        assert s.session_id == "test-session"


# ── Test: MiddlewarePipeline ordering ──────────────────────────────────────


class TestPipelineOrdering:
    @pytest.mark.asyncio
    async def test_hooks_execute_in_priority_order(self):
        """Middleware with lower priority runs first within each hook bucket."""
        mw_low = _RecordingMiddleware("low")
        mw_low.priority = 10
        mw_high = _RecordingMiddleware("high")
        mw_high.priority = 90

        pipeline = MiddlewarePipeline([mw_high, mw_low])  # passed out of order
        state = _make_state()
        state = await pipeline.run_before_llm(state)

        order = state.extra["order"]
        assert order == ["before_llm:low", "before_llm:high"]

    @pytest.mark.asyncio
    async def test_all_four_hooks_run(self):
        mw = _RecordingMiddleware("m")
        pipeline = MiddlewarePipeline([mw])
        state = _make_state()

        state = await pipeline.run_before_llm(state)
        state = await pipeline.run_after_llm(state)
        state = await pipeline.run_before_tool(state)
        state = await pipeline.run_after_tool(state)

        assert state.extra["order"] == [
            "before_llm:m",
            "after_llm:m",
            "before_tool:m",
            "after_tool:m",
        ]

    @pytest.mark.asyncio
    async def test_middleware_only_runs_on_declared_hooks(self):
        """A middleware declaring only 'before_tool' should NOT run in other hooks."""

        class SelectiveMiddleware(Middleware):
            hooks = ["before_tool"]

            async def before_tool(self, state):
                state.extra["ran"] = True
                return state

        pipeline = MiddlewarePipeline([SelectiveMiddleware()])
        state = _make_state()

        state = await pipeline.run_before_llm(state)
        assert "ran" not in state.extra

        state = await pipeline.run_before_tool(state)
        assert state.extra["ran"] is True

    @pytest.mark.asyncio
    async def test_error_isolation(self):
        """A failing middleware should not prevent others from running."""
        good = _RecordingMiddleware("good")
        bad = _FailingMiddleware()

        pipeline = MiddlewarePipeline([bad, good])
        state = _make_state()
        # Should not raise
        state = await pipeline.run_before_llm(state)
        # Good middleware still ran
        assert "before_llm:good" in state.extra.get("order", [])

    @pytest.mark.asyncio
    async def test_repr(self):
        mw = _RecordingMiddleware("x")
        pipeline = MiddlewarePipeline([mw])
        assert "_RecordingMiddleware" in repr(pipeline)


# ── Test: Sync wrappers ───────────────────────────────────────────────────


class TestSyncWrappers:
    def test_run_before_llm_sync(self):
        mw = _RecordingMiddleware("sync")
        pipeline = MiddlewarePipeline([mw])
        state = _make_state()
        result = pipeline.run_before_llm_sync(state)
        assert result.extra["order"] == ["before_llm:sync"]

    def test_run_after_tool_sync(self):
        mw = _RecordingMiddleware("sync")
        pipeline = MiddlewarePipeline([mw])
        state = _make_state()
        result = pipeline.run_after_tool_sync(state)
        assert result.extra["order"] == ["after_tool:sync"]


# ── Test: LoopDetectionMiddleware ──────────────────────────────────────────


class TestLoopDetection:
    def _make_loop_mw(self, threshold: int = 3, window: int = 10):
        return LoopDetectionMiddleware(window_size=window, threshold=threshold)

    @pytest.mark.asyncio
    async def test_no_detection_under_threshold(self):
        """Below threshold: state.aborted stays False."""
        mw = self._make_loop_mw(threshold=3)
        tc = _tool_call_dict("read_file", {"path": "/tmp/x.txt"})

        for _ in range(2):
            state = _make_state(tool_call=tc)
            state = await mw.after_tool(state)
            assert state.aborted is False

    @pytest.mark.asyncio
    async def test_detection_at_threshold(self):
        """At threshold: state.aborted becomes True and interruption message injected."""
        mw = self._make_loop_mw(threshold=3)
        tc = _tool_call_dict("read_file", {"path": "/tmp/x.txt"})

        for i in range(3):
            state = _make_state(tool_call=tc, messages=[])
            state = await mw.after_tool(state)

        assert state.aborted is True
        assert any("loop" in m.get("content", "").lower() for m in state.messages)

    @pytest.mark.asyncio
    async def test_different_args_no_loop(self):
        """Different arguments should not trigger detection."""
        mw = self._make_loop_mw(threshold=3)

        for i in range(5):
            tc = _tool_call_dict("read_file", {"path": f"/tmp/file_{i}.txt"})
            state = _make_state(tool_call=tc, messages=[])
            state = await mw.after_tool(state)
            assert state.aborted is False

    @pytest.mark.asyncio
    async def test_different_tools_no_loop(self):
        """Different tool names should not trigger detection."""
        mw = self._make_loop_mw(threshold=3)

        for name in ["read_file", "write_file", "terminal"]:
            tc = _tool_call_dict(name, {"path": "/tmp/x.txt"})
            state = _make_state(tool_call=tc, messages=[])
            state = await mw.after_tool(state)
            assert state.aborted is False

    @pytest.mark.asyncio
    async def test_per_session_isolation(self):
        """Different sessions should have independent loop tracking."""
        mw = self._make_loop_mw(threshold=3)
        tc = _tool_call_dict("read_file", {"path": "/tmp/x.txt"})

        # Session A: 2 calls (under threshold)
        for _ in range(2):
            state = _make_state(tool_call=tc, messages=[], session_id="sess-A")
            state = await mw.after_tool(state)
            assert state.aborted is False

        # Session B: 2 calls (under threshold, fresh counter)
        for _ in range(2):
            state = _make_state(tool_call=tc, messages=[], session_id="sess-B")
            state = await mw.after_tool(state)
            assert state.aborted is False

    @pytest.mark.asyncio
    async def test_clear_session(self):
        mw = self._make_loop_mw(threshold=3)
        tc = _tool_call_dict("read_file", {"path": "/tmp/x.txt"})

        # 2 calls
        for _ in range(2):
            state = _make_state(tool_call=tc, messages=[], session_id="sess-X")
            await mw.after_tool(state)

        # Clear session
        mw.clear_session("sess-X")

        # After clear, counter resets — 2 more calls should not trigger
        for _ in range(2):
            state = _make_state(tool_call=tc, messages=[], session_id="sess-X")
            state = await mw.after_tool(state)
            assert state.aborted is False

    @pytest.mark.asyncio
    async def test_no_tool_call_is_noop(self):
        mw = self._make_loop_mw()
        state = _make_state(tool_call=None)
        state = await mw.after_tool(state)
        assert state.aborted is False

    @pytest.mark.asyncio
    async def test_window_size_limits_memory(self):
        """History should be capped at window_size."""
        mw = self._make_loop_mw(threshold=100, window=3)
        tc = _tool_call_dict("read_file", {"path": "/tmp/x.txt"})

        for _ in range(5):
            state = _make_state(tool_call=tc, messages=[])
            await mw.after_tool(state)

        # Only 3 entries in deque
        history = mw._history["test-session"]
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_object_style_tool_call(self):
        """Should work with object-style tool calls (attribute access)."""

        class FakeFunction:
            name = "read_file"
            arguments = '{"path": "/tmp/x.txt"}'

        class FakeToolCall:
            id = "call_obj"
            function = FakeFunction()

        mw = self._make_loop_mw(threshold=3)

        for _ in range(3):
            state = _make_state(tool_call=FakeToolCall(), messages=[])
            state = await mw.after_tool(state)

        assert state.aborted is True


class TestComputeCallSignature:
    def test_deterministic(self):
        sig1 = _compute_call_signature("read_file", {"path": "/tmp/a.txt"})
        sig2 = _compute_call_signature("read_file", {"path": "/tmp/a.txt"})
        assert sig1 == sig2

    def test_different_args_different_sig(self):
        sig1 = _compute_call_signature("read_file", {"path": "/tmp/a.txt"})
        sig2 = _compute_call_signature("read_file", {"path": "/tmp/b.txt"})
        assert sig1 != sig2

    def test_different_name_different_sig(self):
        sig1 = _compute_call_signature("read_file", {"path": "/tmp/a.txt"})
        sig2 = _compute_call_signature("write_file", {"path": "/tmp/a.txt"})
        assert sig1 != sig2

    def test_string_args(self):
        sig = _compute_call_signature("tool", "raw string args")
        assert isinstance(sig, str) and len(sig) == 64


# ── Test: ErrorRecoveryMiddleware ──────────────────────────────────────────


class _MockAPIError(Exception):
    """Simulates an API error with status_code."""

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class TestErrorRecovery:
    def _make_mw(self, **kwargs):
        return ErrorRecoveryMiddleware(**kwargs)

    @pytest.mark.asyncio
    async def test_no_error_is_noop(self):
        mw = self._make_mw()
        state = _make_state()
        state = await mw.after_llm(state)
        assert state.retry is False

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_retry(self):
        mw = self._make_mw(max_llm_retries=3, llm_backoff_base=0.01, llm_backoff_max=0.05)
        error = _MockAPIError("rate limited", status_code=429)
        state = _make_state(api_error=error)

        with patch("asyncio.sleep"):  # skip actual sleep
            state = await mw.after_llm(state)

        assert state.retry is True

    @pytest.mark.asyncio
    async def test_server_error_triggers_retry(self):
        mw = self._make_mw(max_llm_retries=3, llm_backoff_base=0.01, llm_backoff_max=0.05)
        error = _MockAPIError("internal server error", status_code=500)
        state = _make_state(api_error=error)

        with patch("asyncio.sleep"):
            state = await mw.after_llm(state)

        assert state.retry is True

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        mw = self._make_mw(max_llm_retries=2, llm_backoff_base=0.01, llm_backoff_max=0.05)
        error = _MockAPIError("rate limited", status_code=429)

        with patch("asyncio.sleep"):
            # First two retries
            for _ in range(2):
                state = _make_state(api_error=error)
                state = await mw.after_llm(state)
                assert state.retry is True

            # Third attempt — should NOT retry (exhausted)
            state = _make_state(api_error=error)
            state = await mw.after_llm(state)
            assert state.retry is False

    @pytest.mark.asyncio
    async def test_non_retryable_error_no_retry(self):
        mw = self._make_mw()
        error = _MockAPIError("bad request", status_code=400)
        state = _make_state(api_error=error)

        state = await mw.after_llm(state)
        assert state.retry is False

    @pytest.mark.asyncio
    async def test_connection_error_is_retryable(self):
        mw = self._make_mw(llm_backoff_base=0.01, llm_backoff_max=0.05)
        error = ConnectionError("connection reset by peer")
        state = _make_state(api_error=error)

        with patch("asyncio.sleep"):
            state = await mw.after_llm(state)

        assert state.retry is True

    @pytest.mark.asyncio
    async def test_tool_timeout_triggers_retry_flag(self):
        mw = self._make_mw(tool_timeout_retry=True)
        tc = _tool_call_dict("terminal", {"command": "sleep 999"}, call_id="tc_1")
        state = _make_state(
            tool_call=tc,
            tool_result="Error: command timed out after 30s",
        )

        state = await mw.after_tool(state)
        assert state.extra.get("_tool_retry") is True

    @pytest.mark.asyncio
    async def test_tool_timeout_no_retry_when_disabled(self):
        mw = self._make_mw(tool_timeout_retry=False)
        tc = _tool_call_dict("terminal", {"command": "sleep 999"}, call_id="tc_1")
        state = _make_state(
            tool_call=tc,
            tool_result="Error: command timed out after 30s",
        )

        state = await mw.after_tool(state)
        assert "_tool_retry" not in state.extra

    @pytest.mark.asyncio
    async def test_tool_success_clears_counter(self):
        mw = self._make_mw()
        tc = _tool_call_dict("read_file", {"path": "/tmp/x"}, call_id="tc_1")
        state = _make_state(tool_call=tc, tool_result="file contents here")

        state = await mw.after_tool(state)
        assert "_tool_retry" not in state.extra

    @pytest.mark.asyncio
    async def test_clear_session(self):
        mw = self._make_mw()
        mw._retry_counts["sess-Z"] = {"llm": 2}
        mw.clear_session("sess-Z")
        assert "sess-Z" not in mw._retry_counts


# ── Test: ToolAuditMiddleware ──────────────────────────────────────────────


class TestToolAudit:
    def _make_mw(self, **kwargs):
        return ToolAuditMiddleware(**kwargs)

    @pytest.mark.asyncio
    async def test_no_tool_call_is_noop(self):
        mw = self._make_mw()
        state = _make_state(tool_call=None)
        state = await mw.before_tool(state)
        assert state.aborted is False

    @pytest.mark.asyncio
    async def test_logs_normal_tool_call(self, caplog):
        mw = self._make_mw(log_all_calls=True)
        tc = _tool_call_dict("read_file", {"path": "/tmp/x.txt"}, call_id="call_audit_1")
        state = _make_state(tool_call=tc)

        with caplog.at_level(logging.INFO, logger="hermes.audit"):
            state = await mw.before_tool(state)

        assert any("read_file" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_dangerous_command_logged(self, caplog):
        mw = self._make_mw(check_dangerous_commands=True)
        tc = _tool_call_dict("terminal", {"command": "rm -rf /"})
        state = _make_state(tool_call=tc)

        with caplog.at_level(logging.WARNING, logger="hermes.audit"):
            # Mock detect_dangerous_command since tools.approval may not be fully importable
            with patch("tools.approval.detect_dangerous_command", return_value=(True, "rm", "Dangerous rm command")):
                state = await mw.before_tool(state)

        assert any("DANGEROUS_COMMAND" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_sensitive_path_logged(self, caplog):
        mw = self._make_mw(sensitive_path_check=True)
        tc = _tool_call_dict("write_file", {"path": "/etc/passwd", "content": "hax"})
        state = _make_state(tool_call=tc)

        with caplog.at_level(logging.WARNING, logger="hermes.audit"):
            state = await mw.before_tool(state)

        assert any("SENSITIVE_FILE_WRITE" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_env_file_flagged(self, caplog):
        mw = self._make_mw(sensitive_path_check=True)
        tc = _tool_call_dict("write_file", {"path": "~/.env", "content": "KEY=VAL"})
        state = _make_state(tool_call=tc)

        with caplog.at_level(logging.WARNING, logger="hermes.audit"):
            state = await mw.before_tool(state)

        assert any("SENSITIVE_FILE_WRITE" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_normal_path_not_flagged(self, caplog):
        mw = self._make_mw(sensitive_path_check=True)
        tc = _tool_call_dict("write_file", {"path": "/home/user/code/app.py", "content": "hi"})
        state = _make_state(tool_call=tc)

        with caplog.at_level(logging.WARNING, logger="hermes.audit"):
            state = await mw.before_tool(state)

        assert not any("SENSITIVE_FILE_WRITE" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_log_all_disabled(self, caplog):
        mw = self._make_mw(log_all_calls=False)
        tc = _tool_call_dict("read_file", {"path": "/tmp/x.txt"})
        state = _make_state(tool_call=tc)

        with caplog.at_level(logging.INFO, logger="hermes.audit"):
            state = await mw.before_tool(state)

        # No audit log entries (tool is not terminal/write_file so no warnings either)
        assert not any("tool_call" in r.message for r in caplog.records)


class TestRedactArgs:
    def test_redacts_sensitive_keys(self):
        args = {"api_key": "sk-12345", "name": "test", "password": "secret"}
        redacted = _redact_args(args)
        assert redacted["api_key"] == "***"
        assert redacted["password"] == "***"
        assert redacted["name"] == "test"

    def test_truncates_long_strings(self):
        args = {"content": "x" * 1000}
        redacted = _redact_args(args)
        assert "1000 chars" in redacted["content"]

    def test_handles_nested_dicts(self):
        args = {"config": {"api_token": "abc", "timeout": 30}}
        redacted = _redact_args(args)
        assert redacted["config"]["api_token"] == "***"
        assert redacted["config"]["timeout"] == 30

    def test_preserves_normal_values(self):
        args = {"path": "/tmp/x.txt", "count": 42, "flag": True}
        redacted = _redact_args(args)
        assert redacted == args


# ── Test: Full pipeline integration ────────────────────────────────────────


class TestFullPipelineIntegration:
    @pytest.mark.asyncio
    async def test_three_middlewares_in_pipeline(self):
        """All three real middlewares in a pipeline — smoke test."""
        loop_mw = LoopDetectionMiddleware(threshold=3)
        error_mw = ErrorRecoveryMiddleware(max_llm_retries=2, llm_backoff_base=0.01)
        audit_mw = ToolAuditMiddleware()

        pipeline = MiddlewarePipeline([loop_mw, error_mw, audit_mw])

        # Run through all hooks with no errors — should be fine
        state = _make_state()
        state = await pipeline.run_before_llm(state)
        state = await pipeline.run_after_llm(state)

        tc = _tool_call_dict("read_file", {"path": "/tmp/x.txt"})
        state.tool_call = tc
        state.tool_result = "file contents"

        state = await pipeline.run_before_tool(state)
        state = await pipeline.run_after_tool(state)

        assert state.aborted is False
        assert state.retry is False

    @pytest.mark.asyncio
    async def test_loop_detected_mid_pipeline(self):
        """Loop detection fires after 3 identical calls in a pipeline."""
        loop_mw = LoopDetectionMiddleware(threshold=3)
        audit_mw = ToolAuditMiddleware()
        pipeline = MiddlewarePipeline([loop_mw, audit_mw])

        tc = _tool_call_dict("terminal", {"command": "ls"})

        for _ in range(3):
            state = _make_state(tool_call=tc, messages=[])
            state = await pipeline.run_after_tool(state)

        assert state.aborted is True
