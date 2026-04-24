"""Tool audit middleware.

Extends the existing ``tools.approval`` dangerous-command detection system
with a middleware hook that runs *before* every tool execution.  The
middleware:

1. Checks terminal commands against ``DANGEROUS_PATTERNS`` (delegating to
   ``tools.approval.detect_dangerous_command``).
2. Checks write targets against sensitive path patterns.
3. Logs all tool invocations (with arguments redacted for sensitive values)
   to a dedicated audit logger.

The middleware does **not** duplicate the approval-prompting logic — that
remains in ``tools.approval``.  This layer is purely observational and
logging, so it can run safely in non-interactive contexts (gateway, cron).

Inspired by DeerFlow's SandboxAuditMiddleware.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from agent.middleware import Middleware, MiddlewareState

logger = logging.getLogger(__name__)

# Separate audit logger — can be routed to a different handler/file.
audit_logger = logging.getLogger("hermes.audit")


# ── Sensitive argument keys whose values should be redacted in logs ──
_SENSITIVE_ARG_KEYS = frozenset({
    "api_key", "token", "password", "secret", "authorization",
    "cookie", "credit_card", "ssn", "private_key",
})

# Sensitive path prefixes that should be flagged in write operations.
_SENSITIVE_PATH_PREFIXES = (
    "/etc/",
    "/var/",
    "/System/",
    "/Library/",
    "/usr/",
)

# Patterns for commands that should always be audited at WARNING level.
_HIGH_RISK_PATTERNS = re.compile(
    r"""(?:rm\s|sudo\s|chmod\s|chown\s|curl\b|wget\b|nc\b|ncat\b|ssh\b|scp\b)""",
    re.IGNORECASE,
)


def _redact_args(args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *args* with sensitive values replaced by ``***``."""
    redacted: dict[str, Any] = {}
    for k, v in args.items():
        kl = k.lower()
        if any(s in kl for s in _SENSITIVE_ARG_KEYS):
            redacted[k] = "***"
        elif isinstance(v, str) and len(v) > 500:
            redacted[k] = v[:250] + "…[" + str(len(v)) + " chars]…" + v[-250:]
        elif isinstance(v, (dict, list)):
            # Recursively redact nested structures (one level deep).
            if isinstance(v, dict):
                redacted[k] = {sk: "***" if any(s in sk.lower() for s in _SENSITIVE_ARG_KEYS) else sv for sk, sv in v.items()}
            else:
                redacted[k] = v
        else:
            redacted[k] = v
    return redacted


class ToolAuditMiddleware(Middleware):
    """Audit tool calls for safety and logging.

    Runs in the ``before_tool`` hook to inspect every tool invocation
    before execution.  Checks for:
    - Dangerous terminal commands (via ``tools.approval.detect_dangerous_command``).
    - Writes to sensitive filesystem paths.
    - High-risk command patterns.

    All tool calls are logged to the ``hermes.audit`` logger.

    Configuration:
        log_all_calls (bool): Log every tool call at INFO level (default True).
        check_dangerous_commands (bool): Run dangerous-command detection for
            ``terminal`` tool calls (default True).
        sensitive_path_check (bool): Flag writes to sensitive paths
            (default True).
    """

    hooks = ["before_tool"]
    name = "ToolAuditMiddleware"
    priority = 5  # Run very early — safety first
    fail_fast = True  # Security audit failure must not be silently swallowed

    def __init__(
        self,
        log_all_calls: bool = True,
        check_dangerous_commands: bool = True,
        sensitive_path_check: bool = True,
    ) -> None:
        self._log_all_calls = log_all_calls
        self._check_dangerous_commands = check_dangerous_commands
        self._sensitive_path_check = sensitive_path_check

    # ── Middleware hook ──────────────────────────────────────────────

    async def before_tool(self, state: MiddlewareState) -> MiddlewareState:
        tool_call = state.tool_call
        if tool_call is None:
            return state

        tool_name, tool_args = self._parse_tool_call(tool_call)
        if not tool_name:
            return state

        redacted = _redact_args(tool_args)

        # ── 1. Dangerous command detection (terminal tool) ───────────
        if (
            self._check_dangerous_commands
            and tool_name == "terminal"
        ):
            cmd = tool_args.get("command", "")
            if cmd:
                self._audit_terminal_command(cmd, state)

        # ── 2. Sensitive path check (file write tools) ───────────────
        if self._sensitive_path_check and tool_name in ("write_file", "patch"):
            path = tool_args.get("path", "")
            if path:
                self._audit_file_write(path, tool_name, state)

        # ── 3. General audit log ─────────────────────────────────────
        if self._log_all_calls:
            audit_logger.info(
                "tool_call session=%s tool=%s call_id=%s args=%s",
                state.session_id or "default",
                tool_name,
                self._call_id(tool_call),
                json.dumps(redacted, ensure_ascii=False, default=str),
            )

        return state

    # ── Audit helpers ───────────────────────────────────────────────

    def _audit_terminal_command(
        self, cmd: str, state: MiddlewareState
    ) -> None:
        """Check a terminal command and log findings."""
        try:
            from tools.approval import detect_dangerous_command
            is_dangerous, pattern_key, description = detect_dangerous_command(cmd)
        except Exception:
            is_dangerous = False
            pattern_key = None
            description = None

        if is_dangerous:
            audit_logger.warning(
                "DANGEROUS_COMMAND session=%s tool=terminal pattern=%s desc=%s cmd=%r",
                state.session_id or "default",
                pattern_key,
                description,
                cmd[:300],
            )
        elif _HIGH_RISK_PATTERNS.search(cmd):
            audit_logger.info(
                "HIGH_RISK_COMMAND session=%s tool=terminal cmd=%r",
                state.session_id or "default",
                cmd[:300],
            )

    def _audit_file_write(
        self, path: str, tool_name: str, state: MiddlewareState
    ) -> None:
        """Check if a file write targets a sensitive path."""
        # Normalise
        norm = path
        if norm.startswith("~"):
            import os
            norm = os.path.expanduser(norm)
        norm = norm.replace("\\", "/")

        for prefix in _SENSITIVE_PATH_PREFIXES:
            if norm.startswith(prefix):
                audit_logger.warning(
                    "SENSITIVE_FILE_WRITE session=%s tool=%s path=%s prefix=%s",
                    state.session_id or "default",
                    tool_name,
                    path,
                    prefix,
                )
                return

        # Check for .env / .ssh patterns
        lower = norm.lower()
        if any(s in lower for s in (".env", ".ssh/", "id_rsa", "id_ed25519")):
            audit_logger.warning(
                "SENSITIVE_FILE_WRITE session=%s tool=%s path=%s reason=config_or_key",
                state.session_id or "default",
                tool_name,
                path,
            )

    # ── Parsing utilities ───────────────────────────────────────────

    @staticmethod
    def _parse_tool_call(tool_call: Any) -> tuple[str, dict[str, Any]]:
        """Extract (tool_name, tool_args_dict) from a tool call."""
        if isinstance(tool_call, dict):
            name = tool_call.get("function", {}).get("name", "")
            raw = tool_call.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            return name, args
        fn = getattr(tool_call, "function", None)
        if fn:
            name = getattr(fn, "name", "")
            raw = getattr(fn, "arguments", "{}")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            return name, args
        return "", {}

    @staticmethod
    def _call_id(tool_call: Any) -> str:
        if isinstance(tool_call, dict):
            return tool_call.get("id", "")
        return getattr(tool_call, "id", "")
