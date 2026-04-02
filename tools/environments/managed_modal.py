"""Managed Modal environment backed by tool-gateway.

Uses ``BaseEnvironment`` for command shaping (``_wrap_command()``) but keeps
its own ``execute()`` override because the HTTP gateway cannot return a
ProcessHandle — all execution is request/response.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import requests
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted
from tools.managed_tool_gateway import resolve_managed_tool_gateway

logger = logging.getLogger(__name__)


def _request_timeout_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class _ManagedModalExecHandle:
    exec_id: str


class ManagedModalEnvironment(BaseEnvironment):
    """Gateway-owned Modal sandbox with Hermes-compatible execute/cleanup.

    Inherits from BaseEnvironment for _wrap_command() (CWD tracking,
    snapshot sourcing) but keeps its own execute() since the HTTP gateway
    cannot return a ProcessHandle.
    """

    _CONNECT_TIMEOUT_SECONDS = _request_timeout_env("TERMINAL_MANAGED_MODAL_CONNECT_TIMEOUT_SECONDS", 1.0)
    _POLL_READ_TIMEOUT_SECONDS = _request_timeout_env("TERMINAL_MANAGED_MODAL_POLL_READ_TIMEOUT_SECONDS", 5.0)
    _CANCEL_READ_TIMEOUT_SECONDS = _request_timeout_env("TERMINAL_MANAGED_MODAL_CANCEL_READ_TIMEOUT_SECONDS", 5.0)
    _client_timeout_grace_seconds = 10.0

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        modal_sandbox_kwargs: Optional[Dict[str, Any]] = None,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)

        self._guard_unsupported_credential_passthrough()

        gateway = resolve_managed_tool_gateway("modal")
        if gateway is None:
            raise ValueError("Managed Modal requires a configured tool gateway and Nous user token")

        self._gateway_origin = gateway.gateway_origin.rstrip("/")
        self._nous_user_token = gateway.nous_user_token
        self._task_id = task_id
        self._persistent = persistent_filesystem
        self._image = image
        self._sandbox_kwargs = dict(modal_sandbox_kwargs or {})
        self._create_idempotency_key = str(uuid.uuid4())
        self._sandbox_id = self._create_sandbox()

    # ------------------------------------------------------------------
    # _run_bash stub — ManagedModal cannot return a ProcessHandle
    # ------------------------------------------------------------------

    def _run_bash(self, cmd_string: str, *, stdin_data: str | None = None):
        raise NotImplementedError(
            "ManagedModalEnvironment is HTTP-based and cannot return a "
            "ProcessHandle. Use execute() directly."
        )

    # ------------------------------------------------------------------
    # execute() override — HTTP request/response model
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
    ) -> dict:
        effective_timeout = timeout or self.timeout

        # Handle stdin via heredoc embedding (gateway has payload support too)
        exec_stdin = stdin_data
        if stdin_data is not None:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            while marker in stdin_data:
                marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            command = f"{command} << '{marker}'\n{stdin_data}\n{marker}"
            exec_stdin = None  # embedded in command now

        exec_command, sudo_stdin = self._prepare_command(command)
        if sudo_stdin is not None:
            exec_command = (
                f"printf '%s\\n' {shlex.quote(sudo_stdin.rstrip())} | {exec_command}"
            )

        # Use _wrap_command for consistent CWD tracking and snapshot sourcing
        wrapped = self._wrap_command(exec_command, cwd)
        effective_cwd = cwd or self.cwd

        # Start the exec via the gateway
        start_result = self._start_exec(wrapped, effective_cwd, effective_timeout, exec_stdin)

        if start_result.get("_immediate"):
            result = {k: v for k, v in start_result.items() if k != "_immediate"}
            self._update_cwd_from_gateway_output(result)
            return result

        handle = start_result.get("_handle")
        if handle is None:
            return self._error_result(
                "Managed Modal exec start did not return an exec handle"
            )

        # Poll loop
        deadline = None
        if self._client_timeout_grace_seconds is not None:
            deadline = time.monotonic() + effective_timeout + self._client_timeout_grace_seconds

        while True:
            if is_interrupted():
                try:
                    self._cancel_exec(handle.exec_id)
                except Exception:
                    pass
                return self._result(
                    "[Command interrupted - Modal sandbox exec cancelled]", 130,
                )

            try:
                result = self._poll_exec(handle)
            except Exception as exc:
                return self._error_result(f"Managed Modal exec poll failed: {exc}")

            if result is not None:
                self._update_cwd_from_gateway_output(result)
                return result

            if deadline is not None and time.monotonic() >= deadline:
                try:
                    self._cancel_exec(handle.exec_id)
                except Exception:
                    pass
                return self._result(
                    f"Managed Modal exec timed out after {effective_timeout}s", 124,
                )

            time.sleep(0.25)

    def _update_cwd_from_gateway_output(self, result: dict) -> None:
        """Best-effort CWD update from the cwdfile written by _wrap_command.

        Since we can't read files from the gateway sandbox directly, we
        only update if there's a subsequent call that reads it.  The
        _wrap_command template writes CWD to the cwdfile which will be
        read on the next execute() call if we had file access.  For now,
        CWD tracking in managed modal relies on the cwd parameter.
        """
        pass

    # ------------------------------------------------------------------
    # Gateway transport
    # ------------------------------------------------------------------

    def _start_exec(self, command: str, cwd: str, timeout: int,
                    stdin_data: str | None) -> dict:
        exec_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "execId": exec_id,
            "command": command,
            "cwd": cwd,
            "timeoutMs": int(timeout * 1000),
        }
        if stdin_data is not None:
            payload["stdinData"] = stdin_data

        try:
            response = self._request(
                "POST",
                f"/v1/sandboxes/{self._sandbox_id}/execs",
                json=payload,
                timeout=10,
            )
        except Exception as exc:
            return {
                **self._error_result(f"Managed Modal exec failed: {exc}"),
                "_immediate": True,
            }

        if response.status_code >= 400:
            return {
                **self._error_result(
                    self._format_error("Managed Modal exec failed", response)
                ),
                "_immediate": True,
            }

        body = response.json()
        status = body.get("status")
        if status in {"completed", "failed", "cancelled", "timeout"}:
            return {
                **self._result(body.get("output", ""), body.get("returncode", 1)),
                "_immediate": True,
            }

        if body.get("execId") != exec_id:
            return {
                **self._error_result(
                    "Managed Modal exec start did not return the expected exec id"
                ),
                "_immediate": True,
            }

        return {"_handle": _ManagedModalExecHandle(exec_id=exec_id)}

    def _poll_exec(self, handle: _ManagedModalExecHandle) -> dict | None:
        try:
            status_response = self._request(
                "GET",
                f"/v1/sandboxes/{self._sandbox_id}/execs/{handle.exec_id}",
                timeout=(self._CONNECT_TIMEOUT_SECONDS, self._POLL_READ_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            return self._error_result(f"Managed Modal exec poll failed: {exc}")

        if status_response.status_code == 404:
            return self._error_result("Managed Modal exec not found")

        if status_response.status_code >= 400:
            return self._error_result(
                self._format_error("Managed Modal exec poll failed", status_response)
            )

        status_body = status_response.json()
        status = status_body.get("status")
        if status in {"completed", "failed", "cancelled", "timeout"}:
            return self._result(
                status_body.get("output", ""),
                status_body.get("returncode", 1),
            )
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self):
        if not getattr(self, "_sandbox_id", None):
            return

        try:
            self._request(
                "POST",
                f"/v1/sandboxes/{self._sandbox_id}/terminate",
                json={
                    "snapshotBeforeTerminate": self._persistent,
                },
                timeout=60,
            )
        except Exception as exc:
            logger.warning("Managed Modal cleanup failed: %s", exc)
        finally:
            self._sandbox_id = None

    # ------------------------------------------------------------------
    # Sandbox creation
    # ------------------------------------------------------------------

    def _create_sandbox(self) -> str:
        cpu = self._coerce_number(self._sandbox_kwargs.get("cpu"), 1)
        memory = self._coerce_number(
            self._sandbox_kwargs.get("memoryMiB", self._sandbox_kwargs.get("memory")),
            5120,
        )
        disk = self._coerce_number(
            self._sandbox_kwargs.get("ephemeral_disk", self._sandbox_kwargs.get("diskMiB")),
            None,
        )

        create_payload = {
            "image": self._image,
            "cwd": self.cwd,
            "cpu": cpu,
            "memoryMiB": memory,
            "timeoutMs": 3_600_000,
            "idleTimeoutMs": max(300_000, int(self.timeout * 1000)),
            "persistentFilesystem": self._persistent,
            "logicalKey": self._task_id,
        }
        if disk is not None:
            create_payload["diskMiB"] = disk

        response = self._request(
            "POST",
            "/v1/sandboxes",
            json=create_payload,
            timeout=60,
            extra_headers={
                "x-idempotency-key": self._create_idempotency_key,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(self._format_error("Managed Modal create failed", response))

        body = response.json()
        sandbox_id = body.get("id")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise RuntimeError("Managed Modal create did not return a sandbox id")
        return sandbox_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _guard_unsupported_credential_passthrough(self) -> None:
        """Managed Modal does not sync or mount host credential files."""
        try:
            from tools.credential_files import get_credential_file_mounts
        except Exception:
            return

        mounts = get_credential_file_mounts()
        if mounts:
            raise ValueError(
                "Managed Modal does not support host credential-file passthrough. "
                "Use TERMINAL_MODAL_MODE=direct when skills or config require "
                "credential files inside the sandbox."
            )

    def _result(self, output: str, returncode: int) -> dict:
        return {"output": output, "returncode": returncode}

    def _error_result(self, output: str) -> dict:
        return self._result(output, 1)

    def _request(self, method: str, path: str, *,
                 json: Dict[str, Any] | None = None,
                 timeout: int = 30,
                 extra_headers: Dict[str, str] | None = None) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self._nous_user_token}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        return requests.request(
            method,
            f"{self._gateway_origin}{path}",
            headers=headers,
            json=json,
            timeout=timeout,
        )

    def _cancel_exec(self, exec_id: str) -> None:
        try:
            self._request(
                "POST",
                f"/v1/sandboxes/{self._sandbox_id}/execs/{exec_id}/cancel",
                timeout=(self._CONNECT_TIMEOUT_SECONDS, self._CANCEL_READ_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            logger.warning("Managed Modal exec cancel failed: %s", exc)

    @staticmethod
    def _coerce_number(value: Any, default: float) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _format_error(prefix: str, response: requests.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                message = payload.get("error") or payload.get("message") or payload.get("code")
                if isinstance(message, str) and message:
                    return f"{prefix}: {message}"
                return f"{prefix}: {json.dumps(payload, ensure_ascii=False)}"
        except Exception:
            pass

        text = response.text.strip()
        if text:
            return f"{prefix}: {text}"
        return f"{prefix}: HTTP {response.status_code}"
