"""Local execution environment — spawn-per-call with snapshot."""

import logging
import os
import platform
import shutil
import signal
import subprocess
import threading

_IS_WINDOWS = platform.system() == "Windows"

from tools.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)

# Hermes-internal env vars that should NOT leak into terminal subprocesses.
# These are loaded from ~/.hermes/.env for Hermes' own LLM/provider calls
# but can break external CLIs (e.g. codex) that also honor them.
# See: https://github.com/NousResearch/hermes-agent/issues/1002
#
# Built dynamically from the provider registry so new providers are
# automatically covered without manual blocklist maintenance.
_HERMES_PROVIDER_ENV_FORCE_PREFIX = "_HERMES_FORCE_"


def _build_provider_env_blocklist() -> frozenset:
    """Derive the blocklist from provider, tool, and gateway config.

    Automatically picks up api_key_env_vars and base_url_env_var from
    every registered provider, plus tool/messaging env vars from the
    optional config registry, so new Hermes-managed secrets are blocked
    in subprocesses without having to maintain multiple static lists.
    """
    blocked: set[str] = set()

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        for pconfig in PROVIDER_REGISTRY.values():
            blocked.update(pconfig.api_key_env_vars)
            if pconfig.base_url_env_var:
                blocked.add(pconfig.base_url_env_var)
    except ImportError:
        pass

    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS
        for name, metadata in OPTIONAL_ENV_VARS.items():
            category = metadata.get("category")
            if category in {"tool", "messaging"}:
                blocked.add(name)
            elif category == "setting" and metadata.get("password"):
                blocked.add(name)
    except ImportError:
        pass

    # Vars not covered above but still Hermes-internal / conflict-prone.
    blocked.update({
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",         # legacy alias
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_TOKEN",         # OAuth token (not in registry as env var)
        "CLAUDE_CODE_OAUTH_TOKEN",
        "LLM_MODEL",
        # Expanded isolation for other major providers (Issue #1002)
        "GOOGLE_API_KEY",          # Gemini / Google AI Studio
        "DEEPSEEK_API_KEY",        # DeepSeek
        "MISTRAL_API_KEY",         # Mistral AI
        "GROQ_API_KEY",            # Groq
        "TOGETHER_API_KEY",        # Together AI
        "PERPLEXITY_API_KEY",      # Perplexity
        "COHERE_API_KEY",          # Cohere
        "FIREWORKS_API_KEY",       # Fireworks AI
        "XAI_API_KEY",             # xAI (Grok)
        "HELICONE_API_KEY",        # LLM Observability proxy
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        # Gateway/runtime config not represented in OPTIONAL_ENV_VARS.
        "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_NAME",
        "DISCORD_HOME_CHANNEL",
        "DISCORD_HOME_CHANNEL_NAME",
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "SLACK_HOME_CHANNEL",
        "SLACK_HOME_CHANNEL_NAME",
        "SLACK_ALLOWED_USERS",
        "WHATSAPP_ENABLED",
        "WHATSAPP_MODE",
        "WHATSAPP_ALLOWED_USERS",
        "SIGNAL_HTTP_URL",
        "SIGNAL_ACCOUNT",
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "SIGNAL_HOME_CHANNEL",
        "SIGNAL_HOME_CHANNEL_NAME",
        "SIGNAL_IGNORE_STORIES",
        "HASS_TOKEN",
        "HASS_URL",
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
        "EMAIL_IMAP_HOST",
        "EMAIL_SMTP_HOST",
        "EMAIL_HOME_ADDRESS",
        "EMAIL_HOME_ADDRESS_NAME",
        "GATEWAY_ALLOWED_USERS",
        # Skills Hub / GitHub app auth paths and aliases.
        "GH_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        # Remote sandbox backend credentials.
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "DAYTONA_API_KEY",
    })
    return frozenset(blocked)


_HERMES_PROVIDER_ENV_BLOCKLIST = _build_provider_env_blocklist()


def _sanitize_subprocess_env(base_env: dict | None, extra_env: dict | None = None) -> dict:
    """Filter Hermes-managed secrets from a subprocess environment.

    `_HERMES_FORCE_<VAR>` entries in ``extra_env`` opt a blocked variable back in
    intentionally for callers that truly need it.  Vars registered via
    :mod:`tools.env_passthrough` (skill-declared or user-configured) also
    bypass the blocklist.
    """
    try:
        from tools.env_passthrough import is_env_passthrough as _is_passthrough
    except Exception:
        _is_passthrough = lambda _: False  # noqa: E731

    sanitized: dict[str, str] = {}

    for key, value in (base_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            continue
        if key not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(key):
            sanitized[key] = value

    for key, value in (extra_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = key[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            sanitized[real_key] = value
        elif key not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(key):
            sanitized[key] = value

    return sanitized


def _find_bash() -> str:
    """Find bash for command execution.

    The fence wrapper uses bash syntax (semicolons, $?, printf), so we
    must use bash — not the user's $SHELL which could be fish/zsh/etc.
    On Windows: uses Git Bash (bundled with Git for Windows).
    """
    if not _IS_WINDOWS:
        return (
            shutil.which("bash")
            or ("/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None)
            or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
            or os.environ.get("SHELL")  # last resort: whatever they have
            or "/bin/sh"
        )

    # Windows: look for Git Bash (installed with Git for Windows).
    # Allow override via env var (same pattern as Claude Code).
    custom = os.environ.get("HERMES_GIT_BASH_PATH")
    if custom and os.path.isfile(custom):
        return custom

    # shutil.which finds bash.exe if Git\bin is on PATH
    found = shutil.which("bash")
    if found:
        return found

    # Check common Git for Windows install locations
    for candidate in (
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "bin", "bash.exe"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate

    raise RuntimeError(
        "Git Bash not found. Hermes Agent requires Git for Windows on Windows.\n"
        "Install it from: https://git-scm.com/download/win\n"
        "Or set HERMES_GIT_BASH_PATH to your bash.exe location."
    )


# Backward compat — process_registry.py imports this name
_find_shell = _find_bash


# Standard PATH entries for environments with minimal PATH (e.g. systemd services).
# Includes macOS Homebrew paths (/opt/homebrew/* for Apple Silicon).
_SANE_PATH = (
    "/opt/homebrew/bin:/opt/homebrew/sbin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


def _make_run_env(env: dict) -> dict:
    """Build a run environment with a sane PATH and provider-var stripping."""
    try:
        from tools.env_passthrough import is_env_passthrough as _is_passthrough
    except Exception:
        _is_passthrough = lambda _: False  # noqa: E731

    merged = dict(os.environ | env)
    run_env = {}
    for k, v in merged.items():
        if k.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = k[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            run_env[real_key] = v
        elif k not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(k):
            run_env[k] = v
    existing_path = run_env.get("PATH", "")
    if "/usr/bin" not in existing_path.split(":"):
        run_env["PATH"] = f"{existing_path}:{_SANE_PATH}" if existing_path else _SANE_PATH
    return run_env


class LocalEnvironment(BaseEnvironment):
    """Run commands directly on the host machine.

    Uses the unified spawn-per-call model:
    - bash -l once at session start to capture env snapshot
    - bash -c for every subsequent command (fast, no shell init overhead)
    - CWD tracked via cwdfile written after each command
    - Process group kill (os.setsid) for clean child cleanup
    - stdin_data support for piping content (bypasses ARG_MAX limits)
    """

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None,
                 **kwargs):
        super().__init__(cwd=cwd or os.getcwd(), timeout=timeout, env=env)
        self.init_session()

    def _run_bash(self, cmd_string: str, *,
                  stdin_data: str | None = None) -> subprocess.Popen:
        run_env = _make_run_env(self.env)
        proc = subprocess.Popen(
            [_find_bash(), "-c", cmd_string],
            text=True,
            cwd="/",  # CWD set inside the script via cd
            env=run_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            preexec_fn=None if _IS_WINDOWS else os.setsid,
        )
        if stdin_data is not None:
            def _write():
                try:
                    proc.stdin.write(stdin_data)
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            threading.Thread(target=_write, daemon=True).start()
        return proc

    def _run_bash_login(self, cmd_string: str) -> subprocess.Popen:
        """For snapshot creation: uses bash -l -c."""
        run_env = _make_run_env(self.env)
        return subprocess.Popen(
            [_find_bash(), "-l", "-c", cmd_string],
            text=True,
            cwd="/",
            env=run_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=None if _IS_WINDOWS else os.setsid,
        )

    def _read_file_in_env(self, path: str) -> str:
        """Local override: direct file read, no subprocess needed."""
        try:
            with open(path) as f:
                return f.read()
        except OSError:
            return ""

    def _kill_process(self, proc):
        """Local override: kill process group for child cleanup."""
        try:
            if _IS_WINDOWS:
                proc.terminate()
            else:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=1.0)
                    return
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    def cleanup(self):
        for p in (self._snapshot_path, self._cwdfile_path):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass
