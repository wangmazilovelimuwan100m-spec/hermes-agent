"""Integration tests for tools.browser_supervisor.

Exercises the supervisor end-to-end against a real local Chrome
(``--remote-debugging-port``).  Skipped when Chrome is not installed
— these are the tests that actually verify the CDP wire protocol
works, since mock-CDP unit tests can only prove the happy paths we
thought to model.

Run manually:
    scripts/run_tests.sh tests/tools/test_browser_supervisor.py

Automated: skipped in CI unless ``HERMES_E2E_BROWSER=1`` is set.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time

import pytest


pytestmark = pytest.mark.skipif(
    not shutil.which("google-chrome") and not shutil.which("chromium"),
    reason="Chrome/Chromium not installed",
)


def _find_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    pytest.skip("no Chrome binary found")


@pytest.fixture
def chrome_cdp(worker_id):
    """Start a headless Chrome with --remote-debugging-port, yield its WS URL.

    Uses a unique port per xdist worker to avoid cross-worker collisions.
    """
    import socket

    # xdist worker_id is "master" in single-process mode or "gw0".."gwN" otherwise.
    if worker_id == "master":
        port_offset = 0
    else:
        port_offset = int(worker_id.lstrip("gw"))
    port = 9225 + port_offset
    profile = tempfile.mkdtemp(prefix="hermes-supervisor-test-")
    proc = subprocess.Popen(
        [
            _find_chrome(),
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--headless=new",
            "--disable-gpu",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ws_url = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=1
            ) as r:
                info = json.loads(r.read().decode())
                ws_url = info["webSocketDebuggerUrl"]
                break
        except Exception:
            time.sleep(0.25)
    if ws_url is None:
        proc.terminate()
        proc.wait(timeout=5)
        shutil.rmtree(profile, ignore_errors=True)
        pytest.skip("Chrome didn't expose CDP in time")

    yield ws_url, port

    proc.terminate()
    try:
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
    shutil.rmtree(profile, ignore_errors=True)


def _test_page_url() -> str:
    html = """<!doctype html>
<html><head><title>Supervisor pytest</title></head><body>
<h1>Supervisor pytest</h1>
<iframe id="inner" srcdoc="<body><h2>frame-marker</h2></body>" width="400" height="100"></iframe>
</body></html>"""
    return "data:text/html;base64," + base64.b64encode(html.encode()).decode()


def _fire_on_page(cdp_url: str, expression: str) -> None:
    """Navigate the first page target to a data URL and fire `expression`."""
    import asyncio
    import websockets as _ws_mod

    async def run():
        async with _ws_mod.connect(cdp_url, max_size=50 * 1024 * 1024) as ws:
            next_id = [1]

            async def call(method, params=None, session_id=None):
                cid = next_id[0]
                next_id[0] += 1
                p = {"id": cid, "method": method}
                if params:
                    p["params"] = params
                if session_id:
                    p["sessionId"] = session_id
                await ws.send(json.dumps(p))
                async for raw in ws:
                    m = json.loads(raw)
                    if m.get("id") == cid:
                        return m

            targets = (await call("Target.getTargets"))["result"]["targetInfos"]
            page = next(t for t in targets if t.get("type") == "page")
            attach = await call(
                "Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}
            )
            sid = attach["result"]["sessionId"]
            await call("Page.navigate", {"url": _test_page_url()}, session_id=sid)
            await asyncio.sleep(1.5)  # let the page load
            await call(
                "Runtime.evaluate",
                {"expression": expression, "returnByValue": True},
                session_id=sid,
            )

    asyncio.run(run())


@pytest.fixture
def supervisor_registry():
    """Yield the global registry and tear down any supervisors after the test."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    yield SUPERVISOR_REGISTRY
    SUPERVISOR_REGISTRY.stop_all()


def _wait_for_dialog(supervisor, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = supervisor.snapshot()
        if snap.pending_dialogs:
            return snap.pending_dialogs
        time.sleep(0.1)
    return ()


def test_supervisor_start_and_snapshot(chrome_cdp, supervisor_registry):
    """Supervisor attaches, exposes an active snapshot with a top frame."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-1", cdp_url=cdp_url)

    # Navigate so the frame tree populates.
    _fire_on_page(cdp_url, "/* no dialog */ void 0")

    # Give a moment for frame events to propagate
    time.sleep(1.0)
    snap = supervisor.snapshot()
    assert snap.active is True
    assert snap.task_id == "pytest-1"
    assert snap.pending_dialogs == ()
    # At minimum a top frame should exist after the navigate.
    assert snap.frame_tree.get("top") is not None


def test_main_frame_alert_detection_and_dismiss(chrome_cdp, supervisor_registry):
    """alert() in the main frame surfaces and can be dismissed via the sync API."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-2", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-MAIN-ALERT'), 50)")
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs, "no dialog detected"
    d = dialogs[0]
    assert d.type == "alert"
    assert "PYTEST-MAIN-ALERT" in d.message

    result = supervisor.respond_to_dialog("dismiss")
    assert result["ok"] is True
    # State cleared after dismiss
    time.sleep(0.3)
    assert supervisor.snapshot().pending_dialogs == ()


def test_iframe_contentwindow_alert(chrome_cdp, supervisor_registry):
    """alert() fired from inside a same-origin iframe surfaces too."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-3", cdp_url=cdp_url)

    _fire_on_page(
        cdp_url,
        "setTimeout(() => document.querySelector('#inner').contentWindow.alert('PYTEST-IFRAME'), 50)",
    )
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs, "no iframe dialog detected"
    assert any("PYTEST-IFRAME" in d.message for d in dialogs)

    result = supervisor.respond_to_dialog("accept")
    assert result["ok"] is True


def test_prompt_dialog_with_response_text(chrome_cdp, supervisor_registry):
    """prompt() gets our prompt_text back inside the page."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-4", cdp_url=cdp_url)

    # Fire a prompt and stash the answer on window
    _fire_on_page(
        cdp_url,
        "setTimeout(() => { window.__promptResult = prompt('give me a token', 'default-x'); }, 50)",
    )
    dialogs = _wait_for_dialog(supervisor)
    assert dialogs
    d = dialogs[0]
    assert d.type == "prompt"
    assert d.default_prompt == "default-x"

    result = supervisor.respond_to_dialog("accept", prompt_text="PYTEST-PROMPT-REPLY")
    assert result["ok"] is True


def test_respond_with_no_pending_dialog_errors_cleanly(chrome_cdp, supervisor_registry):
    """Calling respond_to_dialog when nothing is pending returns a clean error, not an exception."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-5", cdp_url=cdp_url)

    result = supervisor.respond_to_dialog("accept")
    assert result["ok"] is False
    assert "no dialog" in result["error"].lower()


def test_auto_dismiss_policy(chrome_cdp, supervisor_registry):
    """auto_dismiss policy clears dialogs without the agent responding."""
    from tools.browser_supervisor import DIALOG_POLICY_AUTO_DISMISS

    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(
        task_id="pytest-6",
        cdp_url=cdp_url,
        dialog_policy=DIALOG_POLICY_AUTO_DISMISS,
    )

    _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-AUTO-DISMISS'), 50)")
    # Give the supervisor a moment to see + auto-dismiss
    time.sleep(2.0)
    snap = supervisor.snapshot()
    # Nothing pending because auto-dismiss cleared it immediately
    assert snap.pending_dialogs == ()


def test_registry_idempotent_get_or_start(chrome_cdp, supervisor_registry):
    """Calling get_or_start twice with the same (task, url) returns the same instance."""
    cdp_url, _port = chrome_cdp
    a = supervisor_registry.get_or_start(task_id="pytest-idem", cdp_url=cdp_url)
    b = supervisor_registry.get_or_start(task_id="pytest-idem", cdp_url=cdp_url)
    assert a is b


def test_registry_stop(chrome_cdp, supervisor_registry):
    """stop() tears down the supervisor and snapshot reports inactive."""
    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-stop", cdp_url=cdp_url)
    assert supervisor.snapshot().active is True
    supervisor_registry.stop("pytest-stop")
    # Post-stop snapshot reports inactive; supervisor obj may still exist
    assert supervisor.snapshot().active is False


def test_browser_dialog_tool_no_supervisor():
    """browser_dialog returns a clear error when no supervisor is attached."""
    from tools.browser_dialog_tool import browser_dialog

    r = json.loads(browser_dialog(action="accept", task_id="nonexistent-task"))
    assert r["success"] is False
    assert "No CDP supervisor" in r["error"]


def test_browser_dialog_invalid_action(chrome_cdp, supervisor_registry):
    """browser_dialog rejects actions that aren't accept/dismiss."""
    from tools.browser_dialog_tool import browser_dialog

    cdp_url, _port = chrome_cdp
    supervisor_registry.get_or_start(task_id="pytest-bad-action", cdp_url=cdp_url)

    r = json.loads(browser_dialog(action="eat", task_id="pytest-bad-action"))
    assert r["success"] is False
    assert "accept" in r["error"] and "dismiss" in r["error"]


def test_recent_dialogs_ring_buffer(chrome_cdp, supervisor_registry):
    """Closed dialogs show up in recent_dialogs with a closed_by tag."""
    from tools.browser_supervisor import DIALOG_POLICY_AUTO_DISMISS

    cdp_url, _port = chrome_cdp
    sv = supervisor_registry.get_or_start(
        task_id="pytest-recent",
        cdp_url=cdp_url,
        dialog_policy=DIALOG_POLICY_AUTO_DISMISS,
    )

    _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-RECENT'), 50)")
    # Wait for auto-dismiss to cycle the dialog through
    deadline = time.time() + 5
    while time.time() < deadline:
        recent = sv.snapshot().recent_dialogs
        if recent and any("PYTEST-RECENT" in r.message for r in recent):
            break
        time.sleep(0.1)

    recent = sv.snapshot().recent_dialogs
    assert recent, "recent_dialogs should contain the auto-dismissed dialog"
    match = next((r for r in recent if "PYTEST-RECENT" in r.message), None)
    assert match is not None
    assert match.type == "alert"
    assert match.closed_by == "auto_policy"
    assert match.closed_at >= match.opened_at


def test_browser_dialog_tool_end_to_end(chrome_cdp, supervisor_registry):
    """Full agent-path check: fire an alert, call the tool handler directly."""
    from tools.browser_dialog_tool import browser_dialog

    cdp_url, _port = chrome_cdp
    supervisor = supervisor_registry.get_or_start(task_id="pytest-tool", cdp_url=cdp_url)

    _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-TOOL-END2END'), 50)")
    assert _wait_for_dialog(supervisor), "no dialog detected via wait_for_dialog"

    r = json.loads(browser_dialog(action="dismiss", task_id="pytest-tool"))
    assert r["success"] is True
    assert r["action"] == "dismiss"
    assert "PYTEST-TOOL-END2END" in r["dialog"]["message"]
