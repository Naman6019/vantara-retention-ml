"""Smoke test: the API-only dashboard renders every view without exceptions.

Runs an in-process uvicorn server, points the dashboard at it, and drives each
view with Streamlit's AppTest. Requires built artifacts + a populated DB.
"""

import os
import socket
import threading
import time

import pytest
import uvicorn


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def api_server():
    from api.main import app

    port = _free_port()
    os.environ["API_URL"] = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.2)
    yield port
    server.should_exit = True
    thread.join(timeout=10)


def test_dashboard_views_render(api_server):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    dashboard = Path(__file__).resolve().parent.parent / "frontend" / "dashboard.py"
    at = AppTest.from_file(str(dashboard), default_timeout=60)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    for view in [
        "📊 Executive CRM Cockpit",
        "🎯 At-Risk Priority (Value at Risk)",
        "👥 Customer Segmentation & Personas",
        "🔍 Customer 360 & XAI",
        "📈 Revenue Trends & Forecast",
        "🩺 Model Health & Drift",
        "🎯 Campaign Planner (Budget)",
        "📁 Batch Scoring & Export",
    ]:
        at.sidebar.radio[0].set_value(view).run()
        assert not at.exception, f"{view}: {[str(e) for e in at.exception]}"
