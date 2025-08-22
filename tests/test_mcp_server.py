import os
import sys

# Ensure src/ is on sys.path so tests can import the package when running from repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fastapi.testclient import TestClient
from readable_fetch.mcp_server.server import app

client = TestClient(app)


def test_tools_endpoint():
    resp = client.get("/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    names = [t["name"] for t in data["tools"]]
    assert "fetch_and_markdownify" in names
    assert "fetch_and_markdownify_gpt" in names


def test_invoke_bad_tool():
    resp = client.post("/invoke", json={"tool": "nonexistent", "args": {}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False


def test_invoke_fetch_missing_url():
    # Should return validation error from Pydantic inside the endpoint
    resp = client.post("/invoke", json={"tool": "fetch_and_markdownify", "args": {}})
    # FastAPI/Pydantic returns 422 for model validation errors
    assert resp.status_code == 422
