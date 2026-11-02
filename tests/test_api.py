import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_subsets_lists_fd001_fd002_fd004():
    res = client.get("/api/model/subsets")
    assert res.status_code == 200
    subsets = {s["subset"] for s in res.json()}
    assert {"FD001", "FD002", "FD004"} <= subsets


@pytest.mark.parametrize("subset", ["FD001", "FD002", "FD004"])
def test_list_engines_for_each_subset(subset):
    res = client.get("/api/engines", params={"subset": subset})
    assert res.status_code == 200
    engines = res.json()
    assert len(engines) > 0
    assert "unit" in engines[0]
    assert "deep_analysis" in engines[0]


def test_get_single_engine_includes_shap():
    engines = client.get("/api/engines", params={"subset": "FD001"}).json()
    unit = engines[0]["unit"]
    res = client.get(f"/api/engines/{unit}", params={"subset": "FD001"})
    assert res.status_code == 200
    assert res.json()["unit"] == unit


def test_get_unknown_engine_is_404():
    res = client.get("/api/engines/999999", params={"subset": "FD001"})
    assert res.status_code == 404


def test_unknown_subset_is_400():
    res = client.get("/api/engines", params={"subset": "FD003"})
    assert res.status_code == 400


def test_analyze_engine_without_api_keys_is_503(monkeypatch):
    monkeypatch.setattr("app.routers.engines.OPENAI_API_KEY", None)
    monkeypatch.setattr("app.routers.engines.COHERE_API_KEY", None)
    res = client.post("/api/engines/1/analyze", params={"subset": "FD001"})
    assert res.status_code == 503
