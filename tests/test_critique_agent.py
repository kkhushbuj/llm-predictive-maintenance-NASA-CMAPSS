import json

from app.agents.critique_agent import _build_payload


def test_build_payload_includes_shap_stability_when_given():
    payload = json.loads(
        _build_payload(
            {"unit": 1},
            {"verdict": "Novel"},
            None,
            shap_stability={"stable": True, "top_sensor_agreement_rate": 1.0},
        )
    )
    assert payload["shap_stability"]["stable"] is True


def test_build_payload_shap_stability_defaults_to_none():
    payload = json.loads(_build_payload({"unit": 1}, {"verdict": "Novel"}, None))
    assert payload["shap_stability"] is None
