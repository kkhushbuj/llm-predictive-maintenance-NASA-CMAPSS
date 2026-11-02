"""Exercises data_service against the real precomputed results already
checked into models/results/ — no mocking needed, these files ship with the
repo and are the actual data the dashboard reads."""
from app.services import data_service


def test_get_risk_scores_normalizes_deep_analysis_field_for_fd001():
    scores = data_service.get_risk_scores("FD001")
    assert scores, "expected FD001 risk_scores.json to be present"
    assert any(s["deep_analysis"] for s in scores)


def test_get_risk_scores_normalizes_top15_risk_into_deep_analysis_for_multicond():
    scores = data_service.get_risk_scores("FD002")
    assert scores, "expected FD002 risk_scores.json to be present"
    flagged = [s for s in scores if s["deep_analysis"]]
    assert flagged
    assert all(s["top15_risk"] for s in flagged)


def test_get_agent_summary_flattens_multicond_nested_schema():
    summary = data_service.get_agent_summary("FD002")
    # Should look like the FD001 flat shape, not the raw nested all_engines/... shape.
    assert "all_engines" not in summary
    assert {"total_engines_analyzed", "confirmed", "contradicted", "novel"} <= summary.keys()


def test_get_agent_summary_passes_through_fd001_flat_schema_unchanged():
    summary = data_service.get_agent_summary("FD001")
    assert {"total_engines_analyzed", "confirmed", "contradicted", "novel"} <= summary.keys()


def test_get_shap_stability_returns_empty_dict_when_not_yet_computed():
    # shap_stability.json requires a trained checkpoint to produce and isn't
    # shipped in the repo, so this should degrade gracefully, not error.
    assert data_service.get_shap_stability("FD001") == {}


def test_get_subsets_lists_all_three_subsets_with_engine_counts():
    subsets = {s["subset"]: s["n_test_engines"] for s in data_service.get_subsets()}
    assert subsets.get("FD001") == 100
    assert subsets.get("FD002") == 259
    assert subsets.get("FD004") == 248
