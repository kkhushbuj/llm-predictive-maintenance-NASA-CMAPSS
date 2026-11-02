"""Reads pre-computed ML/agent artifacts from disk for the API to serve.

All heavy computation (LSTM training, SHAP, agent pipeline) runs offline via
the ml/ scripts and backend/app/agents/graph.py; the API is a thin read layer
over their JSON/parquet outputs so the dashboard stays fast. These files are
small and rarely read (once per dashboard load), and they change every time
an offline pipeline reruns, so they are read fresh rather than cached in
process memory — a cache would otherwise keep serving stale/empty results
after a background pipeline finishes.

Results live under `models/results/` for FD001 (the original single-condition
run) and under `models/results/<SUBSET>/` for the multi-condition subsets
(FD002, FD004) produced by `ml/train_multicond.py` / `batch_multicond.py`.
"""
import json

import pandas as pd

from app.config import AGENT_ANALYSIS_DIR, KNOWLEDGE_BASE_PATH, RESULTS_DIR

PROCESSED_DIR = RESULTS_DIR.parent.parent / "data" / "processed"

SUBSETS = ["FD001", "FD002", "FD004"]


def _base_dir(subset: str):
    return RESULTS_DIR if subset == "FD001" else RESULTS_DIR / subset


def _agent_analysis_dir(subset: str):
    return AGENT_ANALYSIS_DIR if subset == "FD001" else _base_dir(subset) / "agent_analysis"


def _load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_subsets():
    """Which C-MAPSS subsets have precomputed results on disk, with basic counts."""
    out = []
    for subset in SUBSETS:
        risk = get_risk_scores(subset)
        if not risk and subset != "FD001":
            continue
        out.append({"subset": subset, "n_test_engines": len(risk)})
    return out


def get_risk_scores(subset: str = "FD001"):
    scores = _load_json(_base_dir(subset) / "risk_scores.json") or []
    # FD001 flags the deep-analysis sample as "deep_analysis"; the multicond
    # pipeline calls the equivalent field "top15_risk". Normalize to one name
    # so the dashboard doesn't need to branch on subset.
    for s in scores:
        s.setdefault("deep_analysis", s.get("top15_risk", False))
    return scores


def get_shap_explanations(subset: str = "FD001"):
    data = _load_json(_base_dir(subset) / "shap_explanations.json") or []
    return {e["unit"]: e for e in data}


def get_training_metrics(subset: str = "FD001"):
    return _load_json(_base_dir(subset) / "training_metrics.json") or {}


def get_knowledge_base(subset: str = "FD001"):
    path = _base_dir(subset) / "literature_knowledge_base.json" if subset != "FD001" else KNOWLEDGE_BASE_PATH
    return _load_json(path) or {}


def get_agent_analysis(unit: int, subset: str = "FD001"):
    return _load_json(_agent_analysis_dir(subset) / f"engine_{unit}.json")


def get_shap_stability(subset: str = "FD001"):
    """Per-engine SHAP top-sensor stability across reruns with different
    background samples (see ml/shap_stability.py). None per engine, and an
    empty dict overall, until that offline script has been run for the
    subset — it requires a trained checkpoint, which is not always present."""
    data = _load_json(_base_dir(subset) / "shap_stability.json") or []
    return {s["unit"]: s for s in data}


def get_agent_summary(subset: str = "FD001"):
    summary = _load_json(_agent_analysis_dir(subset) / "_summary.json") or {}
    if "all_engines" not in summary:
        return summary
    # Multicond batch pipeline nests counts under all_engines/top15_highest_risk_secondary;
    # flatten to the same shape the FD001 pipeline (graph.py) already produces.
    all_engines = summary["all_engines"]
    return {
        "subset": summary.get("subset", subset),
        "total_engines_analyzed": summary.get("engines_analyzed", all_engines.get("n", 0)),
        "confirmed": all_engines.get("confirmed", 0),
        "contradicted": all_engines.get("contradicted", 0),
        "novel": all_engines.get("novel", 0),
        "critiqued": summary.get("critiqued", 0),
        "critique_agrees": summary.get("critique_agrees", 0),
        "top15_highest_risk_secondary": summary.get("top15_highest_risk_secondary"),
        "criterion": summary.get("criterion"),
    }


def get_engine_sensor_series(unit: int, subset: str = "FD001"):
    """Full sensor time series for one test engine, for charting."""
    path = PROCESSED_DIR / f"test_{subset}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    engine_df = df[df["unit"] == unit].sort_values("cycle")
    if engine_df.empty:
        return None
    return engine_df.to_dict(orient="records")
