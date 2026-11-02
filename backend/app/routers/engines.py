from fastapi import APIRouter, HTTPException, Query
from openai import AuthenticationError

from app.agents.graph import analyze_single_engine
from app.config import COHERE_API_KEY, OPENAI_API_KEY
from app.services import data_service
from app.services.data_service import SUBSETS

router = APIRouter(prefix="/api/engines", tags=["engines"])

SubsetParam = Query("FD001", description="C-MAPSS subset: FD001, FD002, or FD004")


def _check_subset(subset: str):
    if subset not in SUBSETS:
        raise HTTPException(400, f"Unknown subset '{subset}' — must be one of {SUBSETS}")


@router.get("")
def list_engines(subset: str = SubsetParam):
    """All test engines with the cheap ML-only risk score, for the given subset."""
    _check_subset(subset)
    return data_service.get_risk_scores(subset)


@router.get("/{unit}")
def get_engine(unit: int, subset: str = SubsetParam):
    _check_subset(subset)
    engines = {e["unit"]: e for e in data_service.get_risk_scores(subset)}
    if unit not in engines:
        raise HTTPException(404, f"Engine {unit} not found in {subset}")

    result = dict(engines[unit])
    shap = data_service.get_shap_explanations(subset).get(unit)
    if shap:
        result["shap"] = shap

    agent_analysis = data_service.get_agent_analysis(unit, subset)
    if agent_analysis:
        result["agent_analysis"] = agent_analysis

    return result


@router.get("/{unit}/sensors")
def get_engine_sensors(unit: int, subset: str = SubsetParam):
    _check_subset(subset)
    series = data_service.get_engine_sensor_series(unit, subset)
    if series is None:
        raise HTTPException(404, f"Sensor data for engine {unit} not found")
    return series


@router.post("/{unit}/analyze")
def analyze_engine(unit: int, subset: str = SubsetParam):
    """On-demand Synthesis -> Hypothesis -> Critique pipeline for any engine,
    not just the pre-flagged top 15 (e.g. a user wants a full literature
    cross-check on a mid-risk engine)."""
    _check_subset(subset)
    if not OPENAI_API_KEY or not COHERE_API_KEY:
        raise HTTPException(
            503,
            "OPENAI_API_KEY and COHERE_API_KEY must be configured on the server to run deep analysis.",
        )

    engines = {e["unit"]: e for e in data_service.get_risk_scores(subset)}
    if unit not in engines:
        raise HTTPException(404, f"Engine {unit} not found in {subset}")

    shap = data_service.get_shap_explanations(subset).get(unit)
    if not shap:
        raise HTTPException(404, f"No SHAP explanation for engine {unit} — run ml/explain_shap.py first.")

    if data_service.get_agent_analysis(unit, subset):
        raise HTTPException(409, f"Engine {unit} has already been analyzed.")

    try:
        return analyze_single_engine(shap, include_critique=True, subset=subset)
    except AuthenticationError:
        raise HTTPException(
            502,
            "OpenAI rejected the configured OPENAI_API_KEY (401) — it's invalid or expired. "
            "Update it in .env and restart the server.",
        )
