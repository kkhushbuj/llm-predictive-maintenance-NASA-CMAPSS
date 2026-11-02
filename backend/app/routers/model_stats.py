from fastapi import APIRouter, HTTPException, Query

from app.services import data_service
from app.services.data_service import SUBSETS

router = APIRouter(prefix="/api/model", tags=["model"])

SubsetParam = Query("FD001", description="C-MAPSS subset: FD001, FD002, or FD004")


def _check_subset(subset: str):
    if subset not in SUBSETS:
        raise HTTPException(400, f"Unknown subset '{subset}' — must be one of {SUBSETS}")


@router.get("/subsets")
def get_subsets():
    """Which C-MAPSS subsets have precomputed results, for the dashboard's subset switcher."""
    return data_service.get_subsets()


@router.get("/metrics")
def get_model_metrics(subset: str = SubsetParam):
    """LSTM test RMSE / NASA score, for the report/write-up view."""
    _check_subset(subset)
    return data_service.get_training_metrics(subset)


@router.get("/knowledge-base")
def get_knowledge_base(subset: str = SubsetParam):
    _check_subset(subset)
    return data_service.get_knowledge_base(subset)


@router.get("/agent-summary")
def get_agent_summary(subset: str = SubsetParam):
    """Counts of Confirmed / Contradicted / Novel across deep-analyzed engines."""
    _check_subset(subset)
    return data_service.get_agent_summary(subset)
