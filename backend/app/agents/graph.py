"""LangGraph orchestration for the per-engine deep-analysis pipeline.

Flow per engine:
  SHAP result -> Synthesis (Confirmed/Contradicted/Novel)
      -> if Contradicted/Novel: Hypothesis
      -> if engine is in the critique sample (1-15): Critique (independent 2nd opinion)
"""
import json
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.critique_agent import critique
from app.agents.hypothesis_agent import investigate
from app.agents.literature_agent import load_knowledge_base
from app.agents.synthesis_agent import synthesize
from app.config import AGENT_ANALYSIS_DIR, RESULTS_DIR
from app.services.data_service import get_shap_stability

CRITIQUE_SAMPLE_SIZE = 15


class EngineState(TypedDict):
    unit: int
    engine_shap: dict
    knowledge_base: dict
    include_critique: bool
    synthesis_result: Optional[dict]
    hypothesis_result: Optional[dict]
    critique_result: Optional[dict]
    shap_stability: Optional[dict]


def synthesis_node(state: EngineState) -> EngineState:
    result = synthesize(state["engine_shap"], state["knowledge_base"])
    return {**state, "synthesis_result": result}


def route_after_synthesis(state: EngineState) -> str:
    verdict = state["synthesis_result"]["verdict"]
    if verdict in ("Contradicted", "Novel"):
        return "hypothesis"
    return "maybe_critique"


def hypothesis_node(state: EngineState) -> EngineState:
    result = investigate(state["engine_shap"], state["synthesis_result"])
    return {**state, "hypothesis_result": result}


def route_to_critique_or_end(state: EngineState) -> str:
    return "critique" if state["include_critique"] else "end"


def critique_node(state: EngineState) -> EngineState:
    result = critique(
        state["engine_shap"],
        state["synthesis_result"],
        state.get("hypothesis_result"),
        state.get("shap_stability"),
    )
    return {**state, "critique_result": result}


def passthrough_node(state: EngineState) -> EngineState:
    return state


def build_graph():
    graph = StateGraph(EngineState)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("hypothesis", hypothesis_node)
    graph.add_node("critique", critique_node)
    # Confirmed engines skip hypothesis but still need a real node to branch
    # from before deciding whether they're in the critique sample.
    graph.add_node("route_confirmed", passthrough_node)

    graph.set_entry_point("synthesis")
    graph.add_conditional_edges(
        "synthesis",
        route_after_synthesis,
        {"hypothesis": "hypothesis", "maybe_critique": "route_confirmed"},
    )
    graph.add_conditional_edges(
        "hypothesis",
        route_to_critique_or_end,
        {"critique": "critique", "end": END},
    )
    graph.add_conditional_edges(
        "route_confirmed",
        route_to_critique_or_end,
        {"critique": "critique", "end": END},
    )
    graph.add_edge("critique", END)

    return graph.compile()


def analyze_single_engine(engine_shap: dict, include_critique: bool = True, subset: str = "FD001") -> dict:
    """On-demand version of the pipeline for one engine (e.g. a user clicking
    "Run deep analysis" on an engine that wasn't in the pre-computed top 15).
    Persists the result the same way run_pipeline does, and updates the
    running _summary.json counts so the dashboard stays consistent."""
    kb_path = (
        RESULTS_DIR / "literature_knowledge_base.json"
        if subset == "FD001"
        else RESULTS_DIR / subset / "literature_knowledge_base.json"
    )
    knowledge_base = load_knowledge_base(kb_path)
    compiled_graph = build_graph()

    stability = get_shap_stability(subset).get(engine_shap["unit"])

    initial_state: EngineState = {
        "unit": engine_shap["unit"],
        "engine_shap": engine_shap,
        "knowledge_base": knowledge_base,
        "include_critique": include_critique,
        "synthesis_result": None,
        "hypothesis_result": None,
        "critique_result": None,
        "shap_stability": stability,
    }
    state = compiled_graph.invoke(initial_state)

    analysis_dir = AGENT_ANALYSIS_DIR if subset == "FD001" else RESULTS_DIR / subset / "agent_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / f"engine_{engine_shap['unit']}.json"
    with open(out_path, "w") as f:
        json.dump(state, f, indent=2)

    summary_path = analysis_dir / "_summary.json"
    summary = {"total_engines_analyzed": 0, "confirmed": 0, "contradicted": 0, "novel": 0, "critiqued": 0}
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
        if "all_engines" in existing:
            # Multicond batch summary (batch_multicond.summarize) uses a nested
            # shape — flatten it to the counters this function increments.
            all_engines = existing["all_engines"]
            existing = {
                "total_engines_analyzed": existing.get("engines_analyzed", all_engines.get("n", 0)),
                "confirmed": all_engines.get("confirmed", 0),
                "contradicted": all_engines.get("contradicted", 0),
                "novel": all_engines.get("novel", 0),
                "critiqued": existing.get("critiqued", 0),
            }
        summary = existing
    summary["total_engines_analyzed"] += 1
    summary[state["synthesis_result"]["verdict"].lower()] += 1
    if include_critique:
        summary["critiqued"] += 1
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return state


def run_pipeline(engine_shap_list: list[dict], deep_analysis_units: set[int]) -> list[dict]:
    """Run the full agent pipeline for the flagged high-risk engines.

    engine_shap_list: output of ml/explain_shap.py (one dict per engine)
    deep_analysis_units: unit numbers flagged by risk_scoring.py for deep analysis
    """
    knowledge_base = load_knowledge_base()
    deep_engines = [e for e in engine_shap_list if e["unit"] in deep_analysis_units]
    critique_units = set(sorted(deep_analysis_units)[:CRITIQUE_SAMPLE_SIZE])
    stability_by_unit = get_shap_stability("FD001")

    compiled_graph = build_graph()
    results = []
    for engine in deep_engines:
        initial_state: EngineState = {
            "unit": engine["unit"],
            "engine_shap": engine,
            "knowledge_base": knowledge_base,
            "include_critique": engine["unit"] in critique_units,
            "synthesis_result": None,
            "hypothesis_result": None,
            "critique_result": None,
            "shap_stability": stability_by_unit.get(engine["unit"]),
        }

        state = compiled_graph.invoke(initial_state)
        results.append(state)
        out_path = AGENT_ANALYSIS_DIR / f"engine_{engine['unit']}.json"
        with open(out_path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"engine {engine['unit']:3d}  verdict={state['synthesis_result']['verdict']:12s}  critique={'yes' if state['include_critique'] else 'no'}")

    summary = {
        "total_engines_analyzed": len(results),
        "confirmed": sum(1 for r in results if r["synthesis_result"]["verdict"] == "Confirmed"),
        "contradicted": sum(1 for r in results if r["synthesis_result"]["verdict"] == "Contradicted"),
        "novel": sum(1 for r in results if r["synthesis_result"]["verdict"] == "Novel"),
        "critiqued": sum(1 for r in results if r["include_critique"]),
    }
    with open(AGENT_ANALYSIS_DIR / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:", summary)
    return results


if __name__ == "__main__":
    import sys

    from app.config import ROOT

    sys.path.insert(0, str(ROOT))
    from ml.risk_scoring import compute_risk_scores

    ranked = compute_risk_scores()
    deep_units = {p["unit"] for p in ranked if p["deep_analysis"]}

    with open(AGENT_ANALYSIS_DIR.parent / "shap_explanations.json") as f:
        shap_list = json.load(f)

    run_pipeline(shap_list, deep_units)
