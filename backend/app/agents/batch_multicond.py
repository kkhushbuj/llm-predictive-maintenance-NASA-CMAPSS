"""Agent pipeline over every test engine of FD002/FD004.

Resumable: engines that already have a result file are skipped, so a run cut
short by quota or network errors can be restarted without repeating calls.

    python -m app.agents.batch_multicond synth FD002 FD004
    python -m app.agents.batch_multicond critique FD002 FD004
    python -m app.agents.batch_multicond recritique FD002 FD004
    python -m app.agents.batch_multicond summary FD002 FD004

`recritique` force-recomputes every engine's critique_result with the current
Critique Agent (e.g. after a prompt change), backing up the prior
critique_result per engine first so old vs new stays comparable.
"""
import json
import sys

from app.agents.critique_agent import critique
from app.agents.graph import build_graph
from app.config import RESULTS_DIR
from app.services.data_service import get_shap_stability

MAX_CONSECUTIVE_FAILURES = 3
FD001_NOVEL_RATE = 13 / 15
TOLERANCE_PP = 0.15


def _paths(subset):
    base = RESULTS_DIR / subset
    out = base / "agent_analysis"
    out.mkdir(exist_ok=True)
    return base, out


def synthesis_pass(subset) -> bool:
    base, out = _paths(subset)
    kb = json.loads((base / "literature_knowledge_base.json").read_text())
    engines = json.loads((base / "shap_explanations.json").read_text())
    stability_by_unit = get_shap_stability(subset)
    graph = build_graph()
    failures = 0
    for i, engine in enumerate(engines, start=1):
        path = out / f"engine_{engine['unit']}.json"
        if path.exists():
            continue
        state = {
            "unit": engine["unit"],
            "engine_shap": engine,
            "knowledge_base": kb,
            "include_critique": False,
            "synthesis_result": None,
            "hypothesis_result": None,
            "critique_result": None,
            "shap_stability": stability_by_unit.get(engine["unit"]),
        }
        try:
            result = graph.invoke(state)
        except Exception as e:
            failures += 1
            print(f"[{subset}] engine {engine['unit']} FAILED ({type(e).__name__}: {e})", flush=True)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[{subset}] {failures} consecutive failures — stopping (quota or outage).", flush=True)
                return False
            continue
        failures = 0
        result.pop("knowledge_base")
        path.write_text(json.dumps(result, indent=2))
        print(f"[{subset}] {i}/{len(engines)} engine {engine['unit']}: {result['synthesis_result']['verdict']}", flush=True)
    return True


def critique_pass(subset) -> bool:
    _, out = _paths(subset)
    stability_by_unit = get_shap_stability(subset)
    failures = 0
    for path in sorted(out.glob("engine_*.json")):
        state = json.loads(path.read_text())
        if state.get("critique_result"):
            continue
        stability = state.get("shap_stability") or stability_by_unit.get(state["unit"])
        try:
            state["critique_result"] = critique(
                state["engine_shap"], state["synthesis_result"], state.get("hypothesis_result"), stability
            )
        except Exception as e:
            failures += 1
            print(f"[{subset}] critique {path.stem} FAILED ({type(e).__name__}: {e})", flush=True)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[{subset}] {failures} consecutive failures — stopping (quota or outage).", flush=True)
                return False
            continue
        failures = 0
        state["include_critique"] = True
        path.write_text(json.dumps(state, indent=2))
        print(f"[{subset}] critiqued {path.stem}: agrees={state['critique_result'].get('agrees_with_pipeline')}", flush=True)
    return True


def recritique_pass(subset) -> bool:
    """Force-recompute critique_result for every engine using the current
    (evidence-based) Critique Agent, always passing real shap_stability data.
    Unlike critique_pass, this does NOT skip engines that already have a
    critique_result — it's meant for re-running after a Critique Agent change,
    to compare old vs new behavior on identical inputs (same engine_shap,
    same synthesis_result/hypothesis_result — those are left untouched).

    The old critique_result for every engine is written once, up front, to
    agent_analysis/_critique_backup_pre_stability.json before anything is
    overwritten, so old vs new stays comparable even after this runs.
    """
    _, out = _paths(subset)
    stability_by_unit = get_shap_stability(subset)
    paths = sorted(out.glob("engine_*.json"))

    backup_path = out / "_critique_backup_pre_stability.json"
    if not backup_path.exists():
        backup = {}
        for path in paths:
            state = json.loads(path.read_text())
            if state.get("critique_result"):
                backup[str(state["unit"])] = state["critique_result"]
        backup_path.write_text(json.dumps(backup, indent=2))
        print(f"[{subset}] backed up {len(backup)} pre-stability critique_results -> {backup_path}", flush=True)
    else:
        print(f"[{subset}] backup already exists at {backup_path}, not overwriting", flush=True)

    failures = 0
    for i, path in enumerate(paths, start=1):
        state = json.loads(path.read_text())
        stability = stability_by_unit.get(state["unit"])
        try:
            state["critique_result"] = critique(
                state["engine_shap"], state["synthesis_result"], state.get("hypothesis_result"), stability
            )
        except Exception as e:
            failures += 1
            print(f"[{subset}] recritique {path.stem} FAILED ({type(e).__name__}: {e})", flush=True)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[{subset}] {failures} consecutive failures — stopping (quota or outage).", flush=True)
                return False
            continue
        failures = 0
        state["include_critique"] = True
        path.write_text(json.dumps(state, indent=2))
        print(
            f"[{subset}] {i}/{len(paths)} recritiqued {path.stem}: "
            f"stable={stability.get('stable') if stability else None} "
            f"agrees={state['critique_result'].get('agrees_with_pipeline')}",
            flush=True,
        )
    return True


def _counts(states):
    verdicts = [s["synthesis_result"]["verdict"] for s in states]
    n = len(verdicts)
    novel = verdicts.count("Novel")
    return {
        "n": n,
        "confirmed": verdicts.count("Confirmed"),
        "contradicted": verdicts.count("Contradicted"),
        "novel": novel,
        "novel_rate": novel / n if n else None,
    }


def summarize(subset) -> dict:
    base, out = _paths(subset)
    risk = {r["unit"]: r for r in json.loads((base / "risk_scores.json").read_text())}
    states = [json.loads(p.read_text()) for p in out.glob("engine_*.json")]
    top15 = [s for s in states if risk[s["unit"]]["top15_risk"]]
    critiqued = [s for s in states if s.get("critique_result")]
    overall = _counts(states)
    rate = overall["novel_rate"]
    summary = {
        "subset": subset,
        "test_engines": len(risk),
        "engines_analyzed": len(states),
        "all_engines": overall,
        "top15_highest_risk_secondary": _counts(top15),
        "critiqued": len(critiqued),
        "critique_agrees": sum(1 for s in critiqued if s["critique_result"].get("agrees_with_pipeline")),
        "criterion": {
            "fd001_novel_rate": FD001_NOVEL_RATE,
            "allowed_range": [FD001_NOVEL_RATE - TOLERANCE_PP, min(1.0, FD001_NOVEL_RATE + TOLERANCE_PP)],
            "same_pattern_holds": None
            if rate is None
            else abs(rate - FD001_NOVEL_RATE) <= TOLERANCE_PP,
            "complete": len(states) == len(risk),
        },
    }
    (out / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    stage, subsets = sys.argv[1], sys.argv[2:]
    for s in subsets:
        ok = {
            "synth": synthesis_pass,
            "critique": critique_pass,
            "recritique": recritique_pass,
            "summary": summarize,
        }[stage](s)
        if ok is False:
            sys.exit(1)
