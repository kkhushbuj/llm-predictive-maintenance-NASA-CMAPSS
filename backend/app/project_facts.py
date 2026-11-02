"""Honest, data-backed answers about what this project does well and where it
falls short. Numbers come from the results files at request time (so they
follow retraining); the qualitative points come from REPORT.md and
REPORT_FD002_FD004.md. Nothing here is written to flatter the project."""
import json

from app.config import RESULTS_DIR
from app.services import data_service

# Plain-LSTM baselines from the literature, scored with capped truth labels.
# Benchmark papers do not always state their scoring convention, so these are
# indicative comparisons, not exact ones.
BASELINE = {
    "FD001": ("a published LSTM baseline", 16.78),
    "FD002": ("a published plain-LSTM baseline", 24.49),
    "FD004": ("a published plain-LSTM baseline", 28.17),
}

FACTS_SHEET = """=== PROJECT FACTS (use for questions about trust, limits, strengths, accuracy) ===
Be honest and balanced. Never exaggerate and never defend the project with invented evidence.
If something was not measured or reported (real-fleet performance, deployment, how well the \
risk ranking matches true failure order), say plainly that it has not been tested.

Strengths (real):
- Competitive accuracy: on the standard capped-truth scoring the LSTM beats published \
plain-LSTM baselines on FD001, FD002 and FD004, but it does not reach 2024-26 state of the art.
- Every test engine has a SHAP explanation of which sensors drive its prediction.
- Literature, synthesis, hypothesis and a critique agent (a separate prompt on the same Cohere model as synthesis) \
cross-check the explanations; the pipeline answers "Novel" instead of forcing a verdict.
- It audits itself: LLM-sourced benchmark rows that were fabricated or mislabeled were caught \
and removed, and evaluation criteria were written down before results existed.
- Sensor choice was re-checked per operating-condition dataset (it kept three sensors FD001 dropped).

Limitations (real):
- Predictions are capped near 125 cycles, so engines with longer true life get large errors \
against raw labels (FD002 28.4 RMSE, FD004 27.3 raw vs about 15 capped).
- Not state of the art; newer hybrid models report lower error.
- The literature knowledge base is thin (about 5 of ~14 informative sensors in depth on FD001, \
and it misses sensor 11 on FD002), so "Novel" mostly means "not covered by the search".
- The critique agent mostly disagrees, always with the same SHAP-artifact argument, and the \
pipeline cannot test that claim, so it is a caution flag, not a verdict.
- LLM parts can be wrong: the literature agent produced fabricated benchmark rows that an audit \
caught, and this chat assistant is a small language model.
- Only NASA's simulated C-MAPSS data was used; the risk score is relative to the fleet, not a \
failure probability; results come from one trained model (seed 42).

Fair summary: a transparent, explainable research prototype for prioritising which engines to \
investigate, not a certified maintenance decision tool."""


def _conventions():
    path = RESULTS_DIR / "metrics_all_conventions.json"
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def gather(subset: str) -> dict:
    metrics = data_service.get_training_metrics(subset) or {}
    conv = _conventions().get(subset, {})
    summary = data_service.get_agent_summary(subset) or {}
    stability = list((data_service.get_shap_stability(subset) or {}).values())

    n = conv.get("n") or metrics.get("n_test_engines")
    over = conv.get("n_true_rul_over_125")
    return {
        "subset": subset,
        "n": n,
        "rmse_cap": metrics.get("test_rmse"),
        "rmse_raw": conv.get("rmse_raw"),
        "rmse_le125": conv.get("rmse_raw_only_engines_le_125"),
        "over125_pct": round(100 * over / n) if over is not None and n else None,
        "analyzed": summary.get("total_engines_analyzed"),
        "confirmed": summary.get("confirmed"),
        "contradicted": summary.get("contradicted"),
        "novel": summary.get("novel"),
        "critiqued": summary.get("critiqued"),
        "agrees": summary.get("critique_agrees"),
        "stab_n": len(stability),
        "stab_identical": sum(1 for s in stability if s.get("top_sensor_agreement_rate") == 1.0),
    }


def _f(value, digits=2):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else None


def _strength_bullets(d: dict) -> list:
    out = []
    name, base = BASELINE.get(d["subset"], ("a published baseline", None))
    rmse = _f(d["rmse_cap"])
    if rmse and base:
        out.append(
            f"- **Competitive accuracy.** {rmse} cycles RMSE on {d['subset']}"
            + (f" ({d['n']} test engines)" if d["n"] else "")
            + f", versus about {base} for {name}. Benchmark scoring conventions vary, so treat this as indicative."
        )
    shap = "- **Every prediction is explained.** SHAP shows which sensors shorten or lengthen an engine's remaining life."
    if d["stab_n"] and d["stab_identical"] / d["stab_n"] >= 0.7:
        shap += f" The top sensor was identical across 5 re-runs for {d['stab_identical']} of {d['stab_n']} engines."
    out.append(shap)
    out.append(
        "- **Cross-checked, not just predicted.** Literature agents compare each driver with published research, "
        "and a critique agent (a separate prompt on the same Cohere model as the synthesis agent) challenges the result."
    )
    out.append(
        "- **Candid about what it doesn't know.** It answers \"Novel\" instead of forcing a verdict, "
        "and an audit caught and removed fabricated benchmark rows from its own LLM literature search."
    )
    return out


def _limit_bullets(d: dict) -> list:
    out = []
    if d["rmse_raw"] and d["rmse_le125"]:
        over = f" ({d['over125_pct']}% of engines live longer than that)" if d["over125_pct"] is not None else ""
        out.append(
            f"- **Long-life engines.** Predictions are capped near 125 cycles{over}, so against the raw labels the error is "
            f"{_f(d['rmse_raw'], 1)} cycles RMSE. On engines with 125 cycles or less it is {_f(d['rmse_le125'], 1)}."
        )
    out.append(
        "- **Not state of the art.** On the common capped scoring it beats plain-LSTM baselines, "
        "but newer published hybrid models report lower error."
    )
    if d["analyzed"]:
        extra = ""
        if d["confirmed"] == 0:
            extra = " Zero were confirmed."
            if d["subset"] == "FD002":
                extra += " The project report traces this to the literature search missing sensor 11, this dataset's main driver."
        out.append(
            f"- **The literature check is thin.** {d['novel']} of {d['analyzed']} analyzed engines came back \"Novel\", "
            f"mostly because the search didn't cover the main sensors, not because the model found something new.{extra}"
        )
    if d["critiqued"] and d["agrees"] is not None:
        out.append(
            f"- **The critique agent is a caution flag, not a verdict.** It agreed with the pipeline on {d['agrees']} of "
            f"{d['critiqued']} engines, nearly always raising the same SHAP-artifact concern, which the pipeline can't test."
        )
    if d["stab_n"] and d["stab_identical"] / d["stab_n"] < 0.7:
        out.append(
            f"- **Less stable explanations here.** The top driving sensor stayed the same across 5 re-runs for only "
            f"{d['stab_identical']} of {d['stab_n']} engines."
        )
    out.append(
        "- **Scope.** Only NASA's simulated C-MAPSS data, from one trained model. The risk score is relative to this fleet, "
        "not a failure probability, and how well the ranking matches true failure order isn't reported."
    )
    return out


def limitations_answer(subset: str) -> str:
    d = gather(subset)
    return (
        "**The main limitations, honestly:**\n"
        + "\n".join(_limit_bullets(d))
        + "\n\n**What that means:** use the output as a prioritised shortlist to investigate, not as a final maintenance decision.\n\n"
        "Want the strengths too, or more detail on any of these?"
    )


def build_lessons_answer() -> str:
    """For 'how do I build / what should I know before building a project like
    this?': decline a generic tutorial, but share what this project's own
    reports say went wrong or mattered, which is grounded and actually useful."""
    d = gather("FD004")
    stability = ""
    if d["stab_n"]:
        stability = f" On FD004 the top sensor stayed identical across 5 re-runs for only {d['stab_identical']} of {d['stab_n']} engines."
    return (
        "**I can't give you a full build tutorial, but I can tell you what this project's own reports learned the hard way.** "
        "If you build something similar, watch for:\n"
        "- **Sensor choice per dataset.** Sensors that look constant in one dataset can carry signal in another. "
        "Copying FD001's list would have hidden the fan fault in FD004.\n"
        "- **Operating conditions.** Normalise sensors within each flight regime, not globally.\n"
        "- **The RUL cap.** Capping targets at 125 cycles drives most of the raw-label error. "
        "Decide your scoring convention up front and report both.\n"
        "- **LLM research needs checking.** The literature agent invented benchmark rows; an audit caught and removed them.\n"
        f"- **Explanation stability.** SHAP's top sensor can shift between runs.{stability}\n"
        "- **Evaluation rules first.** This project wrote its criteria down before seeing any results.\n\n"
        "Want more detail on any of these, or an explanation of how this dashboard works?"
    )


def biggest_flaw_answer(subset: str) -> str:
    """One focused answer for 'what is the biggest flaw?', with a clear
    statement that ranking the flaws is a judgement, not a measured fact."""
    d = gather(subset)
    evidence = []
    if d["analyzed"]:
        evidence.append(
            f"- {d['novel']} of {d['analyzed']} analyzed engines came back \"Novel\": the literature search "
            "simply didn't cover the main driving sensors, so there was nothing to confirm or contradict."
        )
    if d["critiqued"] and d["agrees"] is not None:
        evidence.append(
            f"- The critique agent agreed with the pipeline on {d['agrees']} of {d['critiqued']} engines, "
            "nearly always raising the same SHAP-artifact concern, which the pipeline has no test to settle."
        )
    else:
        evidence.append(
            "- The critique agent keeps raising a SHAP-artifact concern that the pipeline has no test to settle."
        )
    cap = ""
    if d["rmse_raw"]:
        cap = f" The 125-cycle prediction cap comes next: against raw labels the error rises to {_f(d['rmse_raw'], 1)} cycles RMSE."
    return (
        "**By my reading, the biggest flaw is that the project can't yet independently validate its own explanations.**\n\n"
        "The model says which sensors drive each prediction, and the literature and critique agents are meant to check that. "
        "In practice that check is weak:\n"
        + "\n".join(evidence)
        + "\n\nSo treat a SHAP explanation here as a strong lead to investigate, not as proven fact."
        + cap
        + "\n\nThat ranking is my judgement, not a measured result. Want the full list of limitations?"
    )


def strengths_answer(subset: str) -> str:
    d = gather(subset)
    return (
        "**What this project does well:**\n"
        + "\n".join(_strength_bullets(d))
        + "\n\nIt isn't perfect, though. Want the honest limitations too?"
    )


def balanced_answer(subset: str) -> str:
    d = gather(subset)
    strengths = _strength_bullets(d)[:3]
    all_limits = _limit_bullets(d)
    # Always surface weak explanation stability when it applies, even in the short list.
    stability = [b for b in all_limits if b.startswith("- **Less stable")]
    rest = [b for b in all_limits if b not in stability]
    limits = (rest[:2] + stability) if stability else rest[:3]
    return (
        "**Short answer:** trust it as an explainable second opinion for deciding which engines to look at first, "
        "not as an autopilot for maintenance decisions. Here is the honest picture.\n\n"
        "**Why it's worth using**\n" + "\n".join(strengths) + "\n\n"
        "**What to keep in mind**\n" + "\n".join(limits) + "\n\n"
        "**Best use:** rank engines, see which sensors drive the prediction, then verify the important cases yourself.\n\n"
        "Want the full list of limitations or strengths?"
    )
