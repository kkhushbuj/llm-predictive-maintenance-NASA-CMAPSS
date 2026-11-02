"""Cheap ML-only risk score for all engines; flags the highest-risk subset
for expensive LLM-based deep analysis (Synthesis/Hypothesis/Critique agents)."""
import json

from ml.config import RESULTS_DIR

DEEP_ANALYSIS_COUNT = 15


def compute_risk_scores():
    with open(RESULTS_DIR / "test_predictions.json") as f:
        predictions = json.load(f)

    max_rul = max(p["predicted_RUL"] for p in predictions) or 1.0

    for p in predictions:
        # Lower predicted RUL -> higher risk. Scaled 0-100.
        p["risk_score"] = round(100 * (1 - p["predicted_RUL"] / max_rul), 2)

    ranked = sorted(predictions, key=lambda p: p["predicted_RUL"])
    for rank, p in enumerate(ranked, start=1):
        p["risk_rank"] = rank
        p["deep_analysis"] = rank <= DEEP_ANALYSIS_COUNT

    with open(RESULTS_DIR / "risk_scores.json", "w") as f:
        json.dump(ranked, f, indent=2)

    print(f"Ranked {len(ranked)} engines. Top {DEEP_ANALYSIS_COUNT} flagged for deep analysis:")
    for p in ranked[:DEEP_ANALYSIS_COUNT]:
        print(f"  unit {p['unit']:3d}  predicted_RUL={p['predicted_RUL']:.1f}  risk={p['risk_score']}")

    return ranked


if __name__ == "__main__":
    compute_risk_scores()
