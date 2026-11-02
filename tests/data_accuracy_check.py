"""Data accuracy audit for FD001 / FD002 / FD004, run against the real files.

    python tests/data_accuracy_check.py        # prints the full report

tests/test_data_accuracy.py runs the same checks under pytest.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services import data_service  # noqa: E402
from ml.config import COLUMNS, DATA_DIR  # noqa: E402

SUBSETS = ["FD001", "FD002", "FD004"]
SAMPLE = 10
SEED = 7
client = TestClient(app)


def _base(subset):
    return data_service._base_dir(subset)


def _raw(subset, kind):
    return pd.read_csv(DATA_DIR / f"{kind}_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)


def check_risk_scores(subset):
    rs = json.loads((_base(subset) / "risk_scores.json").read_text())
    out = []
    missing = [e.get("unit") for e in rs if any(k not in e for k in ("unit", "predicted_RUL", "true_RUL", "risk_score", "risk_rank"))]
    out.append(("a1 required fields present", not missing, f"{len(rs)} engines, {len(missing)} missing fields"))
    bad_range = [e["unit"] for e in rs if not (0 <= e["risk_score"] <= 100)]
    out.append(("a2 risk_score within 0-100", not bad_range, f"{len(bad_range)} out of range"))
    ranks = sorted(e["risk_rank"] for e in rs)
    out.append(("a3 ranks unique and 1..N", ranks == list(range(1, len(rs) + 1)), f"min={ranks[0]} max={ranks[-1]} N={len(rs)}"))
    df = pd.DataFrame(rs)
    rho = df["risk_score"].corr(df["predicted_RUL"], method="spearman")
    out.append(("a4 risk_score vs predicted_RUL Spearman strongly negative", rho < -0.95, f"rho={rho:.4f}"))
    rank_rho = df["risk_rank"].corr(df["risk_score"], method="spearman")
    out.append(("a5 rank 1 = highest risk_score", rank_rho < -0.95, f"rank-vs-score rho={rank_rho:.4f}"))
    return out


def check_shap_direction(subset):
    shap = json.loads((_base(subset) / "shap_explanations.json").read_text())
    rng = random.Random(SEED)
    picked = rng.sample(shap, min(SAMPLE, len(shap)))
    mismatches = []
    for e in picked:
        for s in e["top_sensors"]:
            expected = "increases_risk" if s["signed_contribution"] < 0 else "decreases_risk"
            if s["direction"] != expected:
                mismatches.append((e["unit"], s["sensor"], s["signed_contribution"], s["direction"]))
    return [("b direction == increases_risk iff signed_contribution < 0", not mismatches,
             f"{len(picked)} engines sampled, {len(mismatches)} mismatches {mismatches[:3]}")]


def check_sensor_series(subset):
    rs = json.loads((_base(subset) / "risk_scores.json").read_text())
    shap = {e["unit"]: e for e in json.loads((_base(subset) / "shap_explanations.json").read_text())}
    parquet = pd.read_parquet(data_service.PROCESSED_DIR / f"test_{subset}.parquet")
    raw_test, raw_train = _raw(subset, "test"), _raw(subset, "train")
    lo, hi = raw_train.min(), raw_train.max()
    rng = random.Random(SEED)
    units = rng.sample([e["unit"] for e in rs], min(SAMPLE, len(rs)))

    api_vs_parquet, cycles_bad, scaling_bad = [], [], []
    for u in units:
        res = client.get(f"/api/engines/{u}/sensors", params={"subset": subset})
        api = pd.DataFrame(res.json()) if res.status_code == 200 else pd.DataFrame()
        pq = parquet[parquet["unit"] == u].sort_values("cycle").reset_index(drop=True)
        if api.empty or len(api) != len(pq) or not np.allclose(api["sensor_2"], pq["sensor_2"]):
            api_vs_parquet.append(u)
        max_cycle = int(raw_test.loc[raw_test["unit"] == u, "cycle"].max())
        if api.empty or len(api) != max_cycle or int(api["cycle"].max()) != max_cycle:
            cycles_bad.append((u, len(api), max_cycle))
        # independent re-derivation of the displayed scaling from raw files
        raw_u = raw_test[raw_test["unit"] == u].sort_values("cycle").reset_index(drop=True)
        for s in shap[u]["top_sensors"]:
            name = s["sensor"]
            if hi[name] == lo[name] or api.empty:
                continue
            expected = (raw_u[name] - lo[name]) / (hi[name] - lo[name])
            if not np.allclose(api[name].to_numpy(), expected.to_numpy(), atol=1e-6):
                scaling_bad.append((u, name))
    return [
        ("c1 API sensor series equals processed parquet rows", not api_vs_parquet, f"{len(units)} engines, bad={api_vs_parquet}"),
        ("c2 cycles on chart == engine max cycle in raw test file", not cycles_bad, f"bad(unit, shown, max)={cycles_bad}"),
        ("c3 displayed values == train-min/max scaling of raw readings", not scaling_bad, f"bad={scaling_bad[:5]}"),
    ]


def check_calibration_points(subset):
    rs = {e["unit"]: e for e in json.loads((_base(subset) / "risk_scores.json").read_text())}
    api = client.get("/api/engines", params={"subset": subset}).json()
    bad = [e["unit"] for e in api if rs[e["unit"]]["true_RUL"] != e["true_RUL"] or rs[e["unit"]]["predicted_RUL"] != e["predicted_RUL"]]
    return [("d calibration points (true, predicted) == risk_scores.json", len(api) == len(rs) and not bad, f"{len(api)} points, {len(bad)} differ")]


def check_dashboard_rmse(subset):
    rs = json.loads((_base(subset) / "risk_scores.json").read_text())
    err = np.array([e["predicted_RUL"] - e["true_RUL"] for e in rs])
    computed = float(np.sqrt(np.mean(err**2)))
    shown = client.get("/api/model/metrics", params={"subset": subset}).json()["test_rmse"]
    return [("e dashboard test_rmse == sqrt(mean(err^2)) within 0.05", abs(computed - shown) <= 0.05,
             f"dashboard={shown:.3f} computed={computed:.3f} diff={abs(computed - shown):.4f}")]


def top_errors(subset, n=10):
    rs = json.loads((_base(subset) / "risk_scores.json").read_text())
    rows = sorted(rs, key=lambda e: -abs(e["predicted_RUL"] - e["true_RUL"]))[:n]
    return [(e["unit"], round(e["predicted_RUL"], 1), e["true_RUL"], round(e["predicted_RUL"] - e["true_RUL"], 1)) for e in rows]


def run_all():
    results = {}
    for subset in SUBSETS:
        checks = []
        for fn in (check_risk_scores, check_shap_direction, check_sensor_series, check_calibration_points, check_dashboard_rmse):
            checks.extend(fn(subset))
        results[subset] = checks
    return results


def main():
    results = run_all()
    failures = 0
    for subset, checks in results.items():
        print(f"\n=== {subset} ===")
        for name, ok, detail in checks:
            failures += not ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        print(f"  f. top 10 by |error| (unit, predicted, true, error):")
        for row in top_errors(subset):
            print("     ", row)
    print(f"\nTOTAL FAILURES: {failures}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
