# FD002 / FD004 — criteria fixed before any results

Written 2026-09-23T13:55:06Z, before any FD002/FD004 model was trained or any
agent run. Not to be edited after results exist; deviations go in the report.

## Evaluation criteria (from the project owner's spec)
- Ground truth: the provided `RUL_FD002.txt` / `RUL_FD004.txt` for the test set.
- Full Synthesis -> Hypothesis -> Critique pipeline on **every** test engine
  (FD002: 259, FD004: 248 — the spec said "100"; the test sets are larger, and
  the intent was "all engines, not a sample").
- "The same pattern holds" = Novel rate within ±15 percentage points of FD001's
  13/15 = 86.7%, i.e. between 71.7% and 100%. Outside that range = a genuinely
  different result, to be investigated and explained.
- Worse RMSE than FD001 is reported as-is, with the reason; no tuning to close it.

## Known confound, stated in advance
FD001's 86.7% comes from the 15 highest-risk engines only; FD002/FD004 cover
all engines. The primary verdict uses the criterion above unchanged. As a
clearly-labelled secondary number, the Novel rate among each subset's 15
highest-risk engines is also reported.

## Data decisions
- Regimes: k-means, k=6, on the 3 operational settings, fit on train only.
- Normalization: z-score per sensor within regime, train statistics.
- Sensor selection: drop a sensor only if it is constant inside every regime
  (same principle as FD001's near-constant rule, applied within regime).
- Inputs: kept sensors + 6 regime one-hot columns.

## Model (trained from scratch per subset; reasoning in the report)
Same architecture as FD001 (2-layer LSTM, hidden 64, dropout 0.3), window 30,
RUL cap 125, Adam 1e-3, batch 256, seed 42. Changes vs FD001, decided now:
engine-level validation split (15% of train engines), target scaled by 125,
max 100 epochs / patience 15. Final model = best validation checkpoint.

## Literature
Separate Literature Agent run per subset; only benchmark rows reported for
that exact subset are used. FD001 benchmark numbers are not reused.

## Addendum 2026-09-23T13:56:27Z (still before any training or agent run)
Smoke test showed kept quasi-binary sensors (sensor_10 in FD002, sensor_16 in
FD004) reach |z| ~ 50 after in-regime scaling because their in-regime std is
tiny; continuous sensors peak at |z| ~ 8.6. Decision: clip in-regime z-scores
to ±10. This caps the flips without altering any continuous sensor.

## Addendum 2026-09-23T18:05:03Z (after model training, BEFORE any Synthesis/Hypothesis/Critique run)
1. Knowledge-base validity rule: sensor_findings for sensors that are constant
   inside every regime of that subset are removed (no finding can apply to a
   sensor that carries no in-regime signal). Removed: FD002 sensor_5,
   sensor_19; FD004 none. Raw agent output kept as *.raw.json.
2. Cohere trial quota (1000/month, ~80 used) cannot cover Synthesis + Critique
   for all 507 engines. Order: Synthesis -> Hypothesis for all engines first
   (the Novel-rate criterion depends only on Synthesis); Critique pass after,
   subject to quota. Any engine not critiqued will be reported as such.
3. Test-set RMSE/score reported against BOTH raw provided RUL (primary, per
   spec) and RUL capped at 125 (the convention FD001's 13.58 used).
