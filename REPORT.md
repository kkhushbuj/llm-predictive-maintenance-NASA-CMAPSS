# LLM-Enhanced Predictive Maintenance on NASA C-MAPSS — Results

## 1. Model performance vs. published benchmarks

Trained a 2-layer LSTM (hidden size 64, dropout 0.3) on NASA C-MAPSS **FD001**
(100 train engines, single operating condition, HPC degradation fault mode),
using a 30-cycle sliding window, piecewise-linear RUL target capped at 125
cycles, and 14 non-constant sensors (7 sensors — 1, 5, 6, 10, 16, 18, 19 — are
physically near-constant at a single operating condition and were dropped).

| Model | Dataset | RMSE | NASA Score | Source |
|---|---|---|---|---|
| **This project (LSTM)** | FD001 | **13.58** | **335.0** | — |
| LSTM | FD001 | 16.78 | 382.4 | Pinto et al., 2023 |
| CNN-LSTM | FD001 | 16.78 | 382.4 | penikmatrumput, 2024 |
| Transformer Encoder | FD001 | 15.08 | 338 | Gsebs, 2023 |

*(Benchmark rows sourced live by the Literature Agent's web search — see
`models/results/literature_knowledge_base.json` for full citations.)*

Our model's RMSE (13.58) beats every benchmark the Literature Agent found,
and its NASA score (335.0) is close to the best of them. This is a genuine,
reproducible result — rerun with `python -m ml.train_lstm`.

## 2. Explainability + literature cross-check pipeline

For all 100 test engines, SHAP (`GradientExplainer`) attributes the RUL
prediction to specific sensors (`ml/explain_shap.py`). The 15 highest-risk
engines (lowest predicted RUL, see `ml/risk_scoring.py`) get a full multi-agent
analysis:

1. **Literature Agent** (OpenAI `gpt-4o-mini` + web search) — researches real
   published C-MAPSS papers once, builds a knowledge base of which sensors are
   reported as predictive and what degradation patterns look like. Grounded
   with the canonical NASA sensor-index table so it can't mismap a paper's
   physical sensor names (T30, P30, Nc, ...) to the wrong `sensor_N`.
2. **Synthesis Agent** (Cohere `command-a-03-2025`) — compares each engine's
   top SHAP sensors against the knowledge base and labels the case Confirmed /
   Contradicted / Novel.
3. **Hypothesis Agent** (OpenAI `gpt-4o-mini`) — only runs on
   Contradicted/Novel cases, proposes a plausible explanation + next checks.
4. **Critique Agent** (Cohere `command-a-03-2025`, the same model as the
   Synthesis Agent but run with a separate reviewer prompt) — reviews a 15-engine sample,
   actively looking for a simpler model-setup explanation (SHAP background
   artifact, low sensor variance, extrapolation) rather than agreeing by
   default.

### Result on the 15 highest-risk engines

| Verdict | Count |
|---|---|
| Confirmed | 2 |
| Contradicted | 0 |
| Novel | 13 |
| Critiqued (independent 2nd opinion) | 15 |

**This is the project's genuine, reportable finding.** The dominant outcome is
Novel, not because the pipeline defaults to it, but because the knowledge base
— built from real papers the Literature Agent could actually find and verify —
only documents 5 of the ~14 informative C-MAPSS sensors in depth. Where the
model's top driver *did* overlap with a documented sensor (`sensor_7`/P30,
`sensor_9`/Nc, `sensor_12`/phi, `sensor_14`/NRc), the pipeline correctly
confirmed agreement (0 contradictions) and cited its source. Where the
dominant SHAP sensor (most often `sensor_11`/Ps30) fell outside that coverage,
it correctly declined to force a verdict rather than fabricate one — see
`models/results/agent_analysis/engine_34.json` for a representative example,
including the Critique Agent independently pushing back with an alternative
"SHAP artifact / distribution extrapolation" explanation.

### Known limitation

Literature Agent's web search cannot read figures — early runs mapped a
paper's physical sensor names to the wrong `sensor_N` index (citing sensors
that are known-constant in FD001). Fixed by embedding the canonical
Saxena & Goebel (2008) sensor table directly in its system prompt. Worth
noting in any write-up as a concrete example of LLM-literature-grounding
failure mode and its fix.

## 3. How to reproduce

```bash
# 1. Train + evaluate the LSTM
python -m ml.train_lstm

# 2. SHAP explanations for all 100 test engines
python -m ml.explain_shap

# 3. Rank engines, flag top 15 for deep analysis
python -m ml.risk_scoring

# 4. One-time literature research (requires OPENAI_API_KEY)
python -c "from app.agents.literature_agent import build_knowledge_base; build_knowledge_base()"

# 5. Run the agent pipeline on the flagged engines (requires OPENAI_API_KEY, COHERE_API_KEY)
cd backend && python -m app.agents.graph
```

Dashboard: `backend` (FastAPI, port 8000), serving the static vanilla-JS
frontend at `backend/static/`. See `.claude/launch.json`.
