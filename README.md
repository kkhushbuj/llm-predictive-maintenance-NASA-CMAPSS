# LLM-Enhanced Predictive Maintenance on NASA C-MAPSS

Remaining Useful Life (RUL) prediction for turbofan engines using a 2-layer
LSTM, trained on the NASA C-MAPSS dataset (FD001/FD002/FD004), paired with a
multi-agent LLM pipeline that explains and independently critiques each
prediction.

Full write-up: [`REPORT.md`](REPORT.md) (FD001) and
[`REPORT_FD002_FD004.md`](REPORT_FD002_FD004.md) (multi-condition extension),
or the combined [`REPORT_FULL.pdf`](REPORT_FULL.pdf).

## What's here

- **`ml/`** — LSTM training, preprocessing, SHAP explainability, and risk
  scoring for the C-MAPSS subsets.
- **`backend/`** — FastAPI app serving the dashboard and the agent pipeline
  (`backend/app/agents/`): Literature, Synthesis, Hypothesis, and Critique
  agents that cross-check each engine's SHAP explanation against real
  published research.
- **`backend/static/`** — the dashboard frontend (vanilla JS/HTML/CSS),
  served directly by the FastAPI app.
- **`data/raw/CMAPSS/`** — the raw NASA C-MAPSS dataset files.
- **`models/results/`** — trained model outputs: metrics, SHAP explanations,
  risk scores, and per-engine agent analysis, per subset (`FD001` at the
  root, `FD002/`, `FD004/`).
- **`docs/`** — preregistration and report-build tooling.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY / COHERE_API_KEY / ANTHROPIC_API_KEY
```

## Reproduce the pipeline (FD001)

```bash
# 1. Train + evaluate the LSTM
python -m ml.train_lstm

# 2. SHAP explanations for all test engines
python -m ml.explain_shap

# 3. Rank engines, flag top 15 for deep analysis
python -m ml.risk_scoring

# 4. One-time literature research (requires OPENAI_API_KEY)
python -c "from app.agents.literature_agent import build_knowledge_base; build_knowledge_base()"

# 5. Run the agent pipeline on the flagged engines (requires OPENAI_API_KEY, COHERE_API_KEY)
cd backend && python -m app.agents.graph
```

FD002/FD004 use the equivalent multi-condition scripts (`ml/train_multicond.py`,
`backend/app/agents/batch_multicond.py`) — see `REPORT_FD002_FD004.md` for
details and current coverage.

## Run the tests

```bash
python -m pytest
```

Most tests run offline. The few that call the OpenAI API are skipped when
`OPENAI_API_KEY` is not set.

## Run the dashboard

```bash
cd backend && python -m uvicorn app.main:app --reload --port 8000
```

Then open `http://localhost:8000`.

## The dashboard

- **Fleet overview** for FD001, FD002 and FD004: stat cards, a fleet-health map
  coloured by risk, a predicted-vs-true RUL chart, a ticker of the highest-risk
  engines, and a sortable, searchable engine table.
- **Engine page:** risk gauge, SHAP sensor attribution, sensor trend, the
  agent analysis (synthesis, hypothesis, critique) and an engine assistant.
- Dark and light themes, keyboard-accessible tables and controls, and all
  motion switches off for visitors who prefer reduced motion.

## The chat assistant

Questions about an engine are answered by a language model from that engine's
own data. Questions about the project itself (its limitations, strengths,
whether to trust it, or lessons for building something similar) are answered
from the results files in `models/results/` through `backend/app/project_facts.py`,
so the numbers stay current and the answers stay honest and two-sided.
Answers stream from `POST /api/chat/stream`; `POST /api/chat` returns the same
answer in one piece.
