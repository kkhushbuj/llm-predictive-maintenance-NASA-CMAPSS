"""Dashboard chat box: follow-up questions about a specific engine, grounded
in that engine's actual prediction/SHAP/agent-analysis data (no free-floating
LLM guesses)."""
import json
import re
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

from app.config import OPENAI_API_KEY
from app.project_facts import (
    FACTS_SHEET,
    balanced_answer,
    biggest_flaw_answer,
    build_lessons_answer,
    limitations_answer,
    strengths_answer,
)
from app.services import data_service

router = APIRouter(prefix="/api/chat", tags=["chat"])


# Must match Charts.riskTier() in backend/static/js/charts.js exactly, so the
# chatbot never states a risk level that contradicts the badge on the page.
def _risk_level(risk_score: float) -> str:
    if risk_score >= 70:
        return "High"
    if risk_score >= 40:
        return "Medium"
    return "Low"


# FD002/FD004 models are trained on RUL / 125 (RUL_CAP), so their stored SHAP
# values are fractions of that cap; FD001's are already in cycles. Must match
# the scale engine.js passes to Charts.shapChart().
_SHAP_TO_CYCLES = {"FD001": 1.0, "FD002": 125.0, "FD004": 125.0}


def _pretty_sensor(name: str) -> str:
    return name.replace("sensor_", "Sensor ")


SYSTEM_PROMPT = """You are the assistant inside TurbineSense, a predictive-maintenance \
dashboard for jet-engine Remaining Useful Life (RUL) prediction on the NASA \
C-MAPSS dataset. Answer in plain English for a non-expert.

=== SYSTEM KNOWLEDGE (use for GENERAL questions) ===
Datasets (the user switches between them at the top of the dashboard). Engine \
numbers repeat across datasets, so always consider which dataset you are in:
- FD001: one operating condition, HPC degradation.
- FD002: six operating conditions, HPC degradation.
- FD004: six operating conditions, HPC plus fan degradation.

Prediction: an LSTM neural network reads the last 30 cycles of an engine's \
sensor readings and predicts its Remaining Useful Life (RUL), the number of \
flight cycles left before failure (capped at 125). "True RUL" is the real value \
from the dataset; "Prediction Error" is predicted minus true.

Risk score: risk_score = 100 x (1 - predicted RUL / the highest predicted RUL \
in the same dataset), so it is relative to that dataset's fleet: the engine \
with the lowest predicted RUL has the highest risk. Engines are ranked by \
predicted RUL (rank 1 = shortest predicted life). Risk level is a fixed tier: \
Low below 40, Medium 40 to 69, High 70 or above. The 15 highest-risk engines \
of each dataset show their deep analysis automatically; for any other engine \
the user clicks "Run deep analysis".

SHAP: a method that splits one engine's prediction into per-sensor \
contributions, in cycles of remaining life. In the "Which sensors drove this \
prediction" chart:
- NEGATIVE contribution = the sensor shortens predicted remaining life = \
raises risk = red bar pointing left.
- POSITIVE contribution = the sensor lengthens predicted remaining life = \
lowers risk = green bar pointing right.

"Run deep analysis" runs a four-agent pipeline on one engine:
1. Literature Agent (run once for the whole project, not per engine): searches \
the web for real published papers on C-MAPSS RUL prediction and builds a \
knowledge base of which sensors are reported as predictive and what \
degradation patterns precede failure.
2. Synthesis Agent: compares the engine's top SHAP sensors with that knowledge \
base and gives a verdict: Confirmed (matches documented findings), Contradicted \
(conflicts with them) or Novel (the literature does not cover those sensors, so \
it cannot be checked), with reasoning and the matched papers.
3. Hypothesis Agent: only runs when the verdict is Contradicted or Novel. It \
proposes a plausible physical explanation and concrete next checks an engineer \
could do.
4. Critique Agent: a separate reviewer prompt run on the same Cohere model as the \
Synthesis Agent, so it is not fully independent. It deliberately looks for simpler explanations rooted in the \
model's own setup (sensor noise, a near-constant sensor, extrapolation outside \
the training data). It may only blame "SHAP noise" when a real stability test \
shows the attribution is unstable. It states whether it agrees with the \
pipeline.

=== HOW TO ANSWER ===
There are two kinds of question. Decide which one you are looking at.

1) GENERAL questions about how the system works (what an agent does, what SHAP \
or the risk score means, how the dashboard works, what the datasets are): \
answer from SYSTEM KNOWLEDGE above. Never say "deep analysis hasn't been run" \
for these, and never ask the user to run anything.

2) ENGINE-SPECIFIC questions about the engine currently open: answer ONLY from \
the engine JSON given in the user message. If the user asks about THIS \
engine's Synthesis, Hypothesis or Critique results and the JSON has no \
"agent_analysis" key, say deep analysis has not been run for this engine yet \
and tell them to click "Run deep analysis". For other engine questions (risk \
level, predicted RUL, which sensors matter), answer from the JSON even when \
agent_analysis is absent.

Rules for engine-specific answers:
- State "risk_level" exactly as given. Never contradict it.
- Direction comes ONLY from the two sensor lists: "sensors_increasing_risk" \
(shorten remaining life) and "sensors_decreasing_risk" (lengthen it). Never \
call a sensor in one list a member of the other.
- Write sensors as "Sensor 9", never "sensor_9".
- Describe SHAP in plain English with the cycle figure, for example: "Sensor 9 \
is shortening predicted remaining life by about 1.1 cycles, which raises this \
engine's risk." or "Sensor 20 is lengthening it by about 0.4 cycles, which \
lowers risk."
- Cycle figures always describe a change in REMAINING LIFE, never in risk. \
Write "shortening remaining life by about 18 cycles, which raises risk"; never \
write "raising the risk by 18 cycles".
- Be concise and use the numbers in the data. If the data does not cover the \
question, say so instead of guessing.

If the user only greets you (hi, hello, hey), reply with ONE short friendly \
line plus 2 or 3 example questions. Format answers with short bullet points and \
**bold** the key numbers.

=== HONESTY ===
Never claim more than the data supports and never invent evidence to defend the \
project. When asked about its weaknesses or whether to trust it, admit the real \
limitations directly, then state the real strengths, using only the PROJECT FACTS \
below. If something was not tested or reported, say so plainly.

=== SCOPE AND LENGTH ===
- Stay inside this dashboard: its datasets, the open engine, the model, SHAP, \
the risk score and the agent pipeline. If asked for something unrelated (how to \
build your own project, write code, general knowledge, other products), decline \
in ONE friendly sentence and offer two things you can help with instead. Never \
write a tutorial or a long checklist.
- Be concise by default: at most about 120 words, and at most 5 bullets of one \
short line each. Do not explain every concept at once.
- For a broad question, give a short summary of the 3 or 4 most relevant points, \
then end with one question offering 2 or 3 specific areas to go deeper on.

""" + FACTS_SHEET

OVERVIEW_HINT = (
    "OVERVIEW MODE: the user asked a broad question about the whole project. Answer in one "
    "lead sentence plus at most 4 bullets of under 20 words each, using only the system knowledge. "
    "Do not explain details of any one topic. End with one short question offering 3 specific "
    "areas to explore next (for example: how the risk score works, how to read the SHAP chart, "
    "what the agents do)."
)

# Questions that ask how to build a project from scratch are outside what this
# dashboard assistant is for. Matching is deliberately tight (needs both a
# building verb and a project-like object) to avoid catching questions such as
# "how can I make sense of this risk score?".
_BUILD_VERB = r"(make(?!\s+sense)|build|create|develop|implement|replicate|recreate|clone)"
_BUILD_OBJECT = r"(project|app|application|dashboard|website|something\s+like\s+this|one\s+like\s+this|own)"
_BUILD_GERUND = r"(making|building|creating|developing|implementing|replicating|recreating|cloning|starting)"
_BUILD_RE = re.compile(
    rf"\b(how\s+(can|do|should|would|could|to)\s+(i|we|you)?\s*\w*\s*{_BUILD_VERB}"
    rf"|i\s+(want|need|would\s+like|'d\s+like|wanna)\s+to\s+{_BUILD_VERB}"
    rf"|(help|teach|guide)\s+me\s+(to\s+)?{_BUILD_VERB}"
    rf"|write\s+(me\s+)?(the\s+|some\s+)?code\s+(for|to))"
    rf".*?\b{_BUILD_OBJECT}\b",
    re.IGNORECASE | re.DOTALL,
)
# Other ways of asking the same thing: "before making this project",
# "requirements for building ...", "what do I need to know to build ...".
_BUILD_ALT_RES = [
    re.compile(rf"\b(before|while|when|about|for)\s+{_BUILD_GERUND}\b.*?\b{_BUILD_OBJECT}\b", re.IGNORECASE | re.DOTALL),
    re.compile(
        rf"\b(requirements?|prerequisites?|roadmap|checklist|tutorial|skills|tools|technologies|steps)\s+"
        rf"(to|for|needed\s+(to|for))\s+({_BUILD_GERUND}|{_BUILD_VERB})\b.*?\b{_BUILD_OBJECT}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"\b(what|which)\b.*?\b(need|needs|require|required|take\s+care|consider|prepare|know)\b.*?"
        rf"\b({_BUILD_GERUND}|{_BUILD_VERB})\b.*?\b{_BUILD_OBJECT}\b",
        re.IGNORECASE | re.DOTALL,
    ),
]

_OVERVIEW_RE = re.compile(
    r"\b(understand|explain|overview|summar(y|ise|ize)|walk\s+me\s+through|tell\s+me\s+(about|everything)|what\s+is|what's|"
    r"important\s+points|key\s+points|main\s+points|factors)\b.*\b(this\s+project|the\s+project|this\s+system|the\s+system|"
    r"this\s+dashboard|the\s+dashboard|this\s+app|this\s+site|whole\s+project|entire\s+project)\b"
    r"|\b(what\s+does\s+this\s+(project|dashboard|site|app)\s+do|how\s+does\s+this\s+(project|dashboard|system)\s+work)\b",
    re.IGNORECASE | re.DOTALL,
)


_LIMITS_RE = re.compile(
    r"\b(limitations?|drawbacks?|flaws?|weakness(es)?|downsides?|shortcomings?|disadvantages?|"
    r"biggest\s+(issue|problem|risk)|what.?s\s+wrong|problems?\s+with|not\s+good\s+about)\b",
    re.IGNORECASE,
)
_STRENGTHS_RE = re.compile(
    r"\b(strengths?|advantages?|benefits?|what.?s\s+good|best\s+(part|thing)|why\s+(use|choose)|"
    r"selling\s+points?|what\s+makes\s+this\s+(good|special|useful))\b",
    re.IGNORECASE,
)
_TRUST_RE = re.compile(
    r"\b(why\s+(should|would|do|can|could)\s+(i|we|anyone|someone|people)\s+(\w+\s+)?(trust|use|rely)|"
    r"(can|should|could)\s+(i|we)\s+(really\s+)?(trust|rely|use)|"
    r"how\s+(reliable|accurate|trustworthy|good)|"
    r"is\s+(it|this|the\s+project)\s+(really\s+)?(reliable|accurate|trustworthy|worth)|"
    r"worth\s+using|trust\s+this)\b",
    re.IGNORECASE,
)


# "the biggest flaw" (singular) wants one focused answer; "the limitations"
# (plural) wants the list.
_SUPERLATIVE_RE = re.compile(r"\b(biggest|main|major|primary|worst|top|single|greatest|number\s+one|most\s+(important|serious|significant))\b", re.IGNORECASE)
_SINGULAR_FLAW_RE = re.compile(r"\b(limitation|drawback|flaw|weakness|downside|shortcoming|disadvantage|problem|issue)\b(?!s)", re.IGNORECASE)


def classify_intent(question: str) -> str:
    """'build' (asks how to make their own project), 'flaw' (the single
    biggest weakness), 'balanced' / 'limits' / 'strengths' (questions about
    trust, flaws or merits), 'overview' (broad 'explain the whole project'),
    or 'general'."""
    q = (question or "").strip()
    if _BUILD_RE.search(q) or any(rx.search(q) for rx in _BUILD_ALT_RES):
        return "build"
    limits, strengths, trust = _LIMITS_RE.search(q), _STRENGTHS_RE.search(q), _TRUST_RE.search(q)
    if trust or (limits and strengths):
        return "balanced"
    if limits and _SUPERLATIVE_RE.search(q) and _SINGULAR_FLAW_RE.search(q):
        return "flaw"
    if limits:
        return "limits"
    if strengths:
        return "strengths"
    if _OVERVIEW_RE.search(q):
        return "overview"
    return "general"


class ChatRequest(BaseModel):
    unit: int
    question: str
    subset: str = "FD001"
    analysis_revealed: bool = False


def _pretty_text(obj):
    """Rewrite sensor_N -> Sensor N throughout nested agent text."""
    if isinstance(obj, str):
        return re.sub(r"\bsensor_(\d+)\b", r"Sensor \1", obj, flags=re.IGNORECASE)
    if isinstance(obj, list):
        return [_pretty_text(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _pretty_text(v) for k, v in obj.items()}
    return obj


def build_engine_context(unit: int, subset: str, include_agent_analysis: bool) -> dict:
    """Clean, pre-interpreted context for the LLM — no raw SHAP for it to
    (mis)read itself. Sign convention (ml/explain_shap.py): a NEGATIVE signed
    contribution shortens predicted RUL = raises risk = red bar on the chart."""
    engines = {e["unit"]: e for e in data_service.get_risk_scores(subset)}
    if unit not in engines:
        return None

    engine = engines[unit]
    context = {
        "dataset": subset,
        "unit": unit,
        "risk_level": _risk_level(engine["risk_score"]),
        "risk_score": engine["risk_score"],
        "risk_rank": engine["risk_rank"],
        "predicted_rul_cycles": round(engine["predicted_RUL"], 1),
        "true_rul_cycles": engine["true_RUL"],
        "prediction_error_cycles": round(engine["predicted_RUL"] - engine["true_RUL"], 1),
    }

    shap = data_service.get_shap_explanations(subset).get(unit)
    if shap:
        scale = _SHAP_TO_CYCLES.get(subset, 1.0)
        increasing, decreasing = [], []
        for s in shap.get("top_sensors", []):
            cycles = s["signed_contribution"] * scale
            entry = {"sensor": _pretty_sensor(s["sensor"]), "cycles": abs(round(cycles, 1))}
            (increasing if cycles < 0 else decreasing).append(entry)
        increasing.sort(key=lambda e: e["cycles"], reverse=True)
        decreasing.sort(key=lambda e: e["cycles"], reverse=True)
        context["sensors_increasing_risk"] = increasing  # shorten remaining life
        context["sensors_decreasing_risk"] = decreasing  # lengthen remaining life

    if include_agent_analysis:
        agent_analysis = data_service.get_agent_analysis(unit, subset)
        if agent_analysis:
            # Only the agents' conclusions — the saved state also embeds the raw
            # engine_shap (importance values, RUL-space direction labels), which
            # the clean sensor lists above replace.
            context["agent_analysis"] = _pretty_text(
                {k: agent_analysis.get(k) for k in ("synthesis_result", "hypothesis_result", "critique_result")}
            )

    return context


def _prepare(req: ChatRequest):
    """Decide how to answer: ('text', ready answer) for the fact-based answers,
    or ('llm', messages) for questions the language model should handle. Raises
    the same HTTP errors for both the normal and the streaming endpoint."""
    context = build_engine_context(req.unit, req.subset, req.analysis_revealed)
    if context is None:
        raise HTTPException(404, f"Engine {req.unit} not found")

    # These answers are written from the results files, not generated, so they
    # stay accurate and consistent and need no API key.
    intent = classify_intent(req.question)
    if intent == "build":
        return "text", build_lessons_answer()
    if intent == "flaw":
        return "text", biggest_flaw_answer(req.subset)
    if intent == "balanced":
        return "text", balanced_answer(req.subset)
    if intent == "limits":
        return "text", limitations_answer(req.subset)
    if intent == "strengths":
        return "text", strengths_answer(req.subset)

    if not OPENAI_API_KEY:
        raise HTTPException(503, "OPENAI_API_KEY is not configured on the server.")

    hint = f"\n\n{OVERVIEW_HINT}" if intent == "overview" else ""
    return "llm", [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Engine data:\n{json.dumps(context, indent=2)}\n\nQuestion: {req.question}{hint}"},
    ]


@router.post("")
def chat(req: ChatRequest):
    kind, payload = _prepare(req)
    if kind == "text":
        return {"answer": payload}
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(model="gpt-4o-mini", max_tokens=450, messages=payload)
    return {"answer": response.choices[0].message.content}


_WORD_RE = re.compile(r"\S+\s*")
_FACT_STREAM_DELAY = 0.004  # small, so the stream is still progressive; the page paces the typing


@router.post("/stream")
def chat_stream(req: ChatRequest):
    """Same answers as POST /api/chat, delivered as a plain-text stream so the
    dashboard can show them as they are produced."""
    kind, payload = _prepare(req)  # errors surface as normal HTTP statuses

    def generate():
        if kind == "text":
            for piece in _WORD_RE.findall(payload):
                yield piece
                time.sleep(_FACT_STREAM_DELAY)
            return
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            stream = client.chat.completions.create(model="gpt-4o-mini", max_tokens=450, messages=payload, stream=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception:
            yield "\n\nSorry, I couldn't finish that answer. Please try again."

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
