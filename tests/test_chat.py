import re

import pytest
from fastapi.testclient import TestClient

from app.config import OPENAI_API_KEY
from app.main import app
from app.routers.chat import build_engine_context, classify_intent

client = TestClient(app)

# These call the real OpenAI API (the key is configured in .env). They are
# skipped, not mocked, when the key is missing.
requires_openai_key = pytest.mark.skipif(
    not OPENAI_API_KEY, reason="OPENAI_API_KEY not configured — live chat call needed"
)

def _chat(unit, question, subset="FD001", revealed=False):
    res = client.post(
        "/api/chat",
        json={"unit": unit, "question": question, "subset": subset, "analysis_revealed": revealed},
    )
    assert res.status_code == 200, res.text
    return res.json()["answer"]


@pytest.mark.parametrize(
    "question,intent",
    [
        ("how can i make this project if i want to make ? what exact pints i need to take care of ?", "build"),
        ("I want to build a dashboard like this", "build"),
        ("help me build this project step by step", "build"),
        ("ok may i know what factors i need to understand or important points i need to understand if i need to understand this project ?", "overview"),
        ("explain this project", "overview"),
        ("how does this project work", "overview"),
        ("why is this engine at this risk level?", "general"),
        ("how can i make sense of this risk score", "general"),
        ("what does the synthesis agent do?", "general"),
        ("hi", "general"),
    ],
)
def test_question_intent_classification(question, intent):
    assert classify_intent(question) == intent


@pytest.mark.parametrize(
    "question,intent",
    [
        ("what is the biggest limitation of this project tell me that ?", "flaw"),
        ("if this project has such huge limitation then why would i trust this project and check any answers ?", "balanced"),
        ("why would i use the project if we have the major drawbacks and flaws", "balanced"),
        ("what are the strengths of this project", "strengths"),
        ("how accurate is this model", "balanced"),
        ("how can i make sense of this risk score", "general"),
        ("which sensors matter most?", "general"),
    ],
)
def test_trust_question_intents(question, intent):
    assert classify_intent(question) == intent


@pytest.mark.parametrize("subset", ["FD001", "FD002", "FD004"])
def test_balanced_answer_is_honest_and_two_sided(subset):
    from app.project_facts import balanced_answer

    answer = balanced_answer(subset)
    assert "Why it's worth using" in answer and "What to keep in mind" in answer
    assert "verify the important cases" in answer  # guidance, not a sales pitch
    assert "state of the art" in answer.lower()  # admits it is not SOTA


def test_weak_stability_is_never_presented_as_a_strength():
    """FD004's top sensor is identical across re-runs for only about half the
    engines, so that figure must appear as a limitation, not a strength."""
    from app.project_facts import limitations_answer, strengths_answer

    assert "Less stable explanations" in limitations_answer("FD004")
    assert "identical across 5 re-runs" not in strengths_answer("FD004")


def test_zero_confirmed_cause_only_claimed_for_fd002():
    from app.project_facts import limitations_answer

    assert "sensor 11" in limitations_answer("FD002")
    assert "sensor 11" not in limitations_answer("FD004")


@pytest.mark.parametrize(
    "question,intent",
    [
        ("what is the biggest flaw of this project ?", "flaw"),
        ("what is the biggest limitation of this project tell me that ?", "flaw"),
        ("what is the main drawback", "flaw"),
        ("what is the limitation of this project ?", "limits"),
        ("what are the main limitations", "limits"),
        ("what are the biggest flaws", "limits"),
        ("if this project has limitation and flaw why would i use your project ?", "balanced"),
    ],
)
def test_flaw_vs_limitations_vs_balanced(question, intent):
    assert classify_intent(question) == intent


def test_biggest_flaw_answer_differs_from_the_full_list():
    from app.project_facts import biggest_flaw_answer, limitations_answer

    single, full = biggest_flaw_answer("FD001"), limitations_answer("FD001")
    assert single != full
    assert "my judgement" in single  # ranking flaws is opinion, and it says so
    assert len(single.split()) < len(full.split()) + 40


def _stream(question, subset="FD001", unit=34):
    chunks = []
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"unit": unit, "question": question, "subset": subset, "analysis_revealed": False},
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/plain")
        for piece in res.iter_text():
            if piece:
                chunks.append(piece)
    return chunks


def test_stream_delivers_the_same_answer_as_the_normal_endpoint():
    # The test client buffers the body, so progressive delivery is checked
    # against the running server instead; here the content must match exactly.
    question = "what is the biggest flaw of this project ?"
    assert "".join(_stream(question)) == _chat(34, question)


def test_fact_answers_are_split_into_many_lossless_pieces():
    from app.routers.chat import _WORD_RE
    from app.project_facts import limitations_answer

    text = limitations_answer("FD002")
    pieces = _WORD_RE.findall(text)
    assert len(pieces) > 50
    assert "".join(pieces) == text  # nothing dropped, newlines and bullets intact


def test_stream_reports_unknown_engine_as_a_real_error():
    res = client.post(
        "/api/chat/stream",
        json={"unit": 99999, "question": "hi", "subset": "FD001", "analysis_revealed": False},
    )
    assert res.status_code == 404


@pytest.mark.parametrize(
    "question",
    [
        "how can i make this project if i want to make ? what exact pints i need to take care of ?",
        "what important points and factors i need to take care of before making this project ?",
        "what do i need to know to build something like this",
        "requirements for building this project",
        "what are the prerequisites to create this dashboard",
        "i want to build a dashboard like this",
        "tips before building a project like this",
    ],
)
def test_build_questions_are_recognised(question):
    assert classify_intent(question) == "build"


@pytest.mark.parametrize(
    "question",
    [
        "how does this project make predictions",
        "what does the model need to make a prediction",
        "what factors should i consider when reading the risk score",
        "what points do i need to take care of when investigating this engine",
        "how can i make sense of this risk score",
        "what is the project about",
        "which sensors do i need to look at first",
    ],
)
def test_normal_questions_are_not_mistaken_for_build_requests(question):
    assert classify_intent(question) != "build"


def test_build_answer_shares_real_lessons_not_a_generic_tutorial():
    from app.project_facts import build_lessons_answer

    answer = build_lessons_answer()
    assert "can't give you a full build tutorial" in answer
    for lesson in ("Sensor choice", "RUL cap", "LLM research", "Explanation stability", "Evaluation rules"):
        assert lesson in answer
    assert len(answer.split()) < 230


def test_engine_27_context_signs_match_chart():
    """Engine 27 (FD001): Sensor 9 has signed_contribution +6.29 (green bar,
    lengthens remaining life, lowers risk); Sensor 20 has -1.14 (red bar,
    shortens remaining life, raises risk)."""
    ctx = build_engine_context(27, "FD001", include_agent_analysis=False)
    assert ctx["risk_level"] == "Low"  # risk_score ~21.5
    raising = {e["sensor"] for e in ctx["sensors_increasing_risk"]}
    lowering = {e["sensor"] for e in ctx["sensors_decreasing_risk"]}
    assert "Sensor 20" in raising
    assert "Sensor 9" in lowering
    assert "shap" not in ctx
    for entry in ctx["sensors_increasing_risk"] + ctx["sensors_decreasing_risk"]:
        assert set(entry) == {"sensor", "cycles"}  # no raw importance


def test_engine_34_high_risk_all_sensors_raise_risk():
    ctx = build_engine_context(34, "FD001", include_agent_analysis=False)
    assert ctx["risk_level"] == "High"
    assert ctx["sensors_increasing_risk"] and not ctx["sensors_decreasing_risk"]


def test_multicond_shap_values_converted_to_cycles():
    """FD002 stores SHAP as fractions of the 125-cycle cap; context must be in cycles."""
    ctx = build_engine_context(116, "FD002", include_agent_analysis=False)
    biggest = max(e["cycles"] for e in ctx["sensors_increasing_risk"] + ctx["sensors_decreasing_risk"])
    assert biggest > 5  # raw stored value is ~0.2


def _first_without_analysis(subset):
    from app.services import data_service

    for e in data_service.get_risk_scores(subset):
        if not e.get("deep_analysis") and data_service.get_agent_analysis(e["unit"], subset) is None:
            return e["unit"]
    return None


@requires_openai_key
@pytest.mark.parametrize("subset,unit", [("FD001", 34), ("FD002", 116), ("FD004", 135)])
def test_chat_returns_200_for_each_subset(subset, unit):
    answer = _chat(unit, "Why is this engine at this risk level?", subset)
    assert answer.strip()


@requires_openai_key
@pytest.mark.parametrize("subset", ["FD001", "FD002", "FD004"])
def test_general_synthesis_question_never_says_not_run(subset):
    unit = _first_without_analysis(subset) or 90
    answer = _chat(unit, "what does the synthesis agent do?", subset)
    assert "hasn't been run" not in answer.lower()
    assert "has not been run" not in answer.lower()
    assert "synthesis" in answer.lower()


@requires_openai_key
def test_engine_specific_critique_question_mentions_running_deep_analysis():
    # Picked dynamically: on-demand "Run deep analysis" clicks add analysis
    # files over time, so no fixed engine stays un-analyzed.
    unit = _first_without_analysis("FD001")
    assert unit is not None
    answer = _chat(unit, "what did the critique agent say for this engine?", "FD001").lower()
    assert re.search(r"run deep analysis|(has|hasn't|has not)( yet)? been run|not (yet )?been run", answer), answer


@requires_openai_key
@pytest.mark.parametrize("subset,unit", [("FD001", 90), ("FD002", 116), ("FD004", 135)])
def test_answers_contain_no_raw_sensor_names(subset, unit):
    for q in ("Which sensors matter most for this engine?", "Why is this engine at this risk level?"):
        assert not re.search(r"sensor_\d", _chat(unit, q, subset))


@requires_openai_key
def test_greeting_is_short():
    answer = _chat(90, "hi")
    assert len(answer.split()) < 60, answer
