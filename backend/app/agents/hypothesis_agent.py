"""Hypothesis Agent (OpenAI GPT-4o-mini).

Only invoked when the Synthesis Agent returns Contradicted or Novel. Proposes
a plausible physical/statistical explanation for the unexpected result and
suggests concrete next checks (not another literature search — it reasons
from the specific engine's data plus what the knowledge base already says).
"""
import json

from openai import OpenAI

from app.agents.throttle import llm_retry
from app.config import OPENAI_API_KEY

SYSTEM_PROMPT = """You are a turbofan-engine diagnostics engineer. A machine \
learning model's explanation for one engine's remaining-useful-life prediction \
either contradicts published research or covers a case the literature hasn't \
documented. Propose your single most plausible explanation, grounded in known \
engine physics and the specific sensors involved, and list concrete next steps \
an engineer could take to verify it (e.g. specific sensor cross-checks, other \
engines to compare against, or data-quality checks).

Respond with ONLY JSON:
{
  "hypothesis": "2-4 sentences",
  "physical_reasoning": "why this is plausible given the sensors/physics involved",
  "next_checks": ["specific, actionable check", "..."],
  "confidence": "high|medium|low"
}"""


@llm_retry
def investigate(engine_shap: dict, synthesis_result: dict, model: str = "gpt-4o-mini") -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set — cannot run Hypothesis Agent.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    user_content = json.dumps(
        {"engine_shap": engine_shap, "synthesis_result": synthesis_result}, indent=2
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)
