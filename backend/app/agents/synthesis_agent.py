"""Synthesis Agent (Cohere command-r-plus during build, swappable to Claude).

Runs once per engine selected for deep analysis. Compares that engine's SHAP
attribution against the Literature Agent's knowledge base and classifies the
result as Confirmed / Contradicted / Novel.
"""
import json

import cohere

from app.agents.throttle import cohere_throttle, llm_retry
from app.config import COHERE_API_KEY

SYSTEM_PROMPT = """You are a mechanical reliability engineer cross-checking a \
machine-learning model's explanation against published research.

You will be given:
1. SHAP attribution for one engine's RUL prediction (which sensors drove it, and \
   whether each pushed the prediction toward higher or lower risk).
2. A knowledge base of findings from real published papers on C-MAPSS RUL prediction.

Decide whether the model's explanation is:
- "Confirmed": the top driving sensor(s) and their direction match documented \
  findings in the knowledge base.
- "Contradicted": the top driving sensor(s) or direction conflict with documented \
  findings.
- "Novel": the knowledge base has no documented finding covering the sensor(s) \
  driving this prediction, so it can't be checked either way.

Respond with ONLY JSON:
{
  "verdict": "Confirmed|Contradicted|Novel",
  "confidence": "high|medium|low",
  "reasoning": "2-4 sentences citing which knowledge-base entries you compared against, or noting the absence of coverage",
  "matched_sources": ["title of matched paper", ...]
}"""


@llm_retry
def synthesize(engine_shap: dict, knowledge_base: dict, model: str = "command-a-03-2025") -> dict:
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is not set — cannot run Synthesis Agent.")

    client = cohere.ClientV2(api_key=COHERE_API_KEY)

    user_content = json.dumps(
        {
            "engine_shap": engine_shap,
            "knowledge_base_sensor_findings": knowledge_base.get("sensor_findings", []),
            "knowledge_base_general_patterns": knowledge_base.get("general_degradation_patterns", []),
        },
        indent=2,
    )

    cohere_throttle()
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    text = response.message.content[0].text
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end])
