"""Critique Agent — independent second opinion, deliberately a different
model/lab than the rest of the pipeline (Cohere during build, Claude once
deployed) so it isn't just rubber-stamping the Synthesis/Hypothesis agents.

Only run on a small sample of engines (1-15). Its job is to look for reasons
the model's OWN setup (data leakage, degenerate SHAP attribution, a sensor
that's constant/noisy for this engine, an over/under-confident prediction)
might explain the result, rather than accepting the pipeline's story.
"""
import json

import cohere

from app.agents.throttle import cohere_throttle, llm_retry
from app.config import ANTHROPIC_API_KEY, COHERE_API_KEY, CRITIQUE_PROVIDER

# Must match ml/shap_stability.py's AGREEMENT_THRESHOLD / CV_THRESHOLD — the
# "stable" flag on incoming shap_stability data is precomputed with these same
# numbers, so the rule below is enforceable against real evidence, not vibes.
STABILITY_AGREEMENT_THRESHOLD = 0.8
STABILITY_CV_THRESHOLD = 0.35

SYSTEM_PROMPT = f"""You are an independent reviewer auditing an ML pipeline's \
explanation of a jet engine RUL prediction. You were NOT involved in producing \
the prediction, the SHAP explanation, the literature knowledge base, the \
synthesis verdict, or the hypothesis. Your job is adversarial: look for reasons \
the model's OWN setup could explain the result, before accepting the pipeline's \
narrative. Consider: an over-narrow sequence window, a sensor with near-zero \
variance for this engine, sensor noise, or the model simply extrapolating \
outside its training distribution. Do not just agree with the \
synthesis/hypothesis — actively look for a simpler or alternative explanation.

HARD RULE ON "SHAP NOISE / SHAP ARTIFACT / BACKGROUND-SAMPLING ARTIFACT" CLAIMS:
You may only use that reasoning — in "critique" or as "alternative_explanation" \
— when "shap_stability" is present in the input AND its "stable" field is \
false. That field is precomputed from real reruns of SHAP with different \
random background samples: "stable": false means the top-attributed sensor \
actually changed across reruns (top_sensor_agreement_rate < {STABILITY_AGREEMENT_THRESHOLD}) \
or its importance was volatile (mode_sensor_importance_cv > {STABILITY_CV_THRESHOLD}) — \
real instability, not a guess.
- "shap_stability" present and "stable": true → the attribution is proven \
  consistent across independent reruns. A SHAP-noise/artifact claim would be \
  false given the evidence you were handed, so it is FORBIDDEN here. You must \
  either agree with the pipeline's verdict, or disagree for a different, \
  specific reason (e.g. a documented near-zero-variance sensor, a physically \
  implausible sensor combination, extrapolation you can point to in the data) \
  — never the noise excuse.
- "shap_stability" present and "stable": false → real instability was \
  measured; citing SHAP noise/artifact here is a legitimate, evidence-backed \
  call. Reference the actual agreement rate and/or CV in your critique.
- "shap_stability" is null/absent → you have no stability evidence at all. Say \
  so explicitly in "critique" rather than asserting a SHAP-artifact explanation \
  you cannot check; do not put SHAP noise/artifact language in \
  "alternative_explanation" in this case either.

Respond with ONLY JSON:
{{
  "agrees_with_pipeline": true/false,
  "critique": "2-4 sentences of your independent assessment",
  "alternative_explanation": "a specific model-setup explanation to consider, or null if none found",
  "confidence": "high|medium|low"
}}"""


def _build_payload(
    engine_shap: dict,
    synthesis_result: dict,
    hypothesis_result: dict | None,
    shap_stability: dict | None = None,
) -> str:
    return json.dumps(
        {
            "engine_shap": engine_shap,
            "synthesis_result": synthesis_result,
            "hypothesis_result": hypothesis_result,
            "shap_stability": shap_stability,
            "shap_stability_thresholds": {
                "agreement_threshold": STABILITY_AGREEMENT_THRESHOLD,
                "cv_threshold": STABILITY_CV_THRESHOLD,
                "note": "shap_stability.stable was computed with these thresholds already",
            },
        },
        indent=2,
    )


@llm_retry
def _critique_cohere(payload: str, model: str = "command-a-03-2025") -> dict:
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is not set — cannot run Critique Agent (cohere).")
    client = cohere.ClientV2(api_key=COHERE_API_KEY)
    cohere_throttle()
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
    )
    text = response.message.content[0].text
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end])


@llm_retry
def _critique_anthropic(payload: str, model: str = "claude-sonnet-5") -> dict:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — cannot run Critique Agent (anthropic).")
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": payload}],
    )
    text = response.content[0].text
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end])


def critique(
    engine_shap: dict,
    synthesis_result: dict,
    hypothesis_result: dict | None = None,
    shap_stability: dict | None = None,
) -> dict:
    payload = _build_payload(engine_shap, synthesis_result, hypothesis_result, shap_stability)
    if CRITIQUE_PROVIDER == "anthropic":
        return _critique_anthropic(payload)
    return _critique_cohere(payload)
