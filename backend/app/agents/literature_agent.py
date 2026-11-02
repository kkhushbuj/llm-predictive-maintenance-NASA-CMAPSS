"""Literature Agent (OpenAI GPT-4o-mini, web-search enabled).

Runs ONCE for the whole project, not per-engine. It researches real published
work on jet-engine / C-MAPSS RUL prediction and produces a structured
knowledge base: which sensors are reported as most predictive, and what
degradation patterns typically precede failure. The Synthesis Agent later
diffs each engine's SHAP result against this knowledge base.
"""
import json

from openai import OpenAI

from app.config import KNOWLEDGE_BASE_PATH, OPENAI_API_KEY

CANONICAL_SENSOR_MAP = """The canonical NASA C-MAPSS sensor_1..sensor_21 index mapping \
(Saxena & Goebel 2008, "Damage Propagation Modeling") is fixed and MUST be used to map any \
physical measurement name (T2, P30, Nf, phi, etc.) found in a paper to its sensor_N index. \
Do not guess or infer a different mapping from a paper's own figure/table numbering — always \
translate to this canonical index:
  sensor_1: T2 (total temperature at fan inlet)
  sensor_2: T24 (total temperature at LPC outlet)
  sensor_3: T30 (total temperature at HPC outlet)
  sensor_4: T50 (total temperature at LPT outlet)
  sensor_5: P2 (pressure at fan inlet)
  sensor_6: P15 (total pressure in bypass-duct)
  sensor_7: P30 (total pressure at HPC outlet)
  sensor_8: Nf (physical fan speed)
  sensor_9: Nc (physical core speed)
  sensor_10: epr (engine pressure ratio, P50/P2)
  sensor_11: Ps30 (static pressure at HPC outlet)
  sensor_12: phi (ratio of fuel flow to Ps30)
  sensor_13: NRf (corrected fan speed)
  sensor_14: NRc (corrected core speed)
  sensor_15: BPR (bypass ratio)
  sensor_16: farB (burner fuel-air ratio)
  sensor_17: htBleed (bleed enthalpy)
  sensor_18: Nf_dmd (demanded fan speed)
  sensor_19: PCNfR_dmd (demanded corrected fan speed)
  sensor_20: W31 (HPT coolant bleed)
  sensor_21: W32 (LPT coolant bleed)
Note: in FD001 (single operating condition), sensor_1, 5, 6, 10, 16, 18, 19 are physically \
near-constant (demanded/inlet-condition sensors unaffected by degradation at fixed operating \
condition) and rarely reported as predictive — be suspicious of any source claiming high \
importance for these in FD001 specifically; double-check the mapping in that case."""

SYSTEM_PROMPT = """You are a research analyst specializing in aircraft engine \
prognostics and health management (PHM). You have web search available. \
Research real, published papers on remaining-useful-life (RUL) prediction for \
turbofan engines, especially work using the NASA C-MAPSS dataset (FD001-FD004) \
and related PHM08 challenge literature. Search the web, read the results, and \
extract concrete, citable findings.

""" + CANONICAL_SENSOR_MAP + """

Return ONLY valid JSON matching this schema:
{
  "sensor_findings": [
    {
      "sensor": "sensor_11",                     // matches C-MAPSS sensor_1..sensor_21 naming
      "physical_meaning": "static pressure at HPC outlet (Ps30)" ,
      "reported_importance": "high|medium|low",
      "typical_pre_failure_pattern": "monotonic increase over final ~30 cycles",
      "sources": [{"title": "...", "authors": "...", "year": 2018, "venue": "...", "url": "..."}]
    }
  ],
  "general_degradation_patterns": [
    {"pattern": "...", "sources": [{"title": "...", "year": 2017, "url": "..."}]}
  ],
  "benchmark_results": [
    {"method": "LSTM", "dataset": "FD001", "rmse": 16.14, "score": 338, "source": {"title": "...", "authors": "...", "year": 2017, "url": "..."}}
  ]
}

Only include facts you can attribute to a real source found via search. If you \
cannot verify a number, omit it rather than guessing."""


def _parse_json_response(text: str) -> dict:
    """The model wraps JSON in ```json fences and occasionally emits a stray
    trailing comma or comment; strip fences and retry a couple of common
    fixups before giving up."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned
    start, end = cleaned.find("{"), cleaned.rfind("}") + 1
    candidate = cleaned[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        import re

        no_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", candidate)
        try:
            return json.loads(no_trailing_commas)
        except json.JSONDecodeError as e:
            debug_path = KNOWLEDGE_BASE_PATH.parent / "_literature_agent_raw_failure.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Wrote raw failing response to {debug_path} for inspection")
            raise


USER_PROMPT = (
    "Research sensor importance and degradation patterns for C-MAPSS "
    "turbofan RUL prediction (focus on FD001: single operating condition, "
    "HPC degradation fault mode). Do at least 4-5 separate searches covering: "
    "(1) which individual C-MAPSS sensors (e.g. T24, T30, T50, P30, Nf, Nc, "
    "Ps30, phi, NRf, NRc, BPR, htBleed, W31, W32 — map these to sensor_2 "
    "through sensor_21) are most predictive of RUL in published feature-"
    "importance/SHAP studies; (2) papers specifically using SHAP or other "
    "explainability methods on C-MAPSS; (3) typical degradation trajectory "
    "shape for HPC degradation (FD001/FD003) vs fan degradation; (4) published "
    "LSTM/CNN/Transformer benchmark RMSE and NASA-score results on FD001. "
    "Aim for at least 6-8 distinct sensor_findings entries if the literature "
    "supports it — do not stop after one source, keep searching for more "
    "sensor-specific findings before writing the final JSON. Every entry must "
    "cite a real, findable source. Return the JSON knowledge base described in "
    "the system prompt."
)


MULTICOND_PROMPTS = {
    "FD002": (
        "C-MAPSS FD002: SIX operating conditions, ONE fault mode (HPC degradation), "
        "260 train / 259 test engines."
    ),
    "FD004": (
        "C-MAPSS FD004: SIX operating conditions, TWO fault modes (HPC degradation "
        "AND fan degradation), 249 train / 248 test engines."
    ),
}


def _multicond_user_prompt(subset: str) -> str:
    return (
        f"Research RUL prediction specifically for {MULTICOND_PROMPTS[subset]} "
        "Do at least 5 separate searches covering: (1) published RMSE and "
        f"PHM08/NASA scoring-function results reported on {subset} specifically "
        "(LSTM, CNN, Transformer and other deep models) — include only numbers a "
        f"source explicitly reports for {subset}; never substitute FD001 or FD003 "
        "numbers; (2) which C-MAPSS sensors are reported as most predictive under "
        "multiple operating conditions, including after operating-condition "
        "normalization; (3) how operating-condition normalization or regime "
        "clustering is done in the literature for this subset"
        + (
            "; (4) which sensors respond to FAN degradation versus HPC degradation, "
            "and how their trajectories differ"
            if subset == "FD004"
            else ""
        )
        + ". Aim for at least 8 benchmark_results rows and 6-8 sensor_findings "
        f"rows. Every benchmark row's \"dataset\" field must be \"{subset}\". Every "
        "entry must cite a real, findable source. Return the JSON knowledge base "
        "described in the system prompt."
    )


def build_knowledge_base(
    model: str = "gpt-4o-mini", max_attempts: int = 3, subset: str = "FD001", out_path=None
) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set — cannot run Literature Agent.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_prompt = USER_PROMPT if subset == "FD001" else _multicond_user_prompt(subset)
    out_path = out_path or KNOWLEDGE_BASE_PATH

    last_error = None
    for attempt in range(1, max_attempts + 1):
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            max_output_tokens=8000,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        try:
            knowledge_base = _parse_json_response(response.output_text)
            break
        except (json.JSONDecodeError, IndexError) as e:
            last_error = e
            print(f"attempt {attempt}/{max_attempts} failed to parse JSON ({e}); retrying...")
    else:
        raise RuntimeError(f"Literature Agent failed after {max_attempts} attempts") from last_error

    if subset != "FD001":
        # The prompt forbids other subsets' numbers; enforce it rather than trust it.
        rows = knowledge_base.get("benchmark_results", [])
        knowledge_base["benchmark_results"] = [r for r in rows if str(r.get("dataset", "")).upper() == subset]
        knowledge_base["benchmark_rows_dropped_wrong_subset"] = len(rows) - len(knowledge_base["benchmark_results"])

    with open(out_path, "w") as f:
        json.dump(knowledge_base, f, indent=2)

    print(f"Saved literature knowledge base to {out_path}")
    return knowledge_base


def load_knowledge_base(path=None) -> dict:
    path = path or KNOWLEDGE_BASE_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run build_knowledge_base() first.")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    build_knowledge_base()
