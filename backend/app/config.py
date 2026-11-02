import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "models" / "results"
KNOWLEDGE_BASE_PATH = RESULTS_DIR / "literature_knowledge_base.json"
AGENT_ANALYSIS_DIR = RESULTS_DIR / "agent_analysis"
AGENT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

CRITIQUE_PROVIDER = os.getenv("CRITIQUE_PROVIDER", "cohere")  # "cohere" during build, "anthropic" after deploy
