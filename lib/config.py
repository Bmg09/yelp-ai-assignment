import os
from dotenv import load_dotenv

load_dotenv()

GATEWAY_URL = "https://ai-gateway.vercel.sh/v1"

CLASSIFIER_CANDIDATES = [
    "openai/gpt-5-nano",
    "deepseek/deepseek-v3.2",
    "google/gemini-2.0-flash",
]

ACTIVE_CLASSIFIER = "deepseek/deepseek-v3.2"

JUDGE = "anthropic/claude-haiku-4.5"

CACHE_DIR = ".pycache"
CACHE_ENABLED = os.environ.get("AI_CACHE", "1") != "0"

CONCURRENCY = 30
RATE_LIMIT_RPS = 20


def gateway_key() -> str:
    k = os.environ.get("AI_GATEWAY_API_KEY")
    if not k:
        raise RuntimeError("AI_GATEWAY_API_KEY not set in env/.env")
    return k
