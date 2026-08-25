import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]

for candidate in (APP_DIR / ".env", APP_DIR.parent / ".env", REPO_ROOT / ".env"):
    load_dotenv(candidate, override=False)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
RESUME_PATH = APP_DIR / "resource" / "resume.json"
VECTORSTORE_DIR = APP_DIR / "chroma_db"
COLLECTION_NAME = "projects"


def require_groq_key():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env in the repo root "
            "and add your key from https://console.groq.com/keys"
        )
    return GROQ_API_KEY
