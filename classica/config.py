"""Shared configuration for Classica."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXTS_DIR = PROJECT_ROOT / "texts"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"

TEXTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

PERSEUS_BASE_URL = (
    "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/"
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_SONNET_MODEL = "claude-sonnet-4-20250514"
