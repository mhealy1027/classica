"""Shared configuration for Classica."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
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

# Pricing per million tokens
_PRICING = {
    CLAUDE_MODEL: {"input": 0.80, "output": 4.00},
    CLAUDE_SONNET_MODEL: {"input": 3.00, "output": 15.00},
}


class CostTracker:
    """Track API token usage and estimated cost across a session."""

    def __init__(self):
        self._usage: dict[str, dict[str, int]] = {}  # model -> {input, output}

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        if model not in self._usage:
            self._usage[model] = {"input": 0, "output": 0}
        self._usage[model]["input"] += input_tokens
        self._usage[model]["output"] += output_tokens

    def total_cost(self) -> float:
        cost = 0.0
        for model, counts in self._usage.items():
            pricing = _PRICING.get(model, _PRICING[CLAUDE_MODEL])
            cost += counts["input"] * pricing["input"] / 1_000_000
            cost += counts["output"] * pricing["output"] / 1_000_000
        return cost

    def summary(self) -> str:
        total_in = sum(c["input"] for c in self._usage.values())
        total_out = sum(c["output"] for c in self._usage.values())
        return (
            f"Estimated API cost: ${self.total_cost():.2f} "
            f"(input: {total_in / 1000:.1f}k tokens, output: {total_out / 1000:.1f}k tokens)"
        )

    def print_summary(self) -> None:
        if self._usage:
            print(f"\n{self.summary()}")


# Global tracker for the current session
cost_tracker = CostTracker()
