"""Prompt templates with few-shot examples for Claude API extraction."""

FEW_SHOT_EXAMPLES = [
    {
        "label": "Simple extraction (Book 1, Ch 45) — Athens sends ships to Corcyra",
        "output": [
            {
                "book": 1,
                "chapter": 45,
                "year_bce": 433,
                "actor": "Athens",
                "purpose": "Corcyra expedition",
                "amount": 10,
                "unit": "triremes",
                "direction": "expenditure",
                "confidence": "high",
                "greek_reference": "",
                "notes": "Initial Athenian intervention at Corcyra.",
            }
        ],
    },
    {
        "label": "Dense extraction (Book 2, Ch 13) — Pericles' accounting of Athenian resources",
        "note": "This passage yielded 9 total extractions covering all financial and military figures listed by Pericles. Extract every distinct data point, not just the first few.",
        "output": [
            {
                "book": 2,
                "chapter": 13,
                "year_bce": 431,
                "actor": "Athens",
                "purpose": "Annual tribute from allies",
                "amount": 600,
                "unit": "talents",
                "direction": "revenue",
                "confidence": "high",
                "greek_reference": "phoros",
                "notes": "Disputed. ATL suggests ~388 talents.",
            },
            {
                "book": 2,
                "chapter": 13,
                "year_bce": 431,
                "actor": "Athens",
                "purpose": "Field hoplites available",
                "amount": 13000,
                "unit": "hoplites",
                "direction": "asset",
                "confidence": "high",
                "greek_reference": "",
                "notes": "",
            },
        ],
    },
    {
        "label": "Wage data (Book 3, Ch 17) — Fleet costs",
        "note": "This passage yielded 3 total extractions including ship counts and per-day wage rates for hoplites and sailors. Extract all distinct figures.",
        "output": [
            {
                "book": 3,
                "chapter": 17,
                "year_bce": 428,
                "actor": "Athens",
                "purpose": "Ships on guard duty",
                "amount": 250,
                "unit": "ships",
                "direction": "asset",
                "confidence": "high",
                "greek_reference": "",
                "notes": "",
            },
        ],
    },
]


def build_system_prompt(schema_text: str) -> str:
    """Build the system prompt for extraction.

    Args:
        schema_text: Rendered schema string from render_schema_for_prompt().

    Returns:
        Complete system prompt string.
    """
    import json

    examples_text = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_text += f"\n### {ex['label']}\n"
        if ex.get("note"):
            examples_text += f"*{ex['note']}*\nOutput (representative subset):\n```json\n"
        else:
            examples_text += "Output:\n```json\n"
        examples_text += json.dumps(ex["output"], indent=2)
        examples_text += "\n```\n"

    return f"""You are a classical scholar and data analyst specializing in ancient Greek texts. Your task is to extract structured financial and military data from passages of ancient texts.

{schema_text}

## Rules
1. Return ONLY a valid JSON array of extraction objects. No commentary, no markdown outside the JSON.
2. Do NOT hallucinate data. Only extract information that is explicitly stated or very clearly implied in the passage.
3. Assign a confidence rating to each extraction:
   - "high": exact number given directly in the text
   - "medium": number implied or requires interpretation
   - "low": inferred from context or ambiguous
4. Flag potential duplicates in the notes field if the same data point appears to be restated.
5. If no extractable financial or military data is found in the passage, return an empty array: []
6. Include the Greek term(s) that anchor each data point in greek_reference when applicable.
7. For amounts, use the number as stated in the text. Do not convert units.

## Few-Shot Examples
{examples_text}

Now extract all financial and military data from the passage provided by the user."""


def build_user_message(passage: dict) -> str:
    """Build the user message containing the passage text.

    Args:
        passage: Dict with 'book', 'chapter', 'greek_text', 'english_text'.

    Returns:
        Formatted user message string.
    """
    return f"""## Passage: Book {passage['book']}, Chapter {passage['chapter']}

### English Translation:
{passage['english_text']}

### Greek Text:
{passage['greek_text']}

Extract all financial and military data from this passage as a JSON array."""
