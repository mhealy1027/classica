"""Tool: extract structured data from a passage using Claude API."""

import json
import uuid
from typing import Any

import anthropic

from classica.config import CLAUDE_MODEL
from classica.prompts.extraction import build_system_prompt, build_user_message
from classica.schemas.base import render_schema_for_prompt


def _validate_extraction(extraction: dict, schema: dict) -> dict:
    """Validate an extraction dict against the schema field definitions.

    Coerces types where possible and checks enum membership.
    Returns the cleaned extraction.
    """
    field_map = {f["name"]: f for f in schema["fields"]}

    cleaned = {}
    for field in schema["fields"]:
        name = field["name"]
        value = extraction.get(name)

        if value is None or value == "":
            cleaned[name] = None if field["type"] == "number" else ""
            continue

        # Type coercion
        if field["type"] == "integer":
            try:
                cleaned[name] = int(value)
            except (ValueError, TypeError):
                cleaned[name] = value
        elif field["type"] == "number":
            try:
                cleaned[name] = float(value)
            except (ValueError, TypeError):
                cleaned[name] = value
        else:
            cleaned[name] = str(value)

        # Enum validation
        if "enum" in field and cleaned[name]:
            if str(cleaned[name]) not in [str(e) for e in field["enum"]]:
                cleaned[name] = str(cleaned[name])

    return cleaned


def extract_passage(
    passage: dict,
    schema: dict,
    few_shot_examples: Any = None,
) -> list[dict]:
    """Extract structured data from a single passage using Claude.

    Args:
        passage: Dict with book, chapter, greek_text, english_text.
        schema: Loaded YAML schema dict.
        few_shot_examples: Unused (built-in examples are used).

    Returns:
        List of extraction dicts, each with an extraction_id added.
    """
    schema_text = render_schema_for_prompt(schema)
    system_prompt = build_system_prompt(schema_text)
    user_message = build_user_message(passage)

    client = anthropic.Anthropic()

    # First attempt
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines)

    # Parse JSON
    try:
        extractions = json.loads(raw_text)
    except json.JSONDecodeError:
        # Retry once
        retry_response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": raw_text},
                {"role": "user", "content": "Please return valid JSON only."},
            ],
        )
        raw_text = retry_response.content[0].text.strip()
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_text = "\n".join(lines)
        extractions = json.loads(raw_text)

    if not isinstance(extractions, list):
        extractions = [extractions]

    # Validate and add IDs
    results = []
    for ext in extractions:
        cleaned = _validate_extraction(ext, schema)
        cleaned["extraction_id"] = str(uuid.uuid4())[:8]
        results.append(cleaned)

    return results
