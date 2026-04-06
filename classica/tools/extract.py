"""Tool: extract structured data from passages using Claude API."""

import json
import uuid
from typing import Any, Optional

import anthropic

from classica.config import CLAUDE_MODEL, cost_tracker
from classica.prompts.extraction import build_system_prompt, build_user_message
from classica.schemas.base import render_schema_for_prompt

# ---------------------------------------------------------------------------
# Keywords for the pre-filter (two-pass extraction)
# ---------------------------------------------------------------------------

ENGLISH_KEYWORDS = [
    "talent", "drachma", "obol", "ship", "trireme", "hoplite", "cavalry",
    "archer", "fleet", "tribute", "pay", "wage", "cost", "expense", "money",
    "silver", "gold", "fund", "reserve", "garrison", "expedition", "force",
    "army", "sail", "soldier", "troops", "mercenary",
]

GREEK_KEYWORDS = [
    "τάλαντ", "δραχμ", "ὀβολ", "ναῦ", "τριήρ", "ὁπλίτ", "ἱππ",
    "φόρο", "μισθό", "χρήμ", "στρατ", "ναυτ",
]


def _passage_matches_keywords(passage: dict) -> bool:
    """Check if a passage contains any financial/military keywords."""
    en = (passage.get("english_text") or "").lower()
    gr = passage.get("greek_text") or ""

    for kw in ENGLISH_KEYWORDS:
        if kw in en:
            return True
    for kw in GREEK_KEYWORDS:
        if kw in gr:
            return True
    return False


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return len(text) // 4


def _validate_extraction(extraction: dict, schema: dict) -> dict:
    """Validate an extraction dict against the schema field definitions.

    Coerces types where possible and checks enum membership.
    Returns the cleaned extraction.
    """
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


def _parse_response(raw_text: str) -> list:
    """Strip markdown fences and parse JSON array from response text."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def _call_api(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_message: str,
    model: str,
) -> tuple[str, int, int]:
    """Make an API call and return (response_text, input_tokens, output_tokens)."""
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    raw_text = response.content[0].text.strip()
    in_tok = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    cost_tracker.record(model, in_tok, out_tok)
    return raw_text, in_tok, out_tok


def extract_passage(
    passage: dict,
    schema: dict,
    few_shot_examples: Any = None,
    model: Optional[str] = None,
) -> list[dict]:
    """Extract structured data from a single passage using Claude.

    Args:
        passage: Dict with book, chapter, greek_text, english_text.
        schema: Loaded YAML schema dict.
        few_shot_examples: Unused (built-in examples are used).
        model: Override model name. Defaults to CLAUDE_MODEL (Haiku).

    Returns:
        List of extraction dicts, each with an extraction_id added.
    """
    model = model or CLAUDE_MODEL
    schema_text = render_schema_for_prompt(schema)
    system_prompt = build_system_prompt(schema_text)
    user_message = build_user_message(passage)

    client = anthropic.Anthropic()

    raw_text, _, _ = _call_api(client, system_prompt, user_message, model)

    # Parse JSON
    try:
        extractions = _parse_response(raw_text)
    except json.JSONDecodeError:
        # Retry once
        retry_resp = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": raw_text},
                {"role": "user", "content": "Please return valid JSON only."},
            ],
        )
        raw_text = retry_resp.content[0].text.strip()
        cost_tracker.record(model, retry_resp.usage.input_tokens, retry_resp.usage.output_tokens)
        extractions = _parse_response(raw_text)

    if not isinstance(extractions, list):
        extractions = [extractions]

    # Validate and add IDs
    results = []
    for ext in extractions:
        cleaned = _validate_extraction(ext, schema)
        cleaned["extraction_id"] = str(uuid.uuid4())[:8]
        results.append(cleaned)

    return results


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def _build_batch_user_message(passages: list[dict]) -> str:
    """Build a single user message containing multiple chapters with delimiters."""
    parts = []
    for p in passages:
        parts.append(f"=== Book {p['book']}, Chapter {p['chapter']} ===")
        parts.append(f"English:\n{p['english_text']}")
        parts.append(f"Greek:\n{p['greek_text']}")
        parts.append("")
    return "\n".join(parts) + (
        "\nExtract all financial and military data from ALL passages above as a JSON array. "
        "Include book and chapter fields in each extraction object."
    )


def _pack_batches(passages: list[dict], max_tokens: int = 3000, solo_threshold: int = 2000) -> list[list[dict]]:
    """Pack passages into batches that stay under max_tokens of passage text."""
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for p in passages:
        text = (p.get("english_text") or "") + (p.get("greek_text") or "")
        tokens = _estimate_tokens(text)

        # Solo send if chapter is very large
        if tokens > solo_threshold:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([p])
            continue

        # Would adding this exceed limit?
        if current_tokens + tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(p)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def extract_passages_batched(
    passages: list[dict],
    schema: dict,
    model: Optional[str] = None,
    skip_filter: bool = False,
) -> list[dict]:
    """Extract from multiple passages using batching and optional keyword pre-filter.

    Args:
        passages: All passages to consider.
        schema: Loaded YAML schema dict.
        model: Override model name. Defaults to CLAUDE_MODEL (Haiku).
        skip_filter: If True, skip the keyword pre-filter and extract from all chapters.

    Returns:
        List of all extraction dicts.
    """
    model = model or CLAUDE_MODEL

    # Two-pass: pre-filter by keyword
    if not skip_filter:
        filtered = [p for p in passages if _passage_matches_keywords(p)]
        skipped = len(passages) - len(filtered)
        if skipped > 0:
            print(f"Skipping {skipped} chapters with no financial/military keywords")
        passages = filtered

    if not passages:
        print("No passages to extract from after filtering.")
        return []

    # Pack into batches
    batches = _pack_batches(passages)
    total_batches = len(batches)

    schema_text = render_schema_for_prompt(schema)
    system_prompt = build_system_prompt(schema_text)
    client = anthropic.Anthropic()

    all_extractions: list[dict] = []

    for i, batch in enumerate(batches, 1):
        first = batch[0]
        last = batch[-1]
        label = (
            f"{first['book']}.{first['chapter']}-{last['book']}.{last['chapter']}"
            if len(batch) > 1
            else f"{first['book']}.{first['chapter']}"
        )
        print(f"[Batch {i}/{total_batches}] Chapters {label} ({len(batch)} chapter{'s' if len(batch) != 1 else ''})...")

        if len(batch) == 1:
            user_message = build_user_message(batch[0])
        else:
            user_message = _build_batch_user_message(batch)

        try:
            raw_text, _, _ = _call_api(client, system_prompt, user_message, model)
            extractions = _parse_response(raw_text)
        except json.JSONDecodeError:
            # Retry once
            try:
                retry_resp = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    temperature=0,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": raw_text},
                        {"role": "user", "content": "Please return valid JSON only."},
                    ],
                )
                cost_tracker.record(model, retry_resp.usage.input_tokens, retry_resp.usage.output_tokens)
                extractions = _parse_response(retry_resp.content[0].text.strip())
            except Exception as e:
                print(f"  -> ERROR: {e}")
                continue
        except Exception as e:
            print(f"  -> ERROR: {e}")
            continue

        if not isinstance(extractions, list):
            extractions = [extractions]

        for ext in extractions:
            cleaned = _validate_extraction(ext, schema)
            cleaned["extraction_id"] = str(uuid.uuid4())[:8]
            all_extractions.append(cleaned)

        print(f"  -> {len(extractions)} extraction(s)")

    return all_extractions
