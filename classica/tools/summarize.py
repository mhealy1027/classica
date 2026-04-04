"""Tool: generate analysis memo from extractions."""

import json

import anthropic

from classica.config import CLAUDE_MODEL


def summarize_findings(
    extractions: list[dict],
    cross_ref_results: dict,
) -> dict:
    """Generate a natural-language analysis memo.

    Args:
        extractions: Full list of extraction dicts.
        cross_ref_results: Output from cross_reference().

    Returns:
        Dict with 'summary', 'key_findings', and 'uncertainties'.
    """
    if not extractions:
        return {
            "summary": "No extractions to summarize.",
            "key_findings": [],
            "uncertainties": [],
        }

    client = anthropic.Anthropic()

    system_prompt = """You are a classical historian and data analyst. Given a set of structured extractions from an ancient text and cross-reference analysis results, write a concise analytical memo.

Return your response as a JSON object with these keys:
- "summary": A 2-3 paragraph analysis of the data
- "key_findings": An array of 3-5 notable findings (strings)
- "uncertainties": An array of key uncertainties or data quality issues (strings)

Return ONLY valid JSON. No markdown, no commentary outside the JSON."""

    user_message = f"""## Extraction Data
Total extractions: {len(extractions)}

Sample extractions (first 20):
{json.dumps(extractions[:20], indent=2, default=str)}

## Cross-Reference Results
Duplicates found: {len(cross_ref_results.get('duplicates', []))}
Inconsistencies found: {len(cross_ref_results.get('inconsistencies', []))}

Running totals:
{json.dumps(cross_ref_results.get('running_totals', {}), indent=2, default=str)}

Duplicate details:
{json.dumps(cross_ref_results.get('duplicates', []), indent=2, default=str)}

Inconsistency details:
{json.dumps(cross_ref_results.get('inconsistencies', []), indent=2, default=str)}

Please provide your analytical memo as JSON."""

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
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines)

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        result = {
            "summary": raw_text,
            "key_findings": [],
            "uncertainties": ["Failed to parse structured response from Claude."],
        }

    return {
        "summary": result.get("summary", ""),
        "key_findings": result.get("key_findings", []),
        "uncertainties": result.get("uncertainties", []),
    }
