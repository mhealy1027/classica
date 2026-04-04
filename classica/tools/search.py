"""Tool: keyword search across passages."""

import re
from typing import Optional


def _highlight_match(text: str, keyword: str, context_chars: int = 50) -> list[str]:
    """Find keyword in text and return highlighted snippets with context."""
    snippets = []
    lower_text = text.lower()
    lower_kw = keyword.lower()
    start = 0

    while True:
        idx = lower_text.find(lower_kw, start)
        if idx == -1:
            break

        ctx_start = max(0, idx - context_chars)
        ctx_end = min(len(text), idx + len(keyword) + context_chars)

        prefix = "..." if ctx_start > 0 else ""
        suffix = "..." if ctx_end < len(text) else ""

        snippet = text[ctx_start:ctx_end]
        # Mark the keyword
        match_start = idx - ctx_start
        match_end = match_start + len(keyword)
        snippet = (
            snippet[:match_start]
            + "**"
            + snippet[match_start:match_end]
            + "**"
            + snippet[match_end:]
        )

        snippets.append(f"{prefix}{snippet}{suffix}")
        start = idx + len(keyword)

    return snippets


def search_passages(
    passages: list[dict],
    keywords: list[str],
    greek_keywords: Optional[list[str]] = None,
) -> list[dict]:
    """Search across all parsed passages for keyword matches.

    Args:
        passages: List of passage dicts (book, chapter, greek_text, english_text).
        keywords: English keywords to search for.
        greek_keywords: Optional Greek keywords to search for.

    Returns:
        List of dicts with book, chapter, keyword, language, snippets.
    """
    all_keywords = [(kw, "english") for kw in keywords]
    if greek_keywords:
        all_keywords.extend((kw, "greek") for kw in greek_keywords)

    results = []

    for passage in passages:
        for keyword, lang in all_keywords:
            # Search English text
            if lang == "english" and passage.get("english_text"):
                snippets = _highlight_match(passage["english_text"], keyword)
                if snippets:
                    results.append({
                        "book": passage["book"],
                        "chapter": passage["chapter"],
                        "keyword": keyword,
                        "language": "english",
                        "snippets": snippets,
                    })

            # Search Greek text
            if lang == "greek" and passage.get("greek_text"):
                snippets = _highlight_match(passage["greek_text"], keyword)
                if snippets:
                    results.append({
                        "book": passage["book"],
                        "chapter": passage["chapter"],
                        "keyword": keyword,
                        "language": "greek",
                        "snippets": snippets,
                    })

        # Also search English keywords in English text (already covered)
        # and search all keywords in both texts for broader matching
        for keyword in keywords:
            if passage.get("greek_text"):
                snippets = _highlight_match(passage["greek_text"], keyword)
                if snippets:
                    # Avoid duplicate entries
                    existing = any(
                        r["book"] == passage["book"]
                        and r["chapter"] == passage["chapter"]
                        and r["keyword"] == keyword
                        and r["language"] == "greek"
                        for r in results
                    )
                    if not existing:
                        results.append({
                            "book": passage["book"],
                            "chapter": passage["chapter"],
                            "keyword": keyword,
                            "language": "greek",
                            "snippets": snippets,
                        })

    return results
