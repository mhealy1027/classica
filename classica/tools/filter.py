"""Tool: filter extractions by various criteria."""

from typing import Optional


def filter_extractions(
    extractions: list[dict],
    actor: Optional[str] = None,
    direction: Optional[str] = None,
    unit: Optional[str] = None,
    confidence: Optional[str] = None,
    book: Optional[int] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> list[dict]:
    """Filter a list of extraction dicts by various criteria.

    Args:
        extractions: List of extraction dicts.
        actor: Filter by actor name (case-insensitive). Comma-separated for OR logic.
        direction: Filter by direction (revenue, expenditure, asset, loss, transfer).
                   Comma-separated for OR logic.
        unit: Filter by unit (talents, triremes, ships, hoplites, etc.).
              Comma-separated for OR logic.
        confidence: Filter by confidence level (high, medium, low).
                    Comma-separated for OR logic.
        book: Filter by exact book number (integer).
        year_min: Minimum year_bce value (inclusive). BCE values: 404 < 431.
                  E.g. year_min=404 excludes years later than 404 BCE.
        year_max: Maximum year_bce value (inclusive).
                  E.g. year_max=431 excludes years earlier than 431 BCE.

    Returns:
        Filtered list of extraction dicts.
    """
    results = list(extractions)

    # Auto-swap if caller passes BCE years in natural order (431, 404) but
    # year_min > year_max because BCE values are numerically inverted.
    if year_min is not None and year_max is not None and year_min > year_max:
        year_min, year_max = year_max, year_min

    if actor:
        actors = {a.strip().lower() for a in actor.split(",")}
        results = [
            r for r in results
            if str(r.get("actor", "")).lower() in actors
        ]

    if direction:
        dirs = {d.strip().lower() for d in direction.split(",")}
        results = [
            r for r in results
            if str(r.get("direction", "")).lower() in dirs
        ]

    if unit:
        units = {u.strip().lower() for u in unit.split(",")}
        results = [
            r for r in results
            if str(r.get("unit", "")).lower() in units
        ]

    if confidence:
        confs = {c.strip().lower() for c in confidence.split(",")}
        results = [
            r for r in results
            if str(r.get("confidence", "")).lower() in confs
        ]

    if book is not None:
        results = [
            r for r in results
            if str(r.get("book", "")) == str(book)
        ]

    if year_min is not None:
        results = [
            r for r in results
            if r.get("year_bce") not in (None, "")
            and _to_int(r["year_bce"]) is not None
            and _to_int(r["year_bce"]) >= year_min
        ]

    if year_max is not None:
        results = [
            r for r in results
            if r.get("year_bce") not in (None, "")
            and _to_int(r["year_bce"]) is not None
            and _to_int(r["year_bce"]) <= year_max
        ]

    return results


def _to_int(val) -> Optional[int]:
    """Safely convert a value to int."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
