"""Tool: compare extractions for consistency and find duplicates."""

from collections import defaultdict
from typing import Any


def cross_reference(extractions: list[dict]) -> dict:
    """Analyze extractions for duplicates, inconsistencies, and running totals.

    Args:
        extractions: Full list of extraction dicts.

    Returns:
        Dict with 'duplicates', 'inconsistencies', 'running_totals' keys.
    """
    duplicates = _find_duplicates(extractions)
    running_totals = _compute_running_totals(extractions)
    inconsistencies = _check_inconsistencies(extractions, running_totals)

    return {
        "duplicates": duplicates,
        "inconsistencies": inconsistencies,
        "running_totals": running_totals,
    }


def _find_duplicates(extractions: list[dict]) -> list[dict]:
    """Group extractions by (year_bce, actor, amount, unit) to find duplicates."""
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for ext in extractions:
        if ext.get("amount") is None:
            continue
        key = (
            ext.get("year_bce"),
            ext.get("actor", "").lower().strip(),
            ext.get("amount"),
            ext.get("unit", "").lower().strip(),
        )
        groups[key].append(ext)

    duplicates = []
    for key, group in groups.items():
        if len(group) > 1:
            duplicates.append({
                "key": {
                    "year_bce": key[0],
                    "actor": key[1],
                    "amount": key[2],
                    "unit": key[3],
                },
                "count": len(group),
                "extraction_ids": [e.get("extraction_id", "?") for e in group],
                "locations": [
                    f"Book {e.get('book')}, Ch {e.get('chapter')}" for e in group
                ],
            })

    return duplicates


def _compute_running_totals(extractions: list[dict]) -> dict[str, Any]:
    """Compute running totals for ships and other key units."""
    totals: dict[str, dict] = {}

    ship_units = {"triremes", "ships"}
    talent_units = {"talents"}

    ship_by_direction: dict[str, float] = defaultdict(float)
    talent_by_direction: dict[str, float] = defaultdict(float)

    for ext in sorted(extractions, key=lambda e: (e.get("book", 0), e.get("chapter", 0))):
        amount = ext.get("amount")
        if amount is None:
            continue

        unit = str(ext.get("unit", "")).lower().strip()
        direction = str(ext.get("direction", "")).lower().strip()

        if unit in ship_units:
            ship_by_direction[direction] += float(amount)
        if unit in talent_units:
            talent_by_direction[direction] += float(amount)

    totals["ships"] = {
        "assets": ship_by_direction.get("asset", 0),
        "expenditures": ship_by_direction.get("expenditure", 0),
        "losses": ship_by_direction.get("loss", 0),
    }
    totals["talents"] = {
        "revenue": talent_by_direction.get("revenue", 0),
        "expenditures": talent_by_direction.get("expenditure", 0),
        "assets": talent_by_direction.get("asset", 0),
    }

    return totals


def _check_inconsistencies(
    extractions: list[dict], running_totals: dict
) -> list[dict]:
    """Check for known cross-references and inconsistencies.

    For example, Thucydides 8.79 says 82 ships remaining after Sicily.
    """
    inconsistencies = []

    # Check: do we have a Book 8, Chapter 79 reference about ships?
    book8_ships = [
        e for e in extractions
        if e.get("book") == 8
        and e.get("chapter") == 79
        and str(e.get("unit", "")).lower() in ("ships", "triremes")
    ]

    ship_totals = running_totals.get("ships", {})
    total_assets = ship_totals.get("assets", 0)
    total_losses = ship_totals.get("losses", 0)
    computed_remaining = total_assets - total_losses

    if book8_ships:
        for ref in book8_ships:
            stated = ref.get("amount", 0)
            if stated and computed_remaining != float(stated):
                inconsistencies.append({
                    "check": "Ship count at 8.79",
                    "stated_value": stated,
                    "computed_value": computed_remaining,
                    "difference": float(stated) - computed_remaining,
                    "note": (
                        f"Thucydides states {stated} ships at 8.79. "
                        f"Running total (assets - losses) = {computed_remaining}."
                    ),
                })

    return inconsistencies
