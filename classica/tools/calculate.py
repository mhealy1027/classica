"""Tools: calculate_expenditure and build_balance_sheet."""

from collections import defaultdict
from typing import Optional

DRACHMAS_PER_TALENT = 6000
OTHER_ANNUAL_REVENUE = 200  # Non-tribute: Laurion silver, harbor dues, court fees, etc.

# ATL-based tribute estimates (gold + silver, in talents)
# Pre-425: ~388 talents per Athenian Tribute List
# 425 reassessment (Thudippus decree): quotas tripled, effective receipts ~900-1000
# Post-Sicily (413): allied revolts, revenue collapses
ATL_TRIBUTE = {
    431: 388, 430: 388, 429: 388, 428: 388, 427: 388, 426: 388,
    425: 600,   # transition year, partial reassessment effect
    424: 1000, 423: 1000, 422: 900, 421: 850,
    420: 800, 419: 750, 418: 700, 417: 650, 416: 600, 415: 550,
    414: 400, 413: 300,  # Sicily disaster (summer 413)
    412: 200, 411: 150, 410: 150,
    409: 200, 408: 200, 407: 200, 406: 150, 405: 100, 404: 50,
}

# Default expenditure estimates (talents/year) derived from known Thucydides data
# and standard scholarship on Athenian war costs
DEFAULT_EXPENDITURES = {
    431: 1500,  # Early war: 100-ship fleet + Potidaea siege beginning
    430: 2000,  # Plague year: double fleet + Potidaea continuing (2000 tal total by surrender)
    429: 1800,  # Naval campaigns in Corinthian Gulf + Plataea siege
    428: 1500,  # Mytilene revolt: extra fleet dispatched
    427: 1400,  # Mytilene aftermath + first Sicily expedition
    426: 1600,  # Demosthenes' western campaigns
    425: 1400,  # Pylos/Sphacteria + post-reassessment revenue boost
    424: 1200,  # Delium + Brasidas' Thrace campaign
    423: 1200,  # Armistice year
    422: 1100,  # Amphipolis campaign (Cleon and Brasidas die)
    421: 800,   # Peace of Nicias: reduced operations
    420: 600,   # Peace period
    419: 700,   # Argive alliance building
    418: 900,   # Mantinea campaign
    417: 700,   # Resumed cold war
    416: 700,   # Melos
    415: 2000,  # Sicilian Expedition departs (massive outlay)
    414: 3000,  # Sicily: full campaign
    413: 4000,  # Sicily disaster + Decelea fortified + emergency levies
    412: 1500,  # Ionian War begins: fleet rebuilding
    411: 1200,  # Oligarchic coup + Hellespont fleet
    410: 1200,  # Cyzicus victory
    409: 1200,  # Continued eastern operations
    408: 1100,  # Alcibiades' campaigns
    407: 1000,  # Notium (Alcibiades exiled again)
    406: 1000,  # Arginusae: last naval victory
    405: 800,   # Aegospotami: fleet destroyed, siege of Athens begins
    404: 500,   # Surrender
}


def calculate_expenditure(
    extractions: list[dict],
    wage_rate: float = 1.0,
    crew_size: int = 200,
    campaign_months: float = 8.0,
) -> dict:
    """Calculate yearly talent expenditures from ship and troop count extractions.

    Converts unit counts to talent costs using thesis wage rate assumptions:
    - Ships/triremes: amount × crew_size × wage_rate × days / 6000
    - Hoplites: amount × wage_rate × days / 6000
    - Cavalry: amount × wage_rate × 2 × days / 6000 (horse fodder doubles cost)

    Only processes extractions with direction 'expenditure' or 'asset'
    (both represent deployed/committed forces).

    Args:
        extractions: List of extraction dicts (filter to desired actor/direction first).
        wage_rate: Drachmas per person per day. Default 1.0 (standard rate).
                   Use 2.0 for Potidaea-rate hoplites (Thucydides 3.17).
        crew_size: Crew members per trireme. Default 200.
        campaign_months: Campaign season in months. Default 8 (240 days).

    Returns:
        Dict mapping year_bce (int) to cost breakdown:
        {naval_talents, hoplite_talents, cavalry_talents, total_talents, sources}.
    """
    days = campaign_months * 30

    by_year: dict = defaultdict(lambda: {
        "naval_talents": 0.0,
        "hoplite_talents": 0.0,
        "cavalry_talents": 0.0,
        "total_talents": 0.0,
        "sources": [],
    })

    for ext in extractions:
        year_raw = ext.get("year_bce")
        amount_raw = ext.get("amount")

        if year_raw in (None, "") or amount_raw in (None, ""):
            continue
        try:
            amount = float(amount_raw)
            year = int(float(year_raw))
        except (ValueError, TypeError):
            continue
        if amount <= 0:
            continue

        unit = str(ext.get("unit", "")).lower().strip()
        direction = str(ext.get("direction", "")).lower().strip()

        if direction != "expenditure":
            continue

        label = f"Bk{ext.get('book')}.{ext.get('chapter')}: {amount:.0f} {unit}"

        if unit in ("ships", "triremes"):
            cost = (amount * crew_size * wage_rate * days) / DRACHMAS_PER_TALENT
            by_year[year]["naval_talents"] += cost
            by_year[year]["sources"].append(f"{label} -> {cost:.1f} tal")

        elif unit == "hoplites":
            cost = (amount * wage_rate * days) / DRACHMAS_PER_TALENT
            by_year[year]["hoplite_talents"] += cost
            by_year[year]["sources"].append(f"{label} -> {cost:.1f} tal")

        elif unit == "cavalry":
            cost = (amount * wage_rate * 2 * days) / DRACHMAS_PER_TALENT
            by_year[year]["cavalry_talents"] += cost
            by_year[year]["sources"].append(f"{label} -> {cost:.1f} tal")

    result = {}
    for year, vals in sorted(by_year.items(), reverse=True):
        result[year] = {
            "naval_talents": round(vals["naval_talents"], 1),
            "hoplite_talents": round(vals["hoplite_talents"], 1),
            "cavalry_talents": round(vals["cavalry_talents"], 1),
            "total_talents": round(
                vals["naval_talents"] + vals["hoplite_talents"] + vals["cavalry_talents"], 1
            ),
            "sources": vals["sources"],
        }

    return result


def build_balance_sheet(
    scenario: str = "thucydides",
    initial_reserve: float = 6540.0,
    iron_reserve: float = 1000.0,
    yearly_revenue: Optional[dict] = None,
    yearly_expenditures: Optional[dict] = None,
    start_year: int = 431,
    end_year: int = 404,
) -> dict:
    """Build a year-by-year Athenian financial balance sheet.

    Three preset revenue scenarios for tribute income:
    - "thucydides": 600 talents/year (Thucydides 2.13, undisputed figure)
    - "average": 458 talents/year (average needed to deplete available
                  reserves by 413 BCE)
    - "atl": ATL pattern (~388 pre-425, ~1000 post-425 reassessment,
              declining after Sicily 413)
    - "custom": Use yearly_revenue dict directly.

    Default initial reserve: 6540 talents (Thuc. 2.13: 6000 coined silver
    + 500 uncoined offerings + 40 gold on Athena Parthenos statue).
    Iron reserve: 1000 talents (Thuc. 2.24 — death penalty for unauthorized use).

    Args:
        scenario: Revenue preset ("thucydides", "average", "atl", "custom").
        initial_reserve: Starting reserve in talents.
        iron_reserve: Emergency reserve not to be touched.
        yearly_revenue: Custom {year_bce: total_talents} dict (for "custom" scenario
                        or to override defaults).
        yearly_expenditures: {year_bce: talents} expenditure dict.
                             Accepts calculate_expenditure() output format.
                             Falls back to DEFAULT_EXPENDITURES if None.
        start_year: Earliest year BCE (default 431).
        end_year: Latest year BCE (default 404).

    Returns:
        Dict with scenario metadata and list of year-by-year rows.
    """
    SCENARIO_DESCRIPTIONS = {
        "thucydides": "600 talents/year tribute (Thucydides 2.13; disputed by ATL)",
        "average": "458 talents/year tribute (average needed to exhaust reserves by 413 BCE)",
        "atl": "ATL pattern: ~388 pre-425 -> ~1000 post-425 Thudippus reassessment -> declining after Sicily",
        "custom": "User-supplied revenue figures",
    }

    # Build tribute stream by year
    tribute_by_year: dict[int, float] = {}
    for y in range(end_year, start_year + 1):
        if scenario == "thucydides":
            tribute_by_year[y] = 600.0
        elif scenario == "average":
            tribute_by_year[y] = 458.0
        elif scenario == "atl":
            tribute_by_year[y] = float(ATL_TRIBUTE.get(y, 300))
        else:
            tribute_by_year[y] = float((yearly_revenue or {}).get(y, 600))

    # Full revenue = tribute + other sources
    if scenario == "custom" and yearly_revenue:
        rev_stream = {int(y): float(v) for y, v in yearly_revenue.items()}
    else:
        rev_stream = {
            y: tribute_by_year[y] + OTHER_ANNUAL_REVENUE
            for y in tribute_by_year
        }

    # Build expenditure stream
    if yearly_expenditures:
        exp_stream: dict[int, float] = {}
        for y, v in yearly_expenditures.items():
            y_int = int(y)
            if isinstance(v, dict):
                exp_stream[y_int] = float(v.get("total_talents", 0))
            else:
                exp_stream[y_int] = float(v)
    else:
        exp_stream = {k: float(v) for k, v in DEFAULT_EXPENDITURES.items()}

    # Year-by-year simulation
    balance = initial_reserve
    rows = []
    depletion_year = None
    broke_iron_reserve_year = None

    for year in range(start_year, end_year - 1, -1):  # 431, 430, ..., 404
        tribute = tribute_by_year.get(year, 0.0)
        total_rev = rev_stream.get(year, tribute + OTHER_ANNUAL_REVENUE)
        exp = exp_stream.get(year, float(DEFAULT_EXPENDITURES.get(year, 1000)))
        net = total_rev - exp
        balance = balance + net
        available = balance - iron_reserve

        below_iron = balance < iron_reserve
        if below_iron and broke_iron_reserve_year is None:
            broke_iron_reserve_year = year

        if available <= 0 and depletion_year is None:
            depletion_year = year

        rows.append({
            "scenario": scenario,
            "year_bce": year,
            "tribute_revenue": round(tribute, 1),
            "other_revenue": OTHER_ANNUAL_REVENUE,
            "total_revenue": round(total_rev, 1),
            "expenditure": round(exp, 1),
            "net": round(net, 1),
            "balance": round(balance, 1),
            "iron_reserve": iron_reserve,
            "available": round(available, 1),
            "below_iron_reserve": below_iron,
        })

    return {
        "scenario": scenario,
        "description": SCENARIO_DESCRIPTIONS.get(scenario, ""),
        "initial_reserve": initial_reserve,
        "iron_reserve": iron_reserve,
        "start_year": start_year,
        "end_year": end_year,
        "years": rows,
        "depletion_year": depletion_year,
        "broke_iron_reserve_year": broke_iron_reserve_year,
        "final_balance": round(balance, 1),
    }
