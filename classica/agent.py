"""Core agent loop for Classica.

The agent takes a natural language research question, plans which tools to
use, executes them autonomously, and produces structured data and an
analysis memo.

Workflow:
  PLAN    -> Claude Sonnet returns a step-by-step JSON execution plan.
  EXECUTE -> Agent iterates through the plan using native tool_use; Claude
             adapts after seeing each result.
  REFLECT -> Claude Sonnet synthesises findings into an analysis memo.
"""

import csv
import json
from pathlib import Path
from typing import Optional

import anthropic

from classica.config import (
    OUTPUT_DIR, CLAUDE_MODEL, CLAUDE_SONNET_MODEL, cost_tracker,
)
from classica.tools.ingest import ingest_text
from classica.tools.extract import extract_passage, extract_passages_batched
from classica.tools.search import search_passages as _search_passages
from classica.tools.cross_reference import cross_reference as _cross_reference
from classica.tools.export import export_data as _export_data
from classica.tools.summarize import summarize_findings as _summarize_findings
from classica.tools.filter import filter_extractions as _filter_extractions
from classica.tools.calculate import calculate_expenditure as _calc_exp
from classica.tools.calculate import build_balance_sheet as _build_bs
from classica.schemas.base import load_schema

# Default to Haiku for planning/execution; Sonnet only for reflection
AGENT_MODEL = CLAUDE_MODEL
REFLECTION_MODEL = CLAUDE_SONNET_MODEL
MAX_ITERATIONS = 20
CACHED_CSV = OUTPUT_DIR / "full_thucydides.csv"

# ---------------------------------------------------------------------------
# Tool definitions for the Anthropic SDK
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "load_cached_extractions",
        "description": (
            "Load previously extracted data from a CSV file on disk. "
            "ALWAYS call this first — it loads 2003 extractions across all 8 books "
            "of Thucydides without expensive re-extraction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "CSV path. Defaults to output/full_thucydides.csv",
                }
            },
            "required": [],
        },
    },
    {
        "name": "ingest_text",
        "description": "Fetch and parse Greek + English texts from Perseus Digital Library.",
        "input_schema": {
            "type": "object",
            "properties": {
                "author": {"type": "string", "description": "e.g. 'thucydides'"},
                "refresh": {"type": "boolean", "description": "Re-download if cached"},
            },
            "required": ["author"],
        },
    },
    {
        "name": "extract_passages",
        "description": (
            "Extract structured data from passages using Claude. Expensive (~917 API calls "
            "for full Thucydides). Only use if cached data doesn't exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "author": {"type": "string"},
                "schema_path": {"type": "string"},
                "books": {"type": "string", "description": "Comma-separated, e.g. '1,2'"},
                "output_path": {"type": "string"},
            },
            "required": ["author", "schema_path"],
        },
    },
    {
        "name": "search_passages",
        "description": "Full-text keyword search across parsed passages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "author": {"type": "string"},
                "keywords": {"type": "string", "description": "Comma-separated English keywords"},
                "greek_keywords": {"type": "string", "description": "Comma-separated Greek keywords"},
            },
            "required": ["author", "keywords"],
        },
    },
    {
        "name": "filter_extractions",
        "description": (
            "Filter the loaded extractions by actor, direction, unit, confidence, book, or year range. "
            "Filtered results are stored and used by calculate_expenditure and other tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "actor": {
                    "type": "string",
                    "description": "Actor name, e.g. 'Athens'. Comma-separated for OR logic.",
                },
                "direction": {
                    "type": "string",
                    "description": "One or more of: revenue, expenditure, asset, loss, transfer. Comma-separated.",
                },
                "unit": {
                    "type": "string",
                    "description": "E.g. 'talents', 'triremes,ships', 'hoplites'. Comma-separated.",
                },
                "confidence": {
                    "type": "string",
                    "description": "One or more of: high, medium, low. Comma-separated.",
                },
                "book": {"type": "integer", "description": "Book number 1-8"},
                "year_min": {
                    "type": "integer",
                    "description": (
                        "Minimum year_bce (inclusive). BCE values increase into the past. "
                        "year_min=404 keeps only years 404 BCE and later (>=404)."
                    ),
                },
                "year_max": {
                    "type": "integer",
                    "description": (
                        "Maximum year_bce (inclusive). "
                        "year_max=431 keeps only years 431 BCE and earlier (<=431)."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "cross_reference",
        "description": "Analyse the current (filtered) extractions for duplicates, inconsistencies, and running totals.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "calculate_expenditure",
        "description": (
            "Calculate yearly talent expenditures from ship and troop counts in the "
            "current filtered extractions, using thesis wage rate assumptions. "
            "Results are stored and automatically used by build_balance_sheet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "wage_rate": {
                    "type": "number",
                    "description": "Drachmas per person per day. Default 1.0. Use 2.0 for Potidaea-rate.",
                },
                "crew_size": {
                    "type": "integer",
                    "description": "Crew per trireme. Default 200.",
                },
                "campaign_months": {
                    "type": "number",
                    "description": "Campaign season in months. Default 8.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "build_balance_sheet",
        "description": (
            "Build a year-by-year Athenian financial balance sheet (431-404 BCE). "
            "When asked to compare tribute scenarios, call this THREE times with "
            "scenario='thucydides', then 'average', then 'atl'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "enum": ["thucydides", "average", "atl", "custom"],
                    "description": (
                        "'thucydides': 600 talents/yr tribute (Thuc. 2.13). "
                        "'average': 458 talents/yr (depletes reserves by 413 BCE). "
                        "'atl': ATL pattern — ~388 pre-425, ~1000 post-425 reassessment, declining after Sicily."
                    ),
                },
                "initial_reserve": {
                    "type": "number",
                    "description": "Starting reserve in talents. Default 6540 (Thuc. 2.13: 6000+500+40).",
                },
                "iron_reserve": {
                    "type": "number",
                    "description": "Untouchable emergency reserve. Default 1000 (Thuc. 2.24).",
                },
                "start_year": {"type": "integer", "description": "Default 431"},
                "end_year": {"type": "integer", "description": "Default 404"},
            },
            "required": ["scenario"],
        },
    },
    {
        "name": "export_data",
        "description": "Export extractions, balance sheets, or expenditure data to CSV or JSON.",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_type": {
                    "type": "string",
                    "enum": ["extractions", "filtered_extractions", "balance_sheet", "expenditure"],
                },
                "format": {"type": "string", "enum": ["csv", "json"], "description": "Default csv"},
                "output_path": {"type": "string", "description": "File path"},
            },
            "required": ["data_type"],
        },
    },
    {
        "name": "summarize_findings",
        "description": "Generate a natural-language analysis memo from the current agent state. Call last.",
        "input_schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Optional focus, e.g. 'balance sheet comparison across scenarios'.",
                }
            },
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_PLANNING_PROMPT = """\
You are Classica Agent, a research assistant for ancient Greek financial history.

Given a research question, return a JSON array of planned execution steps.
Each step: {"tool": "<tool_name>", "args": {<optional args>}, "description": "<human label>"}

Available tools: load_cached_extractions, ingest_text, extract_passages,
search_passages, filter_extractions, cross_reference, calculate_expenditure,
build_balance_sheet, export_data, summarize_findings.

Rules:
- ALWAYS start with load_cached_extractions (file: output/full_thucydides.csv)
- For balance sheet questions run ALL THREE scenarios: thucydides, average, atl
- Filter before calculating
- Export results, then summarize
- Keep plans to 6-12 steps

Return ONLY valid JSON array, no other text."""

_EXECUTION_PROMPT = """\
You are Classica Agent, a research assistant for ancient Greek financial history.
Execute your research plan step by step using the provided tools.

Thesis assumptions (use these defaults):
- Sailors: 1 drachma/day  |  6000 dr = 1 talent/month per ship
- Hoplites: 1 drachma/day (Potidaea: 2 dr/day — Thucydides 3.17)
- Crew per trireme: 200
- Campaign season: 8 months = 240 days

Balance sheet context (431 BCE):
- Initial reserve: 6540 talents (Thuc. 2.13: 6000 coined + 500 uncoined + 40 on Athena)
- Iron reserve: 1000 talents (Thuc. 2.24)
- Available reserve: 5540 talents

Three tribute scenarios to always compare:
1. "thucydides": 600 talents/yr  (Thuc. 2.13, face value)
2. "average"   : 458 talents/yr  (average that depletes reserves by 413)
3. "atl"       : ATL pattern — ~388 pre-425, ~1000 post-425 Thudippus reassessment, declining after Sicily

Execute methodically. The first tool call should be load_cached_extractions.\
"""

_REFLECTION_PROMPT = """\
You are Classica Agent. You have just completed a research workflow on Athenian finances.

Write a comprehensive analysis memo (3-4 paragraphs) covering:
1. What the data shows about Athens' financial position year by year
2. How the three tribute scenarios diverge and what that means for the thesis
3. Key uncertainties and data quality issues
4. Scholarly implications

Be specific about numbers. Reference Thucydides book and chapter where relevant.\
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_state() -> dict:
    return {
        "extractions": [],
        "filtered_extractions": [],
        "passages": [],
        "balance_sheets": {},   # {scenario: build_balance_sheet result}
        "expenditure_data": None,
        "xref_results": None,
        "search_results": [],
        "exports": [],
        "memo": None,
    }


def _load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Coerce numeric fields
            for field in ("book", "chapter", "year_bce"):
                if row.get(field) not in (None, ""):
                    try:
                        row[field] = int(float(row[field]))
                    except (ValueError, TypeError):
                        pass
            if row.get("amount") not in (None, ""):
                try:
                    row["amount"] = float(row["amount"])
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def _parse_plan(text: str) -> list[dict]:
    """Extract a JSON array from Claude's planning response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        plan = json.loads(text)
        if isinstance(plan, list):
            return plan
    except json.JSONDecodeError:
        pass
    # Fallback: minimal plan
    return [
        {"tool": "load_cached_extractions", "args": {}, "description": "Load cached extractions"},
        {"tool": "summarize_findings", "args": {}, "description": "Summarize findings"},
    ]


def _compact_result(name: str, result: dict) -> str:
    """Return a brief human-readable summary of a tool result."""
    if name == "load_cached_extractions":
        return (
            f"Loaded {result.get('count', 0)} extractions "
            f"(high={result.get('high', 0)}, med={result.get('med', 0)}, "
            f"low={result.get('low', 0)})"
        )
    if name == "filter_extractions":
        return f"Filtered to {result.get('count', 0)} extractions"
    if name == "calculate_expenditure":
        yrs = result.get("years_with_data", [])
        return (
            f"Computed costs for {len(yrs)} years: "
            + ", ".join(f"{y}->{result['yearly_totals'].get(y, 0):.0f}tal" for y in yrs[:5])
            + ("..." if len(yrs) > 5 else "")
        )
    if name == "build_balance_sheet":
        dep = result.get("depletion_year")
        iron = result.get("broke_iron_reserve_year")
        return (
            f"Scenario '{result.get('scenario')}': "
            f"iron reserve breached={iron}, depletion={dep}, "
            f"final balance={result.get('final_balance')} talents"
        )
    if name == "export_data":
        return f"Exported {result.get('rows', 0)} rows -> {result.get('path', '?')}"
    if name == "summarize_findings":
        memo = result.get("summary", "")
        return memo[:200] + "..." if len(memo) > 200 else memo
    return json.dumps(result, default=str)[:300]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_tool(name: str, inputs: dict, state: dict, client: anthropic.Anthropic) -> dict:
    """Execute a named tool and return a compact result dict for Claude."""

    if name == "load_cached_extractions":
        raw_path = inputs.get("path") or str(CACHED_CSV)
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            # Try relative to project root
            from classica.config import OUTPUT_DIR
            path = OUTPUT_DIR / Path(raw_path).name
        if not path.exists():
            return {"error": f"File not found: {raw_path}", "count": 0}
        rows = _load_csv(path)
        state["extractions"] = rows
        state["filtered_extractions"] = []
        by_conf = {"high": 0, "medium": 0, "low": 0}
        for r in rows:
            c = str(r.get("confidence", "")).lower()
            if c in by_conf:
                by_conf[c] += 1
        return {
            "count": len(rows),
            "high": by_conf["high"],
            "med": by_conf["medium"],
            "low": by_conf["low"],
            "path": str(path),
        }

    if name == "ingest_text":
        result = ingest_text(
            author=inputs["author"],
            refresh=inputs.get("refresh", False),
        )
        state["passages"] = result["passages"]
        return {
            "author": result["author"],
            "books": result["total_books"],
            "chapters": result["total_chapters"],
        }

    if name == "extract_passages":
        from classica.schemas.base import load_schema

        if not state.get("passages"):
            data = ingest_text(author=inputs["author"])
            state["passages"] = data["passages"]

        schema = load_schema(inputs["schema_path"])
        passages = state["passages"]
        if inputs.get("books"):
            book_nums = [int(b) for b in str(inputs["books"]).split(",")]
            passages = [p for p in passages if p["book"] in book_nums]

        all_ext = extract_passages_batched(passages, schema)

        state["extractions"] = all_ext
        state["filtered_extractions"] = []

        if inputs.get("output_path"):
            _export_data(all_ext, format="csv", output_path=inputs["output_path"])

        return {"count": len(all_ext)}

    if name == "search_passages":
        if not state.get("passages"):
            data = ingest_text(author=inputs["author"])
            state["passages"] = data["passages"]
        keywords = [k.strip() for k in inputs["keywords"].split(",")]
        greek_kw = None
        if inputs.get("greek_keywords"):
            greek_kw = [k.strip() for k in inputs["greek_keywords"].split(",")]
        results = _search_passages(state["passages"], keywords, greek_kw)
        state["search_results"] = results
        return {
            "matches": len(results),
            "sample": [
                {"book": r["book"], "chapter": r["chapter"], "keyword": r["keyword"],
                 "snippet": r["snippets"][0][:100] if r["snippets"] else ""}
                for r in results[:5]
            ],
        }

    if name == "filter_extractions":
        source = state.get("extractions", [])
        filtered = _filter_extractions(source, **inputs)
        state["filtered_extractions"] = filtered
        # Summary stats
        by_dir: dict = {}
        by_unit: dict = {}
        years = set()
        for r in filtered:
            d = str(r.get("direction", "")).lower()
            by_dir[d] = by_dir.get(d, 0) + 1
            u = str(r.get("unit", "")).lower()
            by_unit[u] = by_unit.get(u, 0) + 1
            if r.get("year_bce"):
                try:
                    years.add(int(float(r["year_bce"])))
                except (ValueError, TypeError):
                    pass
        return {
            "count": len(filtered),
            "filters": inputs,
            "by_direction": by_dir,
            "by_unit": dict(sorted(by_unit.items(), key=lambda x: -x[1])[:8]),
            "year_range": [min(years), max(years)] if years else [],
        }

    if name == "cross_reference":
        source = state.get("filtered_extractions") or state.get("extractions", [])
        result = _cross_reference(source)
        state["xref_results"] = result
        return {
            "duplicates": len(result["duplicates"]),
            "inconsistencies": len(result["inconsistencies"]),
            "running_totals": result["running_totals"],
        }

    if name == "calculate_expenditure":
        source = state.get("filtered_extractions") or state.get("extractions", [])
        kwargs = {k: v for k, v in inputs.items() if k in ("wage_rate", "crew_size", "campaign_months")}
        result = _calc_exp(source, **kwargs)
        state["expenditure_data"] = result
        yearly_totals = {y: d["total_talents"] for y, d in result.items()}
        return {
            "years_with_data": sorted(yearly_totals.keys(), reverse=True),
            "yearly_totals": yearly_totals,
            "note": "Costs computed from ship/troop counts × wage rate × campaign days / 6000.",
        }

    if name == "build_balance_sheet":
        scenario = inputs.get("scenario", "thucydides")
        kwargs: dict = {"scenario": scenario}
        if "initial_reserve" in inputs:
            kwargs["initial_reserve"] = float(inputs["initial_reserve"])
        if "iron_reserve" in inputs:
            kwargs["iron_reserve"] = float(inputs["iron_reserve"])
        if "start_year" in inputs:
            kwargs["start_year"] = int(inputs["start_year"])
        if "end_year" in inputs:
            kwargs["end_year"] = int(inputs["end_year"])
        # Auto-inject expenditure data from previous calculate_expenditure call
        if state.get("expenditure_data"):
            kwargs["yearly_expenditures"] = {
                y: d["total_talents"] for y, d in state["expenditure_data"].items()
            }
        result = _build_bs(**kwargs)
        state["balance_sheets"][scenario] = result
        # Build compact year table for Claude
        table = [
            {
                "year": r["year_bce"],
                "revenue": r["total_revenue"],
                "expenditure": r["expenditure"],
                "net": r["net"],
                "balance": r["balance"],
                "available": r["available"],
            }
            for r in result["years"]
        ]
        return {
            "scenario": scenario,
            "description": result["description"],
            "initial_reserve": result["initial_reserve"],
            "iron_reserve": result["iron_reserve"],
            "broke_iron_reserve_year": result["broke_iron_reserve_year"],
            "depletion_year": result["depletion_year"],
            "final_balance": result["final_balance"],
            "year_table": table,
        }

    if name == "export_data":
        data_type = inputs.get("data_type", "extractions")
        fmt = inputs.get("format", "csv")
        output_path = inputs.get("output_path")

        if data_type == "balance_sheet":
            # Combine all scenarios into one CSV
            all_rows = []
            for bs in state.get("balance_sheets", {}).values():
                all_rows.extend(bs.get("years", []))
            if not all_rows:
                return {"error": "No balance sheets built yet"}
            if not output_path:
                output_path = str(OUTPUT_DIR / "athens_balance_sheet.csv")
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            import pandas as pd
            df = pd.DataFrame(all_rows)
            if fmt == "json":
                df.to_json(path, orient="records", indent=2)
            else:
                df.to_csv(path, index=False)
            state["exports"].append(str(path))
            return {"path": str(path), "rows": len(all_rows)}

        if data_type == "expenditure":
            exp = state.get("expenditure_data", {})
            rows = []
            for year, d in sorted(exp.items(), reverse=True):
                rows.append({
                    "year_bce": year,
                    "naval_talents": d["naval_talents"],
                    "hoplite_talents": d["hoplite_talents"],
                    "cavalry_talents": d["cavalry_talents"],
                    "total_talents": d["total_talents"],
                })
            if not rows:
                return {"error": "No expenditure data computed yet"}
            if not output_path:
                output_path = str(OUTPUT_DIR / "athens_expenditure.csv")
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            import pandas as pd
            pd.DataFrame(rows).to_csv(path, index=False)
            state["exports"].append(str(path))
            return {"path": str(path), "rows": len(rows)}

        # extractions / filtered_extractions
        source = (
            state.get("filtered_extractions")
            if data_type == "filtered_extractions"
            else state.get("extractions", [])
        )
        result = _export_data(source, format=fmt, output_path=output_path)
        state["exports"].append(result["path"])
        return {"path": result["path"], "rows": result["total_rows"]}

    if name == "summarize_findings":
        source = state.get("filtered_extractions") or state.get("extractions", [])
        xref = state.get("xref_results") or _cross_reference(source)

        # Enrich the summary context with balance sheet data
        focus = inputs.get("focus", "")
        bs_summary = {}
        for sc, bs in state.get("balance_sheets", {}).items():
            bs_summary[sc] = {
                "depletion_year": bs["depletion_year"],
                "broke_iron_reserve_year": bs["broke_iron_reserve_year"],
                "final_balance": bs["final_balance"],
                "description": bs["description"],
                "first_5_years": bs["years"][:5],
            }

        if bs_summary:
            # Use SONNET for the agent's summarize step (better reasoning)
            c = anthropic.Anthropic()
            sys_p = (
                "You are a classical historian. Write a comprehensive analytical memo "
                "about Athenian finances during the Peloponnesian War based on the "
                "structured data provided. Be specific about numbers and cite Thucydides "
                "by book and chapter. Return JSON with keys: summary, key_findings, uncertainties."
            )
            msg = (
                f"Focus: {focus}\n\n"
                f"Balance sheet scenarios:\n{json.dumps(bs_summary, indent=2, default=str)}\n\n"
                f"Expenditure data (sample):\n"
                + json.dumps(
                    {k: v["total_talents"] for k, v in (state.get("expenditure_data") or {}).items()},
                    default=str,
                )
                + f"\n\nExtraction stats: {len(source)} extractions, "
                f"{xref['running_totals']}"
            )
            resp = c.messages.create(
                model=REFLECTION_MODEL,
                max_tokens=4096,
                temperature=0,
                system=sys_p,
                messages=[{"role": "user", "content": msg}],
            )
            cost_tracker.record(REFLECTION_MODEL, resp.usage.input_tokens, resp.usage.output_tokens)
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                if raw.endswith("```"):
                    raw = raw[:-3]
            try:
                memo = json.loads(raw)
            except json.JSONDecodeError:
                memo = {"summary": raw, "key_findings": [], "uncertainties": []}
        else:
            memo = _summarize_findings(source, xref)

        state["memo"] = memo
        return memo

    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(
    question: str,
    agent_state: Optional[dict] = None,
    messages: Optional[list] = None,
) -> tuple[dict, list]:
    """Run the full agent loop for a research question.

    Args:
        question: Natural language research question.
        agent_state: Persistent state dict (pass from previous run for interactive mode).
        messages: Conversation history (pass from previous run for interactive mode).

    Returns:
        Tuple of (agent_state, messages) for use in follow-up questions.
    """
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if agent_state is None:
        agent_state = _empty_state()
    if messages is None:
        messages = []

    client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Phase 1: Planning
    # ------------------------------------------------------------------
    print("[PLAN] Analyzing your question and creating execution plan...")
    plan_resp = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=1024,
        system=_PLANNING_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Research question: {question}\n\nCreate a JSON execution plan.",
        }],
    )
    cost_tracker.record(AGENT_MODEL, plan_resp.usage.input_tokens, plan_resp.usage.output_tokens)
    plan_text = plan_resp.content[0].text
    plan = _parse_plan(plan_text)

    print(f"[PLAN] Breaking down your question into {len(plan)} steps...")
    for i, step in enumerate(plan, 1):
        desc = step.get("description") or step.get("tool", "?")
        print(f"  {i}. {desc}")

    # ------------------------------------------------------------------
    # Phase 2: Execute via native tool_use
    # ------------------------------------------------------------------
    messages.append({
        "role": "user",
        "content": (
            f"Research question: {question}\n\n"
            f"Execution plan:\n{plan_text}\n\n"
            "Execute this plan step by step using the provided tools. "
            "Start with load_cached_extractions."
        ),
    })

    step_count = 0
    final_text = ""

    while step_count < MAX_ITERATIONS:
        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=4096,
            system=_EXECUTION_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        cost_tracker.record(AGENT_MODEL, response.usage.input_tokens, response.usage.output_tokens)

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    final_text = block.text
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    step_count += 1
                    tool_label = _tool_label(block.name, block.input)
                    print(f"\n[STEP {step_count}] {tool_label}...")

                    try:
                        result = _execute_tool(block.name, block.input, agent_state, client)
                    except Exception as exc:
                        result = {"error": str(exc)}

                    print(f"  -> {_compact_result(block.name, result)}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })

            messages.append({"role": "user", "content": tool_results})

    if step_count >= MAX_ITERATIONS:
        print(f"\n[WARN] Reached max iterations ({MAX_ITERATIONS})")

    # ------------------------------------------------------------------
    # Phase 3: Reflect — if Claude ended without a final text memo,
    #           explicitly ask for one.
    # ------------------------------------------------------------------
    if not final_text and agent_state.get("balance_sheets"):
        print("\n[REFLECT] Generating final analysis memo...")
        bs_summary = {
            sc: {
                "depletion_year": bs["depletion_year"],
                "broke_iron_reserve_year": bs["broke_iron_reserve_year"],
                "final_balance": bs["final_balance"],
                "year_table": bs["years"],
            }
            for sc, bs in agent_state["balance_sheets"].items()
        }
        reflect_resp = client.messages.create(
            model=REFLECTION_MODEL,
            max_tokens=4096,
            system=_REFLECTION_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Balance sheets computed:\n{json.dumps(bs_summary, indent=2, default=str)}\n\n"
                    f"Expenditure data:\n"
                    + json.dumps(
                        {y: d["total_talents"] for y, d in (agent_state.get("expenditure_data") or {}).items()},
                        default=str,
                    )
                    + "\n\nWrite the analysis memo."
                ),
            }],
        )
        cost_tracker.record(REFLECTION_MODEL, reflect_resp.usage.input_tokens, reflect_resp.usage.output_tokens)
        final_text = reflect_resp.content[0].text
        agent_state["memo"] = {"summary": final_text}

    if final_text:
        print("\n" + "=" * 60)
        print("ANALYSIS MEMO")
        print("=" * 60)
        print(final_text)

    if agent_state.get("exports"):
        print("\n[DONE] Exported files:")
        for p in agent_state["exports"]:
            print(f"  {p}")

    cost_tracker.print_summary()

    return agent_state, messages


def _tool_label(name: str, inputs: dict) -> str:
    """Short display label for a tool call."""
    labels = {
        "load_cached_extractions": lambda i: f"Loading cached extractions ({i.get('path', 'full_thucydides.csv')})",
        "filter_extractions": lambda i: f"Filtering extractions ({_fmt_filters(i)})",
        "calculate_expenditure": lambda i: f"Calculating expenditure (wage={i.get('wage_rate',1)} dr/day, crew={i.get('crew_size',200)}, months={i.get('campaign_months',8)})",
        "build_balance_sheet": lambda i: f"Building balance sheet — scenario: {i.get('scenario','?')}",
        "export_data": lambda i: f"Exporting {i.get('data_type','data')} to {i.get('output_path','output/')}",
        "summarize_findings": lambda i: f"Generating analysis memo",
        "cross_reference": lambda i: "Cross-referencing extractions",
        "search_passages": lambda i: f"Searching passages for: {i.get('keywords','')}",
        "ingest_text": lambda i: f"Ingesting text for {i.get('author','')}",
        "extract_passages": lambda i: f"Extracting passages (books={i.get('books','all')})",
    }
    fn = labels.get(name)
    return fn(inputs) if fn else f"{name}({inputs})"


def _fmt_filters(inputs: dict) -> str:
    parts = []
    for k in ("actor", "direction", "unit", "confidence", "book", "year_min", "year_max"):
        if k in inputs:
            parts.append(f"{k}={inputs[k]}")
    return ", ".join(parts) or "no filters"
