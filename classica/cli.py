"""CLI entry point for Classica."""

import argparse
import json
import sys
from pathlib import Path

from classica.config import TEXTS_DIR, OUTPUT_DIR, CLAUDE_MODEL, CLAUDE_SONNET_MODEL, cost_tracker


def _resolve_model(args: argparse.Namespace) -> str:
    """Resolve --model shorthand to full model ID."""
    name = getattr(args, "model", None)
    if name == "sonnet":
        return CLAUDE_SONNET_MODEL
    return CLAUDE_MODEL


def cmd_ingest(args: argparse.Namespace) -> None:
    """Handle the 'ingest' subcommand."""
    from classica.tools.ingest import ingest_text

    result = ingest_text(
        author=args.author,
        refresh=args.refresh,
    )
    print(f"\nIngested {result['author']}:")
    print(f"  Work: {result['work']}")
    print(f"  Books: {result['total_books']}")
    print(f"  Chapters: {result['total_chapters']}")


def cmd_extract(args: argparse.Namespace) -> None:
    """Handle the 'extract' subcommand."""
    from classica.tools.ingest import ingest_text
    from classica.tools.extract import extract_passages_batched
    from classica.tools.export import export_data
    from classica.tools.cross_reference import cross_reference
    from classica.tools.summarize import summarize_findings
    from classica.schemas.base import load_schema

    model = _resolve_model(args)

    # Load schema
    schema = load_schema(args.schema)
    print(f"Schema: {schema['name']}")
    print(f"Model: {model}")

    # Ingest text
    data = ingest_text(author=args.author)

    # Filter passages by book if specified
    passages = data["passages"]
    if args.books:
        book_nums = [int(b) for b in args.books.split(",")]
        passages = [p for p in passages if p["book"] in book_nums]
        print(f"Filtering to books: {book_nums} ({len(passages)} chapters)")

    # Batched extraction with optional keyword pre-filter
    skip_filter = getattr(args, "skip_filter", False)
    all_extractions = extract_passages_batched(
        passages, schema, model=model, skip_filter=skip_filter,
    )

    print(f"\nTotal extractions: {len(all_extractions)}")

    # Cross-reference
    xref = cross_reference(all_extractions)
    if xref["duplicates"]:
        print(f"Potential duplicates: {len(xref['duplicates'])}")
    if xref["inconsistencies"]:
        print(f"Inconsistencies: {len(xref['inconsistencies'])}")

    # Export
    output_path = args.output or str(OUTPUT_DIR / "extractions.csv")
    fmt = "json" if output_path.endswith(".json") else "csv"
    export_result = export_data(all_extractions, format=fmt, output_path=output_path)
    print(f"\nExported {export_result['total_rows']} rows to {export_result['path']}")
    print(f"  Confidence: {export_result['by_confidence']}")

    # Summarize
    print("\nGenerating summary...")
    summary = summarize_findings(all_extractions, xref)
    print(f"\n{'='*60}")
    print("ANALYSIS MEMO")
    print(f"{'='*60}")
    print(summary["summary"])
    if summary["key_findings"]:
        print(f"\nKey Findings:")
        for f in summary["key_findings"]:
            print(f"  - {f}")
    if summary["uncertainties"]:
        print(f"\nUncertainties:")
        for u in summary["uncertainties"]:
            print(f"  - {u}")

    cost_tracker.print_summary()


def cmd_search(args: argparse.Namespace) -> None:
    """Handle the 'search' subcommand."""
    from classica.tools.ingest import ingest_text
    from classica.tools.search import search_passages

    data = ingest_text(author=args.author)
    keywords = [k.strip() for k in args.keywords.split(",")]

    greek_kw = None
    if args.greek_keywords:
        greek_kw = [k.strip() for k in args.greek_keywords.split(",")]

    results = search_passages(
        passages=data["passages"],
        keywords=keywords,
        greek_keywords=greek_kw,
    )

    print(f"\nFound {len(results)} matches:")
    for r in results:
        print(f"\n  Book {r['book']}, Ch {r['chapter']} [{r['language']}] — \"{r['keyword']}\"")
        for s in r["snippets"][:3]:
            print(f"    {s}")


def cmd_list(args: argparse.Namespace) -> None:
    """Handle the 'list' subcommand."""
    xml_files = list(TEXTS_DIR.glob("*.xml"))
    if not xml_files:
        print("No cached texts. Run 'python -m classica ingest --author <name>' first.")
        return

    print("Cached texts:")
    for f in sorted(xml_files):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")


def cmd_agent(args: argparse.Namespace) -> None:
    """Handle the 'agent' subcommand."""
    from classica.agent import run_agent, AGENT_MODEL, REFLECTION_MODEL
    import classica.agent as agent_module

    # Apply model override if specified
    model = _resolve_model(args)
    if model != CLAUDE_MODEL:
        agent_module.AGENT_MODEL = model
        agent_module.REFLECTION_MODEL = model

    if args.interactive:
        print("Classica Agent — Interactive Mode")
        print("Type your research question and press Enter. Ctrl+C to exit.")
        print("Agent state is shared across questions in this session.\n")
        agent_state = None
        messages = None
        try:
            while True:
                try:
                    question = input("Question: ").strip()
                except EOFError:
                    break
                if not question:
                    continue
                print()
                agent_state, messages = run_agent(question, agent_state, messages)
                print()
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
    else:
        run_agent(args.question)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="classica",
        description="Classica: AI-powered structured data extraction from ancient texts",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Fetch and parse texts from Perseus")
    p_ingest.add_argument("--author", required=True, help="Author name (e.g. thucydides)")
    p_ingest.add_argument("--refresh", action="store_true", help="Re-download even if cached")
    p_ingest.set_defaults(func=cmd_ingest)

    # extract
    p_extract = subparsers.add_parser("extract", help="Extract structured data from passages")
    p_extract.add_argument("--author", required=True, help="Author name")
    p_extract.add_argument("--schema", required=True, help="Path to YAML schema file")
    p_extract.add_argument("--books", default=None, help="Comma-separated book numbers (e.g. '2' or '1,2,3')")
    p_extract.add_argument("--output", default=None, help="Output file path")
    p_extract.add_argument("--model", choices=["haiku", "sonnet"], default="haiku",
                           help="Model to use: haiku (default, cheap) or sonnet (higher accuracy)")
    p_extract.add_argument("--skip-filter", action="store_true",
                           help="Skip keyword pre-filter and extract from all chapters")
    p_extract.set_defaults(func=cmd_extract)

    # search
    p_search = subparsers.add_parser("search", help="Search passages for keywords")
    p_search.add_argument("--author", required=True, help="Author name")
    p_search.add_argument("--keywords", required=True, help="Comma-separated keywords")
    p_search.add_argument("--greek-keywords", default=None, help="Comma-separated Greek keywords")
    p_search.set_defaults(func=cmd_search)

    # list
    p_list = subparsers.add_parser("list", help="List cached texts")
    p_list.set_defaults(func=cmd_list)

    # agent
    p_agent = subparsers.add_parser("agent", help="Run the AI research agent")
    p_agent_group = p_agent.add_mutually_exclusive_group(required=True)
    p_agent_group.add_argument("--question", help="Research question to answer")
    p_agent_group.add_argument(
        "--interactive", action="store_true",
        help="Interactive mode: run questions sequentially, sharing agent state",
    )
    p_agent.add_argument("--model", choices=["haiku", "sonnet"], default="haiku",
                         help="Model for planning/execution: haiku (default) or sonnet")
    p_agent.set_defaults(func=cmd_agent)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)
