# Saphes

**Ancient Text Structured Extractor** - a configurable AI pipeline that transforms classical texts into analyst-ready structured data.

## Overview

Saphes is a Python-based CLI tool that extracts structured, tabular data from ancient texts hosted in the [Perseus Digital Library](http://www.perseus.tufts.edu/). Given a text and a user-defined extraction schema, it uses the Claude API to systematically parse each chapter, identify relevant data points, and output analyst-ready CSV files.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest a text from Perseus
python -m classica ingest --author thucydides --work history --editions grc2,eng6

# Run extraction with a schema
python -m classica extract --author thucydides --work history \
  --schema schemas/thucydides_finance.yaml --books 2 \
  --output output/book2_finance.csv

# Run extraction on all books
python -m classica extract --author thucydides --work history \
  --schema schemas/thucydides_finance.yaml \
  --output output/full_finance.csv

# List available cached texts
python -m classica list

# Validate output against known data
python -m classica validate --extracted output/book2_finance.csv \
  --known data/thesis_known_extractions.csv
```

## Architecture

```
classica/
├── classica/           # Python package
│   ├── cli.py          # CLI entry point (argparse)
│   ├── ingest.py       # Perseus XML fetching & parsing
│   ├── extract.py      # Claude API extraction logic
│   ├── schemas/        # Schema classes
│   │   ├── base.py     # Base schema class
│   │   └── thucydides_finance.py
│   ├── prompts/        # Prompt templates
│   │   └── extraction.py
│   └── export.py       # CSV/Excel output
├── texts/              # Cached Perseus XML files
├── output/             # Extraction results
├── schemas/            # User-defined YAML schemas
├── requirements.txt
└── README.md
```

## Schema Configuration

Schemas are defined in YAML files so researchers can create new extraction templates without touching Python code. See `schemas/thucydides_finance.yaml` for an example.

## Proof of Concept

The initial target is Thucydides' *History of the Peloponnesian War*, extracting financial and military data. The architecture is designed to be text-agnostic and schema-configurable for any Perseus text.

## Tech Stack

- **Python 3.11+**
- **Anthropic SDK** — Claude API for extraction
- **lxml** — TEI-XML parsing
- **pandas** — Data manipulation and CSV export
- **PyYAML** — Schema configuration files

## License

MIT
