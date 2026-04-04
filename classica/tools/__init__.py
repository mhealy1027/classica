"""Classica tools — composable functions for the extraction pipeline."""

from classica.tools.ingest import ingest_text
from classica.tools.extract import extract_passage
from classica.tools.search import search_passages
from classica.tools.cross_reference import cross_reference
from classica.tools.export import export_data
from classica.tools.summarize import summarize_findings

__all__ = [
    "ingest_text",
    "extract_passage",
    "search_passages",
    "cross_reference",
    "export_data",
    "summarize_findings",
]
