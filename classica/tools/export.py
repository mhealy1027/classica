"""Tool: export extractions to CSV or JSON."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from classica.config import OUTPUT_DIR
from classica.tools.cross_reference import cross_reference


def export_data(
    extractions: list[dict],
    format: str = "csv",
    output_path: Optional[str] = None,
) -> dict:
    """Export extractions to CSV or JSON.

    Args:
        extractions: List of extraction dicts.
        format: Output format — 'csv' or 'json'.
        output_path: Optional output file path. Defaults to output/ directory.

    Returns:
        Dict with path, total_rows, and by_confidence breakdown.
    """
    if not extractions:
        return {
            "path": "",
            "total_rows": 0,
            "by_confidence": {"high": 0, "medium": 0, "low": 0},
        }

    # Run cross-reference to get duplicate flags
    xref = cross_reference(extractions)
    dup_ids = set()
    for dup in xref["duplicates"]:
        for eid in dup["extraction_ids"]:
            dup_ids.add(eid)

    # Build DataFrame
    df = pd.DataFrame(extractions)
    df["duplicate_flag"] = df.get("extraction_id", pd.Series()).apply(
        lambda x: x in dup_ids if pd.notna(x) else False
    )

    # Determine output path
    if output_path:
        path = Path(output_path)
    else:
        ext = ".csv" if format == "csv" else ".json"
        path = OUTPUT_DIR / f"extractions{ext}"

    path.parent.mkdir(parents=True, exist_ok=True)

    # Export
    if format == "json":
        records = df.to_dict(orient="records")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
    else:
        df.to_csv(path, index=False)

    # Confidence breakdown
    by_confidence = {"high": 0, "medium": 0, "low": 0}
    if "confidence" in df.columns:
        counts = df["confidence"].value_counts().to_dict()
        for k in by_confidence:
            by_confidence[k] = int(counts.get(k, 0))

    return {
        "path": str(path),
        "total_rows": len(df),
        "by_confidence": by_confidence,
    }
