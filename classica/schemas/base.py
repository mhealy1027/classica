"""Schema loader — reads YAML schema definitions."""

from pathlib import Path
from typing import Any, Optional

import yaml


def load_schema(schema_path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML extraction schema.

    Args:
        schema_path: Path to the YAML schema file.

    Returns:
        Dict with 'name', 'description', 'fields' keys.
        Each field has 'name', 'type', 'description', and optional 'enum'.

    Raises:
        FileNotFoundError: If schema file doesn't exist.
        ValueError: If schema is missing required keys.
    """
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    if "name" not in schema:
        raise ValueError("Schema must have a 'name' field")
    if "fields" not in schema or not schema["fields"]:
        raise ValueError("Schema must have a non-empty 'fields' list")

    for field in schema["fields"]:
        if "name" not in field or "type" not in field:
            raise ValueError(
                f"Each field must have 'name' and 'type'. Got: {field}"
            )

    return schema


def render_schema_for_prompt(schema: dict[str, Any]) -> str:
    """Render a schema into a human-readable string for inclusion in prompts.

    Args:
        schema: Loaded schema dict.

    Returns:
        Formatted string describing the schema fields.
    """
    lines = [f"Schema: {schema['name']}", ""]
    if schema.get("description"):
        lines.append(schema["description"].strip())
        lines.append("")

    lines.append("Fields:")
    for field in schema["fields"]:
        type_str = field["type"]
        if "enum" in field:
            type_str += f" (one of: {', '.join(str(v) for v in field['enum'])})"
        nullable = " (nullable)" if field.get("nullable") else ""
        desc = field.get("description", "").strip()
        lines.append(f"  - {field['name']}: {type_str}{nullable}")
        if desc:
            lines.append(f"    {desc}")

    return "\n".join(lines)
