"""Author/work TLG code registry.

Maps human-readable author names to their Perseus TLG identifiers
and available editions.
"""

from typing import Optional

REGISTRY: dict[str, dict] = {
    "thucydides": {
        "tlg_author": "tlg0003",
        "tlg_work": "tlg001",
        "editions": {
            "greek": "perseus-grc2",
            "english": "perseus-eng6",
        },
    },
    "herodotus": {
        "tlg_author": "tlg0016",
        "tlg_work": "tlg001",
        "editions": {
            "greek": "perseus-grc1",
            "english": "perseus-eng1",
        },
    },
    "xenophon": {
        "tlg_author": "tlg0032",
        "tlg_work": "tlg006",
        "editions": {
            "greek": "perseus-grc2",
            "english": "perseus-eng1",
        },
    },
    "homer": {
        "tlg_author": "tlg0012",
        "tlg_work": "tlg001",
        "editions": {
            "greek": "perseus-grc2",
            "english": "perseus-eng2",
        },
    },
}


def get_author(name: str) -> dict:
    """Look up an author by name.

    Args:
        name: Lowercase author name (e.g. 'thucydides').

    Returns:
        Dict with tlg_author, tlg_work, and editions.

    Raises:
        KeyError: If author is not in the registry.
    """
    key = name.lower().strip()
    if key not in REGISTRY:
        available = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(f"Unknown author '{name}'. Available: {available}")
    return REGISTRY[key]


def get_perseus_urls(name: str) -> dict[str, str]:
    """Build Perseus raw GitHub URLs for an author's editions.

    Args:
        name: Lowercase author name.

    Returns:
        Dict with 'greek' and 'english' keys mapping to full URLs.
    """
    from classica.config import PERSEUS_BASE_URL

    author = get_author(name)
    tlg_a = author["tlg_author"]
    tlg_w = author["tlg_work"]
    base = f"{PERSEUS_BASE_URL}{tlg_a}/{tlg_w}/"

    urls = {}
    for lang, edition in author["editions"].items():
        filename = f"{tlg_a}.{tlg_w}.{edition}.xml"
        urls[lang] = base + filename
    return urls


def list_authors() -> list[str]:
    """Return all registered author names."""
    return sorted(REGISTRY.keys())
