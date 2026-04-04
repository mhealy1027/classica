"""Tool: fetch and parse texts from Perseus Digital Library."""

from pathlib import Path
from typing import Optional

import requests
from lxml import etree

from classica.config import TEXTS_DIR
from classica.registry import get_author, get_perseus_urls


def _fetch_or_cache(url: str, cache_path: Path, refresh: bool) -> bytes:
    """Download a URL or read from cache."""
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()

    print(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)
    return resp.content


def _parse_tei_xml(xml_bytes: bytes) -> list[dict]:
    """Parse TEI-XML into a list of {book, chapter, text} dicts.

    Handles both common Perseus XML structures:
      - div[@type='book']/div[@type='chapter']
      - div[@type='textpart'][@subtype='book']/div[@type='textpart'][@subtype='chapter']
    """
    tree = etree.fromstring(xml_bytes)

    # Remove namespace prefixes for easier XPath
    for elem in tree.iter():
        if elem.tag and isinstance(elem.tag, str) and "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
        # Also strip namespace from attributes
        new_attrib = {}
        for k, v in elem.attrib.items():
            if "}" in k:
                k = k.split("}", 1)[1]
            new_attrib[k] = v
        elem.attrib.clear()
        elem.attrib.update(new_attrib)

    passages = []

    # Strategy 1: div[@type='textpart'][@subtype='book']/div[@type='textpart'][@subtype='chapter']
    books = tree.xpath(".//div[@type='textpart'][@subtype='book']")
    if books:
        for book_div in books:
            book_n = book_div.get("n", "0")
            chapters = book_div.xpath("./div[@type='textpart'][@subtype='chapter']")
            for ch_div in chapters:
                ch_n = ch_div.get("n", "0")
                text = " ".join(ch_div.itertext()).strip()
                # Collapse whitespace
                text = " ".join(text.split())
                passages.append({
                    "book": int(book_n),
                    "chapter": int(ch_n),
                    "text": text,
                })
        return passages

    # Strategy 2: div[@type='book']/div[@type='chapter']
    books = tree.xpath(".//div[@type='book']")
    if books:
        for book_div in books:
            book_n = book_div.get("n", "0")
            chapters = book_div.xpath("./div[@type='chapter']")
            for ch_div in chapters:
                ch_n = ch_div.get("n", "0")
                text = " ".join(ch_div.itertext()).strip()
                text = " ".join(text.split())
                passages.append({
                    "book": int(book_n),
                    "chapter": int(ch_n),
                    "text": text,
                })
        return passages

    return passages


def _merge_passages(
    greek_passages: list[dict], english_passages: list[dict]
) -> list[dict]:
    """Merge Greek and English passages by (book, chapter)."""
    eng_map = {(p["book"], p["chapter"]): p["text"] for p in english_passages}

    merged = []
    for gp in greek_passages:
        key = (gp["book"], gp["chapter"])
        merged.append({
            "book": gp["book"],
            "chapter": gp["chapter"],
            "greek_text": gp["text"],
            "english_text": eng_map.get(key, ""),
        })

    # Add any English-only passages
    greek_keys = {(p["book"], p["chapter"]) for p in greek_passages}
    for ep in english_passages:
        key = (ep["book"], ep["chapter"])
        if key not in greek_keys:
            merged.append({
                "book": ep["book"],
                "chapter": ep["chapter"],
                "greek_text": "",
                "english_text": ep["text"],
            })

    merged.sort(key=lambda p: (p["book"], p["chapter"]))
    return merged


def ingest_text(
    author: str,
    work: Optional[str] = None,
    editions: Optional[list[str]] = None,
    refresh: bool = False,
) -> dict:
    """Fetch and parse Greek + English texts from Perseus.

    Args:
        author: Author name (e.g. 'thucydides').
        work: Work identifier (unused — resolved from registry).
        editions: Edition list (unused — resolved from registry).
        refresh: If True, re-download even if cached.

    Returns:
        Dict with keys: author, work, total_books, total_chapters,
        passages (list of {book, chapter, greek_text, english_text}).
    """
    author_info = get_author(author)
    urls = get_perseus_urls(author)

    tlg_a = author_info["tlg_author"]
    tlg_w = author_info["tlg_work"]

    print(f"Ingesting {author} ({tlg_a}.{tlg_w})...")

    # Fetch Greek
    greek_cache = TEXTS_DIR / f"{tlg_a}.{tlg_w}.greek.xml"
    greek_xml = _fetch_or_cache(urls["greek"], greek_cache, refresh)
    greek_passages = _parse_tei_xml(greek_xml)
    print(f"  Greek: {len(greek_passages)} passages parsed")

    # Fetch English
    eng_cache = TEXTS_DIR / f"{tlg_a}.{tlg_w}.english.xml"
    eng_xml = _fetch_or_cache(urls["english"], eng_cache, refresh)
    eng_passages = _parse_tei_xml(eng_xml)
    print(f"  English: {len(eng_passages)} passages parsed")

    # Merge
    passages = _merge_passages(greek_passages, eng_passages)

    books = sorted(set(p["book"] for p in passages))
    total_books = len(books)
    total_chapters = len(passages)

    print(f"  Merged: {total_books} books, {total_chapters} chapters")

    return {
        "author": author,
        "work": f"{tlg_a}.{tlg_w}",
        "total_books": total_books,
        "total_chapters": total_chapters,
        "passages": passages,
    }
