#!/usr/bin/env python3
"""Prepare a separate asc-devkit copy with CANNBot's Markdown cleanup, once per SDK."""
from __future__ import annotations

import argparse
import re
import runpy
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLEANER = ROOT / "operator_runtime_t2a/skills/ascendc-docs-search/scripts/clean_markdown.py"
PROTECTED = re.compile(
    r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ ]{0,3}(?P=fence)[ \t]*\r?$"
    r"|<pre\b[^>]*>.*?</pre>"
    r"|<table\b[^>]*>.*?</table>"
    r"|^[ \t]*\|[^\n]*\|[ \t]*$"
    r"|(?P<inline>`+)[^`\n]+(?P=inline)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)


def prepare(src: Path, dst: Path) -> None:
    src, dst = src.resolve(), dst.resolve()
    if not (src / "docs").is_dir() or src == dst or src in dst.parents:
        raise ValueError("src must contain docs/; dst must be outside src")
    upstream = runpy.run_path(str(CLEANER))
    cleaner = upstream["clean_markdown_file"]
    shutil.copytree(src, dst, symlinks=True)  # Refuse existing destinations.
    changed = before = after = 0
    for path in sorted((dst / "docs").rglob("*.md")):
        if path.is_symlink():
            continue
        original = path.read_bytes()
        text = original.decode("utf-8")
        saved: list[str] = []
        prefix = "POLAR_PRESERVED_DOC_BLOCK_"
        while prefix in text:
            prefix += "_"

        def protect(match: re.Match) -> str:
            block = match.group()
            if block.lower().startswith("<table"):
                # ponytail: retain HTML structure for spans/links/media;
                # extend upstream's table parser only if these dominate context.
                if re.search(r"\b(?:rowspan|colspan)\s*=|<a\b[^>]*\bhref\s*=|<(?:img|pre|code|math|svg)\b|\|", block, re.I) is None:
                    return block
                if re.search(r"<(?:img|pre|code|math|svg)\b", block, re.I) is None:
                    for name in ("remove_anchor_tags", "remove_paragraph_id_tags", "remove_html_attributes"):
                        block = upstream[name](block)
            saved.append(block)
            return f"{prefix}{len(saved) - 1}_END"

        path.write_text(PROTECTED.sub(protect, text), encoding="utf-8")
        if not cleaner(str(path), backup=False, quiet=True):
            raise RuntimeError(f"CANNBot cleanup failed: {path}")
        cleaned = path.read_text(encoding="utf-8")
        for i, block in enumerate(saved):
            marker = f"{prefix}{i}_END"
            if cleaned.count(marker) != 1:
                raise RuntimeError(f"Lost protected content: {path}")
            cleaned = cleaned.replace(marker, block)
        result = cleaned.encode("utf-8")
        path.write_bytes(result)
        before += len(original)
        after += len(result)
        changed += original != result
    print(f"dst={dst}; changed={changed}; docs_bytes={before}->{after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.src, args.dst)
