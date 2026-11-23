#!/usr/bin/env python3
"""Prepare a separate SDK copy, removing only redundant documentation markup."""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
MARKUP = re.compile(
    r"(?P<protected><!--.*?(?:-->|\Z)"
    r"|<(?P<block>pre|code|script|style|math|svg)\b[^>]*>.*?(?:</(?P=block)\s*>|\Z)"
    r"|(?P<inline>`+)(?!`).*?(?<!`)(?P=inline)(?!`)"
    r"|^(?: {4}|\t)[^\n]*(?:\n|\Z))"
    r"|(?P<parent><(?P<parent_tag>p|span|td|th|tr|table|li|ul|h[1-4])\b"
    r"(?:[^\"'<>]|\"[^\"]*\"|'[^']*')*>)"
    r"(?P<parent_gap>[ \t\r\n]*)"
    r"(?P<child_anchor><a[ \t]+name=(?P<child_quote>[\"'])"
    r"(?P<child_name>[^\"'<>\r\n]+)(?P=child_quote)>[ \t]*</a>)"
    r"|(?P<duplicate>(?P<anchor><a[ \t]+name=(?P<quote>[\"'])[^\"'<>\r\n]+(?P=quote)>[ \t]*</a>)"
    r"(?P<gap>[ \t\r\n]*)(?-i:(?P=anchor)))"
    r"|(?P<tag><(?:table|thead|tbody|tfoot|tr|th|td|col|colgroup)\b"
    r"(?:[^\"'<>]|\"[^\"]*\"|'[^']*')*>)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
ATTRIBUTE = re.compile(
    r"(?P<space>\s+)(?P<name>[\w:-]+)\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)')"
)


def normalize_markup(text: str) -> str:
    """Keep source lines, code, anchor targets and semantic table attributes intact."""
    def attribute(match: re.Match) -> str:
        name = match["name"].lower()
        value = match["double"] if match["double"] is not None else match["single"]
        # Arbitrary CSS/classes can carry meaning; only this known table class is redundant.
        if name in {"align", "valign", "width", "height"} or (name == "class" and value == "cellrowborder"):
            return re.sub(r"[^\r\n]", "", match[0])
        return match[0]

    def replace(match: re.Match) -> str:
        if match["parent"]:
            parent = match["parent"]
            ids = [a["double"] if a["double"] is not None else a["single"]
                   for a in ATTRIBUTE.finditer(parent) if a["name"].lower() == "id"]
            # The parent already supplies the exact target; leave all other anchors alone.
            redundant = ids == [match["child_name"]] and not parent.rstrip().endswith("/>")
            if match["parent_tag"].lower() in {"td", "th", "tr", "table"}:
                parent = ATTRIBUTE.sub(attribute, parent)
            return parent + match["parent_gap"] + ("" if redundant else match["child_anchor"])
        if match["tag"]:
            return ATTRIBUTE.sub(attribute, match[0])
        if match["duplicate"]:
            return match["anchor"] + match["gap"]
        return match[0]

    def clean(chunk: str) -> str:
        # Repeat for three or more identical adjacent anchors.
        while True:
            result = MARKUP.sub(replace, chunk)
            if result == chunk:
                return result
            chunk = result

    result, pending = [], []
    fence = None
    for line in text.splitlines(keepends=True):
        match = FENCE.match(line.rstrip("\r\n"))
        if fence is not None:
            result.append(line)
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
        elif match:
            result.append(clean("".join(pending)))
            pending.clear()
            fence = match[1]
            result.append(line)
        else:
            pending.append(line)
    result.append(clean("".join(pending)))
    return "".join(result)


def prepare(src: Path, dst: Path) -> None:
    src, dst = src.resolve(), dst.resolve()
    if not (src / "docs").is_dir() or src == dst or src in dst.parents:
        raise ValueError("src must contain docs/; dst must be outside src")
    shutil.copytree(src, dst, symlinks=True)  # Refuse existing destinations.
    changed = before = after = 0
    for path in sorted((dst / "docs").rglob("*.md")):
        if path.is_symlink():
            continue
        original = path.read_bytes()
        text = original.decode("utf-8")
        cleaned = normalize_markup(text)
        if re.findall(r"\r\n|\r|\n", text) != re.findall(r"\r\n|\r|\n", cleaned):
            raise RuntimeError(f"Changed source lines: {path}")
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
