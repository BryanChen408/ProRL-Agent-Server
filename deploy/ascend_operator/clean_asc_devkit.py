#!/usr/bin/env python3
"""Produce an A2/A3-trimmed copy of asc-devkit: stub out A5-only(950) API pages.

Most conservative policy:
  - A page is stubbed ONLY when its 产品支持情况 table explicitly says
    950 √, A2 ×, A3 ×  (A5-only). Anything unparseable / table-missing /
    ambiguous is KEPT byte-identical.
  - Everything else (all kept pages, examples/, impl/, include/, tests/,
    non-md files, directory structure, filenames) is copied byte-identical.
  - Stub keeps the original filename and the page's first '# ' heading, so
    find/grep discovery and <a href> targets all keep working.

    python3 clean_asc_devkit.py --src /home/docker/asc-devkit-9.0.0 \
        --dst /home/docker/asc-devkit-9.0.0-a2a3
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

SUPPORT_RE = re.compile(r"##\s*产品支持情况(.*?)(?=\n##|\Z)", re.S)
ANCHOR_RE = re.compile(r'<a name="[^"]*"></a>')
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
A2_RE = re.compile(r"Atlas A2[^√×xX]{0,120}?[√×xX]")
A3_RE = re.compile(r"Atlas A3[^√×xX]{0,120}?[√×xX]")
A5_RE = re.compile(r"Ascend 950[^√×xX]{0,80}?[√×xX]")

STUB_NO_EQUIV = (
    "> ⚠️ 本接口仅支持 Ascend 950 系列(A5),Atlas A2/A3 平台(如 910B)无等价接口。\n"
    "> 需要相同语义时换算法路径(用 A2/A3 支持的基础向量指令自行实现);\n"
    "> 可用 `find \"$ASC_DEVKIT_DIR/docs/zh/api/\" -name \"<关键词>*.md\"` 查 A2/A3 支持(√)的接口。\n"
)

# 语义替代策展表:stub 页 -> (目标页, 用法提示)。目标是「读 stub 这一跳直接落到
# 正确文档上,发现成本为零」。仅收语义不等价的页;命名变体(Add-20→Add)走自动同族,
# 950 独有特性(AtomicAdd 等)走 STUB_NO_EQUIV。生成时校验目标页存在且未 stub。
CURATED: dict[str, tuple[str, str]] = {
    "Log.md":     ("Ln.md", "自然对数"),
    "Log2.md":    ("Ln.md", "log2(x) = Ln(x) * 0.6931"),
    "Log10.md":   ("Ln.md", "log10(x) = Ln(x) * 0.4343"),
    "log1pf.md":  ("Ln.md", "log1p(x) = Ln(1+x),先 Adds 加 1"),
    "hlog.md":    ("Ln.md", "自然对数"),
    "hexp.md":    ("Exp.md", "指数"),
    "Divs.md":    ("Div.md", "张量除法;标量除法用 Muls 乘倒数"),
    "Neg.md":     ("Muls.md", "取负 = Muls(x, -1)"),
    "AbsSub.md":  ("Sub.md", "先 Sub 再 Abs(两步)"),
    "sqrt-1.md":  ("Sqrt.md", "平方根"),
    "hsqrt.md":   ("Sqrt.md", "平方根"),
    "BitwiseAnd.md": ("And.md", "按位与"),
    "BitwiseOr.md":  ("Or.md", "按位或"),
    "BitwiseNot.md": ("Not.md", "按位非"),
}


def _verdict(rx: re.Pattern, txt: str) -> bool | None:
    m = rx.search(txt)
    if not m:
        return None
    return m.group(0)[-1] == "√"


def is_a5_only(md_text: str) -> bool:
    """True only on an explicit 950√ / A2× / A3× verdict. Doubt -> False."""
    m = SUPPORT_RE.search(md_text)
    if not m:
        return False
    txt = WS_RE.sub(" ", TAG_RE.sub(" ", ANCHOR_RE.sub("", m.group(1))))
    a5, a2, a3 = _verdict(A5_RE, txt), _verdict(A2_RE, txt), _verdict(A3_RE, txt)
    return a5 is True and a2 is False and a3 is False


def _stem(page: str) -> str:
    return re.split(r"[-(（]", page[:-3] if page.endswith(".md") else page)[0]


def _find_a2_sibling(page: str, kept_pages: set[str]) -> str | None:
    """同词干、未被 stub 的页面:优先完全同词干的基础页(Abs-15→Abs.md),否则最短名。"""
    stem = _stem(page)
    cands = [p for p in kept_pages if p != page and _stem(p) == stem]
    if not cands:
        return None
    exact = [p for p in cands if p == f"{stem}.md"]
    return exact[0] if exact else min(cands, key=len)


def make_stub(md_text: str, page: str, kept_pages: set[str]) -> str:
    heading = next(
        (ln.strip() for ln in md_text.splitlines() if ln.startswith("# ")), "# API"
    )
    heading = ANCHOR_RE.sub("", heading)
    target, hint = None, None
    if page in CURATED:
        t, h = CURATED[page]
        if t in kept_pages:
            target, hint = t, h
    if target is None:
        t = _find_a2_sibling(page, kept_pages)
        if t:
            target, hint = t, "同族 A2/A3 可用变体"
    if target is None:
        return f"{heading}\n\n{STUB_NO_EQUIV}"
    return (
        f"{heading}\n\n"
        f"> ⚠️ 本页接口仅支持 Ascend 950 系列(A5),Atlas A2/A3 平台(如 910B)不可用。\n"
        f"> A2/A3 请改读:`docs/zh/api/{target}`({hint})。\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()

    src, dst = args.src, args.dst
    if not src.is_dir():
        raise SystemExit(f"src not a directory: {src}")
    if dst.exists():
        raise SystemExit(f"dst already exists, refusing to overwrite: {dst}")

    # 1) full byte-identical copy
    shutil.copytree(src, dst, symlinks=True)

    # 2) stub A5-only pages inside docs/ only。先按目录分组判定,再按
    #    「同目录未 stub 页集合」生成带指针的 stub(指针 = 同目录相对路径,两棵树各自独立)。
    from collections import defaultdict

    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for md in sorted(dst.glob("docs/**/*.md")):
        by_dir[md.parent].append(md)

    stubbed: list[str] = []
    for parent, pages in by_dir.items():
        texts = {p: p.read_text(encoding="utf-8", errors="ignore") for p in pages}
        stub_set = {p for p, t in texts.items() if is_a5_only(t)}
        kept_pages = {p.name for p in pages if p not in stub_set}
        for p in sorted(stub_set):
            p.write_text(make_stub(texts[p], p.name, kept_pages), encoding="utf-8")
            stubbed.append(p.relative_to(dst).as_posix())

    print(f"src={src}")
    print(f"dst={dst}")
    print(f"stubbed A5-only pages: {len(stubbed)}")
    for rel in stubbed[:10]:
        print(f"  {rel}")
    if len(stubbed) > 10:
        print(f"  …({len(stubbed) - 10} more)")
    (dst / "A5_ONLY_STUBS.txt").write_text(
        "# pages stubbed as A5-only (950√, A2×, A3×)\n" + "\n".join(stubbed) + "\n",
        encoding="utf-8",
    )
    print(f"stub list written to {dst}/A5_ONLY_STUBS.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
