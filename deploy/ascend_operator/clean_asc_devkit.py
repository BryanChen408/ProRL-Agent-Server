#!/usr/bin/env python3
"""Produce an A2/A3-trimmed copy of asc-devkit: stub out A5-only(950) API pages.

Most conservative policy:
  - A page is stubbed ONLY when its 产品支持情况 table explicitly says
    950 supported, A2/A3 unsupported (symbols or words)  (A5-only). Anything unparseable / table-missing /
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
from pathlib import Path

SUPPORT_RE = re.compile(r"##\s*产品支持情况(.*?)(?=\n##(?:\s|$)|\Z)", re.S)
ANCHOR_RE = re.compile(r'<a name="[^"]*"></a>')
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
A2_RE = re.compile(r"Atlas A2(?:(?!Atlas|Ascend).){0,120}?(不支持|支持|√|×|[xX](?=\s|$))")
A3_RE = re.compile(r"Atlas A3(?:(?!Atlas|Ascend).){0,120}?(不支持|支持|√|×|[xX](?=\s|$))")
A5_RE = re.compile(r"Ascend 950(?:(?!Atlas|Ascend).){0,120}?(不支持|支持|√|×|[xX](?=\s|$))")

STUB_NO_EQUIV = (
    "> 限制的是本页 API，不是算子的实现路线；以下方向和候选均非唯一方案。\n"
    "> 可继续搜索其他受支持接口、组合基础 AscendC 指令，或在任务规则允许时从 C++ host 调用内置 NPU 算子。\n"
    "> 候选仅确认文档标注支持 A2/A3，不保证可直接替换；需核对语义、dtype、shape、布局、边界行为和精度，再编译验证。\n"
    "> 可用 `find \"$ASC_DEVKIT_DIR/docs/zh/api/\" -name \"<关键词>*.md\"` 查产品支持情况明确标注 A2/A3 支持的接口。\n"
)

# List all explicitly supported matching pages as candidates, not guaranteed replacements.
CURATED: dict[str, tuple[str, str]] = {
    "Log.md":     ("Ln.md", "自然对数"),
    "Log2.md":    ("Ln.md", "log2(x) = Ln(x) / ln(2)，常数精度需匹配目标 dtype"),
    "Log10.md":   ("Ln.md", "log10(x) = Ln(x) / ln(10)，常数精度需匹配目标 dtype"),
    "log1pf.md":  ("Ln.md", "可研究 Ln(1+x) 的组合，但小 x 时直接加 1 会损失精度，不能视为数值等价替换"),
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
    verdicts = {m.group(1) in {"√", "支持"} for m in rx.finditer(txt)}
    return verdicts.pop() if len(verdicts) == 1 else None


def is_a5_only(md_text: str) -> bool:
    """True only on an explicit 950√ / A2× / A3× verdict. Doubt -> False."""
    m = SUPPORT_RE.search(md_text)
    if not m:
        return False
    txt = WS_RE.sub(" ", TAG_RE.sub(" ", ANCHOR_RE.sub("", m.group(1))))
    a5, a2, a3 = _verdict(A5_RE, txt), _verdict(A2_RE, txt), _verdict(A3_RE, txt)
    return a5 is True and a2 is False and a3 is False


def is_a2a3_supported(md_text: str) -> bool:
    m = SUPPORT_RE.search(md_text)
    if not m:
        return False
    txt = WS_RE.sub(" ", TAG_RE.sub(" ", ANCHOR_RE.sub("", m.group(1))))
    return _verdict(A2_RE, txt) is True and _verdict(A3_RE, txt) is True


def make_stub(md_text: str, page: str, supported_pages: set[str]) -> str:
    heading = next((ln.strip() for ln in md_text.splitlines() if ln.startswith("# ")), "# API")
    # Keep heading anchors; alternative links always use a verified full path.
    name = Path(page).name
    target_name, hint = CURATED.get(name, (name, "同名接口；需核对语义、dtype 和调用约束"))
    candidates = sorted(p for p in supported_pages if Path(p).name in {name, target_name} and p != page)
    links = "".join(
        f"> - `{p}`（{hint if Path(p).name == target_name else '同名接口；需核对语义、dtype 和调用约束'}）。\n"
        for p in candidates
    )
    return (
        f"{heading}\n\n"
        f"> ⚠️ 本页接口支持 Ascend 950，但不支持 Atlas A2/A3 平台(如 910B)。\n"
        f"{STUB_NO_EQUIV}"
        + (f"> A2/A3 可参考的候选文档（非穷尽）：\n{links}" if links else
           "> 未列出候选不代表没有可行实现，请继续搜索和核对。\n")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if src == dst or src in dst.parents:
        raise SystemExit("dst must be outside src")
    if not src.is_dir():
        raise SystemExit(f"src not a directory: {src}")
    if dst.exists():
        raise SystemExit(f"dst already exists, refusing to overwrite: {dst}")

    # 1) full byte-identical copy
    shutil.copytree(src, dst, symlinks=True)

    # Only explicit support verdicts can authorize filtering or replacement hints.
    pages = sorted(p for p in dst.glob("docs/**/*.md") if not p.is_symlink())
    texts = {p.relative_to(dst).as_posix(): p.read_text(encoding="utf-8") for p in pages}
    supported = {p for p, t in texts.items() if is_a2a3_supported(t)}
    stubbed = sorted(p for p, t in texts.items() if is_a5_only(t))
    for rel in stubbed:
        # Keep alternatives in the same language/compatibility tree.
        tree = "docs/zh/" if rel.startswith("docs/zh/") else "docs/api/"
        candidates = {p for p in supported if p.startswith(tree)}
        (dst / rel).write_text(make_stub(texts[rel], rel, candidates), encoding="utf-8")

    # 3) 修复 docs/api 树里的 Windows 反斜杠交叉引用(href="context\Xxx.md"
    #    → href="context/Xxx.md"):容器里顺着链接 Read 会 404,实测模型踩过。
    #    只动 href 属性内的反斜杠,其他字节不变。
    n_linkfix = 0
    link_re = re.compile(r'(href="[^"]*?)\\([^"]*?\.md")')
    for md in sorted(dst.glob("docs/**/*.md")):
        text = md.read_text(encoding="utf-8", errors="ignore")
        fixed, n = link_re.subn(r"\1/\2", text)
        if n:
            md.write_text(fixed, encoding="utf-8")
            n_linkfix += n

    print(f"src={src}")
    print(f"dst={dst}")
    print(f"stubbed A5-only pages: {len(stubbed)}")
    print(f"fixed backslash hrefs: {n_linkfix}")
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
