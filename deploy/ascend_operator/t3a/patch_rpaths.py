#!/usr/bin/env python3
"""t3a R-paths 补丁:修 4 个跨 skill import 的 parents[N] 路径反推(实体布局下必然断裂)。
用法: python3 patch_rpaths.py <t3a_root_or_workdir> — 就地修补 skills/ 下 4 个脚本。"""
import sys
from pathlib import Path

UPWARD_SEARCH = '''
# [R-paths] 跨 skill import 路径:上游用 parents[N] 反推仓库根,依赖「skill 待在原仓库树里」;
# 实体铺到 workdir/.claude/skills/ 后层级对不上必断(实测 ModuleNotFoundError)。
# 改为逐级向上搜 <name>/scripts,任何布局都能命中。
def _skill_scripts(name):
    here = Path(__file__).resolve()
    for base in here.parents:
        cand = base / name / "scripts"
        if cand.is_dir():
            return cand
    return here.parent

'''

def patch(path: Path, old: str, new: str) -> bool | None:
    t = path.read_text(encoding='utf-8')
    if old in t:
        path.write_text(t.replace(old, new, 1), encoding='utf-8')
        return True
    if new.split('\n')[0] in t or '_skill_scripts' in t:
        return None  # 已是新形态(幂等)
    return False


def main(root: Path) -> None:
    tr = root / "skills" / "tilelang2ascend-translator" / "scripts"
    de = root / "skills" / "tilelang2ascend-tilelang-designer" / "scripts"

    # 1) verification_ascendc.py: parents[5]/ops/ops-profiling → 向上搜
    p = tr / "verification_ascendc.py"
    ok = patch(
        p,
        '''_PERF_SCRIPTS = (
    Path(__file__).resolve().parents[5] / "ops" / "ops-profiling" / "scripts"
)''',
        UPWARD_SEARCH.strip() + '\n_PERF_SCRIPTS = _skill_scripts("ops-profiling")',
    )
    print({True: "✓", False: "✗", None: "✓ (已打)"}[ok], p)

    # 2) validate_ascendc_impl.py: parents[5]/ops/triton-op-verifier 的 import → 内联白名单
    p = tr / "validate_ascendc_impl.py"
    t = p.read_text(encoding='utf-8')
    if '_TRITON_SCRIPTS' not in t:
        print("✓ (已是自包含)", p)
    else:
        import os, re
        # 从上游源(canonical source)逐字提取,不依赖 t2a 的本地改本
        src_root = Path(os.environ.get("CANNBOT_SRC", "/home/docker/cannbot-skills"))
        upstream = src_root / "ops" / "triton-op-verifier" / "scripts" / "validate_triton_impl.py"
        m2 = re.search(r'(ALLOWED_TENSOR_METHODS = \{(?:.|\n)*?\n\})', upstream.read_text(encoding='utf-8'))
        old_block = re.search(r'# 从 validate_triton_impl 导入共享的 tensor 方法白名单(?:.|\n)*?from validate_triton_impl import ALLOWED_TENSOR_METHODS', t)
        if m2 and old_block:
            inline = ('# tensor 方法白名单:与 ops/triton-op-verifier/scripts/validate_triton_impl.py 同源(逐字)。\n'
                      '# [R-paths] 上游用 parents[5] 反推仓库根去 import,实体铺出仓库树后必断 → 内联自包含。\n'
                      + m2.group(1))
            t = t.replace(old_block.group(0), inline, 1)
            p.write_text(t, encoding='utf-8')
            print("✓", p)
        else:
            print("✗ 内联白名单定位失败", p)

    # 3)+4) designer 两个脚本: parents[1]/tilelang2ascend-translator → 向上搜
    for name in ("verification_tilelang.py", "validate_tilelang_impl.py"):
        p = de / name
        ok = patch(
            p,
            'Path(__file__).resolve().parents[1] / "tilelang2ascend-translator" / "scripts"',
            '_skill_scripts("tilelang2ascend-translator")',
        )
        if ok:
            t = p.read_text(encoding='utf-8')
            if '_skill_scripts' not in t.split('_skill_scripts("tilelang2ascend-translator")')[0]:
                t = t.replace('SCRIPT_DIR = Path(__file__).resolve().parent',
                              'SCRIPT_DIR = Path(__file__).resolve().parent\n' + UPWARD_SEARCH, 1)
                p.write_text(t, encoding='utf-8')
        print({True: "✓", False: "✗", None: "✓ (已打)"}[ok], p)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
