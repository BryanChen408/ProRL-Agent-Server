from pathlib import Path
import runpy

import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'deploy/ascend_operator/clean_asc_devkit.py'))

def support(a5='支持', a2='不支持', a3='不支持'):
    return f'# API\n\n## 产品支持情况\n- Ascend 950PR&950DT系列产品：{a5}\n- Atlas A2系列产品：{a2}\n- Atlas A3系列产品：{a3}\n\n## 功能说明\ntext\n'

@pytest.mark.parametrize('values,expected', [
    (('支持','不支持','不支持'),True), (('√','×','×'),True),
    (('支持','支持','不支持'),False), (('不支持','不支持','不支持'),False),
    (('支持','待定','不支持'),False),
])
def test_support_formats(values, expected):
    assert M['is_a5_only'](support(*values)) is expected

def test_missing_and_conflicting_support_is_not_filtered():
    assert not M['is_a5_only']('# API\nAtlas A2：不支持')
    assert not M['is_a5_only'](support().replace('- Atlas A2系列产品：不支持\n',''))
    assert not M['is_a5_only'](support().replace('## 功能说明','- Atlas A2系列产品：支持\n## 功能说明'))
    assert M['is_a2a3_supported'](support(a2='支持',a3='支持'))
    assert not M['is_a2a3_supported'](support(a2='支持'))

def test_overload_subsections_are_checked_together():
    text = support().replace('- Ascend', '### 第一种重载\n- Ascend', 1)
    assert M['is_a5_only'](text)
    text = text.replace('## 功能说明', '### 第二种重载\nAtlas A2：支持\n## 功能说明')
    assert not M['is_a5_only'](text)

def test_nested_hint_uses_verified_full_path_and_correct_log2_formula():
    target='docs/zh/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Ln.md'
    text=M['make_stub'](support(), 'docs/zh/api/reg/Log2.md', {target})
    assert target in text and 'Ln(x) / ln(2)' in text and '* 0.6931' not in text
    assert 'docs/zh/api/Ln.md' not in text
    ambiguous=M['make_stub'](support(), 'docs/zh/api/reg/Log2.md', {target,'docs/zh/api/other/Ln.md'})
    assert target in ambiguous and 'docs/zh/api/other/Ln.md' in ambiguous
    assert '非唯一方案' in ambiguous and '不保证可直接替换' in ambiguous
    assert '无等价接口' not in ambiguous

def test_only_same_name_is_used_for_non_curated_hint():
    assert '可参考' not in M['make_stub'](support(), 'docs/zh/api/Add-20.md', {'docs/zh/api/Add.md'})

def test_open_guidance_without_candidates():
    text=M['make_stub'](support(), 'docs/zh/api/AtomicAdd.md', set())
    assert '组合基础 AscendC 指令' in text and '任务规则允许' in text
    assert '未列出候选不代表没有可行实现' in text

def test_curated_candidates_do_not_hide_same_name_alternatives():
    paths={'docs/zh/api/one/Neg.md', 'docs/zh/api/two/Muls.md'}
    text=M['make_stub'](support(), 'docs/zh/api/reg/Neg.md', paths)
    assert all(p in text for p in paths)
