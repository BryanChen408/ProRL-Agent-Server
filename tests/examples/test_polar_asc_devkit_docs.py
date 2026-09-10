from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[2]
prepare = runpy.run_path(str(ROOT / "deploy/ascend_operator/prepare_asc_devkit_docs.py"))["prepare"]


def test_prepare_preserves_code_tables_and_source(tmp_path):
    src, dst = tmp_path / "sdk", tmp_path / "readable"
    (src / "docs").mkdir(parents=True)
    code = '```cpp\nconst char* s = " id=\\"keep\\"";  \n\n\nauto x = Foo<T>();\n```'
    pre = '<pre class="screen">x &lt; y;  \n</pre>'
    complex_table = '<table><tr><td rowspan="2">A2</td><td>√</td></tr><tr><td>×</td></tr></table>'
    linked_table = '<table><tr><td><a href="other.md">constraints</a><img src="formula.png"></td></tr></table>'
    linked_text_table = '<table id="redundant"><tr><td><a href="other.md">constraints</a></td></tr></table>'
    markdown_table = '| mode | 0: nearest<br>1: toward zero |'
    text = '\n\n'.join([
        '# API<a name="anchor"></a>',
        '<table><tr><th>Product</th><th>Support</th></tr><tr><td>Atlas A2</td><td>√</td></tr></table>',
        code, pre, complex_table, linked_table, linked_text_table, markdown_table,
        '`Foo<T>(" id=\\"keep\\"")`', 'POLAR_PRESERVED_DOC_BLOCK_0_END',
    ]) + '\n'
    (src / "docs/api.md").write_text(text)
    crlf_code = b'```cpp\r\nFoo<T>();  \r\n```\r'
    (src / "docs/crlf.md").write_bytes(b'# CRLF\r\n' + crlf_code + b'\n')
    (src / "kernel.cpp").write_bytes(b"\x00unchanged\xff")
    (src / "A5_ONLY_STUBS.txt").write_text("existing stub list\n")
    original = {p.relative_to(src): p.read_bytes() for p in src.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        prepare(src, src / "nested")
    prepare(src, dst)
    result = (dst / "docs/api.md").read_text()
    assert '<a name="anchor">' not in result
    assert '| Atlas A2 | √ |' in result
    assert '<table><tr><td><a href="other.md">constraints</a></td></tr></table>' in result
    for block in (code, pre, complex_table, linked_table, markdown_table, '`Foo<T>(" id=\\"keep\\"")`'):
        assert block in result
    assert result.count('POLAR_PRESERVED_DOC_BLOCK_0_END') == 1
    assert crlf_code in (dst / "docs/crlf.md").read_bytes()
    assert {p.relative_to(src): p.read_bytes() for p in src.rglob('*') if p.is_file()} == original
    assert (dst / "kernel.cpp").read_bytes() == original[Path("kernel.cpp")]
    assert (dst / "A5_ONLY_STUBS.txt").read_bytes() == original[Path("A5_ONLY_STUBS.txt")]
    with pytest.raises(FileExistsError):
        prepare(src, dst)
