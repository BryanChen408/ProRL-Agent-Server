from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[2]
module = runpy.run_path(str(ROOT / "deploy/ascend_operator/prepare_asc_devkit_docs.py"))
prepare = module["prepare"]
normalize = module["normalize_markup"]


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
    assert '<a name="anchor"></a>' in result
    assert '<td>Atlas A2</td><td>√</td>' in result
    assert linked_text_table in result
    for block in (code, pre, complex_table, linked_table, markdown_table, '`Foo<T>(" id=\\"keep\\"")`'):
        assert block in result
    assert result.count('POLAR_PRESERVED_DOC_BLOCK_0_END') == 1
    assert crlf_code in (dst / "docs/crlf.md").read_bytes()
    assert {p.relative_to(src): p.read_bytes() for p in src.rglob('*') if p.is_file()} == original
    assert (dst / "kernel.cpp").read_bytes() == original[Path("kernel.cpp")]
    assert (dst / "A5_ONLY_STUBS.txt").read_bytes() == original[Path("A5_ONLY_STUBS.txt")]
    with pytest.raises(FileExistsError):
        prepare(src, dst)


def test_only_table_presentation_and_identical_duplicate_anchors_are_removed():
    text = (
        '<table id="table"><tr><td class="cellrowborder" align="left" width="50%" '
        'rowspan="2" colspan="3" headers="h" style="display:none" data-x="width=\'5\'">'
        '<a href="#target">text &lt; value</a><img src="x.png" width="20"></td></tr></table>\r\n'
        '<a name="target"></a>\r\n<a name="target"></a><a name="target"></a>\r\n'
        '<a name="TARGET"></a>\n'
        '<td class="warning" title="x > y" height=\'10\'>other</td>'
    )
    expected = (
        '<table id="table"><tr><td rowspan="2" colspan="3" headers="h" '
        'style="display:none" data-x="width=\'5\'">'
        '<a href="#target">text &lt; value</a><img src="x.png" width="20"></td></tr></table>\r\n'
        '<a name="target"></a>\r\n\r\n<a name="TARGET"></a>\n'
        '<td class="warning" title="x > y">other</td>'
    )
    assert normalize(text) == expected
    assert normalize(expected) == expected


@pytest.mark.parametrize("text", [
    '````cpp\n```\n<td width="1">x</td>\n````\n',
    '~~~html\n<td width="1">x</td>\n~~~~\n',
    '```cpp\n<td width="1">unclosed fence</td>\n',
    '`<td width="1">inline</td>`',
    '``<td width="1">`inline`</td>``',
    '    <td width="1">indented code</td>\n',
    '<pre class="screen"><td width="1">code</td></pre>',
    '<code><td width="1">code</td></code>',
    '<!-- <td width="1">comment</td> -->',
    '<svg><table width="1"></table></svg>',
    '<a name="a"></a><a name="A"></a>',
    '<a name="a"></a><a name="b"></a>',
    '<a href="x" name="a"></a><a href="x" name="a"></a>',
    '| value | meaning<br>next line |\n',
])
def test_protected_content_is_byte_identical(text):
    assert normalize(text) == text


@pytest.mark.parametrize("tag", ["p", "span", "td", "th", "tr", "table", "li", "ul", "h1", "h4"])
def test_parent_id_keeps_target_when_redundant_anchor_is_removed(tag):
    source = f'<{tag} id="exact" title="x > y">\r\n<a name="exact"></a>body</{tag}>\n'
    expected = f'<{tag} id="exact" title="x > y">\r\nbody</{tag}>\n'
    assert normalize(source) == expected
    assert normalize(expected) == expected


@pytest.mark.parametrize("source", [
    '<p id="exact"><a name="EXACT"></a>body</p>',
    '<p><a name="exact"></a>body</p>',
    '<p id="exact">text<a name="exact"></a></p>',
    '<p id="exact"/><a name="exact"></a>',
    '<p id="exact" id="other"><a name="exact"></a></p>',
    '<p data-id="exact"><a name="exact"></a></p>',
    '<p id="exact"><a name="exact" href="other.md"></a></p>',
    '<p id="exact"><a name="exact">label</a></p>',
    '<p id="exact"><a name="exact" class="note"></a></p>',
    '<!-- <p id="exact"><a name="exact"></a> -->',
    '`<p id="exact"><a name="exact"></a>`',
    '```html\n<p id="exact"><a name="exact"></a>\n```\n',
    '<pre><p id="exact"><a name="exact"></a></pre>',
    '    <p id="exact"><a name="exact"></a>\n',
])
def test_unique_semantic_or_code_anchors_are_preserved(source):
    assert normalize(source) == source


def test_parent_anchor_and_table_cleanup_compose():
    source = '<td id="x" width="50%" rowspan="2"><a name="x"></a><a name="x"></a>body</td>'
    assert normalize(source) == '<td id="x" rowspan="2">body</td>'
