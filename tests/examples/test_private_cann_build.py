from pathlib import Path
import runpy,shutil,subprocess,hashlib
import pytest
ROOT=Path(__file__).resolve().parents[2]
SCRIPT_DIR=ROOT/'operator_runtime_t2a/skills/tilelang2ascend-translator/scripts'
MODULE=runpy.run_path(str(SCRIPT_DIR/'build_ascendc.py'))
prepare=MODULE['_private_cann_sdk']

def make_sdk(tmp_path):
    root=tmp_path/'installed-sdk';rules=root/'tools/tikcpp/ascendc_kernel_cmake/legacy_modules'
    shutil.copytree(ROOT/'tests/fixtures/cann_preprocess_9_2',rules)
    (root/'include').mkdir();(root/'include/marker.h').write_text('original header')
    (root/'lib64').mkdir();(root/'lib64/libtest.so').write_bytes(b'original library')
    (rules.parent/'ascendc.cmake').write_text('include(${CMAKE_CURRENT_LIST_DIR}/legacy_modules/function.cmake)')
    return root,rules

def digest(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}

def test_private_rules_and_repeated_use(tmp_path):
    sdk,rules=make_sdk(tmp_path);original=digest(sdk);build=tmp_path/'build'
    view=prepare(sdk,build)
    assert view!=sdk and digest(sdk)==original
    assert (view/'include').is_symlink() and (view/'include/marker.h').read_text()=='original header'
    assert (view/'lib64/libtest.so').read_bytes()==b'original library'
    merged=view/'tools/tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_obj_text.sh'
    assert 'output="${arg}.merged"' in merged.read_text()
    stamp=merged.stat().st_mtime_ns
    assert prepare(sdk,build)==view and merged.stat().st_mtime_ns==stamp
    (rules/'util/merge_obj_text.sh').write_text('unsupported SDK rule')
    with pytest.raises(RuntimeError,match='rules changed'):prepare(sdk,build)


def test_reject_unknown_sdk_without_modifying_installation(tmp_path):
    sdk,rules=make_sdk(tmp_path);(rules/'util/merge_obj_text.sh').write_text('unsupported SDK rule')
    before=digest(sdk)
    with pytest.raises(subprocess.CalledProcessError):prepare(sdk,tmp_path/'build')
    assert digest(sdk)==before and not (tmp_path/'build/_polar_cann').exists()


def test_already_fixed_sdk_is_read_only(tmp_path):
    sdk,rules=make_sdk(tmp_path)
    patch=SCRIPT_DIR/'cann_preprocess.patch'
    subprocess.run(['patch','--batch','--fuzz=0','-p1'],input=patch.read_bytes(),cwd=rules,check=True,capture_output=True)
    before=digest(sdk);prepare(sdk,tmp_path/'build');assert digest(sdk)==before


def test_nonlegacy_build_unchanged(tmp_path):
    sdk=tmp_path/'nonlegacy';sdk.mkdir()
    assert prepare(sdk,tmp_path/'build')==sdk
    assert not (tmp_path/'build').exists()
