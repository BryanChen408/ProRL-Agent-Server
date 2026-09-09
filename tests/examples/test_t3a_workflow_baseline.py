import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
SUMMARIZE = runpy.run_path(str(ROOT / "deploy/ascend_operator/t3a/audit_workflow_baseline.py"))["summarize"]


def test_baseline_match_mismatch_and_empty(tmp_path):
    body = "# Ascend Kernel Developer\n\n## 工作流总览\n```\nPhase 0: old\n```"
    source = "---\nname: developer\n---\n" + body
    assert SUMMARIZE(tmp_path, source)["status"] == "unverified"
    path = tmp_path / "OP__sub_developer.raw.json"
    path.write_text(json.dumps({"messages": [{"role": "system", "content": "SDK header\n" + body}]}))
    report = SUMMARIZE(tmp_path, source)
    assert report["status"] == "matched"
    assert report["variants"][0]["outline"] == "Phase 0: old"
    assert SUMMARIZE(tmp_path, source.replace("old", "new"))["status"] == "unverified"
