from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "ascend_operator" / "sync_polar_repo.sh"


def test_sync_polar_repo_help() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert "Usage:" in result.stdout
    assert "output/" in result.stdout


def test_sync_polar_repo_rejects_legacy_polar_e2e_destination() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "root@host:/home/docker/polar_e2e/"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 2
    assert "legacy polar_e2e destination" in result.stderr
