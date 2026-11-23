"""Check optional wheel installation without Docker, pip, or network access."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest


class TorchWheelsTest(unittest.TestCase):
    def test_empty_wheels_and_install_failure(self):
        dockerfile = Path(__file__).with_name("Dockerfile").read_text()
        body = dockerfile.split("<<'TORCH_SH'\n", 1)[1].split("\nTORCH_SH", 1)[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheels = root / "wheels"
            wheels.mkdir()
            calls = root / "calls"
            python = root / "python3"
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CALLS"\nexit 7\n')
            python.chmod(0o755)
            env = {**os.environ, "PATH": f"{root}:/usr/bin:/bin", "CALLS": str(calls)}
            command = ["bash", "-eo", "pipefail", "-c", body.replace("/tmp/torch-wheels", str(wheels))]
            self.assertEqual(subprocess.run(command, env=env).returncode, 0)
            self.assertFalse(calls.exists(), "Empty wheel directory should preserve installed packages")
            wheel = wheels / "torch-test.whl"
            wheel.touch()
            self.assertEqual(subprocess.run(command, env=env).returncode, 7)
            self.assertEqual(calls.read_text().splitlines(), ["-m", "pip", "install", str(wheel)])


if __name__ == "__main__":
    unittest.main()
