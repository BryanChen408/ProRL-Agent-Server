"""Exercise collection with failed/missing tools, without touching Docker or NPUs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class InspectA5Test(unittest.TestCase):
    def test_failures_do_not_stop_collection_or_launch_containers(self):
        script = Path(__file__).with_name("inspect_a5.sh")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = root / "docker-calls"
            for tool in ("npu-smi", "node", "npm", "claude", "bisheng", "msprof"):
                stub = root / tool
                stub.write_text("#!/bin/sh\necho simulated-failure\nexit 126\n")
                stub.chmod(0o755)
            docker = root / "docker"
            docker.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_CALLS"\nexit 1\n'
            )
            docker.chmod(0o755)
            result = subprocess.run(
                ["bash", str(script), "example:a5"],
                env={**os.environ, "PATH": f"{root}:/usr/bin:/bin", "DOCKER_CALLS": str(calls)},
                text=True, capture_output=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("exit=126; continuing", result.stdout)
            self.assertIn("===== Done =====", result.stdout)
            self.assertIn(f"msprof executable: {root / 'msprof'}", result.stdout)
            self.assertNotIn("$ msprof --version", result.stdout)
            commands = calls.read_text().splitlines()
            self.assertEqual(len(commands), 3)
            for command, prefix in zip(commands, ("version ", "image ls ", "image inspect ")):
                self.assertTrue(command.startswith(prefix), command)
            self.assertTrue(commands[-1].endswith("-- example:a5"))


if __name__ == "__main__":
    unittest.main()
