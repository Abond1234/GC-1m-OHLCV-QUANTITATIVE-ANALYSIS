"""Packaging must not collect incompatible DLLs from unrelated PATH tools."""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_app


class BuildEnvironmentTests(unittest.TestCase):
    def test_windows_path_excludes_external_tools_without_mutating_parent(self):
        original = {"PATH": "unrelated-tools", "SystemRoot": "windows", "KEEP": "yes"}
        with patch.dict(os.environ, original, clear=True), patch.object(sys, "platform", "win32"):
            env = build_app._packaging_environment()
            self.assertEqual(os.environ["PATH"], original["PATH"])
        self.assertEqual(env["KEEP"], "yes")
        self.assertEqual(
            env["PATH"].split(os.pathsep),
            [
                str(Path(sys.executable).parent),
                str(Path(sys.base_prefix)),
                str(Path("windows") / "System32"),
                "windows",
            ],
        )

    def test_non_windows_environment_is_preserved(self):
        with (
            patch.dict(os.environ, {"PATH": "original"}, clear=True),
            patch.object(sys, "platform", "linux"),
        ):
            self.assertEqual(build_app._packaging_environment(), {"PATH": "original"})

    def test_build_uses_clean_subprocess_and_propagates_failure(self):
        with (
            patch.object(build_app.importlib.util, "find_spec", return_value=object()),
            patch.object(build_app, "_render_icon", return_value=Path("icon.ico")),
            patch.object(build_app, "_packaging_environment", return_value={"PATH": "safe"}),
            patch.object(build_app.subprocess, "run") as run,
        ):
            run.side_effect = subprocess.CalledProcessError(1, "PyInstaller")
            with self.assertRaises(subprocess.CalledProcessError):
                build_app.main()
        args, kwargs = run.call_args
        self.assertEqual(args[0][:3], [sys.executable, "-m", "PyInstaller"])
        self.assertIn("--clean", args[0])
        self.assertEqual(kwargs["env"], {"PATH": "safe"})
        self.assertTrue(kwargs["check"])


if __name__ == "__main__":
    unittest.main()
