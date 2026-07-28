"""CLI tests for wrapper-core T1 (#22).

Acceptance criteria covered here:
- `--help` enumerates all four v1 subcommands.
- A malformed config file produces a named parse error (nonzero exit,
  no traceback), not an unhandled exception.
"""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bigtea.cli import SUBCOMMANDS, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_bigtea(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "bigtea", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


class TestHelp(unittest.TestCase):
    def test_help_enumerates_all_four_subcommands(self):
        result = run_bigtea("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("probe", "recommend", "launch", "bench"):
            self.assertIn(name, result.stdout)

    def test_no_subcommand_is_a_usage_error(self):
        result = run_bigtea()
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage", result.stderr.lower())


class TestMalformedConfig(unittest.TestCase):
    def test_malformed_toml_gives_named_error_not_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "broken.toml"
            bad.write_text('[model\nname = "unclosed section', encoding="utf-8")
            result = run_bigtea("probe", "--config", str(bad))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("config error", result.stderr)
        self.assertIn("broken.toml", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("Traceback", result.stdout)

    def test_missing_explicit_config_is_named_error(self):
        result = run_bigtea("bench", "--config", "does-not-exist.toml")
        self.assertEqual(result.returncode, 2)
        self.assertIn("config error", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class TestStubs(unittest.TestCase):
    """Each subcommand parses args, loads config, prints its one stub line."""

    expected_issue = {
        "probe": "#8/#9",
        "recommend": "#2",
        "launch": "#24",
        "bench": "#15/#16",
    }

    def _run_inprocess(self, *argv: str) -> tuple[int, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(argv))
        return code, stdout.getvalue()

    def test_each_stub_points_at_its_ticket(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "ok.toml"
            cfg.write_text('[model]\nname = "test"\n', encoding="utf-8")
            for name in SUBCOMMANDS:
                with self.subTest(subcommand=name):
                    code, out = self._run_inprocess(name, "--config", str(cfg))
                    self.assertEqual(code, 0)
                    lines = [l for l in out.splitlines() if l.strip()]
                    self.assertEqual(len(lines), 1)
                    self.assertIn("not implemented yet", lines[0])
                    self.assertIn(self.expected_issue[name], lines[0])

    def test_example_config_is_valid(self):
        code, out = self._run_inprocess(
            "probe", "--config", str(REPO_ROOT / "bigtea.example.toml")
        )
        self.assertEqual(code, 0)
        self.assertIn("not implemented yet", out)


if __name__ == "__main__":
    unittest.main()
