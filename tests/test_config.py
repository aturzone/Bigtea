"""Config-loading tests for wrapper-core T1 (#22)."""

import contextlib
import tempfile
import unittest
from pathlib import Path

from bigtea.config import Config, ConfigError, load_config


class TestLoadConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.warnings: list[str] = []

    def load(self, path=None):
        return load_config(path, warn=self.warnings.append)

    def write(self, text: str, name: str = "cfg.toml") -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_valid_config_parses_all_sections(self):
        path = self.write(
            '[model]\nname = "qwen3-moe"\ngguf = "/models/q.gguf"\n'
            "[hardware]\nram_gb = 192\n"
            '[launch]\nextra_flags = ["--host", "127.0.0.1"]\n'
        )
        cfg = self.load(path)
        self.assertEqual(cfg.model_name, "qwen3-moe")
        self.assertEqual(cfg.model_gguf, "/models/q.gguf")
        self.assertEqual(cfg.hardware, {"ram_gb": 192})
        self.assertEqual(cfg.launch, {"extra_flags": ["--host", "127.0.0.1"]})
        self.assertEqual(cfg.source, path)
        self.assertEqual(self.warnings, [])

    def test_empty_sections_are_fine(self):
        cfg = self.load(self.write("[model]\n[hardware]\n[launch]\n"))
        self.assertIsNone(cfg.model_name)
        self.assertEqual(cfg.hardware, {})
        self.assertEqual(cfg.launch, {})
        self.assertEqual(self.warnings, [])

    def test_malformed_toml_raises_named_config_error(self):
        path = self.write("[model\nbroken")
        with self.assertRaises(ConfigError) as ctx:
            self.load(path)
        self.assertIn(str(path), str(ctx.exception))

    def test_missing_explicit_path_raises_config_error(self):
        with self.assertRaises(ConfigError) as ctx:
            self.load(self.tmp / "nope.toml")
        self.assertIn("not found", str(ctx.exception))

    def test_missing_default_file_gives_defaults_and_message(self):
        with contextlib.chdir(self.tmp):
            cfg = self.load(None)
        self.assertIsInstance(cfg, Config)
        self.assertIsNone(cfg.source)
        self.assertEqual(len(self.warnings), 1)
        self.assertIn("built-in defaults", self.warnings[0])

    def test_default_file_in_cwd_is_picked_up(self):
        self.write('[model]\nname = "x"\n', name="bigtea.toml")
        with contextlib.chdir(self.tmp):
            cfg = self.load(None)
        self.assertEqual(cfg.model_name, "x")

    def test_unknown_section_and_key_warn_but_load(self):
        cfg = self.load(
            self.write('[model]\nname = "x"\ncolor = "red"\n[mystery]\nfoo = 1\n')
        )
        self.assertEqual(cfg.model_name, "x")
        self.assertEqual(len(self.warnings), 2)
        self.assertTrue(any("mystery" in w for w in self.warnings))
        self.assertTrue(any("model.color" in w for w in self.warnings))

    def test_non_table_section_raises_config_error(self):
        with self.assertRaises(ConfigError) as ctx:
            self.load(self.write("model = 3\n"))
        self.assertIn("[model] must be a table", str(ctx.exception))

    def test_non_string_model_name_raises_config_error(self):
        with self.assertRaises(ConfigError) as ctx:
            self.load(self.write("[model]\nname = 42\n"))
        self.assertIn("model.name must be a string", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
