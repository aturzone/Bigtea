"""Config loading for bigtea.

TOML only, parsed with stdlib ``tomllib`` (zero runtime dependencies).
Every subcommand reads its config through :func:`load_config`.

Schema (minimal for T1; later tickets extend it):

    [model]     name (str), gguf (str path)
    [hardware]  free-form probe overrides — empty is fine
    [launch]    free-form extra launch flags — empty is fine

Rules:
- Malformed TOML raises :class:`ConfigError` (a named, user-readable error),
  never an unhandled ``tomllib.TOMLDecodeError``.
- Unknown sections/keys warn but do not fail.
- No config file at the default location: clear message + built-in defaults.
- An explicitly passed ``--config`` path that does not exist is an error
  (silently ignoring it would mask typos).
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

DEFAULT_CONFIG_NAME = "bigtea.toml"

_KNOWN_SECTIONS = {"model", "hardware", "launch"}
_KNOWN_MODEL_KEYS = {"name", "gguf"}


class ConfigError(Exception):
    """A named, user-readable error in a bigtea config file."""


@dataclass
class Config:
    """Parsed config; ``source is None`` means built-in defaults."""

    model_name: str | None = None
    model_gguf: str | None = None
    hardware: dict = field(default_factory=dict)
    launch: dict = field(default_factory=dict)
    source: Path | None = None


def _default_warn(message: str) -> None:
    print(f"bigtea: warning: {message}", file=sys.stderr)


def load_config(
    path: str | Path | None = None,
    *,
    warn: Callable[[str], None] = _default_warn,
) -> Config:
    """Load a bigtea config.

    ``path=None`` looks for ``bigtea.toml`` in the current directory and
    falls back to built-in defaults (with a message) if it is absent.
    """
    if path is not None:
        file = Path(path)
        if not file.is_file():
            raise ConfigError(f"config file not found: {file}")
        return _parse_file(file, warn)

    file = Path.cwd() / DEFAULT_CONFIG_NAME
    if not file.is_file():
        warn(f"no {DEFAULT_CONFIG_NAME} in {Path.cwd()}; using built-in defaults")
        return Config()
    return _parse_file(file, warn)


def _parse_file(file: Path, warn: Callable[[str], None]) -> Config:
    try:
        with open(file, "rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"failed to parse {file}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {file}: {exc}") from exc

    for section in data:
        if section not in _KNOWN_SECTIONS:
            warn(f"{file}: unknown section [{section}] ignored")

    model = data.get("model", {})
    hardware = data.get("hardware", {})
    launch = data.get("launch", {})
    for section_name, value in (("model", model), ("hardware", hardware), ("launch", launch)):
        if not isinstance(value, dict):
            raise ConfigError(
                f"{file}: [{section_name}] must be a table, got {type(value).__name__}"
            )

    for key in model:
        if key not in _KNOWN_MODEL_KEYS:
            warn(f"{file}: unknown key model.{key} ignored")

    model_name = model.get("name")
    model_gguf = model.get("gguf")
    for label, value in (("model.name", model_name), ("model.gguf", model_gguf)):
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{file}: {label} must be a string, got {type(value).__name__}")

    return Config(
        model_name=model_name,
        model_gguf=model_gguf,
        hardware=dict(hardware),
        launch=dict(launch),
        source=file,
    )
