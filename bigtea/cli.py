"""bigtea CLI skeleton (wrapper-core T1, #22).

argparse only — zero runtime dependencies. The four v1 subcommands are
stubs here: each parses its args, loads/validates the config, and points
at the ticket that implements it.
"""

from __future__ import annotations

import argparse
import sys

from bigtea import __version__
from bigtea.config import ConfigError, load_config

SUBCOMMANDS = ("probe", "recommend", "launch", "bench")

_HELP = {
    "probe": "probe local hardware (RAM bandwidth, VRAM/GPU)",
    "recommend": "recommend a CPU/GPU expert offload split",
    "launch": "generate launch flags and start the inference server",
    "bench": "run a benchmark and report tok/s",
}

_STUB_LINES = {
    "probe": "probe: not implemented yet - see issues #8/#9 (hardware-profiler T1/T2)",
    "recommend": "recommend: not implemented yet - see issue #2 (gap-closure T2)",
    "launch": "launch: not implemented yet - see issue #24 (wrapper-core T3)",
    "bench": "bench: not implemented yet - see issues #15/#16 (benchmark-harness T1/T2)",
}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        default=None,
        help="path to a bigtea TOML config (default: ./bigtea.toml if present)",
    )

    parser = argparse.ArgumentParser(
        prog="bigtea",
        description=(
            "Local MoE inference wrapper: probe hardware, recommend an expert "
            "offload split, launch the server, benchmark tok/s."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        required=True,
        metavar="{" + ",".join(SUBCOMMANDS) + "}",
    )
    for name in SUBCOMMANDS:
        subparsers.add_parser(
            name, parents=[common], help=_HELP[name], description=_HELP[name]
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        load_config(args.config)
    except ConfigError as exc:
        print(f"bigtea: config error: {exc}", file=sys.stderr)
        return 2

    print(_STUB_LINES[args.command])
    return 0


if __name__ == "__main__":
    sys.exit(main())
