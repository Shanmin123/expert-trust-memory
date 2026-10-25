from __future__ import annotations

import argparse
import sys


def smoke_test(args: argparse.Namespace) -> int:
    try:
        import qlib
        from qlib.contrib.data.handler import Alpha158, Alpha360
    except ImportError as exc:
        raise RuntimeError(
            "Qlib is not installed in this Python environment. Use the "
            "Python 3.12 Qlib environment and install requirements-qlib-py312.txt."
        ) from exc

    print(f"Python: {sys.version.split()[0]}")
    print(f"Qlib: {qlib.__version__}")
    print(f"Handlers: {Alpha158.__name__}, {Alpha360.__name__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qlib framework utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke-test", help="Import Qlib and core handlers")
    smoke.set_defaults(func=smoke_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
