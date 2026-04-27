"""Command-line interface for HybridGuard.

Subcommands
-----------
canonicalize   Apply the canonicalization front-end to a string or file.
reproduce      (Phase 2) Reproduce the paper's results end-to-end.
train          (Phase 2) Train a single HybridGuard variant for one seed.
evaluate       (Phase 2) Evaluate trained models and regenerate tables.
version        Print the installed version.

Examples
--------
    hybridguard canonicalize "h\u0435llo w\u00f8rld"          # handle homoglyphs
    hybridguard canonicalize --file note.txt --trace   # show what was decoded
    hybridguard version

The Phase 2 subcommands (reproduce / train / evaluate) currently print a
"not yet ported to library form" notice and point the user at the Colab
orchestrator notebook (``notebooks/HybridGuard_Colab_Orchestrator.ipynb``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .canonicalize import canonicalize


# ---------------------------------------------------------------------------
# `canonicalize` subcommand
# ---------------------------------------------------------------------------

def _cmd_canonicalize(args: argparse.Namespace) -> int:
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    result = canonicalize(text)

    if args.json:
        payload = {
            "canonical": result.canonical,
            "lowercase": result.lowercase,
            "trace": list(result.trace),
            "decoded_payloads": [list(p) for p in result.decoded_payloads],
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    elif args.trace:
        sys.stdout.write(result.canonical + "\n")
        sys.stderr.write(
            f"[trace] applied_transforms={list(result.trace)} "
            f"decoded_payloads={[list(p) for p in result.decoded_payloads]}\n"
        )
    else:
        sys.stdout.write(result.canonical + "\n")
    return 0


# ---------------------------------------------------------------------------
# `version` subcommand
# ---------------------------------------------------------------------------

def _cmd_version(args: argparse.Namespace) -> int:
    sys.stdout.write(f"hybridguard {__version__}\n")
    return 0


# ---------------------------------------------------------------------------
# Phase 2 placeholders
# ---------------------------------------------------------------------------

_PHASE2_MSG = (
    "This subcommand is not yet ported from the Colab notebook to the library.\n"
    "For now, reproduce using the orchestrator notebook:\n"
    "    notebooks/HybridGuard_Colab_Orchestrator.ipynb\n"
    "See paper/README.md for the full reproduction walkthrough.\n"
    "Phase 2 of the repo restructure will move the training / evaluation\n"
    "logic out of the notebook and make `hybridguard {sub}` a first-class entry point.\n"
)


def _cmd_phase2_stub(sub: str):
    def _run(args: argparse.Namespace) -> int:
        sys.stderr.write(_PHASE2_MSG.format(sub=sub))
        return 2
    return _run


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hybridguard",
        description="HybridGuard: prompt-injection detection with a canonicalization front-end.",
    )
    p.add_argument("--version", action="version", version=f"hybridguard {__version__}")
    sub = p.add_subparsers(dest="subcommand", required=True)

    # canonicalize
    s_can = sub.add_parser(
        "canonicalize",
        help="Apply the canonicalization front-end to a string or file.",
        description=(
            "Apply NFKC normalization, zero-width/Tag-block stripping, "
            "Unicode TR39 confusable folding, and depth-bounded "
            "base64/hex/URL/ROT13 unwrapping to the input text."
        ),
    )
    s_can_src = s_can.add_mutually_exclusive_group()
    s_can_src.add_argument("text", nargs="?", default=None,
                           help="Text to canonicalize. If omitted, read from stdin.")
    s_can_src.add_argument("--file", type=str, default=None,
                           help="Read input text from this file instead of stdin.")
    s_can.add_argument("--json", action="store_true",
                       help="Emit a JSON report with canonical text + trace flags.")
    s_can.add_argument("--trace", action="store_true",
                       help="Also print a trace summary to stderr.")
    s_can.set_defaults(func=_cmd_canonicalize)

    # version
    s_ver = sub.add_parser("version", help="Print the installed version.")
    s_ver.set_defaults(func=_cmd_version)

    # Phase 2 stubs
    for sub_name, helptext in (
        ("reproduce", "(Phase 2) Reproduce the paper's results end-to-end."),
        ("train",     "(Phase 2) Train a single HybridGuard variant for one seed."),
        ("evaluate",  "(Phase 2) Evaluate trained models and regenerate LaTeX tables."),
    ):
        s = sub.add_parser(sub_name, help=helptext)
        s.set_defaults(func=_cmd_phase2_stub(sub_name))

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
