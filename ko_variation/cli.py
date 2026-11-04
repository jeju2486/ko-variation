"""Public entry point for direct unitig discovery.

Historical candidate-table scans remain reproducible from the v0.8.3 tag.
Annotation, networks and phenotype helpers have their own console commands.
"""
from __future__ import annotations
import sys
from . import __version__
from .memory_budget import guard_blas_environment


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--version"]:
        print(f"KO-Variation {__version__}")
        return 0
    if arguments and arguments[0] == "discover":
        arguments.pop(0)
    if any(a == "--pairs" or a.startswith("--pairs=") for a in arguments):
        raise SystemExit("Candidate-table scanning was retired from this release; use the v0.8.3 tag in a separate environment.")
    guard_blas_environment(default_threads=1)
    from .discovery import main as discover
    return discover(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
