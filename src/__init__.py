"""Package init.

Sets process-wide environment variables that must be in place *before*
PyTorch and LightGBM both load their OpenMP runtimes. On macOS,
PyTorch ships with Intel OpenMP (`libiomp5.dylib`) and LightGBM uses
LLVM OpenMP (`libomp.dylib`). When both are loaded into the same
process and both are exercised heavily (as happens in `run_all`), the
two runtimes deadlock on a barrier the first time they are used after
each other. `KMP_DUPLICATE_LIB_OK=TRUE` tells Intel OpenMP to tolerate
the second runtime's presence rather than abort or deadlock; this is
the documented Intel workaround.

We set the variable conservatively: only if the user has not already
chosen a value. The variable is harmless on Linux / Windows (Intel
OpenMP simply ignores it on platforms where the duplicate-library
condition cannot arise).

We also force unbuffered stdout so that long-running stages (RF tuning,
MLP sweep) print progress to logs in real time rather than dumping
everything at process exit.
"""
from __future__ import annotations

import os
import sys

# Avoid the Intel/LLVM OpenMP barrier deadlock on macOS when torch and
# lightgbm are both used in the same process. Must be set before either
# library imports its OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Make print() flush immediately even when stdout is piped to `tee` or a
# log file. Equivalent to running with `python -u`. Without this, graders
# running `python -m src.run_all > log.txt` see an empty log for minutes.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    # reconfigure is Python 3.7+ on TextIOWrapper; falls through quietly
    # on streams that do not support it (e.g. some test harnesses).
    pass
