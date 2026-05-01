"""Force a non-interactive Matplotlib backend before pyplot is imported.

Importing this module from any entry point guarantees that figures render
without a display server. This must run *before* `matplotlib.pyplot` is
imported anywhere else in the process.

If the user has already set `MPLBACKEND` in the environment we respect
that choice; otherwise we install `Agg`, which works on Windows / Linux /
CI without Tk or a display.
"""
from __future__ import annotations

import os

_user_override = "MPLBACKEND" in os.environ

import matplotlib

if not _user_override:
    os.environ["MPLBACKEND"] = "Agg"
    matplotlib.use("Agg", force=True)
