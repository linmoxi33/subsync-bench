"""subsync-bench — a workbench for making a downloaded subtitle actually line up.

The one-step tool this is built around (ffsubsync) does exactly one thing: shift a
subtitle to match a video's audio. This package keeps that engine but wraps the
whole job around it:

    ingest -> measure -> align -> preview -> compare -> select -> export -> report

Everything is reachable from the CLI and from the web workbench, and both drive
the same code path.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]

from .errors import BenchError  # noqa: F401
