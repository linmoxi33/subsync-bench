"""Fail the build when test fixtures are missing.

Tests that need media skip cleanly when fixtures are absent, which is right for a
developer without ffmpeg but wrong for CI: a green run with half the suite skipped
is worse than a red one. CI calls this before pytest so a missing fixture is a
hard failure with a clear message.

Kept as a script rather than an inline heredoc because the windows-latest runner
defaults to a non-bash shell and parses heredocs differently.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "fixtures" / "media"
REQUIRED = [
    "film.mp4",
    "silent.mp4",
    "truth.srt",
    "late.srt",
    "early.srt",
    "fps.srt",
]


def main() -> int:
    if not MEDIA.exists():
        print(f"FAIL: {MEDIA} does not exist — run fixtures/build_offline.py")
        return 1
    missing = [n for n in REQUIRED if not (MEDIA / n).exists()]
    if missing:
        print(f"FAIL: missing fixtures: {missing}")
        print("      run `python fixtures/build_offline.py`")
        return 1

    # A zero-byte fixture would satisfy exists() but break every test that uses
    # it, with a confusing downstream error instead of this one.
    empty = [n for n in REQUIRED if (MEDIA / n).stat().st_size == 0]
    if empty:
        print(f"FAIL: empty fixtures: {empty}")
        return 1

    print(f"all {len(REQUIRED)} fixtures present:")
    for n in REQUIRED:
        print(f"  {n:12} {(MEDIA / n).stat().st_size:>9,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
