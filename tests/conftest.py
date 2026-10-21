"""Shared fixtures.

Media fixtures are generated once by ``fixtures/build_media.py`` (needs ffmpeg and
network for edge-tts). Tests that need real audio skip cleanly when the fixtures
are absent — but they must SKIP, never silently pass, and the skip reason has to
say how to build them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "fixtures" / "media"


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.fixture(scope="session")
def media_dir() -> Path:
    if not MEDIA.exists() or not (MEDIA / "film.mp4").exists():
        pytest.skip(
            "media fixtures missing — run `python fixtures/build_media.py` "
            "(requires ffmpeg + network for edge-tts)"
        )
    return MEDIA


@pytest.fixture(scope="session")
def film(media_dir: Path) -> Path:
    return media_dir / "film.mp4"


@pytest.fixture(scope="session")
def silent_film(media_dir: Path) -> Path:
    p = media_dir / "silent.mp4"
    if not p.exists():
        pytest.skip("silent fixture missing — run fixtures/build_media.py")
    return p


@pytest.fixture(scope="session")
def envelope(film: Path):
    from subsync_bench import audio

    if not _have_ffmpeg():
        pytest.skip("ffmpeg not on PATH")
    return audio.envelope_from_media(film)


@pytest.fixture
def srt(tmp_path: Path):
    """Write an SRT from (start, end, text) tuples."""

    def _make(cues, name: str = "test.srt") -> Path:
        from subsync_bench.subio import Cue, write_srt

        objs = [
            Cue(i + 1, float(a), float(b), t) for i, (a, b, t) in enumerate(cues)
        ]
        p = tmp_path / name
        write_srt(objs, p)
        return p

    return _make


@pytest.fixture
def project(tmp_path: Path, film: Path, media_dir: Path):
    """A ready-to-use project with the 'late' subtitle."""
    from subsync_bench.workspace import Project

    sub = media_dir / "late.srt"
    if not sub.exists():
        pytest.skip("late.srt fixture missing — run fixtures/build_media.py")
    return Project.create(tmp_path / "proj", video=film, subtitle=sub)
