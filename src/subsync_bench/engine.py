"""Adapter over ffsubsync.

Design rules, all of them learned from measuring the upstream tool:

1. **Never trust ``score``.** It is an unnormalised correlation peak. We keep it
   for provenance but gate on our own normalised confidence (see ``audio.py``).
2. **Never trust ``retval``.** Measured: every failure mode — silent reference,
   wrong-film subtitles, ``--min-score 1e6`` — returns 0 and still writes an
   output file. So we always verify the output ourselves before accepting it.
3. **Never let upstream write the file we hand to the user.** Upstream writes to
   a staging path inside the project; we re-parse it, validate it, and only then
   promote it. A half-written or silently-unshifted file cannot escape.
4. **Never require ffsubsync to inspect a project.** Parsing, measuring and
   previewing all work without it; only the align step needs the engine.
"""

from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import audio, subio
from .errors import (
    E_ALIGN_FAILED,
    E_ENGINE_MISSING,
    AlignError,
)
from .subio import Cue

#: Bounds that are physically meaningful for subtitle sync. A 24 fps -> 25 fps
#: conversion is 1.0427; anything beyond ~5% is a mis-detection, not a transfer.
FRAMERATE_MIN = 0.90
FRAMERATE_MAX = 1.10


@dataclass
class EngineResult:
    """What the engine claims, plus what we independently verified."""

    offset: float
    framerate_scale: float
    upstream_score: float | None
    upstream_retval: int
    upstream_success_flag: bool
    output_path: Path | None
    log_tail: str = ""
    engine_version: str = "unknown"
    mode: str = "engine"  # "engine" | "fallback"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "offset": round(self.offset, 3),
            "framerate_scale": round(self.framerate_scale, 6),
            "upstream_score": self.upstream_score,
            "upstream_retval": self.upstream_retval,
            "upstream_success_flag": self.upstream_success_flag,
            "log_tail": self.log_tail[-2000:],
            "engine_version": self.engine_version,
            "mode": self.mode,
            "warnings": list(self.warnings),
        }


def engine_available() -> bool:
    try:
        import ffsubsync  # noqa: F401
    except Exception:
        return False
    return True


def engine_version() -> str:
    try:
        from ffsubsync.version import __version__

        return str(__version__)
    except Exception:
        return "unknown"


class _LogCapture(logging.Handler):
    def __init__(self, cap: int = 400) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.cap = cap

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(record.getMessage())
            if len(self.lines) > self.cap:
                del self.lines[: len(self.lines) - self.cap]
        except Exception:  # pragma: no cover - defensive
            pass


def _parse_upstream_metrics(lines: Sequence[str]) -> dict[str, float]:
    """Pull score / offset / scale out of upstream's log lines."""
    out: dict[str, float] = {}
    for ln in lines:
        low = ln.lower()
        for key, name in (
            ("score:", "score"),
            ("offset seconds:", "offset"),
            ("framerate scale factor:", "scale"),
        ):
            if low.startswith(key) or f" {key}" in low:
                tail = ln.split(":", 1)[-1].strip()
                try:
                    out[name] = float(tail.split()[0])
                except (ValueError, IndexError):
                    pass
    return out


def run_engine(
    reference: Path,
    subtitle: Path,
    out_path: Path,
    *,
    max_offset_seconds: float = 120.0,
    framerate_ratio: float | None = None,
    encoding: str | None = None,
    no_fix_framerate: bool = False,
) -> EngineResult:
    """Invoke ffsubsync in-process and stage its output.

    Returns an ``EngineResult`` even when upstream "succeeds"; the caller is
    responsible for verifying. Raises only when the engine itself is unusable.
    """
    if not engine_available():
        raise AlignError(
            "ffsubsync is not installed",
            code=E_ENGINE_MISSING,
            hint="pip install 'subsync-bench[engine]' to enable alignment; "
            "inspect/preview/export still work without it",
        )

    try:
        from ffsubsync.ffsubsync import make_parser
        from ffsubsync.ffsubsync import run as ff_run
    except Exception as exc:  # pragma: no cover - broken install
        raise AlignError(
            f"ffsubsync import failed: {exc}", code=E_ENGINE_MISSING
        ) from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    argv = [
        str(reference),
        "-i", str(subtitle),
        "-o", str(out_path),
        "--max-offset-seconds", str(max_offset_seconds),
    ]
    if encoding:
        argv += ["--encoding", encoding]
    if no_fix_framerate:
        argv += ["--no-fix-framerate"]
    elif framerate_ratio is not None:
        argv += ["--frame-rate", str(framerate_ratio)]

    parser = make_parser()
    args = parser.parse_args(argv)

    cap = _LogCapture()
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(cap)
    root.setLevel(logging.INFO)
    stderr_buf = io.StringIO()
    result: dict[str, Any] = {}
    try:
        with contextlib.redirect_stderr(stderr_buf):
            result = ff_run(args)
    except SystemExit as exc:  # upstream calls sys.exit in some paths
        result = {"retval": int(exc.code or 0), "sync_was_successful": False}
    except Exception as exc:
        raise AlignError(
            f"alignment engine raised: {type(exc).__name__}: {exc}",
            code=E_ALIGN_FAILED,
            context={"log_tail": "\n".join(cap.lines[-20:])},
        ) from exc
    finally:
        root.removeHandler(cap)
        root.setLevel(prev_level)

    metrics = _parse_upstream_metrics(cap.lines)
    offset = result.get("offset_seconds")
    if offset is None:
        offset = metrics.get("offset", 0.0)
    scale = result.get("framerate_scale_factor")
    if scale is None:
        scale = metrics.get("scale", 1.0)

    warnings: list[str] = []
    if not audio.has_ffprobe():
        warnings.append(
            "ffprobe missing: upstream could not enumerate subtitle streams and "
            "fell back to audio-only reference"
        )
    if result.get("retval", 0) != 0:
        warnings.append(f"upstream retval={result.get('retval')}")

    return EngineResult(
        offset=float(offset),
        framerate_scale=float(scale),
        upstream_score=metrics.get("score"),
        upstream_retval=int(result.get("retval", 0) or 0),
        upstream_success_flag=bool(result.get("sync_was_successful", False)),
        output_path=out_path if out_path.exists() else None,
        log_tail="\n".join(cap.lines[-40:]),
        engine_version=engine_version(),
        mode="engine",
        warnings=warnings,
    )


@dataclass
class Applied:
    """The verified outcome of an alignment attempt."""

    cues: list[Cue]
    offset: float
    framerate_scale: float
    confidence: float
    null_confidence: float
    speech_ratio: float
    residuals: list[audio.CueResidual]
    residual_summary: dict
    engine: EngineResult | None
    warnings: list[str] = field(default_factory=list)
    #: True when the result was rejected by our own gate and cues are unchanged.
    rejected: bool = False
    reject_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "offset": round(self.offset, 3),
            "framerate_scale": round(self.framerate_scale, 6),
            "confidence": round(self.confidence, 4),
            "null_confidence": round(self.null_confidence, 4),
            "speech_ratio": round(self.speech_ratio, 4),
            "residual_summary": self.residual_summary,
            "engine": self.engine.to_dict() if self.engine else None,
            "warnings": list(self.warnings),
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "cue_count": len(self.cues),
        }


def apply_shift(
    cues: Sequence[Cue], offset: float, scale: float
) -> list[Cue]:
    """Scale then shift, preserving order and clamping at zero."""
    out: list[Cue] = []
    for c in cues:
        s = c.start * scale + offset
        e = c.end * scale + offset
        s = max(0.0, s)
        e = max(s, e)
        out.append(Cue(c.index, s, e, c.text, True))
    return out


def verify_output(
    staged: Path, expected_count: int
) -> tuple[list[Cue], list[str]]:
    """Re-parse upstream's output and check it is structurally sound.

    Upstream's "success" flag is not evidence. Measured failure mode: with a
    silent reference and ``--skip-sync-on-low-quality``, upstream writes the
    *original* cues to the output path and reports success — an output file
    exists and is valid SRT, it is simply unshifted. Only comparing against our
    own measured confidence catches that, which is why this returns the cues for
    the caller to re-measure rather than trusting the file's existence.
    """
    problems: list[str] = []
    if not staged.exists():
        raise AlignError(
            "engine reported success but wrote no output file",
            code=E_ALIGN_FAILED,
        )
    if staged.stat().st_size == 0:
        raise AlignError(
            "engine wrote an empty output file", code=E_ALIGN_FAILED
        )
    try:
        sub = subio.load(staged)
    except Exception as exc:
        raise AlignError(
            f"engine output is not valid SRT: {exc}", code=E_ALIGN_FAILED
        ) from exc

    if len(sub.cues) != expected_count:
        problems.append(
            f"cue count changed: {expected_count} in -> {len(sub.cues)} out"
        )
    struct = subio.validate_cues(sub.cues)
    if struct:
        problems.extend(struct[:5])
    return sub.cues, problems
