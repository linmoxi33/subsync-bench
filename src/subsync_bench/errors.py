"""Error taxonomy.

Every failure carries a stable code so the CLI, the web workbench and the tests
agree on what happened. Codes are part of the public contract: never renumber,
never reuse.
"""

from __future__ import annotations

from typing import Any

# --- ingest ---------------------------------------------------------------
E_NO_INPUT = "E_NO_INPUT"                     # nothing to work on
E_VIDEO_MISSING = "E_VIDEO_MISSING"           # reference video not found
E_VIDEO_NO_AUDIO = "E_VIDEO_NO_AUDIO"         # container has no audio stream
E_VIDEO_SILENT = "E_VIDEO_SILENT"             # audio present but no speech energy
E_SUB_MISSING = "E_SUB_MISSING"               # subtitle file not found
E_SUB_EMPTY = "E_SUB_EMPTY"                   # parsed to zero cues
E_SUB_MALFORMED = "E_SUB_MALFORMED"           # could not parse
E_SUB_OVERLAP = "E_SUB_OVERLAP"               # negative-duration / reversed cue
E_ENCODING_UNDETECTED = "E_ENCODING_UNDETECTED"

# --- alignment ------------------------------------------------------------
E_ALIGN_FAILED = "E_ALIGN_FAILED"             # engine raised
E_ALIGN_NO_SPEECH = "E_ALIGN_NO_SPEECH"       # reference had no speech to align to
E_ALIGN_OFFSET_TOO_LARGE = "E_ALIGN_OFFSET_TOO_LARGE"
E_ALIGN_LOW_CONFIDENCE = "E_ALIGN_LOW_CONFIDENCE"
E_ALIGN_FRAMERATE_IMPLAUSIBLE = "E_ALIGN_FRAMERATE_IMPLAUSIBLE"
E_ENGINE_MISSING = "E_ENGINE_MISSING"         # ffsubsync not importable
E_FFMPEG_MISSING = "E_FFMPEG_MISSING"

# --- workspace ------------------------------------------------------------
E_PROJECT_EXISTS = "E_PROJECT_EXISTS"
E_PROJECT_MISSING = "E_PROJECT_MISSING"
E_PROJECT_CORRUPT = "E_PROJECT_CORRUPT"
E_RECEIPT_MISSING = "E_RECEIPT_MISSING"
E_RECEIPT_STALE = "E_RECEIPT_STALE"           # inputs changed since the receipt
E_NO_DECISION = "E_NO_DECISION"               # nothing selected to export

# --- export ---------------------------------------------------------------
E_EXPORT_BLOCKED = "E_EXPORT_BLOCKED"         # gate refused the export
E_EXPORT_EXISTS = "E_EXPORT_EXISTS"
E_EXPORT_WRITE_FAILED = "E_EXPORT_WRITE_FAILED"

# --- generic --------------------------------------------------------------
E_BAD_ARGS = "E_BAD_ARGS"
E_INTERNAL = "E_INTERNAL"


class BenchError(Exception):
    """Base error. Always carries a code, a human message and optional context."""

    code: str = E_INTERNAL

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or type(self).code
        self.context = context or {}
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.context:
            out["context"] = self.context
        if self.hint:
            out["hint"] = self.hint
        return out

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


class InputError(BenchError):
    code = E_BAD_ARGS


class MediaError(BenchError):
    code = E_VIDEO_MISSING


class SubtitleError(BenchError):
    code = E_SUB_MALFORMED


class AlignError(BenchError):
    code = E_ALIGN_FAILED


class WorkspaceError(BenchError):
    code = E_PROJECT_CORRUPT


class ExportError(BenchError):
    code = E_EXPORT_WRITE_FAILED


#: Codes that mean "the result exists but must not be trusted". The export gate
#: and the UI both treat these as blocking until a human overrides them.
BLOCKING_CODES = frozenset(
    {
        E_VIDEO_SILENT,
        E_ALIGN_NO_SPEECH,
        E_ALIGN_LOW_CONFIDENCE,
        E_ALIGN_FRAMERATE_IMPLAUSIBLE,
        E_ALIGN_OFFSET_TOO_LARGE,
        E_ALIGN_FAILED,
    }
)
