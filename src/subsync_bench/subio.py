"""Subtitle I/O: parse, serialize, and encode safely.

Deliberately independent of pysubs2 so the workbench can inspect a subtitle file
even when the alignment engine is not installed. Handles the encodings that
actually show up in downloaded subtitle packs (UTF-8, UTF-8-BOM, GBK, CP1252,
UTF-16), and refuses to silently mangle bytes it cannot decode.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from .errors import (
    E_ENCODING_UNDETECTED,
    E_SUB_EMPTY,
    E_SUB_MALFORMED,
    E_SUB_MISSING,
    SubtitleError,
)

_TIME_RE = re.compile(
    r"(?P<h>\d{1,3}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{1,3})"
)
_ARROW_RE = re.compile(r"--\s*>")

#: Encodings tried in order. GB18030 before GBK because it is a superset and
#: decodes strictly more Chinese subtitle packs without loss.
_ENCODING_CHAIN: Sequence[str] = (
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "gb18030",
    "big5",
    "cp1252",
)

#: A cue longer than this is almost always a sign of a broken timecode, not a
#: real subtitle (24h of dialogue is not a subtitle line).
MAX_CUE_SECONDS = 3600.0


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str
    #: True when the cue was modified by a transform (shift/scale/split).
    touched: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shifted(self, offset: float, *, clamp: bool = True) -> Cue:
        start = self.start + offset
        end = self.end + offset
        if clamp:
            start = max(0.0, start)
            end = max(0.0, end)
            if end < start:
                end = start
        return Cue(self.index, start, end, self.text, True)

    def scaled(self, factor: float) -> Cue:
        return Cue(self.index, self.start * factor, self.end * factor, self.text, True)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "text": self.text,
            "touched": self.touched,
        }


@dataclass
class Subtitle:
    cues: list[Cue]
    encoding: str
    path: Path | None = None
    #: Non-fatal parse observations, surfaced in the UI rather than swallowed.
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cues)

    @property
    def span(self) -> tuple[float, float]:
        if not self.cues:
            return (0.0, 0.0)
        return (min(c.start for c in self.cues), max(c.end for c in self.cues))

    def total_text_chars(self) -> int:
        return sum(len(c.text) for c in self.cues)

    def to_dict(self) -> dict:
        return {
            "encoding": self.encoding,
            "path": str(self.path) if self.path else None,
            "cue_count": len(self.cues),
            "span": [round(self.span[0], 3), round(self.span[1], 3)],
            "text_chars": self.total_text_chars(),
            "warnings": list(self.warnings),
        }


def _parse_timecode(raw: str) -> float:
    m = _TIME_RE.search(raw.strip())
    if not m:
        raise SubtitleError(f"unparseable timecode: {raw!r}", code=E_SUB_MALFORMED)
    h = int(m.group("h"))
    mi = int(m.group("m"))
    s = int(m.group("s"))
    ms_raw = m.group("ms")
    # "5" means 500ms, "05" means 50ms, "005" means 5ms
    ms = int(ms_raw.ljust(3, "0"))
    return h * 3600 + mi * 60 + s + ms / 1000.0


def format_timecode(t: float, *, sep: str = ",") -> str:
    if t < 0:
        t = 0.0
    # int() is load-bearing: divmod below needs ints, round() gives a float.
    total_ms = int(round(t * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    mi, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{mi:02d}:{s:02d}{sep}{ms:03d}"


def decode_bytes(raw: bytes, *, override: str | None = None) -> tuple[str, str]:
    """Decode subtitle bytes, returning (text, encoding_used)."""
    if override:
        try:
            return raw.decode(override), override
        except (UnicodeDecodeError, LookupError) as exc:
            raise SubtitleError(
                f"cannot decode as {override}: {exc}",
                code=E_ENCODING_UNDETECTED,
                hint="omit --encoding to auto-detect, or pass a valid codec name",
            ) from exc

    # UTF-16 only when the BOM says so. Do NOT try it speculatively: any even-length
    # byte string "decodes" as UTF-16 without error, so a GBK subtitle comes back
    # as plausible-looking Korean garbage. This was a real bug — a 7-character
    # Chinese SRT decoded to "탖쓎훗믄퓊" with no exception raised.
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return raw.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            pass

    for enc in _ENCODING_CHAIN:
        if enc == "utf-16":
            continue
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        # Reject a "successful" decode that produced replacement garbage: a
        # cp1252 fallback on Chinese text yields many control chars.
        if enc == "cp1252" and _looks_garbled(text):
            continue
        return text, enc
    raise SubtitleError(
        "could not determine text encoding",
        code=E_ENCODING_UNDETECTED,
        hint="pass --encoding gb18030 or convert the file to UTF-8 first",
    )


def _looks_garbled(text: str) -> bool:
    if not text:
        return False
    ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\r\n\t")
    return ctrl / len(text) > 0.02


def parse_text(text: str) -> tuple[list[Cue], list[str]]:
    """Parse SRT-ish text into cues. Tolerant, but reports what it tolerated."""
    warnings: list[str] = []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip a BOM that survived decoding.
    if text.startswith("\ufeff"):
        text = text[1:]

    blocks = re.split(r"\n[ \t]*\n", text.strip())
    cues: list[Cue] = []
    skipped = 0
    auto_index = 0

    for block in blocks:
        lines = block.split("\n")
        # Drop leading blank lines inside a block.
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            continue

        timing_at: int | None = None
        for i, ln in enumerate(lines):
            if _ARROW_RE.search(ln):
                timing_at = i
                break
        if timing_at is None:
            skipped += 1
            continue

        timing_line = lines[timing_at]
        halves = _ARROW_RE.split(timing_line, maxsplit=1)
        if len(halves) != 2:
            skipped += 1
            continue
        try:
            start = _parse_timecode(halves[0])
            end = _parse_timecode(halves[1])
        except SubtitleError:
            skipped += 1
            continue

        body = "\n".join(lines[timing_at + 1 :]).strip("\n")
        auto_index += 1

        idx = auto_index
        if timing_at > 0:
            head = lines[timing_at - 1].strip()
            if head.isdigit():
                idx = int(head)

        if end < start:
            # Reversed cue: upstream engines happily produce these. Record it and
            # swap so downstream math stays sane, but keep it visible.
            warnings.append(
                f"cue {idx}: reversed timecode ({format_timecode(start)} > "
                f"{format_timecode(end)}), swapped"
            )
            start, end = end, start
        if end - start > MAX_CUE_SECONDS:
            warnings.append(f"cue {idx}: implausible duration {end - start:.0f}s")

        cues.append(Cue(idx, start, end, body))

    if skipped:
        warnings.append(f"{skipped} block(s) skipped: no usable timecode")
    if not cues:
        raise SubtitleError("no cues parsed from subtitle text", code=E_SUB_EMPTY)

    cues.sort(key=lambda c: (c.start, c.end))
    overlap = _count_overlaps(cues)
    if overlap:
        warnings.append(f"{overlap} overlapping cue pair(s) — legal but suspicious")
    return cues, warnings


def _count_overlaps(cues: Sequence[Cue]) -> int:
    return sum(1 for a, b in pairwise(cues) if b.start < a.end - 1e-6)


def load(path: Path, *, encoding: str | None = None) -> Subtitle:
    path = Path(path)
    if not path.exists():
        raise SubtitleError(f"subtitle not found: {path}", code=E_SUB_MISSING)
    raw = path.read_bytes()
    if not raw.strip():
        raise SubtitleError(f"subtitle is empty: {path.name}", code=E_SUB_EMPTY)
    text, enc = decode_bytes(raw, override=encoding)
    cues, warnings = parse_text(text)
    return Subtitle(cues=cues, encoding=enc, path=path, warnings=warnings)


def serialize_srt(cues: Iterable[Cue], *, sep: str = ",") -> str:
    out: list[str] = []
    for i, c in enumerate(cues, start=1):
        out.append(str(i))
        out.append(f"{format_timecode(c.start, sep=sep)} --> {format_timecode(c.end, sep=sep)}")
        out.append(c.text)
        out.append("")
    return "\n".join(out)


def write_srt(cues: Iterable[Cue], path: Path, *, encoding: str = "utf-8", sep: str = ",") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = serialize_srt(cues, sep=sep)
    path.write_text(body, encoding=encoding, newline="\n")
    return path


def validate_cues(cues: Sequence[Cue]) -> list[str]:
    """Return structural problems. Empty list means the cue list is sound."""
    problems: list[str] = []
    for c in cues:
        if c.end < c.start - 1e-6:
            problems.append(f"cue {c.index}: end < start")
        if c.duration <= 0:
            problems.append(f"cue {c.index}: non-positive duration")
        if c.start < 0:
            problems.append(f"cue {c.index}: negative start")
        if not c.text.strip():
            problems.append(f"cue {c.index}: empty text")
    return problems
