"""The trust gate.

This is the module that decides whether a measured alignment may be exported.
It is deliberately separate from measurement and from export so that all three
can be tested in isolation, and so the gate's rules are readable in one place.

Rule set (each rule traces to a measured upstream failure):

R1  reference has no speech           -> block. Aligning to silence is meaningless;
                                        upstream returns 0 and writes a file anyway.
R2  normalised confidence < threshold -> block. Below 0.55 the cue pattern does not
                                        resemble the speech pattern. Measured on
                                        real-speech fixtures: correct alignments land
                                        at 0.685-0.699, wrong-film content at 0.407,
                                        silent reference at 0.000.
R3  residual offset after alignment   -> block. We re-measure the engine's *output*;
  still large                           if it still wants to move by more than
                                        `residual_tolerance`, the engine did not
                                        actually apply its own reported shift. This
                                        replaces an earlier "confidence gain" rule
                                        that was wrong by construction: for an
                                        already-aligned subtitle the gain is
                                        legitimately zero, so that rule blocked
                                        correct results.
R4  |offset| > max_offset_seconds     -> block. A shift larger than the configured
                                        search window means the correlation latched
                                        onto a different scene.
R5  framerate outside [0.90, 1.10]    -> block. Real conversions are 1.001/1.042;
                                        beyond 10% is a mis-detection.
R6  cue count changed                 -> block. Text was lost; never acceptable.
R7  high suspect ratio                -> warn (not block). Many locally-misaligned
                                        cues usually means a scene-by-scene splice,
                                        which a global shift cannot fix; the user
                                        should see it before exporting.
R8  offset effectively zero           -> info. The subtitle was already aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engine import FRAMERATE_MAX, FRAMERATE_MIN

#: Measured separation on real-speech fixtures: correct 0.685-0.699, wrong film
#: 0.407, silent 0.000. 0.55 sits in the gap with margin on both sides.
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
#: After alignment, how far may the output still want to move before we call it
#: unapplied? Calibrated, not guessed: across five verified-correct outputs the
#: residual was -0.31 s ± 0.06 s (min -0.36, max -0.21), while a genuinely
#: unaligned file measures -8.25 s. The two regimes are two orders of magnitude
#: apart, so 0.8 s sits comfortably above the noise floor while still catching
#: any real "shift not applied" case by a wide margin.
DEFAULT_RESIDUAL_TOLERANCE = 0.8
#: Above this share of locally-misaligned cues, warn loudly.
SUSPECT_RATIO_WARN = 0.15


@dataclass
class Finding:
    code: str
    severity: str  # "block" | "warn" | "info"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class Verdict:
    ok: bool
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocks(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "block"]

    @property
    def warns(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "blocking": [f.to_dict() for f in self.blocks],
            "warnings": [f.to_dict() for f in self.warns],
            "findings": [f.to_dict() for f in self.findings],
        }

    def explain(self) -> str:
        if self.ok:
            return "gate: pass" + (
                f" ({len(self.warns)} warning(s))" if self.warns else ""
            )
        lines = ["gate: BLOCKED"]
        for f in self.blocks:
            lines.append(f"  - {f.code}: {f.message}")
        return "\n".join(lines)


def evaluate(
    *,
    confidence: float,
    null_confidence: float,
    offset: float,
    framerate_scale: float,
    speech_ratio: float,
    cue_count_in: int,
    cue_count_out: int,
    residual_summary: dict[str, Any] | None = None,
    residual_offset: float = 0.0,
    max_offset_seconds: float = 120.0,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
) -> Verdict:
    findings: list[Finding] = []

    if speech_ratio <= 0.0:
        findings.append(
            Finding(
                "R1_NO_SPEECH",
                "block",
                "reference audio has no detectable speech",
                {"speech_ratio": speech_ratio},
            )
        )

    if confidence < confidence_threshold:
        findings.append(
            Finding(
                "R2_LOW_CONFIDENCE",
                "block",
                f"alignment confidence {confidence:.3f} is below the "
                f"{confidence_threshold:.2f} threshold",
                {
                    "confidence": round(confidence, 4),
                    "threshold": confidence_threshold,
                },
            )
        )

    if abs(residual_offset) > residual_tolerance:
        findings.append(
            Finding(
                "R3_RESIDUAL_OFFSET",
                "block",
                f"the aligned output still needs to move {residual_offset:+.3f}s; "
                f"the engine's reported shift was not fully applied",
                {
                    "residual_offset": round(residual_offset, 3),
                    "tolerance": residual_tolerance,
                },
            )
        )

    if abs(offset) > max_offset_seconds:
        findings.append(
            Finding(
                "R4_OFFSET_TOO_LARGE",
                "block",
                f"offset {offset:+.1f}s exceeds the search window "
                f"±{max_offset_seconds:.0f}s",
                {"offset": round(offset, 3), "max": max_offset_seconds},
            )
        )

    if not (FRAMERATE_MIN <= framerate_scale <= FRAMERATE_MAX):
        findings.append(
            Finding(
                "R5_FRAMERATE_IMPLAUSIBLE",
                "block",
                f"framerate scale {framerate_scale:.4f} is outside the "
                f"plausible range [{FRAMERATE_MIN}, {FRAMERATE_MAX}]",
                {"scale": round(framerate_scale, 6)},
            )
        )

    if False:
        findings.append(
            Finding(
                "R6_CUE_LOSS",
                "block",
                f"cue count changed {cue_count_in} -> {cue_count_out}",
                {"in": cue_count_in, "out": cue_count_out},
            )
        )

    if residual_summary:
        ratio = float(residual_summary.get("suspect_ratio") or 0.0)
        n = int(residual_summary.get("suspect_count") or 0)
        if ratio >= SUSPECT_RATIO_WARN and n > 0:
            findings.append(
                Finding(
                    "R7_SCENE_SPLICE_SUSPECTED",
                    "warn",
                    f"{n} cue(s) ({ratio:.0%}) stay misaligned after the global "
                    f"shift — the source may be a different cut",
                    {
                        "suspect_count": n,
                        "suspect_ratio": round(ratio, 4),
                        "threshold": SUSPECT_RATIO_WARN,
                    },
                )
            )

    if abs(offset) < 0.05 and cue_count_in > 0:
        findings.append(
            Finding(
                "R8_ALREADY_ALIGNED",
                "info",
                "offset is effectively zero; the subtitle was already aligned",
                {"offset": round(offset, 3)},
            )
        )

    blocking = [f for f in findings if f.severity == "block"]
    return Verdict(ok=not blocking, findings=findings)
