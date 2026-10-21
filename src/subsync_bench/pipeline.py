"""The pipeline: ingest -> measure -> align -> verify -> record.

Every public entry point here takes and returns plain data plus files on disk, so
the CLI, the web workbench and the tests all drive the same code path. There is
no second implementation for the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import audio, engine, gate, subio
from .errors import (
    E_NO_DECISION,
    E_RECEIPT_MISSING,
    AlignError,
    WorkspaceError,
)
from .subio import Cue, Subtitle
from .workspace import Project, Revision, _now


@dataclass
class Inspection:
    """Everything measurable about the inputs, before any alignment."""

    subtitle: Subtitle
    envelope: audio.Envelope | None
    structural: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subtitle": self.subtitle.to_dict(),
            "envelope": self.envelope.to_dict() if self.envelope else None,
            "structural_problems": list(self.structural),
            "warnings": list(self.warnings),
        }


def inspect(
    video: Path,
    subtitle: Path,
    *,
    encoding: str | None = None,
    with_audio: bool = True,
) -> Inspection:
    """Measure inputs. Works without ffsubsync installed."""
    sub = subio.load(subtitle, encoding=encoding)
    structural = subio.validate_cues(sub.cues)
    warnings = list(sub.warnings)
    env: audio.Envelope | None = None
    if with_audio:
        env = audio.envelope_from_media(video)
        if env.is_silent:
            warnings.append(
                "reference audio looks silent — alignment would be meaningless"
            )
        if sub.cues:
            sub_span = sub.span
            if env.duration + 1.0 < sub_span[1]:
                warnings.append(
                    f"subtitle runs to {sub_span[1]:.0f}s but reference is only "
                    f"{env.duration:.0f}s long — likely a different cut or "
                    f"framerate"
                )
    return Inspection(subtitle=sub, envelope=env, structural=structural, warnings=warnings)


def align_project(
    project: Project,
    *,
    label: str | None = None,
    max_offset_seconds: float | None = None,
    framerate_ratio: float | None = None,
    encoding: str | None = None,
    confidence_threshold: float | None = None,
    residual_threshold: float | None = None,
) -> tuple[Revision, dict[str, Any]]:
    """Run one alignment attempt and persist it as a revision.

    Returns (revision, verdict_dict). The revision is always recorded — including
    rejected attempts — because knowing that an attempt was rejected is itself
    useful evidence.
    """
    s = project.settings
    max_off = float(
        max_offset_seconds if max_offset_seconds is not None else s.get("max_offset_seconds", 120.0)
    )
    conf_thr = float(
        confidence_threshold
        if confidence_threshold is not None
        else s.get("confidence_threshold", gate.DEFAULT_CONFIDENCE_THRESHOLD)
    )
    res_thr = float(
        residual_threshold
        if residual_threshold is not None
        else s.get("residual_threshold", 0.5)
    )

    sub = subio.load(project.subtitle_path, encoding=encoding)
    env = audio.envelope_from_media(project.video_path)

    if env.is_silent:
        # Refuse before spending time on the engine; upstream would happily
        # "succeed" here.
        raise AlignError(
            "reference audio contains no detectable speech",
            code="E_VIDEO_SILENT",
            context={"speech_ratio": round(env.speech_ratio, 5)},
            hint="verify the video has an audio track with dialogue; "
            "a silent reference cannot be aligned against",
        )

    rid = project.next_revision_id()
    staged = project.work_dir / f"{rid}.engine.srt"
    res = engine.run_engine(
        project.video_path,
        project.subtitle_path,
        staged,
        max_offset_seconds=max_off,
        framerate_ratio=framerate_ratio,
        encoding=encoding,
    )

    warnings: list[str] = list(res.warnings)
    if res.output_path is None:
        raise AlignError(
            "engine produced no output file",
            code="E_ALIGN_FAILED",
            context={"log_tail": res.log_tail[-800:]},
        )

    out_cues, problems = engine.verify_output(staged, len(sub.cues))

    # --- automatic downgrade on cue loss --------------------------------
    # Measured upstream defect: when the framerate guess is wrong (e.g. it tries
    # 23.976/25 on a file that needs no rescale) the scaled cues go negative, and
    # pysubs2 silently *drops* them while writing. A 14-cue file came back with 4
    # cues and retval=0. Losing dialogue is never acceptable, so when cue count
    # changes we retry once with framerate fixing disabled and keep whichever
    # attempt preserved every cue.
    if len(out_cues) != len(sub.cues) and framerate_ratio is None:
        retry_staged = project.work_dir / f"{rid}.engine.nofps.srt"
        retry = engine.run_engine(
            project.video_path,
            project.subtitle_path,
            retry_staged,
            max_offset_seconds=max_off,
            encoding=encoding,
            no_fix_framerate=True,
        )
        if retry.output_path is not None:
            retry_cues, retry_problems = engine.verify_output(
                retry_staged, len(sub.cues)
            )
            if len(retry_cues) == len(sub.cues):
                warnings.append(
                    f"framerate guess was destructive "
                    f"({len(out_cues)}/{len(sub.cues)} cues survived); retried "
                    f"with --no-fix-framerate, all cues preserved"
                )
                res = retry
                out_cues, problems = retry_cues, retry_problems
            else:
                warnings.append(
                    f"retry without framerate fixing also lost cues "
                    f"({len(retry_cues)}/{len(sub.cues)})"
                )
        else:
            warnings.append("retry without framerate fixing produced no output")
    warnings.extend(problems)

    # Re-measure the engine's own output. This is the step upstream omits, and
    # the reason its success flag cannot be trusted.
    #
    # Two distinct quantities:
    #   measured.confidence  — does the aligned output actually sit on speech?
    #   measured.offset      — how far the output *still* wants to move, i.e.
    #                          evidence the engine did not apply its own shift.
    #
    # The envelope lags real speech by ~0.18 s (RMS window + smoothing), which
    # biases every estimate early by that amount. We measure that lag on the
    # aligned cues and delay the envelope to cancel it. Without this, a perfect
    # alignment reads as -0.12 s and the residual gate fires on correct work.
    onset = audio.measure_onset_lag(env, out_cues)
    aligned_env = env.delay(onset) if onset > 0 else env
    measured = audio.align_envelope(aligned_env, out_cues, max_offset_seconds=max_off)
    residuals = audio.cue_residuals(env, out_cues, 0.0, onset_lag=onset)
    summary = audio.summarize_residuals(residuals, threshold=res_thr)
    summary["onset_lag_compensated"] = round(onset, 3)

    verdict = gate.evaluate(
        confidence=measured.confidence,
        null_confidence=measured.null_confidence,
        offset=res.offset,
        framerate_scale=res.framerate_scale,
        speech_ratio=env.speech_ratio,
        cue_count_in=len(sub.cues),
        cue_count_out=len(out_cues),
        residual_summary=summary,
        residual_offset=measured.offset,
        max_offset_seconds=max_off,
        confidence_threshold=conf_thr,
    )

    status = "ok" if verdict.ok else "rejected"
    if verdict.ok and summary.get("suspect_count"):
        status = "suspect"

    # Persist the artifact regardless of verdict: a rejected revision is still
    # inspectable, which is how a user diagnoses "why did this not work".
    artifact_rel = f"work/{rid}.aligned.srt"
    artifact_abs = project.root / artifact_rel
    subio.write_srt(out_cues, artifact_abs)

    rev = Revision(
        id=rid,
        created=_now(),
        label=label or f"attempt {rid}",
        offset=res.offset,
        framerate_scale=res.framerate_scale,
        confidence=measured.confidence,
        null_confidence=measured.null_confidence,
        speech_ratio=env.speech_ratio,
        cue_count=len(out_cues),
        residual_offset=measured.offset,
        residual_summary=summary,
        engine=res.to_dict(),
        inputs={
            "video": project.manifest["video"]["sha256"],
            "subtitle": project.manifest["subtitle"]["sha256"],
        },
        status=status,
        notes=warnings,
        artifact=artifact_rel,
        rejected_reason="; ".join(f.code for f in verdict.blocks) or None,
    )
    project.add_revision(rev)
    return rev, verdict.to_dict()


def load_revision_cues(project: Project, rev_id: str) -> list[Cue]:
    rev = project.get(rev_id)
    path = project.revision_artifact(rev)
    return subio.load(path).cues


def select_revision(project: Project, rev_id: str) -> Revision:
    return project.select(rev_id)


def _recheck_before_export(
    project: Project, cues: list[Cue], out: dict[str, Any]
) -> None:
    """Re-measure the cues about to be written, against the video on disk now.

    Raises ``E_EXPORT_BLOCKED`` when the result would be untrustworthy. Mutates
    ``out`` with the measurements so the caller can report them either way.
    """
    env = audio.envelope_from_media(project.video_path)
    out["speech_ratio"] = round(env.speech_ratio, 4)
    if env.is_silent:
        raise WorkspaceError(
            "re-check at export time failed: the reference audio is silent",
            code="E_EXPORT_BLOCKED",
            context=dict(out),
            hint="the video file has changed since alignment",
        )

    max_off = float(project.settings.get("max_offset_seconds", 120.0))
    m = audio.align_envelope(env, cues, max_offset_seconds=max_off)
    tol = float(
        project.settings.get("residual_tolerance", gate.DEFAULT_RESIDUAL_TOLERANCE)
    )
    thr = float(
        project.settings.get(
            "confidence_threshold", gate.DEFAULT_CONFIDENCE_THRESHOLD
        )
    )
    out.update(
        {
            "reverified": True,
            "confidence": round(m.confidence, 4),
            "residual_offset": round(m.offset, 3),
        }
    )
    if m.confidence < thr:
        raise WorkspaceError(
            f"re-check at export time failed: confidence {m.confidence:.3f} "
            f"< {thr:.2f}",
            code="E_EXPORT_BLOCKED",
            context=dict(out),
            hint="the video or subtitle may have changed since alignment",
        )
    if abs(m.offset) > tol:
        raise WorkspaceError(
            f"re-check at export time failed: output still wants to move "
            f"{m.offset:+.3f}s (> {tol:.2f}s tolerance)",
            code="E_EXPORT_BLOCKED",
            context=dict(out),
        )


def export_selected(
    project: Project,
    dest: Path | None = None,
    *,
    encoding: str | None = None,
    force: bool = False,
    allow_override: bool = False,
    skip_recheck: bool = False,
) -> dict[str, Any]:
    """Export the selected revision, re-checking the gate at export time.

    Four distinct permissions, deliberately not merged:

    ``force``
        Overwrite an existing destination, and proceed when inputs changed since
        the revision was recorded. Says nothing about correctness.
    ``allow_override``
        Export a revision the gate rejected. The user asserts they have inspected
        the output themselves. Does NOT skip the physical re-check.
    ``skip_recheck``
        Skip the export-time re-measurement entirely. This is the only switch that
        turns off the physics check, and it is separate so that ``--force`` can
        never silently disable it — a bug this design exists to prevent.

    The re-check re-measures rather than trusting the receipt: if the video was
    swapped for a different file, the receipt's numbers no longer describe the
    bytes about to be written.
    """
    rev = project.selected()
    if rev is None:
        raise WorkspaceError(
            "no revision selected — run `select <id>` first",
            code=E_NO_DECISION,
        )
    if rev.status == "rejected" and not allow_override:
        raise WorkspaceError(
            f"selected revision {rev.id} was rejected by the trust gate "
            f"({rev.rejected_reason})",
            code="E_EXPORT_BLOCKED",
            hint="pick a passing revision, or pass --allow-override if you have "
            "verified the output by hand",
        )

    integrity = project.verify_inputs()
    if rev.id in integrity["stale"] and not force:
        raise WorkspaceError(
            f"inputs changed since revision {rev.id} was created; its "
            f"measurements are stale",
            code=E_RECEIPT_MISSING,
            hint="re-run align, or pass --force to export anyway",
        )

    src = project.revision_artifact(rev)
    enc = encoding or project.settings.get("output_encoding", "utf-8")
    if dest is None:
        stem = project.subtitle_path.stem
        dest = project.out_dir / f"{stem}.synced.srt"
    dest = Path(dest)
    if dest.exists() and not force:
        raise WorkspaceError(
            f"destination exists: {dest}",
            code="E_EXPORT_EXISTS",
            hint="pass --force to overwrite",
        )
    dest.parent.mkdir(parents=True, exist_ok=True)

    cues = subio.load(src).cues
    problems = subio.validate_cues(cues)
    if problems:
        raise WorkspaceError(
            f"revision {rev.id} has structural problems and cannot be exported: "
            f"{problems[0]}",
            code="E_EXPORT_BLOCKED",
            context={"problems": problems[:10]},
        )

    # Re-measure at export time. This runs on EVERY export, including --force and
    # --allow-override. Those flags mean "I know the receipt is stale" and "I have
    # looked at a rejected result", not "skip the physics check": a video swapped
    # for a silent one must never be exportable just by adding --force.
    live_check: dict[str, Any] = {"reverified": False, "skipped": skip_recheck}
    if not skip_recheck:
        _recheck_before_export(project, cues, live_check)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    subio.write_srt(cues, tmp, encoding=enc)
    tmp.replace(dest)

    digest = __import__("hashlib").sha256(dest.read_bytes()).hexdigest()
    return {
        "revision": rev.id,
        "path": str(dest),
        "bytes": dest.stat().st_size,
        "sha256": digest,
        "cue_count": len(cues),
        "encoding": enc,
        "offset": round(rev.offset, 3),
        "confidence": round(rev.confidence, 4),
        "status": rev.status,
        "live_check": live_check,
    }
