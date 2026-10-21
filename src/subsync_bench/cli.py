"""CLI: the scriptable face of the workbench.

Subcommands mirror the workbench's stages, and each one is usable on its own:

    doctor    check the environment before anything else
    inspect   measure inputs, no engine needed
    align     produce a revision (recorded, gated)
    preview   render timeline + cue diff for a revision
    revs      list revisions and their verdicts
    select    choose which revision export will use
    export    write the final subtitle, re-checking the gate
    report    one self-contained HTML with everything

Exit codes are part of the contract (CI depends on them):
    0  success
    1  unexpected error
    2  bad arguments
    3  environment problem (missing ffmpeg/engine)
    4  trust gate blocked the result
    5  workspace / state problem
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__, audio, engine, pipeline, preview, subio
from .errors import (
    E_BAD_ARGS,
    E_ENGINE_MISSING,
    E_FFMPEG_MISSING,
    BenchError,
)
from .workspace import Project, Revision

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ARGS = 2
EXIT_ENV = 3
EXIT_GATE = 4
EXIT_STATE = 5

#: Missing tooling or unusable media — the user must fix the environment/inputs.
_ENV_CODES = {
    E_FFMPEG_MISSING,
    E_ENGINE_MISSING,
    "E_VIDEO_MISSING",
    "E_VIDEO_NO_AUDIO",
    "E_SUB_MISSING",
    "E_ENCODING_UNDETECTED",
}
#: The result exists but must not be trusted. Distinct from 3 so CI can tell
#: "I could not run" apart from "I ran and refused the answer".
_GATE_CODES = {
    "E_EXPORT_BLOCKED",
    "E_ALIGN_FAILED",
    "E_ALIGN_LOW_CONFIDENCE",
    "E_VIDEO_SILENT",
    "E_ALIGN_NO_SPEECH",
}


def _same_file(a: Path, b: Path) -> bool:
    """Compare by content, not path: users copy fixtures around constantly."""
    try:
        if not a.exists() or not b.exists():
            return False
        if a.stat().st_size != b.stat().st_size:
            return False
        from .workspace import sha256_file

        return sha256_file(a) == sha256_file(b)
    except OSError:
        return False


def _reconfigure_streams() -> None:
    """Windows consoles default to cp936/cp1252; our output is UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):  # pragma: no cover
            pass


def _emit(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _fail(exc: BenchError, *, as_json: bool) -> int:
    if as_json:
        _emit({"ok": False, "error": exc.to_dict()})
    else:
        sys.stderr.write(f"error [{exc.code}] {exc.message}\n")
        if exc.hint:
            sys.stderr.write(f"  hint: {exc.hint}\n")
    if exc.code in _ENV_CODES:
        return EXIT_ENV
    if exc.code in _GATE_CODES:
        return EXIT_GATE
    if exc.code == E_BAD_ARGS:
        return EXIT_ARGS
    if exc.code.startswith("E_PROJECT") or exc.code.startswith("E_RECEIPT") or exc.code in (
        "E_NO_DECISION",
        "E_EXPORT_EXISTS",
    ):
        return EXIT_STATE
    return EXIT_ERROR


# --- doctor ---------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        checks.append(
            {"check": name, "ok": ok, "required": required, "detail": detail}
        )

    import shutil

    ff = shutil.which("ffmpeg")
    add("ffmpeg", bool(ff), ff or "not found on PATH")
    fp = shutil.which("ffprobe")
    add(
        "ffprobe",
        bool(fp),
        fp or "not found — upstream cannot enumerate subtitle streams, "
        "audio-only reference still works",
        required=False,
    )
    add("engine:ffsubsync", engine.engine_available(),
        f"ffsubsync {engine.engine_version()}" if engine.engine_available()
        else "not installed — align unavailable, inspect/preview/export still work",
        required=False)
    try:
        import numpy

        add("numpy", True, numpy.__version__)
    except Exception as exc:  # pragma: no cover
        add("numpy", False, f"missing: {exc}")

    hard_ok = all(c["ok"] for c in checks if c["required"])
    out = {
        "ok": hard_ok,
        "version": __version__,
        "checks": checks,
        "capabilities": {
            "inspect": hard_ok,
            "align": hard_ok and engine.engine_available(),
            "preview": hard_ok,
            "export": hard_ok,
        },
    }
    if args.json:
        _emit(out)
    else:
        sys.stdout.write(f"subsync-bench {__version__}\n")
        for c in checks:
            mark = "OK  " if c["ok"] else ("WARN" if not c["required"] else "FAIL")
            sys.stdout.write(f"  [{mark}] {c['check']:<18} {c['detail']}\n")
        caps = out["capabilities"]
        sys.stdout.write(
            "\ncapabilities: "
            + "  ".join(f"{k}={'yes' if v else 'no'}" for k, v in caps.items())
            + "\n"
        )
    return EXIT_OK if hard_ok else EXIT_ENV


# --- inspect --------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace) -> int:
    insp = pipeline.inspect(
        Path(args.video),
        Path(args.subtitle),
        encoding=args.encoding,
        with_audio=not args.no_audio,
    )
    if args.json:
        _emit({"ok": True, "inspection": insp.to_dict()})
        return EXIT_OK
    d = insp.to_dict()
    s = d["subtitle"]
    sys.stdout.write(f"subtitle : {s['path']}\n")
    sys.stdout.write(
        f"  encoding={s['encoding']}  cues={s['cue_count']}  "
        f"span={s['span'][0]:.1f}-{s['span'][1]:.1f}s  chars={s['text_chars']}\n"
    )
    if d["envelope"]:
        e = d["envelope"]
        sys.stdout.write(
            f"reference: {e['duration']:.1f}s  speech_ratio={e['speech_ratio']:.3f}  "
            f"silent={e['is_silent']}\n"
        )
    if d["structural_problems"]:
        sys.stdout.write("\nstructural problems:\n")
        for p in d["structural_problems"][:20]:
            sys.stdout.write(f"  - {p}\n")
    if d["warnings"]:
        sys.stdout.write("\nwarnings:\n")
        for w in d["warnings"][:20]:
            sys.stdout.write(f"  - {w}\n")
    return EXIT_OK


# --- align ----------------------------------------------------------------


def cmd_align(args: argparse.Namespace) -> int:
    root = Path(args.project)
    if args.video and args.subtitle:
        if (root / "project.json").exists():
            # A project already exists. Silently ignoring the new --video /
            # --subtitle would align the OLD inputs while the user believes they
            # are testing new ones — the exact class of quiet wrongness this tool
            # exists to prevent. Make them say what they mean.
            existing = Project.open(root)
            same_v = _same_file(existing.video_path, Path(args.video))
            same_s = _same_file(existing.subtitle_path, Path(args.subtitle))
            if not (same_v and same_s):
                if not args.replace_inputs:
                    raise BenchError(
                        f"project already exists at {root} with different inputs",
                        code="E_PROJECT_EXISTS",
                        context={
                            "existing_video": str(existing.video_path),
                            "existing_subtitle": str(existing.subtitle_path),
                        },
                        hint="pass --replace-inputs to swap them (existing revisions "
                        "become stale), or use a new project directory",
                    )
                existing.replace_inputs(
                    video=Path(args.video), subtitle=Path(args.subtitle)
                )
            proj = existing
        else:
            proj = Project.create(
                root,
                video=Path(args.video),
                subtitle=Path(args.subtitle),
                name=args.name,
            )
    else:
        proj = Project.open(root)

    rev, verdict = pipeline.align_project(
        proj,
        label=args.label,
        max_offset_seconds=args.max_offset_seconds,
        framerate_ratio=args.framerate_ratio,
        encoding=args.encoding,
        confidence_threshold=args.confidence_threshold,
        residual_threshold=args.residual_threshold,
    )
    ok = rev.status != "rejected"
    out = {
        "ok": ok,
        "revision": rev.to_dict(),
        "verdict": verdict,
        "project": str(proj.root),
    }
    if args.json:
        _emit(out)
    else:
        sys.stdout.write(f"revision {rev.id}  [{rev.status.upper()}]\n")
        sys.stdout.write(
            f"  offset={rev.offset:+.3f}s  scale={rev.framerate_scale:.6f}  "
            f"cues={rev.cue_count}\n"
        )
        sys.stdout.write(
            f"  confidence={rev.confidence:.3f} (no-shift baseline "
            f"{rev.null_confidence:.3f})\n"
        )
        rs = rev.residual_summary
        sys.stdout.write(
            f"  residuals: {rs.get('suspect_count', 0)}/{rs.get('cue_count', 0)} "
            f"suspect, median |r|={rs.get('median_abs_residual', 0):.3f}s\n"
        )
        if verdict["blocking"]:
            sys.stdout.write("\n  BLOCKED:\n")
            for f in verdict["blocking"]:
                sys.stdout.write(f"    - {f['code']}: {f['message']}\n")
        for f in verdict["warnings"]:
            sys.stdout.write(f"  warn: {f['code']}: {f['message']}\n")
        if ok:
            sys.stdout.write(f"\nnext: subsync-bench select {rev.id} && subsync-bench export\n")
    return EXIT_OK if ok else EXIT_GATE


# --- preview --------------------------------------------------------------


def cmd_preview(args: argparse.Namespace) -> int:
    proj = Project.open(Path(args.project))
    if args.revision:
        rev = proj.get(args.revision)
    else:
        # Prefer the explicit selection; fall back to the newest revision. Asking
        # the user to `select` before they may even look at a preview is friction
        # with no safety benefit — preview never writes outside the project.
        rev = proj.selected() or (proj.revisions[-1] if proj.revisions else None)
    if rev is None:
        raise BenchError(
            "no revision to preview",
            code="E_NO_DECISION",
            hint="run `align` first",
        )
    before = subio.load(proj.subtitle_path).cues
    after = pipeline.load_revision_cues(proj, rev.id)

    env: audio.Envelope | None = None
    if not args.no_audio:
        try:
            env = audio.envelope_from_media(proj.video_path)
        except BenchError:
            env = None

    residuals = (
        audio.cue_residuals(env, after, 0.0,
                            onset_lag=audio.measure_onset_lag(env, after))
        if env else None
    )

    spec = preview.TimelineSpec()
    if args.start is not None or args.end is not None:
        spec.view_start = args.start or 0.0
        spec.view_end = args.end

    svg = preview.render_timeline(
        env, before, after, residuals=residuals, spec=spec,
        title=f"{proj.manifest.get('name', 'project')} · {rev.id}",
    )
    rows = preview.build_diff(before, after, residuals)

    out_path = Path(args.out) if args.out else proj.work_dir / f"{rev.id}.preview.html"
    html_doc = _preview_html(proj, rev, svg, rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8", newline="\n")

    if args.svg_out:
        Path(args.svg_out).write_text(svg, encoding="utf-8", newline="\n")

    payload = {
        "ok": True,
        "revision": rev.id,
        "html": str(out_path),
        "cue_rows": len(rows),
        "suspect_rows": sum(1 for r in rows if r.residual is not None and abs(r.residual) >= 0.5),
    }
    if args.json:
        _emit(payload)
    else:
        sys.stdout.write(f"preview -> {out_path}\n")
        sys.stdout.write(
            f"  {payload['cue_rows']} cues, {payload['suspect_rows']} flagged\n"
        )
    return EXIT_OK


def _preview_html(proj: Project, rev: Revision, svg: str, rows) -> str:
    from .report import page

    body = [
        "<section class='card'>",
        "<h2>timeline</h2>",
        svg,
        "</section>",
        "<section class='card'>",
        "<h2>measurements</h2>",
        "<dl class='kv'>",
        f"<dt>revision</dt><dd>{rev.id} — {rev.label}</dd>",
        f"<dt>status</dt><dd class='st-{rev.status}'>{rev.status}</dd>",
        f"<dt>offset</dt><dd>{rev.offset:+.3f}s</dd>",
        f"<dt>framerate scale</dt><dd>{rev.framerate_scale:.6f}</dd>",
        f"<dt>confidence</dt><dd>{rev.confidence:.3f} (baseline {rev.null_confidence:.3f})</dd>",
        f"<dt>cues</dt><dd>{rev.cue_count}</dd>",
        f"<dt>suspect cues</dt><dd>{rev.residual_summary.get('suspect_count', 0)}</dd>",
        "</dl>",
        "</section>",
        "<section class='card'>",
        "<h2>per-cue diff</h2>",
        preview.render_diff_table(rows),
        "</section>",
    ]
    return page(
        title=f"{proj.manifest.get('name', 'project')} · {rev.id} preview",
        body="\n".join(body),
        subtitle=f"revision {rev.id} · {rev.status}",
    )


# --- revs / select / export ----------------------------------------------


def cmd_revs(args: argparse.Namespace) -> int:
    proj = Project.open(Path(args.project))
    revs = proj.revisions
    if args.json:
        _emit({"ok": True, "selected": proj.selected_id,
               "revisions": [r.to_dict() for r in revs]})
        return EXIT_OK
    if not revs:
        sys.stdout.write("no revisions yet — run `align`\n")
        return EXIT_OK
    sel = proj.selected_id
    sys.stdout.write(f"{'id':<6}{'status':<10}{'offset':>10}{'scale':>10}"
                     f"{'conf':>7}{'suspect':>9}  label\n")
    sys.stdout.write("-" * 78 + "\n")
    for r in revs:
        mark = "*" if r.id == sel else " "
        sys.stdout.write(
            f"{mark}{r.id:<5}{r.status:<10}{r.offset:>+10.3f}"
            f"{r.framerate_scale:>10.4f}{r.confidence:>7.3f}"
            f"{r.residual_summary.get('suspect_count', 0):>9}  {r.label}\n"
        )
    return EXIT_OK


def cmd_select(args: argparse.Namespace) -> int:
    proj = Project.open(Path(args.project))
    rev = proj.select(args.revision)
    out = {"ok": True, "selected": rev.id, "status": rev.status}
    if args.json:
        _emit(out)
    else:
        sys.stdout.write(f"selected {rev.id} ({rev.status})\n")
    return EXIT_OK


def cmd_export(args: argparse.Namespace) -> int:
    proj = Project.open(Path(args.project))
    info = pipeline.export_selected(
        proj,
        Path(args.out) if args.out else None,
        encoding=args.encoding,
        force=args.force,
        allow_override=args.allow_override,
        skip_recheck=args.skip_recheck,
    )
    if args.json:
        _emit({"ok": True, "export": info})
    else:
        sys.stdout.write(f"exported -> {info['path']}\n")
        sys.stdout.write(
            f"  {info['cue_count']} cues, {info['bytes']} bytes, "
            f"{info['encoding']}, sha256={info['sha256'][:16]}…\n"
        )
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report

    proj = Project.open(Path(args.project))
    out = Path(args.out) if args.out else proj.root / "report.html"
    path = build_report(proj, out)
    if args.json:
        _emit({"ok": True, "report": str(path)})
    else:
        sys.stdout.write(f"report -> {path}\n")
    return EXIT_OK


# --- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subsync-bench",
        description="A workbench for getting a downloaded subtitle to actually "
        "line up with a video: inspect, align, preview, compare, export.",
    )
    p.add_argument("--version", action="version", version=f"subsync-bench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check ffmpeg / engine availability")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_doctor)

    i = sub.add_parser("inspect", help="measure inputs without aligning")
    i.add_argument("video")
    i.add_argument("subtitle")
    i.add_argument("--encoding", default=None)
    i.add_argument("--no-audio", action="store_true", help="skip envelope extraction")
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_inspect)

    a = sub.add_parser("align", help="align and record a revision")
    a.add_argument("project")
    a.add_argument("--video", default=None)
    a.add_argument("--subtitle", default=None)
    a.add_argument("--name", default=None)
    a.add_argument("--label", default=None)
    a.add_argument("--max-offset-seconds", type=float, default=None)
    a.add_argument("--framerate-ratio", type=float, default=None)
    a.add_argument("--encoding", default=None)
    a.add_argument("--confidence-threshold", type=float, default=None)
    a.add_argument("--residual-threshold", type=float, default=None)
    a.add_argument("--force", action="store_true")
    a.add_argument(
        "--replace-inputs",
        action="store_true",
        help="replace the project's video/subtitle (existing revisions become stale)",
    )
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_align)

    pv = sub.add_parser("preview", help="render timeline + cue diff")
    pv.add_argument("project")
    pv.add_argument("--revision", default=None)
    pv.add_argument("--out", default=None)
    pv.add_argument("--svg-out", default=None)
    pv.add_argument("--start", type=float, default=None)
    pv.add_argument("--end", type=float, default=None)
    pv.add_argument("--no-audio", action="store_true")
    pv.add_argument("--json", action="store_true")
    pv.set_defaults(func=cmd_preview)

    r = sub.add_parser("revs", help="list revisions")
    r.add_argument("project")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_revs)

    s = sub.add_parser("select", help="choose the revision to export")
    s.add_argument("project")
    s.add_argument("revision")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_select)

    e = sub.add_parser("export", help="write the final subtitle")
    e.add_argument("project")
    e.add_argument("--out", default=None)
    e.add_argument("--encoding", default=None)
    e.add_argument("--force", action="store_true")
    e.add_argument("--allow-override", action="store_true",
                   help="export even if the trust gate rejected the revision "
                        "(the user has inspected it by hand)")
    e.add_argument("--skip-recheck", action="store_true",
                   help="DANGEROUS: skip the export-time re-measurement. "
                        "Only for offline/archival re-export of a revision you "
                        "already verified.")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_export)

    rp = sub.add_parser("report", help="build a self-contained HTML report")
    rp.add_argument("project")
    rp.add_argument("--out", default=None)
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    _reconfigure_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    as_json = bool(getattr(args, "json", False))
    try:
        return int(args.func(args))
    except BenchError as exc:
        return _fail(exc, as_json=as_json)
    except KeyboardInterrupt:  # pragma: no cover
        sys.stderr.write("interrupted\n")
        return EXIT_ERROR
    except Exception as exc:  # pragma: no cover - last resort
        if as_json:
            _emit({"ok": False, "error": {"code": "E_INTERNAL", "message": str(exc)}})
        else:
            import traceback

            traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
