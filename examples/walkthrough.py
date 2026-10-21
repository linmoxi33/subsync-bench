"""End-to-end walkthrough that prints what each stage consumes and produces.

This exists to make the *coupling* visible. It is easy to claim "these modules
work together"; this script shows the actual handoff at each boundary:

    import  → measure   : the envelope built here is what the gate later trusts
    measure → align     : the same envelope re-measures the engine's output
    align   → preview   : the per-cue residuals decide which cues get outlined
    preview → compare   : the revision's status decides what export will accept
    compare → export    : the receipt's digests decide whether export is stale
    export  → report    : the re-check numbers land in the report

Run:

    python examples/walkthrough.py            # uses the offline fixtures
    python examples/walkthrough.py --keep     # keep the project directory
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from subsync_bench import audio, pipeline, preview, report, subio  # noqa: E402
from subsync_bench.errors import BenchError  # noqa: E402
from subsync_bench.workspace import Project  # noqa: E402

FIXTURES = ROOT / "fixtures" / "media"


def rule(title: str) -> None:
    print(f"\n{'─' * 74}\n{title}\n{'─' * 74}")


def ensure_fixtures() -> Path:
    film = FIXTURES / "film.mp4"
    if film.exists() and (FIXTURES / "late.srt").exists():
        return FIXTURES
    print("fixtures missing, building them (offline, ffmpeg only)…")
    subprocess.run(
        [sys.executable, str(ROOT / "fixtures" / "build_offline.py")], check=True
    )
    return FIXTURES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the project dir")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    media = ensure_fixtures()
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="subsync-demo-"))
    work.mkdir(parents=True, exist_ok=True)
    proj_dir = work / "project"

    # ---------------------------------------------------------------- import
    rule("1. IMPORT — copy inputs in and hash them")
    proj = Project.create(proj_dir, video=media / "film.mp4", subtitle=media / "late.srt",
                          name="walkthrough")
    print(f"  project      : {proj.root}")
    print(f"  video sha256 : {proj.manifest['video']['sha256'][:32]}…")
    print(f"  sub   sha256 : {proj.manifest['subtitle']['sha256'][:32]}…")
    print("  → these two digests are what every later stage checks against")

    # --------------------------------------------------------------- measure
    rule("2. MEASURE — build the envelope that the gate will trust")
    insp = pipeline.inspect(proj.video_path, proj.subtitle_path)
    env = insp.envelope
    assert env is not None
    print(f"  subtitle : {len(insp.subtitle)} cues, encoding={insp.subtitle.encoding}")
    print(f"  envelope : {env.duration:.1f}s, speech_ratio={env.speech_ratio:.3f}, "
          f"silent={env.is_silent}")
    for w in insp.warnings:
        print(f"  warning  : {w}")
    print("  → this envelope is reused verbatim by align, preview and export")

    # ----------------------------------------------------------------- align
    rule("3. ALIGN — run the engine, then re-measure its own output")
    rev, verdict = pipeline.align_project(proj, label="walkthrough")
    print(f"  revision : {rev.id}  status={rev.status}")
    print(f"  offset   : {rev.offset:+.3f}s   framerate_scale={rev.framerate_scale:.6f}")
    print(f"  confidence: {rev.confidence:.4f}  (no-shift baseline {rev.null_confidence:.4f})")
    print(f"  residual offset after align: {rev.residual_offset:+.3f}s")
    rs = rev.residual_summary
    print(f"  per-cue  : {rs['suspect_count']}/{rs['cue_count']} suspect, "
          f"median |onset Δ|={rs['median_abs_onset_distance']:.3f}s")
    print(f"  gate     : {'PASS' if verdict['ok'] else 'BLOCKED'}")
    for f in verdict["blocking"]:
        print(f"    block: {f['code']} — {f['message']}")
    for f in verdict["warnings"]:
        print(f"    warn : {f['code']} — {f['message']}")
    print("  → the residual list produced here drives the preview highlighting")

    # --------------------------------------------------------------- preview
    rule("4. PREVIEW — the residuals decide which cues get outlined")
    before = subio.load(proj.subtitle_path).cues
    after = pipeline.load_revision_cues(proj, rev.id)
    residuals = audio.cue_residuals(
        env, after, 0.0, onset_lag=audio.measure_onset_lag(env, after)
    )
    suspects = [r.index for r in residuals if r.suspect]
    svg = preview.render_timeline(
        env, before, after, residuals=residuals,
        title=f"{proj.manifest['name']} · {rev.id}",
    )
    preview_path = work / "preview.html"
    preview_path.write_text(
        report.page(
            title="walkthrough preview",
            subtitle=f"revision {rev.id} · {rev.status}",
            body=(
                f"<section class='card'><h2>timeline</h2>{svg}</section>"
                f"<section class='card'><h2>per-cue diff</h2>"
                f"{preview.render_diff_table(preview.build_diff(before, after, residuals))}"
                f"</section>"
            ),
        ),
        encoding="utf-8",
    )
    print(f"  suspect cues : {suspects or 'none'}")
    print(f"  svg bytes    : {len(svg)}   html → {preview_path}")
    print("  → outlines in the SVG come from the same objects printed above")

    # --------------------------------------------------------------- compare
    rule("5. COMPARE — history is kept, and the pipeline is deterministic")
    rev2, _ = pipeline.align_project(proj, label="second pass")
    print(f"  {rev.id}: conf={rev.confidence:.4f} status={rev.status} "
          f"offset={rev.offset:+.3f}s")
    print(f"  {rev2.id}: conf={rev2.confidence:.4f} status={rev2.status} "
          f"offset={rev2.offset:+.3f}s")
    print(f"  revision count: {len(proj.revisions)}  (history is kept, not overwritten)")
    same = (abs(rev.offset - rev2.offset) < 1e-9
            and abs(rev.confidence - rev2.confidence) < 1e-9)
    print(f"  identical numbers on re-run: {same}"
          f"{'  ← deterministic' if same else '  ← NOT deterministic'}")
    if not same:
        print("  NOTE: non-determinism here means receipts cannot be compared across runs")

    chosen = rev if rev.status != "rejected" else rev2
    proj.select(chosen.id)
    print(f"  selected: {chosen.id}")

    # ---------------------------------------------------------------- export
    rule("6. EXPORT — re-measures against the video on disk, then writes")
    info = pipeline.export_selected(proj)
    print(f"  wrote    : {info['path']}")
    print(f"  cues     : {info['cue_count']}  bytes={info['bytes']}")
    print(f"  sha256   : {info['sha256'][:32]}…")
    lc = info["live_check"]
    print(f"  re-check : confidence={lc.get('confidence')} "
          f"residual={lc.get('residual_offset')}s "
          f"speech_ratio={lc.get('speech_ratio')}")
    print("  → export refuses if that re-check disagrees with the receipt")

    # ---------------------------------------------------- integrity + report
    rule("7. INTEGRITY — change an input and watch dependents go stale")
    integrity = proj.verify_inputs()
    print(f"  before edit: stale={integrity['stale'] or 'none'}")
    p = proj.subtitle_path
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    integrity2 = proj.verify_inputs()
    print(f"  after edit : stale={integrity2['stale']}")
    try:
        pipeline.export_selected(proj, work / "should-not-exist.srt")
        print("  UNEXPECTED: export succeeded on a stale revision")
        return 1
    except BenchError as exc:
        print(f"  export refused: [{exc.code}] {exc.message}")
    print("  → this is the coupling that matters: state, not features")

    # ---------------------------------------------------------------- report
    rule("8. REPORT — everything above, in one self-contained file")
    rp = report.build_report(proj, work / "report.html")
    print(f"  report → {rp}  ({rp.stat().st_size} bytes)")

    print(f"\nartifacts in {work}")
    if not args.keep and not args.workdir:
        shutil.rmtree(work, ignore_errors=True)
        print("(removed; pass --keep to inspect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
