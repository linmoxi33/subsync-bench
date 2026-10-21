"""End-to-end: the whole workbench, through the CLI.

This is where "the modules actually cooperate" is proven. Each test drives the
real command line over real media and checks the artefacts on disk, not internal
state.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from subsync_bench import subio

CLI = [sys.executable, "-m", "subsync_bench.cli"]


def run(*args, expect: int | None = 0, cwd: Path | None = None):
    proc = subprocess.run(
        CLI + [str(a) for a in args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd) if cwd else None,
    )
    if expect is not None:
        assert proc.returncode == expect, (
            f"exit {proc.returncode} (wanted {expect})\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    return proc


def json_run(*args, expect: int | None = 0):
    proc = run(*args, "--json", expect=expect)
    # the JSON document is the last object printed
    text = proc.stdout.strip()
    start = text.find("{")
    assert start >= 0, f"no JSON in output:\n{text[-2000:]}"
    return json.loads(text[start:])


# --- doctor ---------------------------------------------------------------


def test_doctor_reports_capabilities():
    data = json_run("doctor")
    assert data["ok"] is True
    assert set(data["capabilities"]) == {"inspect", "align", "preview", "export"}
    names = {c["check"] for c in data["checks"]}
    assert {"ffmpeg", "numpy", "engine:ffsubsync"} <= names


def test_doctor_marks_ffprobe_optional():
    data = json_run("doctor")
    ffprobe = next(c for c in data["checks"] if c["check"] == "ffprobe")
    assert ffprobe["required"] is False


# --- inspect --------------------------------------------------------------


def test_inspect_reports_encoding_and_cues(media_dir):
    # --no-audio is the point of this flag: subtitle introspection must work even
    # when there is no media at all (or no ffmpeg), e.g. on a metadata-only box.
    data = json_run(
        "inspect", media_dir / "late.srt", media_dir / "late.srt", "--no-audio"
    )
    sub = data["inspection"]["subtitle"]
    assert sub["cue_count"] == 14
    assert sub["encoding"]
    assert data["inspection"]["envelope"] is None


def test_inspect_detects_subtitle_longer_than_video(media_dir):
    data = json_run("inspect", media_dir / "film.mp4", media_dir / "late.srt")
    env = data["inspection"]["envelope"]
    sub = data["inspection"]["subtitle"]
    assert env["is_silent"] is False
    assert sub["span"][1] > env["duration"], "fixture is intentionally over-long"
    assert any("different cut" in w or "framerate" in w
               for w in data["inspection"]["warnings"])


def test_inspect_missing_subtitle_exits_state(tmp_path, media_dir):
    proc = run("inspect", media_dir / "film.mp4", tmp_path / "nope.srt", expect=None)
    assert proc.returncode != 0


# --- align ----------------------------------------------------------------


def test_align_creates_revision_and_passes_gate(tmp_path, media_dir):
    data = json_run(
        "align", tmp_path / "p",
        "--video", media_dir / "film.mp4",
        "--subtitle", media_dir / "late.srt",
        "--label", "e2e",
    )
    assert data["ok"] is True
    rev = data["revision"]
    assert rev["status"] in ("ok", "suspect")
    assert rev["offset"] == pytest.approx(-8.4, abs=0.4)
    assert rev["cue_count"] == 14
    assert rev["confidence"] > 0.55
    assert (tmp_path / "p" / "project.json").exists()
    assert (tmp_path / "p" / rev["artifact"]).exists()


def test_align_refuses_silent_reference(tmp_path, media_dir, silent_film):
    proc = run(
        "align", tmp_path / "p",
        "--video", silent_film,
        "--subtitle", media_dir / "late.srt",
        expect=None,
    )
    # A silent reference is a refused answer, not a broken environment: the tool
    # ran, measured, and declined. Exit 4 (gate), not 3 (env).
    assert proc.returncode == 4
    assert "E_VIDEO_SILENT" in proc.stdout + proc.stderr


def test_align_blocks_wrong_content(tmp_path, media_dir, srt):
    """A subtitle describing a different film must be rejected, not silently
    shifted into nonsense."""
    windows = [(1.0, 1.5), (7.0, 7.4), (20.0, 20.5), (31.0, 31.5),
               (44.0, 44.4), (52.0, 52.5), (61.0, 61.4), (70.0, 70.5),
               (79.0, 79.4), (88.0, 88.5), (95.0, 95.4), (102.0, 102.5),
               (106.0, 106.4), (108.0, 108.5)]
    wrong = srt([(a, b, f"other film {i}") for i, (a, b) in enumerate(windows)],
                "wrong.srt")
    data = json_run(
        "align", tmp_path / "p",
        "--video", media_dir / "film.mp4",
        "--subtitle", wrong,
        expect=4,
    )
    assert data["ok"] is False
    assert data["revision"]["status"] == "rejected"
    codes = {f["code"] for f in data["verdict"]["blocking"]}
    assert "R2_LOW_CONFIDENCE" in codes


def test_align_is_repeatable_with_new_revision_ids(tmp_path, media_dir):
    proj = tmp_path / "p"
    a = json_run("align", proj, "--video", media_dir / "film.mp4",
                 "--subtitle", media_dir / "late.srt")
    b = json_run("align", proj)
    assert a["revision"]["id"] == "r001"
    assert b["revision"]["id"] == "r002"


def test_align_refuses_input_swap_without_flag(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    proc = run("align", proj, "--video", media_dir / "film.mp4",
               "--subtitle", media_dir / "early.srt", expect=None)
    assert proc.returncode == 5
    assert "E_PROJECT_EXISTS" in proc.stdout + proc.stderr


def test_align_accepts_input_swap_with_flag(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    data = json_run("align", proj, "--video", media_dir / "film.mp4",
                    "--subtitle", media_dir / "early.srt", "--replace-inputs")
    assert data["revision"]["offset"] == pytest.approx(3.7, abs=0.5)


# --- revs / select --------------------------------------------------------


def test_revs_lists_revisions(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    data = json_run("revs", proj)
    assert len(data["revisions"]) == 1
    assert data["selected"] is None


def test_select_sets_selection(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    data = json_run("select", proj, "r001")
    assert data["selected"] == "r001"
    assert json_run("revs", proj)["selected"] == "r001"


def test_select_unknown_exits_state(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    run("select", proj, "r999", expect=5)


# --- export ---------------------------------------------------------------


def test_export_requires_selection(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    proc = run("export", proj, expect=None)
    assert proc.returncode == 5
    assert "E_NO_DECISION" in proc.stdout + proc.stderr


def test_export_writes_aligned_file(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    json_run("select", proj, "r001")
    data = json_run("export", proj)
    out = Path(data["export"]["path"])
    assert out.exists()
    assert data["export"]["cue_count"] == 14
    assert data["export"]["live_check"]["reverified"] is True


def test_exported_timing_is_actually_correct(tmp_path, media_dir):
    """The point of the whole tool: exported cues land on real speech."""
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    json_run("select", proj, "r001")
    data = json_run("export", proj)

    truth = subio.load(media_dir / "truth.srt").cues
    out = subio.load(Path(data["export"]["path"])).cues
    assert len(out) == len(truth)
    deltas = [o.start - t.start for t, o in zip(truth, out, strict=True)]
    assert max(abs(d) for d in deltas) < 0.5, f"offsets too large: {deltas}"
    assert all(t.text == o.text for t, o in zip(truth, out, strict=True)), (
        "text must be preserved"
    )


def test_export_refuses_rejected_revision(tmp_path, media_dir, srt):
    windows = [(1.0, 1.5), (7.0, 7.4), (20.0, 20.5), (31.0, 31.5),
               (44.0, 44.4), (52.0, 52.5), (61.0, 61.4), (70.0, 70.5),
               (79.0, 79.4), (88.0, 88.5), (95.0, 95.4), (102.0, 102.5),
               (106.0, 106.4), (108.0, 108.5)]
    wrong = srt([(a, b, f"x{i}") for i, (a, b) in enumerate(windows)], "w.srt")
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", wrong, expect=4)
    json_run("select", proj, "r001")
    proc = run("export", proj, expect=None)
    assert proc.returncode == 4
    assert "E_EXPORT_BLOCKED" in proc.stdout + proc.stderr


def test_export_override_allows_rejected(tmp_path, media_dir, srt):
    windows = [(1.0, 1.5), (7.0, 7.4), (20.0, 20.5), (31.0, 31.5),
               (44.0, 44.4), (52.0, 52.5), (61.0, 61.4), (70.0, 70.5),
               (79.0, 79.4), (88.0, 88.5), (95.0, 95.4), (102.0, 102.5),
               (106.0, 106.4), (108.0, 108.5)]
    wrong = srt([(a, b, f"x{i}") for i, (a, b) in enumerate(windows)], "w.srt")
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", wrong, expect=4)
    json_run("select", proj, "r001")
    data = json_run("export", proj, "--allow-override", "--skip-recheck")
    assert Path(data["export"]["path"]).exists()


def test_export_refuses_existing_destination(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    json_run("select", proj, "r001")
    json_run("export", proj)
    proc = run("export", proj, expect=None)
    assert proc.returncode == 5
    assert "E_EXPORT_EXISTS" in proc.stdout + proc.stderr


def test_export_force_overwrites(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    json_run("select", proj, "r001")
    json_run("export", proj)
    data = json_run("export", proj, "--force")
    assert Path(data["export"]["path"]).exists()


def test_export_rechecks_and_blocks_when_video_swapped(tmp_path, media_dir):
    """Export-time re-measurement must catch a video swapped after alignment."""
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    json_run("select", proj, "r001")
    # swap the project's video for something unrelated without touching the manifest
    (proj / "input" / "film.mp4").write_bytes(
        (media_dir / "silent.mp4").read_bytes()
    )
    proc = run("export", proj, expect=None)
    assert proc.returncode in (4, 5)
    # Two independent guards may fire first, and both are correct answers:
    #   E_RECEIPT_MISSING — the input digest no longer matches (caught before work)
    #   E_EXPORT_BLOCKED  — the live re-measurement disagrees with the receipt
    combined = proc.stdout + proc.stderr
    assert "E_RECEIPT_MISSING" in combined or "E_EXPORT_BLOCKED" in combined


def test_export_recheck_blocks_even_with_force(tmp_path, media_dir):
    """--force bypasses the staleness shortcut; the live re-measurement must still
    refuse. Otherwise a swapped video could be exported just by adding --force."""
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    json_run("select", proj, "r001")
    (proj / "input" / "film.mp4").write_bytes(
        (media_dir / "silent.mp4").read_bytes()
    )
    proc = run("export", proj, "--force", expect=None)
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "E_EXPORT_BLOCKED" in combined or "E_VIDEO_SILENT" in combined


def test_force_does_not_bypass_recheck_only_skip_does(tmp_path, media_dir):
    """--force may overwrite files and accept stale receipts, but the physics
    re-check is a separate switch. This test exists because an earlier version
    let --force disable the re-check, which meant a swapped video could be
    exported by adding a flag that users add habitually."""
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    json_run("select", proj, "r001")
    (proj / "input" / "film.mp4").write_bytes((media_dir / "silent.mp4").read_bytes())
    blocked = run("export", proj, "--force", expect=None)
    assert blocked.returncode != 0
    skipped = run("export", proj, "--force", "--skip-recheck", expect=None)
    assert skipped.returncode == 0, "explicit skip-recheck is the documented way out"


# --- preview / report -----------------------------------------------------


def test_preview_writes_self_contained_html(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    data = json_run("preview", proj)
    html = Path(data["html"]).read_text(encoding="utf-8")
    assert "<svg" in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert data["cue_rows"] == 14


def test_preview_window_option(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    data = json_run("preview", proj, "--start", 0, "--end", 20)
    assert Path(data["html"]).exists()


def test_report_contains_timeline_and_diff(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt")
    json_run("select", proj, "r001")
    data = json_run("report", proj)
    html = Path(data["report"]).read_text(encoding="utf-8")
    assert "<svg" in html
    assert "per-cue diff" in html
    assert "trust gate" in html
    assert "14" in html


def test_report_with_no_revisions_still_builds(tmp_path, media_dir):
    proj = tmp_path / "p"
    json_run("align", proj, "--video", media_dir / "film.mp4",
             "--subtitle", media_dir / "late.srt", expect=0)
    # remove all revisions from the manifest but keep inputs
    import json as _json

    mp = proj / "project.json"
    m = _json.loads(mp.read_text(encoding="utf-8"))
    m["revisions"] = []
    m["selected"] = None
    mp.write_text(_json.dumps(m), encoding="utf-8")
    data = json_run("report", proj)
    assert Path(data["report"]).exists()


# --- error handling -------------------------------------------------------


def test_unknown_project_exits_state(tmp_path):
    proc = run("revs", tmp_path / "nope", expect=None)
    assert proc.returncode == 5
    assert "E_PROJECT_MISSING" in proc.stdout + proc.stderr


def test_bad_subcommand_exits_two():
    proc = subprocess.run(
        [*CLI, "not-a-command"], capture_output=True, text=True
    )
    assert proc.returncode == 2


def test_help_works():
    proc = subprocess.run([*CLI, "--help"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0
    for sub in ("doctor", "inspect", "align", "preview", "revs", "select",
                "export", "report"):
        assert sub in proc.stdout
