"""Pipeline behaviour that cannot be reached through the CLI without a broken engine.

The cue-loss retry lives here because it needs the engine to *fail in a specific
way*: write a valid but truncated SRT and report success. A mutation test showed
this branch had no coverage at all — deleting the retry entirely left the suite
green, which is exactly the kind of blind spot mutation testing exists to find.

These tests substitute ``engine.run_engine`` with a scripted fake, so the pipeline
logic is exercised without depending on upstream misbehaving on demand.
"""

from __future__ import annotations

import pytest

from subsync_bench import audio, engine, pipeline, subio
from subsync_bench.errors import E_VIDEO_SILENT, AlignError
from subsync_bench.subio import Cue, write_srt
from subsync_bench.workspace import Project


def _write(path, cues):
    write_srt([Cue(i + 1, a, b, t) for i, (a, b, t) in enumerate(cues)], path)
    return path


class FakeEngine:
    """Records calls and returns scripted outputs.

    Each entry in ``script`` is either an int (number of cues to keep from the
    input, shifted by ``offset``) or None (produce no file at all).
    """

    def __init__(self, script, offset=-8.2):
        self.script = list(script)
        self.offset = offset
        self.calls = []

    def __call__(self, reference, subtitle, out_path, **kwargs):
        self.calls.append(
            {
                "reference": reference,
                "subtitle": subtitle,
                "out": out_path,
                "no_fix_framerate": kwargs.get("no_fix_framerate", False),
            }
        )
        step = self.script.pop(0) if self.script else len(subio.load(subtitle).cues)
        if step is None:
            return engine.EngineResult(
                offset=self.offset, framerate_scale=1.0, upstream_score=1.0,
                upstream_retval=0, upstream_success_flag=True,
                output_path=None, mode="engine",
            )

        cues = subio.load(subtitle).cues
        kept = [
            Cue(c.index, max(0.0, c.start + self.offset),
                max(0.0, c.end + self.offset), c.text)
            for c in cues[:step]
        ]
        write_srt(kept, out_path)
        return engine.EngineResult(
            offset=self.offset, framerate_scale=1.0, upstream_score=1.0,
            upstream_retval=0, upstream_success_flag=True,
            output_path=out_path, mode="engine",
        )


@pytest.fixture
def small_project(tmp_path, media_dir):
    """A project whose subtitle has 8 cues, so truncation is easy to script."""
    if not (media_dir / "film.mp4").exists():
        pytest.skip("media fixtures missing — run fixtures/build_offline.py")
    sub = _write(
        tmp_path / "in.srt",
        [(2.0 + i * 6, 5.0 + i * 6, f"line {i}") for i in range(8)],
    )
    return Project.create(tmp_path / "p", video=media_dir / "film.mp4", subtitle=sub)


def test_cue_loss_triggers_one_retry_without_framerate_fixing(small_project, monkeypatch):
    fake = FakeEngine(script=[4, 8])   # first call drops half, retry keeps all
    monkeypatch.setattr(engine, "run_engine", fake)
    monkeypatch.setattr(pipeline.engine, "run_engine", fake)

    rev, _ = pipeline.align_project(small_project)

    assert len(fake.calls) == 2, "a cue-losing run must be retried exactly once"
    assert fake.calls[0]["no_fix_framerate"] is False
    assert fake.calls[1]["no_fix_framerate"] is True, "retry must disable framerate fixing"
    assert rev.cue_count == 8, "the retry that preserved every cue must be kept"
    assert any("retried" in w for w in rev.notes)


def test_cue_loss_warning_names_the_surviving_count(small_project, monkeypatch):
    fake = FakeEngine(script=[4, 8])
    monkeypatch.setattr(pipeline.engine, "run_engine", fake)
    rev, _ = pipeline.align_project(small_project)
    joined = " ".join(rev.notes)
    assert "4/8" in joined, f"warning should quantify the loss, got: {rev.notes}"


def test_double_cue_loss_keeps_original_and_blocks(small_project, monkeypatch):
    """If the retry also loses cues, the revision must be rejected (R6)."""
    fake = FakeEngine(script=[4, 5])
    monkeypatch.setattr(pipeline.engine, "run_engine", fake)
    rev, verdict = pipeline.align_project(small_project)
    assert len(fake.calls) == 2
    assert rev.status == "rejected"
    codes = {f["code"] for f in verdict["blocking"]}
    assert "R6_CUE_LOSS" in codes
    assert any("also lost cues" in w for w in rev.notes)


def test_retry_without_output_is_reported(small_project, monkeypatch):
    fake = FakeEngine(script=[4, None])
    monkeypatch.setattr(pipeline.engine, "run_engine", fake)
    rev, _ = pipeline.align_project(small_project)
    assert any("produced no output" in w for w in rev.notes)
    assert rev.status == "rejected"


def test_no_retry_when_cue_count_is_preserved(small_project, monkeypatch):
    fake = FakeEngine(script=[8])
    monkeypatch.setattr(pipeline.engine, "run_engine", fake)
    rev, _ = pipeline.align_project(small_project)
    assert len(fake.calls) == 1, "a clean run must not be retried"
    assert rev.cue_count == 8


def test_no_retry_when_framerate_was_pinned_explicitly(small_project, monkeypatch):
    """If the caller pinned the framerate, retrying with it disabled would ignore
    an explicit instruction. Loss must then be reported, not silently re-tried."""
    fake = FakeEngine(script=[4])
    monkeypatch.setattr(pipeline.engine, "run_engine", fake)
    rev, verdict = pipeline.align_project(small_project, framerate_ratio=1.0427)
    assert len(fake.calls) == 1
    assert rev.status == "rejected"
    assert "R6_CUE_LOSS" in {f["code"] for f in verdict["blocking"]}


def test_silent_reference_refused_before_engine_runs(small_project, monkeypatch):
    """The engine must never be invoked when the reference has no speech."""
    called = []

    def explode(*a, **k):
        called.append(1)
        raise AssertionError("engine must not run on a silent reference")

    monkeypatch.setattr(pipeline.engine, "run_engine", explode)

    # swap in a silent video, keeping the project valid on disk
    silent = small_project.root / "input" / "silent.mp4"
    silent.write_bytes((small_project.video_path).read_bytes())
    # force the envelope to read as silent without needing real silence
    monkeypatch.setattr(
        pipeline.audio, "envelope_from_media",
        lambda *a, **k: audio.Envelope(
            __import__("numpy").zeros(1000, dtype="float32")
        ),
    )

    with pytest.raises(AlignError) as exc:
        pipeline.align_project(small_project)
    assert exc.value.code == E_VIDEO_SILENT
    assert called == [], "engine must not be reached"


def test_revision_records_input_digests(small_project, monkeypatch):
    fake = FakeEngine(script=[8])
    monkeypatch.setattr(pipeline.engine, "run_engine", fake)
    rev, _ = pipeline.align_project(small_project)
    assert rev.inputs["video"] == small_project.manifest["video"]["sha256"]
    assert rev.inputs["subtitle"] == small_project.manifest["subtitle"]["sha256"]


def test_revision_artifact_is_written(small_project, monkeypatch):
    fake = FakeEngine(script=[8])
    monkeypatch.setattr(pipeline.engine, "run_engine", fake)
    rev, _ = pipeline.align_project(small_project)
    art = small_project.revision_artifact(rev)
    assert art.exists()
    assert len(subio.load(art).cues) == 8


def test_failed_engine_raises_with_log_tail(small_project, monkeypatch):
    def boom(*a, **k):
        raise AlignError("engine exploded", code="E_ALIGN_FAILED")

    monkeypatch.setattr(pipeline.engine, "run_engine", boom)
    with pytest.raises(AlignError):
        pipeline.align_project(small_project)
