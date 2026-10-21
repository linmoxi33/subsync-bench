"""Engine adapter: staging, verification, and the destructive-framerate path.

The cue-loss tests here are the most important in the file. Upstream silently
drops cues when a framerate guess is wrong; that behaviour must be detected and
survived, because losing dialogue is never an acceptable outcome.
"""

from __future__ import annotations

import pytest

from subsync_bench import engine, subio
from subsync_bench.errors import E_ALIGN_FAILED, E_ENGINE_MISSING, AlignError


def _srt(path, cues):
    from subsync_bench.subio import Cue

    objs = [Cue(i + 1, a, b, t) for i, (a, b, t) in enumerate(cues)]
    subio.write_srt(objs, path)
    return path


def test_engine_available_reports_bool():
    assert isinstance(engine.engine_available(), bool)


def test_engine_version_is_string():
    v = engine.engine_version()
    assert isinstance(v, str) and v


def test_framerate_bounds_are_the_documented_values():
    """Real conversions are 1.001 and 1.0427; the window must contain them."""
    assert engine.FRAMERATE_MIN == 0.90
    assert engine.FRAMERATE_MAX == 1.10
    assert engine.FRAMERATE_MIN < 1.0 < engine.FRAMERATE_MAX
    assert engine.FRAMERATE_MIN < 1.0427 < engine.FRAMERATE_MAX
    assert engine.FRAMERATE_MIN < 1 / 1.0427 < engine.FRAMERATE_MAX


# --- verify_output --------------------------------------------------------


def test_verify_output_accepts_matching_cue_count(tmp_path):
    p = _srt(tmp_path / "ok.srt", [(1.0, 2.0, "a"), (3.0, 4.0, "b")])
    cues, problems = engine.verify_output(p, 2)
    assert len(cues) == 2
    assert problems == []


def test_verify_output_reports_cue_loss(tmp_path):
    """The load-bearing assertion: a 4-cue output for a 14-cue input must be
    reported, not silently accepted."""
    p = _srt(tmp_path / "short.srt", [(1.0, 2.0, "a"), (3.0, 4.0, "b")])
    cues, problems = engine.verify_output(p, 14)
    assert len(cues) == 2
    assert any("cue count changed" in x for x in problems)
    assert any("14" in x for x in problems)


def test_verify_output_reports_structural_problems(tmp_path):
    p = _srt(tmp_path / "bad.srt", [(-5.0, -1.0, "negative"), (3.0, 4.0, "")])
    _, problems = engine.verify_output(p, 2)
    assert problems, "negative start / empty text must be reported"


def test_verify_output_missing_file_raises(tmp_path):
    with pytest.raises(AlignError) as exc:
        engine.verify_output(tmp_path / "nope.srt", 1)
    assert exc.value.code == E_ALIGN_FAILED


def test_verify_output_empty_file_raises(tmp_path):
    p = tmp_path / "empty.srt"
    p.write_bytes(b"")
    with pytest.raises(AlignError) as exc:
        engine.verify_output(p, 1)
    assert exc.value.code == E_ALIGN_FAILED


def test_verify_output_garbage_raises(tmp_path):
    p = tmp_path / "garbage.srt"
    p.write_text("this is not a subtitle at all", encoding="utf-8")
    with pytest.raises(AlignError) as exc:
        engine.verify_output(p, 1)
    assert exc.value.code == E_ALIGN_FAILED


# --- apply_shift ----------------------------------------------------------


def test_apply_shift_moves_and_marks_touched():
    from subsync_bench.subio import Cue

    out = engine.apply_shift([Cue(1, 10.0, 12.0, "x")], -8.0, 1.0)
    assert out[0].start == pytest.approx(2.0)
    assert out[0].touched


def test_apply_shift_clamps_negative_to_zero():
    from subsync_bench.subio import Cue

    out = engine.apply_shift([Cue(1, 1.0, 2.0, "x")], -10.0, 1.0)
    assert out[0].start == 0.0
    assert out[0].end >= out[0].start


def test_apply_shift_scales_before_shifting():
    from subsync_bench.subio import Cue

    out = engine.apply_shift([Cue(1, 100.0, 102.0, "x")], 0.0, 25 / 24)
    assert out[0].start == pytest.approx(100.0 * 25 / 24)


# --- real engine (slow) ---------------------------------------------------


@pytest.mark.skipif(not engine.engine_available(), reason="engine not installed")
def test_run_engine_produces_staged_output(tmp_path, media_dir):
    out = tmp_path / "staged.srt"
    res = engine.run_engine(
        media_dir / "film.mp4", media_dir / "late.srt", out,
        max_offset_seconds=30.0,
    )
    assert res.output_path == out and out.exists()
    assert res.offset == pytest.approx(-8.4, abs=0.5)
    assert res.mode == "engine"
    assert res.engine_version != "unknown"


def test_run_engine_without_engine_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "engine_available", lambda: False)
    with pytest.raises(AlignError) as exc:
        engine.run_engine(
            tmp_path / "v.mp4", tmp_path / "s.srt", tmp_path / "o.srt"
        )
    assert exc.value.code == E_ENGINE_MISSING
    assert "pip install" in (exc.value.hint or "")


@pytest.mark.skipif(not engine.engine_available(), reason="engine not installed")
def test_run_engine_honours_no_fix_framerate(tmp_path, media_dir):
    out = tmp_path / "nofps.srt"
    res = engine.run_engine(
        media_dir / "film.mp4", media_dir / "late.srt", out,
        max_offset_seconds=30.0, no_fix_framerate=True,
    )
    assert res.output_path is not None
    cues, problems = engine.verify_output(out, 14)
    assert len(cues) == 14, "no-fix-framerate must not lose cues"
    assert problems == []
