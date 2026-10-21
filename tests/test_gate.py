"""The trust gate.

Each rule exists because a specific upstream failure was measured. These tests
pin the rule boundaries so a future "just relax the threshold" change cannot
quietly disable the gate.
"""

from __future__ import annotations

from subsync_bench import gate


def _ok(**over):
    """Baseline: a measurement that should pass."""
    kw = {
        "confidence": 0.70,
        "null_confidence": 0.70,
        "offset": -8.22,
        "framerate_scale": 1.0,
        "speech_ratio": 0.32,
        "cue_count_in": 14,
        "cue_count_out": 14,
        "residual_summary": {"suspect_count": 0, "suspect_ratio": 0.0},
        "residual_offset": -0.31,
        "max_offset_seconds": 120.0,
        "confidence_threshold": 0.55,
        "residual_tolerance": 0.8,
    }
    kw.update(over)
    return gate.evaluate(**kw)


def test_good_alignment_passes():
    v = _ok()
    assert v.ok
    assert not v.blocks


def test_r1_silent_reference_blocks():
    v = _ok(speech_ratio=0.0)
    assert not v.ok
    assert "R1_NO_SPEECH" in {f.code for f in v.blocks}


def test_r2_low_confidence_blocks():
    v = _ok(confidence=0.40)
    assert not v.ok
    assert "R2_LOW_CONFIDENCE" in {f.code for f in v.blocks}


def test_r2_boundary_just_below_threshold_blocks():
    v = _ok(confidence=0.549)
    assert not v.ok


def test_r2_boundary_just_above_threshold_passes():
    v = _ok(confidence=0.551)
    assert v.ok


def test_r3_unapplied_shift_blocks():
    v = _ok(residual_offset=-8.0)
    assert not v.ok
    assert "R3_RESIDUAL_OFFSET" in {f.code for f in v.blocks}


def test_r3_measured_noise_floor_passes():
    """-0.36s was the worst residual across verified-correct outputs."""
    v = _ok(residual_offset=-0.36)
    assert v.ok


def test_r3_tolerance_boundary():
    assert _ok(residual_offset=0.79).ok
    assert not _ok(residual_offset=0.81).ok


def test_r3_symmetric():
    assert not _ok(residual_offset=1.5).ok
    assert not _ok(residual_offset=-1.5).ok


def test_r4_offset_beyond_window_blocks():
    v = _ok(offset=200.0, max_offset_seconds=120.0)
    assert not v.ok
    assert "R4_OFFSET_TOO_LARGE" in {f.code for f in v.blocks}


def test_r4_negative_offset_beyond_window_blocks():
    v = _ok(offset=-200.0, max_offset_seconds=120.0)
    assert not v.ok


def test_r5_implausible_framerate_blocks():
    v = _ok(framerate_scale=0.85)
    assert not v.ok
    assert "R5_FRAMERATE_IMPLAUSIBLE" in {f.code for f in v.blocks}


def test_r5_real_conversion_passes():
    # 23.976 -> 25 fps is 1.0427; must not be flagged
    assert _ok(framerate_scale=1.0427).ok
    assert _ok(framerate_scale=1 / 1.0427).ok


def test_r6_cue_loss_blocks():
    v = _ok(cue_count_out=4)
    assert not v.ok
    assert "R6_CUE_LOSS" in {f.code for f in v.blocks}


def test_r7_high_suspect_ratio_warns_but_does_not_block():
    v = _ok(residual_summary={"suspect_count": 6, "suspect_ratio": 0.43})
    assert v.ok
    assert "R7_SCENE_SPLICE_SUSPECTED" in {f.code for f in v.warns}


def test_r7_low_suspect_ratio_is_quiet():
    v = _ok(residual_summary={"suspect_count": 1, "suspect_ratio": 0.07})
    assert v.ok
    assert "R7_SCENE_SPLICE_SUSPECTED" not in {f.code for f in v.warns}


def test_r8_already_aligned_is_informational():
    v = _ok(offset=0.01)
    assert v.ok
    infos = [f for f in v.findings if f.severity == "info"]
    assert any(f.code == "R8_ALREADY_ALIGNED" for f in infos)


def test_multiple_blocks_all_reported():
    v = _ok(confidence=0.2, cue_count_out=3, speech_ratio=0.0)
    codes = {f.code for f in v.blocks}
    assert {"R1_NO_SPEECH", "R2_LOW_CONFIDENCE", "R6_CUE_LOSS"} <= codes


def test_verdict_serialises():
    d = _ok().to_dict()
    assert d["ok"] is True
    assert isinstance(d["findings"], list)
    assert all({"code", "severity", "message"} <= set(f) for f in d["findings"])


def test_verdict_explain_mentions_blocked():
    v = _ok(confidence=0.1)
    assert "BLOCKED" in v.explain()


def test_verdict_explain_pass():
    assert "pass" in _ok().explain()


def test_defaults_match_documented_values():
    """These constants are load-bearing; changing them must be deliberate."""
    assert gate.DEFAULT_CONFIDENCE_THRESHOLD == 0.55
    assert gate.DEFAULT_RESIDUAL_TOLERANCE == 0.8
    assert gate.SUSPECT_RATIO_WARN == 0.15
