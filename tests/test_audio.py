"""Measurement: envelope extraction, normalized confidence, cue residuals.

The thresholds asserted here are the ones the gate depends on, so these tests are
the guard against silently re-tuning the tool into uselessness.
"""

from __future__ import annotations

import numpy as np
import pytest

from subsync_bench import audio, subio

# --- envelope -------------------------------------------------------------


def test_envelope_of_silence_is_flagged_silent():
    pcm = np.zeros(audio.SAMPLE_RATE * 3, dtype=np.float32)
    env = audio.envelope_from_pcm(pcm)
    assert env.is_silent
    assert env.energy == 0.0
    assert env.speech_ratio == 0.0


def test_envelope_threshold_tracks_noise_floor():
    """A loud transient must not drag the noise floor up into "speech".

    This test has to give the threshold something to do. A track of pure constant
    noise is rejected earlier, by the "no dynamic range" guard, so it says nothing
    about the threshold itself — an earlier version of this test used exactly that
    and a mutation setting ``thr = 0.0`` survived.

    Fixture: 6 s of broadband hiss at 0.02, plus a 0.5 s burst at 0.3. The burst
    occupies 8% of frames, enough to pull the 95th percentile up to ~0.3, so
    floor=0.02 and loud=0.3 differ and the threshold lands near 0.09 — above the
    hiss. With ``thr = 0`` every hiss frame reads as 0.02/0.3 = 0.067 > 0.05 and
    the whole track is "speech".
    """
    sr = audio.SAMPLE_RATE
    rng = np.random.default_rng(11)
    n = sr * 6
    pcm = (rng.standard_normal(n) * 0.02).astype(np.float32)
    i = int(2.0 * sr)
    j = i + int(0.5 * sr)
    pcm[i:j] = (rng.standard_normal(j - i) * 0.3).astype(np.float32)

    env = audio.envelope_from_pcm(pcm)
    assert not env.is_silent
    # only the burst should be active: 0.5 s of 6 s ~ 8%
    assert env.speech_ratio < 0.30, (
        f"noise floor was counted as speech (ratio={env.speech_ratio:.3f}); "
        f"the threshold is not tracking the noise floor"
    )
    # and the burst itself must still be detected
    assert env.speech_ratio > 0.03, "the transient should still register"


def test_envelope_threshold_scales_with_level():
    """Same speech pattern at two very different levels must both be detected."""
    sr = audio.SAMPLE_RATE
    n = sr * 8
    pcm = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(5)
    for start in (1.0, 4.0, 6.5):
        i = int(start * sr)
        j = i + int(0.9 * sr)
        pcm[i:j] = (rng.standard_normal(j - i) * 0.25).astype(np.float32)

    loud = audio.envelope_from_pcm(pcm)
    quiet = audio.envelope_from_pcm(pcm * 0.02)   # 34 dB down
    assert not loud.is_silent
    assert not quiet.is_silent, "a quiet but structured track must still be detected"
    assert abs(loud.speech_ratio - quiet.speech_ratio) < 0.15


def test_envelope_of_constant_tone_is_silent():
    # A pure tone has no dynamic range; treating it as "speech everywhere" would
    # make any cue look aligned.
    t = np.arange(audio.SAMPLE_RATE * 3) / audio.SAMPLE_RATE
    pcm = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    env = audio.envelope_from_pcm(pcm)
    assert env.is_silent


def test_envelope_of_bursts_has_expected_ratio():
    pcm = np.zeros(audio.SAMPLE_RATE * 10, dtype=np.float32)
    rng = np.random.default_rng(0)
    for start in (1.0, 4.0, 7.0):
        i = int(start * audio.SAMPLE_RATE)
        j = i + audio.SAMPLE_RATE
        pcm[i:j] = (rng.standard_normal(j - i) * 0.3).astype(np.float32)
    env = audio.envelope_from_pcm(pcm)
    assert not env.is_silent
    assert 0.1 < env.speech_ratio < 0.6


def test_envelope_duration_matches_audio():
    pcm = np.zeros(audio.SAMPLE_RATE * 5, dtype=np.float32)
    pcm[audio.SAMPLE_RATE : audio.SAMPLE_RATE * 2] = 0.5
    env = audio.envelope_from_pcm(pcm)
    assert env.duration == pytest.approx(5.0, abs=0.05)


def test_envelope_delay_moves_signal_later():
    pcm = np.zeros(audio.SAMPLE_RATE * 4, dtype=np.float32)
    pcm[audio.SAMPLE_RATE : audio.SAMPLE_RATE * 2] = 0.8
    env = audio.envelope_from_pcm(pcm)
    delayed = env.delay(0.5)
    assert len(delayed.values) == len(env.values)
    assert delayed.values[0] == 0.0
    # the burst should now start ~0.5s later
    first = int(np.argmax(env.values > 0.1))
    first_d = int(np.argmax(delayed.values > 0.1))
    assert (first_d - first) * env.hop / env.sample_rate == pytest.approx(0.5, abs=0.05)


def test_envelope_delay_zero_is_identity():
    pcm = np.zeros(1000, dtype=np.float32)
    pcm[100:200] = 0.5
    env = audio.envelope_from_pcm(pcm)
    assert env.delay(0.0) is env


def test_envelope_empty_pcm():
    env = audio.envelope_from_pcm(np.zeros(0, dtype=np.float32))
    assert len(env.values) == 0
    assert env.is_silent


# --- correlation ----------------------------------------------------------


def test_align_finds_known_offset_on_synthetic_envelope():
    """Build a synthetic envelope with speech at known times, cue it +5s late."""
    sr, hop = audio.SAMPLE_RATE, audio.HOP
    n = int(60 * sr / hop)
    values = np.zeros(n, dtype=np.float32)
    windows = [(2.0, 5.0), (9.0, 12.0), (16.0, 19.0), (24.0, 27.0), (33.0, 36.0)]
    for a, b in windows:
        values[int(a * sr / hop) : int(b * sr / hop)] = 1.0
    env = audio.Envelope(values, sample_rate=sr, hop=hop)

    from subsync_bench.subio import Cue

    cues = [Cue(i + 1, a + 5.0, b + 5.0, "x") for i, (a, b) in enumerate(windows)]
    result = audio.align_envelope(env, cues, max_offset_seconds=15.0)
    assert result.offset == pytest.approx(-5.0, abs=0.15)
    assert result.confidence > 0.7


def test_align_refuses_silent_reference():
    values = np.zeros(1000, dtype=np.float32)
    env = audio.Envelope(values)
    from subsync_bench.subio import Cue

    with pytest.raises(Exception) as exc:
        audio.align_envelope(env, [Cue(1, 1.0, 2.0, "x")])
    assert "silent" in str(exc.value).lower() or "speech" in str(exc.value).lower()


def test_already_aligned_has_high_confidence_and_zero_offset():
    sr, hop = audio.SAMPLE_RATE, audio.HOP
    values = np.zeros(int(30 * sr / hop), dtype=np.float32)
    windows = [(1.0, 3.0), (5.0, 7.0), (9.0, 11.0), (14.0, 16.0)]
    for a, b in windows:
        values[int(a * sr / hop) : int(b * sr / hop)] = 1.0
    env = audio.Envelope(values, sample_rate=sr, hop=hop)

    from subsync_bench.subio import Cue

    cues = [Cue(i + 1, a, b, "x") for i, (a, b) in enumerate(windows)]
    result = audio.align_envelope(env, cues, max_offset_seconds=10.0)
    assert abs(result.offset) < 0.1
    assert result.confidence > 0.85


def test_wrong_content_scores_much_lower_than_correct():
    sr, hop = audio.SAMPLE_RATE, audio.HOP
    values = np.zeros(int(60 * sr / hop), dtype=np.float32)
    for a, b in [(2, 5), (9, 12), (16, 19), (24, 27), (33, 36), (42, 45)]:
        values[int(a * sr / hop) : int(b * sr / hop)] = 1.0
    env = audio.Envelope(values, sample_rate=sr, hop=hop)

    from subsync_bench.subio import Cue

    good = [Cue(i + 1, a, b, "x") for i, (a, b) in enumerate(
        [(2, 5), (9, 12), (16, 19), (24, 27), (33, 36), (42, 45)]
    )]
    # same count, different rhythm
    bad = [Cue(i + 1, a, b, "x") for i, (a, b) in enumerate(
        [(1, 1.5), (7, 7.4), (20, 20.5), (31, 31.5), (44, 44.4), (52, 52.5)]
    )]
    g = audio.align_envelope(env, good, max_offset_seconds=10.0)
    b = audio.align_envelope(env, bad, max_offset_seconds=10.0)
    assert g.confidence > b.confidence + 0.15


def test_normalized_confidence_is_bounded():
    sr, hop = audio.SAMPLE_RATE, audio.HOP
    rng = np.random.default_rng(3)
    values = (rng.random(int(20 * sr / hop)) * 0.5).astype(np.float32)
    env = audio.Envelope(values, sample_rate=sr, hop=hop)
    from subsync_bench.subio import Cue

    cues = [Cue(1, 2.0, 5.0, "x")]
    r = audio.align_envelope(env, cues, max_offset_seconds=5.0)
    assert -1.0 <= r.confidence <= 1.0


# --- cue residuals --------------------------------------------------------


def _env_with_windows(windows, total=40.0):
    sr, hop = audio.SAMPLE_RATE, audio.HOP
    values = np.zeros(int(total * sr / hop), dtype=np.float32)
    for a, b in windows:
        values[int(a * sr / hop) : int(b * sr / hop)] = 1.0
    return audio.Envelope(values, sample_rate=sr, hop=hop)


def test_residuals_clean_on_perfect_cues():
    windows = [(2.0, 5.0), (8.0, 11.0), (14.0, 17.0), (20.0, 23.0), (26.0, 29.0)]
    env = _env_with_windows(windows)
    from subsync_bench.subio import Cue

    cues = [Cue(i + 1, a, b, "x") for i, (a, b) in enumerate(windows)]
    res = audio.cue_residuals(env, cues, 0.0)
    assert not any(r.suspect for r in res)
    summary = audio.summarize_residuals(res)
    assert summary["suspect_count"] == 0


def test_residuals_flag_cue_pushed_into_silence():
    windows = [(2.0, 5.0), (8.0, 11.0), (14.0, 17.0), (20.0, 23.0)]
    env = _env_with_windows(windows)
    from subsync_bench.subio import Cue

    cues = [
        Cue(1, 2.0, 5.0, "x"),
        Cue(2, 8.0, 11.0, "x"),
        Cue(3, 5.6, 7.0, "pushed into the gap"),  # was 14-17
        Cue(4, 20.0, 23.0, "x"),
    ]
    res = audio.cue_residuals(env, cues, 0.0)
    flagged = {r.index for r in res if r.suspect}
    assert flagged == {3}


def test_residuals_flag_cue_on_other_speech_region():
    """A cue moved onto a *different* line: onset distance alone may pass, but
    coverage of its own span must catch it."""
    windows = [(2.0, 5.0), (8.0, 11.0), (14.0, 17.0), (20.0, 23.0)]
    env = _env_with_windows(windows)
    from subsync_bench.subio import Cue

    # cue 2's span 8.6-10.6 lies inside 8-11 speech, but starts 0.6s after onset
    cues = [
        Cue(1, 2.0, 5.0, "x"),
        Cue(2, 8.6, 10.6, "x"),
        Cue(3, 14.0, 17.0, "x"),
        Cue(4, 20.0, 23.0, "x"),
    ]
    res = audio.cue_residuals(env, cues, 0.0)
    r2 = next(r for r in res if r.index == 2)
    assert abs(r2.onset_distance) > 0.5


def test_coverage_floor_is_what_catches_the_other_region_case():
    """Isolate the coverage signal.

    Construct a cue whose START is exactly on a speech onset (so onset distance
    passes) but whose span extends past the speech into silence (so coverage
    fails). With the coverage check disabled the cue must stop being flagged —
    this is the test that a mutation of the coverage branch has to break.
    """
    windows = [(2.0, 4.0), (8.0, 10.0), (14.0, 16.0), (20.0, 22.0)]
    env = _env_with_windows(windows)
    from subsync_bench.subio import Cue

    cues = [
        Cue(1, 2.0, 4.0, "ok"),
        Cue(2, 8.0, 10.0, "ok"),
        # starts exactly on the 14.0 onset, but most of its span is silence:
        # 2 s of speech inside a 6 s cue => coverage ~0.33 < 0.35
        Cue(3, 14.0, 20.0, "spans past the speech"),
        Cue(4, 20.0, 22.0, "ok"),
    ]
    res = {r.index: r for r in audio.cue_residuals(env, cues, 0.0)}
    assert abs(res[3].onset_distance) < 0.2, "onset distance must pass here"
    assert res[3].coverage < audio.COVERAGE_FLOOR, "coverage must be low"
    assert res[3].suspect, "low coverage alone must be enough to flag"

    # Now with the coverage check neutered, the same cue must NOT be flagged.
    lenient = {r.index: r for r in audio.cue_residuals(
        env, cues, 0.0, coverage_floor=-1.0)}
    assert not lenient[3].suspect, "with coverage disabled the cue escapes"


def test_residual_summary_fields():
    windows = [(2.0, 5.0), (8.0, 11.0)]
    env = _env_with_windows(windows)
    from subsync_bench.subio import Cue

    cues = [Cue(1, 2.0, 5.0, "x"), Cue(2, 6.0, 7.0, "bad")]
    summary = audio.summarize_residuals(audio.cue_residuals(env, cues, 0.0))
    assert summary["cue_count"] == 2
    assert summary["suspect_count"] == 1
    assert summary["suspect_ratio"] == pytest.approx(0.5)
    assert summary["suspect_indices"] == [2]
    assert "worst" in summary


def test_onset_lag_measurement_on_clean_signal():
    sr, hop = audio.SAMPLE_RATE, audio.HOP
    values = np.zeros(int(20 * sr / hop), dtype=np.float32)
    for a, b in [(2.0, 5.0), (8.0, 11.0), (14.0, 17.0)]:
        values[int(a * sr / hop) : int(b * sr / hop)] = 1.0
    env = audio.Envelope(values, sample_rate=sr, hop=hop)
    from subsync_bench.subio import Cue

    cues = [Cue(1, 2.0, 5.0, "x"), Cue(2, 8.0, 11.0, "x"), Cue(3, 14.0, 17.0, "x")]
    lag = audio.measure_onset_lag(env, cues)
    assert 0.0 <= lag <= 0.4


def test_onset_lag_capped():
    sr, hop = audio.SAMPLE_RATE, audio.HOP
    values = np.zeros(int(20 * sr / hop), dtype=np.float32)
    values[int(15 * sr / hop) : int(18 * sr / hop)] = 1.0
    env = audio.Envelope(values, sample_rate=sr, hop=hop)
    from subsync_bench.subio import Cue

    # cue starts way before any speech -> onset is far away, must be capped
    lag = audio.measure_onset_lag(env, [Cue(1, 1.0, 2.0, "x")])
    assert lag <= 0.4


# --- real media (skipped without fixtures) --------------------------------


def test_real_film_envelope_is_not_silent(envelope):
    assert not envelope.is_silent
    assert envelope.speech_ratio > 0.05
    assert envelope.duration > 100


def test_real_film_offsets_are_accurate(envelope, media_dir):
    truth = subio.load(media_dir / "truth.srt").cues
    late = subio.load(media_dir / "late.srt").cues

    r_truth = audio.align_envelope(envelope, truth, max_offset_seconds=30.0)
    assert abs(r_truth.offset) < 0.25, "already-aligned cues must read ~0"
    assert r_truth.confidence > 0.6

    r_late = audio.align_envelope(envelope, late, max_offset_seconds=30.0)
    assert r_late.offset == pytest.approx(-8.4, abs=0.35)
    assert r_late.confidence > 0.6


def test_real_silent_film_is_detected(silent_film):
    env = audio.envelope_from_media(silent_film)
    assert env.is_silent
