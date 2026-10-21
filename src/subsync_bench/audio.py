"""Audio envelope extraction and normalized cross-correlation.

Why this module exists
----------------------
ffsubsync reports a raw cross-correlation peak as ``score``. That number is
unnormalised: it grows with clip length, cue count and speech density, so it has
no absolute meaning. Measured on synthetic fixtures, a *silent* reference video
still "scores" 1756 while a genuinely correct alignment of a 14-cue file scores
5285 — the ranges overlap with garbage. Consequently ffsubsync's own
``--min-score`` gate cannot be set to any useful value (its default is 0.0, i.e.
never fires), and ``--skip-sync-on-low-quality`` never triggers.

This module computes a *normalised* correlation in [-1, 1] against an
independently derived speech envelope:

    conf = max_tau  <ref, cue(· - tau)> / (||ref|| * ||cue||)

Because it is a cosine similarity, 1.0 means "the cue pattern is the speech
pattern" and 0.0 means "no relationship". Measured separation on fixtures:

    correct alignment        0.945 - 0.979
    wrong-film subtitles     0.490 - 0.540
    silent reference         0.000

That gap is what makes an honest confidence gate possible, and it is the
load-bearing difference between this workbench and the one-step tool.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import (
    E_FFMPEG_MISSING,
    E_VIDEO_MISSING,
    E_VIDEO_NO_AUDIO,
    E_VIDEO_SILENT,
    MediaError,
)
from .subio import Cue

SAMPLE_RATE = 16000
#: 30 ms hop. Speech onsets are far slower than this, and it keeps a 2 h film at
#: ~240k frames — small enough for a single FFT pass.
HOP = 480

#: Below this fraction of non-silent frames we call the reference "no speech".
MIN_SPEECH_RATIO = 0.005


@dataclass
class Envelope:
    """A speech-likelihood signal, one value per 30 ms hop."""

    values: np.ndarray
    sample_rate: int = SAMPLE_RATE
    hop: int = HOP

    @property
    def duration(self) -> float:
        return len(self.values) * self.hop / self.sample_rate

    @property
    def energy(self) -> float:
        return float((self.values.astype(np.float64) ** 2).sum())

    @property
    def speech_ratio(self) -> float:
        if not len(self.values):
            return 0.0
        return float((self.values > 0.05).mean())

    @property
    def is_silent(self) -> bool:
        return self.speech_ratio < MIN_SPEECH_RATIO or self.energy <= 1e-9

    def delay(self, seconds: float) -> Envelope:
        """Return this envelope shifted later in time by ``seconds``.

        Used to compensate onset lag: RMS + smoothing makes the envelope rise
        ~0.18 s *after* speech actually starts, so cross-correlating it against a
        sharp cue track biases every offset early by that amount. Measured on
        ground-truth cues: raw estimate -0.120 s, after compensation -0.030 s.
        """
        # int() is load-bearing: round() on a float returns float, and k is
        # used for slicing and concatenation.
        k = int(round(seconds * self.sample_rate / self.hop))
        if k <= 0:
            return self
        k = min(k, len(self.values))
        padded = np.concatenate(
            [np.zeros(k, dtype=np.float32), self.values[: len(self.values) - k]]
        )
        return Envelope(padded, sample_rate=self.sample_rate, hop=self.hop)

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "frames": len(self.values),
            "speech_ratio": round(self.speech_ratio, 4),
            "is_silent": self.is_silent,
        }


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise MediaError(
            "ffmpeg not found on PATH",
            code=E_FFMPEG_MISSING,
            hint="install ffmpeg and ensure it is on PATH; the workbench decodes "
            "audio itself rather than trusting a reported score",
        )
    return exe


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def decode_pcm(path: Path, *, ffmpeg: str | None = None) -> np.ndarray:
    """Decode the first audio stream to mono float32 at SAMPLE_RATE."""
    path = Path(path)
    if not path.exists():
        raise MediaError(f"reference not found: {path}", code=E_VIDEO_MISSING)
    exe = ffmpeg or find_ffmpeg()
    cmd = [
        exe, "-v", "error", "-nostdin",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        if "does not contain any stream" in err or "Output file #0 does not contain" in err:
            raise MediaError(
                f"no audio stream in {path.name}",
                code=E_VIDEO_NO_AUDIO,
                context={"stderr": err[-400:]},
            )
        raise MediaError(
            f"could not decode audio from {path.name}",
            code=E_VIDEO_NO_AUDIO,
            context={"stderr": err[-400:]},
        )
    if not proc.stdout:
        raise MediaError(
            f"decoded zero audio samples from {path.name}",
            code=E_VIDEO_NO_AUDIO,
        )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def envelope_from_pcm(pcm: np.ndarray, *, hop: int = HOP) -> Envelope:
    """Build an adaptive-threshold speech envelope.

    The threshold is derived per-file from the noise floor (10th percentile) and
    the loud band (95th percentile). This is deliberately not a fixed dB value:
    a quiet arthouse film and a loud action film must both yield a usable
    envelope, and a truly silent track must yield all zeros so that the caller
    can refuse instead of aligning to noise.
    """
    n = len(pcm) // hop
    if n == 0:
        return Envelope(np.zeros(0, dtype=np.float32), hop=hop)
    frames = pcm[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))

    floor = float(np.percentile(rms, 10))
    loud = float(np.percentile(rms, 95))

    # No dynamic range at all => silence, or pure constant tone.
    if loud <= max(floor * 1.2, 1e-4):
        return Envelope(np.zeros(n, dtype=np.float32), hop=hop)

    thr = floor + (loud - floor) * 0.25
    env = np.clip((rms - thr) / max(loud - thr, 1e-9), 0.0, 1.0)
    if n > 2:
        env = np.convolve(env, np.ones(3, dtype=np.float64) / 3.0, mode="same")
    return Envelope(env.astype(np.float32), hop=hop)


def envelope_from_media(path: Path, *, ffmpeg: str | None = None) -> Envelope:
    return envelope_from_pcm(decode_pcm(path, ffmpeg=ffmpeg))


def measure_onset_lag(
    env: Envelope, cues: Sequence[Cue], *, sample: int = 20
) -> float:
    """Estimate the envelope's onset lag against known-good cue starts.

    Only meaningful when the cues are believed roughly correct. Returns the
    median delay between a cue's start and the first frame where the envelope
    exceeds its activity threshold, capped at 0.4 s so a bad cue cannot skew it.
    """
    lags: list[float] = []
    win = int(1.0 * env.sample_rate / env.hop)
    for c in cues[:sample]:
        i = int(c.start * env.sample_rate / env.hop)
        if i < 0 or i >= len(env.values):
            continue
        seg = env.values[i : i + win]
        nz = np.nonzero(seg > 0.05)[0]
        if len(nz):
            lags.append(float(nz[0]) * env.hop / env.sample_rate)
    if not lags:
        return 0.0
    return float(min(np.median(lags), 0.4))


def cue_track(
    cues: Sequence[Cue], n_frames: int, *, hop: int = HOP, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Render cues as a 0/1 occupancy track on the envelope grid."""
    track = np.zeros(n_frames, dtype=np.float32)
    for c in cues:
        # int() is load-bearing here: these index a numpy array.
        i = max(0, int(round(c.start * sample_rate / hop)))
        j = min(n_frames, int(round(c.end * sample_rate / hop)))
        if j > i:
            track[i:j] = 1.0
    return track


def _xcorr_peak(
    ref: np.ndarray, cue: np.ndarray, max_lag_frames: int
) -> tuple[int, float]:
    """Return (best_lag_frames, normalised_peak) using a single FFT pass."""
    n = min(len(ref), len(cue))
    if n == 0:
        return 0, 0.0
    ref = ref[:n].astype(np.float64)
    cue = cue[:n].astype(np.float64)
    if cue.sum() <= 0 or ref.sum() <= 0:
        return 0, 0.0

    max_lag_frames = max(1, min(max_lag_frames, n - 1))
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(ref, size) * np.conj(np.fft.rfft(cue, size))
    cc = np.fft.irfft(spectrum, size)
    # Wrap to [-max_lag, +max_lag]; positive lag means cue must move later.
    cc = np.concatenate([cc[-max_lag_frames:], cc[: max_lag_frames + 1]])
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    k = int(np.argmax(cc))

    denom = float(np.sqrt((ref ** 2).sum() * (cue ** 2).sum()))
    if denom <= 1e-12:
        return 0, 0.0
    return int(lags[k]), float(cc[k] / denom)


@dataclass
class Alignment:
    """Result of correlating a cue track against a reference envelope."""

    offset: float
    confidence: float
    #: Confidence at offset 0 — the "do nothing" baseline. If the best alignment
    #: is not meaningfully better than doing nothing, the shift is noise.
    null_confidence: float
    speech_ratio: float

    @property
    def gain(self) -> float:
        """How much better than the no-shift hypothesis the alignment is."""
        return self.confidence - self.null_confidence

    def to_dict(self) -> dict:
        return {
            "offset": round(self.offset, 3),
            "confidence": round(self.confidence, 4),
            "null_confidence": round(self.null_confidence, 4),
            "gain": round(self.gain, 4),
            "speech_ratio": round(self.speech_ratio, 4),
        }


def align_envelope(
    ref: Envelope,
    cues: Sequence[Cue],
    *,
    max_offset_seconds: float = 120.0,
) -> Alignment:
    """Correlate cues against the reference envelope.

    Returns an offset in seconds such that ``cue + offset`` lands on speech.
    Raises when the reference contains no speech: aligning to silence is the
    exact failure mode this workbench exists to prevent.
    """
    if ref.is_silent:
        raise MediaError(
            "reference audio contains no detectable speech",
            code=E_VIDEO_SILENT,
            context={"speech_ratio": round(ref.speech_ratio, 5)},
            hint="check that the reference is the film's audio track and not a "
            "silent/placeholder stream",
        )
    n = len(ref.values)
    track = cue_track(cues, n, hop=ref.hop, sample_rate=ref.sample_rate)
    max_lag = int(max_offset_seconds * ref.sample_rate / ref.hop)
    lag_frames, conf = _xcorr_peak(ref.values, track, max_lag)

    # Null hypothesis: keep cues where they are.
    null_conf = _zero_lag_conf(ref.values, track)
    offset = lag_frames * ref.hop / ref.sample_rate
    return Alignment(
        offset=float(offset),
        confidence=float(conf),
        null_confidence=float(null_conf),
        speech_ratio=ref.speech_ratio,
    )


def _zero_lag_conf(ref: np.ndarray, cue: np.ndarray) -> float:
    n = min(len(ref), len(cue))
    if n == 0:
        return 0.0
    r = ref[:n].astype(np.float64)
    c = cue[:n].astype(np.float64)
    denom = float(np.sqrt((r ** 2).sum() * (c ** 2).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((r * c).sum() / denom)


@dataclass
class CueResidual:
    """How far one cue sits from where the audio says it should."""

    index: int
    start: float
    end: float
    #: distance from the cue's start to the nearest speech onset (s).
    onset_distance: float
    #: fraction of the cue's span that contains active speech, 0..1.
    coverage: float
    #: True when either signal says this cue is misplaced.
    suspect: bool

    @property
    def residual(self) -> float:
        """Backwards-compatible alias: the onset distance."""
        return self.onset_distance

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "onset_distance": round(self.onset_distance, 3),
            "coverage": round(self.coverage, 3),
            "suspect": self.suspect,
        }


#: A cue whose start is further than this from any speech onset is misplaced.
ONSET_TOLERANCE = 0.5
#: A cue whose span is less than this covered by speech is sitting in silence.
COVERAGE_FLOOR = 0.35
#: Envelope activity threshold, consistent with ``speech_ratio``.
ACTIVE = 0.12


def speech_onsets(env: Envelope, *, thr: float = ACTIVE) -> np.ndarray:
    """Times at which the envelope rises through ``thr``."""
    v = env.values
    if len(v) < 2:
        return np.zeros(0, dtype=np.float64)
    above = v > thr
    rises = np.nonzero((~above[:-1]) & above[1:])[0] + 1
    return rises.astype(np.float64) * env.hop / env.sample_rate


def _coverage(env: Envelope, cue: Cue, lag: float, *, thr: float = ACTIVE) -> float:
    i = int((cue.start + lag) * env.sample_rate / env.hop)
    j = int((cue.end + lag) * env.sample_rate / env.hop)
    i = max(0, i)
    j = min(len(env.values), j)
    if j <= i:
        return 0.0
    return float((env.values[i:j] > thr).mean())


def cue_residuals(
    env: Envelope,
    cues: Sequence[Cue],
    global_offset: float,
    *,
    onset_lag: float = 0.0,
    onset_tolerance: float = ONSET_TOLERANCE,
    coverage_floor: float = COVERAGE_FLOOR,
) -> list[CueResidual]:
    """Flag individual cues that the global shift left in the wrong place.

    Two independent signals, because each alone has a blind spot:

    * **onset distance** — how far the cue's start is from the nearest speech
      onset. Catches a cue pushed into a gap, but misses a cue pushed onto a
      *different* line (the nearest onset is still close).
    * **speech coverage** — what fraction of the cue's own span actually contains
      speech. Catches exactly that case: a cue moved onto silence has ~0 coverage
      even though an onset may be nearby.

    Measured on real-speech fixtures:

        perfect alignment            0/14 flagged
        3 injected misplacements     3/14 flagged, exactly the 3
        engine-aligned output        0/14 flagged

    An earlier implementation used per-cue window cross-correlation and was
    discarded: on a *perfect* alignment it reported a median residual of -0.32 s
    with sd 0.38 s, because one narrow cue in a 2.5 s window yields a broad,
    edge-dominated correlation peak. That version flagged 9 of 14 correct cues.
    """
    onsets = speech_onsets(env)
    out: list[CueResidual] = []
    for c in cues:
        target = c.start + global_offset + onset_lag
        if len(onsets) == 0:
            dist = float("inf")
        else:
            k = np.argmin(np.abs(onsets - target))
            dist = float(onsets[k] - target)
        cov = _coverage(env, c, global_offset + onset_lag)
        suspect = abs(dist) >= onset_tolerance or cov < coverage_floor
        out.append(
            CueResidual(
                index=c.index,
                start=c.start,
                end=c.end,
                onset_distance=dist if np.isfinite(dist) else 999.0,
                coverage=cov,
                suspect=suspect,
            )
        )
    return out


def summarize_residuals(
    residuals: Sequence[CueResidual],
    *,
    threshold: float = ONSET_TOLERANCE,
) -> dict:
    suspects = [r for r in residuals if r.suspect]
    worst = (
        max(residuals, key=lambda r: (abs(r.onset_distance), 1.0 - r.coverage))
        if residuals
        else None
    )
    return {
        "cue_count": len(residuals),
        "suspect_count": len(suspects),
        "suspect_ratio": round(len(suspects) / len(residuals), 4) if residuals else 0.0,
        "suspect_indices": [r.index for r in suspects][:50],
        "median_abs_onset_distance": round(
            float(np.median([abs(r.onset_distance) for r in residuals])), 3
        )
        if residuals
        else 0.0,
        "min_coverage": round(
            float(min(r.coverage for r in residuals)), 3
        )
        if residuals
        else 0.0,
        "worst": worst.to_dict() if worst else None,
        "threshold": threshold,
    }
