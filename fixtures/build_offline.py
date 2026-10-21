"""Build CI-safe media fixtures without network access.

`build_media.py` produces realistic fixtures using edge-tts, but that needs
network and a TTS service. CI cannot rely on either, so this script synthesises
speech-like audio locally with ffmpeg alone.

The audio is formant-ish: a carrier tone amplitude-modulated by syllable-rate
envelopes with pauses, which gives both webrtcvad (upstream's VAD) and our own
RMS envelope something structured to find. It is NOT real speech — so these
fixtures are only used to prove the plumbing works end to end (project → align →
gate → export). The accuracy thresholds in test_audio.py's real-media tests use
the edge-tts fixtures, which skip cleanly when absent.

That separation is deliberate: a fixture that no VAD recognises cannot test a
sync tool, and a fixture that needs the network cannot gate a build.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).parent / "media"

#: speech windows in seconds, with deliberate gaps
WINDOWS = [
    (2.0, 5.5), (9.0, 12.5), (16.0, 19.5), (23.0, 27.0), (31.0, 34.5),
    (38.0, 42.0), (46.0, 50.0), (54.0, 58.0), (62.0, 65.5), (69.0, 73.0),
    (77.0, 81.0), (85.0, 89.0), (93.0, 97.0), (101.0, 105.0),
]
TOTAL = 110.0
LINES = [
    "The first thing you notice is how quiet the street becomes after midnight.",
    "She kept the letter in a drawer for eleven years before opening it.",
    "Nobody told the engineers that the bridge had already been rebuilt.",
    "There is a particular kind of silence that follows a decision.",
    "He counted the windows twice and got a different answer both times.",
    "The train was late, which meant the whole plan had to be rethought.",
    "Somewhere below the city, water was moving through pipes nobody watched.",
    "It was not the ending anyone expected, but it was the one that fit.",
    "They agreed to meet at the corner where the bookshop used to be.",
    "Every map of the region disagreed about where the river actually went.",
    "The recording ran for six hours and contained almost no useful speech.",
    "By morning the snow had covered the tracks completely.",
    "A single lamp stayed on in the building across the courtyard.",
    "He said he would explain everything, and then he said nothing at all.",
]


def ff(*args: str) -> None:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", "replace")[-1500:] + "\n")
        raise SystemExit(1)


def fmt(t: float) -> str:
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def build_speech_wav(dest: Path) -> None:
    """One 110 s track: syllable-modulated bursts inside each speech window."""
    # Build a volume expression: inside a window, gate the carrier at ~4 Hz to
    # imitate syllables; outside, silence.
    parts = []
    for a, b in WINDOWS:
        parts.append(
            f"between(t,{a:.3f},{b:.3f})*"
            f"(0.55+0.45*sin(2*PI*4.2*(t-{a:.3f})))"
        )
    vol = "+".join(parts)

    # Two carriers give a rough formant impression; the second is modulated so
    # the spectrum is not a single pure tone (a pure tone has no dynamic range
    # and our envelope extractor correctly calls it silence).
    expr = (
        "0.5*sin(2*PI*180*t)+0.3*sin(2*PI*320*t)+0.2*sin(2*PI*900*t)"
    )
    ff(
        "-f", "lavfi",
        "-i", f"aevalsrc='({expr})*({vol})':s=44100:d={TOTAL}",
        "-ac", "1", "-ar", "44100",
        str(dest),
    )


def build() -> None:
    OUT.mkdir(exist_ok=True)
    wav = OUT / "speech.wav"
    build_speech_wav(wav)

    film = OUT / "film.mp4"
    ff(
        "-f", "lavfi", "-i", f"color=c=0x101010:s=640x360:r=24:d={TOTAL}",
        "-i", str(wav),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-shortest",
        str(film),
    )

    silent = OUT / "silent.mp4"
    ff(
        "-f", "lavfi", "-i", f"color=c=0x101010:s=640x360:r=24:d={TOTAL}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(TOTAL),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(silent),
    )

    def write_srt(name: str, shift: float, scale: float = 1.0) -> None:
        blocks = []
        for i, ((a, b), text) in enumerate(zip(WINDOWS, LINES, strict=True)):
            s = max(0.2, a * scale + shift)
            e = max(s + 0.5, b * scale + shift)
            blocks.append(f"{i+1}\n{fmt(s)} --> {fmt(e)}\n{text}\n")
        (OUT / name).write_text("\n".join(blocks), encoding="utf-8")

    write_srt("truth.srt", 0.0)
    write_srt("late.srt", 8.4)
    write_srt("early.srt", -3.7)
    write_srt("fps.srt", 0.0, scale=25 / 24)

    print(f"built {film.name} ({TOTAL:.0f}s), silent.mp4, and 4 subtitle variants")


if __name__ == "__main__":
    build()
