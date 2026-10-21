"""Build fixtures from REAL synthesised speech.

Why this matters: the first fixture used modulated pink noise as a "speech
surrogate". ffsubsync's VAD (webrtcvad) is trained on real speech and found only
5 scattered segments in it, so the engine aligned garbage and dropped 10 of 14
cues. A fixture that no VAD recognises cannot test a sync tool.

This script synthesises actual sentences with edge-tts, places each at a known
timestamp, and emits the matching SRT with exact ground truth.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import edge_tts

HERE = Path(__file__).parent
OUT = HERE / "media"
OUT.mkdir(exist_ok=True)

SENTENCES = [
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

# Ground truth speech windows, seconds. Gaps are deliberate: a sync tool has to
# find the *pattern* of speech, not just its presence.
WINDOWS = [
    (2.0, 6.0), (9.0, 13.5), (17.0, 20.5), (24.0, 28.0), (31.5, 35.0),
    (39.0, 43.0), (47.0, 51.0), (55.0, 59.5), (63.0, 66.5), (70.0, 74.0),
    (78.0, 82.0), (86.0, 90.0), (94.0, 97.5), (101.0, 105.0),
]
TOTAL = 110.0


async def synth(text: str, dest: Path) -> None:
    comm = edge_tts.Communicate(text, "en-US-AriaNeural", rate="+8%")
    await comm.save(str(dest))


def ff(*args: str) -> None:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True)
    if r.returncode != 0:
        print(r.stderr.decode("utf-8", "replace")[-1200:])
        raise SystemExit(1)


async def main() -> None:
    clips = []
    for i, (sent, (a, b)) in enumerate(zip(SENTENCES, WINDOWS, strict=True)):
        raw = OUT / f"clip{i:02d}.mp3"
        if not raw.exists():
            await synth(sent, raw)
        dur = b - a
        # fit the clip into its window: pad with silence, never stretch
        norm = OUT / f"clip{i:02d}.wav"
        ff("-i", str(raw), "-af",
           f"apad=whole_dur={dur:.3f},atrim=0:{dur:.3f},aresample=44100",
           "-ac", "1", "-ar", "44100", str(norm))
        clips.append((norm, a, dur))
        print(f"  clip {i:02d}: {a:6.2f}s +{dur:.2f}s  {sent[:44]}")

    # lay every clip on a silent 110 s bed at its true offset
    inputs: list[str] = []
    filters: list[str] = []
    for i, (path, a, _d) in enumerate(clips):
        inputs += ["-i", str(path)]
        filters.append(f"[{i}:a]adelay={int(a*1000)}|{int(a*1000)}[d{i}]")
    mix_inputs = "".join(f"[d{i}]" for i in range(len(clips)))
    filters.append(f"{mix_inputs}amix=inputs={len(clips)}:normalize=0[mixed]")
    filters.append(f"[mixed]apad=whole_dur={TOTAL}[out]")

    wav = OUT / "speech.wav"
    ff(*inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
       "-ac", "1", "-ar", "44100", str(wav))

    # mux a black video track so it behaves like a real media file
    video = OUT / "film.mp4"
    ff("-f", "lavfi", "-i", f"color=c=0x101010:s=640x360:r=24:d={TOTAL}",
       "-i", str(wav), "-c:v", "libx264", "-preset", "ultrafast",
       "-c:a", "aac", "-b:a", "96k", "-shortest", str(video))

    # also a silent version, to prove the gate refuses it
    silent = OUT / "silent.mp4"
    ff("-f", "lavfi", "-i", f"color=c=0x101010:s=640x360:r=24:d={TOTAL}",
       "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
       "-t", str(TOTAL), "-c:v", "libx264", "-preset", "ultrafast",
       "-c:a", "aac", str(silent))

    # ground-truth SRT
    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int(t % 3600 // 60)
        s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    lines = []
    for i, ((a, b), sent) in enumerate(zip(WINDOWS, SENTENCES, strict=True)):
        lines.append(f"{i+1}\n{fmt(a)} --> {fmt(b)}\n{sent}\n")
    (OUT / "truth.srt").write_text("\n".join(lines), encoding="utf-8")

    # +8.4 s drift, the realistic "downloaded from a different release" case
    SHIFT = 8.4
    lines = []
    for i, ((a, b), sent) in enumerate(zip(WINDOWS, SENTENCES, strict=True)):
        lines.append(f"{i+1}\n{fmt(a+SHIFT)} --> {fmt(b+SHIFT)}\n{sent}\n")
    (OUT / "late.srt").write_text("\n".join(lines), encoding="utf-8")

    # -3.7 s, subtitles arriving early
    lines = []
    for i, ((a, b), sent) in enumerate(zip(WINDOWS, SENTENCES, strict=True)):
        lines.append(f"{i+1}\n{fmt(max(0.2, a-3.7))} --> {fmt(b-3.7)}\n{sent}\n")
    (OUT / "early.srt").write_text("\n".join(lines), encoding="utf-8")

    # 23.976 -> 25 fps drift: timestamps scale by 24/25 then shift
    lines = []
    for i, ((a, b), sent) in enumerate(zip(WINDOWS, SENTENCES, strict=True)):
        lines.append(f"{i+1}\n{fmt(a*25/24)} --> {fmt(b*25/24)}\n{sent}\n")
    (OUT / "fps.srt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nbuilt {video.name} ({TOTAL:.0f}s, {len(clips)} speech windows)")
    print("  truth.srt  late.srt (+8.4s)  early.srt (-3.7s)  fps.srt (x25/24)")


if __name__ == "__main__":
    asyncio.run(main())
