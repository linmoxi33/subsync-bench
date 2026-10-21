# subsync-bench

**A workbench for getting a downloaded subtitle to actually line up with a video.**

Built around [`ffsubsync`](https://github.com/smacke/ffsubsync) — a tool that does
exactly one thing: shift a subtitle so its speech pattern matches the video's
audio, then print an offset. This project keeps that engine and wraps the whole
job around it:

```
import → measure → align → preview → compare → select → export → report
```

Every stage is separately inspectable, and the export step re-measures the result
instead of trusting a receipt.

---

## Why this exists

`ffsubsync` answers *"what offset makes these two patterns correlate best?"*
It does not answer *"is that offset trustworthy?"* Measured behaviour of 0.5.1:

| Situation | `ffsubsync` | This workbench |
|---|---|---|
| Reference audio is silent | exits **0**, writes a shifted file anyway | refuses before running (`E_VIDEO_SILENT`) |
| Subtitle is from a different film | exits **0**, writes a confidently wrong file | confidence 0.285 < 0.55 → **blocked** |
| `--min-score 1000000` | **ignored**; still writes output | n/a — gate uses normalized confidence |
| `--skip-sync-on-low-quality` | never fires with default `--min-score 0.0` | replaced by a calibrated gate |
| Wrong framerate guess | silently **drops cues** (14 → 4 measured) | retries without framerate fixing; cue loss blocks export |
| Which specific cues are still wrong | not reported | per-cue onset distance + speech coverage |

The two numbers behind the gate, measured on real synthesised speech:

```
correct alignment      confidence 0.685 – 0.699
different-film subs    confidence 0.285 – 0.407
silent reference       confidence 0.000
```

The upstream `score` cannot support this: it is an unnormalised correlation peak
that grows with clip length and speech density, so a silent reference still
"scores" 1756 while a correct 14-cue alignment scores 5285 — overlapping ranges.

---

## Install

```bash
pip install -e ".[engine]"        # CLI + alignment engine
pip install -e ".[engine,web]"    # + browser workbench
```

Requires `ffmpeg` on `PATH`. `ffprobe` is optional (upstream uses it only to
enumerate embedded subtitle tracks; without it, alignment falls back to the audio
track and reports a warning).

---

## CLI

```bash
subsync-bench doctor                      # check ffmpeg / engine / numpy
subsync-bench inspect video.mkv sub.srt   # measure inputs, no engine needed
subsync-bench align  proj --video video.mkv --subtitle sub.srt
subsync-bench preview proj                # timeline + per-cue diff → HTML
subsync-bench revs    proj                # list attempts with verdicts
subsync-bench select  proj r001
subsync-bench export  proj                # re-measures, then writes
subsync-bench report  proj                # one self-contained HTML
```

Exit codes are part of the contract:

| code | meaning |
|---|---|
| 0 | success |
| 1 | unexpected error |
| 2 | bad arguments |
| 3 | environment/input problem (missing ffmpeg, no audio, unreadable subtitle) |
| 4 | ran, but the result was **refused** by the trust gate |
| 5 | workspace/state problem (no project, nothing selected, destination exists) |

`3` vs `4` is deliberate: CI can distinguish "I could not run" from "I ran and
declined the answer".

---

## Browser workbench

```bash
subsync-bench-serve --open          # http://127.0.0.1:8765
```

Binds to loopback. Upload a video + subtitle, align, inspect the timeline, switch
between attempts, and export — the same pipeline the CLI uses, not a parallel
implementation.

---

## What each stage actually does

**Import.** Copies inputs into the project and hashes them. Originals are never
modified, so any experiment is safe.

**Measure.** Decodes audio with ffmpeg and builds a speech envelope with a
per-file adaptive threshold (10th percentile noise floor, 95th percentile loud
band). A silent track yields an all-zero envelope, which is how "no speech" is
detected rather than assumed.

**Align.** Calls `ffsubsync` in-process. Its output goes to a staging path inside
the project — never straight to the user — and is re-parsed and validated before
being accepted. If cue count changes, the attempt is retried with framerate
fixing disabled.

**Preview.** A self-contained SVG: speech envelope, cues before, cues after, and
dashed connectors showing how far each cue moved. Suspect cues are outlined. No
JS, no external assets.

**Compare.** Every attempt is a *revision* with a JSON receipt: input digests,
engine version, parameters, offset, confidence, per-cue residuals. Revisions
survive across sessions; a changed input marks dependents **stale** rather than
silently reusing them.

**Export.** Re-measures the cues about to be written against the video on disk
*now*. Three separate permissions, deliberately not merged:

| flag | allows |
|---|---|
| `--force` | overwrite an existing file; accept a stale receipt |
| `--allow-override` | export a revision the gate rejected (you inspected it) |
| `--skip-recheck` | skip the export-time re-measurement — the only switch that does |

`--force` never disables the re-check. An earlier version conflated them, which
meant a video swapped for a silent one could be exported by adding a flag users
add habitually. There is a regression test for exactly that.

---

## Layout

```
src/subsync_bench/
  subio.py      SRT parsing/serialising, encoding detection (UTF-8/GBK/UTF-16…)
  audio.py      envelope, normalized correlation, per-cue residuals
  engine.py     ffsubsync adapter, staging, output verification
  gate.py       the trust rules R1–R8
  pipeline.py   the stages, wired together
  workspace.py  project manifest, revisions, receipts, input integrity
  preview.py    timeline SVG + cue diff table
  report.py     self-contained HTML report
  cli.py        command line
  webapp.py     HTTP layer over the same pipeline
  web/index.html  the workbench UI
fixtures/build_media.py   builds real-speech test media (edge-tts + ffmpeg)
tests/                    148 tests
```

---

## Tests

```bash
pytest -q                              # 148 tests, ~2 min (align is slow)
pytest -q --cov=subsync_bench          # 89% statement coverage
```

Media fixtures are generated by `fixtures/build_media.py` (needs `ffmpeg` and
network for `edge-tts`). Tests that need them **skip with an instruction**, never
pass silently.

The fixture design matters: the first attempt used modulated pink noise as a
"speech surrogate", and `ffsubsync`'s VAD — which is trained on real speech —
found only 5 scattered segments in it, aligned garbage, and dropped 10 of 14
cues. A fixture no VAD recognises cannot test a sync tool.

---

## Known limits

- **Global shift only.** A subtitle from a different *cut* (scenes reordered)
  cannot be fixed by one offset. The workbench detects this (R7, suspect ratio)
  and names the offending cue numbers, but does not auto-split scenes.
- **Cue text is never translated or edited.** Only timing changes.
- **ASS/SSA styling is not preserved.** Parsing is SRT-oriented; other formats
  are accepted for input but output is always SRT.
- **`ffprobe` absent** means embedded subtitle tracks cannot be used as a
  reference; audio-only reference still works.
- **No GPU path.** A 2-hour film takes tens of seconds on CPU; acceptable, not
  instant.

---

## License

MIT. `ffsubsync` is MIT by Stephen Macke; see `NOTICE`.
