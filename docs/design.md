# Design notes

Why the workbench is shaped the way it is, and which decisions were reversed after
measurement.

## The starting point

`ffsubsync` is a good engine wrapped in a one-line contract: shift the subtitle,
print the offset. The contract has three gaps that matter as soon as a human has
to trust the output:

1. **It does not know whether the answer is right.** `score` is an unnormalised
   correlation peak.
2. **It never refuses.** Every failure mode measured returns `retval=0` and
   writes an output file.
3. **There is no state.** No record of what was tried, with which parameters,
   against which inputs.

The workbench adds exactly those three things and nothing else. It does not
re-implement alignment; the engine is still `ffsubsync`.

## Measured upstream behaviour (0.5.1)

All figures below come from `fixtures/build_media.py` output — 110 s of real
synthesised speech in 14 windows — and are reproduced in the test suite.

| Input | `retval` | output written | upstream `score` | our confidence |
|---|---|---|---|---|
| correct (`truth.srt`) | 0 | yes | 8129 | 0.699 |
| +8.4 s late | 0 | yes | 8144 | 0.699 |
| −3.7 s early | 0 | yes | 7788 | 0.685 |
| different film | 0 | yes | 1722–2675 | 0.285–0.407 |
| silent reference | 0 | **yes** | 1756 | 0.000 (refused) |
| `--min-score 1000000` | 0 | yes | unchanged | — |

Two conclusions drove the design:

**`score` cannot be thresholded.** 1756 (silent, garbage) sits between 1722
(different film, garbage) and 5285 (correct, on a different fixture). The ranges
overlap, so any single `--min-score` either passes garbage or rejects good work.
Hence `--min-score` defaults to 0.0 and `--skip-sync-on-low-quality` never fires.

**`retval` cannot be trusted.** A silent reference produces a valid SRT
containing the *unshifted* cues. The file is well-formed; it is simply wrong.
Only re-measuring the output catches this.

## The normalized confidence

Instead of the raw peak, compute a cosine similarity between the speech envelope
and the cue occupancy track, maximised over lag:

```
conf(τ) = <ref, cue(· − τ)> / (‖ref‖ · ‖cue‖)
```

This is bounded in [−1, 1], so 1.0 means "the cue pattern is the speech pattern"
independent of clip length, cue count or speech density. That boundedness is what
makes a fixed threshold meaningful.

Two details that took iteration:

**Adaptive threshold, not fixed dB.** The envelope threshold is derived per file
from the 10th and 95th percentile of frame RMS. A fixed dB value works for one
mastering style and fails for the next; the percentile approach also yields
all-zero output for a silent track, which is how "no speech" is detected rather
than guessed.

**Onset lag compensation.** RMS framing plus 3-frame smoothing makes the envelope
rise ~0.18 s *after* speech starts. Cross-correlating that against a sharp cue
track biases every estimate early. Measured on ground-truth cues: raw estimate
−0.120 s, after compensation −0.030 s. Without the compensation, a *perfect*
alignment reads as a real error and the gate rejects correct work.

## Per-cue residuals: the first implementation was wrong

Goal: name the specific cues a global shift left misplaced.

**Attempt 1 — local cross-correlation.** For each cue, take a ±2.5 s window of
the envelope, correlate against the cue's own occupancy, compare the local lag to
the global shift.

Result on **perfectly aligned** cues: median residual −0.315 s, sd 0.384 s,
9 of 14 cues flagged. Useless. Cause: one narrow cue inside a 2.5 s window gives a
broad correlation peak dominated by window-edge effects, not by real misalignment.

**Attempt 2 — onset distance + coverage.** Two independent signals:

- *onset distance* — how far the cue's start is from the nearest speech onset.
  Catches a cue pushed into a gap.
- *coverage* — fraction of the cue's own span containing speech. Catches a cue
  pushed onto a *different* line, where an onset is nearby but the span is empty.

Result:

| input | flagged |
|---|---|
| perfect alignment | 0 / 14 |
| 3 injected misplacements | 3 / 14 (exactly those 3) |
| engine-aligned output | 0 / 14 |

The discarded version is documented in `audio.cue_residuals`'s docstring so it is
not reintroduced as an "improvement".

## The gate

Eight rules, each traceable to a measured failure:

| rule | blocks when | why |
|---|---|---|
| R1 | reference has no speech | aligning to silence is meaningless |
| R2 | confidence < 0.55 | measured gap: correct ≥ 0.685, wrong ≤ 0.407 |
| R3 | output still wants to move > 0.8 s | engine did not apply its own shift |
| R4 | \|offset\| > search window | correlation latched onto another scene |
| R5 | framerate scale outside [0.90, 1.10] | real conversions are 1.001 / 1.0427 |
| R6 | cue count changed | text was lost |
| R7 | >15% of cues still misaligned | **warns** — usually a different cut |
| R8 | offset ≈ 0 | informational: already aligned |

R3 replaced an earlier "confidence gain over no-shift baseline" rule that was
**wrong by construction**: for an already-aligned subtitle the gain is
legitimately zero, so that rule blocked every correct no-op. Caught by running the
suite against `truth.srt`.

R3's tolerance is calibrated, not chosen: across five verified-correct outputs the
residual was −0.31 s ± 0.06 s, while an unaligned file measures −8.25 s. Two
orders of magnitude apart, so 0.8 s is safe in both directions.

## Destructive framerate guessing

When the framerate guess is wrong (it tries 23.976/25 on a file that needs no
rescale), scaled cues go negative and `pysubs2` silently **drops** them while
writing. Measured: a 14-cue file came back with 4 cues, `retval=0`.

The workbench compares cue counts and retries once with `--no-fix-framerate`. If
the retry also loses cues, R6 blocks the export. Losing dialogue is never an
acceptable outcome.

## Export permissions are separate on purpose

| flag | grants | does NOT grant |
|---|---|---|
| `--force` | overwrite destination, accept stale receipt | skipping the re-check |
| `--allow-override` | export a gate-rejected revision | skipping the re-check |
| `--skip-recheck` | skip the re-check | anything else |

An earlier version had `--allow-override` imply "skip the re-check", and `--force`
short-circuit the whole verification block. Both were caught by a test that swaps
the video for a silent file and asserts the export is refused. `--force` is a flag
users add habitually; it must never disable a safety check.

## Project state

Inputs are copied in and hashed. Every attempt becomes a revision with a JSON
receipt: input digests, engine version, parameters, offset, confidence, per-cue
residuals, verdict.

`verify_inputs` re-hashes from **disk**, not from the manifest. Comparing
revision-vs-manifest only detects "the manifest was updated"; it misses a file
edited in place, where the manifest still holds the old hash and every revision
silently looks fresh. That bug was found by a test and fixed.

## Preview

Self-contained SVG, no JS, no external assets — it has to survive leaving the
workbench (email, ticket, archive). Three lanes: speech envelope, cues before,
cues after, with dashed connectors showing displacement.

Two rendering bugs were caught only by looking at the output:

- The envelope was drawn as one bar per 30 ms frame, which at 1000 px wide reads
  as a comb of stripes rather than loudness. Replaced with a filled area polygon.
- That polygon's closing points were emitted as `"y,x"` instead of `"x,y"`,
  stretching the fill across the entire canvas. Structurally valid SVG, visually
  wrong. The test suite now asserts every coordinate is inside the viewport.

Neither would have been caught by asserting on numbers.

## What is deliberately not done

- **Scene-by-scene realignment.** R7 reports it; fixing it needs a different
  algorithm (per-scene offset search), not a bigger threshold.
- **Translation or text editing.** Timing only.
- **ASS/SSA style preservation.** Output is SRT.
- **GPU acceleration.** Tens of seconds for a feature film on CPU is acceptable.
