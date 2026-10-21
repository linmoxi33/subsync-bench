# The trust gate, rule by rule

Each rule exists because a specific failure was measured. The numbers come from
`fixtures/build_media.py` (110 s of real synthesised speech, 14 windows) and are
reproduced in `tests/test_gate.py` and `tests/test_e2e.py`.

## R1 — reference has no speech

```python
if speech_ratio <= 0.0: block
```

`ffsubsync` on a silent reference: `retval=0`, output file written, `score: 1756`,
`offset: 0.14`. The file is valid SRT containing the *unshifted* cues. Nothing in
the output says "this is meaningless".

The workbench decodes the audio itself and builds an envelope with a per-file
adaptive threshold. A silent track yields all zeros → `speech_ratio == 0` →
refuse before the engine even runs. Cost: one ffmpeg decode (~1 s for a film).

## R2 — confidence below threshold

```python
if confidence < 0.55: block
```

Measured separation on real speech:

```
correct alignment           0.685 – 0.699
different film              0.285 – 0.407
silent reference            0.000
```

0.55 sits in the gap with ≥0.13 margin on both sides.

This is only possible because the metric is **normalised** (cosine similarity,
bounded in [−1, 1]). Upstream's `score` is a raw correlation peak that scales with
clip length, cue count and speech density; measured values for garbage (1722,
1756, 2675) overlap correct values on other fixtures (5285). No fixed threshold on
that number can work, which is why `--min-score` defaults to 0.0 and never fires.

## R3 — the engine did not apply its own shift

```python
if abs(residual_offset) > 0.8: block
```

We re-measure the *output*: how far would it still want to move? For a correctly
applied shift this is ~0.

Tolerance is calibrated, not chosen. Across five verified-correct outputs:

```
-0.360  -0.210  -0.270  -0.360  -0.360   →  mean -0.31, sd 0.06
```

An output that never got shifted measures −8.25 s. Two orders of magnitude apart,
so 0.8 s is safe against measurement noise while catching any real non-application.

**This rule replaced a wrong one.** The first version blocked when
`confidence - null_confidence < 0.05` ("shifting did not help"). For an
already-aligned subtitle the gain is *legitimately* zero — so that rule rejected
every correct no-op. Caught by running the suite against `truth.srt`; the fix is
recorded here so it is not reintroduced.

## R4 — offset beyond the search window

```python
if abs(offset) > max_offset_seconds: block
```

Default window is ±120 s. An offset at the edge of the window means the
correlation latched onto a different scene rather than finding the true shift.
The wrong-film fixture produced `+40.19 s` with `scale 0.959` — a combination that
can only come from a spurious match.

## R5 — implausible framerate scale

```python
if not (0.90 <= framerate_scale <= 1.10): block
```

Real conversions are narrow: 23.976→24 is 1.001, 23.976→25 is 1.0427,
25→24 is 0.96. Anything beyond ±10% is a mis-detection, not a transfer.

Note that 0.96 is *inside* the range on purpose: the wrong-film case produced
0.959 and R5 correctly does not fire — the case is caught by R2 instead. A rule
that fired on both would be double-counting one defect.

## R6 — cue count changed

```python
if cue_count_in != cue_count_out: block
```

This is the most consequential rule, because the loss is **silent**. When the
framerate guess is wrong, scaled timestamps go negative; `pysubs2` logs
`Skipped subtitle at index N: Start time < 0 seconds` at INFO level and writes
the rest. Measured: 14 cues in, 4 out, `retval=0`.

Losing 10 of 14 lines of dialogue is never acceptable. The workbench:

1. compares cue counts after every engine run;
2. on mismatch, retries once with `--no-fix-framerate`;
3. if the retry also loses cues, R6 blocks the export.

## R7 — many cues still misaligned (warn only)

```python
if suspect_ratio >= 0.15: warn
```

A global shift is an average. If the subtitle comes from a different *cut* —
scenes reordered, footage added — no single offset fixes it, and the overall
confidence can still look acceptable because most cues match.

This is a **warning**, not a block: the result may still be the best available
option, and the user may know the source is imperfect. What matters is that they
see it, and that the specific cue numbers are listed.

## R8 — already aligned (info only)

```python
if abs(offset) < 0.05: info
```

Useful signal: "you did not need to run this". Informational so it never blocks.

---

## Why the gate is a separate module

`gate.evaluate()` is a pure function of numbers. It has no file access, no engine,
no state. That makes it:

- **testable exhaustively** — 22 tests cover every boundary;
- **readable in one screen** — the rules are the whole file;
- **reusable** — the CLI, the web UI and the HTML report all call the same
  function, so they cannot disagree about what "trustworthy" means.

The alternative — scattering `if confidence < 0.55: raise` through the pipeline —
is how tools end up with the UI saying "OK" while the exporter refuses.
