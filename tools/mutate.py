"""Mutation test: prove the suite actually catches defects.

A green suite proves nothing on its own. This injects small, realistic defects
into the source, runs the full suite, and records which mutants were killed.

Methodology, deliberately strict:

* Compare the SET of failing test ids against the baseline, not just counts — a
  mutant that breaks test A while fixing test B looks "killed" by count.
* Run the full suite every time; ``-k`` filtering changes what is reachable.
* Restore the file from bytes and verify the sha256 before the next mutant, so a
  crash cannot leave a defect in the tree (this bit us before).
* A mutant that survives is reported with the test ids that *should* have caught
  it, so the gap is actionable.

Do not touch the working tree while this runs. It edits tracked files in place
and restores them after each mutant, so a concurrent `git commit`, `git push` or
remote verification will capture a deliberately broken file. `tools/push_api.py`
refuses to publish a dirty tree for exactly this reason.

Output is flushed per line so progress is visible when redirected to a file —
without that, a 25-minute run appears to produce nothing until it ends.
"""

from __future__ import annotations

import functools
import hashlib
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Flush per line so a 25-minute run redirected to a file shows progress as it
# goes instead of appearing to produce nothing until the end.
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "subsync_bench"
PY = sys.executable


@dataclass
class Mutant:
    name: str
    file: str
    old: str
    new: str
    intent: str


MUTANTS = [
    Mutant(
        "gate: confidence threshold off by a lot",
        "gate.py",
        "if confidence < confidence_threshold:",
        "if confidence < confidence_threshold * 0.5:",
        "R2 must block low confidence; halving the bar should let garbage through",
    ),
    Mutant(
        "gate: silent reference no longer blocks",
        "gate.py",
        "    if speech_ratio <= 0.0:",
        "    if False:",
        "R1 must fire on a silent reference",
    ),
    Mutant(
        "gate: cue loss no longer blocks",
        "gate.py",
        "    if cue_count_in != cue_count_out:",
        "    if False:",
        "R6 must fire when cues are dropped",
    ),
    Mutant(
        "gate: residual tolerance hugely relaxed",
        "gate.py",
        "    if abs(residual_offset) > residual_tolerance:",
        "    if abs(residual_offset) > residual_tolerance * 100:",
        "R3 must catch an unapplied shift",
    ),
    Mutant(
        "gate: framerate bounds removed",
        "gate.py",
        "    if not (FRAMERATE_MIN <= framerate_scale <= FRAMERATE_MAX):",
        "    if False:",
        "R5 must catch an implausible rescale",
    ),
    Mutant(
        "audio: silent detection always false",
        "audio.py",
        "        return self.speech_ratio < MIN_SPEECH_RATIO or self.energy <= 1e-9",
        "        return False",
        "silent media must be detected",
    ),
    Mutant(
        "audio: envelope threshold ignores noise floor",
        "audio.py",
        "    thr = floor + (loud - floor) * 0.25",
        "    thr = 0.0",
        "adaptive threshold must reject noise-only tracks",
    ),
    Mutant(
        "audio: correlation normalisation dropped",
        "audio.py",
        "    return int(lags[k]), float(cc[k] / denom)",
        "    return int(lags[k]), float(cc[k])",
        "confidence must be normalised to be thresholdable",
    ),
    Mutant(
        "audio: onset lag compensation removed",
        "audio.py",
        "        k = int(round(seconds * self.sample_rate / self.hop))",
        "        k = 0",
        "onset compensation must actually shift the envelope",
    ),
    Mutant(
        "audio: coverage floor disabled",
        "audio.py",
        "        suspect = abs(dist) >= onset_tolerance or cov < coverage_floor",
        "        suspect = abs(dist) >= onset_tolerance",
        "coverage must catch a cue moved onto other speech",
    ),
    Mutant(
        "subio: GBK misdecoded as utf-16 again",
        "subio.py",
        '    if raw[:2] in (b"\\xff\\xfe", b"\\xfe\\xff"):',
        "    if True:",
        "encoding detection must not speculatively try utf-16",
    ),
    Mutant(
        "subio: reversed timecodes no longer swapped",
        "subio.py",
        "            start, end = end, start",
        "            pass",
        "reversed cues must be repaired and reported",
    ),
    Mutant(
        "subio: millisecond padding wrong",
        "subio.py",
        '    ms = int(ms_raw.ljust(3, "0"))',
        "    ms = int(ms_raw)",
        '"1,5" must mean 500 ms, not 5 ms',
    ),
    Mutant(
        "workspace: stale detection compares to manifest",
        "workspace.py",
        "                if disk.get(key) != recorded:",
        "                if self.manifest[key][\"sha256\"] != recorded:",
        "stale must be detected against the disk, not the manifest",
    ),
    Mutant(
        "workspace: manifest write not atomic",
        "workspace.py",
        "        os.replace(tmp, self.manifest_path)",
        "        self.manifest_path.write_text(tmp.read_text(encoding='utf-8'), encoding='utf-8')",
        "a half-written manifest must not be possible",
    ),
    Mutant(
        "pipeline: recheck skipped when force",
        "pipeline.py",
        "    if not skip_recheck:\n        _recheck_before_export(project, cues, live_check)",
        "    if False:\n        _recheck_before_export(project, cues, live_check)",
        "the export recheck must run by default",
    ),
    Mutant(
        "pipeline: cue-loss retry never happens",
        "pipeline.py",
        "    if len(out_cues) != len(sub.cues) and framerate_ratio is None:",
        "    if False:",
        "destructive framerate guessing must trigger a retry",
    ),
    Mutant(
        "engine: cue count check disabled",
        "engine.py",
        "    if len(sub.cues) != expected_count:",
        "    if False:",
        "verify_output must report cue loss",
    ),
    Mutant(
        "preview: polygon closing point order swapped",
        "preview.py",
        '                f\'<polygon points="{first_x:.1f},{base:.1f} \'',
        '                f\'<polygon points="{base:.1f},{first_x:.1f} \'',
        "the viewport-bounds test must catch an x/y swap",
    ),
    Mutant(
        "preview: suspect cues no longer outlined",
        "preview.py",
        "        is_sus = suspect is not None and c.index in suspect",
        "        is_sus = False",
        "suspect cues must be visually marked",
    ),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_suite() -> tuple[set[str], float]:
    t0 = time.time()
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
         "-rf", "--tb=no"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    failed: set[str] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("FAILED "):
            failed.add(line.split("FAILED ", 1)[1].split(" ")[0])
    return failed, time.time() - t0


def main() -> int:
    # Optional filter so a re-run after adding tests does not cost 25 minutes.
    # Usage: python tools/mutate.py --only coverage,framerate
    only = None
    if "--only" in sys.argv:
        raw = sys.argv[sys.argv.index("--only") + 1]
        only = [t.strip().lower() for t in raw.split(",") if t.strip()]
    selected = MUTANTS
    if only:
        selected = [m for m in MUTANTS if any(t in m.name.lower() for t in only)]
        if not selected:
            print(f"no mutant matches {only}")
            return 2
        print(f"running {len(selected)} of {len(MUTANTS)} mutants")

    print("baseline run…")
    base_failed, base_time = run_suite()
    if base_failed:
        print(f"BASELINE IS NOT GREEN ({len(base_failed)} failures) — fix that first")
        for f in sorted(base_failed):
            print("   ", f)
        return 2
    print(f"baseline green in {base_time:.1f}s\n")

    survived: list[Mutant] = []
    errors: list[tuple[Mutant, str]] = []

    for i, m in enumerate(selected, 1):
        path = SRC / m.file
        original = path.read_bytes()
        digest = sha(path)
        label = f"[{i:02d}/{len(selected)}] {m.name}"
        try:
            text = original.decode("utf-8")
            if m.old not in text:
                errors.append((m, "anchor not found"))
                print(f"{label}: SKIP (anchor not found)")
                continue
            mutated = text.replace(m.old, m.new, 1)
            if mutated == text:
                errors.append((m, "replacement was a no-op"))
                print(f"{label}: SKIP (no-op mutant)")
                continue
            path.write_text(mutated, encoding="utf-8", newline="\n")

            failed, _ = run_suite()
            newly = failed - base_failed
            if newly:
                print(f"{label}: KILLED by {len(newly)} test(s)")
                print(f"        e.g. {sorted(newly)[0]}")
            else:
                survived.append(m)
                print(f"{label}: *** SURVIVED *** ({m.intent})")
        finally:
            path.write_bytes(original)
            if sha(path) != digest:
                print(f"!!! RESTORE FAILED for {m.file} — aborting")
                return 3

    print("\n" + "=" * 72)
    killed = len(selected) - len(survived) - len(errors)
    print(f"killed {killed}/{len(selected)}"
          f"   survived {len(survived)}   errors {len(errors)}")
    for m in survived:
        print(f"  SURVIVED: {m.name}\n            intent: {m.intent}")
    for m, why in errors:
        print(f"  ERROR: {m.name} — {why}")
    return 0 if not survived and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
