"""Verify a published release end to end.

    python tools/verify_release.py v0.1.0

1. download every release asset (and fail loudly if none arrive — an empty
   directory once made a hash check "pass" against zero files)
2. compare sha256 against the local dist/ artefacts
3. install the downloaded wheel into a clean venv
4. run the real align -> select -> export chain against the fixtures
5. print the offsets so a silently-wrong install cannot look like a pass

Exit codes: 0 ok, 1 verification failed, 2 usage/environment problem.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = "linmoxi33"
REPO = "subsync-bench"


def gh(args: list[str]) -> bytes:
    proc = subprocess.run(["gh", "api", *args], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-600:])
    return proc.stdout


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    return ok


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python tools/verify_release.py <tag>")
        return 2
    tag = sys.argv[1]

    login = subprocess.run(["gh", "api", "user", "--jq", ".login"],
                           capture_output=True, text=True).stdout.strip()
    if login != OWNER:
        print(f"ABORT: active gh account is {login}, expected {OWNER}")
        return 2

    release = json.loads(gh([f"repos/{OWNER}/{REPO}/releases/tags/{tag}"]))
    assets = release.get("assets", [])
    if not assets:
        print(f"FAIL: release {tag} has no assets")
        return 1
    print(f"release {tag}: {len(assets)} asset(s)")

    work = Path(tempfile.mkdtemp(prefix="subsync-relcheck-"))
    ok = True
    try:
        # 1) download
        for a in assets:
            data = gh(["-H", "Accept: application/octet-stream",
                       f"repos/{OWNER}/{REPO}/releases/assets/{a['id']}"])
            (work / a["name"]).write_bytes(data)
        downloaded = [p.name for p in work.iterdir() if p.is_file()]
        ok &= check(f"downloaded {len(downloaded)} file(s)",
                    len(downloaded) == len(assets), ", ".join(downloaded))

        # 2) hashes
        local = ROOT / "dist"
        if not local.exists():
            print("FAIL: no local dist/ to compare against — run `python -m build`")
            return 1
        for name in downloaded:
            local_f = local / name
            if not local_f.exists():
                ok &= check(f"{name}: local artefact present", False)
                continue
            got = hashlib.sha256((work / name).read_bytes()).hexdigest()
            want = hashlib.sha256(local_f.read_bytes()).hexdigest()
            ok &= check(f"{name} sha256", got == want,
                        "identical" if got == want else f"{got[:12]} != {want[:12]}")

        # 3) clean install
        venv_dir = work / "venv"
        venv.create(venv_dir, with_pip=True)
        py = venv_dir / "Scripts" / "python.exe"
        whl = work / "subsync_bench-0.1.0-py3-none-any.whl"
        if not whl.exists():
            print("FAIL: wheel not among the assets")
            return 1
        r = subprocess.run([str(py), "-m", "pip", "install", "-q",
                            "-i", "https://mirrors.aliyun.com/pypi/simple/",
                            str(whl)], capture_output=True, text=True)
        ok &= check("clean venv install", r.returncode == 0, r.stderr[-200:] if r.returncode else "")

        # the engine is an extra; without it align is unavailable by design
        r = subprocess.run([str(py), "-m", "pip", "install", "-q",
                            "-i", "https://mirrors.aliyun.com/pypi/simple/",
                            "ffsubsync>=0.5.1"], capture_output=True, text=True)
        ok &= check("engine installed", r.returncode == 0)

        # 4) real chain
        media = ROOT / "fixtures" / "media"
        proj = work / "proj"
        r = subprocess.run(
            [str(py), "-m", "subsync_bench.cli", "align", str(proj),
             "--video", str(media / "film.mp4"),
             "--subtitle", str(media / "late.srt"), "--label", "release check"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok &= check("align", r.returncode == 0, r.stdout[-300:] if r.returncode else "")

        r = subprocess.run(
            [str(py), "-m", "subsync_bench.cli", "select", str(proj), "r001"],
            capture_output=True, text=True)
        ok &= check("select", r.returncode == 0)

        r = subprocess.run(
            [str(py), "-m", "subsync_bench.cli", "export", str(proj)],
            capture_output=True, text=True)
        ok &= check("export", r.returncode == 0)

        # 5) the exported timing must actually be right
        sys.path.insert(0, str(ROOT / "src"))
        from subsync_bench import subio

        truth = subio.load(media / "truth.srt").cues
        out = subio.load(proj / "out" / "late.synced.srt").cues
        deltas = [o.start - t.start for t, o in zip(truth, out, strict=True)]
        worst = max(abs(d) for d in deltas)
        ok &= check(f"max timing error {worst:.3f}s < 0.5s", worst < 0.5,
                    f"{len(out)} cues, all deltas {min(deltas):+.3f}..{max(deltas):+.3f}")
        ok &= check("text preserved",
                    all(t.text == o.text for t, o in zip(truth, out, strict=True)))

        print(f"\nartifacts kept in {work}")
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if not ok:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
