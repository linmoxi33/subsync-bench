"""Verify that what is on GitHub is byte-identical to what is on disk.

Blob SHAs are not sufficient evidence: `.gitattributes` with `eol=lf` normalises
content on checkout, so a "matching" blob can still download differently. This
downloads every file from the API and compares sha256 against the local file,
reporting line-ending-only differences separately instead of lumping them in.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

OWNER = "linmoxi33"
REPO = "subsync-bench"
BRANCH = "main"
ROOT = Path(__file__).resolve().parent.parent


def gh(args: list[str]) -> bytes:
    proc = subprocess.run(["gh", "api", *args], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-800:])
    return proc.stdout


def remote_tree() -> dict[str, str]:
    """path -> blob sha, recursively."""
    data = json.loads(
        gh([f"repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"])
    )
    if data.get("truncated"):
        print("WARNING: tree listing truncated")
    return {
        e["path"]: e["sha"]
        for e in data.get("tree", [])
        if e["type"] == "blob"
    }


def blob_bytes(sha: str) -> bytes:
    return gh([f"repos/{OWNER}/{REPO}/git/blobs/{sha}"])


def main() -> int:
    login = subprocess.run(["gh", "api", "user", "--jq", ".login"],
                           capture_output=True, text=True).stdout.strip()
    if login != OWNER:
        print(f"ABORT: active account {login} != {OWNER}")
        return 2

    remote = remote_tree()
    # The authoritative set is what git tracks, not what is on disk: the working
    # tree also holds build artefacts, caches and the fixtures, none of which are
    # meant to be published. Comparing rglob against the remote produced 26
    # spurious "missing" entries.
    tracked = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True,
                             text=True, encoding="utf-8", errors="replace").stdout
    local_files = [ROOT / line for line in tracked.splitlines() if line.strip()]
    local_files = [p for p in local_files if p.exists()]

    missing, differ, eol_only, ok = [], [], [], 0
    for p in sorted(local_files):
        rel = p.relative_to(ROOT).as_posix()
        if rel not in remote:
            missing.append(rel)
            continue
        local = p.read_bytes()
        got = blob_bytes(remote[rel])
        # the blobs API returns base64 JSON
        try:
            payload = json.loads(got)
            if payload.get("encoding") == "base64":
                import base64

                got = base64.b64decode(payload["content"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        if hashlib.sha256(local).hexdigest() == hashlib.sha256(got).hexdigest():
            ok += 1
        elif local.replace(b"\r\n", b"\n") == got.replace(b"\r\n", b"\n"):
            eol_only.append(rel)
        else:
            differ.append((rel, len(local), len(got)))

    extra = sorted(set(remote) - {p.relative_to(ROOT).as_posix() for p in local_files})

    print(f"identical      : {ok}/{len(local_files)}")
    if eol_only:
        print(f"line-endings   : {len(eol_only)} (content equal after LF normalisation)")
        for r in eol_only:
            print(f"    {r}")
    if differ:
        print(f"CONTENT DIFFERS: {len(differ)}")
        for rel, a, b in differ:
            print(f"    {rel}  local={a}B remote={b}B")
    if missing:
        print(f"MISSING REMOTE : {len(missing)}")
        for r in missing:
            print(f"    {r}")
    if extra:
        print(f"EXTRA REMOTE   : {len(extra)}")
        for r in extra:
            print(f"    {r}")

    failed = bool(differ or missing)
    print("\nRESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
