"""Push a local git tree to GitHub through the Git Data API.

Why not `git push`: HTTPS to github.com is blocked on this machine (connection
reset / timeout after ~20 s), while `gh api` works. So the tree is uploaded as
blobs and assembled server-side.

Constraints that shaped this (all previously observed):

* Endpoint paths must NOT start with "/" — Git Bash rewrites a leading-slash
  argument as a filesystem path ("invalid API endpoint: .../user/repos").
* A fresh repository rejects blob uploads with 409 until it has one commit, so
  the first file goes in via the Contents API.
* Request bodies go through --input files: a >30 KB base64 payload on argv hits
  Windows' WinError 206 (command line too long).
* POST /git/trees needs base_tree, otherwise the tree is replaced rather than
  extended.
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from pathlib import Path

OWNER = "linmoxi33"
REPO = "subsync-bench"
BRANCH = "main"
ROOT = Path(__file__).resolve().parent.parent

#: files that must not be uploaded even if git tracked them
SKIP_SUFFIX = {".mp4", ".mp3", ".wav", ".pyc", ".zip", ".png"}


def gh(args: list[str], *, payload: dict | None = None, raw: bool = False):
    """Call gh api, routing the body through a temp file when present.

    ``raw=True`` returns stdout verbatim — ``--jq .login`` prints a bare string,
    not JSON, and blindly json.loads-ing it raises a confusing decode error.
    """
    cmd = ["gh", "api", *args]
    tmp_name = None
    if payload is not None:
        fd, tmp_name = tempfile.mkstemp(suffix=".json")
        with open(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        cmd += ["--input", tmp_name]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"gh api failed ({proc.returncode}): {' '.join(cmd[:4])}\n"
                f"{proc.stderr[-1200:]}"
            )
        out = proc.stdout.strip()
        if raw:
            return out
        return json.loads(out) if out else {}
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


def tracked_files() -> list[Path]:
    proc = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    files = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        p = ROOT / line
        if p.suffix.lower() in SKIP_SUFFIX:
            print(f"  skip (binary/asset): {line}")
            continue
        if p.exists():
            files.append(p)
    return files


def main() -> int:
    login = gh(["user", "--jq", ".login"], raw=True)
    if login != OWNER:
        print(f"ABORT: active gh account is {login}, expected {OWNER}")
        return 2
    print(f"active account: {login}")

    # Refuse to publish a dirty tree. This is not tidiness: the mutation test in
    # tools/mutate.py edits tracked source files in place while it runs, so a push
    # started at the wrong moment would publish a deliberately broken file. A
    # clean `git status` is the only cheap proof that what is on disk is what was
    # committed.
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace").stdout.strip()
    if dirty:
        print("ABORT: working tree is not clean; commit or stash first")
        for line in dirty.splitlines()[:20]:
            print("   ", line)
        print("")
        print("If tools/mutate.py is running, wait for it to finish - it edits")
        print("tracked files in place and restores them after each mutant.")
        return 3

    files = tracked_files()
    print(f"uploading {len(files)} files")

    # 1) establish a base commit.
    #    A brand-new repository rejects blob uploads (409) until it has one
    #    commit, so seed it through the Contents API. If the branch already
    #    exists, reuse its tip instead — PUT on an existing path needs the old
    #    blob sha, and omitting it fails with HTTP 422 "sha wasn't supplied".
    try:
        ref = gh([f"repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}"])
        parent = ref["object"]["sha"]
        commit_obj = gh([f"repos/{OWNER}/{REPO}/git/commits/{parent}"])
        base_tree = commit_obj["tree"]["sha"]
        seeded_now = False
        print(f"existing branch: commit={parent[:10]} tree={base_tree[:10]}")
    except RuntimeError:
        seed = ROOT / ".gitignore"
        seed_b64 = base64.b64encode(seed.read_bytes()).decode()
        seeded = gh(["-X", "PUT", f"repos/{OWNER}/{REPO}/contents/.gitignore"],
                    payload={"message": "seed", "content": seed_b64,
                             "branch": BRANCH})
        base_tree = seeded["commit"]["tree"]["sha"]
        parent = seeded["commit"]["sha"]
        seeded_now = True
        print(f"seeded empty repo: tree={base_tree[:10]} commit={parent[:10]}")

    # 2) blobs
    tree_entries = []
    for i, p in enumerate(files, 1):
        rel = p.relative_to(ROOT).as_posix()
        if rel == ".gitignore" and seeded_now:
            continue  # already written by the seed commit above
        content = base64.b64encode(p.read_bytes()).decode()
        blob = gh(["-X", "POST", f"repos/{OWNER}/{REPO}/git/blobs"],
                  payload={"content": content, "encoding": "base64"})
        tree_entries.append({"path": rel, "mode": "100644", "type": "blob",
                             "sha": blob["sha"]})
        print(f"  [{i:02d}/{len(files)}] {rel}")

    # 3) tree (extend the seed tree, do not replace it)
    tree = gh(["-X", "POST", f"repos/{OWNER}/{REPO}/git/trees"],
              payload={"base_tree": base_tree, "tree": tree_entries})
    print(f"tree: {tree['sha'][:10]} ({len(tree.get('tree', []))} entries)")

    # 4) commit
    msg = subprocess.run(["git", "log", "-1", "--pretty=%B"], cwd=str(ROOT),
                         capture_output=True, text=True, encoding="utf-8").stdout
    commit = gh(["-X", "POST", f"repos/{OWNER}/{REPO}/git/commits"],
                payload={"message": msg.strip() or "initial commit",
                         "tree": tree["sha"], "parents": [parent]})
    print(f"commit: {commit['sha'][:10]}")

    # 5) move the branch
    gh(["-X", "PATCH", f"repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}"],
       payload={"sha": commit["sha"], "force": True})
    print(f"branch {BRANCH} -> {commit['sha'][:10]}")
    print(f"https://github.com/{OWNER}/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
