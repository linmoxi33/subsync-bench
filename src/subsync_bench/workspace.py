"""Project workspace: the spine that makes the steps cooperate.

A one-step tool is stateless — you run it, you get a file, you forget. A workbench
needs memory, because the expensive facts (which video, which subtitle, which
shift was applied, whether that shift was trustworthy) must survive across
sessions and be re-checkable.

Layout::

    <project>/
        project.json          manifest: inputs, revisions, selection, settings
        input/                original video + subtitle, never modified
        work/                 staged engine output, envelopes cache
        out/                  exported subtitles
        receipts/             one JSON per revision: full measurements

Key invariants:

* **Inputs are immutable.** Originals are copied in and hashed; the subtitle you
  started from is always recoverable, so any experiment is safe to run.
* **Every revision is reproducible from its receipt.** The receipt records input
  digests, engine version, parameters and measurements. If an input's digest no
  longer matches, the revision is marked *stale* rather than silently reused.
* **Selection is explicit.** A project has at most one selected revision; export
  exports the selection. Nothing is exported implicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import (
    E_PROJECT_CORRUPT,
    E_PROJECT_EXISTS,
    E_PROJECT_MISSING,
    E_RECEIPT_MISSING,
    E_SUB_MISSING,
    E_VIDEO_MISSING,
    WorkspaceError,
)

MANIFEST_NAME = "project.json"
SCHEMA_VERSION = 1


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Revision:
    """One alignment attempt, fully measured and independently re-verifiable."""

    id: str
    created: str
    label: str
    #: offset applied to cues, in seconds (after framerate scaling)
    offset: float
    framerate_scale: float
    #: our normalised confidence, not upstream's raw score
    confidence: float
    null_confidence: float
    speech_ratio: float
    cue_count: int
    #: How far the aligned output still wants to move (should be ~0).
    residual_offset: float = 0.0
    residual_summary: dict[str, Any] = field(default_factory=dict)
    engine: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    #: "ok" | "suspect" | "rejected"
    status: str = "ok"
    notes: list[str] = field(default_factory=list)
    #: relative path to the aligned subtitle inside the project
    artifact: str | None = None
    rejected_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created": self.created,
            "label": self.label,
            "offset": round(self.offset, 3),
            "framerate_scale": round(self.framerate_scale, 6),
            "confidence": round(self.confidence, 4),
            "null_confidence": round(self.null_confidence, 4),
            "speech_ratio": round(self.speech_ratio, 4),
            "cue_count": self.cue_count,
            "residual_offset": round(self.residual_offset, 3),
            "residual_summary": self.residual_summary,
            "engine": self.engine,
            "inputs": self.inputs,
            "status": self.status,
            "notes": list(self.notes),
            "artifact": self.artifact,
            "rejected_reason": self.rejected_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Revision:
        return cls(
            id=d["id"],
            created=d.get("created", ""),
            label=d.get("label", ""),
            offset=float(d.get("offset", 0.0)),
            framerate_scale=float(d.get("framerate_scale", 1.0)),
            confidence=float(d.get("confidence", 0.0)),
            null_confidence=float(d.get("null_confidence", 0.0)),
            speech_ratio=float(d.get("speech_ratio", 0.0)),
            cue_count=int(d.get("cue_count", 0)),
            residual_offset=float(d.get("residual_offset", 0.0)),
            residual_summary=d.get("residual_summary", {}),
            engine=d.get("engine", {}),
            inputs=d.get("inputs", {}),
            status=d.get("status", "ok"),
            notes=list(d.get("notes", [])),
            artifact=d.get("artifact"),
            rejected_reason=d.get("rejected_reason"),
        )


@dataclass
class Project:
    root: Path
    manifest: dict[str, Any]

    # ---- construction ---------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def receipt_dir(self) -> Path:
        return self.root / "receipts"

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        video: Path,
        subtitle: Path,
        name: str | None = None,
    ) -> Project:
        root = Path(root)
        if (root / MANIFEST_NAME).exists():
            raise WorkspaceError(
                f"project already exists at {root}",
                code=E_PROJECT_EXISTS,
                hint="pass --force to reuse, or pick another directory",
            )
        video = Path(video)
        subtitle = Path(subtitle)
        if not video.exists():
            raise WorkspaceError(f"video not found: {video}", code=E_VIDEO_MISSING)
        if not subtitle.exists():
            raise WorkspaceError(
                f"subtitle not found: {subtitle}", code=E_SUB_MISSING
            )

        for sub in ("input", "work", "out", "receipts"):
            (root / sub).mkdir(parents=True, exist_ok=True)

        v_dst = root / "input" / video.name
        s_dst = root / "input" / subtitle.name
        shutil.copy2(video, v_dst)
        shutil.copy2(subtitle, s_dst)

        manifest = {
            "schema": SCHEMA_VERSION,
            "name": name or root.name,
            "created": _now(),
            "video": {
                "path": f"input/{video.name}",
                "sha256": sha256_file(v_dst),
                "bytes": v_dst.stat().st_size,
            },
            "subtitle": {
                "path": f"input/{subtitle.name}",
                "sha256": sha256_file(s_dst),
                "bytes": s_dst.stat().st_size,
            },
            "revisions": [],
            "selected": None,
            "settings": {
                "max_offset_seconds": 120.0,
                "confidence_threshold": 0.55,
                "residual_threshold": 0.5,
                "residual_tolerance": 0.8,
                "output_encoding": "utf-8",
            },
        }
        proj = cls(root=root, manifest=manifest)
        proj.save()
        return proj

    # ---- persistence ----------------------------------------------------
    @classmethod
    def open(cls, root: Path) -> Project:
        root = Path(root)
        mp = root / MANIFEST_NAME
        if not mp.exists():
            raise WorkspaceError(
                f"no project at {root}", code=E_PROJECT_MISSING
            )
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise WorkspaceError(
                f"{MANIFEST_NAME} is not valid JSON: {exc}",
                code=E_PROJECT_CORRUPT,
            ) from exc
        if not isinstance(manifest, dict) or "schema" not in manifest:
            raise WorkspaceError(
                f"{MANIFEST_NAME} missing schema field", code=E_PROJECT_CORRUPT
            )
        if manifest["schema"] > SCHEMA_VERSION:
            raise WorkspaceError(
                f"project schema {manifest['schema']} is newer than supported "
                f"{SCHEMA_VERSION}",
                code=E_PROJECT_CORRUPT,
            )
        return cls(root=root, manifest=manifest)

    def replace_inputs(self, *, video: Path, subtitle: Path) -> dict[str, Any]:
        """Swap the project's inputs, preserving revision history.

        Existing revisions are kept but their recorded input digests no longer
        match, so ``verify_inputs`` reports them stale and export refuses them
        unless forced. Keeping them is deliberate: "I tried the other release and
        it scored worse" is information worth retaining.
        """
        video = Path(video)
        subtitle = Path(subtitle)
        if not video.exists():
            raise WorkspaceError(f"video not found: {video}", code=E_VIDEO_MISSING)
        if not subtitle.exists():
            raise WorkspaceError(
                f"subtitle not found: {subtitle}", code=E_SUB_MISSING
            )
        old = {
            "video": self.manifest["video"]["path"],
            "subtitle": self.manifest["subtitle"]["path"],
        }
        for src, key in ((video, "video"), (subtitle, "subtitle")):
            dst = self.input_dir / src.name
            # Resolve first: on Windows, copying a file onto itself raises
            # PermissionError (WinError 32), which would surface as a confusing
            # failure when the user simply re-points at the project's own input.
            try:
                if dst.exists() and dst.resolve() == src.resolve():
                    self.manifest[key] = {
                        "path": f"input/{dst.name}",
                        "sha256": sha256_file(dst),
                        "bytes": dst.stat().st_size,
                    }
                    continue
            except OSError:
                pass
            if dst.exists() and sha256_file(dst) != sha256_file(src):
                # Do not clobber a previous input of the same name.
                dst = self.input_dir / f"{src.stem}__{sha256_file(src)[:8]}{src.suffix}"
            shutil.copy2(src, dst)
            self.manifest[key] = {
                "path": f"input/{dst.name}",
                "sha256": sha256_file(dst),
                "bytes": dst.stat().st_size,
            }
        self.save()
        return {"replaced": old, "with": {
            "video": self.manifest["video"]["path"],
            "subtitle": self.manifest["subtitle"]["path"],
        }}

    def save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, self.manifest_path)

    # ---- accessors ------------------------------------------------------
    @property
    def video_path(self) -> Path:
        return self.root / self.manifest["video"]["path"]

    @property
    def subtitle_path(self) -> Path:
        return self.root / self.manifest["subtitle"]["path"]

    @property
    def settings(self) -> dict[str, Any]:
        return self.manifest.setdefault("settings", {})

    @property
    def revisions(self) -> list[Revision]:
        return [Revision.from_dict(d) for d in self.manifest.get("revisions", [])]

    @property
    def selected_id(self) -> str | None:
        return self.manifest.get("selected")

    def get(self, rev_id: str) -> Revision:
        for r in self.revisions:
            if r.id == rev_id:
                return r
        raise WorkspaceError(
            f"revision {rev_id} not found",
            code=E_RECEIPT_MISSING,
            context={"available": [r.id for r in self.revisions]},
        )

    def selected(self) -> Revision | None:
        sid = self.selected_id
        if not sid:
            return None
        try:
            return self.get(sid)
        except WorkspaceError:
            return None

    # ---- input integrity ------------------------------------------------
    def verify_inputs(self) -> dict[str, Any]:
        """Re-hash inputs from disk and mark dependent revisions stale.

        Deliberately compares revisions against the *disk* digest, not against
        the manifest's recorded digest. Comparing revision-vs-manifest only
        detects "the manifest was updated"; it misses the far more common case
        of a file edited in place, where the manifest still holds the old hash
        and every revision silently looks fresh.
        """
        report: dict[str, Any] = {"video": {}, "subtitle": {}, "stale": []}
        disk: dict[str, str | None] = {}
        for key, path in (("video", self.video_path), ("subtitle", self.subtitle_path)):
            rec = self.manifest[key]
            if not path.exists():
                report[key] = {"present": False, "matches": False}
                disk[key] = None
                continue
            digest = sha256_file(path)
            disk[key] = digest
            report[key] = {
                "present": True,
                "matches": digest == rec["sha256"],
                "sha256": digest,
            }
        for r in self.revisions:
            for key in ("video", "subtitle"):
                recorded = r.inputs.get(key)
                if not recorded:
                    continue
                if disk.get(key) != recorded:
                    report["stale"].append(r.id)
                    break
        return report

    # ---- revisions ------------------------------------------------------
    def next_revision_id(self) -> str:
        n = len(self.manifest.get("revisions", [])) + 1
        while True:
            rid = f"r{n:03d}"
            if all(d["id"] != rid for d in self.manifest.get("revisions", [])):
                return rid
            n += 1

    def add_revision(self, rev: Revision) -> Revision:
        self.manifest.setdefault("revisions", []).append(rev.to_dict())
        receipt = self.receipt_dir / f"{rev.id}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps(rev.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.save()
        return rev

    def select(self, rev_id: str) -> Revision:
        rev = self.get(rev_id)
        self.manifest["selected"] = rev_id
        self.save()
        return rev

    def revision_artifact(self, rev: Revision) -> Path:
        if not rev.artifact:
            raise WorkspaceError(
                f"revision {rev.id} has no artifact",
                code=E_RECEIPT_MISSING,
            )
        p = self.root / rev.artifact
        if not p.exists():
            raise WorkspaceError(
                f"revision {rev.id} artifact missing on disk: {rev.artifact}",
                code=E_RECEIPT_MISSING,
            )
        return p

    # ---- summary --------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        revs = self.revisions
        sel = self.selected()
        return {
            "root": str(self.root),
            "name": self.manifest.get("name"),
            "created": self.manifest.get("created"),
            "video": self.manifest["video"]["path"],
            "subtitle": self.manifest["subtitle"]["path"],
            "revision_count": len(revs),
            "selected": sel.to_dict() if sel else None,
            "settings": dict(self.settings),
            "revisions": [r.to_dict() for r in revs],
        }
