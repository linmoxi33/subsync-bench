"""Local web workbench.

A thin HTTP layer over the same pipeline the CLI uses — no second implementation.
The browser gets what the CLI cannot give: a file picker, a progress bar during
the slow align step, and the timeline rendered inline next to the cue table.

Run with ``subsync-bench-serve`` (see ``server.py``) or ``uvicorn`` directly.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from . import __version__, audio, engine, pipeline, preview, report, subio
from .errors import BenchError
from .workspace import Project

#: Where uploaded projects live. Kept out of the package tree so `pip install -e`
#: does not accumulate user data inside the source checkout.
WORKSPACE_ROOT = Path(tempfile.gettempdir()) / "subsync-bench-workspaces"


class OpenRequest(BaseModel):
    project: str


class AlignRequest(BaseModel):
    project: str
    label: str | None = None
    max_offset_seconds: float | None = None
    confidence_threshold: float | None = None


class SelectRequest(BaseModel):
    project: str
    revision: str


class ExportRequest(BaseModel):
    project: str
    out: str | None = None
    force: bool = False
    allow_override: bool = False
    skip_recheck: bool = False


def _err(exc: BenchError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"ok": False, "error": exc.to_dict()})


def create_app() -> FastAPI:
    app = FastAPI(title="subsync-bench", version=__version__, docs_url="/api/docs")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        import shutil as _sh

        return {
            "ok": True,
            "version": __version__,
            "ffmpeg": bool(_sh.which("ffmpeg")),
            "ffprobe": bool(_sh.which("ffprobe")),
            "engine": engine.engine_available(),
            "engine_version": engine.engine_version(),
        }

    @app.post("/api/upload")
    async def upload(
        video: UploadFile = File(...),
        subtitle: UploadFile = File(...),
        name: str = Form("project"),
    ) -> Any:
        """Create a project from an uploaded video + subtitle."""
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip() or "project"
        root = WORKSPACE_ROOT / f"{safe_name}-{_short_id()}"
        root.mkdir(parents=True, exist_ok=True)
        incoming = root / "incoming"
        incoming.mkdir(exist_ok=True)

        v_path = incoming / _safe_filename(video.filename or "video.mp4")
        s_path = incoming / _safe_filename(subtitle.filename or "subtitle.srt")
        try:
            await _save_upload(video, v_path)
            await _save_upload(subtitle, s_path)
            proj = Project.create(root / "proj", video=v_path, subtitle=s_path, name=safe_name)
        except BenchError as exc:
            shutil.rmtree(root, ignore_errors=True)
            return _err(exc)
        finally:
            shutil.rmtree(incoming, ignore_errors=True)

        insp = pipeline.inspect(proj.video_path, proj.subtitle_path)
        return {
            "ok": True,
            "project": str(proj.root),
            "inspection": insp.to_dict(),
        }

    @app.post("/api/align")
    def align(req: AlignRequest) -> Any:
        try:
            proj = Project.open(Path(req.project))
            rev, verdict = pipeline.align_project(
                proj,
                label=req.label,
                max_offset_seconds=req.max_offset_seconds,
                confidence_threshold=req.confidence_threshold,
            )
        except BenchError as exc:
            return _err(exc)
        return {"ok": rev.status != "rejected", "revision": rev.to_dict(),
                "verdict": verdict, "project": str(proj.root)}

    @app.post("/api/select")
    def select(req: SelectRequest) -> Any:
        try:
            proj = Project.open(Path(req.project))
            rev = proj.select(req.revision)
        except BenchError as exc:
            return _err(exc)
        return {"ok": True, "selected": rev.id, "status": rev.status}

    @app.get("/api/revs")
    def revs(project: str) -> Any:
        try:
            proj = Project.open(Path(project))
        except BenchError as exc:
            return _err(exc)
        return {
            "ok": True,
            "selected": proj.selected_id,
            "revisions": [r.to_dict() for r in proj.revisions],
            "integrity": proj.verify_inputs(),
        }

    @app.get("/api/timeline.svg")
    def timeline(project: str, revision: str | None = None,
                 start: float | None = None, end: float | None = None) -> Response:
        try:
            proj = Project.open(Path(project))
            rev = (
                proj.get(revision) if revision
                else proj.selected() or (proj.revisions[-1] if proj.revisions else None)
            )
            if rev is None:
                raise BenchError("no revision yet", code="E_NO_DECISION")
            before = subio.load(proj.subtitle_path).cues
            after = pipeline.load_revision_cues(proj, rev.id)
            env = audio.envelope_from_media(proj.video_path)
            residuals = audio.cue_residuals(
                env, after, 0.0, onset_lag=audio.measure_onset_lag(env, after)
            )
        except BenchError as exc:
            return _err(exc)

        spec = preview.TimelineSpec()
        if start is not None:
            spec.view_start = start
        if end is not None:
            spec.view_end = end
        svg = preview.render_timeline(
            env, before, after, residuals=residuals, spec=spec,
            title=f"{proj.manifest.get('name', 'project')} · {rev.id}",
        )
        return Response(content=svg, media_type="image/svg+xml",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/diff")
    def diff(project: str, revision: str | None = None) -> Any:
        try:
            proj = Project.open(Path(project))
            rev = (
                proj.get(revision) if revision
                else proj.selected() or (proj.revisions[-1] if proj.revisions else None)
            )
            if rev is None:
                raise BenchError("no revision yet", code="E_NO_DECISION")
            before = subio.load(proj.subtitle_path).cues
            after = pipeline.load_revision_cues(proj, rev.id)
            env = audio.envelope_from_media(proj.video_path)
            residuals = audio.cue_residuals(
                env, after, 0.0, onset_lag=audio.measure_onset_lag(env, after)
            )
        except BenchError as exc:
            return _err(exc)
        rows = preview.build_diff(before, after, residuals)
        return {"ok": True, "revision": rev.id, "rows": [r.to_dict() for r in rows]}

    @app.post("/api/export")
    def export(req: ExportRequest) -> Any:
        try:
            proj = Project.open(Path(req.project))
            info = pipeline.export_selected(
                proj,
                Path(req.out) if req.out else None,
                force=req.force,
                allow_override=req.allow_override,
                skip_recheck=req.skip_recheck,
            )
        except BenchError as exc:
            return _err(exc)
        body = Path(info["path"]).read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "export": info, "preview_text": body[:4000]}

    @app.get("/api/download")
    def download(path: str) -> Response:
        p = Path(path)
        if not p.exists() or p.suffix.lower() != ".srt":
            raise HTTPException(status_code=404, detail="not found")
        return Response(
            content=p.read_bytes(),
            media_type="application/x-subrip",
            headers={"Content-Disposition": f'attachment; filename="{p.name}"'},
        )

    @app.get("/api/report", response_class=HTMLResponse)
    def full_report(project: str) -> str:
        try:
            proj = Project.open(Path(project))
        except BenchError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
        out = proj.root / "report.html"
        report.build_report(proj, out)
        return out.read_text(encoding="utf-8")

    return app


def _short_id() -> str:
    import uuid

    return uuid.uuid4().hex[:8]


def _safe_filename(name: str) -> str:
    keep = "-_. "
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip()
    return cleaned or "file.bin"


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        while True:
            chunk = await upload.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


_INDEX_HTML = (Path(__file__).parent / "web" / "index.html").read_text(
    encoding="utf-8"
) if (Path(__file__).parent / "web" / "index.html").exists() else "<h1>UI missing</h1>"
