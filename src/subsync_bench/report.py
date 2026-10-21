"""Self-contained HTML report.

One file, no external assets, no JS required. Openable from disk, attachable to a
ticket, archivable next to the subtitle. That constraint is deliberate: the
workbench's output has to survive leaving the workbench.
"""

from __future__ import annotations

from pathlib import Path

from . import audio, preview, subio
from .workspace import Project

_CSS = """
:root{
  --bg:#14110f; --panel:#1d1916; --panel2:#252019; --ink:#f2e9dc;
  --muted:#a2937f; --amber:#e8a33d; --terra:#c9613c; --sage:#8fa876;
  --line:#332b25;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1060px;margin:0 auto;padding:44px 24px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:30px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-0.01em;font-weight:650}
.sub{color:var(--muted);font-size:13px;letter-spacing:0.04em;text-transform:uppercase}
h2{font-size:13px;letter-spacing:0.14em;text-transform:uppercase;color:var(--muted);
  margin:0 0 16px;font-weight:600}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:22px 24px;margin-bottom:20px}
svg{display:block;width:100%;height:auto}
dl.kv{display:grid;grid-template-columns:190px 1fr;gap:8px 18px;margin:0;font-size:14px}
dl.kv dt{color:var(--muted)}
dl.kv dd{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.st-ok{color:var(--sage)} .st-suspect{color:var(--amber)} .st-rejected{color:var(--terra)}
table.diff{width:100%;border-collapse:collapse;font-size:12.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
table.diff th{text-align:left;color:var(--muted);font-weight:500;padding:7px 8px;
  border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel);
  font-size:11px;letter-spacing:0.08em;text-transform:uppercase}
table.diff td{padding:6px 8px;border-bottom:1px solid rgba(51,43,37,.55);vertical-align:top}
table.diff tr.suspect td{background:rgba(201,97,60,.10)}
table.diff tr.suspect td:nth-child(5){color:var(--terra);font-weight:600}
td.txt{color:var(--muted);max-width:380px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font-family:inherit}
td.more{color:var(--muted);text-align:center;padding:12px}
.findings{list-style:none;padding:0;margin:0}
.findings li{padding:9px 12px;border-left:2px solid var(--line);margin-bottom:8px;
  background:var(--panel2);border-radius:0 6px 6px 0;font-size:13.5px}
.findings li.block{border-left-color:var(--terra)}
.findings li.warn{border-left-color:var(--amber)}
.findings li.info{border-left-color:var(--muted)}
.findings code{color:var(--amber);font-size:11.5px;letter-spacing:0.04em}
.note{color:var(--muted);font-size:12.5px;margin:14px 0 0}
footer{color:var(--muted);font-size:12px;border-top:1px solid var(--line);
  padding-top:16px;margin-top:34px}
.hero{display:flex;gap:26px;flex-wrap:wrap}
.metric{background:var(--panel2);border-radius:8px;padding:14px 18px;min-width:132px}
.metric .n{font-size:22px;font-family:ui-monospace,monospace;letter-spacing:-0.02em}
.metric .l{font-size:11px;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;
  margin-top:3px}
"""


def page(*, title: str, body: str, subtitle: str = "") -> str:
    sub = f'<div class="sub">{_esc(subtitle)}</div>' if subtitle else ""
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header><h1>{_esc(title)}</h1>{sub}</header>
{body}
</div></body></html>
"""


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_report(project: Project, out: Path | None = None) -> Path:
    from . import gate as gate_mod
    from .workspace import _now

    out = Path(out) if out else project.root / "report.html"
    before = subio.load(project.subtitle_path).cues
    revs = project.revisions
    selected = project.selected()

    env: audio.Envelope | None = None
    try:
        env = audio.envelope_from_media(project.video_path)
    except Exception:
        env = None

    integrity = project.verify_inputs()

    sections: list[str] = []

    # --- hero ------------------------------------------------------------
    hero_metrics = [
        (str(len(before)), "input cues"),
        (str(len(revs)), "revisions"),
        (selected.id if selected else "—", "selected"),
        (
            f"{selected.confidence:.2f}" if selected else "—",
            "confidence",
        ),
    ]
    hero = "".join(
        f'<div class="metric"><div class="n">{_esc(value)}</div>'
        f'<div class="l">{_esc(label)}</div></div>'
        for value, label in hero_metrics
    )
    sections.append(f'<section class="card"><div class="hero">{hero}</div></section>')

    # --- inputs ----------------------------------------------------------
    v = project.manifest["video"]
    s = project.manifest["subtitle"]
    integrity_note = ""
    if integrity["stale"]:
        integrity_note = (
            f"<p class='note'>stale revisions (inputs changed): "
            f"{_esc(', '.join(integrity['stale']))}</p>"
        )
    sections.append(
        '<section class="card"><h2>inputs</h2><dl class="kv">'
        f"<dt>project</dt><dd>{_esc(project.root)}</dd>"
        f"<dt>video</dt><dd>{_esc(v['path'])}</dd>"
        f"<dt>video sha256</dt><dd>{_esc(v['sha256'][:24])}…</dd>"
        f"<dt>subtitle</dt><dd>{_esc(s['path'])}</dd>"
        f"<dt>subtitle sha256</dt><dd>{_esc(s['sha256'][:24])}…</dd>"
        + (f"<dt>speech ratio</dt><dd>{env.speech_ratio:.3f}</dd>" if env else "")
        + "</dl>" + integrity_note + "</section>"
    )

    # --- revisions table -------------------------------------------------
    if revs:
        rows = []
        for r in revs:
            mark = " ●" if selected and r.id == selected.id else ""
            rows.append(
                f"<tr><td>{_esc(r.id)}{mark}</td>"
                f"<td class='st-{r.status}'>{_esc(r.status)}</td>"
                f"<td>{r.offset:+.3f}s</td>"
                f"<td>{r.framerate_scale:.5f}</td>"
                f"<td>{r.confidence:.3f}</td>"
                f"<td>{r.residual_summary.get('suspect_count', 0)}</td>"
                f"<td class='txt'>{_esc(r.label)}</td></tr>"
            )
        sections.append(
            '<section class="card"><h2>revisions</h2><table class="diff"><thead><tr>'
            "<th>id</th><th>status</th><th>offset</th><th>scale</th>"
            "<th>conf</th><th>suspect</th><th>label</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></section>"
        )

    # --- selected detail -------------------------------------------------
    if selected:
        after = pipeline_cues(project, selected.id)
        residuals = (
            audio.cue_residuals(env, after, 0.0,
                                onset_lag=audio.measure_onset_lag(env, after))
            if env else None
        )
        svg = preview.render_timeline(
            env, before, after, residuals=residuals,
            title=f"{project.manifest.get('name', 'project')} · {selected.id}",
        )
        verdict = gate_mod.evaluate(
            confidence=selected.confidence,
            null_confidence=selected.null_confidence,
            offset=selected.offset,
            framerate_scale=selected.framerate_scale,
            speech_ratio=selected.speech_ratio,
            cue_count_in=len(before),
            cue_count_out=selected.cue_count,
            residual_summary=selected.residual_summary,
            max_offset_seconds=float(project.settings.get("max_offset_seconds", 120.0)),
            confidence_threshold=float(
                project.settings.get("confidence_threshold", 0.55)
            ),
        )
        findings = "".join(
            f"<li class='{f.severity}'><code>{_esc(f.code)}</code> — "
            f"{_esc(f.message)}</li>"
            for f in verdict.findings
        )
        sections.append(
            f'<section class="card"><h2>timeline · {_esc(selected.id)}</h2>{svg}</section>'
        )
        sections.append(
            f'<section class="card"><h2>trust gate</h2>'
            f"<ul class='findings'>{findings or '<li>no findings</li>'}</ul></section>"
        )
        rows = preview.build_diff(before, after, residuals)
        sections.append(
            '<section class="card"><h2>per-cue diff</h2>'
            + preview.render_diff_table(rows)
            + "</section>"
        )

    body = "\n".join(sections)
    doc = page(
        title=project.manifest.get("name", "subtitle project"),
        body=body,
        subtitle=f"subtitle alignment report · generated {_now()}",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8", newline="\n")
    return out


def pipeline_cues(project: Project, rev_id: str):
    from . import pipeline

    return pipeline.load_revision_cues(project, rev_id)
