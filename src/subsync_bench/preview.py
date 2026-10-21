"""Preview: turn measurements into something a human can look at.

ffsubsync prints three numbers. You cannot tell from `offset seconds: -7.450`
whether the subtitle landed on dialogue or on a car crash. This module renders
the evidence:

* a **timeline** SVG overlaying the speech envelope, the cue track before the
  shift, and the cue track after — the visual answer to "did it land?"
* a **cue diff table** listing per-cue before/after times plus the residual, so a
  misaligned cue can be located by number and fixed by hand.

Everything is generated as a self-contained SVG string: no JS, no external
assets, renders in any browser and embeds in an HTML report.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

from . import audio
from .subio import Cue, format_timecode

# Palette: warm editorial, matching the project's house style. No cold blue.
BG = "#14110f"
PANEL = "#1d1916"
INK = "#f2e9dc"
MUTED = "#a2937f"
AMBER = "#e8a33d"
TERRA = "#c9613c"
SAGE = "#8fa876"
SAGE_LIGHT = "#b9cf9f"
GRID = "#332b25"


@dataclass
class TimelineSpec:
    width: int = 1000
    height: int = 320
    lanes: int = 3
    #: Left gutter for lane labels. Labels live here, not inside the plot, so
    #: they never collide with the envelope fill or the cue blocks.
    gutter: float = 104.0
    #: seconds of context to show; None = whole file
    view_start: float | None = None
    view_end: float | None = None


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _lane_y(i: int, spec: TimelineSpec) -> tuple[float, float]:
    pad_top = 34.0
    pad_bottom = 46.0  # room for axis + legend, which must not overlap lanes
    usable = spec.height - pad_top - pad_bottom
    lane_h = usable / spec.lanes
    y0 = pad_top + i * lane_h
    return y0, lane_h


def render_timeline(
    env: audio.Envelope | None,
    cues_before: Sequence[Cue],
    cues_after: Sequence[Cue] | None = None,
    *,
    residuals: Sequence[audio.CueResidual] | None = None,
    spec: TimelineSpec | None = None,
    title: str = "alignment preview",
) -> str:
    spec = spec or TimelineSpec()
    W, H = spec.width, spec.height

    span_candidates = [c.end for c in cues_before]
    if cues_after:
        span_candidates += [c.end for c in cues_after]
    if env is not None and env.duration:
        span_candidates.append(env.duration)
    total = max(span_candidates) if span_candidates else 1.0
    total = max(total, 1.0)

    t0 = spec.view_start if spec.view_start is not None else 0.0
    t1 = spec.view_end if spec.view_end is not None else total
    if t1 <= t0:
        t1 = t0 + 1.0
    view = t1 - t0

    x_left = spec.gutter
    x_right = W - 10.0

    def x_of(t: float) -> float:
        return x_left + (t - t0) / view * (x_right - x_left)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" '
        f'aria-label="{_esc(title)}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="{BG}" rx="10"/>')
    parts.append(
        f'<text x="12" y="18" fill="{INK}" font-size="12" '
        f'letter-spacing="0.08em">{_esc(title.upper())}</text>'
    )
    parts.append(
        f'<text x="{W - 12}" y="18" fill="{MUTED}" font-size="10" '
        f'text-anchor="end">{_esc(format_timecode(t0, sep="."))} - '
        f'{_esc(format_timecode(t1, sep="."))}</text>'
    )

    # --- time grid ------------------------------------------------------
    step = _nice_step(view)
    t = (int(t0 / step) + 1) * step
    while t < t1:
        x = x_of(t)
        parts.append(
            f'<line x1="{x:.1f}" y1="{_lane_y(0, spec)[0] - 4:.1f}" '
            f'x2="{x:.1f}" y2="{_lane_y(2, spec)[0] + _lane_y(2, spec)[1] - 6:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{H - 30:.1f}" fill="{MUTED}" font-size="9" '
            f'text-anchor="middle">{int(t // 60)}:{int(t % 60):02d}</text>'
        )
        t += step

    # --- lane 1: speech envelope ---------------------------------------
    y0, lh = _lane_y(0, spec)
    parts.append(
        f'<text x="12" y="{y0 + 9:.1f}" fill="{MUTED}" font-size="9" '
        f'letter-spacing="0.06em">SPEECH</text>'
    )
    if env is not None and len(env.values):
        base = y0 + lh - 6
        max_h = lh - 16
        i0 = max(0, int(t0 / env.duration * len(env.values))) if env.duration else 0
        i1 = min(len(env.values), int(t1 / env.duration * len(env.values))) if env.duration else len(env.values)
        # Draw as a filled area, not per-frame bars. Per-frame bars turn a 30 ms
        # envelope into a comb at any realistic width: the eye reads stripes
        # instead of speech. A single polygon reads as loudness.
        pts: list[str] = []
        step_px = max(1, (i1 - i0) // int(W - 8))
        for k in range(i0, i1, step_px):
            chunk = env.values[k : k + step_px]
            v = float(chunk.max()) if len(chunk) else 0.0
            if v < 0.03:
                v = 0.0
            x = x_of(k * env.hop / env.sample_rate)
            pts.append(f"{x:.1f},{base - v * max_h:.1f}")
        if pts:
            # A closed polygon: start at the baseline under the first sample,
            # trace the envelope, then return along the baseline. Closing points
            # must be written as "x,y" like every other point — getting the order
            # wrong here silently stretches the fill across the whole canvas.
            first_x = x_of(max(i0 * env.hop / env.sample_rate, t0))
            last_x = x_of(min(i1 * env.hop / env.sample_rate, t1))
            parts.append(
                f'<polygon points="{first_x:.1f},{base:.1f} '
                + " ".join(pts)
                + f' {last_x:.1f},{base:.1f}" '
                f'fill="{SAGE}" fill-opacity="0.55" stroke="{SAGE_LIGHT}" '
                f'stroke-width="1.1" stroke-linejoin="round"/>'
            )
        parts.append(
            f'<line x1="{x_left:.1f}" y1="{base:.1f}" x2="{x_right:.1f}" '
            f'y2="{base:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
    else:
        parts.append(
            f'<text x="12" y="{y0 + lh / 2:.1f}" fill="{TERRA}" font-size="10">'
            f'no envelope available</text>'
        )

    # --- lane 2: cues before -------------------------------------------
    y0, lh = _lane_y(1, spec)
    parts.append(
        f'<text x="12" y="{y0 + 9:.1f}" fill="{TERRA}" font-size="9" '
        f'letter-spacing="0.06em">INPUT</text>'
    )
    _draw_cues(parts, cues_before, x_of, y0, lh, TERRA, t0, t1)

    # --- lane 3: cues after --------------------------------------------
    y0, lh = _lane_y(2, spec)
    label = "ALIGNED" if cues_after else "ALIGNED · none"
    parts.append(
        f'<text x="12" y="{y0 + 9:.1f}" fill="{AMBER}" font-size="9" '
        f'letter-spacing="0.06em">{label}</text>'
    )
    suspect_ids = set()
    if residuals:
        suspect_ids = {r.index for r in residuals if r.suspect}

    # Displacement connectors: without them the two cue lanes are just two rows
    # of blocks and the reader cannot tell what actually moved, or by how much.
    # Note both endpoints are computed from _lane_y() in SVG user space; the
    # lanes are laid out top-down, so lane 1's bottom is above lane 2's top.
    if cues_after:
        after_by_idx = {c.index: c for c in cues_after}
        y1_before, h1_before = _lane_y(1, spec)
        y_before = y1_before + h1_before - 6
        y_after = y0 + 12
        for c in cues_before:
            a = after_by_idx.get(c.index)
            if a is None:
                continue
            xb, xa = x_of(c.start), x_of(a.start)
            if abs(xa - xb) < 0.6:
                continue
            col = TERRA if c.index in suspect_ids else AMBER
            parts.append(
                f'<line x1="{xb:.1f}" y1="{y_before:.1f}" x2="{xa:.1f}" '
                f'y2="{y_after:.1f}" stroke="{col}" stroke-width="1" '
                f'stroke-opacity="0.55" stroke-dasharray="2 2"/>'
            )

    if cues_after:
        _draw_cues(parts, cues_after, x_of, y0, lh, AMBER, t0, t1, suspect=suspect_ids)
    else:
        parts.append(
            f'<line x1="{x_left:.1f}" y1="{y0 + lh / 2:.1f}" x2="{x_right:.1f}" '
            f'y2="{y0 + lh / 2:.1f}" stroke="{GRID}" stroke-width="1" '
            f'stroke-dasharray="3 4"/>'
        )

    # --- legend ---------------------------------------------------------
    lx = 12.0
    ly = H - 16.0
    for color, text in ((SAGE, "speech"), (TERRA, "before"), (AMBER, "after")):
        parts.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="8" height="8" fill="{color}" rx="1"/>')
        parts.append(f'<text x="{lx + 12:.1f}" y="{ly + 7.5:.1f}" fill="{MUTED}" font-size="9">{text}</text>')
        lx += 74
    if suspect_ids:
        parts.append(
            f'<text x="{W - 12}" y="{ly + 7.5:.1f}" fill="{TERRA}" font-size="9" '
            f'text-anchor="end">{len(suspect_ids)} suspect cue(s) outlined</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _draw_cues(
    parts: list[str],
    cues: Sequence[Cue],
    x_of,
    y0: float,
    lh: float,
    color: str,
    t0: float,
    t1: float,
    *,
    suspect: set | None = None,
) -> None:
    bar_y = y0 + 16
    bar_h = max(6.0, lh - 26)
    for c in cues:
        if c.end < t0 or c.start > t1:
            continue
        x1 = x_of(max(c.start, t0))
        x2 = x_of(min(c.end, t1))
        w = max(1.5, x2 - x1)
        is_sus = suspect is not None and c.index in suspect
        stroke = f' stroke="{TERRA}" stroke-width="1.6"' if is_sus else ""
        parts.append(
            f'<rect x="{x1:.1f}" y="{bar_y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" '
            f'fill="{color}" opacity="{0.9 if not is_sus else 0.55}" rx="1.5"{stroke}/>'
        )


def _nice_step(view: float) -> float:
    for cand in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600):
        if view / cand <= 14:
            return float(cand)
    return 3600.0


# --- cue-level diff ------------------------------------------------------


@dataclass
class CueDiffRow:
    index: int
    start_before: float
    end_before: float
    start_after: float | None
    end_after: float | None
    residual: float | None
    coverage: float | None
    suspect: bool
    text: str

    @property
    def delta(self) -> float | None:
        if self.start_after is None:
            return None
        return self.start_after - self.start_before

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "before": [round(self.start_before, 3), round(self.end_before, 3)],
            "after": (
                [round(self.start_after, 3), round(self.end_after, 3)]
                if self.start_after is not None
                else None
            ),
            "delta": round(self.delta, 3) if self.delta is not None else None,
            "residual": round(self.residual, 3) if self.residual is not None else None,
            "coverage": round(self.coverage, 3) if self.coverage is not None else None,
            "suspect": bool(self.suspect),
            "text": self.text,
        }


def build_diff(
    cues_before: Sequence[Cue],
    cues_after: Sequence[Cue] | None,
    residuals: Sequence[audio.CueResidual] | None = None,
) -> list[CueDiffRow]:
    after_by_idx = {c.index: c for c in cues_after} if cues_after else {}
    res_by_idx = {r.index: r for r in residuals} if residuals else {}
    rows: list[CueDiffRow] = []
    for c in cues_before:
        a = after_by_idx.get(c.index)
        r = res_by_idx.get(c.index)
        rows.append(
            CueDiffRow(
                index=c.index,
                start_before=c.start,
                end_before=c.end,
                start_after=a.start if a else None,
                end_after=a.end if a else None,
                residual=r.onset_distance if r else None,
                coverage=r.coverage if r else None,
                suspect=r.suspect if r else False,
                text=c.text,
            )
        )
    return rows


def render_diff_table(rows: Sequence[CueDiffRow], *, limit: int = 200) -> str:
    """Self-contained HTML table of per-cue before/after."""
    head = (
        "<table class='diff'><thead><tr>"
        "<th>#</th><th>before</th><th>after</th><th>Δ</th>"
        "<th>onset Δ</th><th>cover</th><th>text</th>"
        "</tr></thead><tbody>"
    )
    body: list[str] = []
    for r in rows[:limit]:
        cls = " class='suspect'" if r.suspect else ""
        after = (
            f"{format_timecode(r.start_after, sep='.')[:11]}"
            if r.start_after is not None
            else "—"
        )
        delta = f"{r.delta:+.3f}s" if r.delta is not None else "—"
        resid = f"{r.residual:+.3f}s" if r.residual is not None else "—"
        conf = f"{r.coverage:.2f}" if r.coverage is not None else "—"
        body.append(
            f"<tr{cls}><td>{r.index}</td>"
            f"<td>{format_timecode(r.start_before, sep='.')[:11]}</td>"
            f"<td>{after}</td><td>{delta}</td><td>{resid}</td><td>{conf}</td>"
            f"<td class='txt'>{_esc(r.text[:80])}</td></tr>"
        )
    more = ""
    if len(rows) > limit:
        more = f"<tr><td colspan='7' class='more'>… {len(rows) - limit} more cues</td></tr>"
    return head + "".join(body) + more + "</tbody></table>"
