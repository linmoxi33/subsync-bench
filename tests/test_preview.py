"""Preview rendering and cue diff.

SVG output is checked structurally (well-formed, no NaN, geometry inside the
viewport) rather than by string equality — string tests on generated SVG break on
every cosmetic tweak while catching none of the real bugs. The bug these catch:
a closed polygon whose points were emitted as "y,x" instead of "x,y", which
silently stretched the fill across the whole canvas.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from subsync_bench import audio, preview
from subsync_bench.subio import Cue


def _env(windows, total=40.0):
    import numpy as np

    sr, hop = audio.SAMPLE_RATE, audio.HOP
    values = np.zeros(int(total * sr / hop), dtype=np.float32)
    for a, b in windows:
        values[int(a * sr / hop) : int(b * sr / hop)] = 1.0
    return audio.Envelope(values, sample_rate=sr, hop=hop)


WINDOWS = [(2.0, 5.0), (8.0, 11.0), (14.0, 17.0), (20.0, 23.0)]


def _cues(shift=0.0):
    return [Cue(i + 1, a + shift, b + shift, f"line {i+1}") for i, (a, b) in enumerate(WINDOWS)]


def _parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def test_svg_is_well_formed():
    svg = preview.render_timeline(_env(WINDOWS), _cues(), _cues())
    root = _parse(svg)
    assert root.tag.endswith("svg")


def test_svg_has_no_nan_or_inf():
    svg = preview.render_timeline(_env(WINDOWS), _cues(), _cues())
    assert "nan" not in svg.lower()
    assert "inf" not in svg.lower().replace("info", "")


def test_svg_without_envelope_still_renders():
    svg = preview.render_timeline(None, _cues(), None)
    root = _parse(svg)
    assert root.tag.endswith("svg")


def test_svg_without_after_lane_renders():
    svg = preview.render_timeline(_env(WINDOWS), _cues(), None)
    assert "ALIGNED" in svg


def test_polygon_points_are_x_comma_y():
    """Regression: closing points were emitted as 'y,x', which stretched the fill
    across the canvas because SVG then read a huge x coordinate.

    The assertion has to be specific. An earlier version only required "at least
    two points share the max y" — trivially true, because every silent frame sits
    on the baseline (935 such points in the 40 s fixture), so an x/y swap passed.
    What distinguishes a correct polygon is that it starts at the left edge and
    ends at the right edge, both on the same baseline.
    """
    spec = preview.TimelineSpec()
    svg = preview.render_timeline(_env(WINDOWS), _cues(), _cues(), spec=spec)
    root = _parse(svg)
    ns = "{http://www.w3.org/2000/svg}"
    polys = root.findall(f".//{ns}polygon")
    assert polys, "envelope should be drawn as a filled polygon"

    for poly in polys:
        coords = [p for p in poly.get("points", "").split() if p]
        assert len(coords) >= 4
        parsed = []
        for c in coords:
            x_s, y_s = c.split(",")
            parsed.append((float(x_s), float(y_s)))

        xs = [x for x, _ in parsed]
        ys = [y for _, y in parsed]
        assert min(xs) >= 0 and max(xs) <= spec.width, (
            f"x out of viewport: {min(xs)}..{max(xs)}"
        )
        assert min(ys) >= 0 and max(ys) <= spec.height, (
            f"y out of viewport: {min(ys)}..{max(ys)}"
        )

        first_x, first_y = parsed[0]
        last_x, last_y = parsed[-1]
        assert first_x == pytest.approx(spec.gutter, abs=0.6), (
            f"polygon must start at the left gutter ({spec.gutter}), got x={first_x}"
        )
        assert last_x == pytest.approx(spec.width - 10.0, abs=0.6), (
            f"polygon must end at the right edge, got x={last_x}"
        )
        assert first_y == pytest.approx(last_y, abs=0.01), (
            f"closing points must share a baseline: {first_y} vs {last_y}"
        )
        assert first_y == pytest.approx(max(ys), abs=0.01)


def test_cue_rects_stay_inside_viewport():
    svg = preview.render_timeline(_env(WINDOWS), _cues(), _cues())
    root = _parse(svg)
    ns = "{http://www.w3.org/2000/svg}"
    for rect in root.findall(f".//{ns}rect"):
        x = float(rect.get("x", 0))
        w = float(rect.get("width", 0))
        if w <= 1 or w >= 900:  # skip background / tiny marks
            continue
        assert x >= -1 and x + w <= 1010, f"rect escapes viewport: x={x} w={w}"


def test_shifted_cues_produce_connector_lines():
    svg = preview.render_timeline(_env(WINDOWS), _cues(0.0), _cues(-8.0))
    root = _parse(svg)
    ns = "{http://www.w3.org/2000/svg}"
    dashed = [
        ln for ln in root.findall(f".//{ns}line")
        if ln.get("stroke-dasharray") == "2 2"
    ]
    assert len(dashed) >= len(WINDOWS) - 1


def test_identical_cues_produce_no_connectors():
    svg = preview.render_timeline(_env(WINDOWS), _cues(), _cues())
    root = _parse(svg)
    ns = "{http://www.w3.org/2000/svg}"
    dashed = [
        ln for ln in root.findall(f".//{ns}line")
        if ln.get("stroke-dasharray") == "2 2"
    ]
    assert dashed == []


def test_suspect_cues_are_outlined():
    """Assert the geometry, not the caption.

    An earlier version of this test only checked for the legend text
    "suspect cue(s) outlined", which survives even if no rect is ever stroked —
    a mutation that disabled the outlining entirely passed. So: count stroked
    rects and require exactly the number of suspect cues.
    """
    env = _env(WINDOWS)
    after = [
        Cue(1, 2.0, 5.0, "ok"),
        Cue(2, 8.0, 11.0, "ok"),
        Cue(3, 5.4, 7.0, "moved into gap"),   # suspect
        Cue(4, 20.0, 23.0, "ok"),
    ]
    res = audio.cue_residuals(env, after, 0.0)
    suspects = [r for r in res if r.suspect]
    assert len(suspects) == 1, "fixture must produce exactly one suspect"

    svg = preview.render_timeline(env, _cues(), after, residuals=res)
    root = _parse(svg)
    ns = "{http://www.w3.org/2000/svg}"
    stroked = [
        r for r in root.findall(f".//{ns}rect")
        if r.get("stroke") and float(r.get("stroke-width", "0")) >= 1.0
    ]
    assert len(stroked) == len(suspects), (
        f"{len(stroked)} outlined rects for {len(suspects)} suspect cues"
    )
    # and the caption must agree
    assert "suspect cue" in svg


def test_no_suspects_means_no_outlines():
    env = _env(WINDOWS)
    cues = _cues()
    res = audio.cue_residuals(env, cues, 0.0)
    assert not any(r.suspect for r in res)
    svg = preview.render_timeline(env, cues, cues, residuals=res)
    root = _parse(svg)
    ns = "{http://www.w3.org/2000/svg}"
    stroked = [
        r for r in root.findall(f".//{ns}rect")
        if r.get("stroke") and float(r.get("stroke-width", "0")) >= 1.0
    ]
    assert stroked == []


def test_view_window_clips_to_range():
    svg = preview.render_timeline(
        _env(WINDOWS), _cues(), _cues(),
        spec=preview.TimelineSpec(view_start=5.0, view_end=15.0),
    )
    assert "00:00:05" in svg
    assert "00:00:15" in svg


def test_custom_title_is_escaped():
    svg = preview.render_timeline(
        _env(WINDOWS), _cues(), None, title='a <b> & "c"'
    )
    assert "&lt;b&gt;" in svg
    assert "<b>" not in svg


def test_nice_step_scales():
    assert preview._nice_step(10) <= 10
    assert preview._nice_step(3600) <= 3600
    assert preview._nice_step(100000) > 0


# --- diff table -----------------------------------------------------------


def test_build_diff_pairs_by_index():
    before = _cues()
    after = _cues(-8.0)
    rows = preview.build_diff(before, after)
    assert len(rows) == len(before)
    assert rows[0].delta == pytest.approx(-8.0, abs=0.01)


def test_build_diff_without_after():
    rows = preview.build_diff(_cues(), None)
    assert all(r.start_after is None and r.delta is None for r in rows)


def test_diff_row_serialises():
    rows = preview.build_diff(_cues(), _cues(-8.0))
    d = rows[0].to_dict()
    assert set(d) >= {"index", "before", "after", "delta", "suspect", "text"}


def test_diff_table_html_escapes_text():
    rows = preview.build_diff([Cue(1, 1.0, 2.0, "<script>x</script>")], None)
    html = preview.render_diff_table(rows)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_diff_table_limits_rows():
    many = [Cue(i, i * 1.0, i * 1.0 + 0.5, f"t{i}") for i in range(1, 60)]
    html = preview.render_diff_table(preview.build_diff(many, None), limit=10)
    assert html.count("<tr") <= 12
    assert "more cues" in html


def test_diff_marks_suspect_rows():
    env = _env(WINDOWS)
    after = [Cue(1, 2.0, 5.0, "a"), Cue(2, 5.4, 7.0, "bad")]
    res = audio.cue_residuals(env, after, 0.0)
    html = preview.render_diff_table(preview.build_diff(after, after, res))
    assert "class='suspect'" in html
