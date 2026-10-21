"""Subtitle parsing and serialization.

These tests target the real-world mess: mixed line endings, BOM, GBK, missing
indexes, reversed timecodes, arrow variants. Each case here came from a class of
file that actually breaks subtitle tooling.
"""

from __future__ import annotations

import pytest

from subsync_bench import subio
from subsync_bench.errors import E_SUB_EMPTY, SubtitleError


def test_parses_basic_srt():
    text = "1\n00:00:01,000 --> 00:00:02,500\nhello\n\n2\n00:00:03,000 --> 00:00:04,000\nworld\n"
    cues, warnings = subio.parse_text(text)
    assert len(cues) == 2
    assert cues[0].start == pytest.approx(1.0)
    assert cues[0].end == pytest.approx(2.5)
    assert cues[0].text == "hello"
    assert not warnings


def test_multiline_cue_text_preserved():
    text = "1\n00:00:01,000 --> 00:00:02,000\nline one\nline two\n"
    cues, _ = subio.parse_text(text)
    assert cues[0].text == "line one\nline two"


def test_crlf_and_cr_normalised():
    text = "1\r\n00:00:01,000 --> 00:00:02,000\r\nhello\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,000\r\nworld\r\n"
    cues, _ = subio.parse_text(text)
    assert len(cues) == 2
    assert cues[1].text == "world"


def test_bom_stripped():
    text = "\ufeff1\n00:00:01,000 --> 00:00:02,000\nhello\n"
    cues, _ = subio.parse_text(text)
    assert len(cues) == 1
    assert not cues[0].text.startswith("\ufeff")


def test_arrow_with_spaces_variant():
    text = "1\n00:00:01,000 -- > 00:00:02,000\nhello\n"
    cues, _ = subio.parse_text(text)
    assert cues[0].start == pytest.approx(1.0)


def test_dot_millisecond_separator_accepted():
    text = "1\n00:00:01.500 --> 00:00:02.750\nhello\n"
    cues, _ = subio.parse_text(text)
    assert cues[0].start == pytest.approx(1.5)
    assert cues[0].end == pytest.approx(2.75)


def test_missing_index_gets_sequential_number():
    text = "00:00:01,000 --> 00:00:02,000\nhello\n\n00:00:03,000 --> 00:00:04,000\nworld\n"
    cues, _ = subio.parse_text(text)
    assert [c.index for c in cues] == [1, 2]


def test_reversed_timecode_is_swapped_and_reported():
    text = "1\n00:00:05,000 --> 00:00:02,000\nbackwards\n"
    cues, warnings = subio.parse_text(text)
    assert cues[0].start == pytest.approx(2.0)
    assert cues[0].end == pytest.approx(5.0)
    assert any("reversed" in w for w in warnings)


def test_out_of_order_cues_are_sorted():
    text = (
        "1\n00:00:05,000 --> 00:00:06,000\nlater\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nearlier\n"
    )
    cues, _ = subio.parse_text(text)
    assert [c.text for c in cues] == ["earlier", "later"]


def test_empty_text_raises():
    with pytest.raises(SubtitleError) as exc:
        subio.parse_text("\n\n\n")
    assert exc.value.code == E_SUB_EMPTY


def test_block_without_timecode_skipped_with_warning():
    text = "1\n00:00:01,000 --> 00:00:02,000\nok\n\ngarbage block\n\n2\n00:00:03,000 --> 00:00:04,000\nok2\n"
    cues, warnings = subio.parse_text(text)
    assert len(cues) == 2
    assert any("skipped" in w for w in warnings)


def test_unparseable_timecode_block_skipped():
    text = "1\n00:xx:01,000 --> 00:00:02,000\nbad\n\n2\n00:00:03,000 --> 00:00:04,000\ngood\n"
    cues, _ = subio.parse_text(text)
    assert len(cues) == 1
    assert cues[0].text == "good"


def test_millisecond_padding_interpretation():
    # "5" must mean 500ms, not 5ms — a real off-by-100x bug in naive parsers
    assert subio._parse_timecode("00:00:01,5") == pytest.approx(1.5)
    assert subio._parse_timecode("00:00:01,05") == pytest.approx(1.05)
    assert subio._parse_timecode("00:00:01,005") == pytest.approx(1.005)


def test_format_timecode_roundtrip():
    for t in (0.0, 1.0, 61.5, 3599.999, 3600.0, 7322.123):
        s = subio.format_timecode(t)
        assert subio._parse_timecode(s) == pytest.approx(t, abs=1e-3)


def test_format_timecode_clamps_negative():
    assert subio.format_timecode(-5.0) == "00:00:00,000"


def test_serialize_renumbers_from_one():
    from subsync_bench.subio import Cue

    cues = [Cue(7, 1.0, 2.0, "a"), Cue(9, 3.0, 4.0, "b")]
    out = subio.serialize_srt(cues)
    assert out.startswith("1\n")
    assert "\n2\n" in out


def test_serialize_uses_lf_only():
    from subsync_bench.subio import Cue

    out = subio.serialize_srt([Cue(1, 1.0, 2.0, "a")])
    assert "\r" not in out


# --- encodings ------------------------------------------------------------


def test_decode_utf8_bom():
    text, enc = subio.decode_bytes("hello".encode("utf-8-sig"))
    assert text == "hello"
    assert enc == "utf-8-sig"


def test_decode_gbk_chinese():
    raw = "中文字幕测试".encode("gbk")
    text, enc = subio.decode_bytes(raw)
    assert text == "中文字幕测试"
    # GB18030 is tried before GBK because it is a strict superset; either name is
    # an acceptable answer as long as the text round-trips.
    assert enc in ("gb18030", "gbk")


def test_decode_utf16_with_bom():
    raw = "hello world".encode("utf-16")
    text, enc = subio.decode_bytes(raw)
    assert text == "hello world"
    assert enc == "utf-16"


def test_decode_explicit_override_wins():
    raw = b"abc"
    text, enc = subio.decode_bytes(raw, override="utf-8")
    assert (text, enc) == ("abc", "utf-8")


def test_decode_bad_override_raises():
    with pytest.raises(SubtitleError):
        subio.decode_bytes(b"abc", override="not-a-codec")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(SubtitleError) as exc:
        subio.load(tmp_path / "nope.srt")
    assert exc.value.code == "E_SUB_MISSING"


def test_load_empty_file_raises(tmp_path):
    p = tmp_path / "empty.srt"
    p.write_bytes(b"   \n\n")
    with pytest.raises(SubtitleError) as exc:
        subio.load(p)
    assert exc.value.code == E_SUB_EMPTY


def test_load_detects_encoding_and_roundtrips(tmp_path):
    p = tmp_path / "zh.srt"
    p.write_bytes(
        "1\n00:00:01,000 --> 00:00:02,000\n你好世界\n".encode("gb18030")
    )
    sub = subio.load(p)
    assert sub.cues[0].text == "你好世界"
    assert sub.encoding in ("gb18030", "gbk")


# --- validation -----------------------------------------------------------


def test_validate_flags_negative_and_empty():
    from subsync_bench.subio import Cue

    problems = subio.validate_cues(
        [Cue(1, -1.0, 2.0, "x"), Cue(2, 3.0, 3.0, "  ")]
    )
    assert any("negative" in p for p in problems)
    assert any("empty text" in p for p in problems)


def test_validate_clean_cues_has_no_problems():
    from subsync_bench.subio import Cue

    assert subio.validate_cues([Cue(1, 1.0, 2.0, "x")]) == []


def test_cue_shift_clamps_at_zero():
    from subsync_bench.subio import Cue

    c = Cue(1, 1.0, 2.0, "x").shifted(-5.0)
    assert c.start == 0.0
    assert c.end == 0.0
    assert c.touched


def test_cue_scale_and_duration():
    from subsync_bench.subio import Cue

    c = Cue(1, 10.0, 12.0, "x").scaled(1.0427)
    assert c.duration == pytest.approx(2.0 * 1.0427)


def test_subtitle_span_and_chars():
    from subsync_bench.subio import Cue, Subtitle

    s = Subtitle([Cue(1, 2.0, 3.0, "abc"), Cue(2, 5.0, 6.0, "de")], "utf-8")
    # span is (earliest start, latest END), not latest start
    assert s.span == (2.0, 6.0)
    assert s.total_text_chars() == 5
    assert len(s) == 2


def test_impossibly_long_cue_flagged():
    text = "1\n00:00:00,000 --> 09:00:00,000\nweird\n"
    _, warnings = subio.parse_text(text)
    assert any("implausible duration" in w for w in warnings)


def test_overlap_reported_not_rejected():
    text = (
        "1\n00:00:01,000 --> 00:00:05,000\na\n\n"
        "2\n00:00:03,000 --> 00:00:06,000\nb\n"
    )
    cues, warnings = subio.parse_text(text)
    assert len(cues) == 2
    assert any("overlapping" in w for w in warnings)
