"""Workspace: project state, revisions, input integrity.

The invariants tested here are what make the workbench a workbench rather than a
script: inputs are immutable and hashed, every revision is reproducible, and
stale measurements are detected instead of silently reused.
"""

from __future__ import annotations

import json

import pytest

from subsync_bench.errors import (
    E_PROJECT_CORRUPT,
    E_PROJECT_EXISTS,
    E_PROJECT_MISSING,
    E_RECEIPT_MISSING,
    WorkspaceError,
)
from subsync_bench.workspace import Project, Revision, sha256_file


def test_create_copies_inputs_and_hashes(tmp_path, srt):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"not really a video")
    sub = srt([(1.0, 2.0, "hi")], "s.srt")
    proj = Project.create(tmp_path / "p", video=video, subtitle=sub)

    assert proj.video_path.exists()
    assert proj.subtitle_path.exists()
    assert proj.manifest["video"]["sha256"] == sha256_file(video)
    assert proj.manifest["subtitle"]["sha256"] == sha256_file(sub)
    assert proj.manifest["schema"] == 1
    assert proj.selected_id is None


def test_create_twice_raises(tmp_path, srt):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    sub = srt([(1.0, 2.0, "hi")], "s.srt")
    Project.create(tmp_path / "p", video=video, subtitle=sub)
    with pytest.raises(WorkspaceError) as exc:
        Project.create(tmp_path / "p", video=video, subtitle=sub)
    assert exc.value.code == E_PROJECT_EXISTS


def test_open_missing_raises(tmp_path):
    with pytest.raises(WorkspaceError) as exc:
        Project.open(tmp_path / "nope")
    assert exc.value.code == E_PROJECT_MISSING


def test_open_corrupt_json_raises(tmp_path):
    (tmp_path / "project.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(WorkspaceError) as exc:
        Project.open(tmp_path)
    assert exc.value.code == E_PROJECT_CORRUPT


def test_open_future_schema_raises(tmp_path):
    (tmp_path / "project.json").write_text(
        json.dumps({"schema": 99}), encoding="utf-8"
    )
    with pytest.raises(WorkspaceError) as exc:
        Project.open(tmp_path)
    assert exc.value.code == E_PROJECT_CORRUPT


def test_save_and_reopen_roundtrip(project):
    project.manifest["name"] = "renamed"
    project.save()
    again = Project.open(project.root)
    assert again.manifest["name"] == "renamed"


def test_manifest_write_is_atomic(project):
    """A crash mid-save must not leave a truncated manifest."""
    project.save()
    tmp = project.manifest_path.with_suffix(".json.tmp")
    assert not tmp.exists(), "temp file must be renamed, not left behind"
    json.loads(project.manifest_path.read_text(encoding="utf-8"))


def test_revision_ids_are_sequential_and_unique(project):
    ids = [project.next_revision_id()]
    project.add_revision(_rev(ids[0]))
    ids.append(project.next_revision_id())
    assert ids == ["r001", "r002"]


def _rev(rid: str, **over) -> Revision:
    kw = {
        "id": rid,
        "created": "2026-01-01T00:00:00",
        "label": "test",
        "offset": -8.2,
        "framerate_scale": 1.0,
        "confidence": 0.7,
        "null_confidence": 0.7,
        "speech_ratio": 0.3,
        "cue_count": 14,
    }
    kw.update(over)
    return Revision(**kw)


def test_add_revision_writes_receipt_file(project):
    project.add_revision(_rev("r001"))
    receipt = project.receipt_dir / "r001.json"
    assert receipt.exists()
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["id"] == "r001"
    assert data["confidence"] == 0.7


def test_get_unknown_revision_raises(project):
    with pytest.raises(WorkspaceError) as exc:
        project.get("r999")
    assert exc.value.code == E_RECEIPT_MISSING


def test_select_and_read_back(project):
    project.add_revision(_rev("r001"))
    project.select("r001")
    assert project.selected_id == "r001"
    assert project.selected().id == "r001"
    assert Project.open(project.root).selected_id == "r001"


def test_select_unknown_raises(project):
    with pytest.raises(WorkspaceError):
        project.select("r404")


def test_verify_inputs_clean(project):
    report = project.verify_inputs()
    assert report["video"]["matches"] is True
    assert report["subtitle"]["matches"] is True
    assert report["stale"] == []


def test_verify_inputs_detects_video_replacement(project):
    project.add_revision(
        _rev("r001", inputs={
            "video": project.manifest["video"]["sha256"],
            "subtitle": project.manifest["subtitle"]["sha256"],
        })
    )
    project.video_path.write_bytes(b"completely different bytes")
    report = project.verify_inputs()
    assert report["video"]["matches"] is False
    assert "r001" in report["stale"]


def test_verify_inputs_detects_subtitle_edit(project):
    project.add_revision(
        _rev("r001", inputs={
            "video": project.manifest["video"]["sha256"],
            "subtitle": project.manifest["subtitle"]["sha256"],
        })
    )
    p = project.subtitle_path
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    report = project.verify_inputs()
    assert report["subtitle"]["matches"] is False
    assert "r001" in report["stale"]


def test_verify_inputs_missing_file(project):
    project.video_path.unlink()
    report = project.verify_inputs()
    assert report["video"]["present"] is False


def test_revision_artifact_missing_raises(project):
    project.add_revision(_rev("r001", artifact="work/r001.aligned.srt"))
    with pytest.raises(WorkspaceError) as exc:
        project.revision_artifact(project.get("r001"))
    assert exc.value.code == E_RECEIPT_MISSING


def test_revision_artifact_without_path_raises(project):
    project.add_revision(_rev("r001"))
    with pytest.raises(WorkspaceError):
        project.revision_artifact(project.get("r001"))


def test_replace_inputs_preserves_history(project):
    project.add_revision(_rev("r001", inputs={
        "video": project.manifest["video"]["sha256"],
        "subtitle": project.manifest["subtitle"]["sha256"],
    }))
    new_sub = project.root.parent / "other.srt"
    new_sub.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nother\n", encoding="utf-8"
    )
    info = project.replace_inputs(video=project.video_path, subtitle=new_sub)
    assert len(project.revisions) == 1, "history must survive"
    assert "other.srt" in info["with"]["subtitle"]
    # the old revision is now stale
    assert "r001" in project.verify_inputs()["stale"]


def test_replace_inputs_does_not_clobber_same_name(project):
    original = project.subtitle_path
    original_bytes = original.read_bytes()
    # a different file with the same basename
    other_dir = project.root.parent / "incoming"
    other_dir.mkdir()
    clash = other_dir / original.name
    clash.write_text("1\n00:00:05,000 --> 00:00:06,000\ndifferent\n", encoding="utf-8")

    project.replace_inputs(video=project.video_path, subtitle=clash)
    # the previous input of that name must still exist untouched
    assert original.exists()
    assert original.read_bytes() == original_bytes
    assert project.subtitle_path != original


def test_summary_shape(project):
    project.add_revision(_rev("r001"))
    project.select("r001")
    s = project.summary()
    assert s["revision_count"] == 1
    assert s["selected"]["id"] == "r001"
    assert "settings" in s


def test_revision_roundtrip_through_dict():
    r = _rev("r001", residual_offset=-0.31, status="suspect", notes=["n1"])
    d = r.to_dict()
    back = Revision.from_dict(d)
    assert back.id == r.id
    assert back.residual_offset == pytest.approx(-0.31)
    assert back.status == "suspect"
    assert back.notes == ["n1"]


def test_settings_defaults_present(project):
    s = project.settings
    assert s["confidence_threshold"] == 0.55
    assert s["residual_tolerance"] == 0.8
    assert s["max_offset_seconds"] == 120.0


def test_sha256_of_known_content(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256_file(p) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
