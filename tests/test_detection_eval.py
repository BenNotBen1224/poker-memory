import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import detection_eval as evaluation


def _diagnostic_dataset(tmp_path, *, clip_bytes=b"video-one"):
    manifest, row, clip_hash = _dataset(tmp_path, clip_bytes=clip_bytes)
    row.update(permission="unknown", permission_details="", development_only=False)
    manifest.write_text(json.dumps({"schema_version": 1, "study_mode": "private_ai_diagnostic", "hands": [row]}))
    prep_path = tmp_path / row["review_prep_path"]
    prep = json.loads(prep_path.read_text())
    prep["decoded_complete"] = True
    prep_path.write_text(json.dumps(prep))
    ref_path = tmp_path / row["reference_path"]
    ref = json.loads(ref_path.read_text())
    ref.update(study_mode="private_ai_diagnostic", authoring="independent_ai",
               ai_provenance={"model": "test-ai", "review_method": "independent full-frame pixel review",
                              "independent_of_detector": True})
    ref["review"]["reviewer_type"] = "ai"
    ref["fields"]["ante_kind"] = {"status": "unknown", "value": None}
    ref_path.write_text(json.dumps(ref))
    return manifest, row, clip_hash


def _matching_diagnostic_dataset(tmp_path, *, clip_bytes=b"video-one"):
    manifest, row, clip_hash = _diagnostic_dataset(tmp_path, clip_bytes=clip_bytes)
    path = tmp_path / row["reference_path"]
    ref = json.loads(path.read_text())
    scoring = _scoring_reference()
    for name, field in scoring["fields"].items():
        field["evidence_frame_ranges"] = [[0, 2]]
    for label in scoring["cards"] + scoring["events"]:
        label["evidence_frame_ranges"] = [[0, 2]]
    ref.update(scoring)
    ref["fields"]["ante_kind"] = {"status": "visible", "value": "uniform", "evidence_frame_ranges": [[0, 2]]}
    path.write_text(json.dumps(ref))
    return manifest, row, clip_hash


def test_ai_reference_is_private_and_never_certified_even_with_verified_rights(tmp_path):
    manifest, row, _ = _diagnostic_dataset(tmp_path)
    result = evaluation.validate_diagnostic_manifest(manifest)
    assert result["valid"]
    assert result["diagnostic_ready_hands"] == 1
    assert result["admitted_hands"] == 0
    assert result["hands"][0]["permission"] == "unknown"
    assert result["human_certified"] is False
    assert result["hands"][0]["certified_benchmark_admitted"] is False
    assert not evaluation.validate_manifest(manifest, 1)["valid"]
    data = json.loads(manifest.read_text())
    data.pop("study_mode")
    data["hands"][0].update(permission="open_license", permission_details="test license")
    manifest.write_text(json.dumps(data))
    strict = evaluation.validate_manifest(manifest, 1)
    assert strict["admitted_hands"] == 0
    assert any("independent_manual" in error["error"] for error in strict["errors"])
    ref_path = tmp_path / row["reference_path"]
    ref = json.loads(ref_path.read_text())
    ref.pop("study_mode")
    ref["authoring"] = "independent_manual"
    ref_path.write_text(json.dumps(ref))
    strict = evaluation.validate_manifest(manifest, 1)
    assert strict["admitted_hands"] == 0
    assert any("AI reviews" in error["error"] for error in strict["errors"])


def test_human_certified_corpus_cannot_be_scored_as_ai_diagnostics(tmp_path):
    manifest, _, _ = _dataset(tmp_path)
    result = evaluation.validate_diagnostic_manifest(manifest)
    assert not result["valid"]
    assert result["admitted_hands"] == 0
    assert result["diagnostic_ready_hands"] == 0


@pytest.mark.parametrize("damage", ["coverage", "clip", "ledger_hash", "reference_hash", "eof",
                                  "provenance", "missing_field", "missing_ante_kind", "development",
                                  "bad_pts", "bad_pixel", "bad_timebase", "bad_row"])
def test_diagnostic_requires_complete_ai_coverage_and_exact_pixel_identity(tmp_path, damage):
    manifest, row, _ = _diagnostic_dataset(tmp_path)
    ref_path = tmp_path / row["reference_path"]
    prep_path = tmp_path / row["review_prep_path"]
    ref, prep = json.loads(ref_path.read_text()), json.loads(prep_path.read_text())
    if damage == "coverage":
        ref["review"]["reviewed_frame_ranges"] = [[0, 1]]
    elif damage == "clip":
        (tmp_path / row["clip_path"]).write_bytes(b"different clip")
    elif damage == "ledger_hash":
        prep["frame_ledger_sha256"] = "1" * 64
    elif damage == "reference_hash":
        ref["frame_ledger_sha256"] = "1" * 64
    elif damage == "eof":
        prep["decoded_complete"] = False
    elif damage == "provenance":
        ref["ai_provenance"]["independent_of_detector"] = False
    elif damage in {"missing_field", "missing_ante_kind"}:
        del ref["fields"]["stacks" if damage == "missing_field" else "ante_kind"]
    elif damage == "development":
        data = json.loads(manifest.read_text())
        del data["hands"][0]["development_only"]
        manifest.write_text(json.dumps(data))
    else:
        ledger = prep_path.with_name("frames.jsonl")
        frames = [json.loads(line) for line in ledger.read_text().splitlines()]
        if damage == "bad_pts":
            frames[1]["pts"] = 0
        elif damage == "bad_pixel":
            frames[1]["rgb_sha256"] = "not a hash"
        elif damage == "bad_timebase":
            frames[1]["time_base"] = "1/0"
        else:
            frames[1] = []
        ledger.write_text("".join(json.dumps(frame) + "\n" for frame in frames))
        prep["frame_ledger_sha256"] = ref["frame_ledger_sha256"] = evaluation._sha256(ledger)
    prep_path.write_text(json.dumps(prep))
    ref_path.write_text(json.dumps(ref))
    result = evaluation.validate_diagnostic_manifest(manifest)
    assert not result["valid"]
    assert result["diagnostic_ready_hands"] == result["admitted_hands"] == 0
    assert result["errors"]


def test_diagnostic_cli_compares_literal_fields_cards_events_and_keeps_inferences(tmp_path, monkeypatch, capsys):
    manifest, _, _ = _matching_diagnostic_dataset(tmp_path)
    monkeypatch.setattr(evaluation, "_verify_detector_freeze", lambda *_: {})
    output = _scoring_result()
    output["observations"] = {"observations": [{"kind": "observed"}, {"kind": "unknown"}, {"kind": "inferred"}]}
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *_, **__: subprocess.CompletedProcess(
        args=[], returncode=0, stdout="__DETECTION_RESULT__" + json.dumps(output), stderr=""))
    path = tmp_path / "private-results.json"
    code = evaluation.main(["diagnostic-score", str(manifest), "--out", str(path),
                            "--detector-freeze", "freeze.json", "--freeze-sha256", "0" * 64])
    assert code == 0
    report = json.loads(path.read_text())
    assert report["notice"] in capsys.readouterr().out
    assert report["admitted_hands"] == 0
    assert report["verdict"] == "PRIVATE_DIAGNOSTIC_REFERENCE_AGREEMENT"
    row = report["hands"][0]
    assert row["permission"] == "unknown"
    assert set(row["metrics"]["visible_fields"]) == evaluation.REQUIRED_FIELDS | {"ante_kind"}
    assert row["metrics"]["cards"]["recognized_visible"] == 9
    assert row["metrics"]["events"]["exact_matches"] == 5
    assert row["metrics"]["exact_hand_match"]
    assert row["detector_observation_kinds"] == {"observed": 1, "unknown": 1, "inferred": 1}
    assert row["metrics"]["unknown_or_inferred_fields"][0]["status"] == "inferred"
    assert row["reference_classifications"]["ai_provenance"]["independent_of_detector"] is True
    assert row["raw_detector_output"]["record"] == output["record"]
    assert report["compute_cost_usd_estimate"] is None
    assert report["automatic_retries"] == 0
    before = path.read_bytes()
    with pytest.raises(ValueError, match="preserve every attempt"):
        evaluation._run_diagnostic(manifest, path, tmp_path / "freeze.json", "0" * 64, 120, None)
    assert path.read_bytes() == before


@pytest.mark.parametrize("failure", ["timeout", "no_result", "invalid_json", "uncertain", "malformed_success"])
def test_diagnostic_keeps_every_failure_and_its_elapsed_cost(tmp_path, monkeypatch, failure):
    manifest, _, _ = _matching_diagnostic_dataset(tmp_path)
    monkeypatch.setattr(evaluation, "_verify_detector_freeze", lambda *_: {})
    def run(*args, **kwargs):
        assert kwargs["timeout"] == 2
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 2, output=b"partial stdout", stderr=b"partial stderr")
        output = "" if failure == "no_result" else "__DETECTION_RESULT__" + {
            "invalid_json": "{",
            "uncertain": json.dumps({"ok": False, "exception": "uncertain",
                                    "stage": "pokergo_sequence", "evidence": {"issues": ["unreadable"]}}),
            "malformed_success": json.dumps({"ok": True}),
        }.get(failure, "")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=output, stderr="raw failure")
    monkeypatch.setattr(evaluation.subprocess, "run", run)
    clock = iter([0, 1, 4, 5])
    monkeypatch.setattr(evaluation.time, "perf_counter", lambda: next(clock))
    report, code = evaluation._run_diagnostic(manifest, tmp_path / "result.json", tmp_path / "freeze.json",
                                             "0" * 64, 2, 36)
    assert code == 1
    assert report["attempted_hands"] == 1
    assert report["cold_subprocess_seconds_total_all_attempts"] == 3
    assert report["compute_cost_usd_estimate"] == 0.03
    assert report["admitted_hands"] == 0
    row = report["hands"][0]
    assert row["status"] == ("TIMED_OUT" if failure == "timeout" else "FAILED")
    if failure == "timeout":
        assert row["stdout"] == "partial stdout" and row["stderr"] == "partial stderr"
        assert row["exit_code"] is None
    if failure == "uncertain":
        assert row["raw_detector_output"]["evidence"] == {"issues": ["unreadable"]}


def test_failed_freeze_prevents_calls_and_preserves_rejection_report(tmp_path, monkeypatch):
    manifest, _, _ = _diagnostic_dataset(tmp_path)
    def fail(*_):
        raise ValueError("frozen asset mismatch")
    monkeypatch.setattr(evaluation, "_verify_detector_freeze", fail)
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not call detector"))
    report, code = evaluation._run_diagnostic(manifest, tmp_path / "result.json", tmp_path / "freeze.json",
                                             "0" * 64, 120, 36)
    assert code == 1
    assert report["attempted_hands"] == 0
    assert report["cold_subprocess_seconds_total_all_attempts"] is None
    assert report["compute_cost_usd_estimate"] is None
    assert report["hands"][0]["status"] == "REJECTED"
    assert report["freeze_or_input_integrity_error"] == "frozen asset mismatch"


def test_detector_output_is_invalidated_when_clip_changes_during_call(tmp_path, monkeypatch):
    manifest, row, _ = _matching_diagnostic_dataset(tmp_path)
    monkeypatch.setattr(evaluation, "_verify_detector_freeze", lambda *_: {})
    def run(*_, **__):
        (tmp_path / row["clip_path"]).write_bytes(b"changed while running")
        return subprocess.CompletedProcess(args=[], returncode=0,
            stdout="__DETECTION_RESULT__" + json.dumps(_scoring_result()), stderr="")
    monkeypatch.setattr(evaluation.subprocess, "run", run)
    report, code = evaluation._run_diagnostic(manifest, tmp_path / "result.json", tmp_path / "freeze.json",
                                             "0" * 64, 120, None)
    assert code == 1
    assert report["diagnostic_scored_hands"] == 0
    assert report["attempted_hands"] == 1
    assert report["hands"][0]["status"] == "INVALIDATED"
    assert report["hands"][0]["raw_detector_output"]["ok"] is True


def test_certified_score_rejects_ai_mode_without_a_worker_call(tmp_path, monkeypatch):
    manifest, _, _ = _diagnostic_dataset(tmp_path)
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("no certified worker"))
    report, code = evaluation._run_score(manifest, tmp_path / "strict.json", 1, None)
    assert code == 1
    assert report["admitted_hands"] == report["scored_hands"] == 0
    assert report["hands"][0]["status"] == "REJECTED"


def test_worker_retains_normal_reading_observations_and_uncertain_stage(tmp_path, monkeypatch, capsys):
    from poker_memory import reader
    from poker_memory.model import HandRecord, Stakes
    from poker_memory.reader.pokergo import PokerGoObservations, UncertainHand
    evidence = PokerGoObservations(issues=["test diagnostic evidence"])
    record = HandRecord(index=1, button=2, hero={"seat": 1, "cards": "AsAh"},
                        seats=[{"seat": 1, "alias": "A", "stack": 100}, {"seat": 2, "alias": "B", "stack": 100}],
                        events=["s1 fold"])
    reading = SimpleNamespace(record=record, stakes=Stakes(sb=1, bb=2), issues=[], wins={},
                              seconds={"total": 1}, pictures=1, ocr_reads=1, observations=evidence)
    monkeypatch.setattr(reader, "read_hand", lambda *_: reading)
    assert evaluation._worker(tmp_path / "synthetic.mp4") == 0
    result = json.loads(capsys.readouterr().out.removeprefix("__DETECTION_RESULT__"))
    assert result["observations"]["issues"] == ["test diagnostic evidence"]
    def uncertain(*_):
        raise UncertainHand(evidence)
    monkeypatch.setattr(reader, "read_hand", uncertain)
    assert evaluation._worker(tmp_path / "synthetic.mp4") == 1
    result = json.loads(capsys.readouterr().out.removeprefix("__DETECTION_RESULT__"))
    assert result["stage"] == "pokergo_sequence"
    assert result["evidence"]["issues"] == ["test diagnostic evidence"]


@pytest.mark.parametrize("damage", [None, "freeze_digest", "source_hash", "model_hash", "versions", "source_set"])
def test_freeze_verifier_checks_real_current_source_assets_and_versions(tmp_path, damage):
    import importlib.metadata
    import platform
    from pathlib import Path

    root = Path(evaluation.__file__).resolve().parents[1]
    def identity(path):
        return {"path": str(path.resolve()), "sha256": evaluation._sha256(path), "bytes": path.stat().st_size}
    config = tmp_path / "config.yaml"
    config.write_text("test OCR config\n")
    environment = tmp_path / "environment.txt"
    environment.write_text("test installed-state snapshot\n")
    frozen = {
        "kind": "frozen_development_detector", "python": platform.python_version(),
        "production_sources": [identity(path) for path in sorted((root / "src" / "poker_memory").rglob("*.py"))],
        "models": [identity(path) for path in sorted((root / "src" / "poker_memory" / "reader" / "models").glob("*"))
                   if path.is_file()],
        "dependency_manifests": [identity(root / "pyproject.toml"), identity(root / "web" / "package-lock.json")],
        "ocr_default_config": identity(config), "environment_freeze": identity(environment),
        "installed_versions": sorted(({"name": d.metadata["Name"], "version": d.version}
                                      for d in importlib.metadata.distributions()), key=lambda d: d["name"].casefold()),
    }
    if damage == "source_hash":
        frozen["production_sources"][0]["sha256"] = "0" * 64
    elif damage == "model_hash":
        frozen["models"][0]["sha256"] = "0" * 64
    elif damage == "versions":
        frozen["installed_versions"][0]["version"] = "not installed"
    elif damage == "source_set":
        frozen["production_sources"].pop()
    path = tmp_path / "detector-freeze.json"
    path.write_text(json.dumps(frozen))
    digest = "0" * 64 if damage == "freeze_digest" else evaluation._sha256(path)
    if damage is None:
        assert evaluation._verify_detector_freeze(path, digest)["kind"] == "frozen_development_detector"
    else:
        with pytest.raises(ValueError):
            evaluation._verify_detector_freeze(path, digest)


@pytest.mark.parametrize("mismatch", ["ante_kind", "card", "event", "timing"])
def test_diagnostic_agreement_rejects_literal_recognition_mismatches(tmp_path, monkeypatch, mismatch):
    manifest, _, _ = _matching_diagnostic_dataset(tmp_path)
    output = _scoring_result()
    if mismatch == "ante_kind":
        output["stakes"]["ante_kind"] = "big_blind"
    elif mismatch == "card":
        output["record"]["hero"]["cards"] = "AdAh"
    elif mismatch == "event":
        output["record"]["events"][-1] = "s1 call 2 @00:05"
    else:
        output["record"]["events"][-1] = "s1 fold @00:20"
    monkeypatch.setattr(evaluation, "_verify_detector_freeze", lambda *_: {})
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *_, **__: subprocess.CompletedProcess(
        args=[], returncode=0, stdout="__DETECTION_RESULT__" + json.dumps(output), stderr=""))
    report, code = evaluation._run_diagnostic(manifest, tmp_path / "result.json", tmp_path / "freeze.json",
                                             "0" * 64, 120, None)
    assert code == 1
    assert report["diagnostic_scored_hands"] == 1
    assert report["admitted_hands"] == 0
    assert report["hands"][0]["metrics"]["exact_hand_match"] is False


def test_private_totals_include_success_failure_and_timeout_without_retries(tmp_path, monkeypatch):
    rows = []
    for index in range(3):
        directory = tmp_path / str(index)
        directory.mkdir()
        _, row, _ = _matching_diagnostic_dataset(directory, clip_bytes=f"synthetic fixture {index}".encode())
        row.update(id=f"hand-{index}", source_url=f"https://example.test/event-{index}", development_only=index == 0)
        for key in ("clip_path", "reference_path", "review_prep_path"):
            row[key] = str(directory / row[key])
        rows.append(row)
    manifest = tmp_path / "combined.json"
    manifest.write_text(json.dumps({"schema_version": 1, "study_mode": "private_ai_diagnostic", "hands": rows}))
    replies = iter([
        subprocess.CompletedProcess(args=[], returncode=0,
                                    stdout="__DETECTION_RESULT__" + json.dumps(_scoring_result()), stderr=""),
        subprocess.CompletedProcess(args=[], returncode=1,
                                    stdout="__DETECTION_RESULT__" + json.dumps({"ok": False, "exception": "failed"}),
                                    stderr="failure stderr"),
        subprocess.TimeoutExpired([], 2, output=b"timeout stdout", stderr=b"timeout stderr"),
    ])
    def run(*_, **__):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply
    clock = iter([0, 1, 3, 4, 9, 10, 13, 15])
    monkeypatch.setattr(evaluation.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(evaluation, "_verify_detector_freeze", lambda *_: {})
    monkeypatch.setattr(evaluation.subprocess, "run", run)
    report, code = evaluation._run_diagnostic(manifest, tmp_path / "combined-results.json", tmp_path / "freeze.json",
                                             "0" * 64, 2, 36)
    assert code == 1
    assert [row["status"] for row in report["hands"]] == ["DIAGNOSTIC_SCORED", "FAILED", "TIMED_OUT"]
    assert report["failure_counts"] == {"DIAGNOSTIC_SCORED": 1, "FAILED": 1, "TIMED_OUT": 1}
    assert report["attempted_hands"] == 3
    assert report["cold_subprocess_seconds_total_all_attempts"] == 10
    assert report["compute_cost_usd_estimate"] == 0.1
    assert report["development_scored_hands"] == 1
    assert report["nondevelopment_diagnostic_scored_hands"] == report["admitted_hands"] == 0
    assert report["automatic_retries"] == 0


def _reference(video_hash, ledger_hash, *, review_ranges=((0, 2),), field_ranges=((0, 1),)):
    fields = {
        name: {"status": "unknown", "value": None}
        for name in sorted(evaluation.REQUIRED_FIELDS)
    }
    return {
        "schema_version": 1,
        "authoring": "independent_manual",
        "video_sha256": video_hash,
        "frame_ledger_sha256": ledger_hash,
        "review": {
            "status": "complete",
            "reviewer": "reviewer-a",
            "completed_at": "2026-11-04T00:00:00Z",
            "frame_count": 3,
            "reviewed_frame_ranges": [list(r) for r in review_ranges],
        },
        "fields": fields,
        "cards": [{
            "owner": "board",
            "card": "As",
            "status": "visible",
            "evidence_frame_ranges": [list(r) for r in field_ranges],
        }],
        "events": [{
            "text": "s1 fold",
            "status": "visible",
            "time_interval_seconds": [1.0, 2.0],
            "evidence_frame_ranges": [list(r) for r in field_ranges],
        }],
    }


def _dataset(tmp_path, *, clip_bytes=b"video-one", review_ranges=((0, 2),),
             field_ranges=((0, 1),), ref_exists=True):
    clip = tmp_path / "hand.webm"
    clip.write_bytes(clip_bytes)
    clip_hash = hashlib.sha256(clip_bytes).hexdigest()
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir()
    ledger_path = prep_dir / "frames.jsonl"
    ledger_path.write_text("".join(json.dumps({
        "index": i, "pts": i, "time_base": "1/24", "width": 10, "height": 10,
        "rgb_sha256": "0" * 64,
    }) + "\n" for i in range(3)))
    ledger_hash = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    (prep_dir / "preparation.json").write_text(json.dumps({
        "status": "PREPARED", "video_sha256": clip_hash, "frame_count": 3,
        "frame_ledger_sha256": ledger_hash,
    }))
    reference = tmp_path / "reference.json"
    if ref_exists:
        reference.write_text(json.dumps(_reference(clip_hash, ledger_hash, review_ranges=review_ranges,
                                                   field_ranges=field_ranges)))
    row = {
        "id": "hand-1",
        "source_url": "https://example.test/event",
        "permission": "open_license",
        "permission_details": "CC BY 4.0",
        "source_recording_sha256": clip_hash,
        "clip_sha256": clip_hash,
        "source_recording": "event-2026-1",
        "hand_interval": {"start_seconds": 10, "end_seconds": 30},
        "clip_path": clip.name,
        "reference_path": reference.name,
        "review_prep_path": "prep/preparation.json",
        "synthetic": False,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "hands": [row]}))
    return manifest, row, clip_hash


def test_manifest_admits_only_complete_independent_full_frame_reference(tmp_path):
    manifest, _, _ = _dataset(tmp_path)

    result = evaluation.validate_manifest(manifest, require_hands=1)

    assert result["valid"]
    assert result["admitted_hands"] == 1


def test_duplicate_clip_hash_is_rejected(tmp_path):
    manifest, first, _ = _dataset(tmp_path)
    second_clip = tmp_path / "copy.webm"
    second_clip.write_bytes((tmp_path / first["clip_path"]).read_bytes())
    second = {**first, "id": "hand-2", "clip_path": second_clip.name,
              "hand_interval": {"start_seconds": 40, "end_seconds": 60}}
    data = json.loads(manifest.read_text())
    data["hands"].append(second)
    manifest.write_text(json.dumps(data))

    result = evaluation.validate_manifest(manifest, require_hands=1)

    assert result["admitted_hands"] == 1
    assert any("duplicate clip SHA-256" in error["error"] for error in result["errors"])


def test_overlapping_hands_from_same_source_are_rejected(tmp_path):
    manifest, first, _ = _dataset(tmp_path)
    second_clip = tmp_path / "hand-two.webm"
    second_clip.write_bytes(b"video-two")
    second_hash = hashlib.sha256(b"video-two").hexdigest()
    second_prep = tmp_path / "prep-two"
    second_prep.mkdir()
    second_ledger = second_prep / "frames.jsonl"
    second_ledger.write_text("".join(json.dumps({
        "index": i, "pts": i, "time_base": "1/24", "width": 10, "height": 10,
        "rgb_sha256": "0" * 64,
    }) + "\n" for i in range(3)))
    second_ledger_hash = hashlib.sha256(second_ledger.read_bytes()).hexdigest()
    (second_prep / "preparation.json").write_text(json.dumps({
        "status": "PREPARED", "video_sha256": second_hash, "frame_count": 3,
        "frame_ledger_sha256": second_ledger_hash,
    }))
    second_ref = tmp_path / "reference-two.json"
    second_ref.write_text(json.dumps(_reference(second_hash, second_ledger_hash)))
    second = {**first, "id": "hand-2", "clip_sha256": second_hash, "clip_path": second_clip.name,
              "source_recording_sha256": first["source_recording_sha256"],
              "reference_path": second_ref.name, "review_prep_path": "prep-two/preparation.json",
              "hand_interval": {"start_seconds": 20, "end_seconds": 40}}
    data = json.loads(manifest.read_text())
    data["hands"][0]["source_url"] = "https://www.youtube.com/watch?v=videoA111"
    data["hands"].append(second)
    manifest.write_text(json.dumps(data))

    result = evaluation.validate_manifest(manifest, require_hands=1)

    assert result["admitted_hands"] == 1
    assert any("source hand interval overlaps hand-1" in error["error"] for error in result["errors"])


def test_distinct_youtube_recordings_do_not_collide_on_watch_path(tmp_path):
    manifest, first, _ = _dataset(tmp_path)
    first["source_url"] = "https://www.youtube.com/watch?v=videoA111"
    first["source_recording_sha256"] = "a" * 64
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "hands": [first]}))
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second_manifest, second, _ = _dataset(second_dir, clip_bytes=b"video-two")
    second["id"] = "hand-2"
    second["source_url"] = "https://www.youtube.com/watch?v=videoB222"
    second["source_recording_sha256"] = "b" * 64
    second["source_recording"] = "recording-b"
    second["hand_interval"] = {"start_seconds": 20, "end_seconds": 40}
    for key in ("clip_path", "reference_path", "review_prep_path"):
        source = second_dir / second[key]
        second[key] = str(source.resolve())
    data = json.loads((tmp_path / "manifest.json").read_text())
    data["hands"].append(second)
    (tmp_path / "manifest.json").write_text(json.dumps(data))

    result = evaluation.validate_manifest(tmp_path / "manifest.json", require_hands=2)

    assert result["valid"]
    assert result["admitted_hands"] == 2


def test_aliases_of_same_youtube_recording_overlap_and_are_rejected(tmp_path):
    manifest, first, _ = _dataset(tmp_path)
    first["source_url"] = "https://www.youtube.com/watch?v=sameVideo1"
    first["source_recording_sha256"] = "c" * 64
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second_manifest, second, _ = _dataset(second_dir, clip_bytes=b"video-two")
    second["id"] = "hand-2"
    second["source_url"] = "https://youtu.be/sameVideo1"
    second["source_recording_sha256"] = "d" * 64
    second["source_recording"] = "alternate-copy-of-recording"
    second["hand_interval"] = {"start_seconds": 20, "end_seconds": 40}
    for key in ("clip_path", "reference_path", "review_prep_path"):
        second[key] = str((second_dir / second[key]).resolve())
    data = {"schema_version": 1, "hands": [first, second]}
    manifest.write_text(json.dumps(data))

    result = evaluation.validate_manifest(manifest, require_hands=1)

    assert result["admitted_hands"] == 1
    assert any("source hand interval overlaps hand-1" in error["error"] for error in result["errors"])


def test_known_wpt_demo_is_excluded_even_when_manifest_hash_matches(tmp_path, monkeypatch):
    manifest, _, _ = _dataset(tmp_path)
    monkeypatch.setattr(evaluation, "_sha256", lambda _: evaluation.CALIBRATION_SHA256)
    data = json.loads(manifest.read_text())
    data["hands"][0]["clip_sha256"] = evaluation.CALIBRATION_SHA256
    manifest.write_text(json.dumps(data))

    result = evaluation.validate_manifest(manifest, require_hands=0)

    assert result["admitted_hands"] == 0
    assert any("known WPT demo is excluded" in error["error"] for error in result["errors"])


def test_incomplete_full_frame_review_is_not_admitted(tmp_path):
    manifest, _, _ = _dataset(tmp_path, review_ranges=((0, 1),))

    result = evaluation.validate_manifest(manifest, require_hands=0)

    assert result["admitted_hands"] == 0
    assert any("do not cover the entire clip" in error["error"] for error in result["errors"])


def test_malformed_evidence_frame_range_is_not_admitted(tmp_path):
    manifest, _, _ = _dataset(tmp_path, field_ranges=((0, 3),))
    data = json.loads(manifest.read_text())
    ref_path = tmp_path / data["hands"][0]["reference_path"]
    reference = json.loads(ref_path.read_text())
    reference["fields"]["stakes"] = {
        "status": "visible", "value": [1, 2, 0], "evidence_frame_ranges": [[0, 3]],
    }
    ref_path.write_text(json.dumps(reference))

    result = evaluation.validate_manifest(manifest, require_hands=0)

    assert result["admitted_hands"] == 0
    assert any("visible field 'stakes'" in error["error"] for error in result["errors"])


def test_missing_reference_is_an_explicit_failure(tmp_path):
    manifest, _, _ = _dataset(tmp_path, ref_exists=False)

    result = evaluation.validate_manifest(manifest, require_hands=0)

    assert result["admitted_hands"] == 0
    assert any("cannot read independent reference" in error["error"] for error in result["errors"])


def test_synthetic_rows_are_never_admitted(tmp_path):
    manifest, _, _ = _dataset(tmp_path)
    data = json.loads(manifest.read_text())
    data["hands"][0]["synthetic"] = True
    manifest.write_text(json.dumps(data))

    result = evaluation.validate_manifest(manifest, require_hands=0)

    assert result["admitted_hands"] == 0
    assert any("synthetic rows are not admitted" in error["error"] for error in result["errors"])


def test_require_hands_100_fails_with_a_short_corpus(tmp_path):
    manifest, _, _ = _dataset(tmp_path)

    result = evaluation.validate_manifest(manifest, require_hands=100)

    assert result["admitted_hands"] == 1
    assert not result["valid"]
    assert result["errors"][-1]["error"] == "only 1 distinct fully referenced hands admitted; 100 required"


def test_decoder_only_or_repeatability_cannot_yield_a_pass():
    assert evaluation._overall_verdict(100, 100, False, 100, True, 100) == "INCOMPLETE_OR_MISMATCH"
    assert evaluation._overall_verdict(100, 100, True, 100, True, 100) == "MATCHES_REFERENCES"
    assert evaluation._overall_verdict(99, 99, True, 100, True, 99) == "INCOMPLETE_OR_MISMATCH"
    assert evaluation._overall_verdict(1, 1, True, 1, False, 1) == "INCOMPLETE_OR_MISMATCH"


def _scoring_result():
    return {
        "ok": True,
        "record": {
            "button": 2,
            "hero": {"seat": 1, "cards": "AsAh"},
            "seats": [
                {"seat": 2, "alias": "Joe", "stack": 80, "hole_cam": "QhQc"},
                {"seat": 1, "alias": "Hero", "stack": 100, "hole_cam": None},
            ],
            "events": [
                "s2 show QhQc @00:01",
                "flop 2d3h4s @00:02",
                "turn 5c @00:03",
                "river 6d @00:04",
                "s1 fold @00:05",
            ],
        },
        "stakes": {"sb": 1, "bb": 2, "ante": 0},
        "issues": [],
        "wins": {},
        "reader_seconds": {"total": 0.2},
        "pictures": 10,
        "ocr_reads": 3,
        "read_and_structured_serialization_seconds": 0.21,
        "structured_serialization_seconds": 0.001,
        "library_import_seconds": 0.1,
    }


def _scoring_reference():
    result = _scoring_result()
    fields = {
        "stakes": [1, 2, 0],
        "button": 2,
        "hero_seat": 1,
        "player_names": [{"seat": 1, "alias": "hero"}, {"seat": 2, "alias": "Joe"}],
        "stacks": [{"seat": 1, "stack": 100}, {"seat": 2, "stack": 80}],
        "opponent_hole_cards": [{"seat": 2, "cards": ["Qc", "Qh"]}],
        "seat_numbers": [1, 2],
        "board_cards": ["2d", "3h", "4s", "5c", "6d"],
    }
    ref = {
        "fields": {
            name: {"status": "visible", "value": value, "evidence_frame_ranges": [[0, 9]]}
            for name, value in fields.items()
        },
        "cards": [
            {"owner": owner, "card": card, "status": "visible", "evidence_frame_ranges": [[0, 9]]}
            for owner, card in [
                ("seat:1", "As"), ("seat:1", "Ah"), ("seat:2", "Qc"), ("seat:2", "Qh"),
                ("board", "2d"), ("board", "3h"), ("board", "4s"), ("board", "5c"), ("board", "6d"),
            ]
        ],
        "events": [
            {"text": line.split("@")[0].strip(), "status": "visible",
             "time_interval_seconds": [0.0, 10.0], "evidence_frame_ranges": [[0, 9]]}
            for line in result["record"]["events"]
        ],
    }
    ref["fields"]["seat_map_inference"] = {
        "status": "inferred", "value": "seat numbers follow reader convention",
        "evidence_frame_ranges": [[0, 9]],
    }
    return ref


def test_scoring_compares_every_required_visible_field_and_card_owner_once():
    result = _scoring_result()
    reference = _scoring_reference()

    metrics = evaluation._score_hand(reference, result)

    assert set(metrics["visible_fields"]) == evaluation.REQUIRED_FIELDS
    assert all(value["match"] for value in metrics["visible_fields"].values())
    assert metrics["cards"] == {
        "expected_visible": 9,
        "recognized_visible": 9,
        "missing": [],
        "extra": [],
        "exact": True,
        "unknown_or_inferred_reference": 0,
    }
    assert metrics["events"]["exact_matches"] == 5
    assert metrics["events"]["all_timing_within_reference"]
    assert metrics["unknown_or_inferred_fields"] == [{
        "field": "seat_map_inference", "status": "inferred",
        "value": "seat numbers follow reader convention",
    }]
    assert metrics["exact_hand_match"]
    assert evaluation._predicted_cards(result["record"]) == [
        ("board", "2d"), ("board", "3h"), ("board", "4s"), ("board", "5c"), ("board", "6d"),
        ("seat:1", "Ah"), ("seat:1", "As"), ("seat:2", "Qc"), ("seat:2", "Qh"),
    ]


def test_score_entrypoint_keeps_rejected_rows_and_cannot_pass_them(tmp_path, monkeypatch):
    result = _scoring_result()
    reference = _scoring_reference()
    validation = {
        "valid": False,
        "admitted_hands": 1,
        "required_hands": 1,
        "errors": [{"id": "bad-hand", "error": "missing independent reference"}],
        "hands": [
            {"id": "good-hand", "status": "ADMITTED", "reference": reference, "clip_path": "clip.mp4"},
            {"id": "bad-hand", "status": "REJECTED", "errors": ["missing independent reference"]},
        ],
    }
    marker = "__DETECTION_RESULT__" + json.dumps(result)
    monkeypatch.setattr(evaluation, "validate_manifest", lambda *_args, **_kwargs: validation)
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess(
        args=[sys.executable], returncode=0, stdout=marker, stderr=""))

    report, exit_code = evaluation._run_score(tmp_path / "manifest.json", tmp_path / "score.json",
                                              require_hands=1, compute_usd_per_hour=None)

    assert exit_code == 1
    assert report["verdict"] == "INCOMPLETE_OR_MISMATCH"
    assert report["manifest_rows"] == 2
    assert report["rejected_or_failed_rows"] == 1
    assert [hand["status"] for hand in report["hands"]] == ["SCORED", "REJECTED"]
    assert report["criteria"]["product_performance_targets_assessed"] is False
    assert report["paid_api_cost_usd"] is None


def test_score_entrypoint_returns_success_only_for_complete_reference_match(tmp_path, monkeypatch):
    result = _scoring_result()
    validation = {
        "valid": True,
        "admitted_hands": 1,
        "required_hands": 1,
        "errors": [],
        "hands": [
            {"id": "good-hand", "status": "ADMITTED", "reference": _scoring_reference(),
             "clip_path": "clip.mp4"},
        ],
    }
    marker = "__DETECTION_RESULT__" + json.dumps(result)
    monkeypatch.setattr(evaluation, "validate_manifest", lambda *_args, **_kwargs: validation)
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess(
        args=[sys.executable], returncode=0, stdout=marker, stderr=""))

    report, exit_code = evaluation._run_score(tmp_path / "manifest.json", tmp_path / "score.json",
                                              require_hands=1, compute_usd_per_hour=None)

    assert report["verdict"] == "MATCHES_REFERENCES"
    assert exit_code == 0


def test_cost_and_elapsed_total_include_failed_detector_attempt(tmp_path, monkeypatch):
    result = _scoring_result()
    reference = _scoring_reference()
    validation = {
        "valid": True, "admitted_hands": 2, "errors": [],
        "hands": [
            {"id": "good", "status": "ADMITTED", "reference": reference, "clip_path": "one.mp4"},
            {"id": "failed", "status": "ADMITTED", "reference": reference, "clip_path": "two.mp4"},
        ],
    }
    replies = iter([
        subprocess.CompletedProcess(args=[], returncode=0,
                                    stdout="__DETECTION_RESULT__" + json.dumps(result), stderr=""),
        subprocess.CompletedProcess(args=[], returncode=1,
                                    stdout="__DETECTION_RESULT__" + json.dumps({
                                        "ok": False, "exception": "decoder error",
                                        "worker_failure_elapsed_seconds": 4.5,
                                    }), stderr="decoder error"),
    ])
    clock_values = iter([10.0, 12.0, 20.0, 25.0])
    monkeypatch.setattr(evaluation, "validate_manifest", lambda *_args, **_kwargs: validation)
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *_args, **_kwargs: next(replies))
    monkeypatch.setattr(evaluation.time, "perf_counter", lambda: next(clock_values))

    report, exit_code = evaluation._run_score(tmp_path / "manifest.json", tmp_path / "score.json",
                                              require_hands=1, compute_usd_per_hour=36.0)

    assert exit_code == 1
    assert [row["status"] for row in report["hands"]] == ["SCORED", "FAILED"]
    assert report["attempted_hands"] == 2
    assert report["attempted_hands_missing_cold_timing"] == 0
    assert report["cold_subprocess_seconds_total"] == 7.0
    assert report["compute_cost_usd_estimate"] == 0.07
    assert report["read_seconds_including_decode_total"] == 0.2


def test_missing_run_measurements_remain_unknown_not_zero_cost(tmp_path, monkeypatch):
    validation = {
        "valid": False, "admitted_hands": 0, "errors": [{"id": None, "error": "no hands"}], "hands": [],
    }
    monkeypatch.setattr(evaluation, "validate_manifest", lambda *_args, **_kwargs: validation)

    report, exit_code = evaluation._run_score(tmp_path / "manifest.json", tmp_path / "score.json",
                                              require_hands=100, compute_usd_per_hour=36.0)

    assert exit_code == 1
    assert report["attempted_hands"] == 0
    assert report["cold_subprocess_seconds_total"] is None
    assert report["compute_cost_usd_estimate"] is None


class _Frame:
    width = 2
    height = 2
    time_base = Fraction(1, 24)

    def __init__(self, index):
        self.pts = index
        self.time = index / 24
        self.index = index

    def to_ndarray(self, *, format):
        assert format == "rgb24"
        return np.full((2, 2, 3), self.index, dtype=np.uint8)


class _Container:
    def __init__(self, frames):
        self.frames = frames
        self.streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 24))])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def decode(self, _stream):
        return iter(self.frames)


def _fake_av(monkeypatch, frame_count):
    frames = [_Frame(i) for i in range(frame_count)]
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda _path: _Container(frames)))


def test_exact_bounded_run_at_eof_is_prepared_and_extracts_pixel_exact_png(tmp_path, monkeypatch):
    video = tmp_path / "small.mp4"
    video.write_bytes(b"fixture video bytes")
    out = tmp_path / "prep"
    _fake_av(monkeypatch, 3)

    preparation = evaluation._write_preparation(video, out, max_frames=3, extract_frames=False)
    evidence = evaluation._extract_frame_range(video, out / "preparation.json",
                                                tmp_path / "exact-frames", 1, 1)

    assert preparation["status"] == "PREPARED"
    assert preparation["frame_count"] == 3
    assert preparation["review_status"] == "NOT REVIEWED"
    assert evidence["status"] == "PREPARED_NOT_REVIEWED"
    assert evidence["frames"][0]["index"] == 1
    assert np.array_equal(np.asarray(Image.open(tmp_path / "exact-frames" / "frame-00000001.png")),
                          np.full((2, 2, 3), 1, dtype=np.uint8))


def test_partial_preparation_resumes_and_zero_new_frame_resume_finishes(tmp_path, monkeypatch):
    video = tmp_path / "small.mp4"
    video.write_bytes(b"fixture video bytes")
    out = tmp_path / "prep"
    _fake_av(monkeypatch, 3)

    partial = evaluation._write_preparation(video, out, max_frames=2, extract_frames=False)
    complete = evaluation._write_preparation(video, out, max_frames=2, extract_frames=False)
    metadata_path = out / "preparation.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "PREPARED_PARTIAL"
    metadata_path.write_text(json.dumps(metadata))
    zero_new_resume = evaluation._write_preparation(video, out, max_frames=2, extract_frames=False)

    assert partial["status"] == "PREPARED_PARTIAL" and partial["frame_count"] == 2
    assert complete["status"] == "PREPARED" and complete["frame_count"] == 3
    assert zero_new_resume["status"] == "PREPARED"
    assert zero_new_resume["frame_count"] == 3


@pytest.fixture
def interrupted_encoded_preparation(tmp_path, monkeypatch):
    import av

    video = tmp_path / "encoded.mp4"
    with av.open(str(video), "w") as box:
        stream = box.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(8):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 25, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)
    original_open = av.open

    class Interrupted:
        def __enter__(self):
            self.box = original_open(str(video))
            self.streams = self.box.streams
            return self

        def __exit__(self, *_args):
            self.box.close()

        def decode(self, stream):
            for i, frame in enumerate(self.box.decode(stream)):
                if i == 3:
                    raise KeyboardInterrupt("simulated interruption after three flushed rows")
                yield frame

    out = tmp_path / "prep"
    with monkeypatch.context() as patch:
        patch.setattr(av, "open", lambda *_args: Interrupted())
        with pytest.raises(KeyboardInterrupt):
            evaluation._write_preparation(video, out, None, False)
    return video, out


def test_interrupted_real_encoded_rows_are_revalidated_before_normal_resume(interrupted_encoded_preparation):
    video, out = interrupted_encoded_preparation
    before = (out / "frames.jsonl").read_bytes()
    with pytest.raises(ValueError, match="counts differ"):
        evaluation._write_preparation(video, out, None, False)
    recovered = evaluation._recover_preparation(video, out)
    assert recovered["frame_count"] == 3
    assert recovered["status"] == "PREPARED_PARTIAL"
    assert recovered["review_status"] == "NOT REVIEWED"
    assert recovered["decoded_complete"] is False
    assert (out / "frames.jsonl").read_bytes() == before
    partial = evaluation._write_preparation(video, out, 2, False)
    assert partial["frame_count"] == 5 and partial["status"] == "PREPARED_PARTIAL"
    complete = evaluation._write_preparation(video, out, None, False)
    assert complete["frame_count"] == 8 and complete["status"] == "PREPARED"
    assert complete["review_status"] == "NOT REVIEWED"
    template = json.loads((out / "reference-template.json").read_text())
    assert template["review"]["status"] == "not_reviewed"
    assert template["review"]["reviewed_frame_ranges"] == []


@pytest.mark.parametrize("damage", ["pixels", "pts", "index", "truncated", "source", "path"])
def test_recovery_rejects_damage_without_rewriting_metadata(interrupted_encoded_preparation, damage):
    video, out = interrupted_encoded_preparation
    meta = out / "preparation.json"
    ledger = out / "frames.jsonl"
    if damage == "source":
        video.write_bytes(video.read_bytes() + b"changed")
    elif damage == "path":
        other = video.with_name("copy.mp4")
        other.write_bytes(video.read_bytes())
        video = other
    elif damage == "truncated":
        ledger.write_bytes(ledger.read_bytes() + b'{"index":3')
    else:
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        key = {"pixels": "rgb_sha256", "pts": "pts", "index": "index"}[damage]
        rows[1][key] = "0" * 64 if damage == "pixels" else 999
        ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = meta.read_bytes()
    with pytest.raises((ValueError, json.JSONDecodeError)):
        evaluation._recover_preparation(video, out)
    assert meta.read_bytes() == before


def test_recovery_never_certifies_eof_and_rejects_completed_metadata(interrupted_encoded_preparation):
    video, out = interrupted_encoded_preparation
    evaluation._recover_preparation(video, out)
    evaluation._write_preparation(video, out, None, False)
    meta = out / "preparation.json"
    complete = json.loads(meta.read_text())
    with pytest.raises(ValueError, match="incomplete"):
        evaluation._recover_preparation(video, out)
    complete["status"] = "PREPARED_PARTIAL"
    meta.write_text(json.dumps(complete))
    recovered = evaluation._recover_preparation(video, out)
    assert recovered["frame_count"] == 8
    assert recovered["status"] == "PREPARED_PARTIAL"
    assert recovered["decoded_complete"] is False
    finished = evaluation._write_preparation(video, out, None, False)
    assert finished["status"] == "PREPARED"
    assert finished["review_status"] == "NOT REVIEWED"
