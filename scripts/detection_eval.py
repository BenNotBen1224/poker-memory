"""Prepare and score an independently reviewed, one-hand-per-clip detector corpus.

Manifest example:
{
  "schema_version": 1,
  "hands": [{
    "id": "hand-001",
    "source_url": "https://example.org/recording",
    "permission": "open_license",
    "permission_details": "CC BY 4.0",
    "source_recording_sha256": "<sha256 of the original recording>",
    "clip_sha256": "<sha256 of the one-hand clip>",
    "source_recording": "recording-001",
    "hand_interval": {"start_seconds": 120.0, "end_seconds": 240.0},
    "clip_path": "clips/hand-001.mp4",
    "reference_path": "references/hand-001.json",
    "review_prep_path": "review-prep/hand-001/preparation.json",
    "synthetic": false
  }]
}

Run `python scripts/detection_eval.py --help` for the validation, preparation,
and scoring commands. Certified references must be independently authored by a
human. Separate private AI diagnostics never enter certified benchmark totals.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

SCHEMA_VERSION = 1
CALIBRATION_SHA256 = "5d0d3f43e883affe89413562a60a5f11ec7aa69afc295defd211db25f51b4f79"
CALIBRATION_URL_PART = "Incredibly_Unbelievable_Fold_on_the_World_Poker_Tour.webm"
PERMISSIONS = {"open_license", "all_participants"}
LABEL_STATUSES = {"visible", "unknown", "inferred"}
REQUIRED_FIELDS = {"stakes", "button", "hero_seat", "player_names", "stacks",
                   "opponent_hole_cards", "seat_numbers", "board_cards"}
MAX_EXTRACT_FRAMES_PER_RUN = 300
DIAGNOSTIC_MODE = "private_ai_diagnostic"
DIAGNOSTIC_NOTICE = "Private diagnostic; AI-reviewed; not human-certified; not admitted to the certified 100-hand benchmark."


def _diagnostic_labels() -> dict:
    return {"study_mode": DIAGNOSTIC_MODE, "notice": DIAGNOSTIC_NOTICE,
            "ai_reviewed": True, "human_certified": False, "admitted_hands": 0,
            "certified_benchmark_admitted": False}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _source_keys(row: dict) -> set[str]:
    keys = {f"recording-sha256:{row['source_recording_sha256'].lower()}"}
    parsed = urlparse(row["source_url"])
    if parsed.hostname:
        host = parsed.hostname.casefold()
        video_id = None
        if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            query = dict(item.split("=", 1) for item in parsed.query.split("&") if "=" in item)
            video_id = query.get("v")
            parts = [part for part in parsed.path.split("/") if part]
            if not video_id and len(parts) == 2 and parts[0] in {"shorts", "embed", "live"}:
                video_id = parts[1]
        elif host == "youtu.be":
            video_id = parsed.path.strip("/").split("/", 1)[0] or None
        if video_id:
            keys.add(f"youtube:{video_id}")
        else:
            keys.add(f"url:{host}{parsed.path.rstrip('/')}")
    return keys


def _load_prep(path: Path, clip_hash: str) -> tuple[dict | None, str | None]:
    try:
        prep = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, f"cannot read review preparation: {error}"
    if prep.get("status") != "PREPARED":
        return None, "full-frame ledger is not complete (must be PREPARED)"
    if prep.get("video_sha256") != clip_hash:
        return None, "full-frame ledger is for a different clip"
    count = prep.get("frame_count")
    if not isinstance(count, int) or count < 1:
        return None, "full-frame ledger has no valid decoded frame count"
    ledger_path = path.with_name("frames.jsonl")
    try:
        with ledger_path.open("r", encoding="utf-8") as ledger:
            ledger_count = 0
            for expected, line in enumerate(ledger):
                item = json.loads(line)
                required = ("index", "pts", "time_base", "width", "height", "rgb_sha256")
                if not isinstance(item, dict) or item.get("index") != expected or any(key not in item for key in required):
                    return None, f"invalid frame ledger entry at index {expected}"
                ledger_count += 1
    except OSError as error:
        return None, f"cannot read full-frame ledger: {error}"
    except (ValueError, json.JSONDecodeError) as error:
        return None, f"malformed full-frame ledger: {error}"
    if ledger_count != count:
        return None, f"full-frame ledger count mismatch: summary {count}, ledger {ledger_count}"
    try:
        ledger_hash = _sha256(ledger_path)
    except OSError as error:
        return None, f"cannot hash full-frame ledger: {error}"
    if prep.get("frame_ledger_sha256") != ledger_hash:
        return None, "full-frame ledger SHA-256 does not match its preparation summary"
    return prep, None


def _ranges_cover_all(ranges: object, frame_count: int) -> bool:
    if not isinstance(ranges, list) or not ranges:
        return False
    valid: list[tuple[int, int]] = []
    for pair in ranges:
        if (not isinstance(pair, list) or len(pair) != 2 or
                not all(isinstance(n, int) and not isinstance(n, bool) for n in pair)):
            return False
        start, end = pair
        if start < 0 or end < start or end >= frame_count:
            return False
        valid.append((start, end))
    valid.sort()
    cursor = 0
    for start, end in valid:
        if start > cursor:
            return False
        cursor = max(cursor, end + 1)
    return cursor >= frame_count


def _valid_evidence(ranges: object, frame_count: int) -> bool:
    if not isinstance(ranges, list) or not ranges:
        return False
    return all(
        isinstance(pair, list) and len(pair) == 2 and
        all(isinstance(n, int) and not isinstance(n, bool) for n in pair) and
        0 <= pair[0] <= pair[1] < frame_count
        for pair in ranges
    )


def _valid_field_value(name: str, value: object) -> bool:
    if name == "ante_kind":
        return value in ("uniform", "big_blind")
    if name == "stakes":
        return (isinstance(value, list) and len(value) == 3 and
                all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in value) and
                value[0] > 0 and value[1] > 0)
    if name in {"button", "hero_seat"}:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if name == "seat_numbers":
        return (isinstance(value, list) and all(isinstance(n, int) and not isinstance(n, bool) and n > 0
                                                for n in value) and len(set(value)) == len(value))
    if name in {"player_names", "stacks", "opponent_hole_cards"}:
        if not isinstance(value, list):
            return False
        for item in value:
            if (not isinstance(item, dict) or not isinstance(item.get("seat"), int) or
                    isinstance(item.get("seat"), bool) or item["seat"] < 1):
                return False
            if name == "player_names" and (
                    not isinstance(item.get("alias"), str) or not item["alias"].strip()):
                return False
            if name == "stacks" and (
                    not isinstance(item.get("stack"), int) or isinstance(item.get("stack"), bool) or
                    item["stack"] <= 0):
                return False
            if name == "opponent_hole_cards":
                cards = item.get("cards")
                if (not isinstance(cards, list) or len(cards) != 2 or
                        any(not isinstance(card, str) or len(card) != 2 or
                            card[0] not in "23456789TJQKA" or card[1] not in "shdc" for card in cards) or
                        len(set(cards)) != 2):
                    return False
        return True
    if name == "board_cards":
        return (isinstance(value, list) and len(value) <= 5 and all(
            isinstance(card, str) and len(card) == 2 and
            card[0] in "23456789TJQKA" and card[1] in "shdc" for card in value) and
            len(set(value)) == len(value))
    return True


def _normalized_field_value(name: str, value: object) -> object:
    if name in {"seat_numbers"} and isinstance(value, list):
        return sorted(value)
    if name in {"player_names", "stacks", "opponent_hole_cards"} and isinstance(value, list):
        rows = []
        for item in value:
            row = dict(item)
            if "alias" in row:
                row["alias"] = row["alias"].casefold()
            if "cards" in row:
                row["cards"] = sorted(row["cards"])
            rows.append(row)
        return sorted(rows, key=lambda row: row["seat"])
    return value


def _validate_reference(path: Path, clip_hash: str, frame_count: int,
                        ledger_hash: str) -> tuple[dict | None, str | None]:
    return _validate_reference_mode(path, clip_hash, frame_count, ledger_hash, diagnostic=False)


def _validate_reference_mode(path: Path, clip_hash: str, frame_count: int,
                             ledger_hash: str, *, diagnostic: bool) -> tuple[dict | None, str | None]:
    try:
        ref = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, f"cannot read independent reference: {error}"
    if ref.get("schema_version") != SCHEMA_VERSION:
        return None, "reference schema_version must be 1"
    if diagnostic:
        if ref.get("study_mode") != DIAGNOSTIC_MODE or ref.get("authoring") != "independent_ai":
            return None, "private diagnostic reference requires study_mode and authoring='independent_ai'"
        provenance = ref.get("ai_provenance")
        if (not isinstance(provenance, dict) or
                provenance.get("independent_of_detector") is not True or
                not all(isinstance(provenance.get(key), str) and provenance[key].strip()
                        for key in ("model", "review_method"))):
            return None, "AI provenance requires model, review_method and independent_of_detector=true"
    elif ref.get("authoring") != "independent_manual" or ref.get("study_mode") not in (None, "certified"):
        return None, "certified reference must declare authoring='independent_manual' and cannot be diagnostic"
    if "detector_output" in ref or "ocr_output" in ref:
        return None, "reference must not contain production detector/OCR output"
    if ref.get("video_sha256") != clip_hash:
        return None, "reference is for a different clip"
    if ref.get("frame_ledger_sha256") != ledger_hash:
        return None, "reference is for a different decoded frame ledger"
    review = ref.get("review")
    if not isinstance(review, dict) or review.get("status") != "complete":
        return None, "independent full-frame review is not complete"
    if diagnostic and review.get("reviewer_type") != "ai":
        return None, "private diagnostic review requires reviewer_type='ai'"
    if not diagnostic and (review.get("reviewer_type") not in (None, "human") or "ai_provenance" in ref):
        return None, "AI reviews cannot be admitted as human-certified references"
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        return None, "completed review must identify its reviewer"
    if not isinstance(review.get("completed_at"), str) or not review["completed_at"].strip():
        return None, "completed review must include completed_at"
    try:
        datetime.fromisoformat(review["completed_at"].replace("Z", "+00:00"))
    except ValueError:
        return None, "completed_at must be an ISO-8601 timestamp"
    if review.get("frame_count") != frame_count:
        return None, "review frame_count does not match the complete decoded ledger"
    if not _ranges_cover_all(review.get("reviewed_frame_ranges"), frame_count):
        return None, "reviewed frame ranges do not cover the entire clip"

    fields = ref.get("fields")
    if not isinstance(fields, dict):
        return None, "reference fields must be an object; mark unavailable values unknown/inferred explicitly"
    missing_fields = REQUIRED_FIELDS - fields.keys()
    if diagnostic:
        missing_fields |= {"ante_kind"} - fields.keys()
    if missing_fields:
        return None, "reference must explicitly classify these fields as visible, unknown, or inferred: " + \
            ", ".join(sorted(missing_fields))
    for name, field in fields.items():
        if not isinstance(field, dict) or field.get("status") not in LABEL_STATUSES:
            return None, f"field {name!r} must have status visible, unknown, or inferred"
        status = field["status"]
        if status == "visible":
            if "value" not in field or not _valid_evidence(field.get("evidence_frame_ranges"), frame_count):
                return None, f"visible field {name!r} requires a value and supporting frame ranges"
            if not _valid_field_value(name, field["value"]):
                return None, f"visible field {name!r} has an invalid value shape"
        elif status == "unknown" and field.get("value", None) is not None:
            return None, f"unknown field {name!r} must have a null or absent value"
        elif status == "inferred":
            if "value" not in field:
                return None, f"inferred field {name!r} requires its value to be stated"
            if not _valid_field_value(name, field["value"]):
                return None, f"inferred field {name!r} has an invalid value shape"
            if not _valid_evidence(field.get("evidence_frame_ranges"), frame_count):
                return None, f"inferred field {name!r} requires supporting frame ranges"

    cards = ref.get("cards")
    if not isinstance(cards, list):
        return None, "reference cards must be a list (use an empty list and explicit unknown fields when none are visible)"
    for i, card in enumerate(cards):
        if not isinstance(card, dict) or card.get("status") not in LABEL_STATUSES:
            return None, f"card {i} must have a valid status"
        if card["status"] == "visible":
            owner, token = card.get("owner"), card.get("card")
            if (not isinstance(owner, str) or not owner or not isinstance(token, str) or len(token) != 2 or
                    token[0] not in "23456789TJQKA" or token[1] not in "shdc" or
                    not _valid_evidence(card.get("evidence_frame_ranges"), frame_count)):
                return None, f"visible card {i} requires owner, canonical rank+suit, and evidence frame ranges"
        elif card["status"] == "unknown" and card.get("card", None) is not None:
            return None, f"unknown card {i} must have null or absent card value"
        elif card["status"] == "inferred":
            token = card.get("card")
            if (not isinstance(card.get("owner"), str) or not card["owner"] or
                    not isinstance(token, str) or len(token) != 2 or
                    token[0] not in "23456789TJQKA" or token[1] not in "shdc" or
                    not _valid_evidence(card.get("evidence_frame_ranges"), frame_count)):
                return None, f"inferred card {i} requires owner, value, and supporting frame ranges"
    visible_card_keys = [(item["owner"], item["card"]) for item in cards if item.get("status") == "visible"]
    if len(visible_card_keys) != len(set(visible_card_keys)):
        return None, "reference contains duplicate visible owner/card entries"

    events = ref.get("events")
    if not isinstance(events, list):
        return None, "reference events must be a list"
    for i, event in enumerate(events):
        if not isinstance(event, dict) or event.get("status") not in LABEL_STATUSES:
            return None, f"event {i} must have a valid status"
        if event["status"] == "visible":
            interval = event.get("time_interval_seconds")
            if (not isinstance(event.get("text"), str) or
                    not isinstance(interval, list) or len(interval) != 2 or
                    not all(isinstance(n, (int, float)) and not isinstance(n, bool) and
                            math.isfinite(n) for n in interval) or
                    interval[0] < 0 or interval[1] < interval[0] or
                    not _valid_evidence(event.get("evidence_frame_ranges"), frame_count)):
                return None, f"visible event {i} requires text, an ordered time interval, and frame evidence"
            try:
                _event_key(event["text"])
            except (ImportError, ValueError) as error:
                return None, f"visible event {i} has invalid event text: {error}"
        elif event["status"] == "unknown" and event.get("text", None) is not None:
            return None, f"unknown event {i} must have null or absent text"
        elif event["status"] == "inferred":
            if (not isinstance(event.get("text"), str) or
                    not _valid_evidence(event.get("evidence_frame_ranges"), frame_count)):
                return None, f"inferred event {i} requires text and supporting frame ranges"
            try:
                _event_key(event["text"])
            except (ImportError, ValueError) as error:
                return None, f"inferred event {i} has invalid event text: {error}"
    return ref, None


def validate_manifest(path: Path, require_hands: int = 0) -> dict:
    return _validate_manifest(path, require_hands, diagnostic=False)


def validate_diagnostic_manifest(path: Path) -> dict:
    result = _validate_manifest(path, 0, diagnostic=True)
    result.update(_diagnostic_labels())
    result["diagnostic_ready_hands"] = sum(row["status"] == "DIAGNOSTIC_READY" for row in result["hands"])
    if not result["hands"]:
        result["errors"].append({"id": None, "error": "private diagnostic manifest must contain hands"})
        result["valid"] = False
    return result


def _validate_manifest(path: Path, require_hands: int, *, diagnostic: bool) -> dict:
    errors: list[dict] = []
    hands: list[dict] = []
    try:
        manifest = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"schema_version": SCHEMA_VERSION, "hands": [], "errors": [{"id": None, "error": str(error)}],
                "admitted_hands": 0, "required_hands": require_hands, "valid": False}
    rows = manifest.get("hands")
    if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(rows, list):
        return {"schema_version": SCHEMA_VERSION, "hands": [], "errors": [
            {"id": None, "error": "manifest must have schema_version 1 and a hands list"}
        ], "admitted_hands": 0, "required_hands": require_hands, "valid": False}
    mode_error = None
    if diagnostic and manifest.get("study_mode") != DIAGNOSTIC_MODE:
        mode_error = "diagnostic manifest requires study_mode='private_ai_diagnostic'"
    if not diagnostic and manifest.get("study_mode") not in (None, "certified"):
        mode_error = "certified validation rejects diagnostic manifests"
    if mode_error:
        errors.append({"id": None, "error": mode_error})

    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    seen_sources: list[tuple[set[str], float, float, str]] = []
    base = path.resolve().parent
    for index, row in enumerate(rows):
        identity = row.get("id") if isinstance(row, dict) else None
        item_errors: list[str] = []
        if mode_error:
            item_errors.append(mode_error)
        if not isinstance(row, dict):
            errors.append({"id": None, "error": f"hands[{index}] must be an object"})
            continue
        required = ("id", "source_url", "permission", "permission_details", "source_recording_sha256",
                    "clip_sha256",
                    "source_recording", "hand_interval",
                    "clip_path", "reference_path", "review_prep_path", "synthetic")
        for field in required:
            if field not in row:
                item_errors.append(f"missing required field {field}")
        if isinstance(identity, str) and identity.strip():
            if identity in seen_ids:
                item_errors.append("duplicate hand id")
            seen_ids.add(identity)
        else:
            item_errors.append("id must be a non-empty string")
        if row.get("permission") not in (PERMISSIONS | {"unknown"} if diagnostic else PERMISSIONS):
            item_errors.append("permission must be open_license or all_participants" +
                               (" or explicitly unknown for private diagnostics" if diagnostic else ""))
        if (not isinstance(row.get("permission_details"), str) or
                (row.get("permission") != "unknown" and not row["permission_details"].strip())):
            item_errors.append("permission_details must name the license or participant-consent basis")
        if diagnostic and not isinstance(row.get("development_only"), bool):
            item_errors.append("development_only must explicitly distinguish calibration from new diagnostic hands")
        if row.get("synthetic") is not False:
            item_errors.append("synthetic rows are not admitted")
        source_url = row.get("source_url")
        parsed_url = urlparse(source_url) if isinstance(source_url, str) else None
        if not parsed_url or parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            item_errors.append("source_url must be an absolute http(s) URL")
        interval = row.get("hand_interval")
        start = end = None
        if isinstance(interval, dict):
            start, end = interval.get("start_seconds"), interval.get("end_seconds")
        if (not isinstance(start, (int, float)) or isinstance(start, bool) or
                not isinstance(end, (int, float)) or isinstance(end, bool) or
                not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start):
            item_errors.append("hand_interval must have finite seconds with 0 <= start < end")
            start = end = None
        if not isinstance(row.get("source_recording"), str) or not row["source_recording"].strip():
            item_errors.append("source_recording must identify the original recording")
        original_hash = row.get("source_recording_sha256")
        if not isinstance(original_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", original_hash):
            item_errors.append("source_recording_sha256 must be the 64-digit hash of the original recording")

        paths: dict[str, Path] = {}
        for key in ("clip_path", "reference_path", "review_prep_path"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                candidate = Path(value)
                paths[key] = (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            else:
                item_errors.append(f"{key} must be a path")

        actual_hash = None
        clip_path = paths.get("clip_path")
        if clip_path:
            if not clip_path.is_file():
                item_errors.append(f"clip does not exist: {clip_path}")
            else:
                try:
                    actual_hash = _sha256(clip_path)
                except OSError as error:
                    item_errors.append(f"cannot hash clip: {error}")
                if actual_hash:
                    supplied = row.get("clip_sha256")
                    if not isinstance(supplied, str) or supplied.lower() != actual_hash:
                        item_errors.append("manifest clip_sha256 does not match clip bytes")
                    if actual_hash == CALIBRATION_SHA256:
                        item_errors.append("known WPT demo is excluded from held-out hands (calibration only)")
                    prior = seen_hashes.get(actual_hash)
                    if prior:
                        item_errors.append(f"duplicate clip SHA-256 also used by {prior}")
                    else:
                        seen_hashes[actual_hash] = str(identity)
            if isinstance(source_url, str) and CALIBRATION_URL_PART.casefold() in source_url.casefold():
                item_errors.append("known WPT demo source is excluded from held-out hands (calibration only)")

        if (start is not None and end is not None and isinstance(row.get("source_recording"), str)
                and re.fullmatch(r"[0-9a-fA-F]{64}", str(original_hash))
                and parsed_url and parsed_url.scheme in {"http", "https"} and parsed_url.netloc):
            keys = _source_keys(row)
            for old_keys, old_start, old_end, old_id in seen_sources:
                if keys & old_keys and start < old_end and old_start < end:
                    item_errors.append(f"source hand interval overlaps {old_id}")
                    break
            seen_sources.append((keys, start, end, str(identity)))

        ref = None
        prep = None
        if not item_errors and actual_hash and paths.get("review_prep_path"):
            prep, prep_error = _load_prep(paths["review_prep_path"], actual_hash)
            if prep_error:
                item_errors.append(prep_error)
            elif diagnostic:
                try:
                    _check_diagnostic_ledger(paths["review_prep_path"], prep)
                except (OSError, ValueError) as error:
                    item_errors.append(str(error))
        if not item_errors and prep and paths.get("reference_path"):
            ref, ref_error = _validate_reference_mode(paths["reference_path"], actual_hash, prep["frame_count"],
                                                      prep["frame_ledger_sha256"], diagnostic=diagnostic)
            if ref_error:
                item_errors.append(ref_error)

        ready_status = "DIAGNOSTIC_READY" if diagnostic else "ADMITTED"
        result_row = {"id": identity, "status": ready_status if not item_errors else "REJECTED",
                      "errors": item_errors, "sha256": actual_hash}
        if diagnostic:
            result_row.update(_diagnostic_labels())
            result_row.update({"permission": row.get("permission"),
                               "permission_details": row.get("permission_details"),
                               "development_only": row.get("development_only")})
        if not item_errors:
            result_row.update({"clip_path": str(clip_path), "reference_path": str(paths["reference_path"]),
                               "review_prep_path": str(paths["review_prep_path"]),
                               "reference": ref, "frame_count": prep["frame_count"],
                               "reference_sha256": _sha256(paths["reference_path"]),
                               "frame_ledger_sha256": prep["frame_ledger_sha256"],
                               "preparation_sha256": _sha256(paths["review_prep_path"]),
                               "source_recording_sha256": original_hash,
                               "hand_interval": interval})
        hands.append(result_row)
        errors.extend({"id": identity, "error": error} for error in item_errors)

    admitted = sum(hand["status"] == "ADMITTED" for hand in hands)
    if admitted < require_hands:
        errors.append({"id": None, "error": f"only {admitted} distinct fully referenced hands admitted; "
                                             f"{require_hands} required"})
    return {"schema_version": SCHEMA_VERSION, "hands": hands, "errors": errors,
            "admitted_hands": admitted, "required_hands": require_hands,
            "valid": not errors and admitted >= require_hands}


def _check_diagnostic_ledger(path: Path, prep: dict) -> None:
    from fractions import Fraction

    if prep.get("decoded_complete") is not True:
        raise ValueError("private diagnostic preparation must explicitly record decoded_complete=true at EOF")
    if not isinstance(prep["frame_count"], int) or isinstance(prep["frame_count"], bool):
        raise ValueError("diagnostic frame_count must be an integer, not a boolean")
    previous_time = None
    dimensions = None
    with path.with_name("frames.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            frame = json.loads(line)
            pts, width, height = frame["pts"], frame["width"], frame["height"]
            if (not isinstance(frame["index"], int) or isinstance(frame["index"], bool) or frame["index"] != index or
                    not isinstance(pts, int) or isinstance(pts, bool) or
                    not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in (width, height)) or
                    not re.fullmatch(r"[0-9a-f]{64}", str(frame["rgb_sha256"]))):
                raise ValueError(f"invalid private diagnostic pixel/PTS identity at frame {index}")
            try:
                time_base = Fraction(frame["time_base"])
            except (TypeError, ValueError, ZeroDivisionError) as error:
                raise ValueError(f"invalid timebase at frame {index}") from error
            presentation_time = pts * time_base
            if time_base <= 0 or (previous_time is not None and presentation_time <= previous_time):
                raise ValueError(f"non-increasing frame PTS at index {index}")
            if dimensions is not None and dimensions != (width, height):
                raise ValueError("changing frame dimensions are unsupported for diagnostic clips")
            previous_time, dimensions = presentation_time, (width, height)


def _recover_preparation(video: Path, out: Path) -> dict:
    import av

    meta_path, ledger_path = out / "preparation.json", out / "frames.jsonl"
    meta = _read_json(meta_path)
    if meta.get("status") not in {"PREPARING", "PREPARED_PARTIAL"}:
        raise ValueError("recovery requires an incomplete preparation")
    if meta.get("video_sha256") != _sha256(video) or meta.get("video_path") != str(video.resolve()):
        raise ValueError("recovery source differs from the original prepared video")
    count = 0
    with ledger_path.open(encoding="utf-8") as ledger, av.open(str(video)) as container:
        frames = iter(container.decode(container.streams.video[0]))
        for index, line in enumerate(ledger):
            item = json.loads(line)
            frame = next(frames, None)
            if frame is None:
                raise ValueError(f"recovery ledger exceeds source at index {index}")
            expected = {
                "index": index, "pts": frame.pts,
                "time_base": str(frame.time_base or container.streams.video[0].time_base),
                "time_seconds": float(frame.time) if frame.time is not None else None,
                "width": frame.width, "height": frame.height,
                "rgb_sha256": hashlib.sha256(frame.to_ndarray(format="rgb24").tobytes()).hexdigest(),
            }
            if item != expected:
                raise ValueError(f"recovery frame differs from source at index {index}")
            count += 1
    meta.update({
        "status": "PREPARED_PARTIAL", "review_status": "NOT REVIEWED",
        "frame_count": count, "decoded_complete": False, "updated_at": _utc(),
        "frame_ledger": "frames.jsonl", "frame_ledger_sha256": _sha256(ledger_path),
        "recovery": "Every existing ledger row revalidated against decoded source; EOF not certified.",
    })
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp.replace(meta_path)
    return meta


def _write_preparation(video: Path, out: Path, max_frames: int | None, extract_frames: bool) -> dict:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("video preparation requires the vision extra: pip install -e \".[vision]\"") from error

    if extract_frames and (max_frames is None or max_frames > MAX_EXTRACT_FRAMES_PER_RUN):
        raise ValueError(f"--extract-frames requires --max-frames <= {MAX_EXTRACT_FRAMES_PER_RUN}")
    video_hash = _sha256(video)
    out.mkdir(parents=True, exist_ok=True)
    meta_path, ledger_path = out / "preparation.json", out / "frames.jsonl"
    existing = 0
    if ledger_path.exists():
        if not meta_path.exists():
            raise ValueError("frames.jsonl exists without preparation.json; do not guess whether it is resumable")
        meta = _read_json(meta_path)
        if meta.get("video_sha256") != video_hash or meta.get("video_path") != str(video.resolve()):
            raise ValueError("resume source differs from the original prepared video")
        with ledger_path.open("r", encoding="utf-8") as ledger:
            existing = 0
            for expected, line in enumerate(ledger):
                item = json.loads(line)
                required = ("index", "pts", "time_base", "width", "height", "rgb_sha256")
                if item.get("index") != expected or any(key not in item for key in required):
                    raise ValueError(f"existing frame ledger is invalid at index {expected}")
                existing += 1
        if meta.get("frame_count") != existing:
            raise ValueError("preparation summary and frame ledger counts differ")
        if meta.get("frame_ledger_sha256") != _sha256(ledger_path):
            raise ValueError("existing frame ledger SHA-256 does not match its preparation summary")
        if meta.get("status") == "PREPARED":
            return meta
    elif meta_path.exists():
        raise ValueError("preparation.json exists without frames.jsonl")
    else:
        meta = {
            "schema_version": SCHEMA_VERSION, "status": "PREPARING", "review_status": "NOT REVIEWED",
            "video_path": str(video.resolve()), "video_sha256": video_hash, "frame_count": 0,
            "decoded_complete": False, "created_at": _utc(),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    decoded_here = 0
    decoded_complete = False
    source_frame_count = 0
    frame_dir = out / "frames"
    if extract_frames:
        from PIL import Image
        frame_dir.mkdir(exist_ok=True)
    with av.open(str(video)) as container, ledger_path.open("a", encoding="utf-8") as ledger:
        stream = container.streams.video[0]
        time_base = stream.time_base
        last_time = None
        for index, frame in enumerate(container.decode(stream)):
            source_frame_count = index + 1
            if frame.time is not None:
                timestamp = float(frame.time)
                if last_time is not None and timestamp < last_time:
                    raise ValueError(f"decoder did not return presentation order at frame {index}")
                last_time = timestamp
            if index < existing:
                continue
            if max_frames is not None and decoded_here >= max_frames:
                break
            rgb = frame.to_ndarray(format="rgb24")
            item = {
                "index": index,
                "pts": frame.pts,
                "time_base": str(frame.time_base or time_base),
                "time_seconds": float(frame.time) if frame.time is not None else None,
                "width": frame.width,
                "height": frame.height,
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
            ledger.write(json.dumps(item, separators=(",", ":")) + "\n")
            ledger.flush()
            if extract_frames:
                Image.fromarray(rgb).save(frame_dir / f"{index:08d}.png")
            decoded_here += 1
        else:
            decoded_complete = True
            if source_frame_count != existing + decoded_here:
                raise ValueError(f"source frame count changed since preparation: decoded {source_frame_count}, "
                                 f"ledger expects {existing + decoded_here}")

    if existing + decoded_here < 1:
        raise ValueError("video contains no decodable frames")
    meta.update({
        "status": "PREPARED" if decoded_complete else "PREPARED_PARTIAL",
        "review_status": "NOT REVIEWED",
        "frame_count": existing + decoded_here,
        "decoded_complete": decoded_complete,
        "updated_at": _utc(),
        "frame_ledger": "frames.jsonl",
        "frame_ledger_sha256": _sha256(ledger_path),
        "full_resolution_evidence": "Original video plus per-frame PTS/hash ledger; optional bounded PNGs are in frames/.",
    })
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp.replace(meta_path)
    _write_review_html(out / "review.html")
    if decoded_complete:
        template_path = out / "reference-template.json"
        if not template_path.exists():
            fields = {
                name: {"status": "unknown", "value": None}
                for name in sorted(REQUIRED_FIELDS)
            }
            template = {
                "schema_version": SCHEMA_VERSION,
                "authoring": "independent_manual",
                "video_sha256": video_hash,
                "frame_ledger_sha256": meta["frame_ledger_sha256"],
                "review": {
                    "status": "not_reviewed", "reviewer": "", "completed_at": "",
                    "frame_count": meta["frame_count"], "reviewed_frame_ranges": [],
                },
                "fields": fields,
                "cards": [],
                "events": [],
            }
            template_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return meta


def _write_review_html(path: Path) -> None:
    html = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Frame review preparation — NOT REVIEWED</title>
<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}
video{width:100%;max-height:75vh;background:#111}button{padding:.5rem 1rem;margin:.5rem .3rem .5rem 0}
code{overflow-wrap:anywhere}.warning{color:#8a3000;font-weight:700}</style>
<h1>Full-frame review preparation</h1>
<p class="warning">PREPARED is not REVIEWED. This viewer does not create labels or claim human review.</p>
<label>Choose the original video <input id="videoFile" type="file" accept="video/*"></label>
<label>Choose frames.jsonl <input id="ledgerFile" type="file"></label>
<label>Choose preparation.json <input id="prepFile" type="file"></label>
<video id="video" controls></video>
<p id="position">Load both files to begin.</p>
<button id="prev">Previous frame</button><button id="next">Next frame</button>
<p>Video seeking by PTS is approximate and does not prove which decoded frame the browser displayed. If a
detail is ambiguous, extract the exact decoded frame by index with the evaluation tool (lossless PNG, checked
against its RGB hash) and inspect it at full resolution. This viewer never marks frames reviewed.</p>
<script>
let entries=[], at=0, prep=null, source=null;
const video=document.querySelector('#video'), pos=document.querySelector('#position');
const hex=bytes=>[...new Uint8Array(bytes)].map(b=>b.toString(16).padStart(2,'0')).join('');
async function checkReady(){
 if(!source||!prep||!entries.length)return false;
 if(prep.status!=='PREPARED'||prep.decoded_complete!==true||entries.length!==prep.frame_count)
  throw Error('Preparation is partial or the ledger frame count differs.');
 if(hex(await crypto.subtle.digest('SHA-256',await source.arrayBuffer()))!==prep.video_sha256)
  throw Error('Selected original video SHA-256 differs from preparation.json.');
 const ledgerText=await document.querySelector('#ledgerFile').files[0].text();
 if(hex(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(ledgerText)))!==prep.frame_ledger_sha256)
  throw Error('Selected frames.jsonl SHA-256 differs from preparation.json.');
 return true;
}
document.querySelector('#videoFile').onchange=e=>{source=e.target.files[0]||null;if(source)video.src=URL.createObjectURL(source);show()};
document.querySelector('#ledgerFile').onchange=async e=>{
 if(!e.target.files[0])return;
 try{entries=(await e.target.files[0].text()).trim().split(/\\r?\\n/).filter(Boolean).map(JSON.parse);at=0;show()}
 catch(error){pos.textContent=error.message}
};
document.querySelector('#prepFile').onchange=async e=>{
 if(!e.target.files[0])return;
 try{prep=JSON.parse(await e.target.files[0].text());await show()}
 catch(error){pos.textContent=error.message}
};
async function show(){
 if(!entries.length)return;
 try{
  if(!await checkReady()){pos.textContent='Load the original video, ledger, and preparation.json.';return}
  const x=entries[at];pos.textContent=
   `PREPARED — NOT REVIEWED | frame ${x.index+1}/${entries.length} | index ${x.index} | PTS ${x.pts} × ${x.time_base} = ${x.time_seconds}s`;
  if(x.time_seconds!==null)video.currentTime=x.time_seconds;
 }catch(error){pos.textContent=`Cannot verify frame source: ${error.message}`}
}
document.querySelector('#prev').onclick=()=>{at=Math.max(0,at-1);show()};
document.querySelector('#next').onclick=()=>{at=Math.min(entries.length-1,at+1);show()};
</script>"""
    path.write_text(html, encoding="utf-8")


def _extract_frame_range(video: Path, preparation_path: Path, out: Path,
                         start_index: int, count: int) -> dict:
    if start_index < 0 or count < 1 or count > MAX_EXTRACT_FRAMES_PER_RUN:
        raise ValueError(f"request 1..{MAX_EXTRACT_FRAMES_PER_RUN} frames with a nonnegative start index")
    prep, error = _load_prep(preparation_path, _sha256(video))
    if error or prep is None:
        raise ValueError(error or "invalid full-frame preparation")
    stop = min(start_index + count, prep["frame_count"])
    if start_index >= stop:
        raise ValueError(f"start index {start_index} is outside {prep['frame_count']} frames")
    ledger_path = preparation_path.with_name("frames.jsonl")
    with ledger_path.open("r", encoding="utf-8") as ledger_file:
        entries = [json.loads(line) for line in ledger_file if line.strip()]
    try:
        import av
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("frame extraction requires the vision extra: pip install -e \".[vision]\"") from error

    out.mkdir(parents=True, exist_ok=True)
    extracted = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index < start_index:
                continue
            if index >= stop:
                break
            rgb = frame.to_ndarray(format="rgb24")
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            expected = entries[index]
            if digest != expected["rgb_sha256"]:
                raise ValueError(f"decoded frame {index} does not match the prepared RGB hash")
            if frame.width != expected["width"] or frame.height != expected["height"]:
                raise ValueError(f"decoded frame {index} dimensions differ from the ledger")
            image_path = out / f"frame-{index:08d}.png"
            Image.fromarray(rgb).save(image_path)
            extracted.append({
                "index": index, "pts": expected["pts"], "time_base": expected["time_base"],
                "time_seconds": expected.get("time_seconds"), "width": expected["width"],
                "height": expected["height"], "rgb_sha256": digest, "png": image_path.name,
            })
    if len(extracted) != stop - start_index:
        raise ValueError(f"video ended before requested frame range; extracted {len(extracted)}")
    result = {
        "status": "PREPARED_NOT_REVIEWED",
        "video_sha256": prep["video_sha256"],
        "frame_ledger_sha256": prep["frame_ledger_sha256"],
        "frame_count": prep["frame_count"],
        "frames": extracted,
    }
    (out / "frame-evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _worker(clip: Path) -> int:
    import_started = time.perf_counter()
    try:
        from poker_memory.reader import read_hand
        import_seconds = time.perf_counter() - import_started
        read_started = time.perf_counter()
        reading = read_hand(clip)
        result = {
            "ok": True,
            "record": reading.record.model_dump(mode="json"),
            "stakes": reading.stakes.model_dump(mode="json"),
            "issues": reading.issues,
            "wins": reading.wins,
            "reader_seconds": reading.seconds,
            "pictures": reading.pictures,
            "ocr_reads": reading.ocr_reads,
            "library_import_seconds": import_seconds,
            "observations": reading.observations.model_dump(mode="json") if reading.observations else None,
        }
        serialization_started = time.perf_counter()
        json.dumps(result, separators=(",", ":"))
        serialization_seconds = time.perf_counter() - serialization_started
        result["structured_serialization_seconds"] = serialization_seconds
        result["read_and_structured_serialization_seconds"] = time.perf_counter() - read_started
    except Exception as error:
        result = {"ok": False, "exception": traceback.format_exc(),
                  "worker_failure_elapsed_seconds": time.perf_counter() - import_started}
        if hasattr(error, "evidence"):
            result["evidence"] = error.evidence.model_dump(mode="json")
            result["stage"] = error.stage
    print("__DETECTION_RESULT__" + json.dumps(result, separators=(",", ":")))
    return 0 if result["ok"] else 1


def _event_key(line: str) -> tuple:
    from poker_memory.model import parse_event

    event = parse_event(1, line)
    return event.kind, event.seat, event.verb, event.amount, event.street, tuple(event.cards)


def _predicted_cards(record: dict) -> list[tuple[str, str]]:
    from poker_memory.model import parse_cards, parse_event

    cards: set[tuple[str, str]] = set()
    hero = record.get("hero", {})
    if hero.get("seat") and hero.get("cards"):
        cards.update((f"seat:{hero['seat']}", card) for card in parse_cards(hero["cards"]))
    for seat in record.get("seats", []):
        if seat.get("hole_cam"):
            cards.update((f"seat:{seat['seat']}", card) for card in parse_cards(seat["hole_cam"]))
    for line in record.get("events", []):
        event = parse_event(1, line)
        if event.kind == "board":
            cards.update(("board", card) for card in event.cards)
        elif event.kind == "show" and event.seat is not None:
            cards.update((f"seat:{event.seat}", card) for card in event.cards)
    return sorted(cards)


def _predicted_fields(record: dict, stakes: dict) -> dict[str, object]:
    from poker_memory.model import parse_cards, parse_event

    seats = record.get("seats", [])
    opponents: dict[int, set[str]] = {}
    for seat in seats:
        if seat.get("hole_cam"):
            opponents.setdefault(seat["seat"], set()).update(parse_cards(seat["hole_cam"]))
    for line in record.get("events", []):
        event = parse_event(1, line)
        if event.kind == "show" and event.seat is not None:
            opponents.setdefault(event.seat, set()).update(event.cards)
    hero_seat = record.get("hero", {}).get("seat")
    board_cards = []
    for line in record.get("events", []):
        event = parse_event(1, line)
        if event.kind == "board":
            board_cards.extend(event.cards)
    return {
        "stakes": [stakes.get("sb"), stakes.get("bb"), stakes.get("ante", 0)],
        "ante_kind": stakes.get("ante_kind", "uniform"),
        "button": record.get("button"),
        "hero_seat": hero_seat,
        "player_names": [{"seat": seat["seat"], "alias": seat["alias"]}
                         for seat in seats],
        "stacks": [{"seat": seat["seat"], "stack": seat["stack"]} for seat in seats],
        "opponent_hole_cards": [
            {"seat": seat, "cards": sorted(cards)}
            for seat, cards in sorted(opponents.items()) if seat != hero_seat
        ],
        "seat_numbers": [seat["seat"] for seat in sorted(seats, key=lambda seat: seat["seat"])],
        "board_cards": board_cards,
    }


def _score_hand(reference: dict, result: dict) -> dict:
    from poker_memory.model import parse_event

    record = result["record"]
    stakes = result["stakes"]
    predicted_fields = _predicted_fields(record, stakes)
    visible_fields = {name: field for name, field in reference["fields"].items()
                      if field.get("status") == "visible"}
    field_rows = {}
    for name, field in visible_fields.items():
        actual = predicted_fields.get(name)
        expected = _normalized_field_value(name, field.get("value"))
        actual = _normalized_field_value(name, actual)
        field_rows[name] = {"expected": expected, "actual": actual, "match": expected == actual}

    expected_cards = Counter((item["owner"], item["card"]) for item in reference["cards"]
                             if item.get("status") == "visible")
    actual_cards = Counter(_predicted_cards(record))
    card_rows = {
        "expected_visible": sum(expected_cards.values()),
        "recognized_visible": sum((expected_cards & actual_cards).values()),
        "missing": [{"owner": owner, "card": card, "count": count}
                    for (owner, card), count in (expected_cards - actual_cards).items()],
        "extra": [{"owner": owner, "card": card, "count": count}
                  for (owner, card), count in (actual_cards - expected_cards).items()],
        "exact": expected_cards == actual_cards,
        "unknown_or_inferred_reference": sum(item.get("status") in {"unknown", "inferred"}
                                              for item in reference["cards"]),
    }

    expected_events = [event for event in reference["events"] if event.get("status") == "visible"]
    actual_events = record.get("events", [])
    expected_keys = [_event_key(event["text"]) for event in expected_events]
    actual_keys = [_event_key(line) for line in actual_events]
    matcher = difflib.SequenceMatcher(a=expected_keys, b=actual_keys, autojunk=False)
    paired: list[tuple[int, int]] = []
    missing: list[int] = []
    extra: list[int] = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            paired.extend(zip(range(a0, a1), range(b0, b1)))
        elif tag == "delete":
            missing.extend(range(a0, a1))
        elif tag == "insert":
            extra.extend(range(b0, b1))
        else:
            shared = min(a1 - a0, b1 - b0)
            paired.extend(zip(range(a0, a0 + shared), range(b0, b0 + shared)))
            missing.extend(range(a0 + shared, a1))
            extra.extend(range(b0 + shared, b1))
    timing = []
    mismatches = []
    for expected_index, actual_index in paired:
        expected = expected_events[expected_index]
        actual = actual_events[actual_index]
        try:
            parsed = parse_event(actual_index + 1, actual)
            predicted_range = ([parsed.time.earliest_ms / 1000, parsed.time.latest_ms / 1000]
                               if parsed.time else None)
            ref_range = expected["time_interval_seconds"]
            gap = None if predicted_range is None else max(
                0.0, ref_range[0] - predicted_range[1], predicted_range[0] - ref_range[1])
            timing.append({"event": expected["text"], "reference_interval_seconds": ref_range,
                           "detected_interval_seconds": predicted_range, "error_seconds": gap,
                           "within_reference_interval": gap == 0.0})
        except (ValueError, KeyError) as error:
            timing.append({"event": expected.get("text"), "error": str(error), "error_seconds": None,
                           "within_reference_interval": False})
        if expected_keys[expected_index] != actual_keys[actual_index]:
            mismatches.append({"reference": expected["text"], "detected": actual})
    for index in missing:
        mismatches.append({"reference": expected_events[index]["text"], "detected": None})
    for index in extra:
        mismatches.append({"reference": None, "detected": actual_events[index]})
    event_rows = {
        "reference_visible": len(expected_events),
        "detected": len(actual_events),
        "exact_matches": sum(expected_keys[i] == actual_keys[j] for i, j in paired),
        "missing": [expected_events[i]["text"] for i in missing],
        "extra": [actual_events[i] for i in extra],
        "mismatches": mismatches,
        "timing": timing,
        "all_timing_within_reference": bool(timing) and all(t["within_reference_interval"] for t in timing),
        "unknown_or_inferred_reference": sum(event.get("status") in {"unknown", "inferred"}
                                              for event in reference["events"]),
    }
    unscored = [{"field": name, "status": field["status"], "value": field.get("value")}
                for name, field in reference["fields"].items() if field["status"] != "visible"]
    detector_issues = result.get("issues", [])
    has_observations = bool(field_rows or expected_cards or expected_events)
    exact = (has_observations and all(row["match"] for row in field_rows.values()) and card_rows["exact"] and
             not event_rows["missing"] and not event_rows["extra"] and not event_rows["mismatches"] and
             event_rows["all_timing_within_reference"] and not detector_issues)
    return {
        "visible_fields": field_rows,
        "field_coverage": {"visible_reference_fields": len(visible_fields),
                           "recognized_fields": sum(row["actual"] is not None for row in field_rows.values())},
        "cards": card_rows,
        "events": event_rows,
        "unknown_or_inferred_fields": unscored,
        "reader_issues": detector_issues,
        "exact_hand_match": exact,
        "read_seconds_including_decode": result["reader_seconds"].get("total"),
        "read_and_structured_serialization_seconds":
            result.get("read_and_structured_serialization_seconds"),
        "structured_serialization_seconds": result.get("structured_serialization_seconds"),
        "library_import_seconds": result.get("library_import_seconds"),
        "reader_phase_seconds": result["reader_seconds"],
        "cold_subprocess_seconds": result.get("cold_subprocess_seconds"),
        "cold_startup_and_import_overhead_seconds":
            (result.get("cold_subprocess_seconds") or 0) -
            (result.get("read_and_structured_serialization_seconds") or 0),
        "pictures_sampled_by_detector": result.get("pictures"),
        "ocr_reads": result.get("ocr_reads"),
    }


def _overall_verdict(admitted_hands: int, scored_hands: int, all_exact: bool,
                     require_hands: int, validation_clean: bool, manifest_rows: int) -> str:
    if (validation_clean and admitted_hands >= require_hands and scored_hands == admitted_hands and
            manifest_rows == admitted_hands and all_exact):
        return "MATCHES_REFERENCES"
    return "INCOMPLETE_OR_MISMATCH"


def _run_score(manifest_path: Path, out_path: Path, require_hands: int,
               compute_usd_per_hour: float | None) -> tuple[dict, int]:
    validation = validate_manifest(manifest_path, require_hands)
    rows = []
    for hand in validation["hands"]:
        if hand["status"] != "ADMITTED":
            rows.append({"id": hand["id"], "status": "REJECTED", "errors": hand["errors"]})
            continue
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "_worker", hand["clip_path"]],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        except Exception:
            rows.append({"id": hand["id"], "status": "FAILED",
                         "cold_subprocess_seconds": time.perf_counter() - started,
                         "exception": traceback.format_exc()})
            continue
        cold_seconds = time.perf_counter() - started
        marker = next((line[len("__DETECTION_RESULT__"):] for line in reversed(proc.stdout.splitlines())
                       if line.startswith("__DETECTION_RESULT__")), None)
        if marker is None:
            rows.append({"id": hand["id"], "status": "FAILED", "exit_code": proc.returncode,
                         "cold_subprocess_seconds": cold_seconds, "stdout": proc.stdout, "stderr": proc.stderr,
                         "error": "worker returned no structured result"})
            continue
        try:
            result = json.loads(marker)
        except json.JSONDecodeError as error:
            rows.append({"id": hand["id"], "status": "FAILED", "exit_code": proc.returncode,
                         "cold_subprocess_seconds": cold_seconds, "stdout": proc.stdout, "stderr": proc.stderr,
                         "error": f"invalid worker result: {error}"})
            continue
        result["cold_subprocess_seconds"] = cold_seconds
        row = {"id": hand["id"], "exit_code": proc.returncode, "cold_subprocess_seconds": cold_seconds,
               "stdout": proc.stdout, "stderr": proc.stderr}
        if not result.get("ok") or proc.returncode != 0:
            row.update({"status": "FAILED", "exception": result.get("exception"),
                        "worker_failure_elapsed_seconds": result.get("worker_failure_elapsed_seconds")})
        else:
            try:
                row.update({"status": "SCORED", "metrics": _score_hand(hand["reference"], result)})
                row["raw_detector_output"] = {
                    "record": result["record"], "stakes": result["stakes"], "issues": result["issues"],
                    "wins": result["wins"],
                    "reader_seconds": result["reader_seconds"], "pictures": result["pictures"],
                    "ocr_reads": result["ocr_reads"],
                    "read_and_structured_serialization_seconds":
                        result.get("read_and_structured_serialization_seconds"),
                    "structured_serialization_seconds": result.get("structured_serialization_seconds"),
                    "library_import_seconds": result.get("library_import_seconds"),
                }
            except Exception:
                row.update({"status": "FAILED", "exception": traceback.format_exc()})
        rows.append(row)

    scored = [row for row in rows if row["status"] == "SCORED"]
    all_exact = bool(scored) and all(row["metrics"]["exact_hand_match"] for row in scored)
    all_processed = len(scored) == validation["admitted_hands"]
    verdict = _overall_verdict(validation["admitted_hands"], len(scored), all_exact, require_hands,
                               not validation["errors"], len(validation["hands"]))
    read_seconds = sum(row["metrics"].get("read_seconds_including_decode") or 0 for row in scored)
    timed_attempts = [row["cold_subprocess_seconds"] for row in rows
                      if row.get("status") != "REJECTED" and row.get("cold_subprocess_seconds") is not None]
    cold_seconds = sum(timed_attempts) if timed_attempts else None
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc(),
        "manifest_path": str(manifest_path.resolve()),
        "verdict": verdict,
        "verdict_scope": "reference agreement only; not a product-performance pass and does not assess the <$1s/$0.10 targets",
        "criteria": {
            "minimum_distinct_admitted_hands": require_hands,
            "all_admitted_hands_scored": all_processed,
            "every_scored_hand_exact_on_visible_fields_cards_events_and_timing": all_exact,
            "all_manifest_rows_admitted_with_no_validation_errors": not validation["errors"],
            "decoder_only_or_repeatability_only_is_not_a_reference_match": True,
            "product_performance_targets_assessed": False,
        },
        "admitted_hands": validation["admitted_hands"],
        "scored_hands": len(scored),
        "manifest_rows": len(validation["hands"]),
        "rejected_or_failed_rows": len(rows) - len(scored),
        "attempted_hands": len(timed_attempts),
        "attempted_hands_missing_cold_timing": sum(
            row.get("status") != "REJECTED" and row.get("cold_subprocess_seconds") is None for row in rows),
        "validation_errors": validation["errors"],
        "read_seconds_including_decode_total": read_seconds,
        "cold_subprocess_seconds_total": cold_seconds,
        "paid_api_cost_usd": None,
        "compute_cost_usd_estimate": (
            cold_seconds * compute_usd_per_hour / 3600
            if compute_usd_per_hour is not None and cold_seconds is not None else None),
        "compute_usd_per_hour_assumption": compute_usd_per_hour,
        "hands": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    return report, 0 if verdict == "MATCHES_REFERENCES" else 1


def _verify_detector_freeze(path: Path, expected_hash: str) -> dict:
    import importlib.metadata
    import platform

    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or _sha256(path) != expected_hash:
        raise ValueError("detector freeze SHA-256 mismatch")
    frozen = _read_json(path)
    root = Path(__file__).resolve().parents[1]
    expected_sources = {str(p.resolve()) for p in (root / "src" / "poker_memory").rglob("*.py")}
    sources = frozen.get("production_sources", [])
    if (frozen.get("kind") != "frozen_development_detector" or len(sources) != len(expected_sources) or
            {item.get("path") for item in sources} != expected_sources):
        raise ValueError("freeze must bind all current production source files")
    models = frozen.get("models", [])
    expected_models = {str(p.resolve()) for p in (root / "src" / "poker_memory" / "reader" / "models").glob("*")
                       if p.is_file()}
    if {item.get("path") for item in models} != expected_models:
        raise ValueError("freeze does not bind the exact local model/notice asset set")
    manifests = frozen.get("dependency_manifests", [])
    if {item.get("path") for item in manifests} != {str(root / "pyproject.toml"), str(root / "web" / "package-lock.json")}:
        raise ValueError("freeze must bind the dependency manifest and lock snapshot")
    for item in sources + models + manifests + [
            frozen["ocr_default_config"], frozen["environment_freeze"]]:
        asset = Path(item["path"])
        if asset.stat().st_size != item["bytes"] or _sha256(asset) != item["sha256"]:
            raise ValueError(f"frozen asset mismatch: {asset}")
    installed = sorted(({"name": d.metadata["Name"], "version": d.version}
                        for d in importlib.metadata.distributions()), key=lambda d: d["name"].casefold())
    if installed != frozen.get("installed_versions") or platform.python_version() != frozen.get("python"):
        raise ValueError("installed dependency/Python versions differ from frozen environment")
    return frozen


def _check_diagnostic_input(hand: dict) -> None:
    for path_key, hash_key in (("clip_path", "sha256"), ("reference_path", "reference_sha256"),
                               ("review_prep_path", "preparation_sha256")):
        if _sha256(Path(hand[path_key])) != hand[hash_key]:
            raise ValueError(f"diagnostic input changed after validation: {path_key}")
    if _sha256(Path(hand["review_prep_path"]).with_name("frames.jsonl")) != hand["frame_ledger_sha256"]:
        raise ValueError("diagnostic ledger changed after validation")


def _timeout_text(value: str | bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def _run_diagnostic(manifest_path: Path, out_path: Path, freeze_path: Path, freeze_hash: str,
                    timeout_seconds: float, compute_usd_per_hour: float | None) -> tuple[dict, int]:
    if out_path.exists():
        raise ValueError("private diagnostic report already exists; choose a new path to preserve every attempt")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("diagnostic timeout must be finite and positive")
    if compute_usd_per_hour is not None and (not math.isfinite(compute_usd_per_hour) or compute_usd_per_hour < 0):
        raise ValueError("compute cost assumption must be finite and nonnegative")
    run_started = time.perf_counter()
    evaluator_hash = _sha256(Path(__file__))
    validation = validate_diagnostic_manifest(manifest_path)
    freeze_error = None
    try:
        _verify_detector_freeze(freeze_path, freeze_hash)
    except (OSError, ValueError, KeyError, TypeError) as error:
        freeze_error = str(error)
    rows = []
    integrity_error = freeze_error
    for hand in validation["hands"]:
        row = {key: value for key, value in hand.items() if key != "reference"}
        row["reference_classifications"] = hand.get("reference")
        row["detector_attempted"] = False
        if hand["status"] != "DIAGNOSTIC_READY" or integrity_error:
            row["status"] = "REJECTED"
            if integrity_error:
                row["errors"] = row["errors"] + [integrity_error]
            rows.append(row)
            continue
        try:
            _verify_detector_freeze(freeze_path, freeze_hash)
            _check_diagnostic_input(hand)
            if _sha256(Path(__file__)) != evaluator_hash:
                raise ValueError("evaluator changed during diagnostic run")
        except (OSError, ValueError, KeyError, TypeError) as error:
            integrity_error = str(error)
            row.update(status="REJECTED", errors=[integrity_error])
            rows.append(row)
            continue
        row["detector_attempted"] = True
        row["command"] = [sys.executable, "-B", str(Path(__file__).resolve()), "_worker", hand["clip_path"]]
        started = time.perf_counter()
        try:
            proc = subprocess.run(row["command"], capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=timeout_seconds)
            row.update(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                       cold_subprocess_seconds=time.perf_counter() - started)
            marker = next((line[len("__DETECTION_RESULT__"):] for line in reversed(proc.stdout.splitlines())
                           if line.startswith("__DETECTION_RESULT__")), None)
            if marker is None:
                raise ValueError("worker returned no structured result")
            result = json.loads(marker)
            row["raw_detector_output"] = result
            if not result.get("ok") or proc.returncode != 0:
                row.update(status="FAILED", error=result.get("exception", "worker failure"))
            else:
                result["cold_subprocess_seconds"] = row["cold_subprocess_seconds"]
                row.update(status="DIAGNOSTIC_SCORED", metrics=_score_hand(hand["reference"], result))
                if result.get("read_and_structured_serialization_seconds") is None:
                    row["metrics"]["cold_startup_and_import_overhead_seconds"] = None
                observations = result.get("observations")
                row["detector_observation_kinds"] = (
                    dict(Counter(item["kind"] for item in observations.get("observations", [])))
                    if isinstance(observations, dict) else None)
        except subprocess.TimeoutExpired as error:
            row.update(status="TIMED_OUT", exit_code=None, error=str(error), timeout_seconds=timeout_seconds,
                       stdout=_timeout_text(error.stdout), stderr=_timeout_text(error.stderr))
        except Exception:
            row.update(status="FAILED", error=traceback.format_exc())
        if "cold_subprocess_seconds" not in row:
            row["cold_subprocess_seconds"] = time.perf_counter() - started
        try:
            _verify_detector_freeze(freeze_path, freeze_hash)
            _check_diagnostic_input(hand)
            if _sha256(Path(__file__)) != evaluator_hash:
                raise ValueError("evaluator changed during diagnostic run")
        except (OSError, ValueError, KeyError, TypeError) as error:
            integrity_error = str(error)
            row.update(status="INVALIDATED", integrity_error=integrity_error)
        rows.append(row)
    scored = [row for row in rows if row["status"] == "DIAGNOSTIC_SCORED"]
    attempts = [row for row in rows if row["detector_attempted"]]
    cold_total = sum(row["cold_subprocess_seconds"] for row in attempts) if attempts else None
    exact = bool(scored) and len(scored) == len(rows) and all(row["metrics"]["exact_hand_match"] for row in scored)
    success = exact and validation["valid"] and not integrity_error
    report = {
        **_diagnostic_labels(), "schema_version": 1, "created_at": _utc(),
        "verdict": "PRIVATE_DIAGNOSTIC_REFERENCE_AGREEMENT" if success else "PRIVATE_DIAGNOSTIC_INCOMPLETE_OR_MISMATCH",
        "verdict_scope": "AI reference agreement only; no human certification, generalization or product-performance pass",
        "manifest_path": str(manifest_path.resolve()), "detector_freeze_path": str(freeze_path.resolve()),
        "detector_freeze_sha256": freeze_hash, "evaluator_sha256": evaluator_hash,
        "diagnostic_ready_hands": validation["diagnostic_ready_hands"], "diagnostic_scored_hands": len(scored),
        "development_scored_hands": sum(row.get("development_only") is True for row in scored),
        "nondevelopment_diagnostic_scored_hands": sum(row.get("development_only") is False for row in scored),
        "attempted_hands": len(attempts), "failure_counts": dict(Counter(row["status"] for row in rows)),
        "validation_errors": validation["errors"], "freeze_or_input_integrity_error": integrity_error,
        "cold_subprocess_seconds_total_all_attempts": cold_total,
        "evaluation_elapsed_seconds_including_validation": time.perf_counter() - run_started,
        "timeout_seconds_per_hand": timeout_seconds, "automatic_retries": 0,
        "paid_api_cost_usd": None, "compute_usd_per_hour_assumption": compute_usd_per_hour,
        "compute_cost_usd_estimate": (cold_total * compute_usd_per_hour / 3600
                                    if cold_total is not None and compute_usd_per_hour is not None else None),
        "cost_scope": "All detector subprocess attempts, including errors/timeouts; segmentation and AI review cost unmeasured, not zero",
        "hands": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report, 0 if success else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    validate_parser = subs.add_parser("validate", help="validate manifest, clip hashes, complete ledgers and references")
    validate_parser.add_argument("manifest", type=Path)
    validate_parser.add_argument("--require-hands", type=int, default=100)
    prepare_parser = subs.add_parser("prepare", help="decode every frame and make a review ledger/player")
    prepare_parser.add_argument("video", type=Path)
    prepare_parser.add_argument("--out", type=Path, required=True)
    prepare_parser.add_argument("--max-frames", type=int, help="decode at most this many new frames (bounded/resumable)")
    prepare_parser.add_argument("--resume", action="store_true", help="continue an existing ledger for the same source")
    prepare_parser.add_argument("--recover-interrupted", action="store_true",
                                help="revalidate all existing frame rows against source and repair incomplete metadata only")
    prepare_parser.add_argument("--extract-frames", action="store_true",
                                help=f"also save full-resolution PNGs (requires --max-frames <= {MAX_EXTRACT_FRAMES_PER_RUN})")
    extract_parser = subs.add_parser("extract-frames", help="extract exact ledger-verified full-resolution PNG frames")
    extract_parser.add_argument("video", type=Path)
    extract_parser.add_argument("--preparation", type=Path, required=True)
    extract_parser.add_argument("--start-index", type=int, required=True)
    extract_parser.add_argument("--count", type=int, required=True)
    extract_parser.add_argument("--out", type=Path, required=True)
    score_parser = subs.add_parser("score", help="run the real detector on admitted clips and score independent references")
    score_parser.add_argument("manifest", type=Path)
    score_parser.add_argument("--out", type=Path, required=True)
    score_parser.add_argument("--require-hands", type=int, default=100)
    score_parser.add_argument("--compute-usd-per-hour", type=float)
    diagnostic_validate = subs.add_parser("diagnostic-validate", help=DIAGNOSTIC_NOTICE, description=DIAGNOSTIC_NOTICE)
    diagnostic_validate.add_argument("manifest", type=Path)
    diagnostic_score = subs.add_parser("diagnostic-score", help=DIAGNOSTIC_NOTICE, description=DIAGNOSTIC_NOTICE)
    diagnostic_score.add_argument("manifest", type=Path)
    diagnostic_score.add_argument("--out", type=Path, required=True)
    diagnostic_score.add_argument("--detector-freeze", type=Path, required=True)
    diagnostic_score.add_argument("--freeze-sha256", required=True)
    diagnostic_score.add_argument("--timeout-seconds", type=float, default=120)
    diagnostic_score.add_argument("--compute-usd-per-hour", type=float)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        return _worker(Path(args.clip))
    if args.command == "validate":
        result = validate_manifest(args.manifest, args.require_hands)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["valid"] else 1
    if args.command == "diagnostic-validate":
        result = validate_diagnostic_manifest(args.manifest.resolve())
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["valid"] else 1
    if args.command == "diagnostic-score":
        try:
            report, code = _run_diagnostic(args.manifest.resolve(), args.out.resolve(),
                                          args.detector_freeze.resolve(), args.freeze_sha256,
                                          args.timeout_seconds, args.compute_usd_per_hour)
        except (OSError, ValueError) as error:
            print(json.dumps({**_diagnostic_labels(), "error": str(error)}), file=sys.stderr)
            return 1
        print(f"{DIAGNOSTIC_NOTICE}\n{report['verdict']}: {report['diagnostic_scored_hands']} diagnostic scored; "
              f"0 certified admitted; report: {args.out}")
        return code
    if args.command == "prepare":
        if args.recover_interrupted:
            if args.resume or args.max_frames is not None or args.extract_frames:
                parser.error("--recover-interrupted cannot be combined with resume, limits, or extraction")
            try:
                result = _recover_preparation(args.video.resolve(), args.out.resolve())
            except Exception as error:
                print(f"recovery failed: {error}", file=sys.stderr)
                return 1
            print(json.dumps(result, indent=2))
            return 2
        if args.max_frames is not None and args.max_frames < 1:
            parser.error("--max-frames must be positive")
        if args.resume and not (args.out / "frames.jsonl").exists():
            parser.error("--resume requires an existing frames.jsonl")
        if not args.resume and (args.out / "frames.jsonl").exists():
            parser.error("output already has a frame ledger; use --resume")
        try:
            result = _write_preparation(args.video.resolve(), args.out.resolve(), args.max_frames, args.extract_frames)
        except Exception as error:
            print(f"prepare failed: {error}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PREPARED" else 2
    if args.command == "extract-frames":
        try:
            result = _extract_frame_range(args.video.resolve(), args.preparation.resolve(),
                                          args.out.resolve(), args.start_index, args.count)
        except Exception as error:
            print(f"frame extraction failed: {error}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "score":
        if args.require_hands < 1:
            parser.error("--require-hands must be positive")
        if args.compute_usd_per_hour is not None and (
                not math.isfinite(args.compute_usd_per_hour) or args.compute_usd_per_hour < 0):
            parser.error("--compute-usd-per-hour must be a finite nonnegative number")
        report, code = _run_score(args.manifest.resolve(), args.out.resolve(), args.require_hands,
                                  args.compute_usd_per_hour)
        print(f"{report['verdict']}: {report['scored_hands']} scored / "
              f"{report['admitted_hands']} admitted; report: {args.out}")
        return code
    return 2


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        worker_parser = argparse.ArgumentParser()
        worker_parser.add_argument("mode")
        worker_parser.add_argument("clip")
        worker_args = worker_parser.parse_args(sys.argv[1:])
        raise SystemExit(_worker(Path(worker_args.clip)))
    raise SystemExit(main())
