import sqlite3

import pytest

from conftest import hand
from poker_memory.store import Conflict, NotFound, StaleRevision, Store

KEY = "2026-10-31-home-game"


@pytest.fixture()
def store(tmp_path, doc):
    s = Store(tmp_path / "pm.db")
    s.import_session(doc)
    return s


def with_raise(record, to):
    events = [e.replace("s3 raise 75", f"s3 raise {to}") for e in record.events]
    return record.model_copy(update={"events": events, "result": None})


def test_import_replays_every_hand_and_survives_restart(tmp_path, doc):
    Store(tmp_path / "pm.db").import_session(doc)
    reopened = Store(tmp_path / "pm.db")
    hands = reopened.hands(KEY)
    assert [(h.index, h.revision, h.active_revision, h.status) for h in hands] == [
        (7, 1, 1, "verified"), (8, 1, 1, "verified"), (9, 1, 1, "verified")]
    assert reopened.session(KEY).status == "live"
    assert hands[1].replay.payoffs == {1: 0, 2: 96, 3: -216, 4: 120}


def test_reimport_is_rejected(store, doc):
    with pytest.raises(Conflict):
        store.import_session(doc)


def test_end_session(store):
    assert store.end_session(KEY).status == "ended"
    with pytest.raises(NotFound):
        store.end_session("nope")


def test_valid_correction_becomes_active_and_keeps_history(store, doc):
    fixed = store.correct(KEY, 7, with_raise(hand(doc, 7), 95), base_revision=1, note="raise was 95")
    assert (fixed.revision, fixed.active_revision, fixed.status) == (2, 2, "verified")
    assert store.hand(KEY, 7).revision == 2
    assert "s3 raise 75 @01:55.4" in store.hand(KEY, 7, revision=1).record.events
    assert [r.note for r in store.revisions(KEY, 7)] == ["imported", "raise was 95"]


def test_invalid_correction_is_kept_but_not_activated(store, doc):
    bad = store.correct(KEY, 7, with_raise(hand(doc, 7), 21), base_revision=1, note="typo")
    assert (bad.revision, bad.active_revision, bad.status) == (2, 1, "invalid")
    assert bad.replay.issues[0].code == "illegal_amount"
    current = store.hand(KEY, 7)
    assert (current.revision, current.status) == (1, "verified")
    # The next correction must build on the latest revision (2), even though 1 is active.
    with pytest.raises(StaleRevision):
        store.correct(KEY, 7, with_raise(hand(doc, 7), 95), base_revision=1, note="retry")
    assert store.correct(KEY, 7, with_raise(hand(doc, 7), 95), base_revision=2, note="retry").active_revision == 3


def test_correction_must_target_the_same_hand(store, doc):
    with pytest.raises(Conflict):
        store.correct(KEY, 9, hand(doc, 7), base_revision=1, note="wrong hand")


def test_revisions_cannot_be_edited_or_deleted(store, tmp_path):
    db = sqlite3.connect(tmp_path / "pm.db")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute("UPDATE hand_revisions SET note = 'x'")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute("DELETE FROM hand_revisions")
