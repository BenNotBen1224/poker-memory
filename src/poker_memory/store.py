"""SQLite storage. Hand revisions are append-only (enforced by triggers); a hand's
active revision only ever points at a verified replay."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel

from .engine import Replay, replay
from .model import HandRecord, SessionDoc, SessionInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  key TEXT PRIMARY KEY,
  info TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('live', 'ended')),
  created_at TEXT NOT NULL,
  ended_at TEXT
);
CREATE TABLE IF NOT EXISTS hands (
  id INTEGER PRIMARY KEY,
  session_key TEXT NOT NULL REFERENCES sessions(key),
  hand_index INTEGER NOT NULL,
  active_revision INTEGER,
  latest_revision INTEGER NOT NULL,
  UNIQUE (session_key, hand_index)
);
CREATE TABLE IF NOT EXISTS hand_revisions (
  hand_id INTEGER NOT NULL REFERENCES hands(id),
  revision INTEGER NOT NULL,
  record TEXT NOT NULL,
  replay TEXT NOT NULL,
  status TEXT NOT NULL,
  engine_version TEXT NOT NULL,
  note TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (hand_id, revision)
);
CREATE TRIGGER IF NOT EXISTS hand_revisions_no_update BEFORE UPDATE ON hand_revisions
BEGIN SELECT RAISE(ABORT, 'hand revisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS hand_revisions_no_delete BEFORE DELETE ON hand_revisions
BEGIN SELECT RAISE(ABORT, 'hand revisions are immutable'); END;
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY,
  session_key TEXT NOT NULL REFERENCES sessions(key),
  state TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY,
  conversation_id INTEGER NOT NULL REFERENCES conversations(id),
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analysis_runs (
  id INTEGER PRIMARY KEY,
  turn_id INTEGER NOT NULL REFERENCES turns(id),
  tool TEXT NOT NULL,
  session_key TEXT NOT NULL,
  hand_index INTEGER,
  revision INTEGER,
  input TEXT NOT NULL,
  output TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


class StaleRevision(Conflict):
    pass


class StoredSession(BaseModel):
    info: SessionInfo
    status: str
    created_at: str
    ended_at: str | None


class StoredHand(BaseModel):
    session_key: str
    index: int
    revision: int
    active_revision: int | None
    latest_revision: int
    status: str
    note: str
    created_at: str
    record: HandRecord
    replay: Replay


class ToolRun(BaseModel):
    tool: str
    hand_index: int | None = None
    revision: int | None = None
    input: dict[str, Any]
    output: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        try:
            db.executescript(SCHEMA)
        finally:
            db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One connection per unit of work; BEGIN IMMEDIATE makes check-then-write atomic."""
        db = sqlite3.connect(self.path, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    # sessions

    def import_session(self, doc: SessionDoc) -> StoredSession:
        key = doc.session.key
        indexes = [h.index for h in doc.hands]
        if len(set(indexes)) != len(indexes):
            raise Conflict("duplicate hand numbers in the import")
        with self._tx() as db:
            if db.execute("SELECT 1 FROM sessions WHERE key = ?", (key,)).fetchone():
                raise Conflict(f"session {key!r} already exists")
            db.execute("INSERT INTO sessions (key, info, status, created_at) VALUES (?, ?, 'live', ?)",
                       (key, doc.session.model_dump_json(), _now()))
            for record in doc.hands:
                hand_id = db.execute(
                    "INSERT INTO hands (session_key, hand_index, latest_revision) VALUES (?, ?, 0)",
                    (key, record.index)).lastrowid
                self._append(db, hand_id, 0, record, doc.session, "imported")
        return self.session(key)

    def sessions(self) -> list[StoredSession]:
        with self._tx() as db:
            rows = db.execute("SELECT * FROM sessions ORDER BY created_at DESC, key").fetchall()
        return [_session(r) for r in rows]

    def session(self, key: str) -> StoredSession:
        with self._tx() as db:
            row = db.execute("SELECT * FROM sessions WHERE key = ?", (key,)).fetchone()
        if not row:
            raise NotFound(f"no session {key!r}")
        return _session(row)

    def end_session(self, key: str) -> StoredSession:
        with self._tx() as db:
            if not db.execute("SELECT 1 FROM sessions WHERE key = ?", (key,)).fetchone():
                raise NotFound(f"no session {key!r}")
            db.execute("UPDATE sessions SET status = 'ended', ended_at = coalesce(ended_at, ?) WHERE key = ?",
                       (_now(), key))
        return self.session(key)

    # hands

    def hands(self, key: str) -> list[StoredHand]:
        """Each hand at its active revision, or its latest one if nothing verified yet."""
        self.session(key)
        with self._tx() as db:
            rows = db.execute(
                "SELECT h.*, r.* FROM hands h JOIN hand_revisions r ON r.hand_id = h.id"
                " AND r.revision = coalesce(h.active_revision, h.latest_revision)"
                " WHERE h.session_key = ? ORDER BY h.hand_index", (key,)).fetchall()
        return [_hand(r) for r in rows]

    def hand(self, key: str, index: int, revision: int | None = None) -> StoredHand:
        with self._tx() as db:
            row = db.execute(
                "SELECT h.*, r.* FROM hands h JOIN hand_revisions r ON r.hand_id = h.id"
                " AND r.revision = coalesce(?, h.active_revision, h.latest_revision)"
                " WHERE h.session_key = ? AND h.hand_index = ?", (revision, key, index)).fetchone()
        if not row:
            raise NotFound(f"no hand {index} (revision {revision or 'current'}) in session {key!r}")
        return _hand(row)

    def revisions(self, key: str, index: int) -> list[StoredHand]:
        with self._tx() as db:
            rows = db.execute(
                "SELECT h.*, r.* FROM hands h JOIN hand_revisions r ON r.hand_id = h.id"
                " WHERE h.session_key = ? AND h.hand_index = ? ORDER BY r.revision", (key, index)).fetchall()
        if not rows:
            raise NotFound(f"no hand {index} in session {key!r}")
        return [_hand(r) for r in rows]

    def correct(self, key: str, index: int, record: HandRecord, base_revision: int, note: str) -> StoredHand:
        """Append a new revision. It becomes active only if it replays as verified;
        an invalid correction is kept for review and the old active revision stays."""
        if record.index != index:
            raise Conflict(f"record is hand {record.index}, not hand {index}")
        info = self.session(key).info
        with self._tx() as db:
            row = db.execute("SELECT id, latest_revision FROM hands WHERE session_key = ? AND hand_index = ?",
                             (key, index)).fetchone()
            if not row:
                raise NotFound(f"no hand {index} in session {key!r}")
            if row["latest_revision"] != base_revision:
                raise StaleRevision(
                    f"hand {index} is at revision {row['latest_revision']}; reload before correcting revision {base_revision}")
            revision = self._append(db, row["id"], row["latest_revision"], record, info, note.strip() or "correction")
        return self.hand(key, index, revision)

    @staticmethod
    def _append(db, hand_id: int, latest: int, record: HandRecord, info: SessionInfo, note: str) -> int:
        result = replay(record, info.stakes)
        revision = latest + 1
        db.execute(
            "INSERT INTO hand_revisions (hand_id, revision, record, replay, status, engine_version, note, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (hand_id, revision, record.model_dump_json(), result.model_dump_json(), result.status,
             result.engine_version, note, _now()))
        active = ", active_revision = ?" if result.status == "verified" else ""
        db.execute(f"UPDATE hands SET latest_revision = ?{active} WHERE id = ?",
                   (revision, revision, hand_id) if active else (revision, hand_id))
        return revision

    # conversations

    def new_conversation(self, key: str) -> int:
        self.session(key)
        with self._tx() as db:
            return db.execute("INSERT INTO conversations (session_key, state, created_at) VALUES (?, '{}', ?)",
                              (key, _now())).lastrowid

    def conversation(self, conversation_id: int) -> tuple[str, dict]:
        with self._tx() as db:
            row = db.execute("SELECT session_key, state FROM conversations WHERE id = ?",
                             (conversation_id,)).fetchone()
        if not row:
            raise NotFound(f"no conversation {conversation_id}")
        return row["session_key"], json.loads(row["state"])

    def save_turn(self, conversation_id: int, question: str, answer: dict, state: dict, runs: list[ToolRun]) -> int:
        with self._tx() as db:
            key = db.execute("SELECT session_key FROM conversations WHERE id = ?", (conversation_id,)).fetchone()[0]
            turn_id = db.execute("INSERT INTO turns (conversation_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                                 (conversation_id, question, json.dumps(answer), _now())).lastrowid
            db.execute("UPDATE conversations SET state = ? WHERE id = ?", (json.dumps(state), conversation_id))
            for run in runs:
                db.execute(
                    "INSERT INTO analysis_runs (turn_id, tool, session_key, hand_index, revision, input, output, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (turn_id, run.tool, key, run.hand_index, run.revision, json.dumps(run.input),
                     json.dumps(run.output), _now()))
        return turn_id

    def turns(self, conversation_id: int) -> list[dict]:
        with self._tx() as db:
            rows = db.execute("SELECT id, question, answer, created_at FROM turns WHERE conversation_id = ? ORDER BY id",
                              (conversation_id,)).fetchall()
        return [{"id": r["id"], "question": r["question"], "answer": json.loads(r["answer"]),
                 "created_at": r["created_at"]} for r in rows]

    def tool_runs(self, turn_id: int) -> list[ToolRun]:
        with self._tx() as db:
            rows = db.execute("SELECT * FROM analysis_runs WHERE turn_id = ? ORDER BY id", (turn_id,)).fetchall()
        return [ToolRun(tool=r["tool"], hand_index=r["hand_index"], revision=r["revision"],
                        input=json.loads(r["input"]), output=json.loads(r["output"])) for r in rows]


def _session(row: sqlite3.Row) -> StoredSession:
    return StoredSession(info=SessionInfo.model_validate_json(row["info"]), status=row["status"],
                         created_at=row["created_at"], ended_at=row["ended_at"])


def _hand(row: sqlite3.Row) -> StoredHand:
    return StoredHand(
        session_key=row["session_key"], index=row["hand_index"], revision=row["revision"],
        active_revision=row["active_revision"], latest_revision=row["latest_revision"], status=row["status"],
        note=row["note"], created_at=row["created_at"],
        record=HandRecord.model_validate_json(row["record"]), replay=Replay.model_validate_json(row["replay"]))
