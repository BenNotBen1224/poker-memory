"""HTTP API. Thin: validation at the edge, everything else in PokerMemory."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .gto import warm
from .memory import Answer, PokerMemory
from .model import HandRecord, SessionDoc
from .review import KEY as REVIEW_KEY, Lens, Review, build_review, verify_case
from .search import combo_name, hero_cards, summary
from .store import Conflict, NotFound, StoredHand


class AskBody(BaseModel):
    question: str
    session_key: str | None = None
    conversation_id: int | None = None


class CorrectionBody(BaseModel):
    base_revision: int
    note: str = ""
    record: HandRecord


def _hand_row(h: StoredHand) -> dict:
    return {"index": h.index, "revision": h.revision, "active_revision": h.active_revision,
            "latest_revision": h.latest_revision, "status": h.status, "summary": summary(h),
            "hero_cards": hero_cards(h), "combo": combo_name(hero_cards(h)),
            "dealt_at": h.record.dealt_at, "issues": [i.model_dump() for i in h.replay.issues]}


def create_app(data_dir: str | Path) -> FastAPI:
    memory = PokerMemory.open(data_dir)
    warm()
    app = FastAPI(title="Poker Memory", version="0.1.0")
    app.state.memory = memory

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, e: NotFound):
        return JSONResponse({"detail": str(e)}, status_code=404)

    @app.exception_handler(Conflict)
    async def _conflict(_: Request, e: Conflict):
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.exception_handler(ValueError)
    async def _bad(_: Request, e: ValueError):
        return JSONResponse({"detail": str(e)}, status_code=422)

    @app.get("/api/sessions")
    def sessions():
        out = []
        for s in memory.store.sessions():
            hands = memory.store.hands(s.info.key)
            out.append({**s.info.model_dump(), "status": s.status, "ended_at": s.ended_at,
                        "hand_count": len(hands), "verified": sum(h.status == "verified" for h in hands),
                        "video_url": f"/api/sessions/{s.info.key}/video" if s.info.video else None})
        return out

    @app.get("/api/demo/wpt/review", response_model=Review, response_model_exclude_none=True)
    def demo_review(lens: Lens = "decision"):
        session = memory.store.session(REVIEW_KEY)
        hand = memory.store.hand(REVIEW_KEY, 1)
        return build_review(session, hand, memory.video_path(REVIEW_KEY), lens)

    @app.get("/api/demo/wpt/video")
    def demo_video():
        session = memory.store.session(REVIEW_KEY)
        hand = memory.store.hand(REVIEW_KEY, 1)
        path = memory.video_path(REVIEW_KEY)
        verify_case(session, hand, path)
        return FileResponse(path, media_type="video/webm")

    @app.post("/api/sessions", status_code=201)
    def import_session(doc: SessionDoc):
        if doc.session.video:
            raise HTTPException(422, "upload sessions without video here; attach a consented or openly licensed video with the CLI")
        return memory.import_session(doc)

    @app.post("/api/sessions/{key}/end")
    def end_session(key: str):
        return memory.end_session(key)

    @app.get("/api/sessions/{key}/hands")
    def hands(key: str):
        return [_hand_row(h) for h in memory.store.hands(key)]

    @app.get("/api/sessions/{key}/hands/{index}")
    def hand(key: str, index: int, revision: int | None = None):
        return memory.store.hand(key, index, revision)

    @app.get("/api/sessions/{key}/hands/{index}/revisions")
    def revisions(key: str, index: int):
        return [{"revision": r.revision, "status": r.status, "note": r.note, "created_at": r.created_at,
                 "issues": [i.model_dump() for i in r.replay.issues]} for r in memory.store.revisions(key, index)]

    @app.put("/api/sessions/{key}/hands/{index}")
    def correct(key: str, index: int, body: CorrectionBody):
        return memory.correct_hand(key, index, body.base_revision, body.record, body.note)

    @app.post("/api/ask", response_model=Answer)
    def ask(body: AskBody):
        return memory.ask(body.question, body.session_key, body.conversation_id)

    @app.get("/api/conversations/{conversation_id}")
    def conversation(conversation_id: int):
        key, _ = memory.store.conversation(conversation_id)
        return {"session_key": key, "turns": memory.store.turns(conversation_id)}

    @app.get("/api/sessions/{key}/video")
    def video(key: str):
        path = memory.video_path(key)
        if not path or not path.is_file():
            raise NotFound(f"session {key!r} has no video")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "video/mp4")

    return app


def app_from_env() -> FastAPI:
    return create_app(os.environ.get("POKER_MEMORY_DATA", "data"))
