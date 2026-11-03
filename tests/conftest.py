from pathlib import Path

import pytest

from poker_memory.model import SessionDoc

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture()
def doc() -> SessionDoc:
    return SessionDoc.load(EXAMPLES / "home_game.yaml")


def hand(doc: SessionDoc, index: int):
    return next(h for h in doc.hands if h.index == index)
