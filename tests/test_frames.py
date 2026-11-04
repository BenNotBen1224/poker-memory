"""scripts/frames.py must return the picture from the second it is asked for."""

import subprocess
import sys
from pathlib import Path

import pytest

imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
Image = pytest.importorskip("PIL.Image")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import frames  # noqa: E402

SECONDS = 12
FPS = 10


@pytest.fixture(scope="module")
def staircase(tmp_path_factory) -> Path:
    """A 12 s clip whose gray level is 20 x the whole second it is in, so any frame says when it came from."""
    path = tmp_path_factory.mktemp("clip") / "staircase.mp4"
    source = f"color=c=black:s=64x64:r={FPS}:d={SECONDS},format=gray,geq=lum='20*floor(N/{FPS})'"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i", source,
         "-c:v", "mpeg4", "-q:v", "2", str(path)],
        check=True,
    )
    return path


def second_of(path: Path) -> int:
    return round(Image.open(path).convert("L").getpixel((32, 32)) / 20)


def test_one_frame_comes_from_the_second_asked_for(staircase, tmp_path):
    [(at, path)] = frames.grab(staircase, 5.5, 5.5, 10.0, 64, None, tmp_path)
    assert (at, second_of(path)) == (5.5, 5)


def test_each_frame_of_a_sheet_comes_from_its_own_label(staircase, tmp_path):
    found = frames.grab(staircase, 2.5, 8.5, 2.0, 64, None, tmp_path)
    assert [(at, second_of(path)) for at, path in found] == [(2.5, 2), (4.5, 4), (6.5, 6), (8.5, 8)]
