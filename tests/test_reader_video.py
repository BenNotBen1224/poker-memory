from fractions import Fraction

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from poker_memory.reader.video import Clip, keyframe, stretch  # noqa: E402


@pytest.fixture(params=["libx264", "libvpx-vp9"])
def encoded_video(tmp_path, request):
    codec = request.param
    path = tmp_path / ("delayed.mp4" if codec == "libx264" else "immediate.webm")
    with av.open(str(path), "w") as box:
        stream = box.add_stream(codec, rate=24)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.gop_size = 12
        if codec == "libx264":
            stream.codec_context.max_b_frames = 2
            stream.options = {"x264-params": "scenecut=0:open-gop=0:b-adapt=0"}
        else:
            stream.options = {"lag-in-frames": "0", "auto-alt-ref": "0"}
        for i in range(41):
            pixels = np.full((48, 64, 3), (i * 5, 200 - i * 4, i * 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = i + 7
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)
    return path, codec


def _ordinary_decode(path):
    with av.open(str(path)) as box:
        return [
            (float(frame.pts * frame.time_base), frame.to_ndarray(format="yuv420p").copy())
            for frame in box.decode(video=0)
        ]


def _assert_picture(picture, expected):
    time, planes = expected
    height, width = picture.y.shape
    assert picture.time == time
    assert np.array_equal(picture.y, planes[:height])
    assert np.array_equal(picture.u, planes[height:height + height // 4].reshape(height // 2, width // 2))
    assert np.array_equal(picture.v, planes[height + height // 4:].reshape(height // 2, width // 2))


@pytest.mark.parametrize("threads", [1, 4])
def test_gop_continuation_matches_complete_decode_including_reordered_tail(encoded_video, threads):
    path, codec_name = encoded_video
    expected = _ordinary_decode(path)
    assert len(expected) == 41
    assert expected[0][0] > 0
    clip = Clip.read(path)
    assert len(clip.keys) >= 3
    if codec_name == "libx264":
        from poker_memory.reader.video import _decoder

        assert _decoder(clip, threads).decode(clip.packets[0]) == []
        assert any(packet.pts != packet.dts for packet in clip.packets)
    pictures = []
    for k in range(len(clip.keys)):
        decoder, first = keyframe(clip, k, threads)
        pictures.append(first)
        pictures.extend(stretch(clip, k, decoder, every=1))
    assert len(pictures) == 41
    for actual, want in zip(pictures, expected, strict=True):
        _assert_picture(actual, want)


def test_sampling_is_by_presented_frame_and_keeps_final_tail(encoded_video):
    path, _ = encoded_video
    clip = Clip.read(path)
    expected = _ordinary_decode(path)
    for k, start in enumerate(clip.keys):
        lower = clip.times[start]
        upper = clip.times[clip.keys[k + 1]] if k + 1 < len(clip.keys) else float("inf")
        group = [frame for frame in expected if lower <= frame[0] < upper]
        assert group
        decoder, first = keyframe(clip, k)
        _assert_picture(first, group[0])
        selected = list(stretch(clip, k, decoder, every=5))
        tail = {len(group) - 1} if k == len(clip.keys) - 1 and len(group) > 1 else set()
        indexes = sorted(set(range(5, len(group), 5)) | tail)
        assert len(selected) == len(indexes)
        for picture, i in zip(selected, indexes, strict=True):
            _assert_picture(picture, group[i])
