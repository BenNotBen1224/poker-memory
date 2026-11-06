import numpy as np
import pytest

from poker_memory.reader.suits import read_suit


# Synthetic software fixtures; never benchmark footage or independent references.
GLYPHS = {
    "h": ".#####.#####.\n############.\n#############\n#############\n#############\n#############\n#############\n############.\n.###########.\n.##########..\n..#########..\n...#######...\n....#####....\n....####.....\n......#......",
    "d": ".....##.....\n....####....\n...#####....\n...######...\n..########..\n.##########.\n############\n############\n############\n.##########.\n..########..\n...######...\n....#####...\n....####....\n.....##.....",
    "c": ".....####.....\n....######....\n....######....\n....######....\n.....#####....\n..###.######..\n.############.\n.#############\n##############\n.############.\n..###.##.###..\n......##......\n......##......\n.....####.....\n....######....",
    "s": ".....###.....\n....#####....\n...#######...\n..#########..\n.###########.\n############.\n#############\n#############\n#############\n#############\n############.\n..#########..\n.....##......\n....#####....\n...######....",
}


def crop(suit):
    shape = np.array([[ch == "#" for ch in row] for row in GLYPHS[suit].splitlines()])
    rgb = np.full((17, 16, 3), 225, np.uint8)
    left = (16 - shape.shape[1]) // 2
    interior = rgb[:15, left:left + shape.shape[1]]
    interior[shape] = (195, 15, 30) if suit in ("h", "d") else (35, 35, 35)
    return rgb


@pytest.mark.parametrize("suit", ["h", "d", "c", "s"])
def test_reads_literal_suits_from_stable_pixels(suit):
    first = crop(suit)
    second = first.copy()
    second[0, 0] = 220
    assert read_suit(second, first) == suit


def test_requires_agreement_not_just_one_readable_frame():
    assert read_suit(crop("h"), crop("h")) == "h"
    assert read_suit(crop("h"), crop("d")) is None


@pytest.mark.parametrize("suit", ["h", "d", "c", "s"])
def test_rejects_occlusion_absent_and_dim_glyphs(suit):
    stable = crop(suit)
    assert read_suit(stable, stable) == suit
    occluded = stable.copy()
    occluded[:7] = 225
    assert read_suit(occluded, occluded) is None
    assert read_suit(np.full_like(stable, 225), stable) is None
    assert read_suit(np.full_like(stable, 30), stable) is None
    dim = (stable.astype(float) * 0.55).astype(np.uint8)
    assert read_suit(dim, dim) is None


def test_color_does_not_invent_a_shape():
    unknown = np.full((17, 16, 3), 225, np.uint8)
    unknown[:15, 3:12] = (195, 15, 30)
    assert read_suit(crop("d"), crop("d")) == "d"
    assert read_suit(unknown, unknown) is None
    blended = ((crop("h").astype(np.uint16) + crop("d").astype(np.uint16)) // 2).astype(np.uint8)
    assert read_suit(blended, blended) is None
    wrong_color = crop("h")
    wrong_color[wrong_color.mean(axis=2) < 150] = (35, 35, 35)
    assert read_suit(wrong_color, wrong_color) is None


def test_red_raster_diamond_competes_with_heart_not_black_spade():
    # Software raster reproducing the measured one-pixel-wide diamond variant.
    rows = (".....##......", "....####.....", "....#####....", "...#######...", "..#########..",
            ".##########..", "############.", "#############", "############.", ".##########..",
            "..########...", "...#######...", "....#####....", ".....###.....", "......#......")
    rgb = np.full((16, 16, 3), 225, np.uint8)
    mask = np.array([[c == "#" for c in row] for row in rows])
    rgb[:15, 1:14][mask] = (195, 15, 30)
    assert read_suit(rgb, rgb.copy()) == "d"
    black = rgb.copy()
    black[black.mean(axis=2) < 150] = (35, 35, 35)
    assert read_suit(black, black) is None


def test_black_club_raster_variant_preserves_transition_guards():
    rows = ("....####.....", "...######....", "...######....", "...######....", "....#####....",
            ".#######.##..", "############.", "#############", "#############", "#############",
            ".####.#.####.", "......#......", ".....##......", "....####.....", "...######....")
    rgb = np.full((17, 16, 3), 225, np.uint8)
    mask = np.array([[c == "#" for c in row] for row in rows])
    rgb[:15, 1:14][mask] = (35, 35, 35)
    assert read_suit(rgb, rgb.copy()) == "c"
    assert read_suit(crop("s"), rgb) is None
    occluded = rgb.copy()
    occluded[:7] = 225
    assert read_suit(occluded, occluded) is None
    blended = ((rgb.astype(np.uint16) + crop("s").astype(np.uint16)) // 2).astype(np.uint8)
    assert read_suit(blended, blended) is None
    red = rgb.copy()
    red[red.mean(axis=2) < 150] = (195, 15, 30)
    assert read_suit(red, red) is None


@pytest.mark.parametrize("suit,rows", [
    ("c", ("....#####....", "...######....", "...######....", "...######....",
           "....#####....", ".##########..", "############.", "#############",
           "#############", "#############", ".####.#.####.", ".....##......",
           ".....##......", "....#####....", "....#####....")),
    ("s", ("......###.....", "....######....", "...########...", "..##########..",
           "..###########.", ".############.", ".############.", ".#############",
           "##############", ".#############", ".############.", "..######.###..",
           "......##......", ".....####.....", "....######....")),
])
def test_black_raster_caps_and_lobes_reject_incompatible_candidate_shapes(suit, rows):
    mask = np.array([[pixel == "#" for pixel in row] for row in rows])
    rgb = np.full((17, 16, 3), 225, np.uint8)
    left = (16 - mask.shape[1]) // 2
    rgb[:15, left:left + mask.shape[1]][mask] = (35, 35, 35)
    assert read_suit(rgb, rgb.copy()) == suit
    assert read_suit(rgb, crop("h")) is None
    occluded = rgb.copy()
    occluded[:7] = 225
    assert read_suit(occluded, occluded) is None
    faded = ((rgb.astype(np.uint16) + 225) // 2).astype(np.uint8)
    assert read_suit(faded, faded) is None


def test_topology_alone_cannot_create_a_black_suit():
    rgb = crop("s")
    rgb[7:11, 1:15] = 225
    assert read_suit(rgb, rgb.copy()) is None
    ambiguous = ((crop("c").astype(np.uint16) + crop("s").astype(np.uint16)) // 2).astype(np.uint8)
    assert read_suit(ambiguous, ambiguous) is None


@pytest.mark.parametrize("rgb", [np.zeros((20, 20, 3), np.uint8), np.zeros((17, 16), np.uint8),
                                 np.zeros((17, 16, 3), float)])
def test_unsupported_crop_contract_raises(rgb):
    with pytest.raises(ValueError, match="native uint8 RGB"):
        read_suit(rgb, crop("s"))
