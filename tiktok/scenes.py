"""Everything drawn on top of the video, with the sound cues that go with it."""
from __future__ import annotations

import gfx
import timeline as tl
from gfx import BLACK, RED, WHITE, YELLOW
from overlays import DARK, ChipRow, Circle, Flip, Hud, Spr, Captions, build_pages, voice_words
from timeline import S, T

TITLE_TOP = 200


class Scene:
    def __init__(self):
        self.items, self.cues = [], []

    def cue(self, t, kind, gain_db):
        self.cues.append((t, kind, gain_db))

    def title(self, text, t0, t1, anim="pop", cy=None):
        spr = gfx.box_text(text, 58, "Bold", 960)
        self.items.append(Spr(spr, 540, cy, t0, t1, anim, top=None if cy else TITLE_TOP))
        if anim == "pop":
            self.cue(t0, "pop", -20)

    def add(self, spr, cx, cy, t0, t1, anim="pop", sound="pop", gain=-20, **kw):
        self.items.append(Spr(spr, cx, cy, t0, t1, anim, **kw))
        if sound:
            self.cue(t0, sound, gain)

    def flip(self, back, front, cx, cy, t0, tflip, t1):
        self.items.append(Flip(back, front, cx, cy, t0, tflip, t1))
        self.cue(t0, "flick", -10)
        self.cue(tflip, "flick", -10)


def dark_pill(text, size=42, rot=0.0):
    return gfx.pill(text, size, bg=DARK, fg=WHITE, rot=rot)


def build() -> Scene:
    sc = Scene()
    end = tl.DURATION
    back = lambda w: gfx.card("", "", w, back=True)

    # hook + rewind
    sc.title("he folded *three kings* 😳", T("hook"), T("hook", 1.6))
    sc.title("he folded *three kings* 😳\nonly 2 hands could beat him", T("hook", 1.6), T("rewind"), anim="none")
    sc.add(dark_pill("🙈 no spoilers", 40, rot=6), 185, 1150, T("hook", 0.6), T("rewind"))
    sc.add(gfx.stroke_text("⏪ rewind", 96), 540, 940, T("rewind"), T("ian"), anim="slam", sound=None)

    # who is who
    sc.title("this is ian 👋\nhe has *pocket kings* 👑", T("ian"), T("joe"))
    sc.add(dark_pill("WPT final table · 5 left"), 540, 470, T("ian", 0.2), T("joe"), sound=None)
    kd, ks = gfx.card("K", "♦", 140), gfx.card("K", "♠", 140)
    sc.flip(back(140), kd, 250, 1100, T("ian", 0.35), S("ian", 24.44), T("joe"))
    sc.flip(back(140), ks, 410, 1100, T("ian", 0.47), S("ian", 24.44) + 0.12, T("joe"))
    sc.add(gfx.pill("cowboys = kings 🤠", 44), 700, 1150, S("ian", 24.9), T("joe"))
    sc.title("vs *joe mckeehen* 🏆", T("joe"), T("flop"), cy=255)
    sc.add(dark_pill("2015 world series of poker champ", 40), 540, 362, T("joe", 0.35), T("flop"), sound=None)
    for i, x in enumerate((250, 410)):
        sc.add(back(140), x, 1100, T("joe", 0.3 + 0.12 * i), T("flop"), sound="flick", gain=-10)

    # the hand
    board = [("A", "♥", T("flop", 0.35)), ("7", "♠", T("flop", 0.55)), ("5", "♥", T("flop", 0.75)),
             ("J", "♣", T("turn", 0.4)), ("K", "♣", T("river", 1.4))]
    sc.items.append(Hud(T("flop"), T("tank_hf"), board))
    for _, _, tb in board:
        sc.cue(tb, "flick", -10)
    sc.title("flop: ian bets, joe calls", T("flop"), T("turn"))
    sc.title("turn: joe bets, ian calls", T("turn"), T("river"))
    sc.title("river… 👀", T("river"), T("river", 1.4))
    sc.title("river K♣ = *three kings* for ian 🔥", T("river", 1.4), T("bet_a"))
    sc.title("ian bets *800K* 💰", T("bet_a"), T("shove"))
    sc.title("joe thinks… 🤔", T("shove"), T("shove", 2.45))
    sc.title("joe goes *all in* 💀", T("shove", 2.45), T("wow"))
    sc.add(gfx.stroke_text("*ALL IN* 💀", 150, max_width=1000), 540, 900, T("shove", 2.45), T("wow"),
           anim="slam", sound=None)
    sc.add(dark_pill("ian rn: 😶", 44), 540, 470, T("react", 0.1), T("wow"))
    sc.title("ian has to call *2.14 million* 😬", T("wow"), T("stacks_a"))

    # only two hands beat him
    sc.title("only *2 hands* could beat him 👇", T("stacks_a"), T("poll"))
    for i, (rank, x) in enumerate((("A", 215), ("A", 345), ("Q", 595), ("10", 725))):
        sc.add(gfx.card(rank, "", 120), x, 1010, T("stacks_a", 0.25 + 0.12 * i + (0.2 if i > 1 else 0)), T("poll"),
               sound="flick", gain=-10)
    sc.add(gfx.stroke_text("or", 60), 470, 1010, T("stacks_a", 0.55), T("poll"), sound=None)
    sc.add(dark_pill("three aces", 38), 280, 1140, T("stacks_a", 0.5), T("poll"), sound=None)
    sc.add(dark_pill("a straight", 38), 660, 1140, T("stacks_a", 0.95), T("poll"), sound=None)

    # poll
    sc.title("would you call? 🤔", T("poll"), T("tank_hf"))
    sc.add(gfx.box_text("comment *CALL* or *FOLD* 👇", 60, "Black", 900, bg=RED, fg=WHITE, hl=YELLOW, rot=-3),
           540, 820, T("poll", 0.25), T("tank_hf"), anim="slam", sound=None)
    sc.add(gfx.dots(3, 10), 540, 1010, T("poll", 0.7), T("tank_hf"), sound=None)
    sc.add(gfx.stroke_text("calling only needs to win\n*3 out of 10* 🎯", 56, max_width=900), 540, 1130,
           T("poll", 0.9), T("tank_hf"), anim="up", sound=None)
    t, hi = T("poll", 0.25), True
    while t < T("tank_hf") - 0.1:
        sc.cue(t, "tick_hi" if hi else "tick_lo", -10)
        t, hi = t + 0.5, not hi

    # the tank
    sc.title("ian thinks… and thinks… 🤔", T("tank_hf"), T("fold_b", 0.78))
    fills = [T("tank_hf", 0.5), T("tank_hf", 2.0), T("tank_nw", 0.9), T("tank_30a", 0.6), T("tank_30a", 2.3),
             T("tank_kd", 0.5)]
    sc.items.append(ChipRow(T("tank_hf"), T("fold_b", 0.78), fills))
    for tf in fills:
        sc.cue(tf, "chips", -10)
    sc.add(gfx.pill("6 time chips = *3 extra minutes* ⏱", 40), 540, 580, T("tank_kd", 0.9), T("fold_b", 0.78))
    sc.add(dark_pill("joe waiting like 😐", 44), 540, 1180, T("tank_joe", 0.15), T("tank_30a"))

    # the fold
    sc.title("he actually did it 😱", T("fold_b", 0.78), T("regret"))
    sc.add(gfx.stroke_text("HE *FOLDED* 😱", 140, max_width=1000), 540, 900, T("fold_b", 0.78), T("fold_b", 2.4),
           anim="slam", sound=None)
    sc.title("then joe reportedly told him 👇", T("regret"), T("build"))
    sc.add(gfx.box_text("“you're going to *regret* that fold for the rest of your life” 😬", 56, "Bold", 900),
           540, 940, T("regret", 0.4), T("build"), sound=None)
    sc.add(dark_pill("(via WPT.com)", 36), 540, 1150, T("regret", 0.9), T("build"), anim="fade", sound=None)
    # the answer lands when the 10 shows its face (flip at +0.37, face from half of the 0.22 s flip)
    answer = T("reveal", 0.37 + 0.11)
    sc.title("so… what did joe have? 👀", T("build"), answer)

    # the reveal
    sc.title("joe had *Q-10* 😱", answer, T("reveal", 2.6))
    sc.flip(back(170), gfx.card("Q", "♣", 170), 455, 650, T("reveal"), T("reveal", 0.25), T("reveal", 2.6))
    sc.flip(back(170), gfx.card("10", "♦", 170), 635, 650, T("reveal"), T("reveal", 0.37), T("reveal", 2.6))
    sc.items.append(Circle(447, 1098, 440, 120, T("reveal", 1.0), 0.5, T("read")))
    sc.title("straight = *best possible hand* 💀", T("reveal", 2.6), T("read"))
    row = [("A", "♥", True), ("K", "♣", True), ("Q", "♣", False), ("J", "♣", True), ("10", "♦", False)]
    for i, ((rank, suit, dim), x) in enumerate(zip(row, (276, 408, 540, 672, 804))):
        sc.add(gfx.card(rank, suit, 118, dim=dim), x, 650, T("reveal", 2.6 + 0.06 * i), T("read"),
               sound="flick", gain=-12)

    # aftermath
    sc.title("ian guessed it: *Q-10* 🎯", T("read"), T("twist"))
    sc.add(dark_pill("(he guessed hearts lol)", 40), 540, 470, T("read", 0.9), T("twist"))
    sc.title("plot twist 🤯", T("twist"), T("outro"))
    sc.add(gfx.box_text("ian later said it “*still might have been a terrible fold*”", 54, "Bold", 900),
           540, 900, T("twist", 0.6), T("outro"))
    sc.add(gfx.stroke_text("to call, he only needed\nto win *3 in 10* 🧮", 54, max_width=900), 540, 1190,
           T("twist", 2.4), T("outro"), anim="up", sound=None)
    sc.title("ian still finished *2nd* 🥈 ($201K)", T("outro"), end + 1)
    sc.add(dark_pill("joe finished 3rd 🥉", 42), 540, 470, T("outro", 0.6), end + 1)
    sc.add(gfx.box_text("would *YOU* have folded? 👇", 60, "Black", 900, bg=RED, fg=WHITE, hl=YELLOW, rot=-3),
           540, 1150, T("outro", 1.2), end + 1, anim="slam", sound=None)

    sc.items.append(Captions(build_pages(voice_words())))
    sc.items.append(Spr(gfx.text_line("🎥 World Poker Tour · CC BY 3.0", 26, "Bold", WHITE, shadow=3),
                        490, 1508, 0, end + 1, anim="none", alpha=0.8))
    return sc


def draw(sc: Scene, canvas, t: float) -> None:
    for item in sc.items:
        item.draw(canvas, t)
