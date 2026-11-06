"""Build preview.html: the finished cut, its chapters, the checks it passed, and a source for every on-screen fact.

Re-runnable:  python make_preview.py   ->  preview.html, out/poster.jpg, out/thumbs/*.jpg
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from html import escape
from pathlib import Path
from string import Template

import av

from timeline import DURATION, S, T

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MP4 = OUT / "steinman-fold-tiktok.mp4"
THUMBS = OUT / "thumbs"
VIDEO_URL = "https://github.com/BenNotBen1224/poker-memory/releases/download/v0.1.0/steinman-fold-tiktok.mp4"
REVEAL = T("reveal")
X0, X1 = 40, 994  # shared SVG x-range so the timeline and the caption chart line up

CHAPTERS = [  # label, first segment, thumbnail time in the cut
    ("hook", "hook", 1.2),
    ("meet ian", "ian", 6.8),
    ("meet joe", "joe", 9.2),
    ("the board", "flop", 17.7),
    ("bet + shove", "bet_a", 26.8),
    ("the big think", "wow", 45.0),
    ("he folds", "fold_a", 52.4),
    ("joe's jab", "regret", 58.5),
    ("the reveal", "reveal", 63.5),
    ("his guess", "read", 67.5),
    ("plot twist", "twist", 72.9),
    ("the ending", "outro", 77.0),
]

COMMONS = "https://commons.wikimedia.org/wiki/File:Incredibly_Unbelievable_Fold_on_the_World_Poker_Tour.webm"
FOOTAGE = ("the footage", COMMONS)
WPT = ("WPT.com", "https://www.worldpokertour.com/article/steinman-fold/")
POKERNEWS = ("PokerNews", "https://www.pokernews.com/news/2018/03/ian-steinman-kings-set-fold-mckeehen-30175.htm")
RESULTS = ("WPT results", "https://www.worldpokertour.com/live-update/hands-171-david-larson-wins-wpt-rolling-thunder-"
           "295128-ian-steinman-out-2nd-201428-main-tour-wpt-rolling-thunder-season-2017-2018-4-25000750000-150000")
WSOP = ("WSOP.com", "https://www.wsop.com/news/joe-mckeehen-wins-2015-wsop-main-event-championship/")
MATH = ("math", None)

FACTS = [  # what the video says, why it holds, sources
    ("ian has pocket kings", "K♦ K♠ shown on the broadcast", [FOOTAGE, POKERNEWS]),
    ("joe = 2015 world series of poker champ", "won the 2015 Main Event", [WSOP]),
    ("WPT final table · 5 left", "five players remained", [POKERNEWS]),
    ("flop: ian bets, joe calls · turn: joe bets, ian calls", "150K bet + call, then 370K bet + call", [POKERNEWS]),
    ("river K♣ = three kings", "board A♥ 5♥ 7♠ J♣ K♣", [FOOTAGE, POKERNEWS]),
    ("ian bets 800K → joe all in → 2.14M to call", "800K lead; joe's shove is 2.94M total", [FOOTAGE, POKERNEWS]),
    ("only 2 hands could beat him", "no pair and at most 2 of a suit on the board → only A-A or Q-10", [WPT, MATH]),
    ("calling only needs to win 3 in 10", "pay 2.14M for a 7.29M pot = 29%", [MATH]),
    ("6 time chips = 3 extra minutes", "“six of his 30-second time chips”", [WPT]),
    ("joe reportedly said “you're going to regret that fold…”", "the rail heard it and told WPT.com", [WPT]),
    ("joe had Q-10", "Q♣ 10♦ shown on the broadcast", [FOOTAGE, POKERNEWS]),
    ("straight = best possible hand", "no flush or full house is possible on this board", [MATH]),
    ("ian guessed Q-10 (hearts lol)", "“queen-ten of hearts”", [WPT]),
    ("“still might have been a terrible fold”", "ian, about an hour after the hand", [WPT]),
    ("ian finished 2nd ($201K), joe 3rd", "$201,428 and $131,081", [RESULTS]),
]

CAPTION = ("he folded three kings at a WPT final table 😳 only 2 hands could beat him… would you have called? 👇\n\n"
           "#poker #pokertok #wpt #holdem #fyp")
CREDIT = ("Footage: “Incredibly Unbelievable Fold on the World Poker Tour” by World Poker Tour, CC BY 3.0 "
          "(creativecommons.org/licenses/by/3.0), via Wikimedia Commons. "
          "Edited: cut, cropped, captions and graphics added. Not endorsed by WPT.")


def bound_verify() -> tuple[dict, str]:
    """verify.json, refusing to show it unless it was written for this exact video file."""
    v = json.loads((OUT / "verify.json").read_text(encoding="utf-8"))
    size = MP4.stat().st_size
    if (HERE / v["file"]).resolve() != MP4.resolve() or v["format"]["bytes"] != size:
        raise SystemExit(f"verify.json describes {v['file']} ({v['format']['bytes']} bytes), not {MP4} ({size}).")
    if (OUT / "verify.json").stat().st_mtime < MP4.stat().st_mtime:
        raise SystemExit("verify.json is older than the video; run verify.py first.")
    return v, hashlib.sha256(MP4.read_bytes()).hexdigest()


def grab(times: list[float]) -> dict[float, tuple[float, object]]:
    """First decoded frame at or after each time (seconds into the cut), with its real timestamp."""
    got = {}
    with av.open(str(MP4)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        for t in sorted(times):
            c.seek(int(t / vs.time_base), stream=vs, backward=True)
            for fr in c.decode(vs):
                if fr.time >= t - 1e-3:
                    got[t] = (fr.time, fr.to_image())
                    break
    return got


def xt(t: float) -> float:
    return X0 + t / DURATION * (X1 - X0)


def timeline_svg(starts: list[float]) -> str:
    r = xt(REVEAL)
    parts = [
        f'<rect x="{X0}" y="4" width="{r - X0:.1f}" height="18" rx="4" class="hid"/>',
        f'<text x="{X0 + 8}" y="17" class="lab hidt">🙈 joe&#39;s cards hidden</text>',
        f'<rect x="{r:.1f}" y="4" width="{X1 - r:.1f}" height="18" rx="4" class="shown"/>',
        f'<text x="{r + 6:.1f}" y="17" class="lab showt">👀 shown</text>',
    ]
    for i, ((label, _, _), t0, t1) in enumerate(zip(CHAPTERS, starts, starts[1:] + [DURATION])):
        a, b = xt(t0), xt(t1)
        parts.append(f'<g><title>{i + 1}. {escape(label)} · {t0:.1f}s</title>'
                     f'<rect x="{a + 1:.1f}" y="28" width="{b - a - 2:.1f}" height="30" rx="4" class="c{i % 2}"/>'
                     f'<text x="{(a + b) / 2:.1f}" y="48" class="num">{i + 1}</text></g>')
    for s in range(0, int(DURATION) + 1, 10):
        parts.append(f'<line x1="{xt(s):.1f}" x2="{xt(s):.1f}" y1="60" y2="66" class="tick"/>'
                     f'<text x="{xt(s):.1f}" y="79" class="axis">{s}s</text>')
    parts.append(f'<line x1="{r:.1f}" x2="{r:.1f}" y1="2" y2="62" class="rev"/>')
    parts.append(f'<line id="ph" x1="{X0}" x2="{X0}" y1="0" y2="64" class="ph"/>')
    return f'<svg id="tl" viewBox="0 0 1000 84">{"".join(parts)}</svg>'


def caption_svg(rows: list[dict]) -> str:
    mid, k = 66, 250  # y of "on time"; SVG units per second of offset
    parts = [f'<rect x="{X0}" y="{mid - 0.1 * k}" width="{X1 - X0}" height="{0.2 * k}" class="okband"/>']
    for s, label in ((0.2, "+0.2s"), (0.1, ""), (0.0, "0"), (-0.1, ""), (-0.2, "−0.2s")):
        y = mid - s * k
        parts.append(f'<line x1="{X0}" x2="{X1}" y1="{y}" y2="{y}" class="{"zero" if s == 0 else "grid"}"/>')
        if label:
            parts.append(f'<text x="{X0 - 6}" y="{y + 4}" class="axis end">{label}</text>')
    parts.append(f'<text x="{X1 - 4}" y="{mid - 0.2 * k - 4}" class="axis end">caption a bit early ↑</text>')
    parts.append(f'<text x="{X1 - 4}" y="{mid + 0.2 * k + 13}" class="axis end">caption a bit late ↓</text>')
    for row in rows:
        off = row["median_offset_s"]
        parts.append(f'<circle cx="{xt(row["start"]):.1f}" cy="{mid - off * k:.1f}" r="5.5" class="dot">'
                     f'<title>{escape(row["clip"])} · {off:+.3f}s · “{escape(row["caption"])}”</title></circle>')
    return f'<svg viewBox="0 0 1000 140">{"".join(parts)}</svg>'


def plain_checks(v: dict) -> list[tuple[bool, str, str]]:
    worst = max(abs(row["median_offset_s"]) for row in v["captions"])
    w, h = v["format"]["size"]
    fps = float(Fraction(str(v["format"]["avg_rate"])))
    words = {
        "format": f"tiktok size · {w}×{h} · {fps:.3f} fps",
        "all frames decode": f"all {v['log']['frames']:,} frames play",
        "frame log": "every frame comes from the right moment",
        "hole cards masked before reveal": "joe's cards stay hidden until the reveal",
        "no voice clip crosses the reveal": "no commentary leaks past the reveal",
        "no spoiler words in source clips": "no spoiler words in the clips (4 speech models)",
        "no spoiler words heard before reveal": "no spoilers heard before the reveal",
        "no commentator names heard": "no commentator names heard",
        "loudness": f"loudness {float(v['loudness']['lufs']):.0f} LUFS (phone-ready)",
        "audio in sync with the timeline": f"sound in sync ({float(v['av']['lag_ms']):.0f} ms off)",
        "audio length": "sound and picture end together",
        "captions on time": f"captions within {worst:.2f} s of the voice",
    }
    return [(c["ok"] in (True, "True"), words.get(c["name"], c["name"]), f"{c['name']}: {c['detail']}")
            for c in v["checks"]]


def link(src: tuple[str, str | None]) -> str:
    name, url = src
    return f'<a href="{escape(url)}" target="_blank" rel="noopener">{escape(name)}</a>' if url else escape(name)


def mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


PAGE = Template(r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>he folded three kings · tiktok cut</title>
<style>
@font-face { font-family: TT; src: url("fonts/TikTokSans.ttf"); font-weight: 300 900; }
:root { --bg:#0e1014; --card:#171a21; --line:#262b36; --txt:#e9ecf2; --mut:#8a93a6; --pink:#fe2c55; --cyan:#25f4ee; --ok:#3ddc84; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--txt); font: 15px/1.45 TT, system-ui, sans-serif; }
.wrap { display: grid; grid-template-columns: auto 1fr; gap: 24px; padding: 22px; max-width: 1500px; margin: auto; }
.left { position: sticky; top: 16px; align-self: start; }
video { height: min(84vh, 800px); aspect-ratio: 9/16; background: #000; border-radius: 18px; display: block;
        box-shadow: -4px 0 0 0 var(--cyan), 4px 0 0 0 var(--pink); }
.now { margin-top: 10px; color: var(--mut); font-size: 14px; text-align: center; }
.now b { color: var(--txt); }
h1 { font-size: 28px; margin: 0 0 2px; font-weight: 800; }
h1 em { font-style: normal; color: var(--pink); }
.sub { color: var(--mut); margin-bottom: 12px; }
.stat { display: inline-block; padding: 3px 11px; border-radius: 99px; background: var(--card); border: 1px solid var(--line);
        margin: 0 6px 14px 0; font-size: 13px; }
.stat.ok { color: var(--ok); border-color: #1f5a3b; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; margin-bottom: 16px; }
h2 { font-size: 15px; margin: 0 0 10px; font-weight: 700; }
h2 small { color: var(--mut); font-weight: 400; margin-left: 6px; }
.strip { display: grid; grid-template-columns: repeat(auto-fill, minmax(92px, 1fr)); gap: 10px; }
.ch { all: unset; cursor: pointer; text-align: center; font-size: 12px; color: var(--mut); }
.ch img { width: 100%; aspect-ratio: 9/16; object-fit: cover; border-radius: 10px; border: 2px solid transparent; display: block; margin-bottom: 4px; }
.ch:hover img { border-color: var(--cyan); }
.ch.on img { border-color: var(--pink); }
.ch.on { color: var(--txt); }
svg { width: 100%; height: auto; display: block; font-family: TT, system-ui, sans-serif; }
#tl { cursor: pointer; }
.hid { fill: #3a1d24; } .shown { fill: #173326; }
.lab { font-size: 11px; } .hidt { fill: #ff9fb0; } .showt { fill: #9ff0c2; }
.c0 { fill: #2b3140; } .c1 { fill: #353d50; }
.num { fill: #e9ecf2; font-size: 12px; text-anchor: middle; font-weight: 700; }
.tick { stroke: #4a5266; } .axis { fill: var(--mut); font-size: 10.5px; text-anchor: middle; } .axis.end { text-anchor: end; }
.rev { stroke: var(--pink); stroke-width: 2.5; }
.ph { stroke: var(--cyan); stroke-width: 2.5; }
.okband { fill: #12301f; } .grid { stroke: #2a303c; stroke-dasharray: 3 4; } .zero { stroke: var(--ok); stroke-width: 1.5; }
.dot { fill: var(--cyan); stroke: #0e1014; stroke-width: 1.5; }
.checks { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px; }
.pill { background: #11261b; border: 1px solid #1f5a3b; color: #bff3d3; border-radius: 10px; padding: 7px 10px; font-size: 13.5px; }
.pill.bad { background: #2a1116; border-color: #6b1f2c; color: #ffc2cc; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
td, th { text-align: left; padding: 7px 8px; border-top: 1px solid var(--line); vertical-align: top; }
th { color: var(--mut); font-weight: 600; border-top: 0; }
td:first-child { font-weight: 600; }
a { color: var(--cyan); text-decoration: none; } a:hover { text-decoration: underline; }
.kit { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.kit > div { display: flex; flex-direction: column; }
.box { flex: 1; background: #0f1218; border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; white-space: pre-wrap; font-size: 14px; }
button.cp { align-self: flex-start; margin-top: 6px; background: var(--pink); color: #fff; border: 0; border-radius: 8px; padding: 6px 12px; font: inherit; font-weight: 700; cursor: pointer; }
ul.notes { margin: 0; padding: 0; list-style: none; } ul.notes li { margin: 6px 0; }
footer { color: var(--mut); font-size: 12px; padding: 0 22px 22px; max-width: 1500px; margin: auto; overflow-wrap: anywhere; }
@media (max-width: 900px) { .wrap { grid-template-columns: 1fr; } .left { position: static; } video { height: auto; width: 100%; } .kit { grid-template-columns: 1fr; } }
</style></head>
<body>
<div class="wrap">
  <div class="left">
    <video id="v" src="$video" poster="$poster" controls playsinline preload="metadata"></video>
    <div class="now" id="now">tap a chapter to jump</div>
  </div>
  <main>
    <h1>he folded <em>three kings</em> 😳</h1>
    <div class="sub">tiktok cut · real WPT footage + commentary · ready to post</div>
    <div>$stats</div>
    <section><h2>chapters <small>tap to jump</small></h2><div class="strip">$strip</div></section>
    <section><h2>timeline <small>tap to jump · pink line = the reveal</small></h2>$timeline</section>
    <section><h2>checks on the final file <small>$passed of $total passed</small></h2><div class="checks">$checks</div></section>
    <section><h2>captions vs. voice <small>each dot = one spoken line · 0 = right on time · hover for the words</small></h2>$capchart</section>
    <section><h2>is every fact true? <small>each claim on screen and where it comes from</small></h2>
      <table><tr><th>on screen</th><th>why it holds</th><th>source</th></tr>$facts</table></section>
    <section><h2>post kit</h2><div class="kit">
      <div><div class="box" id="cap">$caption</div><button class="cp" data-copy="cap">copy caption</button></div>
      <div><div class="box" id="credit">$credit</div><button class="cp" data-copy="credit">copy credit</button></div>
    </div></section>
    <section><h2>good to know</h2><ul class="notes">$notes</ul></section>
  </main>
</div>
<footer>$footer</footer>
<script>
const DUR = $duration, REV = $reveal, X0 = $x0, X1 = $x1, CH = $chapters;
const v = document.getElementById('v'), ph = document.getElementById('ph'), now = document.getElementById('now');
const chips = [...document.querySelectorAll('.ch')];
const fmt = t => Math.floor(t / 60) + ':' + String(Math.floor(t) % 60).padStart(2, '0');
function jump(t) { v.currentTime = Math.min(DUR - 0.05, Math.max(0, t)); v.play(); }
chips.forEach(c => c.addEventListener('click', () => jump(+c.dataset.t + 0.01)));
const tl = document.getElementById('tl');
tl.addEventListener('click', e => {
  const r = tl.getBoundingClientRect(), u = (e.clientX - r.left) / r.width * 1000;
  jump((u - X0) / (X1 - X0) * DUR);
});
function show() {
  const t = v.currentTime, x = X0 + t / DUR * (X1 - X0);
  ph.setAttribute('x1', x); ph.setAttribute('x2', x);
  let i = 0; CH.forEach((c, k) => { if (t >= c.t - 0.02) i = k; });
  chips.forEach((c, k) => c.classList.toggle('on', k === i));
  now.textContent = '';
  const b = document.createElement('b'); b.textContent = (i + 1) + '. ' + CH[i].label;
  now.append(b, ' · ' + fmt(t) + ' / ' + fmt(DUR) + (t < REV ? ' · 🙈 cards hidden' : ' · 👀 cards shown'));
}
v.addEventListener('timeupdate', show); v.addEventListener('seeked', show);
document.querySelectorAll('button.cp').forEach(b => b.addEventListener('click', () => {
  const text = document.getElementById(b.dataset.copy).innerText, label = b.textContent;
  const done = () => { b.textContent = 'copied ✓'; setTimeout(() => { b.textContent = label; }, 1500); };
  const fallback = () => { const ta = document.createElement('textarea'); ta.value = text; document.body.append(ta);
    ta.select(); document.execCommand('copy'); ta.remove(); done(); };
  if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done, fallback); else fallback();
}));
</script>
</body></html>
""")


def main() -> None:
    v, sha = bound_verify()
    starts = [T(seg) for _, seg, _ in CHAPTERS]
    times = [t for _, _, t in CHAPTERS]
    for (label, _, t), t0, t1 in zip(CHAPTERS, starts, starts[1:] + [DURATION]):
        assert t0 <= t < t1, f"thumbnail for {label} at {t}s is outside its chapter {t0:.2f}-{t1:.2f}s"

    frames = grab(times)
    THUMBS.mkdir(exist_ok=True)
    for old in THUMBS.glob("*.jpg"):
        old.unlink()
    strip = []
    for i, ((label, _, t), t0) in enumerate(zip(CHAPTERS, starts)):
        real_t, img = frames[t]
        name = f"{i + 1:02d}-{label.replace(' ', '-').replace('+', 'and').replace(chr(39), '')}.jpg"
        img.resize((180, 320)).save(THUMBS / name, quality=84)
        print(f"thumb {name}: asked {t:.2f}s, got frame at {real_t:.3f}s")
        strip.append(f'<button class="ch" data-t="{t0:.3f}" title="{escape(label)} · starts {mmss(t0)}">'
                     f'<img src="out/thumbs/{name}" alt="{escape(label)}"><span>{i + 1} · {escape(label)}</span><br>'
                     f'<small>{mmss(t0)}</small></button>')
    poster_t, poster = frames[CHAPTERS[0][2]]
    poster.resize((540, 960)).save(OUT / "poster.jpg", quality=86)

    checks = plain_checks(v)
    passed = sum(ok for ok, _, _ in checks)
    mb = MP4.stat().st_size / 1e6
    w, h = v["format"]["size"]
    stats = "".join(f'<span class="stat{c}">{escape(s)}</span>' for c, s in [
        ("", f"⏱ {DURATION:.1f} s"), ("", f"📐 {w}×{h}"), ("", f"💾 {mb:.1f} MB"),
        (" ok" if passed == len(checks) else "", f"✅ {passed}/{len(checks)} checks"),
    ])
    notes = [
        "🎙 sound = real WPT commentary, plus sound effects and a quiet hum made in code. no AI voice, no music.",
        "🎨 picture = real WPT footage. captions, cards and stickers are drawn by code. no AI images.",
        f"🤏 one mumbled word (~{S('tank_30b', 374.91):.0f} s): the caption says “make a decision”; it could be "
        "“make my decision”. the sound fits “a” best, and the meaning is the same.",
        "📱 when posting: add a trending sound at about 5–10% volume so the commentary stays clear.",
    ]
    page = PAGE.substitute(
        video=VIDEO_URL, poster="out/poster.jpg", stats=stats, strip="".join(strip),
        timeline=timeline_svg(starts), passed=passed, total=len(checks),
        checks="".join(f'<div class="pill{"" if ok else " bad"}" title="{escape(detail)}">{"✓" if ok else "✗"} '
                       f'{escape(text)}</div>' for ok, text, detail in checks),
        capchart=caption_svg(v["captions"]),
        facts="".join(f"<tr><td>{escape(a)}</td><td>{escape(b)}</td><td>{' · '.join(link(s) for s in srcs)}</td></tr>"
                      for a, b, srcs in FACTS),
        caption=escape(CAPTION), credit=escape(CREDIT), notes="".join(f"<li>{escape(n)}</li>" for n in notes),
        footer=escape(f"file: out/{MP4.name} · {MP4.stat().st_size:,} bytes · sha256 {sha} · checks from verify.py "
                      f"(verify.json written after this file, same size) · verdict {v['verdict']} · "
                      f"rebuild this page: python make_preview.py"),
        duration=f"{DURATION:.4f}", reveal=f"{REVEAL:.4f}", x0=X0, x1=X1,
        chapters=json.dumps([{"label": label, "t": round(t0, 3)} for (label, _, _), t0 in zip(CHAPTERS, starts)]),
    )
    (HERE / "preview.html").write_text(page, encoding="utf-8")
    print(f"poster from {poster_t:.3f}s; checks {passed}/{len(checks)}; sha256 {sha}")
    print(f"wrote {HERE / 'preview.html'}")


if __name__ == "__main__":
    main()
