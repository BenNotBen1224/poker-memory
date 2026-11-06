# poker-memory

Local-first poker hand memory and post-session review.

You enter a session hand by hand. Every hand is replayed by [PokerKit](https://github.com/uoftcprg/pokerkit) so
only legal hands are used, stored in SQLite with full version history, and then you can ask about it in plain
English. Answers show where each number came from, and every moment links to the session video.

```
enter hands ─▶ replay check (PokerKit) ─▶ SQLite (every version kept)
                                              │
        phone-friendly web app ◀── API ◀── ask in English ─▶ pot odds · equity · river solver · think time · video moment
```

This is the first working slice. `examples/home_game.yaml` is synthetic (fictional players, a generated video).
`examples/real/` is one hand from a real, in-person tournament recording: see
[Try it on a real recording](#try-it-on-a-real-recording).

## Run it

Needs Python 3.12+ and Node 20+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,video]"
.\.venv\Scripts\poker-memory.exe demo            # loads examples/home_game.yaml and ends the session
.\.venv\Scripts\poker-memory.exe ask "Was my river fold to the raise reasonable? what if he bluffs 10%"
.\.venv\Scripts\poker-memory.exe serve           # API on http://127.0.0.1:8000
```

In a second terminal:

```powershell
cd web
npm install
npm run build
npm run start                                     # app on http://127.0.0.1:3000
```

The web app forwards `/api/*` to the API. Set `POKER_MEMORY_API` to point it elsewhere. Data lives in `data/`
(change with `--data` or `POKER_MEMORY_DATA`).

Tests: `.\.venv\Scripts\python.exe -m pytest -q tests`

## Read-only WPT decision demo

Open **http://127.0.0.1:3000/demo** for the focused Ian/Joe hand study. Its default decision lens
does not fetch/render opponent camera cards or load the unredacted video. Explicitly select
**Broadcast & outcome** to reveal the camera evidence, play the CC BY 3.0 source and seek the
reviewed action timeline. The diagram stays at the river decision; selecting a timeline action
changes the evidence cue, not the diagram's decision snapshot.

Use a **new isolated directory**, not an existing session database:

```powershell
# First obtain the exact openly licensed clip if absent:
.\.venv\Scripts\python.exe scripts\fetch_real_video.py
.\.venv\Scripts\python.exe scripts\setup_review_demo.py --data C:\path\to\new-demo-runtime
$env:POKER_MEMORY_DATA = 'C:\path\to\new-demo-runtime'
.\.venv\Scripts\python.exe -m uvicorn poker_memory.api:app_from_env --factory --host 127.0.0.1 --port 8000
# In a second terminal, from web:
npm run build
npm run start -- --hostname 127.0.0.1
```

The read-only `GET /api/demo/wpt/review?lens=decision|broadcast` returns a typed `Review`:
case/revision/source identity, exact replay-derived `Decision`, timeline, one-hand player context,
testable hypotheses, attributed reporting and limitations. Only the broadcast lens includes
`Hindsight` and its checked `/api/demo/wpt/video` URL. No `/api/ask`, conversation writes or solver
run is used. Video seeking uses HTTP Range support. Missing, changed, live, unverified or
pending-correction evidence fails explicitly. Case SHA-256 binds the exact normalized session
and transcription; video SHA-256 binds the existing reviewed source receipt.

All amounts are conditional on that record: extra call **2,140,000**, pot **5,150,000**,
called final pot **7,290,000**, break-even **214/729 = 29.3552812071%**. The control is an
illustrative share of shoves Ian beats, not a measured bluff frequency; ties excluded.
Future fold EV is zero; call advantage is `p * 7,290,000 - 2,140,000`. Full-hand recorded net
(-1,490,000) is separate from known-card losing-call net (-3,630,000), and neither determines
optimal play against an unknown range. Broadcast starting stacks are disputed in
[PokerNews's correction](https://www.pokernews.com/news/2018/03/ian-steinman-kings-set-fold-mckeehen-30175.htm);
do not infer exact physical-table survival or ICM effects. Ian's
[reported post-hand reasoning](https://www.cardplayer.com/poker-news/22600-poker-pro-unsure-if-epic-fold-was-a-smart-play)
is attributed, not measured player statistics. No long-run profile or validated payout model is available.

This is a reviewed transcription, not a new automated accuracy result. WPT automation remains
26/27 with a known +3.1s fold-time mismatch. Approximate, possibly edited video clocks are not
reliable thinking times. Existing frozen detector snapshots and failed reports are unchanged;
adding the review module means a new current-source freeze would include the new source too.

Verification:
`.\.venv\Scripts\python.exe -B -m pytest tests\test_review.py tests\test_api.py -q`;
from `web`, `npm run build` and `npm run typecheck`.
For a rerunnable real-browser check, paste `scripts\verify_review_demo.js` into the browser console
on `/demo` (requires modern Chromium). It exercises slider values, default/redacted and explicit
broadcast lenses, decoded playback, timeline seeking, visible source credit, request-error/retry behavior,
and an actual 390px iframe viewport. It restores the decision lens and returns/stores a JSON receipt
as `window.__reviewDemoProof`; it does not write application data.

## Entering a session

See `examples/home_game.yaml`. One line per event, with the video time after `@`:

```
s3 raise 75 @01:55.4      # bet/raise amounts are "to" totals for the street
flop Qs7c2d @01:02
s3 show QdJs              # or: s3 muck
```

Unknown opponent cards stay unknown. Import with `poker-memory import FILE.yaml`, then `poker-memory end KEY`.

Stakes default to a per-player ante: existing records without `ante_kind` still mean
`uniform`, and an omitted `ante` still means zero. A big-blind ante must be explicit:

```yaml
stakes:
  sb: 50000
  bb: 100000
  ante: 100000
  ante_kind: big_blind
```

Only the big blind pays that ante, separately from its live blind; five players start
with 250000 in the pot, not 650000. Heads-up, the button posts the small blind and the
other player pays the big blind and ante (no automatic heads-up ante reduction).
The supported big-blind-ante rule is **ante first**: a short big blind posts as much
of the ante as its stack allows, then any remaining chips toward its blind. The ante
is dead money in the main pot, not an uncalled bet to return or a live blind credit.
Blind-first variants are not supported. Replay uses PokerKit's actual posted-blind
amounts for calls; bet/raise amounts remain street totals excluding the ante.
The API/SQLite record retains `ante_kind`, and the session list labels `BB ante`
versus `ante per player`. This model support does not add recognition of BB-ante graphics.

Video is only accepted with `consent: all_participants` (everyone at the table agreed) or `consent: open_license`
(the video's license allows it; the file must also name the `source:` with title, author, URL and license, and the
app shows that credit under the video). Video can only be added with the CLI, never through the upload API.

## What you can ask

| Ask | What happens |
| --- | --- |
| "Was my river fold to the raise reasonable?" | Finds the hand and decision; shows pot odds, think times, video moments, and the river solver's answer |
| "Was my river bet of 800000 a good size?" | River solver: what checking and each bet size were worth with your hand |
| "What if he bluffs 10%?" | River what-if: equity and a fold/call verdict, labeled as an assumption |
| "What if he has 77, AK?" | Equity against that range (exact on the river, seeded simulation earlier) |
| "Show me the AQ hand" then "#2" | Several matches become a pick list |
| "hand 9", "on the flop", "last hand" | Narrow by hand, street, or recency |

Every fact is labeled **observed** (entered), **derived** (computed), **estimated** (depends on your assumption),
or **unknown**. Cards shown later in the hand are listed as hindsight and never used in the math.

## River solver estimates (GTO)

When the decision you ask about is on the river with two players left, the answer adds estimated move values
under ranges inferred from the betting actions. The web app draws one bar per move and marks yours and the
highest estimated-value move. These are model estimates, not proof of optimal play. On the real hand below,
the estimated call value is about 4.3M chips higher than folding; a 465,300 river bet is estimated about 15,000
above the 800,000 bet. That small estimated gap could change under different assumed ranges.

How it works (`src/poker_memory/gto.py`, `solver.py`, `handrank.py`, written here, no outside solver code):
- Ranges come from the betting only. Both players start with every hand; each call keeps the hands with enough
  equity against the other player's range to pay the price. Cards shown later are never used.
- The river is solved as a game tree with discounted regret matching. The bets it can choose are check, the sizes
  actually bet on that river, and all-in, with up to two legal raises (each at least the size of the last bet).
- The solver's exploitability target is range-averaged; it is not a per-hand error bound or a guarantee about
  the EV estimates for this hand. Values are in chips; tournament payout effects (ICM) are ignored. The mix
  and move estimates can vary with the model and assumed ranges.
- Historical measurements on one shared, busy 8-core VM found a 0.84 s median and 1.15 s slowest first HTTP
  question in a 10-run trial. Those observations are not a latency guarantee. Lookup tables and the common
  first preflop range step are warmed when the server starts. `127.0.0.1` was faster than `localhost` on that
  Windows machine. Deeper stacks or more bet sizes make the tree bigger.

Other spots (turn and earlier, three or more players) say "No solver result for this spot" and why.

## Fixing a hand

Edit the event lines on the hand page (or `PUT /api/sessions/{key}/hands/{n}`). Every save is a new version.
A version only goes into use if it replays correctly; broken versions are kept with the reason. Saves must be
based on the newest version, so two editors can't silently overwrite each other.

## Try it on a real recording

`examples/real/wpt_rolling_thunder.yaml` is one hand from the World Poker Tour Rolling Thunder final table
(five players left, 7 min 27 s). The video is "Incredibly Unbelievable Fold on the World Poker Tour" by
World Poker Tour, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/),
[source](https://commons.wikimedia.org/wiki/File:Incredibly_Unbelievable_Fold_on_the_World_Poker_Tour.webm).
The 35 MB file is not in git. The script downloads it and checks its SHA-1 (`9e4f94ad…`).

```powershell
.\.venv\Scripts\python.exe scripts\fetch_real_video.py
.\.venv\Scripts\poker-memory.exe --data data\real import examples\real\wpt_rolling_thunder.yaml
.\.venv\Scripts\poker-memory.exe --data data\real end wpt-rolling-thunder-final-table
.\.venv\Scripts\poker-memory.exe --data data\real ask "Was my river fold to the raise reasonable?"
.\.venv\Scripts\poker-memory.exe --data data\real serve
```

The hand was read from the video by eye, frame by frame. To check it yourself, take one frame at each event's
stated time and look at it:

```powershell
.\.venv\Scripts\python.exe scripts\event_frames.py examples\real\wpt_rolling_thunder.yaml --out events
```

`scripts/frames.py` makes contact sheets and zoomed crops (see `--help`). Frames and crops are adapted material
(cropped, resized, labeled); keep the credit above if you share them.

What to know:
- Event times are within about 1 to 3 seconds. The overlay's pot figure can lag by several seconds on a big bet,
  so times come from the chips, the action label or the card itself. The video clock may include edits, so think
  times are video-clock times, not stopwatch times.
- Seat numbers and the button are inferred; the video never shows them.
- The broadcast's hole-card cameras show the opponents' cards. The players could not see them, so they are kept out
  of every number and only mentioned afterwards as hindsight. The broadcast's own win percentages are not used.

## Read a hand from the broadcast automatically

`poker-memory read VIDEO...` reads one hand from a broadcast's own TV graphics and prints it as YAML, in the
`stakes:` and `hands:` shapes a session file uses. Add a `session:` block (key, title, consent, source) to import it.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"
.\.venv\Scripts\poker-memory.exe read examples\real\wpt_rolling_thunder_fold.webm
```

It runs on the CPU, with no GPU and no paid service:
- It decodes the keyframes (about one every 5 s). Where two neighbouring keyframes' graphics differ, it decodes
  about four pictures a second between them, and stops once the graphics settle. After the last keyframe it reads
  on to the end. That is 542 of the clip's 10,729 pictures.
- A 1.9 MB text-reading model reads each distinct piece of text (a name, an action line, a card rank) once. It is
  PaddleOCR's English-and-digits recogniser, under Apache-2.0 (see `src/poker_memory/reader/models/NOTICE`). The
  whole hand takes 82 reads.
- Who holds which cards, the seats, the button and the blinds come from the graphics' layout and the broadcast's
  rules, not from a card or seat detector. Before the flop, panels show up in acting order, which gives the button
  and the blinds. The reader numbers seats from the small blind, which is seat 1.

For the 7.5-minute clip on an 8-core CPU (AMD EPYC 7763, a shared VM):

| Run | Time |
| --- | --- |
| Already running: a server, or the second video onwards in one `read` | 0.99 to 1.0 s |
| Fresh process, including starting Python and loading the libraries and the model | about 2.3 s |

Most of the time goes on decoding the video. The machine is shared, so single reads vary from 0.99 to 1.12 s, and
the median sits on the 1 s line.
`scripts\bench_fast_read.py VIDEO` measures these times. Compare the result with the hand read by eye like this:

```powershell
.\.venv\Scripts\python.exe scripts\check_fast_read.py examples\real\wpt_rolling_thunder.yaml
```

The existing WPT calibration yields 26 of 27 checks and exits with a mismatch: the fold reads 3.1 s late,
because the graphics clear Ian's panel that long after he lets go of his cards. Complete hand construction
supports this broadcast's graphics layout (WPT, 1280×720), and reads one hand per video. WPT stacks are never
on screen, so a stack is the least it could be, unless an all-in shows it.

The packet decoder handles delayed/reordered H.264 frames as well as the VP9 calibration clip. It retains the
decoder between a keyframe and its following samples, uses decoded-frame presentation timestamps, and drains
delayed frames at EOF. Codec handling alone does not add support for new broadcast layouts.

`poker_memory.reader.suits.read_suit(current_rgb, previous_rgb)` is a separate
**development-calibrated** red/black suit-shape tool, used by the native observation path below.
It accepts native 16x16 or 16x17 RGB suit-interior crops (no rank or card border).
Both frames must independently match a shape, agree, and pass visibility/colour
checks; absent, dim, occluded, ambiguous, or changing glyphs return `None`.
Unsupported crop dimensions/types raise `ValueError`. Its four templates came
from candidate63 frame 245491, plus a club raster variant from frame 245251;
separate development frame pairs checked all four
suits. Confident colour restricts the competing shapes to hearts/diamonds or
clubs/spades; it never selects a suit without matching its glyph. This avoids a
black spade falsely reducing a readable red diamond's shape margin. Shape,
visibility and two-frame thresholds are unchanged. This is not a held-out accuracy claim. WPT's existing
four-colour extraction remains unchanged.
The additional club raster is eligible only when its pixels show the constriction
below the top lobe and wider side lobes; it cannot reduce a smooth spade's margin.
Stable candidate63 Peters cards now read as 4c2c without a player/card lookup.
Absent, sliding, occluded, dim and conflicting crops still reject. Calibration
and validation are separate development frames, not independent held-out hands.

Candidate56 subsequently exposed native rank crops including the card's bottom
border/background: the local recognizer read stable 9/8/7 pixels as `g`/`Q`/`Z`.
Native ranks now use the 18-pixel card interior instead of a 21-pixel crop; no
rank alias or player/card lookup was added. WPT crops are unchanged.
If that tight crop yields no supported rank, the same local recognizer tries
four pixels of paper-colour padding below it, never the neighbouring border.
This restores required context for narrow hook-shaped ranks. A valid tight-crop
rank is never overridden by the padded attempt; unsupported outputs stay unknown.
Black suits require compatible visible structure before full-template matching:
a club's top-lobe constriction and wider side lobes, or a spade's narrow tip
and expanding cap. An incompatible black shape cannot reduce a readable
glyph's runner-up margin. The absolute shape and two-frame/visibility thresholds
remain unchanged; the existing margin applies when multiple eligible shapes
remain. Topology alone cannot create a suit, and ambiguous, faded or occluded
glyphs still return unknown. No new model/template was added.
These rules use candidate56 development evidence. Its original frozen diagnostic
remains a failure; later development reads are not retrospective benchmark passes.

`read_hand` recognizes the scoped 854x480 PokerGO development layout from its
BLINDS and BB ANTE labels. It uses the existing local OCR model and suit-shape
callable, retaining pixel observations in `Reading.observations` on success or
`UncertainHand.evidence` on rejection.
Evidence includes names/graphic slots, displayed stacks and cyclic button/SB/BB
markers, stakes with explicit BB ante, individual hole-card ranks/suits and
owners, displayed pot, and action/win labels. Unsupported glyphs and numeric
transitions remain unknown. Unrecognized panel owners are explicitly unknown;
their cards/actions are not assigned to a guessed player. Header stacks with fewer than three recognized
digits are unsupported in this development path; they are not silently accepted
as tiny stacks. Decimal K/M amounts retain punctuation; an amount exceeding the
last confirmed displayed stack is rejected rather than divided or reconstructed
from the pot. This conservative bound can also reject legitimate wagers when
the displayed stack is stale.

The minimal builder supports **five-player unopened preflop hands with one
explicit raise and every other player explicitly folding**. It requires stable
header players/stacks/roles established before actions, labels in cyclic preflop
order, readable winner cards, and matching displayed initial/raised pots and net-win
label. Shared PokerKit replay must finish legally before a `Reading` is returned.
Missing folds, calls/checks/all-ins, conflicting roles/cards/pots, illegal raises
and postflop sequences remain rejected. Unknown opponent camera cards are retained
as unknown, not filled from manual labels. A terminal preflop replay establishes
that no board was dealt (`board_from_replay`, inferred); it does not implement a
general board detector.

The reader still requires a **manually pre-cut single hand**. Clip preparation,
source frame/PTS mapping and manual boundary review are separate from automatic
reader time; production code contains no candidate source indexes. Logical seat
IDs follow the observed relative header order and verified action order, not
printed physical chair numbers. Starting stacks use stable pre-action header
values, checked against replay and the pot/net-win labels. Old/new graphic updates
must not be mixed; changing order/roles or inconsistent values reject the cut.
Labels require consecutive sample agreement. Observation time brackets describe
sampled graphic recognition, not certified visibility onset or physical-action
time; event timestamps are sampled confirmation times.

Unsupported input prints structured uncertain YAML and exits **2**, without a
`hands` key. Successful preflop output is a normal `Reading`/hand YAML with explicit
diagnostics. Panel disappearance is never a fold in this path. Camera cards are
hindsight observations, not decision-time opponent knowledge. It was exercised on candidate63 only:
development material, NOT REVIEWED, excluded from held-out scoring, and never
imported into the consent-gated application.

The rules were tuned on this one hand and are checked against the same hand, so they are not yet proven on
another. The WPT answer key's times are good to 1 to 3 s. A WPT read that misses something prints it as a `#` comment and
still exits normally. The hero is the player who acted last.

## Prepare and score a reviewed detector corpus

`scripts/detection_eval.py` validates a real-video hand manifest, prepares a ledger of every decoded frame, and
runs the actual detector against independently authored references. It rejects synthetic hands, duplicate clip
hashes, overlapping intervals from one source recording, the WPT demo above, incomplete full-frame reviews, and
missing/malformed references. Scoring defaults to requiring 100 admitted hands; fewer hands cannot produce a
passing result.

Each manifest hand needs `id`, `source_url`, `permission` (`open_license` or `all_participants`),
`permission_details`, separate `source_recording_sha256` and `clip_sha256` hashes, a stable `source_recording`
identifier, a half-open `hand_interval` in source-recording seconds, and paths to `clip_path`,
`review_prep_path`, and `reference_path`.
References must be manually authored with `authoring: "independent_manual"`, a completed review covering every
frame, explicit visible/unknown/inferred statuses, and evidence frame ranges for visible values. Do not copy
detector or OCR output into the reference. The preparation output is only marked `PREPARED`, never `REVIEWED`.

```powershell
.\.venv\Scripts\python.exe -B scripts\detection_eval.py prepare clips\hand-001.mp4 --out review-prep\hand-001
# Optional bounded pilot; exit 2 means this ledger is partial, not ready for review/scoring.
.\.venv\Scripts\python.exe -B scripts\detection_eval.py prepare clips\hand-001.mp4 --out review-prep\pilot --max-frames 120
.\.venv\Scripts\python.exe -B scripts\detection_eval.py prepare clips\hand-001.mp4 --out review-prep\pilot --resume
.\.venv\Scripts\python.exe -B scripts\detection_eval.py validate corpus.json --require-hands 100
.\.venv\Scripts\python.exe -B scripts\detection_eval.py score corpus.json --out results.json --require-hands 100
```

If a process is interrupted between flushing frame rows and finalizing its metadata, normal resume deliberately
rejects the stale count/hash. Preserve the interrupted directory first. On a working copy, run
`prepare VIDEO --out COPY --recover-interrupted`: this re-decodes and exactly compares every existing ledger row
(including source frame PTS, dimensions, and RGB hash) before atomically repairing incomplete metadata.
It rejects changed source bytes/paths, corrupt rows, and truncated JSON rather than discarding evidence.
Recovery exits 2 with `PREPARED_PARTIAL` / `NOT REVIEWED`, even if its rows happen to reach EOF; it does not append
frames or certify completion. Then run `prepare VIDEO --out COPY --resume` to finish normally.

`review.html` is approximate timestamp navigation and does not prove which exact frame the browser displays.
`frames.jsonl` records each frame index, PTS/time base, timestamp, dimensions, and pixel hash. For exact review,
`extract-frames VIDEO --preparation review-prep\hand-001\preparation.json --start-index 100 --count 20 --out evidence`
extracts up to 300 ledger-verified, full-resolution PNGs at a time; it does not create PNGs for every frame by
default. Preparation itself can resume by frame index. Reviewers must independently watch the whole clip and fill
`reference-template.json`; use extracted exact frames when a visual detail or timestamp is ambiguous.
Scoring reports exact visible-card owner/rank/suit agreement, event differences and time error against labeled
intervals, visible coverage, unknown/inferred fields, reader issues, cold subprocess time, and the reader's full
read time including video scan/decode. Paid API cost is null (the detector uses no paid API); compute cost is only
estimated if an hourly rate is supplied. `MATCHES_REFERENCES` requires the requested number of complete
references, no rejected rows, and exact visible-field/card/event/timing matches without reader issues. It means
only reference agreement; it is not a product-performance pass and does not assess the historical under-one-second
or <$0.10 targets. Normal exit or repeatable output alone cannot produce this verdict.

### Separate private AI diagnostic study

`diagnostic-validate` and `diagnostic-score` use a separate mode:
**private diagnostic, AI-reviewed, not human-certified, not admitted to the certified 100-hand benchmark**.
Neither command changes preparation review fields, application consent rules, or source permissions.
Unknown source rights stay `permission: "unknown"` with no inferred license. A successful diagnostic exit
means agreement with the supplied AI reference, not generalization, permission, human certification, or a
product-speed/cost pass. Certified `validate`/`score` reject this mode and AI provenance even if rights are stated.

A diagnostic manifest uses the same hand/path/source-interval fields above, plus top-level
`study_mode: "private_ai_diagnostic"` and an explicit boolean `development_only` on every hand.
The latter keeps calibration hands separate from new diagnostic hands; neither count is certified admission.
Do not relabel a development hand as held-out. References require:

```json
{
  "schema_version": 1,
  "study_mode": "private_ai_diagnostic",
  "authoring": "independent_ai",
  "ai_provenance": {
    "model": "actual reviewing model or AI system",
    "review_method": "independent visual inspection of every exact decoded frame",
    "independent_of_detector": true
  },
  "video_sha256": "<actual clip SHA-256>",
  "frame_ledger_sha256": "<actual complete frames.jsonl SHA-256>",
  "review": {
    "status": "complete",
    "reviewer_type": "ai",
    "reviewer": "independent AI review identifier",
    "completed_at": "2026-11-05T00:00:00Z",
    "frame_count": 3,
    "reviewed_frame_ranges": [[0, 2]]
  },
  "fields": {},
  "cards": [],
  "events": []
}
```

This is a shape example, not a valid reference or a supplied answer key: fill all eight required fields
(`stakes`, `button`, `hero_seat`, `player_names`, `stacks`, `opponent_hole_cards`, `seat_numbers`,
`board_cards`) and also `ante_kind` (`uniform`/`big_blind`) using visible/unknown/inferred classifications.
Unavailable fields must be explicitly unknown, not omitted. Visible/inferred fields, cards and events require
exact frame evidence; visible events also need time intervals. Inferred values are retained but not scored as
visible facts. Logical seat IDs or a board established only by terminal replay should be labeled inferred,
not printed/visually observed facts. Do not use detector/OCR output to author the independent reference.

Preparation must be `PREPARED` with `decoded_complete: true`, matching clip and ledger hashes, contiguous frame
indices, valid RGB hashes/dimensions and strictly increasing PTS/timebases. Complete reviewed ranges must cover
every ledger frame. These are integrity/declared-review gates, not proof that an AI's visual judgment is correct.
Source recording hashes/intervals are declared provenance; this command verifies the actual clip and prepared
ledger, not source reuse rights or manual segmentation accuracy.

```powershell
.\.venv\Scripts\python.exe -B scripts\detection_eval.py diagnostic-validate private-corpus.json
.\.venv\Scripts\python.exe -B scripts\detection_eval.py diagnostic-score private-corpus.json --out private-results-001.json --detector-freeze detector-freeze.json --freeze-sha256 ACTUAL_FREEZE_SHA256 --timeout-seconds 120
```

Scoring requires a SHA-bound detector freeze containing all 19 current production source hashes, exact local
model/notice assets, dependency manifest/installed-version snapshot, Python version, OCR config and environment
snapshot. They are checked before/after attempts along with clip/reference/ledger identity. The current development
freeze records actual OCR runtime settings; no portable hash-locked Python environment is claimed where no Python
lock exists. Detector tuning is separate work, never an implicit response to a new reference mismatch.

Reports retain raw normal-reader output and observations (or stage-specific uncertainty), the same eight-field,
card-owner/rank/suit, event and timing comparisons, ante-kind discrimination, reference classifications,
missing/extra values, rejections, errors and timeout stdout/stderr. Each detector call has a finite timeout;
there are no automatic retries. Cold subprocess totals and optional hourly compute estimates include failures
and timeouts. Startup/imports are included in cold totals; segmentation and AI-review time/cost remain unmeasured,
not zero. An existing output path is refused so a later attempt cannot overwrite an earlier failure.
Independent references must be supplied before new diagnostic detector runs; adding this tooling is not a
100-hand test result.

## Limits of this slice

- Review only after a session ends; live sessions are refused.
- The solver covers only two-player river decisions, and its ranges come from a simple betting rule.
- English questions are matched by rules, not a language model; unusual wording may not be understood.
- Equity is against one opponent's range; multiway spots say so.
- Hands are typed in, or read from one broadcast's graphics layout; there is no voice capture yet.

## Later

More broadcast layouts, hand entry by voice, player notes across sessions, range presets, solving the turn and
earlier, sharing.
