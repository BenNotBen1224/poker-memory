# poker-memory

Local-first poker hand memory and post-session review.

You enter a session hand by hand. Every hand is replayed by [PokerKit](https://github.com/uoftcprg/pokerkit) so
only legal hands are used, stored in SQLite with full version history, and then you can ask about it in plain
English. Answers show where each number came from, and every moment links to the session video.

```
enter hands ─▶ replay check (PokerKit) ─▶ SQLite (every version kept)
                                              │
        phone-friendly web app ◀── API ◀── ask in English ─▶ pot odds · equity · think time · video moment
```

This is the first working slice. Everything in `examples/` is synthetic: fictional players and a generated video.

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

## Entering a session

See `examples/home_game.yaml`. One line per event, with the video time after `@`:

```
s3 raise 75 @01:55.4      # bet/raise amounts are "to" totals for the street
flop Qs7c2d @01:02
s3 show QdJs              # or: s3 muck
```

Unknown opponent cards stay unknown. Import with `poker-memory import FILE.yaml`, then `poker-memory end KEY`.
Video is only accepted when `consent: all_participants`.

## What you can ask

| Ask | What happens |
| --- | --- |
| "Was my river fold to the raise reasonable?" | Finds the hand and decision; shows pot odds, think times, video moments |
| "What if he bluffs 10%?" | River what-if: equity and a fold/call verdict, labeled as an assumption |
| "What if he has 77, AK?" | Equity against that range (exact on the river, seeded simulation earlier) |
| "Show me the AQ hand" then "#2" | Several matches become a pick list |
| "hand 9", "on the flop", "last hand" | Narrow by hand, street, or recency |

Every fact is labeled **observed** (entered), **derived** (computed), **estimated** (depends on your assumption),
or **unknown**. Cards shown later in the hand are listed as hindsight and never used in the math.

## Fixing a hand

Edit the event lines on the hand page (or `PUT /api/sessions/{key}/hands/{n}`). Every save is a new version.
A version only goes into use if it replays correctly; broken versions are kept with the reason. Saves must be
based on the newest version, so two editors can't silently overwrite each other.

## Limits of this slice

- Review only after a session ends; live sessions are refused.
- No solver: nothing here claims to be GTO.
- English questions are matched by rules, not a language model; unusual wording may not be understood.
- Equity is against one opponent's range; multiway spots say so.
- Hands are typed in; there is no video or voice capture yet.

## Later

Hand entry from video/voice, player notes across sessions, range presets, solver integration, sharing.
