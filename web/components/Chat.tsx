"use client";

import Link from "next/link";
import { useState } from "react";
import { api, type Answer, type Solver } from "@/lib/api";

const EXAMPLES = [
  "Was my river fold to the raise reasonable?",
  "What if he bluffs 10%?",
  "What if he has 77, AK?",
];

const SYNTHETIC_EXAMPLES = ["Show me the AQ hand", "Was my river call good in hand 9?"];

type Turn = { question: string; answer?: Answer; error?: string };

export function Chat({ sessionKey, synthetic, onSeek }: {
  sessionKey: string;
  synthetic: boolean;
  onSeek?: (ms: number) => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [conversation, setConversation] = useState<number | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);

  async function ask(question: string) {
    question = question.trim();
    if (!question || busy) return;
    setBusy(true);
    setText("");
    setTurns((t) => [...t, { question }]);
    try {
      const answer = await api<Answer>("/api/ask", {
        method: "POST",
        body: JSON.stringify({ question, session_key: sessionKey, conversation_id: conversation }),
      });
      if (answer.conversation_id != null) setConversation(answer.conversation_id);
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, answer } : turn)));
    } catch (e) {
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, error: String(e) } : turn)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <h2>Ask about a hand</h2>
      {turns.map((turn, i) => (
        <div key={i}>
          <div className="question"><span>{turn.question}</span></div>
          {turn.error && <div className="card error">{turn.error}</div>}
          {turn.answer && <AnswerCard answer={turn.answer} onSeek={onSeek} onPick={ask} />}
          {!turn.answer && !turn.error && <div className="card muted">Thinking…</div>}
        </div>
      ))}
      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          ask(text);
        }}
      >
        <input
          className="grow"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. Was my river fold reasonable?"
          aria-label="Question"
        />
        <button className="primary" disabled={busy || !text.trim()}>Ask</button>
      </form>
      <div className="row" style={{ marginTop: 8 }}>
        {(synthetic ? [...EXAMPLES, ...SYNTHETIC_EXAMPLES] : EXAMPLES).map((q) => (
          <button key={q} type="button" className="chip" disabled={busy} onClick={() => ask(q)}>
            {q}
          </button>
        ))}
      </div>
    </section>
  );
}

const chips = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 2, signDisplay: "exceptZero" });

function SolverBars({ s }: { s: Solver }) {
  const lo = Math.min(0, ...s.ev);
  const span = Math.max(0, ...s.ev) - lo || 1;
  return (
    <div className="solver" aria-label="Solver value of each move">
      {s.moves.map((m, k) => (
        <div key={m} className="solver-row">
          <span>
            {m}
            {k === s.best && <b className="tag best"> best</b>}
            {k === s.taken && <b className="tag you"> you</b>}
          </span>
          <span className="solver-track">
            <span
              className={`solver-bar${k === s.best ? " best" : ""}${k === s.taken ? " you" : ""}`}
              style={{ left: `${((Math.min(0, s.ev[k]) - lo) / span) * 100}%`, width: `${(Math.abs(s.ev[k]) / span) * 100}%` }}
            />
          </span>
          <span className="solver-ev">{chips.format(s.ev[k])}</span>
        </div>
      ))}
    </div>
  );
}

function AnswerCard({ answer, onSeek, onPick }: {
  answer: Answer;
  onSeek?: (ms: number) => void;
  onPick: (q: string) => void;
}) {
  return (
    <div className="card">
      <div className="row">
        <span className={`badge ${answer.kind === "analysis" ? "good" : "warn"}`}>{answer.kind.replace("_", " ")}</span>
        {answer.hand != null && (
          <Link href={`/s/${encodeURIComponent(answer.session_key ?? "")}/h/${answer.hand}`}>
            Hand {answer.hand}, revision {answer.revision}
          </Link>
        )}
      </div>
      {answer.understood && <div className="muted">I understood: {answer.understood}</div>}

      {answer.facts.length ? (
        <ul className="facts">
          {answer.facts.map((f) => (
            <li key={f.name}>
              <span className={`badge ${f.kind}`}>{f.kind}</span>
              <div className="grow">
                {f.text}
                {f.name === "gto" && f.kind === "estimated" && <SolverBars s={f.value as Solver} />}
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p>{answer.text}</p>
      )}

      {answer.assumptions.length > 0 && (
        <div className="muted">
          {answer.assumptions.map((a) => <div key={a}>Assumption: {a}</div>)}
        </div>
      )}

      {answer.candidates.length > 0 && (
        <div className="row" style={{ marginTop: 8 }}>
          {answer.candidates.map((c, i) => (
            <button key={c.index} type="button" onClick={() => onPick(`#${i + 1}`)}>
              #{i + 1}: {c.summary}
            </button>
          ))}
        </div>
      )}

      {answer.evidence.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div className="muted">Video moments</div>
          <div className="row">
            {answer.evidence.map((ev) => (
              <button key={ev.label + ev.start_ms} type="button" disabled={!onSeek} onClick={() => onSeek?.(ev.start_ms)}>
                ▶ {ev.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {answer.tools.length > 0 && (
        <details style={{ marginTop: 8 }}>
          <summary>How this was worked out ({answer.tools.map((t) => t.tool).join(", ")})</summary>
          <pre>{JSON.stringify(answer.tools, null, 2)}</pre>
        </details>
      )}
    </div>
  );
}
