"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Video, type VideoHandle } from "@/components/Video";
import { api, clock, type Revision, type Session, type StoredHand } from "@/lib/api";

export default function HandPage() {
  const params = useParams<{ key: string; index: string }>();
  const key = decodeURIComponent(params.key);
  const index = Number(params.index);
  const base = `/api/sessions/${encodeURIComponent(key)}/hands/${index}`;

  const [session, setSession] = useState<Session | null>(null);
  const [hand, setHand] = useState<StoredHand | null>(null);
  const [revisions, setRevisions] = useState<Revision[]>([]);
  const [viewing, setViewing] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState("");
  const video = useRef<VideoHandle>(null);

  const load = useCallback(
    async (revision: number | null) => {
      try {
        const [all, h, revs] = await Promise.all([
          api<Session[]>("/api/sessions"),
          api<StoredHand>(revision ? `${base}?revision=${revision}` : base),
          api<Revision[]>(`${base}/revisions`),
        ]);
        setSession(all.find((s) => s.key === key) ?? null);
        setHand(h);
        setRevisions(revs);
        setError("");
      } catch (e) {
        setError(String(e));
      }
    },
    [base, key],
  );

  useEffect(() => {
    load(viewing);
  }, [load, viewing]);

  if (error && !hand) return <p className="error">{error}</p>;
  if (!hand) return <p className="muted">Loading…</p>;

  const alias = Object.fromEntries(hand.record.seats.map((s) => [String(s.seat), s.alias]));
  const heroSeat = hand.record.hero.seat;
  const seek = session?.video_url ? (ms: number) => video.current?.seek(ms) : undefined;

  return (
    <>
      {session?.video_url && <Video ref={video} src={session.video_url} source={session.source} />}
      <p className="muted">
        <Link href={`/s/${encodeURIComponent(key)}`}>← {session?.title ?? key}</Link>
      </p>
      <h1>
        Hand {hand.index} <span className="cards">{hand.record.hero.cards}</span>
      </h1>
      <div className="row">
        <span className="badge">revision {hand.revision}</span>
        <span className={`badge ${hand.status === "verified" ? "good" : "bad"}`}>
          {hand.status === "verified" ? "checked by replay" : "does not replay"}
        </span>
        {hand.revision === hand.active_revision ? (
          <span className="badge good">in use</span>
        ) : (
          <span className="badge warn">not in use (in use: {hand.active_revision ?? "none"})</span>
        )}
        {hand.replay.board.length > 0 && <span className="cards">Board {hand.replay.board.join(" ")}</span>}
      </div>
      {hand.replay.issues.map((i) => (
        <p key={`${i.seq}-${i.code}`} className="error">
          {i.seq != null ? `Line ${i.seq}: ` : ""}
          {i.message}
        </p>
      ))}

      <h2>Step by step</h2>
      <div className="card" style={{ overflowX: "auto" }}>
        <table className="steps">
          <tbody>
            {hand.replay.steps.map((s, n) => (
              <tr key={n} className={s.seat === heroSeat ? "hero" : ""}>
                <td>
                  {s.time ? (
                    <button type="button" className="chip" disabled={!seek} onClick={() => seek?.(s.time!.earliest_ms)}>
                      {clock(s.time.earliest_ms)}
                    </button>
                  ) : (
                    <span className="muted">no time</span>
                  )}
                </td>
                <td>
                  <div>{s.label}</div>
                  <div className="muted">
                    {Object.entries(s.after.stacks)
                      .map(([seat, chips]) => `${alias[seat] ?? seat} ${chips}`)
                      .join(" · ")}
                  </div>
                </td>
                <td className="num">
                  <div>pot {s.before.pot} → {s.after.pot}</div>
                  {s.before.to_call ? <div className="muted">to call {s.before.to_call}</div> : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {saved && <p className="card" role="status">{saved}</p>}
      {error && <p className="error">{error}</p>}
      <Correction
        key={`${hand.revision}-${hand.latest_revision}`}
        hand={hand}
        url={base}
        onSaved={(message) => {
          setSaved(message);
          setViewing(null);
          load(null);
        }}
      />

      <h2>Versions</h2>
      {revisions.map((r) => (
        <div key={r.revision} className="card row">
          <strong>Revision {r.revision}</strong>
          <span className={`badge ${r.status === "verified" ? "good" : "bad"}`}>{r.status}</span>
          {r.revision === hand.active_revision && <span className="badge good">in use</span>}
          <span className="grow muted">{r.note || "original entry"}</span>
          <button type="button" disabled={r.revision === hand.revision} onClick={() => setViewing(r.revision)}>
            View
          </button>
          {r.issues.map((i) => (
            <div key={`${i.seq}-${i.code}`} className="error" style={{ width: "100%" }}>
              {i.message}
            </div>
          ))}
        </div>
      ))}
    </>
  );
}

function Correction({ hand, url, onSaved }: { hand: StoredHand; url: string; onSaved: (message: string) => void }) {
  const [events, setEvents] = useState(hand.record.events.join("\n"));
  const [note, setNote] = useState("");
  const [failure, setFailure] = useState("");
  const [busy, setBusy] = useState(false);

  async function save(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      const lines = events.split("\n").map((l) => l.trim()).filter(Boolean);
      const saved = await api<StoredHand>(url, {
        method: "PUT",
        body: JSON.stringify({ base_revision: hand.latest_revision, note, record: { ...hand.record, events: lines } }),
      });
      onSaved(
        saved.status === "verified"
          ? `Saved revision ${saved.revision}. It replays correctly and is now in use.`
          : `Saved revision ${saved.revision}, but it does not replay: ${saved.replay.issues.map((i) => i.message).join("; ")}. The version in use did not change.`,
      );
    } catch (err) {
      setFailure(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <details className="card">
      <summary>Fix this hand</summary>
      <form onSubmit={save}>
        <p className="muted">
          One line per event, for example <code>s3 raise 75 @01:55.4</code>. Saving makes a new version; old versions are
          kept. Based on revision {hand.latest_revision}.
        </p>
        <textarea rows={Math.min(18, hand.record.events.length + 2)} value={events} onChange={(e) => setEvents(e.target.value)} aria-label="Events" />
        <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="What did you fix?" style={{ marginTop: 6 }} aria-label="Note" />
        <div className="row" style={{ marginTop: 6 }}>
          <button className="primary" disabled={busy}>Save new version</button>
          {failure && <span className="error">{failure}</span>}
        </div>
      </form>
    </details>
  );
}
