"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type Session } from "@/lib/api";

export default function Home() {
  const [sessions, setSessions] = useState<Session[] | null>(null);
  const [error, setError] = useState("");

  async function load() {
    try {
      setSessions(await api<Session[]>("/api/sessions"));
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function end(key: string) {
    try {
      await api(`/api/sessions/${encodeURIComponent(key)}/end`, { method: "POST" });
      await load();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <>
      <h1>Sessions</h1>
      {error && <p className="error">{error}</p>}
      {sessions === null && !error && <p className="muted">Loading…</p>}
      {sessions?.length === 0 && (
        <p className="muted">No sessions yet. Run <code>poker-memory demo</code> to load the sample game.</p>
      )}
      {sessions?.map((s) => (
        <div className="card" key={s.key}>
          <div className="row">
            <Link className="grow" href={`/s/${encodeURIComponent(s.key)}`}>
              <strong>{s.title}</strong>
            </Link>
            <span className={`badge ${s.status === "ended" ? "good" : "warn"}`}>{s.status}</span>
          </div>
          <div className="row muted" style={{ marginTop: 6 }}>
            <span>{s.key}</span>
            <span>
              {s.stakes.sb}/{s.stakes.bb} {s.stakes.currency}
            </span>
            <span>
              {s.verified}/{s.hand_count} hands checked
            </span>
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            {s.synthetic && <span className="badge">synthetic</span>}
            <span className={`badge ${s.consent === "all_participants" ? "good" : "bad"}`}>
              {s.consent === "all_participants" ? "everyone consented" : "no consent"}
            </span>
            {s.video_url && <span className="badge">video</span>}
            {s.status !== "ended" && (
              <button type="button" onClick={() => end(s.key)}>
                End session
              </button>
            )}
          </div>
        </div>
      ))}
    </>
  );
}
