"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Chat } from "@/components/Chat";
import { Video, type VideoHandle } from "@/components/Video";
import { api, type HandRow, type Session } from "@/lib/api";

export default function SessionPage() {
  const key = decodeURIComponent(useParams<{ key: string }>().key);
  const [session, setSession] = useState<Session | null>(null);
  const [hands, setHands] = useState<HandRow[]>([]);
  const [error, setError] = useState("");
  const video = useRef<VideoHandle>(null);

  useEffect(() => {
    Promise.all([api<Session[]>("/api/sessions"), api<HandRow[]>(`/api/sessions/${encodeURIComponent(key)}/hands`)])
      .then(([all, rows]) => {
        setSession(all.find((s) => s.key === key) ?? null);
        setHands(rows);
      })
      .catch((e) => setError(String(e)));
  }, [key]);

  if (error) return <p className="error">{error}</p>;
  if (!session) return <p className="muted">Loading…</p>;

  return (
    <>
      {session.video_url && <Video ref={video} src={session.video_url} source={session.source} />}
      <h1>{session.title}</h1>
      {session.status !== "ended" && (
        <p className="card">This session is still open. Analysis starts after it ends.</p>
      )}
      <Chat
        sessionKey={key}
        synthetic={session.synthetic}
        onSeek={session.video_url ? (ms) => video.current?.seek(ms) : undefined}
      />
      <h2>Hands</h2>
      {hands.map((h) => (
        <Link key={h.index} href={`/s/${encodeURIComponent(key)}/h/${h.index}`} style={{ textDecoration: "none", color: "inherit" }}>
          <div className="card row">
            <strong>#{h.index}</strong>
            <span className="cards">{h.hero_cards.join(" ")}</span>
            <span className="grow muted">{h.summary}</span>
            <span className={`badge ${h.status === "verified" ? "good" : "bad"}`}>
              {h.status === "verified" ? "checked" : "needs fix"}
            </span>
          </div>
        </Link>
      ))}
    </>
  );
}
