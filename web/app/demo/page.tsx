"use client";

import { useEffect, useRef, useState } from "react";
import { api, Source } from "@/lib/api";
import styles from "./review.module.css";

type Review = {
  revision: number;
  case_sha256: string;
  lens: "decision" | "broadcast";
  source: Source;
  decision: {
    hero: string; opponent: string; hero_cards: string[]; board: string[];
    pot_before_call: number; extra_call: number; called_final_pot: number;
    break_even_share: number; fold_future_ev: number;
    stack_if_fold: number; stack_if_losing_call: number;
    scenarios: { share_percent: number; call_advantage: number }[];
  };
  timeline: { seq: number; label: string; street: string; time_ms: number; time_text: string; pot_after: number }[];
  players: { alias: string; position: string; initial_stack: number; observed_actions: number; hand_count: number; long_run_profile: string }[];
  hypotheses: { title: string; observation: string; possible_explanation: string; limitation: string; evidence_needed: string }[];
  reported_context: { title: string; text: string; author: string; published: string; url: string; kind: string; retrieval: string }[];
  limitations: string[];
  hindsight?: {
    opponent_cards: string[]; hero_hand: string; opponent_hand: string;
    actual_hero_net: number; known_card_call_net: number; additional_loss_avoided: number; video_url: string;
  };
};

const chips = (value: number) => value.toLocaleString("en-US", { maximumFractionDigits: 0 });
const compact = (value: number) => `${(value / 1_000_000).toFixed(2)}M`;
const signed = (value: number) => `${value > 0 ? "+" : ""}${compact(value)}`;
const symbols: Record<string, string> = { h: "♥", d: "♦", c: "♣", s: "♠" };

function Cards({ cards, hidden = false }: { cards?: string[]; hidden?: boolean }) {
  return <div className={styles.cards}>{(hidden ? ["??", "??"] : cards ?? []).map((card, i) =>
    <span key={i} className={`${styles.playingCard} ${"hd".includes(card[1]) ? styles.red : ""} ${hidden ? styles.cardBack : ""}`}
      aria-label={hidden ? "Opponent card unknown at decision time" : card}>
      {hidden ? "?" : <>{card[0] === "T" ? "10" : card[0]}<small>{symbols[card[1]]}</small></>}
    </span>)}</div>;
}

export default function DemoPage() {
  const [lens, setLens] = useState<"decision" | "broadcast">("decision");
  const [review, setReview] = useState<Review | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [share, setShare] = useState(30);
  const [selected, setSelected] = useState(14);
  const [videoError, setVideoError] = useState("");
  const video = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let active = true;
    setReview(null);
    setError("");
    setVideoError("");
    api<Review>(`/api/demo/wpt/review?lens=${lens}`).then(value => {
      if (active) setReview(value);
    }).catch(e => { if (active) setError(e instanceof Error ? e.message : "Review request failed"); });
    return () => { active = false; };
  }, [lens, retry]);

  const chooseMoment = (seq: number, ms: number) => {
    setSelected(seq);
    if (lens === "broadcast" && video.current) video.current.currentTime = ms / 1000;
  };

  if (error) return <div className={styles.demo}><section className={styles.panel} role="alert">
    <span className={styles.eyebrow}>Review unavailable</span><h1>We cannot verify this example.</h1>
    <p>{error}</p><p>No substitute analysis is shown. The ended session, active transcription and source video must match the reviewed case.</p>
    <button onClick={() => setRetry(retry + 1)}>Retry verification</button></section></div>;
  if (!review || review.lens !== lens) return <div className={styles.demo} role="status">Verifying the reviewed hand and source video…</div>;

  const d = review.decision;
  const scenario = d.scenarios[share];
  const moment = review.timeline.find(m => m.seq === selected) ?? review.timeline[0];
  const h = lens === "broadcast" ? review.hindsight : undefined;
  const x = (p: number) => 62 + p * 5.3;
  const y = (ev: number) => 176 - ev / d.called_final_pot * 150;
  const thresholdX = x(d.break_even_share * 100);

  return <div className={styles.demo}>
    <header className={styles.heading}>
      <div><span className={styles.eyebrow}>POKER MEMORY / ONE-HAND STUDY</span>
        <h1>A big fold. <span>A better question.</span></h1>
        <p>Ian vs Joe · WPT Rolling Thunder · Five players remaining</p></div>
      <div className={styles.provenance}><span className={styles.dot} /> Reviewed transcription
        <small>Replay verified · Revision {review.revision}<br />Not a fully accurate automated read</small></div>
    </header>

    <nav className={styles.lenses} aria-label="Review lens">
      <button aria-pressed={lens === "decision"} onClick={() => setLens("decision")}>01 · Decision lens <small>Opponent cards hidden</small></button>
      <button aria-pressed={lens === "broadcast"} onClick={() => setLens("broadcast")}>02 · Broadcast &amp; outcome <small>Reveal hindsight + source video</small></button>
    </nav>

    <div className={styles.grid}>
      <section className={`${styles.panel} ${styles.tablePanel}`} aria-label="River hand diagram">
        <div className={styles.sectionTitle}><h2>The river decision</h2><span className={styles.tag}>OBSERVED</span></div>
        <p className={styles.subtle}>After Joe&apos;s shove, before Ian&apos;s fold. Earlier chips are already committed.</p>
        <div className={styles.pokerTable}>
          <div className={styles.opponent}><b>Joe · BB <span>inferred</span></b>
            <Cards cards={h?.opponent_cards} hidden={!h} />
            <small>{h ? "Broadcast camera · hindsight only" : "Cards unknown to Ian"}</small></div>
          <div className={styles.board}><span className={styles.eyebrow}>POT BEFORE CALL</span><strong>{compact(d.pot_before_call)}</strong>
            <Cards cards={d.board} /></div>
          <div className={styles.hero}><Cards cards={d.hero_cards} /><b>Ian · SB <span>inferred</span></b><small>Recorded action: fold</small></div>
        </div>
        <div className={styles.priceStrip}>
          <div><small>Extra to call</small><strong>{compact(d.extra_call)}</strong></div>
          <div><small>Final pot if called</small><strong>{compact(d.called_final_pot)}</strong></div>
          <div><small>Break-even share</small><strong>{(d.break_even_share * 100).toFixed(4)}%</strong></div>
        </div>
        <p className={styles.subtle}>Record-based replay · chips, not money. Under the broadcast stacks: {chips(d.stack_if_fold)} left if folding; {chips(d.stack_if_losing_call)} after a losing call. <strong>Published reporting disputes those stacks; actual survival implications are unknown.</strong></p>
      </section>

      <section className={`${styles.panel} ${styles.scenario}`}>
        <div className={styles.sectionTitle}><h2>When would calling pay?</h2><span className={`${styles.tag} ${styles.amber}`}>WHAT-IF</span></div>
        <p>Move an assumption, not a player statistic.</p>
        <label htmlFor="share">Share of Joe&apos;s shoves Ian beats <strong>{share}%</strong></label>
        <input id="share" type="range" min="0" max="100" step="1" value={share} onChange={e => setShare(Number(e.target.value))} />
        <p className={styles.assumption}>Includes bluffs <em>and</em> weaker value hands. Ties excluded (0%); all remaining hands beat Ian. The initial 30% is an illustration, not an estimate. No observed frequency or default opponent range.</p>
        <div className={styles.ev} data-testid="scenario-ev">
          <span>Conditional call advantage over folding</span><strong className={scenario.call_advantage >= 0 ? styles.positive : styles.negative}>{scenario.call_advantage > 0 ? "+" : ""}{chips(scenario.call_advantage)} chips</strong>
          <small>Decision-point fold future EV = {d.fold_future_ev}. Sunk chips excluded.</small>
        </div>
        <svg className={styles.chart} viewBox="0 0 640 258" role="img" aria-label="Conditional call advantage crosses zero at 29.3553 percent">
          <defs><linearGradient id="lineGradient"><stop stopColor="#f4b77a" /><stop offset="1" stopColor="#7ee6c4" /></linearGradient></defs>
          <text x="18" y="23">Call − fold · chips</text>
          <line x1="62" y1={y(0)} x2="592" y2={y(0)} stroke="#7b8890" strokeDasharray="4 5" />
          <text x="21" y={y(0) + 5}>0</text>
          <line x1={thresholdX} y1="45" x2={thresholdX} y2="207" stroke="#f4b77a" strokeDasharray="3 5" />
          <path d={`M ${x(0)} ${y(d.scenarios[0].call_advantage)} L ${x(100)} ${y(d.scenarios[100].call_advantage)}`} fill="none" stroke="url(#lineGradient)" strokeWidth="4" />
          <circle cx={x(share)} cy={y(scenario.call_advantage)} r="7" fill="#ecfff8" stroke="#113c35" strokeWidth="3" />
          <text x={thresholdX + 8} y="61">29.36% break-even</text>
          <text x="62" y="230">0%</text><text x="560" y="230">100%</text>
          <text x="165" y="251">Hypothetical share of shoves Ian beats</text>
        </svg>
        <div className={styles.formula}>p × {chips(d.called_final_pot)} − {chips(d.extra_call)}</div>
        <p className={styles.subtle}>This is sensitivity arithmetic, not advice about Joe. Broad-range solver +EV is not a player-specific verdict. No payout/ICM estimate.</p>
      </section>
    </div>

    <section className={styles.panel}>
      <div className={styles.sectionTitle}><h2>Follow the evidence</h2><span className={styles.tag}>REVIEWED ACTIONS</span></div>
      <p className={styles.subtle}>Select a moment. In broadcast mode, the video seeks to its approximate clock; it does not autoplay.</p>
      <div className={styles.timeline} aria-label="Action timeline">
        {review.timeline.map(m => <button key={m.seq} aria-pressed={selected === m.seq} onClick={() => chooseMoment(m.seq, m.time_ms)}>
          <small>{m.time_text} · {m.street}</small><b>{m.label}</b></button>)}
      </div>
      <div className={styles.focus}><span>Selected evidence · {moment.time_text}</span><b>{moment.label}</b><span>Pot after action {compact(moment.pot_after)}</span></div>
      {h ? <div className={styles.broadcast}>
        <div className={styles.warning}>HINDSIGHT: the unredacted broadcast contains hidden-card overlays. These were not available to the players.</div>
        <video ref={video} src={h.video_url} controls playsInline preload="metadata"
          onLoadedMetadata={() => { if (video.current) video.current.currentTime = moment.time_ms / 1000; }}
          onError={() => setVideoError("Source video failed to load or is no longer eligible. No replacement footage is shown.")} />
        {videoError && <p role="alert" className={styles.negative}>{videoError}</p>}
        <p className={styles.credit}>“<a href={review.source.url}>{review.source.title}</a>” by {review.source.author} · <a href={review.source.license_url}>{review.source.license}</a> · Shown unchanged.</p>
        <div className={styles.outcomes}>
          <div><span className={styles.tag}>RECORDED OUTCOME</span><h3>Ian folded: {signed(h.actual_hero_net)}</h3><p>Record-based full-hand net, including earlier contributions; not a verified physical-table stack claim.</p></div>
          <div><span className={styles.tag}>KNOWN-CARD COUNTERFACTUAL</span><h3>Call: {signed(h.known_card_call_net)}</h3><p>Under the reviewed record, Joe&apos;s {h.opponent_hand.toLowerCase()} beats Ian&apos;s {h.hero_hand.toLowerCase()}. Folding avoided an extra {compact(h.additional_loss_avoided)} against these revealed cards.</p></div>
        </div>
        <p className={styles.warning}>Outcome is not strategy: saving chips against this particular hand does not establish that folding has higher expected value against an unknown range.</p>
      </div> : <div className={styles.locked}><span>◈</span><div><b>Broadcast stays hidden in the decision lens</b><p>The video includes Joe&apos;s cards. Select “Broadcast &amp; outcome” above to reveal the source and seek this moment.</p></div></div>}
    </section>

    <section className={styles.context}>
      <div><span className={styles.eyebrow}>CONTEXT, NOT A PERSONALITY TEST</span><h2>One hand is not a player profile.</h2><p>No measured bluff rate, long-run tendencies, motives or reliable thinking times are available.</p></div>
      {review.players.map(p => <div className={styles.profile} key={p.alias}><h3>{p.alias}<span>n = {p.hand_count} hand</span></h3>
        <p>{p.position} · inferred</p><b>{chips(p.initial_stack)} broadcast starting chips</b><p>Actual starting stacks disputed.</p><p>{p.observed_actions} observed actions · Long-run profile: {p.long_run_profile}</p></div>)}
    </section>

    <section className={styles.panel} aria-label="Attributed reporting">
      <div className={styles.sectionTitle}><h2>What the reporting adds</h2><span className={`${styles.tag} ${styles.amber}`}>ATTRIBUTED, NOT MEASURED</span></div>
      <div className={styles.reports}>{review.reported_context.map(item => <article key={item.url}>
        <h3>{item.title}</h3><p>{item.text}</p><a href={item.url}>{item.author} · {item.published}</a>
        <p className={styles.subtle}>{item.retrieval}</p>
      </article>)}</div>
    </section>

    <section><div className={styles.sectionTitle}><h2>Why might behavior and outcome differ?</h2><span className={`${styles.tag} ${styles.amber}`}>TESTABLE HYPOTHESES</span></div>
      <div className={styles.hypotheses}>{review.hypotheses.map((item, i) => <article className={styles.panel} key={item.title}>
        <span className={styles.hypothesisNumber}>0{i + 1}</span><h3>{item.title}</h3>
        <p><b>Observed</b> {item.observation}</p><p><b>Possible explanation</b> {item.possible_explanation}</p>
        <p className={styles.subtle}><b>Limit</b> {item.limitation}</p><div className={styles.needed}><b>To test this</b><p>{item.evidence_needed}</p></div>
      </article>)}</div>
    </section>
    <footer className={styles.panel}><h2>Accuracy &amp; limits</h2><ul>{review.limitations.map(text => <li key={text}>{text}</li>)}</ul>
      <small>Case binding SHA-256 <code>{review.case_sha256}</code> · Read-only review, no conversation writes or new detection run.</small></footer>
  </div>;
}
