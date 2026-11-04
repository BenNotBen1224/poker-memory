export type VideoTime = { earliest_ms: number; latest_ms: number; text: string };

export type Source = { title: string; author: string; url: string; license: string; license_url: string };

export type Session = {
  key: string;
  title: string;
  consent: "all_participants" | "open_license" | "none";
  synthetic: boolean;
  stakes: { sb: number; bb: number; ante: number; currency: string };
  source: Source | null;
  status: string;
  ended_at: string | null;
  hand_count: number;
  verified: number;
  video_url: string | null;
};

export type Issue = { seq: number | null; code: string; message: string };

export type HandRow = {
  index: number;
  revision: number;
  active_revision: number | null;
  latest_revision: number;
  status: string;
  summary: string;
  hero_cards: string[];
  combo: string;
  dealt_at: string | null;
  issues: Issue[];
};

export type Snapshot = {
  street: string;
  board: string[];
  pot: number;
  stacks: Record<string, number>;
  to_call: number | null;
};

export type Step = {
  seq: number | null;
  raw: string;
  kind: string;
  seat: number | null;
  label: string;
  time: VideoTime | null;
  before: Snapshot;
  after: Snapshot;
};

export type HandRecord = {
  index: number;
  button: number;
  seats: { seat: number; alias: string; stack: number; note?: string; hole_cam?: string | null }[];
  hero: { seat: number; cards: string };
  dealt_at: string | null;
  events: string[];
  result: Record<string, number> | null;
};

export type StoredHand = {
  session_key: string;
  index: number;
  revision: number;
  active_revision: number | null;
  latest_revision: number;
  status: string;
  note: string;
  created_at: string;
  record: HandRecord;
  replay: { status: string; steps: Step[]; issues: Issue[]; board: string[]; payoffs: Record<string, number> | null };
};

export type Revision = { revision: number; status: string; note: string; created_at: string; issues: Issue[] };

export type Fact = { name: string; kind: "observed" | "derived" | "estimated" | "unknown"; text: string; value?: unknown };
export type Solver = { moves: string[]; ev: number[]; taken: number; best: number };
export type Evidence = { label: string; start_ms: number; end_ms: number };
export type ToolRun = { tool: string; hand_index: number | null; revision: number | null; input: object; output: object };

export type Answer = {
  kind: "analysis" | "candidates" | "not_found" | "needs_fix" | "refused";
  text: string;
  understood: string;
  conversation_id: number | null;
  session_key: string | null;
  hand: number | null;
  revision: number | null;
  facts: Fact[];
  candidates: { index: number; revision: number; summary: string }[];
  evidence: Evidence[];
  video_url: string | null;
  assumptions: string[];
  tools: ToolRun[];
};

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(body?.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`);
  return body as T;
}

export function clock(ms: number): string {
  const s = ms / 1000;
  const m = Math.floor(s / 60);
  const rest = (s - m * 60).toFixed(1).padStart(4, "0");
  return `${String(m).padStart(2, "0")}:${rest}`;
}
