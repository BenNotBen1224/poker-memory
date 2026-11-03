"use client";

import { useImperativeHandle, useRef } from "react";
import type { Source } from "@/lib/api";

export type VideoHandle = { seek: (ms: number) => void };

export function Video({ src, source, ref }: { src: string; source?: Source | null; ref: React.Ref<VideoHandle> }) {
  const video = useRef<HTMLVideoElement>(null);
  useImperativeHandle(ref, () => ({
    seek(ms: number) {
      const v = video.current;
      if (!v) return;
      v.currentTime = ms / 1000;
      v.play().catch(() => {});
      v.scrollIntoView({ block: "nearest", behavior: "smooth" });
    },
  }));
  return (
    <div className="video">
      <video ref={video} src={src} controls playsInline preload="metadata" />
      {source && (
        <p className="muted credit">
          “<a href={source.url}>{source.title}</a>” by {source.author}, <a href={source.license_url}>{source.license}</a>.
          Shown unchanged.
        </p>
      )}
    </div>
  );
}
