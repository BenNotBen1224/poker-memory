"use client";

import { useImperativeHandle, useRef } from "react";

export type VideoHandle = { seek: (ms: number) => void };

export function Video({ src, ref }: { src: string; ref: React.Ref<VideoHandle> }) {
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
    </div>
  );
}
