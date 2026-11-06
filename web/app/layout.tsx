import type { Metadata, Viewport } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Poker Memory",
  description: "Review your poker hands after the session, with checked math and video moments.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="top">
          <Link href="/">Poker Memory</Link>
          <Link href="/demo">Hand study</Link>
          <span className="muted">after-session review</span>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
