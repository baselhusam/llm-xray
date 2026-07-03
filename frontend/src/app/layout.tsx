import type { Metadata } from "next";
import { Hanken_Grotesk, Spline_Sans_Mono } from "next/font/google";
import "./globals.css";

import { SITE_URL, buildOgImageUrl } from "@/lib/share";

// Brand type: Hanken Grotesk (humanist sans, everything human-facing) +
// Spline Sans Mono (labels, code, the instrument's voice).
const hanken = Hanken_Grotesk({
  variable: "--font-hanken",
  subsets: ["latin"],
});

const splineMono = Spline_Sans_Mono({
  variable: "--font-spline",
  subsets: ["latin"],
});

const TITLE = "LLM X-Ray";
const DESCRIPTION = "Type a prompt. Watch an AI think. Share the X-ray.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: TITLE, template: "%s · LLM X-Ray" },
  description: DESCRIPTION,
  openGraph: {
    title: TITLE,
    description: DESCRIPTION,
    url: SITE_URL,
    siteName: TITLE,
    images: [{ url: buildOgImageUrl({ prompt: "" }), width: 1200, height: 630 }],
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
    images: [buildOgImageUrl({ prompt: "" })],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      // Dark mode is the default — visualizations are designed for dark backgrounds.
      className={`dark ${hanken.variable} ${splineMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
