import type { Metadata } from "next";

import { XRayApp } from "@/components/XRayApp";
import { buildOgImageUrl } from "@/lib/share";
import { MODEL_LABEL } from "@/lib/xray-protocol";

type SearchParams = Promise<{ [key: string]: string | string[] | undefined }>;

/** First value of a possibly-repeated query param. */
function one(v: string | string[] | undefined): string | undefined {
  return (Array.isArray(v) ? v[0] : v)?.trim() || undefined;
}

export async function generateMetadata({
  searchParams,
}: {
  searchParams: SearchParams;
}): Promise<Metadata> {
  const sp = await searchParams;
  const prompt = one(sp.prompt)?.slice(0, 160);
  if (!prompt) return {}; // inherit the layout's default card

  const token = one(sp.token);
  const probRaw = one(sp.prob);
  const prob = probRaw !== undefined ? Number(probRaw) : undefined;
  const image = buildOgImageUrl({
    prompt,
    token,
    prob: prob !== undefined && !Number.isNaN(prob) ? prob : undefined,
  });

  const title = `“${prompt}”`;
  const description = `Watch ${MODEL_LABEL} process “${prompt}” — attention patterns and per-layer predictions, layer by layer.`;

  return {
    title,
    description,
    openGraph: { title: `${title} · LLM X-Ray`, description, images: [image] },
    twitter: { card: "summary_large_image", title: `${title} · LLM X-Ray`, description, images: [image] },
  };
}

export default async function Home({
  searchParams,
}: {
  searchParams: SearchParams;
}) {
  const sp = await searchParams;
  const initialPrompt = one(sp.prompt)?.slice(0, 200);
  return <XRayApp initialPrompt={initialPrompt} />;
}
