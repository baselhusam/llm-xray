import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a self-contained server bundle (.next/standalone/server.js) so the
  // Docker runtime image stays minimal. public/ + .next/static are copied in
  // by the Dockerfile, since standalone does not include them.
  output: "standalone",
};

export default nextConfig;
