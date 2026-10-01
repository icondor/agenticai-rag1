// next.config.ts

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["pdfjs-dist"],
  allowedDevOrigins: ["192.168.50.170"],
};

export default nextConfig;
