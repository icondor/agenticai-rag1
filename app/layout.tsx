// app/layout.tsx

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lab 11 — RAG peste o bibliotecă",
  description: "RAG clasic vs. agentic, pe aceeași întrebare",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ro">
      <body className="min-h-full antialiased">{children}</body>
    </html>
  );
}
