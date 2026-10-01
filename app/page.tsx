// app/page.tsx
//
// Pagina de start trimite direct la demo-ul bibliotecii.

import { redirect } from "next/navigation";

export default function Home() {
  redirect("/biblioteca");
}
