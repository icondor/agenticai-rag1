// scripturi/indexeaza.ts
//
// Indexează PDF-urile din `biblioteca/`. Rulează O DATĂ, când se schimbă
// corpusul — nu la fiecare pornire de server.
//
//   pnpm indexeaza

import { indexeazaBiblioteca } from "@/lib/rag/indexare";

const folder = process.argv[2] ?? "biblioteca";

console.log(`Indexez PDF-urile din «${folder}»…\n`);
const raport = await indexeazaBiblioteca(folder);

console.log("\n═══ GATA ═══");
console.table(
  raport.carti.map((c) => ({
    carte: c.titlu,
    pagini: c.pagini,
    chunkuri: c.chunkuri,
    secunde: (c.durataMs / 1000).toFixed(1),
  })),
);
console.log(`total: ${raport.totalChunkuri} chunkuri · ${(raport.durataTotalaMs / 1000).toFixed(1)}s`);
console.log(`embeddings: ${raport.eticheta}`);
