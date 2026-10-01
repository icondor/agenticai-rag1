// scripturi/indexeaza-demo.ts
//
// Indexează corpusul sintetic din `verificare/corpus-test.ts`, ca aplicația
// să pornească fără cărțile din bibliotecă.
//
//   pnpm indexeaza:demo
//
// ⚠️ Folosește embeddings REALE (costă câțiva bani, foarte puțini) — spre
//    deosebire de `pnpm verifica`, care merge cu embeddings false.

import { chunkStructural } from "@/lib/rag/chunker";
import { creeazaEmbeddingsCuCache, etichetaEmbeddings } from "@/lib/rag/embeddings";
import { VectorStoreSimplu } from "@/lib/rag/vector-store";
import { CALE_INDEX } from "@/lib/rag/indexare";
import { CORPUS_TEST } from "@/verificare/corpus-test";

const store = new VectorStoreSimplu(creeazaEmbeddingsCuCache(), {});

let total = 0;
for (const carte of CORPUS_TEST) {
  const bucati = await chunkStructural(carte.randuri, carte.titlu);
  await store.addDocuments(bucati);
  console.log(`  ${carte.titlu}: ${bucati.length} chunkuri`);
  total += bucati.length;
}

await store.salveaza(CALE_INDEX, etichetaEmbeddings());
console.log(`\n${total} chunkuri · ${etichetaEmbeddings()} · salvat în ${CALE_INDEX}`);
