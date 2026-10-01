// scripturi/evalueaza.ts
//
// Măsoară calitatea retrieverului pe indexul curent.
//
//   pnpm evalueaza
//
// 🎯 Rulează-l ÎNAINTE și DUPĂ orice schimbare de chunking, de model de
//    embeddings sau de k. Fără cifre, «e mai bun acum» e o părere.

import { incarcaIndex } from "@/lib/rag/indexare";
import { comparaConfiguratii, evalueazaRetrieval, afiseazaRaport } from "@/lib/rag/evaluare";
import { etichetaEmbeddings } from "@/lib/rag/embeddings";
import { CAZURI_TEST } from "@/verificare/corpus-test";

const store = await incarcaIndex();

// ⚠️ Cazurile de test sunt scrise pentru corpusul sintetic (pnpm indexeaza:demo).
//    Pe cărțile tale, cifrele au sens doar dacă scrii cazuri pentru ele,
//    în același format ca CAZURI_TEST din verificare/corpus-test.ts.
console.log(
  "⚠️  Cazurile de evaluare sunt pentru corpusul demo. Pentru cărțile tale, " +
    "scrie cazuri proprii în verificare/corpus-test.ts.",
);
console.log(`index: ${store.numarDocumente} chunkuri · ${etichetaEmbeddings()}`);
console.log(`cărți: ${store.valoriDistincte("carte_sursa").join(", ")}`);

afiseazaRaport(await evalueazaRetrieval(store, CAZURI_TEST, { k: 5 }), "configurația curentă");

await comparaConfiguratii(
  [
    { nume: "k=3", store, k: 3 },
    { nume: "k=5", store, k: 5 },
    { nume: "k=10", store, k: 10 },
    { nume: "k=5 + MMR", store, k: 5, cuDiversitate: true },
  ],
  CAZURI_TEST,
);

console.log(
  "\nPentru comparația de embeddings: schimbă EMBEDDINGS_PROVIDER, rulează\n" +
    "din nou `pnpm indexeaza:demo` (sau `pnpm indexeaza`) și apoi `pnpm evalueaza`.",
);
