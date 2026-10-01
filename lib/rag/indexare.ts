// lib/rag/indexare.ts
//
// ═══════════════════════════════════════════════════════════════════════
// PIPELINE-UL DE INDEXARE, CAP-COADĂ
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 CEA MAI IMPORTANTĂ IDEE DIN LAB, și cea pe care tutorialele o ascund:
//
//    RAG are DOUĂ pipeline-uri, complet separate.
//
//    ┌─ INDEXARE (offline) ─────────────────────────────────────────┐
//    │  PDF → extragere → chunking → embeddings → vector store      │
//    │  Rulează O DATĂ, când se schimbă corpusul.                    │
//    │  Durează minute. Costă bani. Nu se întâmplă în timpul chat-ului│
//    └───────────────────────────────────────────────────────────────┘
//
//    ┌─ INTEROGARE (online) ────────────────────────────────────────┐
//    │  întrebare → embedding → căutare → context → model → răspuns │
//    │  Rulează la FIECARE întrebare. Durează milisecunde + un apel. │
//    └───────────────────────────────────────────────────────────────┘
//
//    Fișierul ăsta e primul pipeline. Nu e chemat niciodată din interfață.
//
// ⚠️ Confuzia dintre cele două e cauza numărul unu a aplicațiilor de RAG
//    lente: cineva reindexează la fiecare cerere, fără să-și dea seama.

import { readdir, stat } from "node:fs/promises";
import { extname, join } from "node:path";
import type { Document } from "@langchain/core/documents";
import { extrageCarte } from "@/lib/rag/extrage-pdf";
import { chunkStructural, statistici } from "@/lib/rag/chunker";
import { creeazaEmbeddingsCuCache, etichetaEmbeddings } from "@/lib/rag/embeddings";
import { VectorStoreSimplu } from "@/lib/rag/vector-store";

export const CALE_INDEX = ".cache/index-biblioteca.json";

export interface RezultatIndexare {
  carti: { titlu: string; pagini: number; chunkuri: number; durataMs: number }[];
  totalChunkuri: number;
  durataTotalaMs: number;
  eticheta: string;
}

/**
 * Indexează toate PDF-urile dintr-un folder.
 *
 * @param folder  unde stau cărțile, ex. «biblioteca/»
 * @param caleIndex  unde salvăm vectorii
 */
export async function indexeazaBiblioteca(
  folder: string,
  caleIndex = CALE_INDEX,
): Promise<RezultatIndexare> {
  const pornit = Date.now();
  const fisiere = (await readdir(folder)).filter(
    (f) => extname(f).toLowerCase() === ".pdf",
  );

  if (fisiere.length === 0) {
    throw new Error(
      `Niciun PDF în «${folder}». Pune cărțile acolo sau folosește corpusul de test.`,
    );
  }

  const embeddings = creeazaEmbeddingsCuCache();
  const store = new VectorStoreSimplu(embeddings, {});
  const raport: RezultatIndexare["carti"] = [];
  let toate: Document[] = [];

  for (const fisier of fisiere) {
    const pornitCarte = Date.now();
    // Titlul cărții = numele fișierului, fără extensie. Apare în citări,
    // deci denumește fișierele cum vrei să scrie în răspuns.
    const titlu = fisier.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ");

    console.log(`\n[indexare] ${titlu}`);
    const carte = await extrageCarte(join(folder, fisier), titlu);
    console.log(
      `  extras: ${carte.statistici.pagini} pagini, ${carte.statistici.randuri} rânduri`,
    );
    console.log(`  roluri: ${JSON.stringify(carte.statistici.peRoluri)}`);

    const bucati = await chunkStructural(carte.randuri, titlu);
    const st = statistici(bucati);
    console.log(
      `  chunkuri: ${st.chunkuri} · medie ${st.lungimeMedie} car. · ${st.cuAncora} cu ancoră`,
    );

    toate = toate.concat(bucati);
    raport.push({
      titlu,
      pagini: carte.statistici.pagini,
      chunkuri: bucati.length,
      durataMs: Date.now() - pornitCarte,
    });
  }

  // ⚠️ Embeddings-urile se cer în LOT, nu unul câte unul.
  //    Un apel cu 500 de texte e mult mai rapid decât 500 de apeluri —
  //    latența de rețea domină, nu calculul.
  console.log(`\n[indexare] calculez embeddings pentru ${toate.length} chunkuri…`);
  await store.addDocuments(toate);

  await store.salveaza(caleIndex, etichetaEmbeddings());
  console.log(`[indexare] salvat în ${caleIndex}`);

  return {
    carti: raport,
    totalChunkuri: toate.length,
    durataTotalaMs: Date.now() - pornit,
    eticheta: etichetaEmbeddings(),
  };
}

// ═══════════════════════════════════════════════════════════════════════
// Încărcarea indexului, pentru pipeline-ul de interogare
// ═══════════════════════════════════════════════════════════════════════
//
// ⚠️ Ținut pe globalThis, ca checkpointer-ul din Lab 8: în dev, Next.js
//    recompilează modulele la fiecare editare și golește starea de modul.
//    Fără asta, reîncarci de pe disc câteva mii de vectori la fiecare
//    salvare de fișier.
//
// 🔍 Ținem minte și data fișierului. Dacă rulezi `pnpm indexeaza` cu
//    serverul pornit, următoarea întrebare vede indexul nou, fără restart.

const globalIndex = globalThis as typeof globalThis & {
  __storeBiblioteca?: { store: VectorStoreSimplu; modificatLa: number; cale: string };
};

export async function incarcaIndex(caleIndex = CALE_INDEX) {
  let modificatLa: number;
  try {
    modificatLa = (await stat(caleIndex)).mtimeMs;
  } catch {
    throw new Error(
      `Nu găsesc indexul la «${caleIndex}». Rulează întâi: pnpm indexeaza`,
    );
  }

  const inMemorie = globalIndex.__storeBiblioteca;
  if (inMemorie && inMemorie.cale === caleIndex && inMemorie.modificatLa === modificatLa) {
    return inMemorie.store;
  }

  const store = new VectorStoreSimplu(creeazaEmbeddingsCuCache(), {});
  const n = await store.incarca(caleIndex);

  // ⚠️ Indexul și întrebările trebuie să vină de la ACELAȘI model de
  //    embeddings. Dacă ai schimbat EMBEDDINGS_PROVIDER sau EMBEDDINGS_MODEL
  //    după indexare, scorurile n-ar mai însemna nimic — oprim aici.
  const eticheta = etichetaEmbeddings();
  if (store.eticheta && store.eticheta !== eticheta) {
    throw new Error(
      `Indexul a fost construit cu «${store.eticheta}», dar configurația curentă ` +
        `e «${eticheta}». Reindexează (pnpm indexeaza) sau revino la modelul vechi.`,
    );
  }

  console.log(`[index] încărcat: ${n} chunkuri · ${eticheta}`);
  globalIndex.__storeBiblioteca = { store, modificatLa, cale: caleIndex };
  return store;
}

/** Golește indexul din memorie — forțează reîncărcarea de pe disc. */
export function uitaIndexul() {
  globalIndex.__storeBiblioteca = undefined;
}
