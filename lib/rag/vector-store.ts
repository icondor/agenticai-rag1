// lib/rag/vector-store.ts
//
// ═══════════════════════════════════════════════════════════════════════
// PASUL 5 DIN PIPELINE — VECTOR STORE
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 Demistificarea zilei: un vector store e un ARRAY de vectori plus o
//    funcție de similaritate. Atât. Tot restul — indexare aproximativă,
//    persistență, sharding — sunt optimizări care contează la milioane de
//    documente, nu la câteva mii de chunk-uri de comentariu.
//
//    Fișierul ăsta are ~50 de linii de logică reală. Le scriem noi ca să
//    se vadă că nu e magie.
//
// ⚠️ DE CE NU FOLOSIM `MemoryVectorStore`:
//    În v1, singura implementare în memorie trăiește în `@langchain/classic`
//    — pachetul legacy pe care cursul îl predă ca «recunoaște și evită».
//    `@langchain/core` are doar clasa abstractă.
//
//    Deci: ori încălcăm regula de zero legacy, ori aducem un pachet cu
//    dependențe native (dureri de instalare), ori scriem 50 de linii.
//    Alegem a treia, și iese și cea mai bună lecție.
//
// 🔍 Extindem `VectorStore` din core, nu scriem o clasă de la zero. Așa
//    primim gratis `similaritySearch`, `similaritySearchWithScore` și,
//    cel mai important, `asRetriever()` — care ne leagă de restul
//    ecosistemului LangChain fără niciun adaptor.

import { VectorStore } from "@langchain/core/vectorstores";
import { Document, type DocumentInterface } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";

interface Intrare {
  vector: number[];
  document: DocumentInterface;
}

/**
 * Similaritatea cosinus: cât de mult „arată în aceeași direcție" doi vectori.
 *
 * 🔍 Rezultatul e între -1 și 1. Pentru embeddings de text, practic mereu
 *    între 0 și 1: 1 înseamnă identice ca sens, 0 înseamnă fără legătură.
 *
 * 🌍 Intuiția: nu contează cât de LUNGI sunt vectorii, ci unghiul dintre ei.
 *    Un paragraf de 50 de cuvinte și unul de 500 despre același subiect
 *    arată în aceeași direcție, chiar dacă au „mărimi" diferite.
 */
export function cosinus(a: number[], b: number[]): number {
  // ⚠️ Fără verificarea asta, vectori de lungimi diferite dau NaN, iar
  //    sortarea după NaN întoarce fragmente la întâmplare — fără eroare.
  //    Se întâmplă când indexul a fost construit cu alt model de embeddings.
  if (a.length !== b.length) {
    throw new Error(
      `Vectori de dimensiuni diferite (${a.length} vs ${b.length}): indexul a fost ` +
        `construit cu alt model de embeddings. Reindexează.`,
    );
  }

  let produs = 0;
  let normaA = 0;
  let normaB = 0;

  for (let i = 0; i < a.length; i++) {
    produs += a[i] * b[i];
    normaA += a[i] * a[i];
    normaB += b[i] * b[i];
  }

  const numitor = Math.sqrt(normaA) * Math.sqrt(normaB);
  return numitor === 0 ? 0 : produs / numitor;
}

export type FiltruMetadate = (document: DocumentInterface) => boolean;

export class VectorStoreSimplu extends VectorStore {
  private intrari: Intrare[] = [];

  /**
   * Ce model de embeddings a produs vectorii — salvat odată cu indexul.
   * Vectorii a două modele diferite nu se pot compara între ei.
   */
  eticheta?: string;

  /** Tipul filtrului — o funcție peste document, nu un obiect de query. */
  declare FilterType: FiltruMetadate;

  _vectorstoreType(): string {
    return "simplu";
  }

  // ─── Scriere ────────────────────────────────────────────────────────

  async addVectors(vectori: number[][], documente: DocumentInterface[]) {
    vectori.forEach((vector, i) =>
      this.intrari.push({ vector, document: documente[i] }),
    );
  }

  async addDocuments(documente: DocumentInterface[]) {
    // Aici se cheamă modelul de embeddings — partea care costă și durează.
    const vectori = await this.embeddings.embedDocuments(
      documente.map((d) => d.pageContent),
    );
    return this.addVectors(vectori, documente);
  }

  // ─── Citire ─────────────────────────────────────────────────────────

  /**
   * Căutarea propriu-zisă: compari întrebarea cu FIECARE document.
   *
   * ⚠️ Da, e O(n). Pentru 5.000 de chunk-uri de 768 de dimensiuni înseamnă
   *    ~4 milioane de înmulțiri — câteva milisecunde. Pentru 5 milioane de
   *    chunk-uri ar fi inacceptabil, și de acolo încep indexările aproximative
   *    (HNSW, IVF) pe care le au pgvector, Qdrant, Pinecone.
   *
   * 🎯 Filtrul se aplică ÎNAINTE de sortare — «pre-filtering». Diferența
   *    față de a filtra rezultatele după căutare e importantă: dacă ceri
   *    k=5 și apoi filtrezi, poți rămâne cu 1 rezultat. Aici ceri 5 DINTRE
   *    cele care trec filtrul.
   */
  async similaritySearchVectorWithScore(
    vectorIntrebare: number[],
    k: number,
    filtru?: FiltruMetadate,
  ): Promise<[DocumentInterface, number][]> {
    return this.intrari
      .filter((intrare) => (filtru ? filtru(intrare.document) : true))
      .map(
        (intrare) =>
          [intrare.document, cosinus(vectorIntrebare, intrare.vector)] as [
            DocumentInterface,
            number,
          ],
      )
      .sort((a, b) => b[1] - a[1])
      .slice(0, k);
  }

  // ─── Diversitate ────────────────────────────────────────────────────

  /**
   * MMR — Maximal Marginal Relevance.
   *
   * 🎯 Problema pe care o rezolvă: cele mai apropiate 5 chunk-uri pot fi
   *    aproape identice între ele. Dacă o carte repetă aceeași idee în trei
   *    locuri, primești de trei ori aceeași informație și pierzi trei sloturi
   *    din contextul plătit.
   *
   *    MMR alege iterativ: la fiecare pas ia documentul care maximizează
   *    relevanța față de întrebare MINUS similaritatea cu ce ai ales deja.
   *
   * @param lambda 1 = doar relevanță (ca similaritySearch), 0 = doar diversitate
   */
  async cautaCuDiversitate(
    vectorIntrebare: number[],
    k: number,
    optiuni: { lambda?: number; candidati?: number; filtru?: FiltruMetadate } = {},
  ): Promise<[DocumentInterface, number][]> {
    const lambda = optiuni.lambda ?? 0.5;
    const nrCandidati = optiuni.candidati ?? k * 4;

    // Pornim de la un bazin mai mare de candidați, apoi alegem k diverși.
    const candidati = this.intrari
      .filter((i) => (optiuni.filtru ? optiuni.filtru(i.document) : true))
      .map((i) => ({ ...i, scor: cosinus(vectorIntrebare, i.vector) }))
      .sort((a, b) => b.scor - a.scor)
      .slice(0, nrCandidati);

    const alese: typeof candidati = [];

    while (alese.length < k && candidati.length > 0) {
      let celMaiBunIndex = 0;
      let celMaiBunScor = -Infinity;

      candidati.forEach((candidat, index) => {
        const asemanareCuAlese =
          alese.length === 0
            ? 0
            : Math.max(...alese.map((a) => cosinus(candidat.vector, a.vector)));

        const scorMmr = lambda * candidat.scor - (1 - lambda) * asemanareCuAlese;
        if (scorMmr > celMaiBunScor) {
          celMaiBunScor = scorMmr;
          celMaiBunIndex = index;
        }
      });

      alese.push(candidati.splice(celMaiBunIndex, 1)[0]);
    }

    return alese.map((a) => [a.document, a.scor] as [DocumentInterface, number]);
  }

  // ─── Persistență ────────────────────────────────────────────────────
  //
  // 🎯 Nu e „persistență reală" — e un fișier JSON. Dar rezolvă problema
  //    practică a labului: reindexarea unei cărți durează minute, iar tu
  //    repornești serverul de zece ori pe oră.
  //
  // ⚠️ La scară, JSON-ul devine imposibil: 100.000 de vectori × 768 de
  //    dimensiuni înseamnă sute de MB citiți în RAM la fiecare pornire.
  //    Acolo intră pgvector.

  async salveaza(cale: string, eticheta = this.eticheta) {
    const { mkdir, writeFile } = await import("node:fs/promises");
    const { dirname } = await import("node:path");
    await mkdir(dirname(cale), { recursive: true });
    this.eticheta = eticheta;
    await writeFile(
      cale,
      JSON.stringify({
        eticheta,
        intrari: this.intrari.map((i) => ({
          vector: i.vector,
          pageContent: i.document.pageContent,
          metadata: i.document.metadata,
        })),
      }),
    );
  }

  async incarca(cale: string) {
    const { readFile } = await import("node:fs/promises");
    type IntrareSalvata = {
      vector: number[];
      pageContent: string;
      metadata: Record<string, unknown>;
    };
    const fisier = JSON.parse(await readFile(cale, "utf8")) as
      | { eticheta?: string; intrari: IntrareSalvata[] }
      | IntrareSalvata[]; // formatul vechi, fără etichetă

    const brut = Array.isArray(fisier) ? fisier : fisier.intrari;
    this.eticheta = Array.isArray(fisier) ? undefined : fisier.eticheta;

    this.intrari = brut.map((i) => ({
      vector: i.vector,
      document: new Document({ pageContent: i.pageContent, metadata: i.metadata }),
    }));

    return this.intrari.length;
  }

  get numarDocumente() {
    return this.intrari.length;
  }

  /** Valorile distincte ale unui câmp de metadate — pentru interfață. */
  valoriDistincte(camp: string): string[] {
    const valori = new Set<string>();
    for (const i of this.intrari) {
      const v = i.document.metadata?.[camp];
      if (v !== undefined && v !== null) valori.add(String(v));
    }
    return [...valori].sort();
  }

  // ─── Constructor de conveniență ─────────────────────────────────────

  static async dinDocumente(
    documente: DocumentInterface[],
    embeddings: EmbeddingsInterface,
  ) {
    const store = new VectorStoreSimplu(embeddings, {});
    await store.addDocuments(documente);
    return store;
  }
}
