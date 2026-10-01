// lib/rag/evaluare.ts
//
// ═══════════════════════════════════════════════════════════════════════
// EVALUAREA RETRIEVERULUI
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 Întrebarea la care răspunde fișierul ăsta: «RAG-ul meu e bun?»
//
//    Fără măsurători, răspunsul e o părere. Cu ele, e un număr pe care îl
//    poți compara înainte și după o schimbare de chunking, de model sau de k.
//
// ⚠️ DISTINCȚIA CARE SE CONFUNDĂ CEL MAI DES, și merită un slide:
//
//    RETRIEVAL     — a găsit fragmentul corect?     ← măsurăm aici
//    GENERARE      — a scris un răspuns bun din el? ← altă problemă
//
//    Sunt eșecuri diferite, cu remedii diferite. Dacă retrieval-ul n-a
//    adus fragmentul bun, niciun prompt din lume nu salvează răspunsul.
//    Iar dacă l-a adus și răspunsul e prost, chunking-ul nu e vinovat.
//
//    De aceea evaluăm ÎNTÂI retrieval-ul: e mai ieftin (fără apeluri de
//    model), mai rapid și mai ușor de interpretat.

import type { VectorStoreSimplu, FiltruMetadate } from "@/lib/rag/vector-store";
import { PRAG_RELEVANTA } from "@/lib/rag/schemas-biblioteca";

// ─────────────────────────────────────────────────────────────────────────
// Setul de evaluare
// ─────────────────────────────────────────────────────────────────────────

export interface CazEvaluare {
  /** Întrebarea, exact cum ar pune-o un utilizator. */
  intrebare: string;
  /**
   * Cum recunoaștem fragmentul corect.
   *
   * 🎯 De ce o FUNCȚIE și nu un id: id-urile se schimbă la fiecare
   *    reindexare. Un predicat pe metadate — «carte_biblica = Osea și
   *    cap_biblic = 1» — supraviețuiește reindexării.
   */
  eCorect: (metadata: Record<string, unknown>) => boolean;
  /** Ce testează cazul, pentru raport. */
  eticheta?: string;
  /**
   * Caz NEGATIV: corpusul NU conține răspunsul, deci ratarea e succesul.
   *
   * 🎯 De ce setul de evaluare TREBUIE să conțină așa ceva, și de ce e o
   *    greșeală de metodă să nu ai:
   *
   *    Un retriever care întoarce mereu ceva are hit rate bun și e periculos.
   *    Întrebi «ce spune cartea despre economia UE?», el întoarce cele mai
   *    apropiate 5 fragmente — irelevante, dar cu scoruri respectabile — iar
   *    modelul construiește un răspuns plauzibil din ele.
   *
   *    Cazurile negative măsoară exact asta: știe sistemul să spună «nu am»?
   *
   * ⚠️ Un retriever întoarce MEREU k fragmente — nu poate «să nu găsească».
   *    Singurul semnal că n-a găsit nimic util e SCORUL. Deci un caz negativ
   *    e respins corect doar dacă cel mai bun scor stă sub prag. Fără prag,
   *    metrica ar ieși 100% indiferent cât de prost e retrieverul.
   */
  asteptatNegasit?: boolean;
}

export interface RezultatCaz {
  intrebare: string;
  eticheta?: string;
  /** Pe ce poziție a apărut primul fragment corect. 0 = negăsit. */
  pozitie: number;
  /** Scorul primului rezultat, ca să vedem dacă retrieverul era «sigur». */
  scorTop: number;
  gasit: boolean;
  /** Cazul a ieșit cum trebuia? La cele negative, «corect» înseamnă negăsit. */
  corect: boolean;
  negativ: boolean;
}

export interface RaportEvaluare {
  cazuri: RezultatCaz[];
  /** Câte întrebări au avut fragmentul corect ORIUNDE în primele k. */
  hitRate: number;
  /** Câte l-au avut pe prima poziție. */
  hitRateTop1: number;
  /**
   * Mean Reciprocal Rank: media lui 1/poziție.
   *
   * 🔍 De ce e mai bun decât hit rate-ul singur: hit rate-ul nu face
   *    diferența între «corect pe poziția 1» și «corect pe poziția 5».
   *    MRR o face — 1.0 înseamnă mereu primul, 0.2 înseamnă mereu al cincilea.
   *    Contează, pentru că fragmentele de jos se pierd în context.
   */
  mrr: number;
  k: number;
  total: number;
  /**
   * Câte cazuri negative au fost respinse corect.
   *
   * ⚠️ Se raportează SEPARAT de hit rate. Un sistem cu hit rate 100% și
   *    respingere 0% e un sistem care nu spune niciodată «nu știu» — cel
   *    mai periculos tip de RAG.
   */
  respingereCorecta: number | null;
  /** Scorul mediu al primului rezultat la cazurile negative. */
  scorMediuNegativ: number | null;
  /** Pragul folosit ca să decidem că un caz negativ a fost respins. */
  prag: number;
}

// ─────────────────────────────────────────────────────────────────────────
// Evaluarea
// ─────────────────────────────────────────────────────────────────────────

export async function evalueazaRetrieval(
  store: VectorStoreSimplu,
  cazuri: CazEvaluare[],
  optiuni: {
    k?: number;
    filtru?: FiltruMetadate;
    cuDiversitate?: boolean;
    /** Sub acest scor, un rezultat e considerat irelevant. */
    prag?: number;
  } = {},
): Promise<RaportEvaluare> {
  const k = optiuni.k ?? 5;
  const prag = optiuni.prag ?? PRAG_RELEVANTA;
  const rezultate: RezultatCaz[] = [];

  for (const caz of cazuri) {
    const vector = await store.embeddings.embedQuery(caz.intrebare);
    const gasite = optiuni.cuDiversitate
      ? await store.cautaCuDiversitate(vector, k, { filtru: optiuni.filtru })
      : await store.similaritySearchVectorWithScore(vector, k, optiuni.filtru);

    // Poziția primului fragment care trece testul. 1-indexat; 0 = negăsit.
    let pozitie = 0;
    for (let i = 0; i < gasite.length; i++) {
      if (caz.eCorect(gasite[i][0].metadata as Record<string, unknown>)) {
        pozitie = i + 1;
        break;
      }
    }

    const negativ = caz.asteptatNegasit === true;
    const scorTop = gasite[0]?.[1] ?? 0;
    rezultate.push({
      intrebare: caz.intrebare,
      eticheta: caz.eticheta,
      pozitie,
      scorTop,
      gasit: pozitie > 0,
      negativ,
      // La cazurile negative, succesul e ca NIMIC să nu treacă de prag.
      corect: negativ ? scorTop < prag : pozitie > 0,
    });
  }

  // ⚠️ Metricile de retrieval se calculează DOAR pe cazurile pozitive.
  //    Amestecarea lor cu cele negative dă un număr care nu înseamnă nimic.
  const pozitive = rezultate.filter((r) => !r.negativ);
  const negative = rezultate.filter((r) => r.negativ);
  const totalPozitive = pozitive.length || 1;

  return {
    cazuri: rezultate,
    hitRate: pozitive.filter((r) => r.gasit).length / totalPozitive,
    hitRateTop1: pozitive.filter((r) => r.pozitie === 1).length / totalPozitive,
    mrr:
      pozitive.reduce((s, r) => s + (r.pozitie > 0 ? 1 / r.pozitie : 0), 0) /
      totalPozitive,
    k,
    total: pozitive.length,
    respingereCorecta:
      negative.length > 0
        ? negative.filter((r) => r.corect).length / negative.length
        : null,
    scorMediuNegativ:
      negative.length > 0
        ? negative.reduce((s, r) => s + r.scorTop, 0) / negative.length
        : null,
    prag,
  };
}

/** Raport lizibil în consolă. */
export function afiseazaRaport(raport: RaportEvaluare, titlu = "Evaluare") {
  const proc = (x: number) => `${Math.round(x * 100)}%`;

  console.log(`\n── ${titlu} · k=${raport.k} · ${raport.total} întrebări pozitive ──`);
  console.log(
    `   hit rate: ${proc(raport.hitRate)}   top-1: ${proc(raport.hitRateTop1)}   MRR: ${raport.mrr.toFixed(3)}`,
  );

  if (raport.respingereCorecta !== null) {
    console.log(
      `   respingere corectă (prag ${raport.prag}): ${proc(raport.respingereCorecta)}` +
        `   (scor mediu pe cazurile fără răspuns: ${raport.scorMediuNegativ?.toFixed(2)})`,
    );
  }

  const gresite = raport.cazuri.filter((c) => !c.corect);
  if (gresite.length > 0) {
    console.log(`   GREȘITE (${gresite.length}):`);
    for (const r of gresite) {
      const motiv = r.negativ ? "scor peste prag, deși nu există răspuns" : "nu a găsit";
      console.log(
        `     · «${r.intrebare.slice(0, 52)}» — ${motiv} (scor top: ${r.scorTop.toFixed(2)})`,
      );
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Compararea configurațiilor
// ─────────────────────────────────────────────────────────────────────────
//
// 🎯 Asta e ce transformă evaluarea din exercițiu în unealtă. Rulezi
//    ACELAȘI set de întrebări pe configurații diferite și vezi care câștigă:
//
//      - k = 3 vs 5 vs 10
//      - cu MMR vs fără
//      - chunking structural vs naiv
//      - embeddings Google vs OpenAI
//
// ⚠️ Numai una dintre ele se schimbă odată. Dacă schimbi și chunking-ul, și
//    modelul, nu mai știi care a produs diferența.

export interface Configuratie {
  nume: string;
  store: VectorStoreSimplu;
  k?: number;
  cuDiversitate?: boolean;
}

export async function comparaConfiguratii(
  configuratii: Configuratie[],
  cazuri: CazEvaluare[],
) {
  const rezultate: { nume: string; raport: RaportEvaluare }[] = [];

  for (const cfg of configuratii) {
    const raport = await evalueazaRetrieval(cfg.store, cazuri, {
      k: cfg.k,
      cuDiversitate: cfg.cuDiversitate,
    });
    rezultate.push({ nume: cfg.nume, raport });
  }

  console.log("\n── COMPARAȚIE ──");
  console.log(
    "   " +
      "configurație".padEnd(28) +
      "hit rate".padStart(10) +
      "top-1".padStart(9) +
      "MRR".padStart(8),
  );
  for (const { nume, raport } of rezultate) {
    console.log(
      "   " +
        nume.padEnd(28) +
        `${Math.round(raport.hitRate * 100)}%`.padStart(10) +
        `${Math.round(raport.hitRateTop1 * 100)}%`.padStart(9) +
        raport.mrr.toFixed(3).padStart(8),
    );
  }

  return rezultate;
}
