// lib/rag/rag-clasic.ts
//
// ═══════════════════════════════════════════════════════════════════════
// RAG CLASIC — PIPELINE-UL DE INTEROGARE, CA LANȚ LCEL
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 Ăsta e RAG-ul din toate tutorialele, și e important să-l predăm ÎNTÂI,
//    complet și fără dispreț. Funcționează. Pentru multe aplicații reale e
//    exact ce trebuie.
//
//    Patru pași, în ordine fixă, scriși de noi:
//
//      1. întrebarea → vector
//      2. vector → cele mai apropiate k chunkuri
//      3. chunkurile → text pentru prompt
//      4. prompt → model → răspuns
//
// 🌉 LEGĂTURA CU LAB 7, și e cea care leagă tot cursul:
//
//    Asta e un WORKFLOW. Noi am decis: o singură căutare, k fix, fără
//    reformulare, fără a doua încercare. Modelul nu decide nimic despre
//    retrieval — primește contextul de-a gata.
//
//    În Bloc C facem varianta agentică, unde retrieval-ul devine un tool
//    și modelul decide dacă să caute, ce să caute și de câte ori. E fix
//    distincția workflow vs. agent din Lab 7, aplicată la RAG.

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda, RunnableSequence } from "@langchain/core/runnables";
import type { DocumentInterface } from "@langchain/core/documents";
import { createModel, readProviderFromEnv } from "@/lib/llm";
import type { VectorStoreSimplu, FiltruMetadate } from "@/lib/rag/vector-store";
import { PRAG_RELEVANTA } from "@/lib/rag/schemas-biblioteca";

// ─────────────────────────────────────────────────────────────────────────
// Pasul 3 · asamblarea contextului
// ─────────────────────────────────────────────────────────────────────────

/**
 * Transformă chunkurile în textul care intră în prompt.
 *
 * 🎯 Trei decizii, toate cu motiv:
 *
 *    1. FIECARE chunk primește un identificator scurt, [1], [2], [3].
 *       Fără el, modelul nu are cum să citeze precis, iar tu nu ai cum să
 *       verifici. Citarea e singura apărare împotriva halucinației.
 *
 *    2. Metadatele intră în context, nu doar textul. Modelul trebuie să
 *       poată scrie «Osea 1:2, p. 251», nu «undeva în carte».
 *
 *    3. Ordinea: cele mai relevante la ÎNCEPUT și la SFÂRȘIT.
 *       ⚠️ Fenomenul «lost in the middle»: modelele acordă mai puțină
 *          atenție informației din mijlocul unui context lung. Cu 5 chunkuri
 *          nu se simte; cu 20 se simte foarte tare.
 *
 * ⚠️ Numărul unui fragment îl fixăm ÎNAINTE de reordonare (`numere`). Dacă
 *    am numerota după, [2] din răspuns ar fi alt fragment decât [2] de pe
 *    ecran — și verificarea citărilor, singura apărare, ar minți.
 *
 * @param numere  numărul fiecărui document; implicit poziția lui (1, 2, 3…)
 */
export function asambleazaContext(
  documente: DocumentInterface[],
  numere?: number[],
): string {
  return documente
    .map((doc, i) => {
      const m = doc.metadata as Record<string, unknown>;
      const sursa = [
        m.carte_sursa,
        m.ancora,
        m.pagina ? `p. ${m.pagina}` : undefined,
      ]
        .filter(Boolean)
        .join(" · ");

      const sectiune = [m.capitol, m.subsectiune].filter(Boolean).join(" → ");

      return [
        `[${numere?.[i] ?? i + 1}] ${sursa}`,
        sectiune ? `    ${sectiune}` : "",
        doc.pageContent,
      ]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n\n---\n\n");
}

/** Reordonare pentru «lost in the middle»: cele bune la capete. */
export function reordoneazaPentruAtentie<T>(documente: T[]): T[] {
  const inceput: T[] = [];
  const sfarsit: T[] = [];
  documente.forEach((d, i) => (i % 2 === 0 ? inceput.push(d) : sfarsit.unshift(d)));
  return [...inceput, ...sfarsit];
}

// ─────────────────────────────────────────────────────────────────────────
// Pasul 4 · promptul
// ─────────────────────────────────────────────────────────────────────────
//
// ⚠️ Regulile de aici nu sunt stil. Fiecare previne un mod de eșec concret,
//    pe care îl vom provoca în demo:
//
//   - «doar din context»        → previne răspunsul din memoria modelului
//   - «citează [n]»             → face răspunsul verificabil
//   - «spune când nu știi»      → previne inventarea
//   - «nu amesteca sursele»     → previne pasta din trei comentatori

const PROMPT = ChatPromptTemplate.fromMessages([
  [
    "system",
    `Ești un asistent care răspunde STRICT pe baza fragmentelor date din bibliotecă.

REGULI:
- Răspunde DOAR din fragmentele de mai jos. Nu completa din cunoștințe generale.
- Citează fiecare afirmație cu numărul fragmentului: [1], [2].
- Dacă fragmentele nu conțin răspunsul, spune exact asta și oprește-te.
  Nu inventa, nu aproxima, nu «probabil».
- Dacă două fragmente spun lucruri diferite, prezintă-le SEPARAT, cu sursa
  fiecăruia. NU le combina într-o poziție unică.
- Răspunde în limba în care e pusă întrebarea, chiar dacă fragmentele sunt
  în altă limbă. Citatele scurte le poți lăsa în original.

FRAGMENTE:
{context}`,
  ],
  ["human", "{intrebare}"],
]);

// ─────────────────────────────────────────────────────────────────────────
// Lanțul complet
// ─────────────────────────────────────────────────────────────────────────

export interface OptiuniRagClasic {
  k?: number;
  filtru?: FiltruMetadate;
  /** Diversitate în loc de simplă similaritate. */
  cuDiversitate?: boolean;
  /** Oprește apelul de model dacă utilizatorul închide pagina. */
  signal?: AbortSignal;
}

export interface RezultatRag {
  raspuns: string;
  /** În ordinea scorului; fragmentul de pe poziția i are numărul [i + 1]. */
  fragmente: { text: string; metadata: Record<string, unknown>; scor: number }[];
  /** Doar retrieval-ul: embedding + căutare. */
  durataCautareMs: number;
  durataMs: number;
  tokeniInput: number;
  tokeniOutput: number;
}

/**
 * RAG clasic: caută o dată, răspunde o dată.
 *
 * 🔍 Îl construim ca `RunnableSequence` — LCEL, din Lab 5 — ca să se vadă
 *    că e un LANȚ, nu un agent. Pașii sunt fixați de noi, în ordine.
 */
export async function ragClasic(
  store: VectorStoreSimplu,
  intrebare: string,
  optiuni: OptiuniRagClasic = {},
): Promise<RezultatRag> {
  const pornit = Date.now();
  const k = optiuni.k ?? 5;

  // ── Pasul 1-2: întrebare → vector → chunkuri ──────────────────────
  const vector = await store.embeddings.embedQuery(intrebare);
  const rezultate = optiuni.cuDiversitate
    ? await store.cautaCuDiversitate(vector, k, { filtru: optiuni.filtru })
    : await store.similaritySearchVectorWithScore(vector, k, optiuni.filtru);
  const gasite = rezultate.filter(([, scor]) => scor >= PRAG_RELEVANTA);

  const durataCautareMs = Date.now() - pornit;

  if (gasite.length === 0) {
    return {
      raspuns:
        "Nu am găsit niciun fragment relevant în bibliotecă. Încearcă altă formulare " +
        "sau verifică filtrele.",
      fragmente: [],
      durataCautareMs,
      durataMs: Date.now() - pornit,
      tokeniInput: 0,
      tokeniOutput: 0,
    };
  }

  // ── Pasul 3-4: context → model → răspuns ──────────────────────────
  //
  // Numerotăm în ordinea scorului (ca pe ecran), apoi reordonăm.
  const ordonate = reordoneazaPentruAtentie(
    gasite.map(([doc], i) => ({ doc, nr: i + 1 })),
  );

  // 🔍 Lanțul se oprește la model, fără `StringOutputParser`: parserul ar
  //    păstra doar textul și ar arunca `usage_metadata` — adică tokenii.
  //    Fără ei nu putem compara costul clasic vs. agentic, iar comparația
  //    asta e rostul paginii.
  const lant = RunnableSequence.from([
    RunnableLambda.from((intrare: { intrebare: string }) => ({
      intrebare: intrare.intrebare,
      context: asambleazaContext(
        ordonate.map((o) => o.doc),
        ordonate.map((o) => o.nr),
      ),
    })),
    PROMPT,
    createModel(readProviderFromEnv()),
  ]);

  const mesaj = await lant.invoke({ intrebare }, { signal: optiuni.signal });

  return {
    raspuns: mesaj.text,
    fragmente: gasite.map(([doc, scor]) => ({
      text: doc.pageContent,
      metadata: doc.metadata as Record<string, unknown>,
      scor,
    })),
    durataCautareMs,
    durataMs: Date.now() - pornit,
    tokeniInput: mesaj.usage_metadata?.input_tokens ?? 0,
    tokeniOutput: mesaj.usage_metadata?.output_tokens ?? 0,
  };
}
