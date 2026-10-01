// lib/biblioteca-agent.ts
//
// ═══════════════════════════════════════════════════════════════════════
// RAG AGENTIC
// ═══════════════════════════════════════════════════════════════════════
//
// 🌉 Compară cu `lib/rag/rag-clasic.ts`, care e un RunnableSequence:
//
//    CLASIC                              AGENTIC
//    ──────────────────────────────────  ────────────────────────────────
//    noi chemăm retrieval-ul             modelul îl cheamă, ca tool
//    o singură căutare                   câte are nevoie
//    k fix, ales de noi                  modelul cere câte vrea
//    întrebarea merge ca atare           modelul o poate reformula
//    filtrele le punem noi               modelul le deduce din întrebare
//    un apel de model                    2-6 apeluri
//    ~2 secunde                          ~8-20 secunde
//
// 🎯 E aceeași distincție ca în Lab 7 — workflow vs. agent — aplicată la
//    retrieval. Nu e un concept nou de învățat, e unul deja știut, mutat.
//
// ⚠️ Nu e „mai bun". E mai flexibil și mai scump. Slide-ul «când NU vrei
//    agentic» spune exact: dacă întrebările tale sunt uniforme și o singură
//    căutare le acoperă, RAG-ul clasic e răspunsul corect.

import type { StructuredToolInterface } from "@langchain/core/tools";
import { createAgent, modelCallLimitMiddleware, toolCallLimitMiddleware } from "langchain";
import { createModel, readProviderFromEnv } from "@/lib/llm";
import { creeazaBibliotecaTools } from "@/lib/tools/biblioteca-tools";
import type { VectorStoreSimplu } from "@/lib/rag/vector-store";
import { PRAG_RELEVANTA } from "@/lib/rag/schemas-biblioteca";

/**
 * ⚠️ Vezi Lab 8: un pas de graf nu e un apel de model. Cu ~6 căutări în cel
 *    mai rău caz și 2 middleware, 60 e confortabil.
 */
export const LIMITA_RECURSIE = 60;

const PROMPT = `Ești un asistent de studiu peste o bibliotecă de documente și cărți.

CUM LUCREZI:
1. Pentru ORICE întrebare de conținut, cheamă întâi cautaInBiblioteca.
   Nu răspunde din cunoștințele tale generale — biblioteca e sursa.
   Nu presupune domeniul bibliotecii din acest prompt și nu declara o întrebare
   în afara corpusului înainte să cauți. Doar rezultatele căutării pot arăta
   dacă biblioteca conține sau nu răspunsul.
2. Uită-te la scorurile de relevanță. Dacă toate sunt sub ${PRAG_RELEVANTA}, formularea ta
   a fost probabil greșită: reformulează și caută din nou, ALTFEL.
3. Dacă întrebarea are două părți («compară X cu Y»), fă două căutări.
4. Dacă utilizatorul cere părerea unei anumite cărți, folosește filtrul
   carte_sursa. Înainte de a filtra, verifică prin inventarBiblioteca numele
   exacte care există. Folosește carte_biblica doar pentru întrebări biblice.

CUM RĂSPUNZI:
- Citează fiecare afirmație: [1], [2], cu numerele fragmentelor primite.
  Numerele sunt unice între căutări (a doua căutare continuă de la [6]).
- La final, listează sursele: carte, referință, pagină.
- Dacă fragmentele NU conțin răspunsul, spune asta clar. Nu completa din
  memoria ta, nici măcar «probabil». Un «nu am găsit» corect valorează mai
  mult decât un răspuns plauzibil și neverificabil.
- Dacă două surse spun lucruri diferite, prezintă-le SEPARAT, cu atribuire.
  NU le combina într-o poziție unică — dezacordul dintre autori e informație,
  nu zgomot de eliminat.
- Răspunde în limba întrebării, chiar dacă fragmentele sunt în altă limbă.
  Citatele scurte le poți lăsa în original, cu traducere între paranteze.

TON: precis, sobru și fără ironie. Ești un instrument de studiu.`;

export function creeazaBibliotecaAgent(store: VectorStoreSimplu) {
  // ⚠️ Adnotarea de tip, ca în Lab 8: cu tool-uri cu scheme diferite,
  //    inferența lui createAgent cedează și cere responseFormat.
  const unelte: StructuredToolInterface[] = creeazaBibliotecaTools(store);

  const middleware = [
    // Nu are rost să caute de zece ori aceeași întrebare.
    toolCallLimitMiddleware({
      toolName: "cautaInBiblioteca",
      runLimit: 6,
      exitBehavior: "continue" as const,
    }),
    modelCallLimitMiddleware({ runLimit: 12, exitBehavior: "end" as const }),
  ];

  return createAgent({
    model: createModel(readProviderFromEnv()),
    tools: unelte,
    systemPrompt: PROMPT,
    middleware,
  });
}
