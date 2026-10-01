// lib/tools/biblioteca-tools.ts
//
// ═══════════════════════════════════════════════════════════════════════
// RETRIEVAL EXPUS CA TOOL — pasul de la RAG clasic la RAG agentic
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 Schimbarea e mică în cod și uriașă în comportament.
//
//    În RAG clasic, noi chemam retrieval-ul: o dată, cu k fix, înainte de
//    a chema modelul. Modelul primea contextul de-a gata și nu avea nimic
//    de decis.
//
//    Aici retrieval-ul devine un tool. Modelul decide:
//      - DACĂ să caute (poate întrebarea nu cere căutare)
//      - CE să caute (poate reformula, poate împărți în două căutări)
//      - DE CÂTE ORI (dacă prima căutare n-a dat nimic bun, caută altfel)
//      - CU CE FILTRE (dacă întrebarea zice «în Osea», filtrează)
//
// 🌉 E aceeași trecere ca în Lab 7: de la buclă scrisă de tine la buclă
//    ținută de model. Doar că obiectul buclei e retrieval-ul, nu tool-urile
//    în general.
//
// ⚠️ Ce plătești: mai multe apeluri de model, latență mai mare, cost mai
//    mare. Slide-ul «când NU vrei agentic» există exact pentru asta.

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import type { DocumentInterface } from "@langchain/core/documents";
import type { VectorStoreSimplu } from "@/lib/rag/vector-store";
import { PRAG_RELEVANTA } from "@/lib/rag/schemas-biblioteca";

/**
 * Comparație fără majuscule și fără diacritice: «osea» = «Osea»,
 * «romani» = «Romani», «Iesirea» = «Ieșirea». Un filtru care ratează din
 * cauza unei diacritice ascunde exact fragmentul bun, fără nicio eroare.
 */
export function normalizeaza(text: unknown): string {
  return String(text ?? "")
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .trim();
}

// ─────────────────────────────────────────────────────────────────────────
// Schema de intrare
// ─────────────────────────────────────────────────────────────────────────
//
// ⚠️ Chei EXPLICITE, nu `z.record()`. Aceeași capcană ca în Lab 7: schema
//    pleacă spre provider ca JSON Schema, iar unele modele resping cheile
//    dinamice.

export const IntrareCautare = z.object({
  intrebare: z
    .string()
    .min(3)
    .describe(
      "Ce cauți, formulat ca o frază descriptivă. NU cuvinte cheie izolate — " +
        "căutarea e semantică, deci «motivele instaurării regimului fanariot» " +
        "funcționează mai bine decât «motive fanarioți».",
    ),
  carte_biblica: z
    .string()
    .optional()
    .describe(
      "Filtrează după cartea biblică discutată. Folosește numele EXACT cum apare " +
        "în inventarBiblioteca (carti_biblice) — nu-l traduce și nu-l ghici. " +
        "Folosește DOAR dacă întrebarea se referă explicit la o carte anume.",
    ),
  capitol: z
    .number()
    .int()
    .optional()
    .describe(
      "Filtrează după capitolul biblic. Merge DOAR împreună cu carte_biblica; " +
        "fără ea e ignorat (capitolul 2 există în zeci de cărți).",
    ),
  carte_sursa: z
    .string()
    .optional()
    .describe(
      "Filtrează după cartea din bibliotecă (autorul/lucrarea). " +
        "Folosește când utilizatorul cere părerea unei anumite surse.",
    ),
  numar_rezultate: z
    .number()
    .int()
    .min(1)
    .max(10)
    .optional()
    .describe("Câte fragmente să întorc. Implicit 5. Peste 8 rareori ajută."),
});

// ─────────────────────────────────────────────────────────────────────────
// Formatarea rezultatului
// ─────────────────────────────────────────────────────────────────────────

/**
 * Ce vede modelul după căutare.
 *
 * 🎯 Formatul contează la fel de mult ca al descrierii tool-ului. Modelul
 *    trebuie să poată CITA din ce primește, deci fiecare fragment vine cu
 *    identificatorul și cu sursa lui completă.
 *
 * 🔍 Includem și scorul. De ce: dacă toate scorurile sunt mici, modelul
 *    poate să-și dea seama singur că retrieval-ul n-a găsit nimic bun și
 *    să reformuleze — în loc să răspundă din fragmente irelevante.
 *    E o informație pe care RAG-ul clasic o aruncă.
 */
function formateaza(
  rezultate: [DocumentInterface, number][],
  primulId: number,
  avertisment?: string,
) {
  if (rezultate.length === 0) {
    return {
      gasite: 0,
      mesaj:
        "Niciun fragment găsit. Încearcă o formulare mai generală sau scoate filtrele.",
      ...(avertisment ? { avertisment } : {}),
      fragmente: [],
    };
  }

  return {
    gasite: rezultate.length,
    ...(avertisment ? { avertisment } : {}),
    fragmente: rezultate.map(([doc, scor], i) => {
      const m = doc.metadata as Record<string, unknown>;
      return {
        id: primulId + i,
        relevanta: Math.round(scor * 100) / 100,
        carte_sursa: m.carte_sursa ?? "necunoscută",
        capitol: m.capitol ?? null,
        subsectiune: m.subsectiune ?? null,
        referinta: m.ancora ?? null,
        pagina: m.pagina ?? null,
        text: doc.pageContent,
      };
    }),
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Tool factory
// ─────────────────────────────────────────────────────────────────────────
//
// 🔍 Fabrică, nu constantă — ca în Lab 7. Tool-ul are nevoie de store-ul
//    încărcat, iar acela nu există la momentul importului modulului.

export function creeazaBibliotecaTools(store: VectorStoreSimplu) {
  // ⚠️ Numerotarea fragmentelor CONTINUĂ de la o căutare la alta: prima
  //    întoarce [1]-[5], a doua [6]-[10]. Dacă fiecare căutare ar începe de
  //    la [1], la o întrebare comparativă (două căutări) «[1]» din răspuns
  //    ar putea fi oricare din două fragmente. Contorul trăiește în closure,
  //    deci e pe o instanță de tool-uri — adică pe o cerere.
  let urmatorulId = 1;

  const cauta = tool(
    async ({ intrebare, carte_biblica, capitol, carte_sursa, numar_rezultate }) => {
      const k = numar_rezultate ?? 5;

      // ⚠️ Capitolul fără carte ar amesteca Osea 2 cu Romani 2 și cu orice
      //    alt capitol 2 — iar modelul le-ar putea trata ca răspuns la o
      //    singură întrebare. Descrierea spune «necesită carte_biblica», dar
      //    descrierea e o rugăminte, nu o garanție: o impunem aici.
      //    Nu aruncăm eroare — căutăm fără capitol și îi spunem modelului de
      //    ce, ca să poată reveni cu carte_biblica dacă vrea.
      const capitolFolosit = carte_biblica ? capitol : undefined;
      const avertisment =
        capitol !== undefined && !carte_biblica
          ? "Filtrul «capitol» a fost ignorat: funcționează doar împreună cu carte_biblica."
          : undefined;

      // Filtrul se aplică ÎNAINTE de sortare — pre-filtering.
      // Vezi comentariul din vector-store.ts pentru de ce contează.
      const filtru =
        carte_biblica || capitolFolosit !== undefined || carte_sursa
          ? (doc: DocumentInterface) => {
              const m = doc.metadata as Record<string, unknown>;
              if (carte_biblica && normalizeaza(m.carte_biblica) !== normalizeaza(carte_biblica)) return false;
              if (capitolFolosit !== undefined && m.cap_biblic !== capitolFolosit) return false;
              if (carte_sursa && !normalizeaza(m.carte_sursa).includes(normalizeaza(carte_sursa))) return false;
              return true;
            }
          : undefined;

      const vector = await store.embeddings.embedQuery(intrebare);
      const rezultate = await store.similaritySearchVectorWithScore(vector, k, filtru);

      console.log(
        `[cautaInBiblioteca] «${intrebare.slice(0, 50)}»` +
          (carte_biblica ? ` · ${carte_biblica}${capitolFolosit ? " " + capitolFolosit : ""}` : "") +
          (avertisment ? " · capitol ignorat (fără carte)" : "") +
          ` → ${rezultate.length} fragmente`,
      );

      const primulId = urmatorulId;
      urmatorulId += rezultate.length;
      return formateaza(rezultate, primulId, avertisment);
    },
    {
      name: "cautaInBiblioteca",
      description:
        "Caută fragmente relevante în biblioteca de documente și cărți. " +
        "Biblioteca poate conține orice domeniu; nu presupune că o întrebare este " +
        "în afara corpusului înainte să folosești acest tool.\n\n" +
        "CÂND se cheamă:\n" +
        "- pentru ORICE întrebare de conținut. Nu răspunde din memoria ta.\n" +
        "- de mai multe ori, dacă întrebarea are mai multe părți sau dacă " +
        `prima căutare a întors relevanțe mici (sub ${PRAG_RELEVANTA}).\n` +
        "- cu formulări DIFERITE, dacă prima n-a găsit nimic util. Nu repeta " +
        "aceeași căutare.\n\n" +
        "REGULI:\n" +
        "- «intrebare» e o frază descriptivă, nu cuvinte cheie. Căutarea e semantică.\n" +
        "- Folosește filtrele DOAR dacă utilizatorul le cere explicit. Un filtru " +
        "greșit ascunde exact fragmentul bun.\n" +
        "- Dacă vrei să compari două surse, fă DOUĂ căutări cu carte_sursa diferit.\n" +
        "- Numerele fragmentelor sunt unice în toată conversația; citează-le exact.\n\n" +
        "EXEMPLU:\n" +
        "- { intrebare: «măsurile administrative și fiscale ale domnilor fanarioți» }\n" +
        "- Ieșire: { gasite, fragmente: [{ id, relevanta, carte_sursa, referinta, pagina, text }] }",
      schema: IntrareCautare,
    },
  );

  // ── Tool secundar: ce e în bibliotecă ───────────────────────────────
  //
  // 🎯 De ce ne trebuie: fără el, modelul nu are cum să știe ce cărți există
  //    și inventează nume de surse când utilizatorul întreabă «ce ai în
  //    bibliotecă?». Un tool ieftin care previne o halucinație frecventă.
  const inventar = tool(
    async () => ({
      carti: store.valoriDistincte("carte_sursa"),
      carti_biblice: store.valoriDistincte("carte_biblica"),
      total_fragmente: store.numarDocumente,
    }),
    {
      name: "inventarBiblioteca",
      description:
        "Listează cărțile din bibliotecă și, dacă există asemenea metadate, " +
        "cărțile biblice acoperite.\n\n" +
        "CÂND: la întrebări de tipul «ce ai în bibliotecă?», «ce autori acoperi?», " +
        "sau înainte de a filtra pe o carte, ca să verifici că există.\n" +
        "NU inventa niciodată titluri de cărți. Cheamă tool-ul ăsta.",
      schema: z.object({}),
    },
  );

  return [cauta, inventar];
}
