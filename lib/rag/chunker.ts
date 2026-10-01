// lib/rag/chunker.ts
//
// ═══════════════════════════════════════════════════════════════════════
// PASUL 2-3 DIN PIPELINE — CHUNKING ȘI METADATE
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 De ce se taie textul în bucăți, întrebarea pe care o pune toată lumea:
//
//   1. Modelul de embeddings are o limită de intrare (gemini-embedding-001:
//      2048 de tokeni). O carte întreagă nu încape.
//   2. Chiar dacă ar încăpea, un vector pentru 300 de pagini n-ar însemna
//      nimic — ar fi media tuturor ideilor din carte. Cauți «Osea 2» și
//      primești «cartea asta e despre profeți». Inutil.
//   3. Contextul pe care îl dai modelului la generare e limitat și costă.
//      Vrei 5 bucăți relevante, nu 300 de pagini.
//
// Fișierul conține DOUĂ chunkere, intenționat:
//
//   • `chunkNaiv`      — ce face toată lumea: taie la N caractere
//   • `chunkStructural` — taie la unitățile autorului, cu metadate
//
// Le păstrăm pe amândouă ca să le putem compara la rulare, pe același text.

import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { REFERINTA, uneste, type RandExtras } from "@/lib/rag/extrage-pdf";

// ─────────────────────────────────────────────────────────────────────────
// Metadatele unui chunk
// ─────────────────────────────────────────────────────────────────────────

export interface MetadateChunk {
  /** Cartea din care provine, așa cum apare în citări. */
  carte_sursa: string;
  /** «PROFEȚIA ÎN PROFEȚII MICI» */
  parte?: string;
  /** «PROFEȚIA LUI OSEA» */
  capitol?: string;
  /** «The First Cycle of the Sins of Israel and Her Restoration» */
  subsectiune?: string;
  /** Ancora brută: «Osea 1:2–2:1» */
  ancora?: string;
  /** Cartea biblică extrasă din ancoră, pentru filtrare. */
  carte_biblica?: string;
  /** Capitolul biblic, pentru filtrare. */
  cap_biblic?: number;
  /** Pagina din PDF — apare în citare. */
  pagina: number;
  /** Câte caractere are chunk-ul; util în statistici. */
  lungime: number;
}

// ─────────────────────────────────────────────────────────────────────────
// VARIANTA 1 — chunking naiv, la N caractere
// ─────────────────────────────────────────────────────────────────────────
//
// 🎯 Asta e ce găsești în 9 din 10 tutoriale de RAG, și e important să o
//    predăm ÎNTÂI. Funcționează. Pentru multe corpusuri e chiar suficientă.
//
// ⚠️ Ce pierzi:
//    - taie în mijlocul frazei, uneori în mijlocul cuvântului
//    - un chunk poate conține sfârșitul unei idei și începutul alteia
//    - metadate: doar numele fișierului. Nu poți filtra pe nimic.
//
// `chunkSize` și `chunkOverlap` sunt cele două butoane pe care le are:
//   - prea mic  → context insuficient, modelul nu înțelege bucata
//   - prea mare → dilluezi relevanța, plătești tokeni degeaba
//   - overlap-ul salvează frazele tăiate la graniță, cu prețul duplicării

export async function chunkNaiv(
  textIntreg: string,
  carteSursa: string,
  optiuni: { chunkSize?: number; chunkOverlap?: number } = {},
): Promise<Document[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: optiuni.chunkSize ?? 1000,
    chunkOverlap: optiuni.chunkOverlap ?? 200,
    // Ordinea contează: încearcă întâi să taie la paragraf, apoi la frază,
    // apoi la cuvânt, și abia la final oriunde. De asta se numește «recursive».
    separators: ["\n\n", "\n", ". ", " ", ""],
  });

  const bucati = await splitter.splitText(textIntreg);

  return bucati.map(
    (text, i) =>
      new Document({
        pageContent: text,
        metadata: {
          carte_sursa: carteSursa,
          pagina: 0, // nu știm — informația s-a pierdut la extragere
          lungime: text.length,
          index: i,
        },
      }),
  );
}

// ─────────────────────────────────────────────────────────────────────────
// VARIANTA 2 — chunking structural
// ─────────────────────────────────────────────────────────────────────────

/** Din «Osea 1:2–2:1» scoatem cartea și capitolul, pentru filtrare. */
function parseAncora(ancora: string) {
  const m = REFERINTA.exec(ancora.trim());
  if (!m) return {};
  return {
    carte_biblica: m[1].replace(/\s+/g, " ").trim(),
    cap_biblic: Number(m[2]),
  };
}

/**
 * Taie la unitățile semantice ale autorului.
 *
 * 🎯 Ideea centrală: autorul a decis deja unde începe o idee nouă. A marcat-o
 *    tipografic — cu o ancoră în italic, «Osea 1:2–2:1.». Noi nu inventăm
 *    granițe, îl urmăm pe el.
 *
 * 🔍 Ierarhia se MOȘTENEȘTE. Un rând de corp aparține ultimei ancore văzute,
 *    care aparține ultimei subsecțiuni, care aparține ultimului capitol.
 *    De aceea parcurgem rândurile în ordine și ținem un «context curent».
 */
export async function chunkStructural(
  randuri: RandExtras[],
  carteSursa: string,
  optiuni: { maxCaractere?: number } = {},
): Promise<Document[]> {
  const maxCaractere = optiuni.maxCaractere ?? 2500;

  // Contextul curent, actualizat pe măsură ce coborâm prin document.
  let parte: string | undefined;
  let capitol: string | undefined;
  let subsectiune: string | undefined;
  let ancora: string | undefined;
  let paginaStart = 1;
  let acumulate: string[] = [];

  const documente: Document[] = [];

  /** Închide unitatea curentă și o transformă în Document. */
  const inchideUnitatea = async () => {
    if (acumulate.length === 0) return;
    const text = uneste(acumulate);
    acumulate = [];
    if (text.length < 40) return; // fragmente prea mici, nu merită indexate

    const meta: MetadateChunk = {
      carte_sursa: carteSursa,
      parte,
      capitol,
      subsectiune,
      ancora,
      ...parseAncora(ancora ?? ""),
      pagina: paginaStart,
      lungime: text.length,
    };

    // ⚠️ O unitate poate fi prea lungă pentru modelul de embeddings.
    //    Atunci o tăiem în bucăți — DAR fiecare bucată păstrează metadatele
    //    unității. Asta e diferența față de chunking-ul naiv: subîmpărțim,
    //    fără să pierdem apartenența.
    if (text.length <= maxCaractere) {
      documente.push(new Document({ pageContent: text, metadata: meta }));
      return;
    }

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: maxCaractere,
      chunkOverlap: 200,
      separators: ["\n\n", ". ", " ", ""],
    });
    const bucati = await splitter.splitText(text);
    bucati.forEach((b, i) =>
      documente.push(
        new Document({
          pageContent: b,
          metadata: { ...meta, lungime: b.length, parte_din: `${i + 1}/${bucati.length}` },
        }),
      ),
    );
  };

  for (const rand of randuri) {
    switch (rand.rol) {
      case "antet_subsol":
        break; // se aruncă: numere de pagină, colontitluri

      case "titlu_parte":
        await inchideUnitatea();
        parte = rand.text;
        capitol = undefined;
        subsectiune = undefined;
        ancora = undefined;
        break;

      case "titlu_capitol":
        await inchideUnitatea();
        capitol = rand.text;
        subsectiune = undefined;
        ancora = undefined;
        break;

      case "subsectiune":
        await inchideUnitatea();
        subsectiune = rand.text;
        ancora = undefined;
        break;

      case "ancora": {
        // ⚠️ Ancora e ÎNCEPUTUL unei unități noi, dar rândul conține și
        //    primul fragment de text. Închidem ce era înainte, apoi începem
        //    unitatea nouă CU acest rând inclus.
        await inchideUnitatea();
        const potrivire = /^([^.]*\d+:\d+[^.]*)\./.exec(rand.text);
        ancora = potrivire ? potrivire[1].trim() : rand.text.slice(0, 40);
        paginaStart = rand.pagina;
        acumulate.push(rand.text);
        break;
      }

      case "corp":
        if (acumulate.length === 0) paginaStart = rand.pagina;
        acumulate.push(rand.text);
        break;
    }
  }

  await inchideUnitatea();
  return documente;
}

// ─────────────────────────────────────────────────────────────────────────
// Statistici, pentru slide-ul de comparație
// ─────────────────────────────────────────────────────────────────────────

export function statistici(documente: Document[]) {
  const lungimi = documente.map((d) => d.pageContent.length);
  const cuAncora = documente.filter((d) => d.metadata.ancora).length;
  const cuCarteBiblica = documente.filter((d) => d.metadata.carte_biblica).length;

  return {
    chunkuri: documente.length,
    lungimeMedie: Math.round(lungimi.reduce((a, b) => a + b, 0) / (lungimi.length || 1)),
    lungimeMin: Math.min(...lungimi),
    lungimeMax: Math.max(...lungimi),
    cuAncora,
    cuCarteBiblica,
    /** Câte chimpuri de metadate are, în medie — măsura pe ce poți filtra. */
    campuriMetadate: Math.round(
      documente.reduce((s, d) => s + Object.keys(d.metadata).length, 0) /
        (documente.length || 1),
    ),
  };
}
