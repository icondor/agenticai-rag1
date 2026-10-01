// lib/rag/extrage-pdf.ts
//
// ═══════════════════════════════════════════════════════════════════════
// PASUL 1 DIN PIPELINE — ÎNCĂRCAREA DOCUMENTELOR
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 Cea mai subestimată etapă din tot RAG-ul. Toată lumea începe de la
//    chunking, pentru că așa arată tutorialele. Dar dacă extragerea e
//    proastă, tot ce urmează e proastă: embeddings pe text stricat,
//    retrieval slab, citări urâte pe ecran.
//
//    Regula: retrieval-ul tău e cel mult atât de bun cât e extragerea.
//
// ⚠️ CAPCANA TEXTULUI JUSTIFICAT — măsurată pe cartea din lab.
//
//    Într-un PDF cu text justificat, spațierea dintre cuvinte variază de la
//    rând la rând. Un extractor naiv (sau un simplu copy-paste) nu recunoaște
//    capătul de rând ca separator și lipește cuvintele:
//
//      «stated thebasic theology»          în loc de  «stated the basic theology»
//      «call out thechurch»                în loc de  «call out the church»
//      «...and Her RestorationHosea 1:2»   în loc de  două rânduri separate
//
//    Am încercat repararea automată, cu dicționar de 275.000 de cuvinte:
//    recuperează ~99%, dar restul rămâne stricat («churchto») și, la o carte
//    de 300 de pagini, alea sunt câteva sute de cuvinte împrăștiate — unele
//    exact în chunk-urile pe care agentul le citează pe ecran.
//
//    Concluzia: repari la SURSĂ, nu după. De aceea fișierul ăsta există.
//
// ELEMENTE NOI:
//
//   • pdfjs-dist  → biblioteca de citit PDF din browser-ul Firefox, rulată
//                   în Node. Ne dă textul CU informații de font.
//   • getTextContent() → fragmente de text, fiecare cu fontName, dimensiune
//                        și poziție (x, y) pe pagină.

import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";

// ─────────────────────────────────────────────────────────────────────────
// Ce știm despre un rând de text
// ─────────────────────────────────────────────────────────────────────────

/**
 * Rolul unui rând, dedus din tipografie.
 *
 * 🎯 De ce prin FONT și nu prin regex: într-o carte de comentarii, un rând
 *    ca «Osea 1:1.» care deschide o unitate e cules cu italic. Dar o
 *    referință din mijlocul frazei, «(1 Împărați 21:21–24)», e cules cu fontul
 *    de corp. Un regex le confundă. Fontul nu.
 */
export type RolRand =
  | "titlu_parte" // «PROFEȚIA ÎN PROFEȚII MICI» — bold, centrat
  | "titlu_capitol" // «PROFEȚIA LUI OSEA» — corp mărit
  | "subsectiune" // «The First Cycle of the Sins...» — bold italic
  | "ancora" // «Osea 1:2–2:1.» — italic, deschide o unitate
  | "corp" // textul propriu-zis
  | "antet_subsol"; // numere de pagină — se aruncă (vezi deducRol pentru limite)

export interface RandExtras {
  text: string;
  rol: RolRand;
  pagina: number;
  /** Coordonata X a începutului — folosită ca să detectăm centrarea. */
  x: number;
  /** Dimensiunea fontului, în puncte. */
  marime: number;
  /**
   * Fontul DOMINANT al rândului (cele mai multe caractere), numele intern
   * pdfjs, ex. «g_d0_f3». Decide dacă rândul e corp sau subsecțiune.
   */
  font: string;
  /**
   * Fontul cu care ÎNCEPE rândul. Folosit doar la detectarea ancorei: o
   * ancoră începe în italic («Osea 1:2.») și continuă, pe același rând, cu
   * text de corp — deci fontul dominant al rândului poate fi cel de corp.
   * Lipsește la corpusul de test (acolo rolul e dat direct).
   */
  fontInceput?: string;
}

// ─────────────────────────────────────────────────────────────────────────
// Fragmente → rânduri
// ─────────────────────────────────────────────────────────────────────────

export interface Fragment {
  str: string;
  transform: number[];
  fontName: string;
}

/**
 * Grupează fragmentele care stau pe aceeași linie de bază.
 *
 * 🔍 De ce e nevoie: pdfjs întoarce text la nivel de FRAGMENT, nu de rând.
 *    Un rând cu un cuvânt italic în mijloc vine ca trei fragmente separate.
 *    Le regrupăm după coordonata Y (transform[5]), cu toleranță de 2 puncte
 *    pentru micile variații de poziționare.
 */
export function grupeazaInRanduri(fragmente: Fragment[], pagina: number) {
  const randuri: { y: number; fragmente: Fragment[] }[] = [];

  for (const fr of fragmente) {
    if (!fr.str) continue;
    const y = Math.round(fr.transform[5]);
    const ultimul = randuri[randuri.length - 1];

    if (ultimul && Math.abs(ultimul.y - y) <= 2) ultimul.fragmente.push(fr);
    else randuri.push({ y, fragmente: [fr] });
  }

  return randuri
    .map((r) => {
      const cuText = r.fragmente.filter((f) => f.str.trim());
      const primul = cuText[0] ?? r.fragmente[0];

      // ⚠️ Fontul rândului = cel cu cele mai multe CARACTERE, nu cel al
      //    primului fragment. Un rând de corp care începe cu un cuvânt în
      //    italic (*hesed*, *agape*, un titlu de carte) ar fi luat drept
      //    subsecțiune: chunkerul ar tăia pasajul în mijlocul frazei, ar
      //    scoate rândul din text și ar pierde ancora pentru restul lui.
      //    Tăcut — singurul simptom e un număr mare de «subsectiune».
      const caracterePeFont = new Map<string, { n: number; fr: Fragment }>();
      for (const f of cuText) {
        const cheie = `${f.fontName}@${Math.round(Math.abs(f.transform[0]) * 10) / 10}`;
        const v = caracterePeFont.get(cheie) ?? { n: 0, fr: f };
        v.n += f.str.trim().length;
        caracterePeFont.set(cheie, v);
      }
      const principal =
        [...caracterePeFont.values()].sort((a, b) => b.n - a.n)[0]?.fr ?? primul;

      return {
        text: r.fragmente
          .map((f) => f.str)
          .join("")
          .replace(/\s+/g, " ")
          .trim(),
        pagina,
        y: r.y,
        x: Math.min(...r.fragmente.map((f) => f.transform[4])),
        marime: Math.round(Math.abs(principal.transform[0]) * 10) / 10,
        font: principal.fontName,
        fontInceput: primul.fontName,
      };
    })
    .filter((r) => r.text.length > 0);
}

// ─────────────────────────────────────────────────────────────────────────
// Deducerea rolurilor
// ─────────────────────────────────────────────────────────────────────────

/**
 * Află ce combinație (font, mărime) e corpul textului.
 *
 * 🎯 De ce NU hardcodăm numele fonturilor: pdfjs generează nume interne
 *    («g_d0_f2») care diferă de la un PDF la altul, chiar și pentru aceeași
 *    carte re-exportată. Un tabel fix ar merge pe cartea de test și ar cădea
 *    pe următoarea.
 *
 *    În schimb, corpul e întotdeauna combinația CEA MAI FRECVENTĂ. Titlurile
 *    și ancorele sunt, prin definiție, rare. Deci deducem, nu presupunem.
 */
function detecteazaCorpul(
  randuri: { font: string; marime: number; text: string; x: number }[],
) {
  const frecvente = new Map<string, { n: number; caractere: number }>();

  for (const r of randuri) {
    const cheie = `${r.font}@${r.marime}`;
    const v = frecvente.get(cheie) ?? { n: 0, caractere: 0 };
    v.n += 1;
    v.caractere += r.text.length;
    frecvente.set(cheie, v);
  }

  // Sortăm după numărul de CARACTERE, nu de rânduri: un titlu poate apărea
  // des într-o carte cu multe capitole, dar are puține caractere.
  const [cheieCorp] = [...frecvente.entries()].sort(
    (a, b) => b[1].caractere - a[1].caractere,
  )[0];

  const [font, marime] = cheieCorp.split("@");

  // Marginea din stânga a corpului: cea mai frecventă valoare X printre
  // rândurile de corp. Titlurile centrate încep mai la dreapta decât ea.
  const xCorp = randuri
    .filter((r) => `${r.font}@${r.marime}` === cheieCorp)
    .map((r) => Math.round(r.x));
  const frecventaX = new Map<number, number>();
  for (const x of xCorp) frecventaX.set(x, (frecventaX.get(x) ?? 0) + 1);
  const margine = [...frecventaX.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;

  return { font, marime: Number(marime), margine };
}

export interface OptiuniRol {
  fontCorp: string;
  marimeCorp: number;
  /** Marginea din stânga a corpului — reperul față de care judecăm centrarea. */
  margineCorp: number;
}

/**
 * Regexul de referință biblică, la începutul unui rând.
 *
 * Grupul 1 = cartea, grupul 2 = capitolul. Acoperă:
 *   «Osea 1:2–2:1.»   «Romani 5:9»   «Ieșirea 3:14»   «1 Corinteni 13:4»
 *   «Cântarea Cântărilor 2:1»   «Plângerile lui Ieremia 3:22»   «Gen. 1:1»
 *
 * ⚠️ `\p{Lu}` / `\p{Ll}` (cu flag-ul `u`) și nu `[A-Z]` / `[a-z]`: cu
 *    clasele ASCII, «Ieșirea» și «Cântarea» nu se potrivesc, iar ancora
 *    devine subsecțiune — tăcut. La cărți în română, asta înseamnă cam
 *    jumătate din referințe pierdute.
 *
 * Folosit și de chunker.ts, ca să extragă cartea și capitolul.
 */
export const REFERINTA =
  /^((?:[1-3]\s?)?\p{Lu}\p{Ll}+(?:\s+\p{L}+){0,3}?)\.?\s+(\d+):(\d+)/u;

export function deducRol(
  rand: { text: string; x: number; marime: number; font: string; fontInceput?: string },
  o: OptiuniRol,
): RolRand {
  const { fontCorp, marimeCorp, margineCorp } = o;
  const eMaiMare = rand.marime > marimeCorp + 0.5;
  const altFont = rand.font !== fontCorp;
  const incepeAltFont = (rand.fontInceput ?? rand.font) !== fontCorp;
  const totMajuscule = rand.text === rand.text.toUpperCase() && /[A-Z]/.test(rand.text);

  // ⚠️ Centrarea NU se judecă față de lățimea paginii. Un titlu lung, centrat,
  //    începe aproape de marginea din stânga — l-am clasificat greșit prima dată.
  //    Reperul corect e marginea CORPULUI: dacă rândul începe vizibil mai la
  //    dreapta decât textul normal, e centrat.
  const eCentrat = rand.x > margineCorp + 20;

  // Numere de pagină: un rând care conține doar un număr.
  // ⚠️ Colontitlurile cu TEXT (titlul cărții repetat sus pe fiecare pagină)
  //    NU sunt detectate aici — ajung în corp și, de acolo, în chunkuri.
  //    Dacă o carte are colontitluri, se vede imediat în fragmentele de pe
  //    ecran. Fixul tipic: un rând scurt, identic, care apare pe multe pagini.
  if (/^\d{1,4}$/.test(rand.text)) return "antet_subsol";

  // Titlu de parte: majuscule + centrat + mai mare decât corpul.
  if (eMaiMare && totMajuscule && eCentrat) return "titlu_parte";

  // Titlu de capitol: majuscule + mai mare, dar aliniat la stânga.
  if (eMaiMare && totMajuscule) return "titlu_capitol";

  // Orice altceva mai mare rămâne tot titlu.
  if (eMaiMare) return "titlu_capitol";

  // La aceeași mărime, un font diferit înseamnă italic sau bold italic.
  //
  // Ancora: rândul ÎNCEPE în alt font cu o referință biblică. Restul
  // rândului poate fi corp — de aceea ne uităm la fontul de început.
  if (incepeAltFont && REFERINTA.test(rand.text)) return "ancora";

  // Subsecțiune: rândul e în alt font aproape în întregime (fontul DOMINANT
  // e altul). Un cuvânt italic la început nu mai ajunge.
  if (altFont) return "subsectiune";

  return "corp";
}

// ─────────────────────────────────────────────────────────────────────────
// Funcția publică
// ─────────────────────────────────────────────────────────────────────────

export interface CarteExtrasa {
  titlu: string;
  randuri: RandExtras[];
  /** Statistici utile pentru slide-ul de indexare. */
  statistici: {
    pagini: number;
    randuri: number;
    caractere: number;
    peRoluri: Record<string, number>;
  };
}

/**
 * Extrage o carte din PDF, cu rolul fiecărui rând.
 *
 * @param cale     calea către PDF pe disc
 * @param titlu    cum se numește cartea în citări
 */
export async function extrageCarte(
  cale: string,
  titlu: string,
): Promise<CarteExtrasa> {
  const { readFile } = await import("node:fs/promises");
  const date = new Uint8Array(await readFile(cale));

  // useSystemFonts: pdfjs are nevoie de fonturi ca să măsoare corect textul.
  const pdf = await getDocument({ data: date, useSystemFonts: true }).promise;

  // ── Trecerea 1: adunăm toate rândurile, fără să le judecăm ──────────
  const brute: (Omit<RandExtras, "rol"> & { y: number })[] = [];
  for (let nr = 1; nr <= pdf.numPages; nr++) {
    const pagina = await pdf.getPage(nr);
    const continut = await pagina.getTextContent();
    brute.push(...grupeazaInRanduri(continut.items as Fragment[], nr));
  }

  // ⚠️ Un PDF scanat are doar imagini, fără strat de text: pdfjs nu
  //    întoarce nimic. Fără verificarea asta, eroarea ar fi un TypeError
  //    criptic din detecteazaCorpul.
  if (brute.length === 0) {
    throw new Error(
      `«${titlu}»: PDF-ul nu are text extractibil — probabil e scanat. ` +
        `Trece-l întâi printr-un OCR (ex. ocrmypdf) și indexează rezultatul.`,
    );
  }

  // ── Trecerea 2: deducem ce e corpul, apoi rolul fiecărui rând ───────
  //
  // 🎯 De ce două treceri: rolul unui rând se definește RELATIV la restul
  //    cărții. Nu poți ști că 20.2pt e „mai mare" până nu știi că 14.4pt e
  //    normalul. Prima trecere adună, a doua judecă.
  const {
    font: fontCorp,
    marime: marimeCorp,
    margine: margineCorp,
  } = detecteazaCorpul(brute);

  const randuri: RandExtras[] = brute.map((r) => ({
    text: r.text,
    pagina: r.pagina,
    x: r.x,
    marime: r.marime,
    font: r.font,
    fontInceput: r.fontInceput,
    rol: deducRol(r, { fontCorp, marimeCorp, margineCorp }),
  }));

  const peRoluri: Record<string, number> = {};
  for (const r of randuri) peRoluri[r.rol] = (peRoluri[r.rol] ?? 0) + 1;

  return {
    titlu,
    randuri,
    statistici: {
      pagini: pdf.numPages,
      randuri: randuri.length,
      caractere: randuri.reduce((s, r) => s + r.text.length, 0),
      peRoluri,
    },
  };
}

/**
 * Unește rândurile de corp într-un paragraf, corect.
 *
 * ⚠️ Două lucruri pe care extractoarele naive le greșesc:
 *
 *   1. Unesc fără spațiu → «thebasic». Noi unim CU spațiu.
 *   2. Lasă cratima de despărțire → «sanctifi- cation». Noi o reparăm:
 *      dacă rândul se termină cu «-» și următorul începe cu literă mică,
 *      lipim fără spațiu și fără cratimă.
 */
export function uneste(randuri: string[]): string {
  let rezultat = "";

  for (const rand of randuri) {
    // `\p{Ll}` și nu `[a-z]`: «sfin-» + «țenie» trebuie lipit și el.
    if (rezultat.endsWith("-") && /^\p{Ll}/u.test(rand)) {
      rezultat = rezultat.slice(0, -1) + rand; // despărțire în silabe
    } else if (rezultat.length === 0) {
      rezultat = rand;
    } else {
      rezultat += " " + rand;
    }
  }

  return rezultat.replace(/\s+/g, " ").trim();
}
