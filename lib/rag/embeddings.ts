// lib/rag/embeddings.ts
//
// ═══════════════════════════════════════════════════════════════════════
// PASUL 4 DIN PIPELINE — EMBEDDINGS
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 Ce e un embedding, în două fraze:
//
//    Un model care transformă un text într-un vector de numere — la noi,
//    768 sau 3072 de dimensiuni. Texte cu înțeles apropiat primesc vectori
//    apropiați în spațiul ăla, chiar dacă nu au niciun cuvânt comun.
//
//    De asta «pedeapsa lui Dumnezeu pentru păcatele lui Israel» găsește un
//    paragraf despre «divine judgment on the ten tribes», deși nu împart
//    niciun cuvânt. Căutarea pe cuvinte cheie n-ar găsi nimic.
//
// ⚠️ PROBLEMA CARE SURPRINDE PE TOATĂ LUMEA: Anthropic NU are model de
//    embeddings. Nici nu plănuiește; documentația lor îndrumă către alți
//    furnizori. Deci într-o aplicație RAG cu Claude ai OBLIGATORIU doi
//    furnizori: unul care generează, altul care caută.
//
//    E o lecție de arhitectură, nu o neplăcere: modelul care scrie și
//    modelul care găsește sunt lucruri diferite, cu cerințe diferite.
//
// Opțiunile, cu prețuri orientative la 1M de tokeni:
//
//   Google   gemini-embedding-001         cel mai ieftin, 100+ limbi
//   OpenAI   text-embedding-3-small       echilibrat, foarte răspândit
//   Voyage   voyage-3-large               recomandat oficial pentru Claude
//   Ollama   nomic-embed-text             gratis, dar rulează local
//
// Am ales Google pentru că suportă explicit româna și pentru că proiectul
// are deja `@langchain/google-genai` din Lab 1.

import { TaskType } from "@google/generative-ai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { OpenAIEmbeddings } from "@langchain/openai";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";

/**
 * Modelul implicit.
 *
 * ⚠️ `text-embedding-004` a fost scos din uz în ianuarie 2026. Dacă vezi
 *    tutoriale care îl folosesc, sunt vechi.
 */
const MODEL_IMPLICIT = "gemini-embedding-001";

/**
 * Modelul implicit pentru OpenAI.
 *
 * ⚠️ NU «text-embedding-ada-002». Ăla e din 2022, e depreciat, și e mai
 *    scump decât înlocuitorul lui. Îl scriem explicit ca să nu preluăm
 *    valoarea implicită a pachetului, care e chiar ada-002.
 */
const MODEL_OPENAI = "text-embedding-3-small";

/**
 * 🔍 Dimensiunea vectorilor: gemini-embedding-001 produce implicit 3072.
 *    API-ul permite trunchierea Matryoshka (3072 → 1536 → 768) prin
 *    `outputDimensionality`, iar `@langchain/google-genai` o expune
 *    (verificat în 2.3.2; versiunile mai vechi nu o aveau). Noi rămânem pe
 *    implicit: vectori mai mari, dar fără compromis de calitate.
 *
 *    Merită verificat de fiecare dată în fișierul de tipuri al pachetului
 *    instalat — SDK-urile JS și Python nu au mereu aceleași opțiuni.
 */

export type ScopEmbeddings = "documente" | "intrebare";

/**
 * Providerul de embeddings, comutabil din environment.
 *
 * 🎯 De ce e comutabil și nu fix: ca să poți rula ACEEAȘI întrebare, pe
 *    ACELAȘI corpus, cu două modele diferite, și să vezi diferența pe ecran.
 *    E singurul mod onest de a susține afirmația «modelul de embeddings
 *    contează» — altfel e o părere.
 */
export type ProviderEmbeddings = "google" | "openai";

export function citesteProviderEmbeddings(): ProviderEmbeddings {
  const brut = (process.env.EMBEDDINGS_PROVIDER ?? "google").toLowerCase();
  if (brut !== "google" && brut !== "openai") {
    throw new Error(
      `EMBEDDINGS_PROVIDER=«${brut}» nu e valid. Acceptate: google, openai.`,
    );
  }
  return brut;
}

export interface OptiuniEmbeddings {
  provider?: ProviderEmbeddings;
  model?: string;
  scop?: ScopEmbeddings;
}

/**
 * Creează modelul de embeddings, cu aceleași convenții ca `lib/llm.ts`.
 *
 * 🎯 ELEMENT NOU IMPORTANT: `taskType`.
 *
 *    Modelul produce vectori DIFERIȚI pentru același text, în funcție de
 *    rolul lui. Un paragraf care urmează să fie căutat se codifică altfel
 *    decât o întrebare care caută.
 *
 *    Motivul e asimetria: întrebarea «ce zice despre judecata lui Israel?»
 *    și paragraful care răspunde nu seamănă lexical deloc. Modelul e
 *    antrenat să apropie exact perechile astea — dar numai dacă îi spui
 *    care e care.
 *
 *    RETRIEVAL_DOCUMENT → la indexare
 *    RETRIEVAL_QUERY    → la căutare
 *
 * ⚠️ Dacă folosești același taskType pentru amândouă, retrieval-ul merge —
 *    dar mai prost, fără niciun mesaj de eroare. E o pierdere tăcută de
 *    calitate, genul cel mai greu de depistat.
 */
export function creeazaEmbeddings(
  optiuni: OptiuniEmbeddings = {},
): EmbeddingsInterface {
  const provider = optiuni.provider ?? citesteProviderEmbeddings();

  if (provider === "openai") {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "OPENAI_API_KEY lipsă. Setează-l sau pune EMBEDDINGS_PROVIDER=google.",
      );
    }

    // ⚠️ CAPCANĂ VERIFICATĂ ÎN COD: modelul implicit al pachetului
    //    `@langchain/openai` e «text-embedding-ada-002» — modelul din 2022,
    //    depreciat. `new OpenAIEmbeddings()` fără parametri îți dă modelul
    //    vechi, tăcut. De aceea îl scriem EXPLICIT aici.
    //
    //    Diferența, măsurată de OpenAI între ada-002 și 3-large:
    //      MIRACL (multilingv):  31,4% → 54,9%
    //      MTEB (engleză):       61,0% → 64,6%
    //    iar 3-small e și mai ieftin decât ada-002.
    //
    // 🔍 `dimensions` funcționează în pachetul JS de OpenAI, dar NU în cel
    //    de Google. Fiecare SDK e incomplet în alt fel.
    //
    // ⚠️ OpenAI NU are conceptul de taskType: embeddings-urile lui sunt
    //    simetrice. Codul se simplifică, dar pierzi optimizarea pentru
    //    asimetria întrebare-document.
    return new OpenAIEmbeddings({
      apiKey,
      model: optiuni.model ?? process.env.EMBEDDINGS_MODEL ?? MODEL_OPENAI,
      // Trunchiere Matryoshka: vectori de 4× mai mici decât cei impliciți
      // (3072 → 768), cu pierdere mică de calitate.
      // ⚠️ NU face vectorii comparabili cu ai altui model. Modele diferite
      //    produc spații diferite, chiar la aceeași dimensiune — de aceea
      //    orice schimbare de provider sau model cere reindexare.
      dimensions: 768,
    }) as unknown as EmbeddingsInterface;
  }

  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error(
      "GOOGLE_API_KEY lipsă. Embeddings-urile vin de la Google, nu de la providerul de chat. " +
        "Source ~/.llm_gemini sau setează în .env.local.",
    );
  }

  return new GoogleGenerativeAIEmbeddings({
    apiKey,
    model: optiuni.model ?? process.env.EMBEDDINGS_MODEL ?? MODEL_IMPLICIT,
    taskType:
      optiuni.scop === "intrebare"
        ? TaskType.RETRIEVAL_QUERY
        : TaskType.RETRIEVAL_DOCUMENT,
  }) as unknown as EmbeddingsInterface;
}

/**
 * Ambele roluri într-o singură interfață.
 *
 * 🔍 O instanță de embeddings are UN singur taskType. Ca să folosim
 *    RETRIEVAL_DOCUMENT la indexare și RETRIEVAL_QUERY la căutare, ne
 *    trebuie două instanțe — iar clasa asta le rutează transparent, ca
 *    restul codului să nu știe nimic despre asta.
 */
export class EmbeddingsPeScop implements EmbeddingsInterface {
  constructor(
    private pentruDocumente: EmbeddingsInterface,
    private pentruIntrebari: EmbeddingsInterface,
  ) {}

  embedDocuments(texte: string[]) {
    return this.pentruDocumente.embedDocuments(texte);
  }

  embedQuery(text: string) {
    return this.pentruIntrebari.embedQuery(text);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Embeddings cu cache pe disc
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 De ce ne trebuie, foarte practic: indexarea unei cărți de 300 de pagini
//    înseamnă câteva mii de apeluri. Dacă schimbi o virgulă în chunker și
//    reindexezi, plătești din nou și aștepți din nou.
//
//    Cache-ul face diferența dintre «rulez de 3 ori în timpul labului» și
//    «rulez o dată și mă rog să meargă».
//
// ⚠️ Cheia de cache include MODELUL și DIMENSIUNILE. Altfel, dacă schimbi
//    modelul, primești vectori vechi de la altul — și rezultate care nu au
//    nicio legătură, fără niciun mesaj de eroare.

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

export class EmbeddingsCuCache implements EmbeddingsInterface {
  /** Câte texte trimitem odată înainte de a salva pe disc. */
  static LOT = 100;

  private cache = new Map<string, number[]>();
  /** Promisiunea încărcării, ca două cereri simultane să aștepte aceeași. */
  private incarcare?: Promise<void>;

  constructor(
    private baza: EmbeddingsInterface,
    private caleCache = ".cache/embeddings.json",
    private eticheta = MODEL_IMPLICIT,
  ) {}

  private cheie(text: string) {
    return createHash("sha1").update(this.eticheta + "\u0000" + text).digest("hex");
  }

  private incarca() {
    // ⚠️ Un simplu boolean setat la început ar lăsa a doua cerere să treacă
    //    înainte ca fișierul să fie citit — cu cache-ul gol, pe care apoi
    //    l-ar salva peste cel bun. Ținem promisiunea, nu un flag.
    this.incarcare ??= this.citesteDePeDisc();
    return this.incarcare;
  }

  private async citesteDePeDisc() {
    try {
      const brut = await readFile(this.caleCache, "utf8");
      for (const [k, v] of Object.entries(JSON.parse(brut) as Record<string, number[]>)) {
        this.cache.set(k, v);
      }
      console.log(`[embeddings] cache încărcat: ${this.cache.size} vectori`);
    } catch {
      // Nu există încă. Normal la prima rulare.
    }
  }

  private async salveaza() {
    await mkdir(dirname(this.caleCache), { recursive: true });
    await writeFile(this.caleCache, JSON.stringify(Object.fromEntries(this.cache)));
  }

  async embedDocuments(texte: string[]): Promise<number[][]> {
    await this.incarca();

    const lipsa: { index: number; text: string }[] = [];
    const rezultat: number[][] = new Array(texte.length);

    texte.forEach((t, i) => {
      const v = this.cache.get(this.cheie(t));
      if (v) rezultat[i] = v;
      else lipsa.push({ index: i, text: t });
    });

    if (lipsa.length > 0) {
      console.log(
        `[embeddings] ${texte.length - lipsa.length} din cache, ${lipsa.length} de calculat`,
      );
      // ⚠️ În loturi, cu salvare după FIECARE lot. Dacă providerul dă rate
      //    limit la chunk-ul 2.900 din 3.000, ce s-a calculat până atunci
      //    rămâne pe disc — rulezi din nou și plătești doar restul.
      for (let start = 0; start < lipsa.length; start += EmbeddingsCuCache.LOT) {
        const lot = lipsa.slice(start, start + EmbeddingsCuCache.LOT);
        const noi = await this.baza.embedDocuments(lot.map((l) => l.text));
        lot.forEach((l, k) => {
          rezultat[l.index] = noi[k];
          this.cache.set(this.cheie(l.text), noi[k]);
        });
        await this.salveaza();
        if (lipsa.length > EmbeddingsCuCache.LOT) {
          console.log(
            `[embeddings] ${Math.min(start + lot.length, lipsa.length)}/${lipsa.length} salvate`,
          );
        }
      }
    } else {
      console.log(`[embeddings] toate cele ${texte.length} din cache`);
    }

    return rezultat;
  }

  async embedQuery(text: string): Promise<number[]> {
    await this.incarca();
    const cheie = this.cheie(text);
    const gasit = this.cache.get(cheie);
    if (gasit) return gasit;

    const vector = await this.baza.embedQuery(text);
    this.cache.set(cheie, vector);
    await this.salveaza();
    return vector;
  }
}

/**
 * Varianta pe care o folosim peste tot: taskType corect pe fiecare rol,
 * plus cache pe disc.
 *
 * ⚠️ Cheia de cache include modelul ȘI rolul. Altfel, dacă schimbi modelul
 *    sau taskType-ul, primești vectori vechi de la altă configurație — și
 *    rezultate fără nicio legătură, fără niciun mesaj de eroare.
 */
export function creeazaEmbeddingsCuCache(optiuni: OptiuniEmbeddings = {}) {
  const provider = optiuni.provider ?? citesteProviderEmbeddings();
  const model =
    optiuni.model ??
    process.env.EMBEDDINGS_MODEL ??
    (provider === "openai" ? MODEL_OPENAI : MODEL_IMPLICIT);

  // ⚠️ Cache-uri SEPARATE per provider și model. Dacă ai un singur fișier
  //    de cache și schimbi providerul, primești vectori de la modelul vechi
  //    pentru texte pe care le crezi reindexate — și rezultate fără nicio
  //    legătură, fără niciun mesaj de eroare.
  const numeFisier = `${provider}-${model}`.replace(/[^a-zA-Z0-9.-]+/g, "_");
  const documente = new EmbeddingsCuCache(
    creeazaEmbeddings({ provider, model, scop: "documente" }),
    `.cache/emb-${numeFisier}-doc.json`,
    `${provider}:${model}@RETRIEVAL_DOCUMENT`,
  );
  const intrebari = new EmbeddingsCuCache(
    creeazaEmbeddings({ provider, model, scop: "intrebare" }),
    `.cache/emb-${numeFisier}-query.json`,
    `${provider}:${model}@RETRIEVAL_QUERY`,
  );

  return new EmbeddingsPeScop(documente, intrebari);
}

/** Eticheta configurației active — apare în interfață și în citări. */
export function etichetaEmbeddings(optiuni: OptiuniEmbeddings = {}) {
  const provider = optiuni.provider ?? citesteProviderEmbeddings();
  const model =
    optiuni.model ??
    process.env.EMBEDDINGS_MODEL ??
    (provider === "openai" ? MODEL_OPENAI : MODEL_IMPLICIT);
  return `${provider} · ${model}`;
}
