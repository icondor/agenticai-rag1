// verificare/test-lab9.ts
//
// Verificări pentru pipeline-ul de RAG, cu embeddings fals și model fals.
// Fără chei API, fără costuri. Rulează în câteva secunde.
//
//   pnpm verifica

import { existsSync } from "node:fs";
import { createAgent, FakeToolCallingModel, toolCallLimitMiddleware } from "langchain";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type { StructuredToolInterface } from "@langchain/core/tools";

import {
  deducRol,
  extrageCarte,
  grupeazaInRanduri,
  REFERINTA,
  uneste,
  type Fragment as FragmentPdf,
} from "@/lib/rag/extrage-pdf";
import { chunkNaiv, chunkStructural, statistici } from "@/lib/rag/chunker";
import { VectorStoreSimplu, cosinus } from "@/lib/rag/vector-store";
import { asambleazaContext, reordoneazaPentruAtentie } from "@/lib/rag/rag-clasic";
import { creeazaBibliotecaTools } from "@/lib/tools/biblioteca-tools";
import { evalueazaRetrieval } from "@/lib/rag/evaluare";
import {
  citesteProfil,
  incepeSesiune,
  inregistreazaIntrebare,
  profilInPrompt,
} from "@/lib/rag/memorie-studiu";
import { CARTE_A, CARTE_B, CAZURI_TEST } from "@/verificare/corpus-test";

let esecuri = 0;
function verifica(nume: string, conditie: boolean, detaliu = "") {
  console.log(`${conditie ? "PASS" : "FAIL"}  ${nume}${detaliu ? " :: " + detaliu : ""}`);
  if (!conditie) esecuri += 1;
}

/** Embeddings deterministe pe cuvinte — suficient ca retrieval-ul să aibă sens. */
class EmbFals implements EmbeddingsInterface {
  async embedQuery(t: string) {
    return this.v(t);
  }
  async embedDocuments(ts: string[]) {
    return ts.map((t) => this.v(t));
  }
  private v(text: string) {
    const x = new Array(96).fill(0);
    for (const cuvant of text.toLowerCase().split(/\W+/)) {
      if (cuvant.length < 3) continue;
      let h = 0;
      for (const c of cuvant) h = (h * 31 + c.charCodeAt(0)) % 96;
      x[h] += 1;
    }
    return x;
  }
}

async function storeDeTest() {
  const store = new VectorStoreSimplu(new EmbFals(), {});
  const a = await chunkStructural(CARTE_A, "Istoria moderna a romanilor");
  const b = await chunkStructural(CARTE_B, "Fise de istorie romaneasca");
  await store.addDocuments([...a, ...b]);
  return store;
}

// ── T1: extragerea din PDF real ────────────────────────────────────────
//
// Are nevoie de un PDF real (materiale protejate, nu e în repo). Calea se
// poate schimba cu PDF_TEST=...; dacă fișierul lipsește, T1 e sărit.
const PDF_TEST = process.env.PDF_TEST ?? "biblioteca/pagini_251_252.pdf";
if (!existsSync(PDF_TEST)) {
  console.log(`SKIP  T1 — nu găsesc «${PDF_TEST}» (setează PDF_TEST=...)`);
} else {
  const carte = await extrageCarte(PDF_TEST, "carte de test");
  const roluri = carte.statistici.peRoluri;
  verifica("T1 extrage toate cele 5 roluri", Object.keys(roluri).length >= 5, Object.keys(roluri).join(","));
  verifica("T1 detectează ancore", (roluri.ancora ?? 0) >= 3, String(roluri.ancora));
  verifica("T1 aruncă numerele de pagină", (roluri.antet_subsol ?? 0) >= 1);

  const corp = uneste(carte.randuri.filter((r) => r.rol === "corp").map((r) => r.text));
  const lipiri = ["concerningHosea", "exceptthe", "describedas", "kingsof"];
  verifica("T1 zero cuvinte lipite", lipiri.every((l) => !corp.includes(l)));
}

// ── T2: uneste repară despărțirea în silabe ────────────────────────────
{
  verifica(
    "T2 cratima de despărțire se repară",
    uneste(["sanctifi-", "cation este"]) === "sanctification este",
    uneste(["sanctifi-", "cation este"]),
  );
  verifica(
    "T2 rândurile normale se unesc cu spațiu",
    uneste(["primul rand", "al doilea"]) === "primul rand al doilea",
  );
  verifica(
    "T2 despărțirea se repară și înainte de diacritice",
    uneste(["sfin-", "țenia este"]) === "sfințenia este",
    uneste(["sfin-", "țenia este"]),
  );
}

// ── T1b: rolul rândului, din fonturi simulate ─────────────────────────
//
// Fără PDF: construim fragmentele cum le-ar da pdfjs. «corp» = fontul de
// corp, «ital» = italic, ambele la 12pt.
{
  const fr = (str: string, font: string, y: number, x = 72): FragmentPdf => ({
    str,
    fontName: font,
    transform: [12, 0, 0, 12, x, y],
  });
  const randuri = grupeazaInRanduri(
    [
      // un rând de corp care începe cu un cuvânt în italic
      fr("hesed ", "ital", 700),
      fr("înseamnă îndurare statornică și apare des la profeții mici", "corp", 700, 110),
      // o ancoră: referința în italic, apoi text de corp pe același rând
      fr("Osea 1:2. ", "ital", 680),
      fr("Căsătoria lui Osea cu Gomer este un semn profetic pentru popor", "corp", 680, 130),
      // o subsecțiune: tot rândul în italic
      fr("Primul ciclu: păcatul și restaurarea", "ital", 660),
      // o ancoră scurtă, singură pe rând
      fr("Osea 2:1.", "ital", 640),
    ],
    1,
  );
  const o = { fontCorp: "corp", marimeCorp: 12, margineCorp: 72 };
  const roluri = randuri.map((r) => deducRol(r, o));

  verifica("T1b cuvânt italic la început → rămâne corp", roluri[0] === "corp", roluri[0]);
  verifica("T1b fontul rândului = cel cu cele mai multe caractere", randuri[0].font === "corp", randuri[0].font);
  verifica("T1b referință italic + text de corp → ancoră", roluri[1] === "ancora", roluri[1]);
  verifica("T1b rând întreg în italic → subsecțiune", roluri[2] === "subsectiune", roluri[2]);
  verifica("T1b ancoră singură pe rând → ancoră", roluri[3] === "ancora", roluri[3]);
}

// ── T2b: referințele biblice în română ─────────────────────────────────
{
  const carte = (t: string) => REFERINTA.exec(t)?.[1];
  const cap = (t: string) => Number(REFERINTA.exec(t)?.[2]);
  verifica("T2b «Osea 1:2–2:1.»", carte("Osea 1:2–2:1. Text") === "Osea" && cap("Osea 1:2–2:1.") === 1);
  verifica("T2b diacritice: «Ieșirea 3:14»", carte("Ieșirea 3:14. Text") === "Ieșirea", String(carte("Ieșirea 3:14. Text")));
  verifica("T2b număr + nume: «1 Corinteni 13:4»", carte("1 Corinteni 13:4. Text") === "1 Corinteni");
  verifica("T2b mai multe cuvinte: «Cântarea Cântărilor 2:1»", carte("Cântarea Cântărilor 2:1. Text") === "Cântarea Cântărilor");
  verifica("T2b «Plângerile lui Ieremia 3:22»", carte("Plângerile lui Ieremia 3:22. Text") === "Plângerile lui Ieremia");
  verifica("T2b o frază obișnuită nu e referință", !REFERINTA.test("Acesta este un rând de corp despre 3:14."));
}

// ── T3: chunking structural vs naiv ────────────────────────────────────
{
  const struct = await chunkStructural(CARTE_A, "Istoria moderna a romanilor");
  const text = uneste(CARTE_A.filter((r) => r.rol !== "antet_subsol").map((r) => r.text));
  const naiv = await chunkNaiv(text, "Istoria moderna a romanilor", { chunkSize: 400 });

  const sStruct = statistici(struct);
  const sNaiv = statistici(naiv);

  verifica("T3 structural: toate chunkurile au capitol", struct.every((d) => d.metadata.capitol), String(struct.length));
  verifica("T3 structural: toate au subsecțiune", struct.every((d) => d.metadata.subsectiune));
  verifica(
    "T3 naiv: zero metadate de structură",
    naiv.every((d) => !d.metadata.capitol && !d.metadata.subsectiune),
  );
  verifica("T3 structural are mai multe câmpuri de metadate", sStruct.campuriMetadate > sNaiv.campuriMetadate, `${sStruct.campuriMetadate} vs ${sNaiv.campuriMetadate}`);

  const fanariote = struct.filter((d) => d.metadata.capitol === "EPOCA FANARIOTA");
  verifica("T3 ierarhia se moștenește în epoca fanariotă", fanariote.length === 2, String(fanariote.length));
  verifica("T3 subsecțiunea ajunge în metadate", struct.some((d) => String(d.metadata.subsectiune ?? "").includes("Mavrocordat")));
}

// ── T4: vector store — similaritate, filtre, MMR, persistență ──────────
{
  const store = await storeDeTest();
  verifica("T4 corpusul de test se indexează", store.numarDocumente >= 6, String(store.numarDocumente));
  verifica("T4 două surse distincte", store.valoriDistincte("carte_sursa").length === 2, store.valoriDistincte("carte_sursa").join(", "));

  const q = await new EmbFals().embedQuery("reformele lui Constantin Mavrocordat");
  const fara = await store.similaritySearchVectorWithScore(q, 3);
  verifica("T4 similaritySearch întoarce k rezultate", fara.length === 3);

  const cuFiltru = await store.similaritySearchVectorWithScore(q, 5, (d) => d.metadata.carte_sursa === "Fise de istorie romaneasca");
  verifica("T4 filtrul pe sursă restrânge corect", cuFiltru.every(([d]) => d.metadata.carte_sursa === "Fise de istorie romaneasca"), String(cuFiltru.length));

  const mmr = await store.cautaCuDiversitate(q, 3, { lambda: 0.3 });
  const uniceMmr = new Set(mmr.map(([d]) => d.pageContent)).size;
  verifica("T4 MMR întoarce documente distincte", uniceMmr === mmr.length, `${uniceMmr}/${mmr.length}`);

  await store.salveaza("/tmp/lab9-index.json");
  const nou = new VectorStoreSimplu(new EmbFals(), {});
  const n = await nou.incarca("/tmp/lab9-index.json");
  verifica("T4 salvare + reîncărcare", n === store.numarDocumente, `${n}`);
  verifica("T4 asRetriever merge după reîncărcare", (await nou.asRetriever({ k: 2 }).invoke("reforme")).length === 2);
}

// ── T5: cosinus ────────────────────────────────────────────────────────
{
  verifica("T5 vectori identici → 1", Math.abs(cosinus([1, 2, 3], [1, 2, 3]) - 1) < 1e-9);
  verifica("T5 vectori ortogonali → 0", Math.abs(cosinus([1, 0], [0, 1])) < 1e-9);
  verifica("T5 lungimea nu contează, doar direcția", Math.abs(cosinus([1, 1], [5, 5]) - 1) < 1e-9);

  let aAruncat = false;
  try {
    cosinus([1, 2, 3], [1, 2]);
  } catch {
    aAruncat = true;
  }
  verifica("T5 dimensiuni diferite → eroare, nu NaN tăcut", aAruncat);
}

// ── T6: asamblarea contextului ─────────────────────────────────────────
{
  const docs = [
    new Document({
      pageContent: "text despre instaurarea regimului fanariot",
      metadata: {
        carte_sursa: "Istoria moderna a romanilor",
        pagina: 179,
        capitol: "EPOCA FANARIOTA",
        subsectiune: "Instaurarea regimului fanariot",
      },
    }),
    new Document({
      pageContent: "text despre dubla alegere",
      metadata: {
        carte_sursa: "Istoria moderna a romanilor",
        pagina: 229,
        capitol: "UNIREA PRINCIPATELOR",
        subsectiune: "Dubla alegere a lui Alexandru Ioan Cuza",
      },
    }),
  ];
  const ctx = asambleazaContext(docs);
  verifica("T6 fiecare fragment primește identificator", ctx.includes("[1]") && ctx.includes("[2]"));
  verifica("T6 sursa și pagina intră în context", ctx.includes("Istoria moderna") && ctx.includes("p. 179"));
  verifica("T6 secțiunea istorică intră în context", ctx.includes("EPOCA FANARIOTA") && ctx.includes("Dubla alegere"));

  const reordonat = reordoneazaPentruAtentie([1, 2, 3, 4, 5]);
  verifica("T6 lost-in-the-middle: cel mai bun rămâne primul", reordonat[0] === 1, JSON.stringify(reordonat));
  verifica("T6 al doilea ajunge la coadă", reordonat[reordonat.length - 1] === 2, JSON.stringify(reordonat));

  // După reordonare, fiecare fragment își păstrează numărul de pe ecran.
  const ctxReordonat = asambleazaContext([docs[1], docs[0]], [2, 1]);
  verifica(
    "T6 numerele supraviețuiesc reordonării",
    ctxReordonat.indexOf("[2] Istoria") < ctxReordonat.indexOf("[1] Istoria") &&
      ctxReordonat.includes("[2] Istoria") && ctxReordonat.includes("[1] Istoria"),
  );
}

// ── T7: tool-ul de căutare în agent ────────────────────────────────────
{
  const store = await storeDeTest();
  const unelte: StructuredToolInterface[] = creeazaBibliotecaTools(store);

  const agent = createAgent({
    model: new FakeToolCallingModel({
      toolCalls: [
        [{ name: "cautaInBiblioteca", args: { intrebare: "domnii fanarioti reformatori" }, id: "c1" }],
        [{ name: "cautaInBiblioteca", args: { intrebare: "domnii fanarioti reformatori", carte_sursa: "Fise de istorie" }, id: "c2" }],
        [{ name: "inventarBiblioteca", args: {}, id: "c3" }],
        [],
      ],
    }) as never,
    tools: unelte,
    middleware: [toolCallLimitMiddleware({ toolName: "cautaInBiblioteca", runLimit: 6, exitBehavior: "continue" as const })],
  });

  const r = await agent.invoke({ messages: [new HumanMessage("x")] }, { recursionLimit: 60 });
  const toolMsgs = r.messages.filter(ToolMessage.isInstance);
  const continuturi = toolMsgs.map((m) => JSON.parse(String(m.content)) as Record<string, unknown>);

  verifica("T7 trei apeluri de tool executate", toolMsgs.length === 3, String(toolMsgs.length));
  verifica("T7 prima căutare găsește fragmente", Number(continuturi[0].gasite) > 0, String(continuturi[0].gasite));

  const filtrata = continuturi[1].fragmente as { carte_sursa: string }[];
  verifica("T7 filtrul pe sursă funcționează prin tool", filtrata.every((f) => f.carte_sursa === "Fise de istorie romaneasca"), String(filtrata.length));

  const frag = (continuturi[0].fragmente as Record<string, unknown>[])[0];
  verifica("T7 fragmentul are id, relevanță, sursă, referință", ["id", "relevanta", "carte_sursa", "referinta", "pagina"].every((k) => k in frag));
  verifica("T7 inventarul listează ambele cărți", (continuturi[2].carti as string[]).length === 2, JSON.stringify(continuturi[2].carti));

  const idPrima = (continuturi[0].fragmente as { id: number }[]).map((f) => f.id);
  const idA2a = (continuturi[1].fragmente as { id: number }[]).map((f) => f.id);
  verifica(
    "T7 numerele continuă între căutări",
    idA2a[0] === idPrima[idPrima.length - 1] + 1,
    `${JSON.stringify(idPrima)} → ${JSON.stringify(idA2a)}`,
  );
}

// ── T7b: filtrul pe sursă ignoră majusculele și diacriticele ──────────
{
  const store = await storeDeTest();
  const cauta: StructuredToolInterface = creeazaBibliotecaTools(store)[0];
  const r = (await cauta.invoke({
    intrebare: "domnii fanarioti reformatori",
    carte_sursa: "FISE DE ISTORIE ROMÂNEASCĂ",
  })) as unknown as {
    gasite: number;
    fragmente: { carte_sursa: string }[];
  };
  verifica(
    "T7b sursa normalizată găsește titlul fără diacritice",
    r.gasite > 0 && r.fragmente.every((f) => f.carte_sursa === "Fise de istorie romaneasca"),
    String(r.gasite),
  );
}

// ── T7c: filtrul biblic incomplet e ignorat, cu avertisment ────────────
//
// Capabilitatea rămâne opțională, dar corpusul principal de test este istoric.
{
  const store = await storeDeTest();
  const cauta: StructuredToolInterface = creeazaBibliotecaTools(store)[0];
  type R = { gasite: number; avertisment?: string };

  const doarCapitol = (await cauta.invoke({
    intrebare: "reformele domnilor fanarioti",
    capitol: 2,
  })) as unknown as R;
  verifica("T7c capitol fără carte → avertisment pentru model", Boolean(doarCapitol.avertisment));
  verifica(
    "T7c capitol fără carte → căutarea istorică rămâne disponibilă",
    doarCapitol.gasite > 0,
    String(doarCapitol.gasite),
  );
}

// ── T8: evaluarea, cu cazuri negative ──────────────────────────────────
{
  const store = await storeDeTest();
  const raport = await evalueazaRetrieval(store, CAZURI_TEST, { k: 3, prag: 0.6 });

  verifica("T8 metricile se calculează doar pe cazurile pozitive", raport.total === 4, String(raport.total));
  verifica("T8 hit rate rezonabil pe corpusul de test", raport.hitRate >= 0.75, `${Math.round(raport.hitRate * 100)}%`);
  verifica("T8 cazul negativ e raportat separat", raport.respingereCorecta !== null);
  const negativ = raport.cazuri.find((c) => c.negativ)!;
  verifica(
    "T8 întrebarea fără răspuns e respinsă corect (scor sub prag)",
    raport.respingereCorecta === 1 && negativ.scorTop < 0.6,
    negativ.scorTop.toFixed(2),
  );

  // Cu un prag imposibil de mic, orice scor trece → respingerea trebuie să cadă.
  // Dovada că metrica depinde de retriever, nu iese 100% din construcție.
  const faraPrag = await evalueazaRetrieval(store, CAZURI_TEST, { k: 3, prag: -1 });
  verifica("T8 respingerea chiar depinde de scor", faraPrag.respingereCorecta === 0, String(faraPrag.respingereCorecta));
  verifica("T8 MRR între 0 și 1", raport.mrr > 0 && raport.mrr <= 1, raport.mrr.toFixed(3));
}

// ── T9: memoria de lungă durată ────────────────────────────────────────
{
  const id = "cititor-test";
  const gol = await citesteProfil(id);
  verifica("T9 profil nou e gol", gol.sesiuni === 0 && gol.pasajeStudiate.length === 0);

  await incepeSesiune(id);
  await inregistreazaIntrebare(id, {
    intrebare: "Ce reforme au făcut domnii fanarioți?",
    fragmente: [
      { referinta: "Reformele lui Constantin Mavrocordat", carte_sursa: "Istoria moderna a romanilor" },
      { referinta: "Domnii fanarioti reformatori", carte_sursa: "Fise de istorie romaneasca" },
    ],
  });

  const dupa = await citesteProfil(id);
  verifica("T9 pasajele se rețin, cel mai recent primul", dupa.pasajeStudiate[0]?.referinta === "Domnii fanarioti reformatori", dupa.pasajeStudiate.map((p) => p.referinta).join(", "));
  verifica(
    "T9 sursele se numără",
    dupa.surseFavorite["Istoria moderna a romanilor"] === 1 &&
      dupa.surseFavorite["Fise de istorie romaneasca"] === 1,
    JSON.stringify(dupa.surseFavorite),
  );
  verifica("T9 limba e detectată", dupa.limba === "ro", String(dupa.limba));

  await incepeSesiune(id);
  const aDouaSesiune = await citesteProfil(id);
  verifica("T9 sesiunile se numără între apeluri", aDouaSesiune.sesiuni === 2, String(aDouaSesiune.sesiuni));

  const prompt = profilInPrompt(aDouaSesiune);
  verifica("T9 profilul devine text de prompt", prompt.includes("a 2-a sesiune") && prompt.includes("fanarioti"));
  verifica("T9 profil gol → prompt gol", profilInPrompt(gol) === "");

  // Izolarea între cititori — de asta e namespace ierarhic.
  const altul = await citesteProfil("alt-cititor");
  verifica(
    "T9 cititorii sunt izolați",
    altul.sesiuni === 0 &&
      altul.pasajeStudiate.length === 0 &&
      Object.keys(altul.surseFavorite).length === 0,
    JSON.stringify(altul.surseFavorite),
  );
}

console.log(esecuri === 0 ? "\nTOATE TESTELE AU TRECUT" : `\n${esecuri} TESTE PICATE`);
process.exit(esecuri === 0 ? 0 : 1);
