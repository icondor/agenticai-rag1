# Lab 9 — RAG peste o bibliotecă

Pipeline RAG complet, în TypeScript, pe LangChain v1: extragere din PDF cu
detectarea structurii, chunking structural, embeddings comutabile, vector
store scris de la zero, RAG clasic ca lanț LCEL și RAG agentic cu retrieval
expus ca tool — comparate pe aceeași întrebare, în aceeași pagină.

---

## Instalare

```bash
pnpm install
cp .env.example .env.local     # și completează cheile
```

⚠️ **Ai nevoie de DOUĂ chei.** Anthropic nu are model de embeddings, deci
providerul care generează răspunsuri și cel care caută sunt diferiți.
Vezi `.env.example`.

## Prima rulare

```bash
pnpm verifica            # 60 de verificări, fără chei, fără costuri
pnpm indexeaza:demo      # indexează corpusul sintetic (câțiva cenți)
pnpm dev                 # → http://localhost:3000/biblioteca
```

## Cu cărțile tale

```bash
cp cartile-tale/*.pdf biblioteca/
pnpm indexeaza
pnpm evalueaza           # hit rate, MRR, comparație de configurații
```

⚠️ `biblioteca/` e în `.gitignore`. Materialele protejate nu se versionează.

---

## Comenzi

| Comandă | Ce face | Costă? |
|---|---|---|
| `pnpm verifica` | 60 de verificări cu embeddings și model fals | nu |
| `pnpm tipuri` | `tsc --noEmit` | nu |
| `pnpm indexeaza:demo` | indexează corpusul sintetic | câțiva cenți |
| `pnpm indexeaza` | indexează PDF-urile din `biblioteca/` | proporțional cu volumul |
| `pnpm evalueaza` | măsoară retrieverul pe indexul curent | foarte puțin |
| `pnpm dev` | pornește aplicația | la fiecare întrebare |

---

## Structura

```
lib/
  llm.ts                  modelul de chat — copiat din proiectul cursului
  cost.ts                 tabelul de prețuri — copiat
  biblioteca-agent.ts     RAG agentic, peste createAgent
  rag/
    extrage-pdf.ts        PDF → rânduri cu rol dedus din fonturi
    chunker.ts            chunking naiv ȘI structural, pentru comparație
    embeddings.ts         Google/OpenAI comutabil, taskType, cache pe disc
    vector-store.ts       store propriu: cosinus, filtre, MMR, persistență
    indexare.ts           pipeline offline complet
    rag-clasic.ts         lanț LCEL cu citări obligatorii
    evaluare.ts           hit rate, MRR, cazuri negative, comparații
    memorie-studiu.ts     Store — epilog, pregătire pentru Lab 10
    schemas-biblioteca.ts protocolul SSE
  tools/
    biblioteca-tools.ts   retrieval expus ca tool
app/
  api/biblioteca/route.ts o rută, două moduri
  biblioteca/page.tsx     interfața cu comparație
verificare/
  corpus-test.ts          două cărți sintetice, în formatul unei cărți reale
  test-lab9.ts            65 de verificări
scripturi/                indexare, indexare demo, evaluare
```

---

## Cele două pipeline-uri

Ideea pe care tutorialele o ascund: RAG are **două** pipeline-uri complet
separate.

```
INDEXARE (offline)     PDF → extragere → chunking → embeddings → vector store
                       rulează O DATĂ · durează minute · costă bani

INTEROGARE (online)    întrebare → embedding → căutare → context → model
                       rulează la FIECARE întrebare · milisecunde + un apel
```

Confuzia dintre ele e cauza numărul unu a aplicațiilor de RAG lente: cineva
reindexează la fiecare cerere fără să-și dea seama.

---

## Clasic vs. agentic

| | Clasic | Agentic |
|---|---|---|
| Cine cheamă retrieval-ul | noi, în cod | modelul, ca tool |
| Câte căutări | una | câte are nevoie |
| `k` | fix, ales de noi | modelul cere câte vrea |
| Filtrele | le punem noi | modelul le deduce din întrebare |
| Reformulare | nu | da, dacă scorurile sunt mici |
| Apeluri de model | 1 | 2-6 |
| Durată | ~2s | ~8-20s |

E aceeași distincție ca workflow vs. agent din Lab 7, aplicată la retrieval.
Agenticul **nu e mai bun** — e mai flexibil și mai scump.

---

## Capcane, toate verificate prin rulare

| Capcana | Simptomul | Fixul |
|---|---|---|
| Extragere naivă din PDF justificat | `thebasic`, `thechurch` | unire cu spațiu + reparare de cratimă |
| Nume de font hardcodate | merge pe o carte, cade pe următoarea | corpul se deduce ca fiind cel mai frecvent |
| Centrare judecată față de lățimea paginii | titlurile lungi clasificate greșit | reper: marginea corpului |
| `MemoryVectorStore` | e în `@langchain/classic`, pachetul legacy | 50 de linii scrise de noi |
| `new OpenAIEmbeddings()` fără parametri | primești `text-embedding-ada-002`, din 2022 | modelul scris explicit |
| Același `taskType` la indexare și căutare | retrieval mai slab, fără nicio eroare | `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` |
| Un singur cache la schimbarea providerului | vectori de la alt model, tăcut | cache separat per provider |
| Cazuri negative numărate ca eșecuri | metrici fără sens | raportate separat |
| `@langchain/langgraph` tranzitiv | `Cannot find package` cu pnpm | dependență explicită |
| esbuild ≥ 0.28 fără `./` | „entry point cannot be marked as external" | prefix `./` la cale |

---

## Ce NU e aici

- **pgvector** — un slide, fără cod. Vine în alt lab, odată cu persistența.
- **Memorie semantică, prompt dinamic din profil** — Lab 10, lângă checkpointere.
- **Reranking cu cross-encoder** — discutat pe slide, nu implementat.
