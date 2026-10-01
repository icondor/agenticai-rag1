// lib/rag/schemas-biblioteca.ts
//
// ═══════════════════════════════════════════════════════════════════════
// CONTRACTELE — cerere și protocol SSE
// ═══════════════════════════════════════════════════════════════════════
//
// Aceeași disciplină ca în Lab 7 și 8: o singură sursă de adevăr pentru
// forma datelor, folosită de rută, de interfață și de teste.
//
// Ce e specific aici: aceeași rută servește DOUĂ moduri — clasic și agentic
// — ca să le poți compara pe aceeași întrebare, în același ecran.

import { z } from "zod";

// ─────────────────────────────────────────────────────────────────────────
// Pragul de relevanță
// ─────────────────────────────────────────────────────────────────────────
//
// Sub el, un fragment e considerat «probabil irelevant»: agentul reformulează,
// evaluarea socotește întrebarea respinsă, interfața colorează roșu.
//
// ⚠️ Pragul depinde de MODELUL de embeddings. Scorurile cosinus nu au aceeași
//    scară de la un model la altul: la unele, fragmentele bune ies pe la 0.5.
//    0.6 e ales pentru gemini-embedding-001. Când schimbi modelul, rulează
//    `pnpm evalueaza` și uită-te la «scor mediu pe cazurile fără răspuns» —
//    pragul trebuie să stea deasupra lui.
//    E într-un singur loc tocmai ca să-l schimbi o singură dată.

export const PRAG_RELEVANTA = 0.6;

export const ModRag = z.enum(["clasic", "agentic"]);
export type ModRag = z.infer<typeof ModRag>;

export const CerereBiblioteca = z.object({
  intrebare: z.string().min(2),
  mod: ModRag.default("agentic"),
  /** Doar pentru modul clasic: câte fragmente aducem. */
  k: z.number().int().min(1).max(12).optional(),
  /** Doar pentru modul clasic: MMR în loc de similaritate simplă. */
  diversitate: z.boolean().optional(),
  /** Filtre puse din interfață, tot pentru modul clasic. */
  carte_sursa: z.string().optional(),
  carte_biblica: z.string().optional(),
});
export type CerereBiblioteca = z.infer<typeof CerereBiblioteca>;

// ─────────────────────────────────────────────────────────────────────────
// Un fragment, așa cum îl vede interfața
// ─────────────────────────────────────────────────────────────────────────

export const Fragment = z.object({
  id: z.number().int(),
  relevanta: z.number(),
  carte_sursa: z.string(),
  capitol: z.string().nullable(),
  subsectiune: z.string().nullable(),
  referinta: z.string().nullable(),
  pagina: z.number().nullable(),
  text: z.string(),
});
export type Fragment = z.infer<typeof Fragment>;

// ─────────────────────────────────────────────────────────────────────────
// Evenimente
// ─────────────────────────────────────────────────────────────────────────
//
// 🎯 Diferența față de Lab 8: aici evenimentele trebuie să arate PAȘII
//    RETRIEVAL-ULUI, nu doar pe cei ai agentului. Fragmentele găsite sunt
//    partea cea mai instructivă din tot demo-ul — sala trebuie să vadă CE
//    a găsit modelul înainte să scrie răspunsul.

export const EvenimentBiblioteca = z.discriminatedUnion("type", [
  z.object({ type: z.literal("start"), mod: ModRag, eticheta_embeddings: z.string() }),

  /** Modelul a lansat o căutare — în agentic, poate fi una din mai multe. */
  z.object({
    type: z.literal("cautare"),
    /** Leagă căutarea de fragmentele ei — în agentic pot fi mai multe odată. */
    id: z.string(),
    intrebare: z.string(),
    filtre: z.record(z.string(), z.unknown()),
  }),

  /** Ce a găsit căutarea. */
  z.object({
    type: z.literal("fragmente"),
    /** Id-ul căutării care a produs fragmentele. */
    id: z.string(),
    fragmente: z.array(Fragment),
    durata_ms: z.number().nonnegative(),
  }),

  z.object({ type: z.literal("gandeste"), mesaj: z.string() }),

  /** Bucată de răspuns, pentru afișare progresivă. */
  z.object({ type: z.literal("raspuns"), text: z.string() }),

  z.object({
    type: z.literal("done"),
    durata_ms: z.number().nonnegative(),
    nr_cautari: z.number().int().nonnegative(),
    nr_fragmente: z.number().int().nonnegative(),
    tokeni_input: z.number().int().nonnegative(),
    tokeni_output: z.number().int().nonnegative(),
    cost: z.string(),
    model: z.string(),
  }),

  z.object({ type: z.literal("eroare"), mesaj: z.string() }),
]);
export type EvenimentBiblioteca = z.infer<typeof EvenimentBiblioteca>;

// ─────────────────────────────────────────────────────────────────────────
// Întrebări demo
// ─────────────────────────────────────────────────────────────────────────
//
// 🎯 Alese ca să arate lucruri diferite:
//   1. simplă, o căutare ajunge → clasicul se descurcă la fel de bine
//   2. sinteză din mai multe fragmente despre același subiect
//   3. comparativă → agenticul face DOUĂ căutări, clasicul una singură
//   4. un detaliu precis, dintr-un singur fragment
//   5. fără răspuns în corpus → testul de onestitate

export const INTREBARI_DEMO = [
  "De ce a instaurat Poarta Otomană regimul fanariot în Moldova și Țara Românească?",
  "Enumeră câțiva domni fanarioți și explică pe scurt prin ce s-au remarcat.",
  "Compară reformele domnilor fanarioți cu reformele lui Alexandru Ioan Cuza.",
  "Cum a contribuit dubla alegere a lui Alexandru Ioan Cuza la formarea statului român modern?",
  "Ce rol a avut shogunatul Tokugawa în Restaurația Meiji?",
];
