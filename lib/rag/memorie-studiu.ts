// lib/rag/memorie-studiu.ts
//
// ═══════════════════════════════════════════════════════════════════════
// EPILOG — MEMORIE DE LUNGĂ DURATĂ
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 Distincția care se confundă cel mai des, și de care depinde tot ce
//    urmează în Lab 10:
//
//    SCURTĂ DURATĂ (short-term)          LUNGĂ DURATĂ (long-term)
//    ─────────────────────────────────   ──────────────────────────────────
//    mesajele conversației curente       fapte despre utilizator
//    trăiește într-un `thread_id`        trăiește într-un `namespace`
//    checkpointer (Lab 8)                Store (aici)
//    se pierde când începi alt fir       supraviețuiește oricâtor fire
//    «ce am vorbit acum 5 minute»        «ce știu despre omul ăsta»
//
//    Un agent cu checkpointer își amintește conversația. Un agent cu Store
//    își amintește PERSOANA. Sunt lucruri diferite, iar o aplicație serioasă
//    are nevoie de amândouă.
//
// ELEMENT NOU: `InMemoryStore` din `@langchain/langgraph`.
//
//   API-ul e un key-value pe namespace-uri:
//     put(namespace, cheie, valoare)
//     get(namespace, cheie)
//     search(namespace, { filter, limit })
//     delete(namespace, cheie)
//
// ⚠️ «InMemory» — la fel ca `MemorySaver` din Lab 8, trăiește în RAM-ul
//    procesului. Repornești serverul, se pierde. API-ul e IDENTIC cu al
//    unui Store persistent; în Lab 10 se schimbă o linie.
//
// 🌉 Aici e doar epilogul: ce reține aplicația despre un cititor între
//    sesiuni. Memoria semantică, pattern-urile de memorie și promptul
//    dinamic din profil sunt Lab 10, unde stau lângă checkpointere.

import { InMemoryStore } from "@langchain/langgraph";

// ─────────────────────────────────────────────────────────────────────────
// Ce reținem despre un cititor
// ─────────────────────────────────────────────────────────────────────────

export interface ProfilCititor {
  /** Ce pasaje a studiat, cel mai recent primul. */
  pasajeStudiate: { referinta: string; carte_sursa: string; cand: string }[];
  /** Ce cărți din bibliotecă a preferat, după câte întrebări a pus despre ele. */
  surseFavorite: Record<string, number>;
  /** Limba în care pune întrebări — folosită ca să răspundem la fel. */
  limba?: string;
  /** Câte sesiuni a avut. */
  sesiuni: number;
}

/**
 * Un profil gol, NOU la fiecare apel.
 *
 * ⚠️ Funcție, nu constantă. Cu o constantă și `{ ...GOL }`, copia e doar la
 *    primul nivel: `surseFavorite` rămâne ACELAȘI obiect pentru toți
 *    cititorii noi. Primul care scrie în el îl «umple» pentru toți ceilalți.
 */
function profilGol(): ProfilCititor {
  return { pasajeStudiate: [], surseFavorite: {}, sesiuni: 0 };
}

// ─────────────────────────────────────────────────────────────────────────
// Store-ul
// ─────────────────────────────────────────────────────────────────────────
//
// ⚠️ Pe globalThis, ca checkpointer-ul din Lab 8 și ca indexul din
//    `indexare.ts`. În dev, Fast Refresh golește starea de modul, iar un
//    Store care se golește la fiecare salvare de fișier nu demonstrează
//    nimic despre memoria «de lungă durată».

const globalStore = globalThis as typeof globalThis & {
  __storeCititori?: InMemoryStore;
};

export function storeCititori() {
  if (!globalStore.__storeCititori) {
    globalStore.__storeCititori = new InMemoryStore();
  }
  return globalStore.__storeCititori;
}

/**
 * Namespace-ul: o cale ierarhică, nu un simplu string.
 *
 * 🔍 De ce contează: `search(["cititori"])` întoarce toți cititorii, iar
 *    `search(["cititori", "ionel"])` doar pe unul. Ierarhia îți dă și
 *    izolare între utilizatori, și posibilitatea de a interoga peste toți —
 *    ceea ce un dicționar plat nu-ți dă.
 */
function namespaceCititor(idCititor: string) {
  return ["cititori", idCititor] as const;
}

// ─────────────────────────────────────────────────────────────────────────
// Citire și scriere
// ─────────────────────────────────────────────────────────────────────────

export async function citesteProfil(idCititor: string): Promise<ProfilCititor> {
  const store = storeCititori();
  const inregistrare = await store.get([...namespaceCititor(idCititor)], "profil");
  if (!inregistrare) return profilGol();

  // ⚠️ InMemoryStore întoarce OBIECTUL salvat, nu o copie. Copiem și
  //    obiectele din interior, ca modificările noastre să ajungă în store
  //    doar prin `scrieProfil`, explicit.
  const salvat = inregistrare.value as Partial<ProfilCititor>;
  return {
    ...profilGol(),
    ...salvat,
    pasajeStudiate: [...(salvat.pasajeStudiate ?? [])],
    surseFavorite: { ...(salvat.surseFavorite ?? {}) },
  };
}

export async function scrieProfil(idCititor: string, profil: ProfilCititor) {
  const store = storeCititori();
  await store.put([...namespaceCititor(idCititor)], "profil", profil);
}

/**
 * Actualizează profilul după o întrebare.
 *
 * 🎯 Observă ce NU facem: nu chemăm un model ca să «extragă preferințe».
 *    Faptele astea sunt deja în date — ce fragmente au fost folosite, în ce
 *    limbă era întrebarea. Un apel de model în plus ar costa bani ca să
 *    deducă ceva ce știm deja.
 *
 *    Memoria dedusă de model, pentru fapte care NU sunt în date, e Lab 10.
 */
export async function inregistreazaIntrebare(
  idCititor: string,
  date: {
    intrebare: string;
    fragmente: { referinta?: string | null; carte_sursa: string }[];
  },
): Promise<ProfilCititor> {
  const profil = await citesteProfil(idCititor);

  // Limba: euristică simplă pe diacritice și cuvinte frecvente.
  // Suficient pentru a răspunde în limba întrebării; nu e detecție serioasă.
  const pareRomana =
    /[ăâîșț]/i.test(date.intrebare) ||
    /\b(ce|care|cum|despre|spune|zice)\b/i.test(date.intrebare);
  profil.limba = pareRomana ? "ro" : "en";

  for (const f of date.fragmente) {
    profil.surseFavorite[f.carte_sursa] =
      (profil.surseFavorite[f.carte_sursa] ?? 0) + 1;

    if (f.referinta) {
      // Cel mai recent primul, fără duplicate, maximum 20.
      profil.pasajeStudiate = [
        { referinta: f.referinta, carte_sursa: f.carte_sursa, cand: new Date().toISOString() },
        ...profil.pasajeStudiate.filter((p) => p.referinta !== f.referinta),
      ].slice(0, 20);
    }
  }

  await scrieProfil(idCititor, profil);
  return profil;
}

export async function incepeSesiune(idCititor: string) {
  const profil = await citesteProfil(idCititor);
  profil.sesiuni += 1;
  await scrieProfil(idCititor, profil);
  return profil;
}

// ─────────────────────────────────────────────────────────────────────────
// Din profil în prompt
// ─────────────────────────────────────────────────────────────────────────

/**
 * Transformă profilul într-o bucată de prompt.
 *
 * 🎯 Ăsta e pasul care face memoria VIZIBILĂ. Un Store plin de date pe care
 *    nu le pui niciodată în prompt nu schimbă nimic în comportament.
 *
 * ⚠️ Cât de mult pui contează: profilul intră în context la FIECARE apel.
 *    Trei pasaje și două surse, nu douăzeci. Vezi Lab 8, costul contextului.
 */
export function profilInPrompt(profil: ProfilCititor): string {
  if (profil.sesiuni <= 1 && profil.pasajeStudiate.length === 0) return "";

  const linii: string[] = ["CE ȘTII DESPRE CITITORUL ĂSTA:"];

  if (profil.sesiuni > 1) {
    linii.push(`- E a ${profil.sesiuni}-a sesiune. Salută-l ca pe cineva revenit, scurt.`);
  }

  const recente = profil.pasajeStudiate.slice(0, 3);
  if (recente.length > 0) {
    linii.push(
      `- A studiat recent: ${recente.map((p) => p.referinta).join(", ")}. ` +
        `Dacă întrebarea nouă are legătură, poți face trimitere.`,
    );
  }

  const top = Object.entries(profil.surseFavorite)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2);
  if (top.length > 0) {
    linii.push(`- Consultă cel mai des: ${top.map(([s]) => s).join(", ")}.`);
  }

  if (profil.limba) {
    linii.push(`- Pune întrebări în ${profil.limba === "ro" ? "română" : "engleză"}.`);
  }

  return linii.join("\n");
}
