
import type { RandExtras } from "@/lib/rag/extrage-pdf";

function rand(text: string, rol: RandExtras["rol"], pagina = 1): RandExtras {
  return { text, rol, pagina, x: 72, marime: rol === "corp" ? 14.4 : 20.2, font: "test" };
}

/**
 * Corpus sintetic bazat pe fapte prezentate în «O istorie sinceră a poporului
 * român». Formulările sunt scrise pentru curs, nu copiate din carte.
 *
 * Sursa A urmărește instituțiile și reformele epocii moderne.
 */
export const CARTE_A: RandExtras[] = [
  rand("1", "antet_subsol", 1),
  rand("ISTORIA MODERNA A ROMANILOR", "titlu_parte", 1),
  rand("EPOCA FANARIOTA", "titlu_capitol", 179),
  rand("Instaurarea regimului fanariot", "subsectiune", 179),
  rand(
    "Poarta Otomana a instaurat regimul fanariot pentru a-si consolida controlul asupra Moldovei si Tarii Romanesti, intr-un context in care Rusia devenea o mare putere regionala, iar Transilvania intrase sub stapanire habsburgica.",
    "corp",
    179,
  ),
  rand(
    "Domnii pamanteni puteau colabora cu adversarii Portii, pe cand fanariotii erau inalti dregatori otomani considerati mai fideli sultanului. Principatele si-au pastrat autonomia, dar aceasta a fost limitata considerabil.",
    "corp",
    179,
  ),
  rand("Reformele lui Constantin Mavrocordat", "subsectiune", 180),
  rand(
    "Nicolae si Constantin Mavrocordat, alaturi de Alexandru si Constantin Ipsilanti, s-au remarcat printr-un efort reformator amplu. Reforma fiscala a impus schimbari sociale, administrative si judiciare pentru stabilizarea contribuabililor.",
    "corp",
    180,
  ),
  rand(
    "Constantin Mavrocordat a numit ispravnici cu atributii administrative, fiscale si judiciare si a introdus salarizarea dregatorilor. Reformele fanariote au urmarit si modernizarea justitiei prin proceduri scrise si coduri de legi.",
    "corp",
    184,
  ),
  rand("UNIREA PRINCIPATELOR", "titlu_capitol", 229),
  rand("Dubla alegere a lui Alexandru Ioan Cuza", "subsectiune", 229),
  rand(
    "Alexandru Ioan Cuza a fost ales domn al Moldovei la 5 ianuarie 1859 si al Tarii Romanesti la 24 ianuarie 1859. Dubla alegere a aplicat politica faptului implinit si a pus temeliile statului roman modern.",
    "corp",
    229,
  ),
  rand(
    "Sprijinul lui Napoleon al III-lea a contribuit la depasirea opozitiei Turciei si Austriei, iar Conferinta de la Paris a recunoscut ulterior faptul implinit.",
    "corp",
    229,
  ),
  rand("Reformele lui Alexandru Ioan Cuza", "subsectiune", 230),
  rand(
    "In timpul lui Cuza au fost secularizate averile manastiresti, a fost reorganizat invatamantul si a fost realizata reforma agrara. Aceste masuri au consolidat institutiile noului stat si au accelerat modernizarea societatii.",
    "corp",
    231,
  ),
];

/** Sursa B oferă portrete și perspective complementare pentru comparație. */
export const CARTE_B: RandExtras[] = [
  rand("1", "antet_subsol", 1),
  rand("FISE DE ISTORIE ROMANEASCA", "titlu_parte", 1),
  rand("DOMNITORI SI MISCARI POLITICE", "titlu_capitol", 108),
  rand("Stefan cel Mare", "subsectiune", 108),
  rand(
    "Cronicarul Grigore Ureche il descrie pe Stefan cel Mare ca energic, priceput in razboi si capabil sa-si surprinda adversarii. Domnul participa direct la lupte si isi incuraja astfel oastea.",
    "corp",
    108,
  ),
  rand(
    "In politica externa, Stefan a cautat un echilibru intre Polonia, Ungaria si Imperiul Otoman. In 1486 a acceptat plata tributului pentru recunoasterea autonomiei Moldovei.",
    "corp",
    114,
  ),
  rand("Tudor Vladimirescu si miscarea din 1821", "subsectiune", 199),
  rand(
    "Tudor Vladimirescu provenea dintr-o familie de mosneni si capatase experienta militara ca voluntar in armata rusa in razboiul din 1806-1812. A activat apoi in negot si administratie.",
    "corp",
    199,
  ),
  rand(
    "Prestigiul sau in Oltenia si capacitatea de comanda au contribuit la alegerea lui drept conducator al miscarii din 1821. Adunarea norodului i-a oferit o baza populara si militara proprie.",
    "corp",
    201,
  ),
  rand("Domnii fanarioti reformatori", "subsectiune", 180),
  rand(
    "Dintre domnii fanarioti, Nicolae si Constantin Mavrocordat si Alexandru si Constantin Ipsilanti s-au remarcat prin reforme sistematice. Alti domni au urmarit mai ales imbogatirea personala, ceea ce a intrerupt continuitatea reformelor.",
    "corp",
    180,
  ),
  rand(
    "Reformele au modernizat fiscalitatea, administratia si justitia, dar efectele lor au fost limitate de cererile materiale arbitrare ale Portii si de rezistenta boierimii.",
    "corp",
    184,
  ),
];

export const CORPUS_TEST = [
  { titlu: "Istoria moderna a romanilor", randuri: CARTE_A },
  { titlu: "Fise de istorie romaneasca", randuri: CARTE_B },
];

// ─────────────────────────────────────────────────────────────────────────
// Setul de evaluare care merge cu corpusul
// ─────────────────────────────────────────────────────────────────────────
//
// 🎯 Include intenționat toate cele patru tipuri de întrebare pe care le
//    predăm: directă, detaliu precis, portret istoric și fără răspuns.

export const CAZURI_TEST = [
  {
    intrebare: "motivele instaurarii regimului fanariot de catre Poarta Otomana",
    eCorect: (m: Record<string, unknown>) =>
      m.subsectiune === "Instaurarea regimului fanariot",
    eticheta: "directă",
  },
  {
    intrebare: "masurile administrative si judiciare ale lui Constantin Mavrocordat",
    eCorect: (m: Record<string, unknown>) =>
      m.subsectiune === "Reformele lui Constantin Mavrocordat",
    eticheta: "detaliu precis",
  },
  {
    intrebare: "dubla alegere a lui Alexandru Ioan Cuza si statul roman modern",
    eCorect: (m: Record<string, unknown>) =>
      m.subsectiune === "Dubla alegere a lui Alexandru Ioan Cuza",
    eticheta: "directă",
  },
  {
    intrebare: "experienta militara si rolul lui Tudor Vladimirescu in miscarea din 1821",
    eCorect: (m: Record<string, unknown>) =>
      m.subsectiune === "Tudor Vladimirescu si miscarea din 1821",
    eticheta: "portret istoric",
  },
  {
    intrebare: "rolul shogunatului Tokugawa in Restauratia Meiji",
    eCorect: () => false,
    asteptatNegasit: true,
    eticheta: "fără răspuns în corpus",
  },
];
