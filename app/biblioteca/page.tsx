"use client";

// app/biblioteca/page.tsx
//
// Patru zone:
//   1. întrebarea + comutatorul clasic/agentic + parametrii
//   2. căutările lansate — în agentic pot fi mai multe
//   3. fragmentele găsite, cu scor și sursă  ← partea cea mai instructivă
//   4. răspunsul, cu citări
//
// 🎯 De ce fragmentele sunt afișate integral și nu ascunse: în RAG, cel mai
//    important lucru pe care îl poate învăța cineva e să compare CE a găsit
//    retrieval-ul cu CE a scris modelul. Dacă răspunsul conține ceva ce nu e
//    în fragmente, ai o halucinație — și se vede pe loc.

import { useEffect, useRef, useState } from "react";
import {
  EvenimentBiblioteca,
  INTREBARI_DEMO,
  PRAG_RELEVANTA,
  type Fragment,
  type ModRag,
} from "@/lib/rag/schemas-biblioteca";

interface Cautare {
  /** Numărul afișat: Căutarea 1, 2, 3… */
  id: number;
  /** Id-ul de pe server, care leagă căutarea de fragmentele ei. */
  cheie: string;
  intrebare: string;
  filtre: Record<string, unknown>;
  fragmente?: Fragment[];
  durataMs?: number;
}

interface Sumar {
  durataMs: number;
  nrCautari: number;
  nrFragmente: number;
  tokeni: string;
  cost: string;
  model: string;
}

export default function BibliotecaPage() {
  const [intrebare, setIntrebare] = useState(INTREBARI_DEMO[1]);
  const [mod, setMod] = useState<ModRag>("agentic");
  const [k, setK] = useState(5);
  const [diversitate, setDiversitate] = useState(false);

  const [cautari, setCautari] = useState<Cautare[]>([]);
  const [raspuns, setRaspuns] = useState("");
  const [ganduri, setGanduri] = useState<string[]>([]);
  const [sumar, setSumar] = useState<Sumar | null>(null);
  const [eticheta, setEticheta] = useState("");
  const [eroare, setEroare] = useState<string | null>(null);
  const [ocupat, setOcupat] = useState(false);

  const contor = useRef(0);
  const abandon = useRef<AbortController | null>(null);
  useEffect(() => () => abandon.current?.abort(), []);

  async function intreaba(modFolosit: ModRag = mod) {
    if (!intrebare.trim() || ocupat) return;
    setMod(modFolosit);
    setCautari([]);
    setRaspuns("");
    setGanduri([]);
    setSumar(null);
    setEroare(null);
    contor.current = 0;
    setOcupat(true);
    abandon.current = new AbortController();

    try {
      const res = await fetch("/api/biblioteca", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abandon.current.signal,
        body: JSON.stringify({
          intrebare,
          mod: modFolosit,
          k,
          diversitate,
        }),
      });
      if (!res.body) throw new Error("Fără stream");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let rest = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        rest += decoder.decode(value, { stream: true });

        const bucati = rest.split("\n\n");
        rest = bucati.pop() ?? "";

        for (const bucata of bucati) {
          const linie = bucata.split("\n").find((l) => l.startsWith("data:"));
          if (!linie) continue;
          const parsat = EvenimentBiblioteca.safeParse(
            JSON.parse(linie.slice(5).trim()),
          );
          if (parsat.success) aplica(parsat.data);
        }
      }
    } catch (err) {
      const abandonat = err instanceof Error && err.name === "AbortError";
      if (!abandonat) setEroare(err instanceof Error ? err.message : "Eroare");
    } finally {
      setOcupat(false);
    }
  }

  function aplica(ev: EvenimentBiblioteca) {
    if (ev.type === "start") setEticheta(ev.eticheta_embeddings);
    if (ev.type === "cautare") {
      contor.current += 1;
      setCautari((c) => [
        ...c,
        { id: contor.current, cheie: ev.id, intrebare: ev.intrebare, filtre: ev.filtre },
      ]);
    }
    if (ev.type === "fragmente") {
      // Atașăm fragmentele căutării care le-a produs, după id.
      setCautari((c) => {
        const i = c.findIndex((x) => x.cheie === ev.id);
        if (i === -1) return c;
        const copie = [...c];
        copie[i] = { ...copie[i], fragmente: ev.fragmente, durataMs: ev.durata_ms };
        return copie;
      });
    }
    if (ev.type === "gandeste") setGanduri((g) => [...g, ev.mesaj]);
    if (ev.type === "raspuns") setRaspuns((r) => r + (r ? "\n\n" : "") + ev.text);
    if (ev.type === "done") {
      setSumar({
        durataMs: ev.durata_ms,
        nrCautari: ev.nr_cautari,
        nrFragmente: ev.nr_fragmente,
        tokeni: `${ev.tokeni_input} in / ${ev.tokeni_output} out`,
        cost: ev.cost,
        model: ev.model,
      });
    }
    if (ev.type === "eroare") setEroare(ev.mesaj);
  }

  return (
    <main className="mx-auto w-full max-w-4xl p-4">
      <h1 className="mb-1 text-2xl font-bold">Bibliotecă de istorie românească</h1>
      <p className="mb-4 text-sm text-gray-500">
        Aceeași întrebare, două arhitecturi. Clasic: o căutare, k fix, decis de
        noi. Agentic: modelul decide dacă, ce și de câte ori caută.
      </p>

      {/* ── 1 · ÎNTREBAREA ────────────────────────────────────────── */}
      <div className="mb-3 flex flex-wrap gap-2">
        {INTREBARI_DEMO.map((q, i) => (
          <button
            key={i}
            onClick={() => setIntrebare(q)}
            className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-600 hover:bg-gray-200"
          >
            {i + 1}
          </button>
        ))}
        <span className="self-center text-xs text-gray-400">
          exemple: 1 simplă · 2 sinteză · 3 comparativă · 4 detaliu ·
          5 fără răspuns în corpus
        </span>
      </div>

      <textarea
        value={intrebare}
        onChange={(e) => setIntrebare(e.target.value)}
        rows={2}
        className="mb-3 w-full rounded-lg border p-3 text-sm"
      />

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg border p-0.5">
          {(["clasic", "agentic"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMod(m)}
              className={`rounded px-3 py-1 text-sm ${
                mod === m ? "bg-blue-600 text-white" : "text-gray-600"
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        {mod === "clasic" && (
          <>
            <label className="flex items-center gap-1 text-xs text-gray-600">
              k =
              <input
                type="number"
                min={1}
                max={12}
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-14 rounded border px-1 py-0.5"
              />
            </label>
            <label className="flex items-center gap-1 text-xs text-gray-600">
              <input
                type="checkbox"
                checked={diversitate}
                onChange={(e) => setDiversitate(e.target.checked)}
              />
              MMR (diversitate)
            </label>
          </>
        )}

        <button
          onClick={() => intreaba()}
          disabled={ocupat || !intrebare.trim()}
          className="ml-auto rounded-lg bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
        >
          {ocupat ? "Caută…" : "Întreabă"}
        </button>
        <button
          onClick={() => intreaba(mod === "clasic" ? "agentic" : "clasic")}
          disabled={ocupat}
          className="rounded-lg border px-3 py-2 text-sm text-gray-600 disabled:opacity-50"
        >
          Rulează în celălalt mod
        </button>
      </div>

      {eticheta && (
        <p className="mb-3 text-xs text-gray-400">embeddings: {eticheta}</p>
      )}

      {eroare && (
        <div className="mb-4 rounded-lg bg-red-100 p-3 text-sm text-red-800">
          {eroare}
        </div>
      )}

      {/* ── 2-3 · CĂUTĂRI ȘI FRAGMENTE ────────────────────────────── */}
      {cautari.map((c) => (
        <section key={c.id} className="mb-4 rounded-lg border">
          <div className="flex flex-wrap items-center gap-2 border-b bg-gray-50 px-3 py-2 text-xs">
            <span className="font-semibold text-teal-700">
              Căutarea {c.id}
            </span>
            <span className="font-mono text-gray-700">«{c.intrebare}»</span>
            {Object.entries(c.filtre)
              .filter(([, v]) => v !== null && v !== undefined && v !== false)
              .map(([k2, v]) => (
                <span key={k2} className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-800">
                  {k2}: {String(v)}
                </span>
              ))}
            {c.durataMs !== undefined && (
              <span className="ml-auto text-gray-400">{c.durataMs}ms</span>
            )}
          </div>

          <div className="divide-y">
            {(c.fragmente ?? []).map((f) => (
              <div key={f.id} className="p-3 text-xs">
                <div className="mb-1 flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded px-1.5 py-0.5 font-mono ${
                      f.relevanta >= PRAG_RELEVANTA + 0.15
                        ? "bg-green-100 text-green-800"
                        : f.relevanta >= PRAG_RELEVANTA
                          ? "bg-amber-100 text-amber-800"
                          : "bg-red-100 text-red-800"
                    }`}
                  >
                    [{f.id}] {f.relevanta}
                  </span>
                  <span className="font-semibold">{f.carte_sursa}</span>
                  {f.referinta && (
                    <span className="text-teal-700">{f.referinta}</span>
                  )}
                  {f.pagina && <span className="text-gray-400">p. {f.pagina}</span>}
                </div>
                {f.subsectiune && (
                  <div className="mb-1 italic text-gray-500">{f.subsectiune}</div>
                )}
                <p className="whitespace-pre-wrap text-gray-700">{f.text}</p>
              </div>
            ))}
            {!c.fragmente && (
              <div className="p-3 text-xs text-gray-400">se caută…</div>
            )}
          </div>
        </section>
      ))}

      {ganduri.map((g, i) => (
        <p key={i} className="mb-2 text-xs italic text-gray-400">
          💭 {g}
        </p>
      ))}

      {/* ── 4 · RĂSPUNSUL ─────────────────────────────────────────── */}
      {raspuns && (
        <section className="rounded-lg border bg-blue-50 p-4">
          <div className="mb-2 text-xs font-medium uppercase text-gray-500">
            Răspuns
            {sumar && (
              <span className="ml-2 font-normal normal-case text-gray-400">
                {(sumar.durataMs / 1000).toFixed(1)}s · {sumar.nrCautari} căutări ·{" "}
                {sumar.nrFragmente} fragmente · {sumar.tokeni} · {sumar.cost} ·{" "}
                {sumar.model}
              </span>
            )}
          </div>
          <p className="whitespace-pre-wrap text-sm text-gray-800">{raspuns}</p>
          <p className="mt-3 border-t pt-2 text-xs text-gray-400">
            Verifică citările [n] în fragmentele de mai sus. Orice afirmație care
            nu se regăsește acolo e o halucinație.
          </p>
        </section>
      )}

      <aside className="mt-8 rounded-lg border bg-gray-50 p-3 text-[11px] leading-relaxed text-gray-500">
        <p className="mb-1 font-medium text-gray-600">Notițe despre retrieval</p>
        <ul className="list-disc space-y-0.5 pl-4">
          <li>
            <strong>k</strong> este numărul maxim de fragmente candidate; după
            eliminarea celor cu relevanță mică pot rămâne mai puține.
          </li>
          <li>
            <strong>MMR oprit:</strong> sunt alese fragmentele cele mai similare
            semantic cu întrebarea.
          </li>
          <li>
            <strong>MMR pornit:</strong> se echilibrează relevanța cu diversitatea,
            pentru a reduce fragmentele aproape duplicate.
          </li>
          <li>
            Orientativ: pentru o întrebare precisă folosește k 3–5, de regulă fără
            MMR; pentru sinteză sau comparație, k 7–10 și MMR pot ajuta.
          </li>
          <li>Aceste setări se aplică numai modului clasic.</li>
        </ul>
      </aside>
    </main>
  );
}
