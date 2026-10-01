// app/api/biblioteca/route.ts
//
// ═══════════════════════════════════════════════════════════════════════
// O RUTĂ, DOUĂ MODURI — ca să le poți compara pe aceeași întrebare
// ═══════════════════════════════════════════════════════════════════════
//
// 🎯 De ce ambele în același loc: demo-ul central al labului e «aceeași
//    întrebare, două arhitecturi». Dacă ar fi două rute separate, ai avea
//    două pagini și comparația s-ar pierde.
//
//    CLASIC  → un RunnableSequence. Noi chemăm retrieval-ul, o dată.
//    AGENTIC → createAgent. Modelul cheamă retrieval-ul, de câte ori vrea.

import { NextRequest } from "next/server";
import { AIMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import { calculateCost, formatCost } from "@/lib/cost";
import { getActiveModelName, readProviderFromEnv } from "@/lib/llm";
import { incarcaIndex } from "@/lib/rag/indexare";
import { etichetaEmbeddings } from "@/lib/rag/embeddings";
import { ragClasic } from "@/lib/rag/rag-clasic";
import { creeazaBibliotecaAgent, LIMITA_RECURSIE } from "@/lib/biblioteca-agent";
import { normalizeaza } from "@/lib/tools/biblioteca-tools";
import {
  CerereBiblioteca,
  type EvenimentBiblioteca,
  type Fragment,
} from "@/lib/rag/schemas-biblioteca";
import type { DocumentInterface } from "@langchain/core/documents";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  const json: unknown = await request.json().catch(() => null);
  const corp = CerereBiblioteca.safeParse(json);
  if (!corp.success) {
    return Response.json({ error: "Cerere invalidă" }, { status: 400 });
  }
  const cerere = corp.data;
  const pornitLa = Date.now();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const encoder = new TextEncoder();

      // Defensiv, ca în Lab 8: dacă utilizatorul închide tabul, scrierea
      // aruncă, iar dacă asta se întâmplă din catch, eroarea scapă.
      const emite = (ev: EvenimentBiblioteca) => {
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(ev)}\n\n`));
        } catch {
          // clientul a plecat
        }
      };

      let nrCautari = 0;
      let nrFragmente = 0;
      let tokeniInput = 0;
      let tokeniOutput = 0;

      try {
        const provider = readProviderFromEnv();
        emite({
          type: "start",
          mod: cerere.mod,
          eticheta_embeddings: etichetaEmbeddings(),
        });

        // Indexul e încărcat o singură dată și ținut pe globalThis.
        const store = await incarcaIndex();

        // ═══════════════════════════════════════════════════════════
        // MODUL CLASIC — lanț, pași ficși
        // ═══════════════════════════════════════════════════════════
        if (cerere.mod === "clasic") {
          const filtru =
            cerere.carte_sursa || cerere.carte_biblica
              ? (doc: DocumentInterface) => {
                  const m = doc.metadata as Record<string, unknown>;
                  if (
                    cerere.carte_sursa &&
                    !normalizeaza(m.carte_sursa).includes(normalizeaza(cerere.carte_sursa))
                  ) {
                    return false;
                  }
                  if (
                    cerere.carte_biblica &&
                    normalizeaza(m.carte_biblica) !== normalizeaza(cerere.carte_biblica)
                  ) {
                    return false;
                  }
                  return true;
                }
              : undefined;

          emite({
            type: "cautare",
            id: "clasic",
            intrebare: cerere.intrebare,
            filtre: {
              carte_sursa: cerere.carte_sursa ?? null,
              carte_biblica: cerere.carte_biblica ?? null,
              k: cerere.k ?? 5,
              diversitate: cerere.diversitate ?? false,
            },
          });

          const rezultat = await ragClasic(store, cerere.intrebare, {
            k: cerere.k ?? 5,
            filtru,
            cuDiversitate: cerere.diversitate,
            signal: request.signal,
          });
          nrCautari = 1;
          nrFragmente = rezultat.fragmente.length;
          tokeniInput = rezultat.tokeniInput;
          tokeniOutput = rezultat.tokeniOutput;

          emite({
            type: "fragmente",
            id: "clasic",
            fragmente: rezultat.fragmente.map((f, i) => ({
              id: i + 1,
              relevanta: Math.round(f.scor * 100) / 100,
              carte_sursa: String(f.metadata.carte_sursa ?? "necunoscută"),
              capitol: f.metadata.capitol ? String(f.metadata.capitol) : null,
              subsectiune: f.metadata.subsectiune
                ? String(f.metadata.subsectiune)
                : null,
              referinta: f.metadata.ancora ? String(f.metadata.ancora) : null,
              pagina:
                typeof f.metadata.pagina === "number" ? f.metadata.pagina : null,
              text: f.text,
            })),
            // Doar retrieval-ul, ca în agentic — altfel comparația de
            // timp ar include și generarea răspunsului.
            durata_ms: rezultat.durataCautareMs,
          });

          emite({ type: "raspuns", text: rezultat.raspuns });
        }

        // ═══════════════════════════════════════════════════════════
        // MODUL AGENTIC — bucla e a modelului
        // ═══════════════════════════════════════════════════════════
        else {
          const agent = creeazaBibliotecaAgent(store);

          const pasi = await agent.stream(
            { messages: [{ role: "user", content: cerere.intrebare }] },
            {
              streamMode: "updates",
              recursionLimit: LIMITA_RECURSIE,
              // Tab închis → agentul se oprește, nu mai plătim apeluri.
              signal: request.signal,
            },
          );

          // Gărzile din Lab 8: stream-ul livrează același mesaj de mai
          // multe ori. Fără ele, fiecare căutare apare de câteva ori.
          const idMesajeVazute = new Set<string>();
          const cereriVazute = new Set<string>();
          const inCurs = new Map<string, number>();
          // Căutările lansate, ca să le putem închide pe toate — și pe cele
          // blocate de toolCallLimitMiddleware, care nu întorc JSON.
          const idCautari = new Set<string>();

          for await (const pas of pasi as AsyncIterable<
            Record<string, { messages?: unknown[] }>
          >) {
            for (const continut of Object.values(pas)) {
              for (const brut of continut?.messages ?? []) {
                if (!BaseMessage.isInstance(brut)) continue;

                const idMesaj = brut.id ?? `${idMesajeVazute.size}-${brut.getType()}`;
                const mesajNou = !idMesajeVazute.has(idMesaj);
                if (mesajNou) idMesajeVazute.add(idMesaj);

                if (AIMessage.isInstance(brut)) {
                  if (mesajNou) {
                    tokeniInput += brut.usage_metadata?.input_tokens ?? 0;
                    tokeniOutput += brut.usage_metadata?.output_tokens ?? 0;
                  }

                  for (const cerereTool of brut.tool_calls ?? []) {
                    if (!cerereTool.id || cereriVazute.has(cerereTool.id)) continue;
                    cereriVazute.add(cerereTool.id);
                    inCurs.set(cerereTool.id, Date.now());

                    if (cerereTool.name === "cautaInBiblioteca") {
                      nrCautari += 1;
                      idCautari.add(cerereTool.id);
                      const a = cerereTool.args as Record<string, unknown>;
                      emite({
                        type: "cautare",
                        id: cerereTool.id,
                        intrebare: String(a.intrebare ?? ""),
                        filtre: {
                          carte_biblica: a.carte_biblica ?? null,
                          capitol: a.capitol ?? null,
                          carte_sursa: a.carte_sursa ?? null,
                        },
                      });
                    } else {
                      emite({ type: "gandeste", mesaj: `Verifică inventarul bibliotecii…` });
                    }
                  }

                  const faraTools = (brut.tool_calls ?? []).length === 0;
                  // `.text` și nu `.content`: unii provideri întorc conținutul
                  // ca listă de blocuri, nu ca string.
                  const text = brut.text.trim();
                  if (mesajNou && faraTools && text.length > 0) {
                    emite({ type: "raspuns", text });
                  }
                }

                if (
                  ToolMessage.isInstance(brut) &&
                  mesajNou &&
                  idCautari.has(brut.tool_call_id)
                ) {
                  const pornitCautare = inCurs.get(brut.tool_call_id) ?? Date.now();
                  // O căutare blocată de limită întoarce un mesaj de eroare,
                  // nu JSON. Trimitem totuși evenimentul, cu zero fragmente,
                  // ca interfața să nu rămână cu o căutare «în curs».
                  let fragmente: Fragment[] = [];
                  try {
                    const continutTool = JSON.parse(brut.text) as {
                      fragmente?: Fragment[];
                    };
                    fragmente = continutTool.fragmente ?? [];
                  } catch {
                    // căutare blocată sau eșuată
                  }
                  nrFragmente += fragmente.length;
                  emite({
                    type: "fragmente",
                    id: brut.tool_call_id,
                    fragmente,
                    durata_ms: Date.now() - pornitCautare,
                  });
                }
              }
            }
          }
        }

        const model = getActiveModelName(provider);
        emite({
          type: "done",
          durata_ms: Date.now() - pornitLa,
          nr_cautari: nrCautari,
          nr_fragmente: nrFragmente,
          tokeni_input: tokeniInput,
          tokeni_output: tokeniOutput,
          cost: formatCost(calculateCost(model, tokeniInput, tokeniOutput)),
          model,
        });

        console.log(
          `[BIBLIOTECA ${cerere.mod}] ${((Date.now() - pornitLa) / 1000).toFixed(1)}s · ` +
            `${nrCautari} căutări · ${nrFragmente} fragmente · ${tokeniInput} in / ${tokeniOutput} out`,
        );
      } catch (eroare) {
        console.error("[BIBLIOTECA]", eroare);
        emite({
          type: "eroare",
          mesaj:
            eroare instanceof Error
              ? eroare.message
              : "Eroare la interogarea bibliotecii.",
        });
      } finally {
        try {
          controller.close();
        } catch {
          // deja închis
        }
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
