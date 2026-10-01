// lib/cost.ts — copiat din proiectul cursului (agentic-ai-lab7_8/lib/cost.ts)

/**
 * Cost calculator pentru LLM-uri (Lab 1 starter).
 *
 * Pricing actualizat IUNIE 2026. Verifică providerii pentru valorile curente:
 *   - OpenAI:    https://openai.com/api/pricing
 *   - Anthropic: https://www.anthropic.com/pricing#api
 *   - Google:    https://ai.google.dev/gemini-api/docs/pricing
 *
 * Notă pentru Gemini 2.5 Pro și Gemini 3.x Pro: pricing-ul are tier-uri
 * după lungimea prompt-ului. Valorile de mai jos sunt pentru prompt-uri
 * sub 200K tokens (cazul tipic). Peste 200K, prețul se dublează.
 */

const PRICING_PER_M_TOKENS: Record<string, { input: number; output: number }> =
  {
    // ============================================================
    // OpenAI — public + Azure deployments
    // ============================================================
    // Generația GPT-5.x (curent)
    "gpt-5": { input: 1.25, output: 10 },
    "gpt-5.2": { input: 1.75, output: 14 }, // actualizat: era greșit $1.25/$10
    "gpt-5-nano": { input: 0.05, output: 0.4 },
    "gpt-5.4": { input: 2.5, output: 15 }, // lansat 5 martie 2026
    "gpt-5.4-mini": { input: 0.25, output: 2 }, // lansat 17 martie 2026
    "gpt-5.4-nano": { input: 0.2, output: 1.25 }, // lansat 17 martie 2026
    "gpt-5.5": { input: 5, output: 30 }, // flagship curent

    // Generația GPT-4.x (încă disponibilă)
    "gpt-4o": { input: 2.5, output: 10 }, // legacy, încă funcțional
    "gpt-4o-mini": { input: 0.15, output: 0.6 }, // popular pentru dev
    "gpt-4.1": { input: 2.0, output: 8 },
    "gpt-4.1-mini": { input: 0.4, output: 1.6 },
    "gpt-4.1-nano": { input: 0.1, output: 0.4 },

    // Modele de reasoning (o-series)
    o3: { input: 2, output: 8 }, // post-launch cut
    "o4-mini": { input: 1.1, output: 4.4 },

    // ============================================================
    // Anthropic — verificat cu platform.claude.com/docs (iulie 2026)
    // Prețuri standard (global, fără cache/batch). Format API: cratimă, nu punct.
    // ============================================================
    // Curente / flagship
    "claude-fable-5": { input: 10.0, output: 50.0 }, // cel mai capabil model general
    "claude-mythos-5": { input: 10.0, output: 50.0 }, // limited availability (Project Glasswing)
    "claude-opus-4-8": { input: 5.0, output: 25.0 }, // flagship pt. coding agentic
    "claude-sonnet-5": { input: 2.0, output: 10.0 }, // introductory până 31 aug 2026; apoi $3/$15
    "claude-haiku-4-5": { input: 1.0, output: 5.0 }, // cel mai rapid, ieftin

    // Legacy (încă disponibile pe Claude API)
    "claude-opus-4-7": { input: 5.0, output: 25.0 },
    "claude-opus-4-6": { input: 5.0, output: 25.0 },
    "claude-opus-4-5": { input: 5.0, output: 25.0 },
    "claude-sonnet-4-6": { input: 3.0, output: 15.0 },
    "claude-sonnet-4-5": { input: 3.0, output: 15.0 },

    // Deprecated / retired (scumpe — evită pt. lab)
    "claude-opus-4-1": { input: 15.0, output: 75.0 }, // deprecated, retire 5 aug 2026
    "claude-opus-4": { input: 15.0, output: 75.0 }, // retired (except Google Cloud)
    "claude-sonnet-4": { input: 3.0, output: 15.0 }, // retired (except Bedrock/GCloud)
    "claude-haiku-3-5": { input: 0.8, output: 4.0 }, // retired (except Bedrock/GCloud)

    // ============================================================
    // Google Gemini
    // ============================================================
    // Generația 2.5 (la zi)
    "gemini-2.5-flash-lite": { input: 0.1, output: 0.4 }, // actualizat: era $0.075/$0.30
    "gemini-2.5-flash": { input: 0.3, output: 2.5 },
    "gemini-2.5-pro": { input: 1.25, output: 10.0 }, // actualizat output: era $5

    // Generația 3.x (mai nouă)
    "gemini-3-flash-lite": { input: 0.25, output: 1.5 },
    "gemini-3-flash": { input: 0.5, output: 3.0 },
    "gemini-3.1-pro": { input: 2.0, output: 12.0 }, // sub 200K context
    "gemini-3.1-flash-lite": { input: 0.25, output: 1.5 },
    "gemini-3.5-flash": { input: 1.5, output: 9.0 }, // lansat 19 mai 2026

    // ============================================================
    // xAI (Grok) — verificat cu docs.x.ai (iulie 2026)
    // Prețuri standard, prompt < 200k tokens (peste 200k, prețul se dublează).
    // API OpenAI-compatibil: baseURL https://api.x.ai/v1
    // ============================================================
    "grok-4.5": { input: 2.0, output: 6.0 }, // recomandat — cel mai capabil
    "grok-4.3": { input: 1.25, output: 2.5 },
    "grok-4.20-0309-reasoning": { input: 1.25, output: 2.5 },
    "grok-4.20-0309-non-reasoning": { input: 1.25, output: 2.5 },
    "grok-4.20-multi-agent-0309": { input: 1.25, output: 2.5 },
    "grok-build-0.1": { input: 1.0, output: 2.0 }, // cel mai ieftin
  };

export function calculateCost(
  model: string,
  inputTokens: number,
  outputTokens: number,
): number {
  const pricing = PRICING_PER_M_TOKENS[model.toLowerCase()];
  if (!pricing) {
    // Model necunoscut — întoarce 0 ca să nu crape UI-ul.
    // Adaugă model-ul în PRICING_PER_M_TOKENS dacă e nou.
    return 0;
  }
  return (
    (inputTokens / 1_000_000) * pricing.input +
    (outputTokens / 1_000_000) * pricing.output
  );
}

export function formatCost(cost: number): string {
  if (cost === 0) return "n/a";
  if (cost < 0.0001) {
    return `$${(cost * 1_000_000).toFixed(2)} µ`;
  }
  return `$${cost.toFixed(6)}`;
}

// ============================================================
// Text-to-Speech (xAI) — verificat cu docs.x.ai (iulie 2026)
// Se plătește pe CARACTERE (nu tokeni). Voice Pricing: $15 / 1M chars.
// ============================================================
const XAI_TTS_COST_PER_MILLION_CHARS = 15.0;

export function calculateTtsCost(chars: number): number {
  return (chars / 1_000_000) * XAI_TTS_COST_PER_MILLION_CHARS;
}
