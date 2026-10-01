// lib/llm.ts — copiat din proiectul cursului (agentic-ai-lab7_8/lib/llm.ts)

/**
 * Provider abstraction pentru LangChain v1
 *
 * Lab 1-2 (starter):  azureopenai | gemini | openai | anthropic | xai  + reasoning
 * Lab 3/4 (EXTINS):   + provider LOCAL "ollama"
 *                     + createModelWithFallback()  (reziliență cloud → local)
 *                     + pickModel()                (smart routing simplu/greu)
 *
 * Toate adăugările Lab 3/4 sunt marcate cu [+ Lab 3/4]. Sunt ADITIVE — nimic din
 * ce funcționa în Lab 1-2 (chat, specializedchat, playground, reasoning) nu se schimbă.
 *
 *   - LLM_PROVIDER: "azureopenai" | "gemini" | "openai" | "anthropic" | "xai" | "ollama"
 *   - Azure: AZURE_OPENAI_* (5 variabile)
 *   - Gemini: GOOGLE_API_KEY + GEMINI_MODEL
 *   - OpenAI: OPENAI_API_KEY + OPENAI_MODEL
 *   - Anthropic: ANTHROPIC_API_KEY + ANTHROPIC_MODEL
 *   - xAI: XAI_API_KEY + XAI_MODEL (API OpenAI-compatibil)
 *   - Ollama (LOCAL) [+ Lab 3/4]: fără cheie — OLLAMA_MODEL (+ opțional OLLAMA_BASE_URL)
 *
 * Cursantul își sursează fișierul ~/.llm_* potrivit înainte de `pnpm dev`,
 * sau folosește .env.local pentru variabile.
 */

import { AzureChatOpenAI, ChatOpenAI } from "@langchain/openai"; //will use ChatOpenAI for xAI (OpenAI-compatible API) as well
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatOllama } from "@langchain/ollama"; // [+ Lab 3/4] provider local
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";

export type LlmProvider =
  | "azureopenai"
  | "gemini"
  | "openai"
  | "anthropic"
  | "xai"
  | "ollama"; // [+ Lab 3/4] provider local

// Max output tokens — destul pentru chat responses lungi.
// cu cat e mai mare, platim mai mult pe tokeni de output (cost calculator).
const MAX_OUTPUT_TOKENS = 4096;

// Când reasoning e activat, thinking-ul se scumpește pe tokeni de output.
// Ridicăm plafonul ca modelul să nu-și trunchieze gândirea (de exemplu sa semene cu claude.ai).
// budget-ul de thinking trebuie să fie strict sub maxTokens.
const REASONING_MAX_OUTPUT_TOKENS = 16000;
const REASONING_THINKING_BUDGET = 12000;

/**
 * Detectează deployments de reasoning models (gpt-5*, o-series).
 * Acestea cer `max_completion_tokens` în loc de `max_tokens`.
 */
function isReasoningDeployment(deploymentName: string): boolean {
  const name = deploymentName.toLowerCase();
  return name.startsWith("gpt-5") || /^o\d/.test(name);
}

/**
 * Citește providerul din environment. Aruncă eroare descriptivă dacă lipsește
 * sau e invalid.
 */
export function readProviderFromEnv(): LlmProvider {
  const raw = (process.env.LLM_PROVIDER ?? "").toLowerCase();
  if (
    raw === "azureopenai" ||
    raw === "gemini" ||
    raw === "openai" ||
    raw === "anthropic" ||
    raw === "xai" ||
    raw === "ollama" // [+ Lab 3/4]
  ) {
    return raw;
  }
  throw new Error(
    `LLM_PROVIDER lipsește sau invalid (primit "${process.env.LLM_PROVIDER ?? ""}"). ` +
      // [+ Lab 3/4] adăugat "ollama" în lista de valori acceptate
      `Valori acceptate: azureopenai | gemini | openai | anthropic | xai | ollama. ` +
      `Source fișierul ~/.llm_* potrivit sau setează în .env.local.`,
  );
}

/**
 * Întoarce numele modelului activ pentru afișare / cost calculator.
 */
export function getActiveModelName(provider: LlmProvider): string {
  switch (provider) {
    case "azureopenai":
      return process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME ?? "?"; //
    case "gemini":
      return process.env.GEMINI_MODEL ?? "gemini-2.5-flash"; // fallback
    case "openai":
      return process.env.OPENAI_MODEL ?? "gpt-4o-mini";
    case "anthropic":
      return process.env.ANTHROPIC_MODEL ?? "claude-haiku-4-5";
    case "xai":
      return process.env.XAI_MODEL ?? "grok-4.5";
    case "ollama": // [+ Lab 3/4]
      return process.env.OLLAMA_MODEL ?? "qwen3:8b";
  }
}

/**
 * Descriere scurtă a suportului de reasoning pentru provider (pentru UI).
 */
export function getReasoningCapabilities(provider: LlmProvider): string {
  switch (provider) {
    case "anthropic":
      return "Extended thinking (Claude 4.x) — reasoning afișat dacă modelul îl emite.";
    case "gemini":
      return "Thoughts (Gemini 2.5+) — reasoning afișat dacă modelul îl emite.";
    case "openai":
      return "Reasoning summaries doar pentru modele gpt-5 / o-series.";
    case "azureopenai":
      return "Reasoning summaries doar pentru deployment-uri de reasoning.";
    case "xai":
      return "Grok 4.x raționează implicit — reasoning afișat dacă e disponibil.";
    case "ollama": // [+ Lab 3/4]
      return "Modele locale (ex. Qwen3) pot raționa; disponibilitatea depinde de model.";
  }
}

/**
 * Opțiuni pentru createModel.
 * `reasoning` activează extended thinking / reasoning acolo unde providerul +
 * modelul suportă (Anthropic, Gemini, OpenAI/Azure reasoning models).
 */
export interface CreateModelOptions {
  reasoning?: boolean;
}

/**
 * Creează un model LangChain pe baza configurării din environment.
 * Aruncă eroare descriptivă dacă variabilele necesare lipsesc.
 */
export function createModel(
  provider: LlmProvider,
  options: CreateModelOptions = {},
): BaseChatModel {
  const reasoning = options.reasoning ?? false;

  //  bazat pe provider, citim variabilele necesare și construim modelul corespunzător.
  switch (provider) {
    case "azureopenai": {
      const apiKey = process.env.AZURE_OPENAI_API_KEY;
      const instanceName = process.env.AZURE_OPENAI_API_INSTANCE_NAME;
      const deploymentName = process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME;
      const apiVersion = process.env.AZURE_OPENAI_API_VERSION;
      const basePath = process.env.AZURE_OPENAI_BASE_PATH;

      if (!apiKey || !instanceName || !deploymentName || !apiVersion) {
        throw new Error(
          "Variabile Azure OpenAI lipsă. Necesare: AZURE_OPENAI_API_KEY, " +
            "AZURE_OPENAI_API_INSTANCE_NAME, AZURE_OPENAI_API_DEPLOYMENT_NAME, " +
            "AZURE_OPENAI_API_VERSION. Source fișierul ~/.llm_* potrivit.",
        );
      }

      const isReasoningModel = isReasoningDeployment(deploymentName);
      const tokenLimitKey = isReasoningModel
        ? "max_completion_tokens"
        : "max_tokens";

      // Reasoning summaries cer Responses API + un deployment de reasoning.
      const reasoningConfig =
        reasoning && isReasoningModel
          ? {
              useResponsesApi: true,
              reasoning: {
                effort: "medium" as const,
                summary: "auto" as const,
              },
            }
          : {};

      return new AzureChatOpenAI({
        azureOpenAIApiKey: apiKey,
        azureOpenAIApiInstanceName: instanceName,
        azureOpenAIApiDeploymentName: deploymentName,
        azureOpenAIApiVersion: apiVersion,
        ...(basePath ? { azureOpenAIBasePath: basePath } : {}),
        ...reasoningConfig,
        modelKwargs: { [tokenLimitKey]: MAX_OUTPUT_TOKENS },
      }) as unknown as BaseChatModel;
    }

    case "gemini": {
      const apiKey = process.env.GOOGLE_API_KEY;
      if (!apiKey) {
        throw new Error(
          "GOOGLE_API_KEY lipsă. Source ~/.llm_gemini sau setează în .env.local.",
        );
      }
      const modelName = process.env.GEMINI_MODEL ?? "gemini-2.5-flash";
      return new ChatGoogleGenerativeAI({
        apiKey,
        model: modelName,
        maxOutputTokens: reasoning
          ? REASONING_MAX_OUTPUT_TOKENS
          : MAX_OUTPUT_TOKENS,
        ...(reasoning
          ? {
              // thinkingBudget: -1 => dinamic, modelul decide cât gândește.
              thinkingConfig: { includeThoughts: true, thinkingBudget: -1 },
            }
          : {}),
      }) as unknown as BaseChatModel;
    }

    case "openai": {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error(
          "OPENAI_API_KEY lipsă. Setează-l în .env.local sau ~/.llm_openai.",
        );
      }
      const modelName = process.env.OPENAI_MODEL ?? "gpt-4o-mini";
      // Reasoning summaries doar pentru modele de reasoning (gpt-5 / o-series).
      const reasoningConfig =
        reasoning && isReasoningDeployment(modelName)
          ? {
              useResponsesApi: true,
              reasoning: {
                effort: "medium" as const,
                summary: "auto" as const,
              },
            }
          : {};
      return new ChatOpenAI({
        apiKey,
        model: modelName,
        ...reasoningConfig,
      }) as unknown as BaseChatModel;
    }

    case "anthropic": {
      const apiKey = process.env.ANTHROPIC_API_KEY;
      if (!apiKey) {
        throw new Error(
          "ANTHROPIC_API_KEY lipsă. Setează-l în .env.local sau ~/.llm_anthropic.",
        );
      }
      const modelName = process.env.ANTHROPIC_MODEL ?? "claude-haiku-4-5";
      return new ChatAnthropic({
        apiKey,
        model: modelName,
        maxTokens: reasoning ? REASONING_MAX_OUTPUT_TOKENS : MAX_OUTPUT_TOKENS,
        // Extended thinking: budget_tokens trebuie < maxTokens.
        ...(reasoning
          ? {
              thinking: {
                type: "enabled" as const,
                budget_tokens: REASONING_THINKING_BUDGET,
              },
            }
          : {}),
      }) as unknown as BaseChatModel;
    }

    case "xai": {
      const apiKey = process.env.XAI_API_KEY;
      if (!apiKey) {
        throw new Error(
          "XAI_API_KEY lipsă. Setează-l în .env.local sau ~/.llm_xai.",
        );
      }
      const modelName = process.env.XAI_MODEL ?? "grok-4.5";
      // we use the OpenAI-compatible API, so we can use ChatOpenAI class
      return new ChatOpenAI({
        apiKey,
        model: modelName,
        configuration: { baseURL: "https://api.x.ai/v1" },
      }) as unknown as BaseChatModel;
    }

    // [+ Lab 3/4] provider LOCAL — Ollama
    case "ollama": {
      // MODEL LOCAL — Ollama rulează un server pe http://localhost:11434.
      // Nu cere cheie API: trebuie doar `ollama serve` pornit și modelul pull-uit
      // (`ollama pull qwen3:8b`). De-aia e plasa ideală de fallback.
      // Notă: reasoning-ul nu e cablat aici (depinde de model); îl lăsăm implicit.
      const modelName = process.env.OLLAMA_MODEL ?? "qwen3:8b";
      return new ChatOllama({
        model: modelName,
        baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
      }) as unknown as BaseChatModel;
      // Alternativă (endpoint OpenAI-compatible — Ollama sau LM Studio):
      //   return new ChatOpenAI({
      //     model: modelName,
      //     configuration: { baseURL: "http://localhost:11434/v1", apiKey: "ollama" },
      //   }) as unknown as BaseChatModel;
    }
  }
}

// ===========================================================================
// [+ Lab 3/4] Adăugările peste starterul Lab 1-2
// ===========================================================================

/**
 * [+ Lab 3/4] FALLBACK: providerul tău cloud, cu modelul LOCAL ca plasă de siguranță.
 * Dacă primary pică (rate limit / outage), .withFallbacks() trece automat pe local.
 *
 * De ce așa, și nu openai→anthropic→ollama: createModel() validează cheia la
 * construcție, deci un lanț cu provideri pentru care n-ai chei ar arunca pe loc.
 * "Providerul tău + local" cere doar cheia ta + Ollama pornit — rulează la oricine.
 */
export function createModelWithFallback() {
  const primary = createModel(readProviderFromEnv());
  const local = createModel("ollama");
  return primary.withFallbacks([local]);
}

/**
 * [+ Lab 3/4] SMART ROUTING: cererile grele merg pe providerul tău cloud, cele simple pe local.
 * Euristică didactică (regex). În producție clasifici cu un model mic și rapid.
 */
export function pickModel(input: string): BaseChatModel {
  const greu =
    /analizează|explică de ce|compară|demonstrează|rezolvă|de ce/i.test(input);
  return greu ? createModel(readProviderFromEnv()) : createModel("ollama");
}
