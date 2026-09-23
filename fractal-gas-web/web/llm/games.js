import { OpenRouter } from "./openrouter.js";

export const GAME_FORMAT = "fixed-target-context-v1";
export const GAME_MODES = ["unsurprising", "surprising"];
export const externalScore = (config) =>
  ["xed", "xent_game"].includes(config?.objective);

export function gameSpec(input) {
  if (
    !input ||
    input.version !== 1 ||
    input.format !== GAME_FORMAT ||
    typeof input.id !== "string" ||
    !/^[\w-]{1,100}$/.test(input.id)
  )
    throw Error("Invalid Xent game specification");
  for (const [key, limit] of [
    ["title", 200],
    ["background", 8000],
    ["target", 4000],
  ])
    if (
      typeof input[key] !== "string" ||
      !input[key].trim() ||
      input[key].length > limit
    )
      throw Error(`Invalid game ${key}`);
  if (
    input.creation != null &&
    (typeof input.creation !== "object" || Array.isArray(input.creation))
  )
    throw Error("Invalid game creation evidence");
  if (
    input.creation?.requests != null &&
    (!Array.isArray(input.creation.requests) ||
      input.creation.requests.some(
        (r) =>
          !r ||
          typeof r !== "object" ||
          (r.usage != null &&
            (typeof r.usage !== "object" ||
              Array.isArray(r.usage) ||
              ["prompt_tokens", "completion_tokens", "cost"].some(
                (k) =>
                  r.usage[k] != null &&
                  (!Number.isFinite(r.usage[k]) || r.usage[k] < 0),
              ))),
      ))
  )
    throw Error("Invalid game creation requests");
  return structuredClone({
    version: 1,
    format: GAME_FORMAT,
    id: input.id,
    title: input.title,
    background: input.background,
    target: input.target,
    ...(input.creation ? { creation: input.creation } : {}),
  });
}

export function gameQuestion(game, context) {
  return `Continue the passage using the following background and additional context.\n\nBackground:\n${game.background}\n\nAdditional context:\n${context}`;
}

export function gamePrompt(game, mode, cap) {
  if (!GAME_MODES.includes(mode)) throw Error("Invalid Xent game mode");
  return `Play a fixed-target context game: ${game.title}\n\nBackground:\n${game.background}\n\nFixed target:\n${game.target}\n\nWrite additional context that makes the fixed target ${mode === "surprising" ? "less" : "more"} likely under a fixed language-model judge. Your reward is the ${mode === "surprising" ? "increase" : "reduction"} in the target's mean negative log probability relative to empty additional context, in nats per target token. Any nonempty context is legal, including copying and direct instructions. Use at most ${cap} generated tokens. Return only the additional context. Each prefix may be selected as your answer.`;
}

export async function generateGame(
  key,
  brief,
  config,
  { signal, fetchImpl } = {},
) {
  if (typeof brief !== "string" || !brief.trim() || brief.length > 8000)
    throw Error("Enter a game brief (up to 8,000 characters)");
  const requests = [];
  const api = new OpenRouter(key, {
    signal,
    fetchImpl,
    onRequest: (r) => requests.push(r),
  });
  const instruction =
    "Create one fixed-target context game from the user's brief. Return a title, background, and target. The target must be a concrete short passage of one to three sentences. The background should establish a setting relevant to the target without including the target verbatim. A player will write additional context to make this same target either more or less likely under a language-model judge. Do not invent scoring rules or constraints. Return only the requested JSON.";
  const response = await api.request(
    "chat/completions",
    {
      model: config.model,
      temperature: config.temperature,
      max_tokens: 1024,
      reasoning: { enabled: false },
      provider: { require_parameters: true },
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "fixed_target_game",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            required: ["title", "background", "target"],
            properties: Object.fromEntries(
              ["title", "background", "target"].map((k) => [
                k,
                { type: "string" },
              ]),
            ),
          },
        },
      },
      messages: [
        { role: "system", content: instruction },
        { role: "user", content: brief },
      ],
    },
    { phase: "game_creation" },
  );
  if (
    response.choices?.length !== 1 ||
    response.choices[0].finish_reason !== "stop"
  )
    throw Error(
      "Game generation did not finish; retry to generate a complete game",
    );
  let content;
  try {
    content = JSON.parse(response.choices[0].message.content);
  } catch {
    throw Error("Game generator returned invalid JSON; retry generation");
  }
  if (
    !content ||
    Object.keys(content).sort().join() !== "background,target,title"
  )
    throw Error("Game generator returned an invalid game");
  // Credentials never enter the artifact, including accidental echoes.
  const clean = JSON.parse(
    JSON.stringify({
      ...content,
      creation: {
        brief,
        instruction,
        model: config.model,
        returned_model: response.model ?? null,
        provider: response.provider ?? null,
        temperature: config.temperature,
        created_at: Date.now(),
        request_id: response.id ?? null,
        requests,
      },
    }).replaceAll(key, "[redacted]"),
  );
  return gameSpec({
    ...clean,
    version: 1,
    format: GAME_FORMAT,
    id: crypto.randomUUID(),
  });
}
