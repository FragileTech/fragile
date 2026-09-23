import { configuration } from "./config.js";
import { manifest, collectRuns } from "./benchmark-data.js";
import { BenchmarkRunner } from "./benchmark.js";
import { GAME_MODES } from "./games.js";

export function pairManifests(
  settings,
  pair = {
    id: crypto.randomUUID(),
    unsurprising: crypto.randomUUID(),
    surprising: crypto.randomUUID(),
  },
) {
  if (settings.config.objective !== "xent_game")
    throw Error("Select an Xent game first");
  return GAME_MODES.map((mode) => ({
    ...manifest(
      {
        ...settings,
        config: configuration({ ...settings.config, game_mode: mode }),
      },
      pair[mode],
    ),
    pair: structuredClone(pair),
  }));
}

export async function runGamePair(stores, key, options = {}) {
  if (stores.length !== 2) throw Error("Both game benchmarks are required");
  const [a, b] = stores.map((s) => s.manifest);
  const withoutMode = (header) => {
    const copy = structuredClone(header.settings);
    delete copy.config.game_mode;
    delete copy.config.prompt;
    return JSON.stringify(copy);
  };
  if (
    !a.pair ||
    JSON.stringify(a.pair) !== JSON.stringify(b.pair) ||
    a.settings.config.game_mode !== "unsurprising" ||
    b.settings.config.game_mode !== "surprising" ||
    withoutMode(a) !== withoutMode(b)
  )
    throw Error("Incompatible game benchmark pair");
  let sharedMetadata = null;
  for (const store of stores) {
    const { metadata } = collectRuns(store.manifest, await store.readEvents());
    if (metadata) {
      if (
        sharedMetadata &&
        (JSON.stringify(metadata.provider) !==
          JSON.stringify(sharedMetadata.provider) ||
          JSON.stringify(metadata.scoring) !==
            JSON.stringify(sharedMetadata.scoring))
      )
        throw Error(
          "Game pair uses different scoring evidence or generation routes",
        );
      sharedMetadata = metadata;
    }
  }
  for (const store of stores) {
    options.signal?.throwIfAborted();
    const runner = new BenchmarkRunner(store, key, {
      ...options,
      sharedMetadata,
    });
    options.onRunner?.(runner, store.manifest);
    const stop = () => runner.stop();
    options.signal?.addEventListener("abort", stop, { once: true });
    try {
      options.signal?.throwIfAborted();
      await runner.run({ retryIncomplete: options.retryIncomplete ?? false });
    } finally {
      options.signal?.removeEventListener("abort", stop);
    }
    sharedMetadata =
      collectRuns(store.manifest, await store.readEvents()).metadata ??
      sharedMetadata;
  }
}
