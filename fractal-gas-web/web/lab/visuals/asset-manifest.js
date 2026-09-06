export const vehicleModels = Object.freeze([
  "rocket",
  "kart",
  "drone",
  "harvester",
]);
export const propModels = Object.freeze(["dock", "gate", "reactor"]);
export const assetManifest = Object.freeze(
  Object.fromEntries(
    ["futuristic", "steampunk"].map((style) => [
      style,
      Object.freeze(
        Object.fromEntries(
          [...vehicleModels, ...propModels].map((model) => [
            model,
            Object.freeze(
              Object.fromEntries(
                ["high", "low"].map((lod) => [
                  lod,
                  new URL(
                    `../assets/${style}/${model}-${lod}.glb`,
                    import.meta.url,
                  ).href,
                ]),
              ),
            ),
          ]),
        ),
      ),
    ]),
  ),
);
