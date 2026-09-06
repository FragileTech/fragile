import { worldModels } from "./world-catalog.js";
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
          [...vehicleModels, "refinery", ...worldModels].map((model) => [
            model,
            Object.freeze(
              Object.fromEntries(
                ["high", "low"].map((lod) => [
                  lod,
                  new URL(
                    worldModels.includes(model)
                      ? `../assets/${style}/world-${lod}.glb#${model}`
                      : `../assets/${style}/${model}-${lod}.glb`,
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
