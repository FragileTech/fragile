// Shared asset envelopes. Physics always comes from the scene.
export const worldCatalog = Object.freeze({
  "drop-crystal": {
    family: "collectible-drops",
    size: [1, 1, 0.8],
  },
  "drop-nugget": {
    family: "collectible-drops",
    size: [1, 1, 0.8],
  },
  "drop-salvage": {
    family: "collectible-drops",
    size: [1, 1, 0.8],
  },
  "drop-capsule": {
    family: "collectible-drops",
    size: [1, 1, 0.8],
  },
  "drop-core": {
    family: "collectible-drops",
    size: [1, 1, 0.8],
  },
  "drop-pile": {
    family: "collectible-drops",
    size: [1, 1, 0.8],
  },
  "ore-small": {
    family: "ore-rocks",
    size: [2, 2, 1.6],
  },
  "ore-medium": {
    family: "ore-rocks",
    size: [2, 2, 1.6],
  },
  "ore-large": {
    family: "ore-rocks",
    size: [2, 2, 1.6],
  },
  "ore-captured": {
    family: "ore-rocks",
    size: [2, 2, 1.6],
  },
  "capture-clamp": {
    family: "ore-rocks",
    size: [0.6, 0.5, 0.35],
  },
  reactor: {
    family: "gravity-well",
    size: [2.4, 2.4, 2.8],
  },
  dock: {
    family: "recovery-dock",
    size: [2, 2, 0.38],
  },
  beacon: {
    family: "recovery-dock",
    size: [0.45, 0.45, 0.9],
  },
  gate: {
    family: "checkpoints",
    size: [0.45, 2, 2],
  },
  "gate-pylons": {
    family: "checkpoints",
    size: [0.5, 2, 1.6],
  },
  "gate-marker": {
    family: "checkpoints",
    size: [2, 2, 0.12],
  },
  "finish-arch": {
    family: "checkpoints",
    size: [0.6, 2, 1.4],
  },
  rail: {
    family: "arena-scenery",
    size: [2, 0.3, 0.75],
  },
  "corner-inner": {
    family: "arena-scenery",
    size: [0.8, 0.8, 0.75],
  },
  "corner-outer": {
    family: "arena-scenery",
    size: [0.8, 0.8, 0.75],
  },
  bollard: {
    family: "arena-scenery",
    size: [0.45, 0.45, 0.65],
  },
  "rock-obstacle": {
    family: "arena-scenery",
    size: [2, 2, 1.5],
  },
  "floor-tile": {
    family: "arena-scenery",
    size: [2, 2, 0.08],
  },
  lamp: {
    family: "arena-scenery",
    size: [0.5, 0.5, 2],
  },
  "utility-column": {
    family: "arena-scenery",
    size: [0.8, 0.8, 1.8],
  },
  deposit: {
    family: "arena-scenery",
    size: [1.8, 1.8, 1.4],
  },
  gantry: {
    family: "arena-scenery",
    size: [0.65, 2, 1.8],
  },
  "track-straight": {
    family: "racing-scenery",
    size: [4, 3, 0.12],
  },
  "track-bend": {
    family: "racing-scenery",
    size: [4, 4, 0.12],
  },
  kerb: {
    family: "racing-scenery",
    size: [2, 0.4, 0.12],
  },
  guardrail: {
    family: "racing-scenery",
    size: [2, 0.3, 0.5],
  },
  "start-light": {
    family: "racing-scenery",
    size: [0.5, 0.5, 2],
  },
  "pit-station": {
    family: "racing-scenery",
    size: [2, 1.5, 1.5],
  },
  "direction-board": {
    family: "racing-scenery",
    size: [0.5, 1.3, 1.4],
  },
  "tether-fitting": {
    family: "capture-effects",
    size: [0.5, 0.4, 0.5],
  },
  "capture-latch": {
    family: "capture-effects",
    size: [0.6, 0.28, 0.28],
  },
  "thrust-plume": {
    family: "capture-effects",
    size: [1, 0.4, 0.4],
  },
  "intake-swirl": {
    family: "capture-effects",
    size: [1, 1, 0.3],
  },
  "pickup-burst": {
    family: "capture-effects",
    size: [1, 1, 1],
  },
  "rotor-airflow": {
    family: "capture-effects",
    size: [1.5, 1.5, 0.4],
  },
  "path-trail": {
    family: "capture-effects",
    size: [2, 0.4, 0.12],
  },
});
export const worldModels = Object.freeze(Object.keys(worldCatalog));
