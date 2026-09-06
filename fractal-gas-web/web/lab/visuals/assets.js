import * as T from "../vendor/three.module.js";
import { assetManifest } from "./asset-manifest.js";
import { retainAsset, disposeGroup } from "./resources.js";

const cache = new Map();
const pending = new Map();
let loader;
const keyOf = (style, model, lod) => `${style}/${model}/${lod}`;

export function styleAssetsReady(style) {
  return (
    !!assetManifest[style] &&
    Object.entries(assetManifest[style]).every(([model, levels]) =>
      Object.keys(levels).every((lod) => cache.has(keyOf(style, model, lod))),
    )
  );
}

export async function preloadStyle(style) {
  if (!assetManifest[style]) throw new Error(`Unknown visual style: ${style}`);
  if (styleAssetsReady(style)) return;
  if (pending.has(style)) return pending.get(style);
  const promise = (async () => {
    loader ??= import("../vendor/addons/loaders/GLTFLoader.js").then(
      ({ GLTFLoader }) => new GLTFLoader(),
    );
    const gltfLoader = await loader;
    // Limit image decoding and transient memory while loading a full collection.
    for (const [model, levels] of Object.entries(assetManifest[style])) {
      const results = await Promise.allSettled(
        Object.entries(levels).map(async ([lod, url]) => {
          const key = keyOf(style, model, lod);
          if (cache.has(key)) return;
          const gltf = await gltfLoader.loadAsync(url);
          retainAsset(gltf.scene);
          cache.set(key, gltf.scene);
        }),
      );
      const failure = results.find((result) => result.status === "rejected");
      if (failure) throw failure.reason;
    }
  })();
  pending.set(style, promise);
  try {
    await promise;
  } finally {
    pending.delete(style);
  }
}

export function assetModel(style, model, lod = "high", color) {
  const source = cache.get(keyOf(style, model, lod));
  if (!source) return null;
  const root = source.clone(true);
  root.userData.assetStyle = style;
  root.userData.assetModel = model;
  root.userData.lod = lod;
  root.traverse((part) => {
    if (part.userData.wheelRadius) {
      let scale = 1;
      for (let p = part; p; p = p.parent) scale *= p.scale.x;
      part.userData.wheelRadius *= scale;
    }
  });
  // Identification is a small underbody marker; authored bodywork keeps its palette.
  if (color != null) {
    const marker = new T.Mesh(
      new T.RingGeometry(0.13, 0.17, 16),
      new T.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
      }),
    );
    marker.name = "Agent identification marker";
    marker.position.z = 0.025;
    root.add(marker);
  }
  return root;
}

// Call only after all viewports have been disposed; ordinary style changes retain
// the two bounded collections for instant reuse.
export function disposeAssetCache() {
  if (pending.size) throw new Error("Cannot dispose assets while loading");
  for (const asset of cache.values()) disposeGroup(asset, true);
  cache.clear();
}
