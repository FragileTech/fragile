import * as T from "../vendor/three.module.js";

export const stylePalette = {
  futuristic: {
    background: 0x0c1422,
    floor: 0x465563,
    wall: 0x324151,
    accent: 0x75dceb,
    energy: 0x9f82ff,
    ore: 0xb993ef,
    metal: 0x8e9fab,
  },
  steampunk: {
    background: 0x211a15,
    floor: 0x595148,
    wall: 0x4a3c2d,
    accent: 0xe2b56b,
    energy: 0xb8652b,
    ore: 0xb08a42,
    metal: 0x997348,
  },
};

// These are procedural, scene-sized meshes. Authored GLB vehicle materials are
// never sent through this remapping function.
export function themeScenery(group, style) {
  if (style !== "steampunk") return group;
  const seen = new Set();
  const vertexColor = new T.Color();
  group.traverse((object) => {
    if (object.name === "Fragile documentation logo") return;
    for (let p = object; p; p = p.parent)
      if (p.userData.authoredWorld || p.userData.assetModel) return;
    const colors = object.geometry?.getAttribute("color");
    if (colors && !seen.has(colors)) {
      seen.add(colors);
      for (let i = 0; i < colors.count; i++) {
        vertexColor.fromBufferAttribute(colors, i);
        const hsl = vertexColor.getHSL({});
        vertexColor.setHSL(0.095, 0.34, hsl.l);
        colors.setXYZ(i, vertexColor.r, vertexColor.g, vertexColor.b);
      }
      colors.needsUpdate = true;
    }
    for (const material of !object.material
      ? []
      : Array.isArray(object.material)
        ? object.material
        : [object.material]) {
      if (seen.has(material)) continue;
      seen.add(material);
      if (material.color) {
        const hsl = material.color.getHSL({});
        material.color.setHSL(
          hsl.s > 0.2 ? 0.095 : 0.085,
          Math.min(0.55, hsl.s + 0.12),
          hsl.l,
        );
      }
      if (material.emissive && material.emissive.getHex())
        material.emissive.setHex(0xb87930);
    }
  });
  return group;
}
