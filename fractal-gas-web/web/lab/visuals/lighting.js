// Small procedural studio environment gives metal and glass readable reflections.
// No image downloads, post-processing passes, or per-agent lights are required.
import * as T from "../vendor/three.module.js";
import { retainResource } from "./resources.js";
export function laboratoryEnvironment(renderer, style = "futuristic") {
  const width = 256,
    height = 128,
    data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = 4 * (y * width + x),
        upper = y < height / 2;
      let rgb = upper ? [42, 60, 82] : [12, 17, 27];
      if (y > 15 && y < 48 && ((x > 20 && x < 58) || (x > 134 && x < 159)))
        rgb = [205, 228, 255];
      if (y > 32 && y < 78 && x > 204 && x < 222) rgb = [146, 75, 193];
      if (y > 35 && y < 64 && x > 82 && x < 90) rgb = [64, 193, 194];
      if (style === "steampunk") {
        const brightness = Math.max(...rgb);
        rgb = [
          brightness,
          Math.round(brightness * 0.77),
          Math.round(brightness * 0.49),
        ];
      }
      data.set([...rgb, 255], i);
    }
  const texture = new T.DataTexture(data, width, height);
  texture.colorSpace = T.SRGBColorSpace;
  texture.mapping = T.EquirectangularReflectionMapping;
  texture.needsUpdate = true;
  const generator = new T.PMREMGenerator(renderer),
    target = generator.fromEquirectangular(texture);
  generator.dispose();
  texture.dispose();
  return target;
}
let shadowTexture;
export function contactShadow() {
  if (!shadowTexture) {
    const size = 32,
      data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const d = Math.hypot(
          (x - size / 2) / (size / 2),
          (y - size / 2) / (size / 2),
        );
        data[4 * (y * size + x) + 3] = Math.round(
          Math.max(0, 1 - d) ** 2 * 180,
        );
      }
    shadowTexture = new T.DataTexture(data, size, size);
    shadowTexture.needsUpdate = true;
    retainResource(shadowTexture);
  }
  const mesh = new T.Mesh(
    new T.PlaneGeometry(2.1, 1.6),
    new T.MeshBasicMaterial({
      map: shadowTexture,
      transparent: true,
      depthWrite: false,
    }),
  );
  mesh.position.set(-0.1, 0, -0.075);
  mesh.name = "contact-shadow";
  return mesh;
}
