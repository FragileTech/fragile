import assert from "node:assert/strict";
import { GLTFLoader } from "../../web/lab/vendor/addons/loaders/GLTFLoader.js";
export function glbJson(bytes) {
  assert.equal(bytes.readUInt32LE(0), 0x46546c67);
  assert.equal(bytes.readUInt32LE(4), 2);
  assert.equal(bytes.readUInt32LE(8), bytes.length);
  return JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString());
}

// Geometry/metadata round trip uses the production loader. Image decoding itself
// is covered by the browser check; Node does not provide ImageBitmap.
export async function geometryOnly(bytes, json) {
  const model = structuredClone(json);
  delete model.images;
  delete model.textures;
  const stripTextures = (value) => {
    if (!value || typeof value !== "object") return;
    for (const key of Object.keys(value)) {
      if (key.endsWith("Texture")) delete value[key];
      else stripTextures(value[key]);
    }
  };
  stripTextures(model.materials);
  const jsonBuffer = Buffer.from(JSON.stringify(model));
  const padded = Buffer.alloc(Math.ceil(jsonBuffer.length / 4) * 4, 32);
  jsonBuffer.copy(padded);
  const binary = bytes.subarray(20 + bytes.readUInt32LE(12));
  const result = Buffer.alloc(20 + padded.length + binary.length);
  result.writeUInt32LE(0x46546c67, 0);
  result.writeUInt32LE(2, 4);
  result.writeUInt32LE(result.length, 8);
  result.writeUInt32LE(padded.length, 12);
  result.writeUInt32LE(0x4e4f534a, 16);
  padded.copy(result, 20);
  binary.copy(result, 20 + padded.length);
  return (
    await new GLTFLoader().parseAsync(
      result.buffer.slice(
        result.byteOffset,
        result.byteOffset + result.byteLength,
      ),
      "",
    )
  ).scene;
}
