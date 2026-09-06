import { checksum, encode, decode } from "../binary.js";
export async function compress(bytes) {
  const codec = typeof CompressionStream === "function" ? "gzip" : "raw";
  const data =
    codec === "raw"
      ? bytes.slice()
      : new Uint8Array(
          await new Response(
            new Blob([bytes])
              .stream()
              .pipeThrough(new CompressionStream("gzip")),
          ).arrayBuffer(),
        );
  return { codec, data, checksum: checksum(bytes), rawBytes: bytes.byteLength };
}
export async function decompress(record, limit = 128 * 1024 * 1024) {
  if (
    !Number.isInteger(record.rawBytes) ||
    record.rawBytes < 0 ||
    record.rawBytes > limit
  )
    throw new Error("Invalid stored chunk size");
  let bytes;
  if (record.codec === "raw") bytes = new Uint8Array(record.data);
  else if (record.codec === "gzip") {
    const reader = new Blob([record.data])
        .stream()
        .pipeThrough(new DecompressionStream("gzip"))
        .getReader(),
      parts = [];
    let size = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.length;
      if (size > record.rawBytes) {
        await reader.cancel();
        throw new Error("Stored chunk exceeds declared size");
      }
      parts.push(value);
    }
    bytes = new Uint8Array(size);
    let at = 0;
    for (const p of parts) {
      bytes.set(p, at);
      at += p.length;
    }
  } else throw new Error("Unsupported chunk codec");
  if (bytes.length !== record.rawBytes || checksum(bytes) !== record.checksum)
    throw new Error("Stored recording checksum mismatch");
  return bytes;
}
const types = {
  Uint8Array,
  Uint32Array,
  Int32Array,
  Float32Array,
  Float64Array,
};
export function encodeObject(value) {
  return new TextEncoder().encode(
    JSON.stringify(value, (_, v) =>
      ArrayBuffer.isView(v)
        ? { $typed: v.constructor.name, data: encode(v) }
        : v,
    ),
  );
}
export function decodeObject(bytes) {
  return JSON.parse(new TextDecoder().decode(bytes), (_, v) =>
    v?.$typed
      ? decode(
          v.data,
          types[v.$typed] ||
            (() => {
              throw new Error("Unsupported typed array");
            })(),
        )
      : v,
  );
}
