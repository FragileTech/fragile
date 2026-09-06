export function encode(array) {
  const bytes = new Uint8Array(
    array.buffer,
    array.byteOffset,
    array.byteLength,
  );
  let string = "";
  for (let i = 0; i < bytes.length; i += 8192)
    string += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(string);
}
export function decode(value, Type = Uint8Array) {
  if (typeof value !== "string") throw new Error("Invalid binary field");
  const string = atob(value),
    bytes = Uint8Array.from(string, (c) => c.charCodeAt(0));
  if (bytes.length % Type.BYTES_PER_ELEMENT)
    throw new Error("Misaligned binary field");
  return new Type(bytes.buffer);
}
const table = Uint32Array.from({ length: 256 }, (_, i) => {
  for (let k = 0; k < 8; k++) i = (i >>> 1) ^ (i & 1 ? 0xedb88320 : 0);
  return i >>> 0;
});
export function checksum(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = (crc >>> 8) ^ table[(crc ^ byte) & 255];
  return (crc ^ 0xffffffff) >>> 0;
}
