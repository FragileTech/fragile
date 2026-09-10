import { cp, mkdir, readFile, rm } from "node:fs/promises";
const root = new URL("../", import.meta.url);
const pkg = new URL("node_modules/d3-hierarchy/", root);
const version = JSON.parse(
  await readFile(new URL("package.json", pkg)),
).version;
if (version !== "3.1.2")
  throw new Error("LLM layout requires d3-hierarchy 3.1.2");
const dest = new URL("web/llm/vendor/d3-hierarchy/", root);
await rm(dest, { recursive: true, force: true });
await mkdir(dest, { recursive: true });
await cp(new URL("src/", pkg), dest, { recursive: true });
await cp(new URL("LICENSE", pkg), new URL("LICENSE", dest));
console.log("Bundled d3-hierarchy 3.1.2 for offline LLM analysis");

const tokenizerPkg = new URL("node_modules/@huggingface/tokenizers/", root);
if (
  JSON.parse(await readFile(new URL("package.json", tokenizerPkg))).version !==
  "0.2.0"
)
  throw Error("Unexpected tokenizer version");
const tokenizerDest = new URL("web/llm/vendor/tokenizer/", root);
await mkdir(tokenizerDest, { recursive: true });
await cp(
  new URL("dist/tokenizers.mjs", tokenizerPkg),
  new URL("tokenizers.mjs", tokenizerDest),
);
await cp(
  new URL("LICENSE", tokenizerPkg),
  new URL("LIBRARY_LICENSE", tokenizerDest),
);
await cp(new URL("llm/tokenizer/", root), tokenizerDest, { recursive: true });
console.log("Bundled pinned Qwen tokenizer for XED alignment");
