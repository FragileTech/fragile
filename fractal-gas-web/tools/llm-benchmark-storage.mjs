import {
  mkdir,
  open,
  readFile,
  rename,
  unlink,
  truncate,
} from "node:fs/promises";
import { join } from "node:path";
import { hostname } from "node:os";
import {
  validateManifest,
  validateArchive,
} from "../web/llm/benchmark-data.js";

async function atomicJSON(path, value) {
  const tmp = `${path}.${process.pid}.tmp`;
  const file = await open(tmp, "w", 0o600);
  try {
    await file.writeFile(JSON.stringify(value) + "\n");
    await file.sync();
  } finally {
    await file.close();
  }
  await rename(tmp, path);
}
export async function lockDirectory(directory, fn) {
  await mkdir(directory, { recursive: true });
  const path = join(directory, ".writer.lock");
  let lock;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      lock = await open(path, "wx", 0o600);
      break;
    } catch (error) {
      if (error.code !== "EEXIST") throw error;
      let owner;
      try {
        owner = JSON.parse(await readFile(path, "utf8"));
      } catch {
        throw Error(
          "Benchmark writer lock is unreadable; inspect it before removing it",
        );
      }
      if (
        owner.host !== hostname() ||
        !Number.isInteger(owner.pid) ||
        owner.pid < 1
      )
        throw Error("Benchmark has another writer");
      try {
        process.kill(owner.pid, 0);
        throw Error("Benchmark has another active writer");
      } catch (e) {
        if (e.code !== "ESRCH") throw e;
      }
      await unlink(path);
    }
  }
  if (!lock) throw Error("Could not acquire benchmark writer lock");
  try {
    await lock.writeFile(
      JSON.stringify({ pid: process.pid, host: hostname() }),
    );
    await lock.sync();
    return await fn();
  } finally {
    await lock.close();
    await unlink(path);
  }
}
export class DiskBenchmarkStore {
  constructor(directory, header, events) {
    this.directory = directory;
    this.manifest = header;
    this.lastSeq = events.length;
  }
  static async create(directory, header) {
    header = validateManifest(header);
    await mkdir(directory, { recursive: true });
    // The journal header is authoritative; never overwrite an existing benchmark.
    const file = await open(join(directory, "events.fgllmbench"), "wx", 0o600);
    try {
      await file.writeFile(JSON.stringify(header) + "\n");
      await file.sync();
    } finally {
      await file.close();
    }
    const store = new DiskBenchmarkStore(directory, header, []);
    await store.checkpoint();
    return store;
  }
  static async open(directory, { recover = false } = {}) {
    const path = join(directory, "events.fgllmbench");
    const bytes = await readFile(path);
    const end = bytes.lastIndexOf(10) + 1;
    if (!end) throw Error("Invalid benchmark journal header");
    if (end !== bytes.length && recover) await truncate(path, end);
    const [header, ...events] = bytes
      .subarray(0, end)
      .toString("utf8")
      .trimEnd()
      .split("\n")
      .map((line) => JSON.parse(line));
    validateArchive(header, events);
    const store = new DiskBenchmarkStore(
      directory,
      validateManifest(header),
      events,
    );
    if (recover) await store.checkpoint();
    return store;
  }
  async append(event) {
    if (event.seq !== this.lastSeq + 1)
      throw Error("Concurrent benchmark writer");
    const file = await open(
      join(this.directory, "events.fgllmbench"),
      "a",
      0o600,
    );
    try {
      await file.writeFile(JSON.stringify(event) + "\n");
      await file.sync();
    } finally {
      await file.close();
    }
    this.lastSeq = event.seq;
    await this.checkpoint();
  }
  async checkpoint() {
    await atomicJSON(join(this.directory, "manifest.json"), {
      ...this.manifest,
      last_seq: this.lastSeq,
      updated_at: Date.now(),
    });
  }
  async readEvents() {
    const text = await readFile(
      join(this.directory, "events.fgllmbench"),
      "utf8",
    );
    return text
      .slice(0, text.lastIndexOf("\n"))
      .split("\n")
      .slice(1)
      .map((line) => JSON.parse(line));
  }
}
