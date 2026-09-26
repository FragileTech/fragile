import { encode, decode, checksum } from "../lab/binary.js";
import { ENGINE_VERSION, frameInfo } from "./native.js";
import { validateGeometry } from "./geometry.js";
export const RECORDING_LIMIT = 64 * 1024 * 1024;
export class Recording {
  constructor(config, engine = ENGINE_VERSION, retainHistory = true) {
    this.retainHistory = retainHistory;
    this.engine = engine;
    this.config = structuredClone(config);
    this.frames = [];
    this.metadata = [];
    this.bytes = new TextEncoder().encode(
      JSON.stringify(this.config),
    ).byteLength;
    if (this.bytes >= RECORDING_LIMIT)
      throw new Error("Configuration exceeds the 64 MiB recording limit");
  }
  append(frame, metadata = null) {
    validateGeometry(metadata?.geometry, frameInfo(frame).d);
    const copy = metadata === null ? null : structuredClone(metadata);
    if (copy !== null && (typeof copy !== "object" || Array.isArray(copy)))
      throw new Error("Invalid frame metadata");
    const metaBytes =
      copy === null
        ? 0
        : new TextEncoder().encode(JSON.stringify(copy)).byteLength;
    const oldMetaBytes =
      !this.retainHistory && this.metadata[0]
        ? new TextEncoder().encode(JSON.stringify(this.metadata[0])).byteLength
        : 0;
    const oldBytes = !this.retainHistory
      ? (this.frames[0]?.byteLength || 0) + oldMetaBytes
      : 0;
    const nextBytes = this.bytes - oldBytes + frame.byteLength + metaBytes;
    if (nextBytes > RECORDING_LIMIT)
      throw new Error(
        "Recording reached 64 MiB. Save this run and reset to continue.",
      );
    this.bytes = nextBytes;
    if (!this.retainHistory) {
      this.frames = [frame];
      this.metadata = [copy];
    } else {
      this.frames.push(frame);
      this.metadata.push(copy);
    }
  }
  replaceLast(frame, metadata) {
    validateGeometry(metadata?.geometry, frameInfo(frame).d);
    const i = this.frames.length - 1;
    if (
      i < 0 ||
      frameInfo(frame).iteration !== frameInfo(this.frames[i]).iteration
    )
      throw new Error("Population update must preserve the current iteration");
    const copy = structuredClone(metadata);
    const metadataBytes = (value) =>
      value == null
        ? 0
        : new TextEncoder().encode(JSON.stringify(value)).byteLength;
    const bytes =
      this.bytes -
      this.frames[i].byteLength -
      metadataBytes(this.metadata[i]) +
      frame.byteLength +
      metadataBytes(copy);
    if (bytes > RECORDING_LIMIT)
      throw new Error("Population update exceeds the recording limit");
    this.frames[i] = frame;
    this.metadata[i] = copy;
    this.bytes = bytes;
  }
  export() {
    const archiveDeltas = this.metadata.some(
      (m) => m?.controller?.basins?.length > 0,
    );
    let previousArchive = null,
      previousArchiveIndex = -1;
    const metadata = this.metadata.map((m, index) => {
      if (!archiveDeltas || !m?.controller?.basins) return m;
      const serialized = JSON.stringify(m.controller.basins);
      if (serialized === previousArchive) {
        const controller = {
          ...m.controller,
          archive_ref: previousArchiveIndex,
        };
        delete controller.basins;
        return { ...m, controller };
      }
      previousArchive = serialized;
      previousArchiveIndex = index;
      return m;
    });
    return JSON.stringify({
      format: "fgopt",
      version: archiveDeltas
        ? 4
        : this.metadata.some((m) => m?.settings_update)
          ? 3
          : this.config.algorithm === "graph" &&
              this.config.freeze_prefix_after > 0
            ? 2
            : 1,
      engine: this.engine,
      config: this.config,
      frames: this.frames.map((frame, i) => ({
        ...(metadata[i] === null ? {} : { metadata: metadata[i] }),
        data: encode(frame),
        checksum: checksum(
          new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength),
        ),
      })),
    });
  }
}
export function validateFrame(frame, config, previous = -1) {
  if (!(frame instanceof Float64Array) || frame.length < 12 || frame[0] !== 1)
    throw new Error("Invalid recording frame");
  const { n, d, stride, iteration, alive, bestIndex } = frameInfo(frame);
  if (
    !Number.isSafeInteger(n) ||
    n < 1 ||
    n > 1000000 ||
    !Number.isSafeInteger(d) ||
    d !== config.dimensions ||
    d < 1 ||
    d > 4096 ||
    frame.length !== 12 + n * stride ||
    !Number.isSafeInteger(iteration) ||
    iteration <= previous ||
    !Number.isSafeInteger(alive) ||
    alive < 0 ||
    alive > n ||
    !Number.isInteger(bestIndex) ||
    bestIndex < -1 ||
    bestIndex >= n
  )
    throw new Error("Recording frame dimensions or counters are invalid");
  if (frame[3] !== 0 && frame[3] !== 1)
    throw new Error("Invalid velocity flag");
  let count = 0;
  for (let i = 0; i < n; i++) {
    const o = 12 + i * stride,
      active = frame[o + 2 * d + 2];
    if (active !== 0 && active !== 1)
      throw new Error("Invalid walker validity mask");
    if (active) {
      count++;
      for (let k = 0; k <= 2 * d; k++)
        if (!Number.isFinite(frame[o + k]))
          throw new Error("Nonfinite active walker");
    }
    for (const k of [3, 4, 5]) {
      const index = frame[o + 2 * d + k];
      if (!Number.isInteger(index) || index < 0 || index >= n)
        throw new Error("Invalid companion or ancestry index");
    }
    for (const k of [6, 7])
      if (![0, 1].includes(frame[o + 2 * d + k]))
        throw new Error("Invalid walker flags");
  }
  if (count !== alive) throw new Error("Recording alive count mismatch");
  return iteration;
}
export function importRecording(text) {
  if (text.length > 96 * 1024 * 1024)
    throw new Error("Recording file is too large");
  const value = JSON.parse(text);
  if (
    value.format !== "fgopt" ||
    ![1, 2, 3, 4].includes(value.version) ||
    ![
      "fgopt-1",
      "fgopt-2",
      "fgopt-3",
      "fgopt-4",
      "fgopt-5",
      "fgopt-6",
      "fgopt-7",
      "fgopt-8",
      "fgopt-9",
      "fgopt-10",
      ENGINE_VERSION,
    ].includes(value.engine) ||
    !value.config ||
    typeof value.config !== "object" ||
    !Array.isArray(value.frames) ||
    !value.frames.length
  )
    throw new Error("Unsupported optimization recording");
  const recording = new Recording(value.config, value.engine);
  let previous = -1;
  for (const entry of value.frames) {
    const controller = entry.metadata?.controller;
    if (value.version === 4 && controller && "archive_ref" in controller) {
      const reference = controller.archive_ref;
      if (
        !Number.isSafeInteger(reference) ||
        reference < 0 ||
        reference >= recording.frames.length ||
        !Array.isArray(recording.metadata[reference]?.controller?.basins)
      )
        throw new Error("Invalid archive reference");
      controller.basins = structuredClone(
        recording.metadata[reference].controller.basins,
      );
      delete controller.archive_ref;
    }
    if (
      typeof entry.data !== "string" ||
      entry.data.length > ((RECORDING_LIMIT - recording.bytes) * 4) / 3 + 4
    )
      throw new Error("Recording exceeds its memory limit");
    const frame = decode(entry.data, Float64Array);
    if (checksum(new Uint8Array(frame.buffer)) !== entry.checksum)
      throw new Error("Recording checksum mismatch");
    const update =
      value.version >= 3 && entry.metadata?.settings_update === true;
    if (
      update &&
      (frameInfo(frame).iteration !== previous ||
        !entry.metadata?.settings_event ||
        frameInfo(frame).evaluations !==
          frameInfo(recording.frames.at(-1)).evaluations ||
        frameInfo(frame).best !== frameInfo(recording.frames.at(-1)).best)
    )
      throw new Error("Invalid settings update boundary");
    previous = validateFrame(
      frame,
      value.config,
      update ? previous - 1 : previous,
    );
    if (
      value.version === 2 ||
      (value.version >= 3 && value.config.algorithm === "graph")
    ) {
      const meta = entry.metadata;
      if (
        !meta ||
        !Array.isArray(meta.frozen) ||
        !Number.isSafeInteger(meta.activeRootId) ||
        !Number.isSafeInteger(meta.activeRootParentId)
      )
        throw new Error("Invalid frozen Graph metadata");
      const ids = new Set();
      for (const node of meta.frozen) {
        if (
          !Number.isSafeInteger(node.id) ||
          node.id < 0 ||
          !Number.isSafeInteger(node.parentId) ||
          node.parentId < 0 ||
          !Array.isArray(node.x) ||
          node.x.length !== value.config.dimensions ||
          !node.x.every(Number.isFinite) ||
          !Number.isFinite(node.value) ||
          ids.has(node.id)
        )
          throw new Error("Invalid frozen Graph path");
        ids.add(node.id);
      }
    }
    recording.append(frame, entry.metadata ?? null);
  }
  return recording;
}
