import createLlmModule from "./engine/llm.mjs";
export class NativeLlm {
  static async create(config, transition) {
    const module = await createLlmModule({ llmTransition: transition });
    const handle = module.ccall(
      "fgllm_create",
      "number",
      ["string"],
      [JSON.stringify(config)],
    );
    if (!handle) throw new Error(module.ccall("fgllm_error", "string", [], []));
    return new NativeLlm(module, handle);
  }
  constructor(module, handle) {
    this.module = module;
    this.handle = handle;
    this.started = false;
    this.busy = false;
  }
  async advance() {
    if (this.busy || !this.handle) throw new Error("Engine is busy or closed");
    this.busy = true;
    try {
      const code = await this.module.ccall(
        "fgllm_advance",
        "number",
        ["number", "number"],
        [this.handle, this.started ? 0 : 1],
        { async: true },
      );
      if (code !== 0)
        throw new Error(
          this.module.llmError ||
            this.module.ccall("fgllm_error", "string", [], []),
        );
      this.started = true;
      return JSON.parse(
        this.module.ccall(
          "fgllm_snapshot",
          "string",
          ["number"],
          [this.handle],
        ),
      );
    } finally {
      this.busy = false;
    }
  }
  close() {
    if (this.busy) throw new Error("Cannot close a pending engine");
    if (this.handle)
      this.module.ccall("fgllm_destroy", null, ["number"], [this.handle]);
    this.handle = 0;
  }
}
