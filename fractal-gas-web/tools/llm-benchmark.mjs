import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";
import { BenchmarkRunner } from "../web/llm/benchmark.js";
import {
  manifest,
  parseBenchmark,
  exportBenchmark,
  processBenchmark,
  collectRuns,
  fractalRecording,
} from "../web/llm/benchmark-data.js";
import { DiskBenchmarkStore, lockDirectory } from "./llm-benchmark-storage.mjs";
import {
  parseReport,
  processReport,
  verifyReportIdentities,
} from "../web/llm/comparison-report.js";

export async function main(
  args = process.argv.slice(2),
  { env = process.env, runnerOptions = {} } = {},
) {
  const [command, ...rest] = args,
    options = {};
  for (let i = 0; i < rest.length; i++) {
    const key = rest[i];
    if (key === "--retry-incomplete") {
      options.retry = true;
      continue;
    }
    if (
      !["--config", "--output", "--input", "--file"].includes(key) ||
      !rest[i + 1] ||
      rest[i + 1].startsWith("--")
    )
      throw Error(`Invalid argument: ${key}`);
    options[key.slice(2)] = rest[++i];
  }
  if (command === "help" || command === "--help" || !command) {
    console.log(
      "LLM benchmark: run --config settings.json --output DIR | resume --output DIR --retry-incomplete | export --output DIR --file archive.fgllmbench | process --input archive.fgllmbench --output DIR",
    );
    return;
  }
  if (!options.output) throw Error("Specify --output");
  // npm --prefix changes cwd for scripts; resolve paths where the user invoked it.
  const userPath = (value) => resolve(env.INIT_CWD || process.cwd(), value);
  const directory = userPath(options.output);
  if (command === "process") {
    if (!options.input) throw Error("Specify --input");
    const input = await readFile(userPath(options.input), "utf8");
    let report;
    try {
      if (JSON.parse(input)?.format === "fgllmcompare")
        report = await verifyReportIdentities(parseReport(input));
    } catch (e) {
      if (/"format"\s*:\s*"fgllmcompare"/.test(input)) throw e;
    }
    const archive = report
      ? report.source.kind === "benchmark"
        ? report.source
        : null
      : parseBenchmark(input);
    const tables = report
      ? processReport(report)
      : processBenchmark(archive.manifest, archive.events);
    await mkdir(directory, { recursive: true });
    for (const [name, rows] of Object.entries(tables))
      await writeFile(
        join(directory, `${name}.jsonl`),
        rows.map((row) => JSON.stringify(row) + "\n").join(""),
        { flag: "wx", mode: 0o600 },
      );
    await writeFile(
      join(directory, "manifest.json"),
      JSON.stringify(
        report
          ? {
              format: report.format,
              version: report.version,
              id: report.id,
              created_at: report.created_at,
              view: report.view,
              source: report.source.kind,
            }
          : archive.manifest,
        null,
        2,
      ) + "\n",
      { flag: "wx", mode: 0o600 },
    );
    if (archive)
      for (const run of collectRuns(
        archive.manifest,
        archive.events,
      ).runs.values())
        if (run.method === "fractal")
          await writeFile(
            join(directory, `${run.id.replaceAll(":", "-")}.fgllm`),
            JSON.stringify(fractalRecording(run)),
            { flag: "wx", mode: 0o600 },
          );
    if (report?.source.kind === "recording")
      await writeFile(
        join(directory, "current.fgllm"),
        JSON.stringify(report.source.record),
        { flag: "wx", mode: 0o600 },
      );
    return;
  }
  if (command === "export") {
    if (!options.file) throw Error("Specify --file");
    await lockDirectory(directory, async () => {
      const store = await DiskBenchmarkStore.open(directory);
      await writeFile(
        userPath(options.file),
        exportBenchmark(store.manifest, await store.readEvents()),
        { flag: "wx", mode: 0o600 },
      );
    });
    return;
  }
  if (!["run", "resume"].includes(command))
    throw Error("Unknown benchmark command");
  if (command === "run" && !options.config) throw Error("Specify --config");
  if (!env.OPENROUTER_API_KEY)
    throw Error("Set OPENROUTER_API_KEY before generating a benchmark");
  const settings =
    command === "run"
      ? JSON.parse(await readFile(userPath(options.config), "utf8"))
      : null;
  await lockDirectory(directory, async () => {
    const store =
      command === "run"
        ? await DiskBenchmarkStore.create(directory, manifest(settings))
        : await DiskBenchmarkStore.open(directory, { recover: true });
    const runner = new BenchmarkRunner(store, env.OPENROUTER_API_KEY, {
      togetherKey: env.TOGETHER_API_KEY,
      onStatus: (text) => console.log(text),
      ...runnerOptions,
    });
    const stop = () => runner.stop();
    process.once("SIGINT", stop);
    process.once("SIGTERM", stop);
    try {
      await runner.run({ retryIncomplete: options.retry ?? false });
    } finally {
      process.removeListener("SIGINT", stop);
      process.removeListener("SIGTERM", stop);
    }
  });
}
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  main().catch((error) => {
    const key = process.env.OPENROUTER_API_KEY;
    error.message = String(error.message).replaceAll(
      process.env.TOGETHER_API_KEY || "\0",
      "[redacted]",
    );
    console.error(
      key ? String(error.message).replaceAll(key, "[redacted]") : error.message,
    );
    process.exitCode = 1;
  });
}
