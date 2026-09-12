import { metadata } from "./registry.js";
import { createRunModel } from "./run-model.js";

export const demos = metadata
  .filter((entry) => ["II", "III"].includes(entry.part))
  .map((entry) => ({
    ...entry,
    create: (args) => createRunModel({ ...args, id: entry.id }),
  }));
