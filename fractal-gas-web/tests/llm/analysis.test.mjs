import test from "node:test";
import assert from "node:assert/strict";
import {
  AnalysisIndex,
  probability,
  color,
  tokenSpans,
  decisionReason,
} from "../../web/llm/analysis-data.js";
import {
  Recording,
  importRecording,
  RECORDING_LIMIT,
} from "../../web/llm/recording.js";
import { configuration } from "../../web/llm/config.js";
import { NativeLlm } from "../../web/llm/native.js";
import { OpenRouter } from "../../web/llm/openrouter.js";
import { TokenEnvironment } from "../../web/llm/environment.js";
import { fakeOpenRouter } from "./fixtures.mjs";

export function treeRecord() {
  const node = (id, parent, text, logp, status = 0) => ({
    id,
    parent,
    text,
    logp,
    tokens: text.length,
    status,
    token_data: Array.from(
      text.slice(parent === null ? 0 : parent === 0 ? 0 : 1),
    ).map((text) => ({ text, logprob: -1 })),
  });
  const nodes = [
    node(0, null, "", 0),
    node(1, 0, "a", -1),
    node(2, 0, "b", -2),
    node(3, 1, "ac", -1.1),
    node(4, 1, "ad", -1.3, 2),
    node(5, 2, "be", -2.2),
    { ...node(6, 5, "be", -2.2, 1), token_data: [] },
  ];
  const walkers = (ids, parents) =>
    ids.map((node, slot) => ({
      slot,
      node,
      parentSlot: parents?.[slot] ?? 0,
      fitness: 2 + slot,
      fitnessCompanion: 0,
      cloneCompanion: 1,
      cloned: slot === 1,
      leaf: slot > 0,
      alive: true,
      score: nodes[node]?.logp ?? 0,
    }));
  return {
    version: 2,
    config: configuration({ chunk_tokens: 1, sequence_tokens: 2 }),
    nodes,
    snapshots: [
      { node_count: 3, walkers: walkers([1, 2]), decisions: [] },
      {
        node_count: 6,
        walkers: walkers([3, 4]),
        decisions: [
          { slot: 0, evaluated: 1, fitness: 2, distance: 0.2, clone_score: -1 },
          { slot: 1, evaluated: 1, fitness: 4, distance: 0.4, clone_score: 2 },
        ],
      },
      {
        node_count: 7,
        walkers: walkers([3, 4]),
        decisions: [{ slot: 0, evaluated: 3, fitness: 9 }],
      },
    ],
  };
}
test("immutable ancestry, discarded branches, subtree filters and ancestor retention", () => {
  const index = new AnalysisIndex(treeRecord());
  assert.deepEqual(
    index.chain(6).map((n) => n.id),
    [0, 2, 5, 6],
  );
  assert.equal(index.chunk(6), "");
  assert.deepEqual(
    index.model({ step: 3 }).edges.map((e) => [e.source, e.target]),
    [
      ["root", "n:1"],
      ["root", "n:2"],
      ["n:1", "n:3"],
      ["n:1", "n:4"],
      ["n:2", "n:5"],
      ["n:5", "n:6"],
    ],
  );
  assert.deepEqual(
    index.model({ step: 3, filter: "discarded" }).nodes.map((n) => n.id),
    [0, 2, 5, 6],
  );
  assert.deepEqual(
    index
      .model({ step: 3, filter: "subtree", selected: 1 })
      .nodes.map((n) => n.id),
    [0, 1, 3, 4],
  );
  assert.deepEqual(
    index.model({ step: 3, search: "ac" }).nodes.map((n) => n.id),
    [0, 1, 3],
  );
  assert.deepEqual(
    index
      .model({ step: 3, collapsed: new Set(["n:2"]) })
      .nodes.map((n) => n.id),
    [0, 1, 2, 3, 4],
  );
  assert.equal(index.compare(3, 4).shared.id, 1);
  assert.equal(index.compare(3, 6).shared.id, 0);
});
test("Graph uses actual parent slots, preserves duplicate prefix slots and optionally unused slots", () => {
  const record = treeRecord();
  record.config.algorithm = "graph";
  const snapshot = record.snapshots.at(-1);
  snapshot.walkers = [
    { slot: 0, node: 0, parentSlot: 0 },
    { slot: 1, node: 1, parentSlot: 0 },
    { slot: 2, node: 3, parentSlot: 1 },
    { slot: 3, node: 3, parentSlot: 2 },
    { slot: 4, node: null, parentSlot: 0 },
  ];
  const index = new AnalysisIndex(record),
    model = index.model({ step: 3, mode: "graph" });
  assert.deepEqual(
    model.nodes.map((n) => [n.key, n.id]),
    [
      ["root", 0],
      ["g:1", 1],
      ["g:2", 3],
      ["g:3", 3],
    ],
  );
  assert.deepEqual(
    model.edges.map((e) => [e.source, e.target]),
    [
      ["root", "g:1"],
      ["g:1", "g:2"],
      ["g:2", "g:3"],
    ],
  );
  assert.equal(
    index.model({ step: 3, mode: "graph", unused: true }).nodes.at(-1).unused,
    true,
  );
});
test("metrics stay on evaluated prefixes, aggregate repeated decisions and preserve fixed domains", () => {
  const index = new AnalysisIndex(treeRecord());
  assert.deepEqual(index.value(1, "fitness", 2), {
    value: 3,
    count: 2,
    min: 2,
    max: 4,
    step: 2,
  });
  assert.equal(index.value(3, "fitness", 2).value, null);
  assert.equal(index.value(3, "fitness", 3).value, 9);
  assert.equal(index.value(1, "fitness", 1).value, null);
  assert.deepEqual(index.domains.fitness, [2, 9]);
  assert.equal(index.value(1, "clone_probability", 2).value, 0.5);
  assert.equal(index.value(1, "fitness", 3, 1).value, 4);
  const legacy = treeRecord();
  legacy.version = 1;
  const old = new AnalysisIndex(legacy);
  assert.equal(old.value(1, "fitness", 2).value, 2);
  assert.equal(old.value(3, "fitness", 2).value, null);
  assert.equal(old.value(1, "distance", 3).value, null);
  legacy.config.algorithm = "graph";
  const graph = new AnalysisIndex(legacy);
  assert.equal(graph.value(0, "fitness", 1).value, null);
});
test("likelihood objectives, zero-token chunks, log probability scale and Unicode byte overlaps", () => {
  const record = treeRecord();
  record.config.objective = "total";
  const total = new AnalysisIndex(structuredClone(record));
  record.config.objective = "mean";
  const mean = new AnalysisIndex(record);
  assert.ok(Math.abs(total.value(3, "reward", 3).value + 0.1) < 1e-9);
  assert.ok(Math.abs(mean.value(3, "reward", 3).value - 0.45) < 1e-9);
  assert.equal(mean.value(6, "reward", 3).value, 0);
  assert.match(probability(-10000), /^\d\.\d{3}e-4343$/);
  assert.notEqual(
    color("probability", -10000, [-20000, 0]),
    color("probability", -20000, [-20000, 0]),
  );
  assert.equal(color("fitness", null, [0, 1]), "#796e83");
  const tokens = [
    { text: "a�", bytes: [97, 240, 159], logprob: -1 },
    { text: "�é", bytes: [153, 130, 195, 169], logprob: -2 },
    { text: "\n", bytes: [10], logprob: -3 },
  ];
  const spans = tokenSpans(tokens);
  assert.equal(spans.map((s) => s.text).join(""), "a🙂é\n");
  assert.deepEqual(
    spans.find((s) => s.text === "🙂").tokens.map((t) => t.index),
    [0, 1],
  );
});
test("decision explanations separate stochastic score, forcing and protections", () => {
  const d = {
    alive: false,
    leaf: false,
    donor_protected: true,
    best_protected: true,
    invalid_donor: true,
    cloned: false,
  };
  assert.match(decisionReason(d), /Dead walker/);
  assert.match(decisionReason(d), /Graph parent protected/);
  assert.match(decisionReason(d), /Retained/);
  assert.match(
    decisionReason({
      alive: true,
      leaf: true,
      cloned: true,
      clone_score: 2,
      draw: 0.5,
    }),
    /score 2 > draw 0.5/,
  );
});
for (const algorithm of ["wave", "graph"])
  test(`v2 actual ${algorithm} pre-clone references and lossless legacy import`, async () => {
    const c = configuration({
      algorithm,
      walkers: 8,
      chunk_tokens: 2,
      sequence_tokens: 8,
      max_walkers: 32,
    });
    const fake = fakeOpenRouter(),
      api = new OpenRouter("secret", { fetchImpl: fake.fetch }),
      record = new Recording(c);
    record.metadata(await api.prepare(c));
    const env = new TokenEnvironment(c, api, record);
    const native = await NativeLlm.create(
      { ...c, dimensions: api.dimensions },
      (r) => env.transition(r),
    );
    try {
      for (let i = 0; i < 4; i++) env.commit(await native.advance());
      const data = importRecording(record.export()),
        index = new AnalysisIndex(data);
      assert.equal(data.version, 3);
      assert.equal(data.engine, "fgllm-1");
      for (const [i, s] of data.snapshots.entries()) {
        const before =
          data.snapshots[i - 1]?.walkers ??
          (algorithm === "wave" ? s.walkers.map(() => ({ node: 0 })) : []);
        assert.equal(s.decisions.length, before.length);
        for (const d of s.decisions) {
          assert.equal(d.evaluated, before[d.slot].node);
          assert.equal(d.companion, before[d.companion_slot].node);
          assert.equal(d.donor, before[d.donor_slot].node);
          assert.equal(d.result, s.walkers[d.slot].node);
          assert.equal(d.cloned, s.walkers[d.slot].cloned);
          assert.ok(
            Math.abs(
              d.fitness -
                d.distance_norm ** c.distance_coef *
                  d.reward_norm ** c.reward_coef *
                  d.other,
            ) < 1e-5,
          );
          const donor = s.decisions[d.donor_slot];
          assert.equal(d.donor_fitness, donor.fitness);
          if (d.cloned && d.result !== d.donor)
            assert.equal(data.nodes[d.result].parent, d.donor);
        }
      }
      const newest = data.nodes.at(-1);
      assert.equal(index.value(newest.id, "fitness", 4).value, null);
      assert.ok(index.origins[newest.id].length > 0);
      assert.ok(index.origins[newest.id].every((d) => d.result === newest.id));
      const legacy = structuredClone(data);
      legacy.version = 1;
      legacy.snapshots.forEach((s) => delete s.decisions);
      assert.deepEqual(
        importRecording(JSON.stringify(legacy)).nodes,
        data.nodes,
      );
      const bad = structuredClone(data);
      bad.snapshots[1].decisions[0].evaluated = bad.nodes.length;
      assert.throws(() => importRecording(JSON.stringify(bad)), /attribution/);
      record.size = RECORDING_LIMIT;
      const boundary = JSON.stringify(record.data.snapshots);
      assert.throws(() => record.commit([], { decisions: [] }));
      assert.equal(JSON.stringify(record.data.snapshots), boundary);
    } finally {
      native.close();
    }
  });
test("10,000-node model construction keeps all exact links and bounded depth filters", () => {
  const record = treeRecord();
  record.nodes = [record.nodes[0]];
  for (let id = 1; id <= 10000; id++)
    record.nodes.push({
      id,
      parent: Math.floor((id - 1) / 4),
      text: `Branch ${id}`,
      tokens: id % 20,
      logp: -id,
      status: 0,
      token_data: [],
    });
  record.snapshots = [
    {
      node_count: record.nodes.length,
      walkers: [{ node: 9999 }, { node: 10000 }],
      decisions: [],
    },
  ];
  const start = performance.now(),
    index = new AnalysisIndex(record),
    model = index.model({ step: 1 });
  assert.equal(model.nodes.length, 10001);
  assert.equal(model.edges.length, 10000);
  assert.ok(performance.now() - start < 3000);
});
