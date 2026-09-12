import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import init, {
  BrowserGas,
  default_config,
} from "../../web/euclidean-gas/engine/cpu/gas.js";
import {
  demos,
  donorProbabilities,
  enumerateAssignments,
  probabilityField,
  conditionalPositionLaw,
} from "../../web/euclidean-gas/lecture/foundations.js";
import { positions, velocities } from "../../web/euclidean-gas/lecture/math.js";
await init({
  module_or_path: await readFile(
    new URL("../../web/euclidean-gas/engine/cpu/gas_bg.wasm", import.meta.url),
  ),
});
const engine = {
  defaults: async () => JSON.parse(default_config()),
  create: (c) => BrowserGas.create(JSON.stringify(c)),
};
const defaultParams = (d) =>
  Object.fromEntries(d.controls.map((c) => [c.key, c.value]));
function finiteCharts(s) {
  assert.ok(s.charts.length > 0);
  for (const c of s.charts)
    for (const series of c.series ?? [])
      for (const pair of series.points) {
        assert.equal(pair.length, 2);
        assert.ok(pair.every(Number.isFinite), `${c.title}: ${pair}`);
      }
  for (const m of s.metrics ?? [])
    if (typeof m.value === "number")
      assert.ok(Number.isFinite(m.value), m.label);
}
for (const d of demos)
  test(`${d.id} has finite real computations and deterministic reset`, async () => {
    let a, b;
    try {
      const args = { params: defaultParams(d), seed: 7, engine };
      a = await d.create(args);
      finiteCharts(a.snapshot());
      await a.step();
      await a.step();
      const result = a.snapshot();
      finiteCharts(result);
      assert.deepEqual(a.snapshot(), result, "snapshot is pure");
      b = await d.create(args);
      await b.step();
      await b.step();
      assert.deepEqual(b.snapshot(), result);
    } finally {
      a?.dispose();
      b?.dispose();
    }
  });
for (const id of ["I-02", "I-04", "I-10"])
  test(`${id} non-default controls run`, async () => {
    const d = demos.find((x) => x.id === id),
      params = {
        ...defaultParams(d),
        ...(id === "I-02"
          ? { walkers: 5, law: "greedy", count: 4 }
          : id === "I-04"
            ? { survivors: 1, boundary: "periodic" }
            : { velocityWeight: 0 }),
      };
    const m = await d.create({ params, seed: 21, engine });
    try {
      await m.step();
      finiteCharts(m.snapshot());
      if (id === "I-04")
        assert.equal(
          m.snapshot().metrics.find((v) => v.label === "Eligible slots").value,
          16,
        );
    } finally {
      m.dispose();
    }
  });
test("zero survivors is a terminal retained frame, not an unhandled failure", async () => {
  const d = demos.find((d) => d.id === "I-04"),
    m = await d.create({
      params: { ...defaultParams(d), survivors: 0 },
      seed: 7,
      engine,
    });
  try {
    await m.step();
    assert.equal(m.snapshot().done, true);
    assert.equal(m.snapshot().metrics[0].value, 0);
  } finally {
    m.dispose();
  }
});
test("exact small laws normalize and mutual assignments are reciprocal", () => {
  const x = [
    [-1, 0],
    [-0.5, 0.1],
    [0.5, 0],
    [1, 0.1],
    [1.5, 0],
  ];
  for (const law of ["independent", "mutual", "greedy"]) {
    let total = 0;
    enumerateAssignments(x, 0.5, law, (a, w) => {
      total += w;
      if (law !== "independent")
        for (let i = 0; i < a.length; i++) assert.equal(a[a[i]], i);
    });
    assert.ok(Math.abs(total - 1) < 1e-10, law);
  }
  const f = probabilityField(x);
  assert.ok(
    Math.abs(f.persistence + f.joint.reduce((a, b) => a + b) - 1) < 1e-12,
  );
  assert.ok(f.joint.every((x) => x >= 0));
  const q = donorProbabilities(x, 0.2);
  assert.ok(
    q.every(
      (row, i) =>
        row[i] === 0 && Math.abs(row.reduce((a, b) => a + b) - 1) < 1e-12,
    ),
  );
});
test("fixture replacement validates before commit and refreshes translated rewards", async () => {
  const c = await engine.defaults();
  c.walkers = 4;
  c.benchmark = "quadratic";
  c.reward_shift = [1, 0];
  c.potential = "quadratic";
  c.gas.precision = "f64";
  c.gas.boundary = { kind: "unbounded" };
  const r = await engine.create(c);
  try {
    const f = await r.set_population(
      JSON.stringify({
        positions: [
          [1, 0],
          [2, 0],
          [0, 0],
          [1, 1],
        ],
        alive: [true, true, false, true],
      }),
    );
    assert.deepEqual(f.population.rewards.raw, [0, 0.5, 0.5, 0.5]);
    assert.equal(f.population.validity[2].terminated, true);
    const before = r.checkpoint();
    await assert.rejects(
      r.set_population(JSON.stringify({ positions: [[0, 0]] })),
    );
    assert.deepEqual(r.checkpoint(), before);
  } finally {
    r.free();
  }
});
test("BAOAB trace does not change the committed run and final force uses post-A position", async () => {
  const c = await engine.defaults();
  c.walkers = 4;
  c.benchmark = "quadratic";
  c.gas.precision = "f64";
  c.gas.boundary = { kind: "unbounded" };
  c.gas.fitness.reward_exponent = 0;
  c.gas.fitness.diversity_exponent = 0;
  c.gas.kinetic = {
    integrator: {
      kind: "baoab",
      positions: "positions",
      velocities: "velocities",
      dt: 0.04,
      friction: 1,
    },
    noise: {
      innovation: "gaussian",
      geometry: { kind: "isotropic", scale: { kind: "constant", values: [0] } },
    },
  };
  const a = await engine.create(c),
    b = await engine.create(c);
  try {
    a.set_trace(true);
    const fa = await a.step(1),
      fb = await b.step(1);
    assert.deepEqual(fa.population, fb.population);
    assert.deepEqual(fa.report, fb.report);
    assert.deepEqual(
      fa.trace.map((t) => t.stage),
      [
        "pre_clone",
        "literal_clone",
        "post_transform",
        "B1",
        "A1",
        "O",
        "A2",
        "B2",
        "post_kinetic",
      ],
    );
    const stage = (name) => fa.trace.find((t) => t.stage === name),
      A = positions(stage("A2")),
      before = velocities(stage("A2")),
      after = velocities(stage("B2"));
    after.forEach((row, i) =>
      row.forEach((v, j) =>
        assert.ok(Math.abs(v - (before[i][j] - 0.02 * A[i][j])) < 1e-12),
      ),
    );
    assert.deepEqual(
      a.checkpoint(),
      b.checkpoint(),
      "traces are not serialized into checkpoint or RNG",
    );
  } finally {
    a.free();
    b.free();
  }
});
test("restitution trace recovers pair momentum and a-squared relative energy", async () => {
  const d = demos.find((d) => d.id === "I-09");
  for (const restitution of [0, 0.5, 1]) {
    const m = await d.create({
      params: { ...defaultParams(d), restitution },
      seed: 7,
      engine,
    });
    try {
      const s = m.snapshot();
      assert.ok(s.metrics[0].value < 1e-12);
      assert.ok(Math.abs(s.metrics[1].value - s.metrics[2].value) < 1e-12);
    } finally {
      m.dispose();
    }
  }
});

test("every declared Part I control endpoint admits a finite experiment", async () => {
  for (const d of demos)
    for (const control of d.controls) {
      const values =
        control.type === "range"
          ? [control.min, control.max]
          : control.options.map((o) => o.value);
      for (const value of values) {
        if (value === control.value) continue;
        let model;
        try {
          model = await d.create({
            params: { ...defaultParams(d), [control.key]: value },
            seed: 13,
            engine,
          });
          await model.step();
          finiteCharts(model.snapshot());
        } catch (e) {
          throw new Error(`${d.id} ${control.key}=${value}: ${e}`, {
            cause: e,
          });
        } finally {
          model?.dispose();
        }
      }
    }
});

test("exact conditional accepted-copy field agrees with independent engine draws", async () => {
  const x = [
      [-1, 0.2],
      [-0.5, 0.1],
      [0.5, 0],
      [1.2, 0.1],
    ],
    n = 1500;
  const c = await engine.defaults();
  c.walkers = 4;
  c.benchmark = "sphere";
  c.gas.precision = "f64";
  c.gas.boundary = { kind: "unbounded" };
  c.gas.kinetic.integrator.amplitude = 0;
  c.gas.distance_donors.kernel = { kind: "gaussian", width: 0.7 };
  c.gas.cloning_donors.kernel = { kind: "gaussian", width: 0.8 };
  const predicted = probabilityField(x, { width: 0.7, cloneWidth: 0.8 }),
    counts = Array(4).fill(0),
    run = await engine.create(c);
  try {
    for (let i = 0; i < n; i++) {
      await run.set_population(JSON.stringify({ positions: x }));
      const { report: r } = await run.step(1);
      if (r.clone_plan.choices[0].accepted)
        counts[r.clone_plan.sources[r.cloning_companions.indices[0]].slot]++;
    }
    predicted.joint.forEach((p, j) =>
      assert.ok(
        Math.abs(counts[j] / n - p) < 0.045,
        `donor ${j}: observed ${counts[j] / n}, predicted ${p}`,
      ),
    );
  } finally {
    run.free();
  }
});

const teachingCloud = (n) =>
  Array.from({ length: n }, (_, i) => [
    (i < n / 2 ? -1 : 1) + 0.24 * Math.cos(i * 2.3),
    0.48 * Math.sin(i * 1.7),
  ]);

test("I-03 local default includes self and reproduces the chapter weighted sums", async () => {
  const d = demos.find((d) => d.id === "I-03"),
    p = { ...defaultParams(d), stats: "local", outlier: 1.5 };
  const x = Array.from({ length: 8 }, (_, i) => [
    Math.cos((i * Math.PI) / 4) * (i === 7 ? 1.5 : 1),
    Math.sin((i * Math.PI) / 4) * (i === 7 ? 1.5 : 1),
  ]);
  const raw = x.map((q) => -0.5 * q.reduce((s, v) => s + v * v, 0));
  for (const self of ["include", "exclude"]) {
    const model = await d.create({ params: { ...p, self }, seed: 7, engine });
    try {
      const rows = model.snapshot().table.rows;
      x.forEach((q, i) => {
        const weights = x.map((r, j) =>
          self === "exclude" && i === j
            ? 0
            : Math.exp(-q.reduce((s, v, k) => s + (v - r[k]) ** 2, 0) / 0.5),
        );
        const expected =
          weights.reduce((s, w, j) => s + w * raw[j], 0) /
          weights.reduce((a, b) => a + b, 0);
        assert.ok(Math.abs(rows[i][4] - expected) < 1e-12);
      });
      assert.ok(
        Math.abs(
          rows[7][4] - (self === "include" ? -1.0155461394010827 : -0.5),
        ) < 1e-12,
      );
      assert.match(
        model.snapshot().message,
        self === "include"
          ? /include the selected walker/
          : /exclude the selected walker/,
      );
    } finally {
      model.dispose();
    }
  }
  assert.equal(defaultParams(d).self, "include");
});

test("I-08 conditions probabilities on eligible slots and handles singleton/revival/extinction", async () => {
  const x = teachingCloud(5),
    d = demos.find((d) => d.id === "I-08");
  for (const [boundary, recipient, mode, eligible] of [
    [1.1, 0, "voluntary", [0, 2, 4]],
    [0.8, 0, "voluntary", [0, 4]],
    [0.76, 0, "voluntary", [0]],
    [0.76, 1, "revival", [0]],
    [0.4, 0, "extinction", []],
  ]) {
    const law = conditionalPositionLaw(x, recipient, boundary);
    assert.equal(law.mode, mode);
    assert.deepEqual(law.slots, eligible);
    law.joint.forEach((w, j) => {
      if (!eligible.includes(j)) assert.equal(w, 0);
    });
    if (eligible.length)
      assert.ok(
        Math.abs(law.persistence + law.joint.reduce((a, b) => a + b) - 1) <
          1e-12,
      );
    if (mode === "voluntary" && eligible.length === 1)
      assert.equal(law.persistence, 1);
    if (mode === "revival") assert.equal(law.joint[0], 1);
    const model = await d.create({
      params: { ...defaultParams(d), boundary, recipient },
      seed: 7,
      engine,
    });
    try {
      await model.step();
      const s = model.snapshot();
      finiteCharts(s);
      assert.equal(s.done, mode === "extinction");
      assert.equal(
        s.metrics.find((m) => m.label === "Eligible donors").value,
        eligible.length,
      );
    } finally {
      model.dispose();
    }
  }
});

test("I-08 alive-conditioned law agrees with actual WASM clone choices", async () => {
  const x = teachingCloud(5);
  for (const [boundary, recipient] of [
    [1.1, 0],
    [0.8, 0],
    [0.76, 0],
    [0.76, 1],
  ]) {
    const c = await engine.defaults();
    c.walkers = 5;
    c.benchmark = "sphere";
    c.gas.precision = "f64";
    c.initial_lower = -0.1;
    c.initial_upper = 0.1;
    c.gas.boundary = {
      kind: "absorbing_box",
      field: "positions",
      domain: { lower: [-boundary, -boundary], upper: [boundary, boundary] },
    };
    c.gas.kinetic.integrator.amplitude = 0;
    const run = await engine.create(c),
      law = conditionalPositionLaw(x, recipient, boundary),
      counts = Array(5).fill(0),
      draws = 3000;
    try {
      for (let n = 0; n < draws; n++) {
        await run.set_population(JSON.stringify({ positions: x }));
        const { report } = await run.step(1),
          choice = report.clone_plan.choices[recipient];
        if (choice.accepted)
          counts[report.clone_plan.sources[choice.donors[0].pool_index].slot]++;
      }
      counts.forEach((n, j) => {
        if (!law.alive[j]) assert.equal(n, 0);
        const p = law.joint[j],
          error = Math.abs(n / draws - p),
          tolerance = 5 * Math.sqrt((p * (1 - p)) / draws) + 1 / draws;
        assert.ok(
          error <= tolerance,
          `boundary ${boundary}, recipient ${recipient}, donor ${j}: ${n / draws} vs ${p}`,
        );
      });
    } finally {
      run.free();
    }
  }
});

test("I-08 cumulative histogram retains every draw and separates mass from density", async () => {
  const d = demos.find((d) => d.id === "I-08"),
    m = await d.create({
      params: { ...defaultParams(d), boundary: 2, jitter: 0.3 },
      seed: 7,
      engine,
    });
  try {
    for (let i = 0; i < 100; i++) await m.step();
    const s = m.snapshot(),
      count = s.metrics.find(
        (m) => m.label === "Cumulative histogram draws",
      ).value,
      outside = s.metrics.find(
        (m) => m.label === "Samples outside histogram viewport",
      ).value;
    assert.equal(count, 12800);
    assert.equal(
      s.metrics.find((m) => m.label === "Displayed recent positions").value,
      400,
    );
    const hist = s.charts.find((c) =>
      c.title.startsWith("Cumulative post-clone histogram"),
    );
    assert.ok(
      Math.abs(
        hist.series[0].points.reduce((s, p) => s + p[1], 0) +
          outside / count -
          1,
      ) < 1e-12,
    );
    assert.ok(
      hist.series[0].points.every(
        (p) => Math.abs(p[1] * count - Math.round(p[1] * count)) < 1e-8,
      ),
    );
    assert.equal(
      s.charts.find((c) => c.title === "Atomic position components").yLabel,
      "Probability mass",
    );
    assert.equal(
      s.charts.find((c) => c.title === "Continuous offspring components")
        .yLabel,
      "Probability density",
    );
    assert.match(s.message, /Position-only/);
    assert.ok(!d.controls.some((c) => c.key === "temperature"));
    assert.ok(
      !s.charts.some(
        (c) => c.title.includes("phase space") || c.title.includes("Kinetic"),
      ),
    );
  } finally {
    m.dispose();
  }
});

test("I-06 exposes independently seeded multiwell residence and its ensemble mean", async () => {
  const d = demos.find((d) => d.id === "I-06"),
    model = await d.create({
      params: { ...defaultParams(d), landscape: "multiwell" },
      seed: 7,
      engine,
    });
  try {
    for (let i = 0; i < 1000; i++) await model.step();
    const s = model.snapshot(),
      chart = s.charts.find((c) => c.title.startsWith("Independent runs"));
    assert.deepEqual(
      s.table.rows.map((r) => r[0]),
      [7, 19, 43],
    );
    assert.equal(chart.series.length, 4);
    assert.ok(chart.series.every((s) => s.points.length === 400));
    chart.series[3].points.forEach((p, i) =>
      assert.ok(
        Math.abs(
          p[1] -
            chart.series
              .slice(0, 3)
              .reduce((sum, s) => sum + s.points[i][1], 0) /
              3,
        ) < 1e-12,
      ),
    );
    assert.ok(Math.abs(s.table.rows[0][1]) < 0.1);
    assert.ok(s.table.rows[1][1] > 0.8);
    assert.ok(
      s.metrics.find((m) => m.label === "Between-run mean range").value > 0.7,
    );
  } finally {
    model.dispose();
  }
});

test("printed BAOAB pseudocode executes the same force/drift schedule and OU units", async () => {
  const document = fileURLToPath(
    new URL(
      "../../../docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md",
      import.meta.url,
    ),
  );
  // Execute the exact Python printed in the chapter. A scalar array wrapper and
  // standard-library math supply its NumPy interface without an extra dependency.
  const code = String.raw`
import json, math, pathlib, sys, types
text=pathlib.Path(sys.argv[1]).read_text()
source=text[text.index('def Psi_kin_BAOAB('):text.index('def run_euclidean_gas_step(')]
class Array(float):
    shape=(1,1)
    def __add__(self,v): return Array(float(self)+float(v))
    __radd__=__add__
    def __sub__(self,v): return Array(float(self)-float(v))
    def __rsub__(self,v): return Array(float(v)-float(self))
    def __mul__(self,v): return Array(float(self)*float(v))
    __rmul__=__mul__
    def __truediv__(self,v): return Array(float(self)/float(v))
    def __neg__(self): return Array(-float(self))
innovation=0
np=types.SimpleNamespace(exp=math.exp,expm1=math.expm1,sqrt=math.sqrt,random=types.SimpleNamespace(randn=lambda *shape:Array(innovation)))
force_calls=[]
def harmonic(x):
    force_calls.append(float(x))
    return -x
scope=dict(np=np,F=harmonic,u=lambda x:Array(0),psi_v=lambda v,cap:v)
exec(source,scope)
step=scope['Psi_kin_BAOAB']
p=dict(tau=.04,gamma_fric=1,m=1,sigma_v=math.sqrt(.8),sigma_x=0,V_alg=None)
x,v=step(Array(1),Array(.5),p)
result=dict(deterministic=[x,v],force_calls=force_calls.copy(),noise=[])
scope['F']=lambda x:Array(0)
for gamma in [.1,1,5,0]:
    p.update(gamma_fric=gamma,sigma_v=math.sqrt(2*gamma*.4) if gamma else .7)
    innovation=1
    xp,vp=step(Array(0),Array(0),p)
    innovation=-1
    xm,vm=step(Array(0),Array(0),p)
    result['noise'].append(dict(gamma=gamma,variance=((vp-vm)/2)**2))
print(json.dumps(result))
`;
  const result = spawnSync("python3", ["-c", code, document], {
    encoding: "utf8",
  });
  assert.equal(result.status, 0, result.stderr);
  const printed = JSON.parse(result.stdout);
  assert.ok(Math.abs(printed.deterministic[0] - 1.0188235786158624) < 1e-12);
  assert.ok(Math.abs(printed.deterministic[1] - 0.44080245922079786) < 1e-12);
  assert.deepEqual(printed.force_calls, [1, printed.deterministic[0]]);
  printed.noise.forEach(({ gamma, variance }) =>
    assert.ok(
      Math.abs(
        variance -
          (gamma ? 0.4 * (1 - Math.exp(-2 * gamma * 0.04)) : 0.7 * 0.7 * 0.04),
      ) < 1e-12,
    ),
  );
  const c = await engine.defaults();
  c.walkers = 2;
  c.benchmark = "quadratic";
  c.gas.precision = "f64";
  c.gas.boundary = { kind: "unbounded" };
  c.gas.fitness.reward_exponent = c.gas.fitness.diversity_exponent = 0;
  c.gas.kinetic = {
    integrator: {
      kind: "baoab",
      positions: "positions",
      velocities: "velocities",
      dt: 0.04,
      friction: 1,
    },
    noise: {
      innovation: "gaussian",
      geometry: { kind: "isotropic", scale: { kind: "constant", values: [0] } },
    },
  };
  const run = await engine.create(c);
  try {
    await run.set_population(
      JSON.stringify({
        positions: [
          [1, 0],
          [1, 0],
        ],
        velocities: [
          [0.5, 0],
          [0.5, 0],
        ],
      }),
    );
    const f = await run.step(1);
    assert.ok(Math.abs(positions(f)[0][0] - printed.deterministic[0]) < 1e-12);
    assert.ok(Math.abs(velocities(f)[0][0] - printed.deterministic[1]) < 1e-12);
  } finally {
    run.free();
  }
});
