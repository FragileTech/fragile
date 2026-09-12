// Scientific state and all measurements belong to the compiled Rust session.
export function renderSnapshot(raw) {
  const result = raw.result;
  const charts = result
    ? result.plots.map((p) => ({
        title: p.title,
        xLabel: p.x_label,
        yLabel: p.y_label,
        series: p.series.map((s) => ({
          name: s.name,
          points: s.points,
          style:
            s.kind === "bars"
              ? "bars"
              : ["scatter", "points"].includes(s.kind)
                ? "points"
                : "line",
        })),
      }))
    : [];
  if (raw.positions?.length)
    charts.push({
      title: "Executed walker positions",
      xLabel: "Position 1",
      yLabel: "Position 2",
      series: [
        {
          name: "Recorded eligible walkers",
          style: "points",
          points: raw.positions,
        },
      ],
    });
  return {
    step: raw.step,
    time: raw.step,
    done: raw.done,
    result,
    charts,
    metrics:
      result?.metrics.map((m) => ({
        ...m,
        value: m.value === null ? "Inconclusive" : m.value,
      })) || [],
    message: result
      ? [result.model, ...result.notes].join(" · ")
      : "Running the gas to collect the required measurement window…",
    scene: result?.details.scene,
    table: {
      columns: ["Run", "Completed steps", "Required steps"],
      rows: (raw.run_steps || []).map((n, i) => [i, n, raw.budgets[i]]),
    },
  };
}
export async function createRunModel({ id, params, seed, engine }) {
  const session = await engine.lectureCreate({
    id,
    parameters: params,
    seed,
    steps: 96,
  });
  let raw;
  try {
    raw = await session.advance(8);
  } catch (error) {
    session.free();
    throw error;
  }
  return {
    async step() {
      raw = await session.advance(8);
    },
    snapshot() {
      return renderSnapshot(raw);
    },
    archive() {
      return session.evidence();
    },
    checkpoint() {
      return session.checkpoint();
    },
    dispose() {
      session.free();
    },
  };
}
