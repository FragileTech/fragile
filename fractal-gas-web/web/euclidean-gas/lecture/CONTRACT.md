# Lecture experiment modules

Each family module exports `demos`, an array of descriptors. Modules execute inside
the lecture Web Worker and must not access the DOM. Every demo performs actual
computations; reference-model outputs identify that model in their caption.

```javascript
export const demos = [{
  id: "I-01", part: "I", title: "One complete update",
  kind: "WASM experiment", // or "Mathematical model", or "WASM + reference"
  question: "Which operation moved this walker?",
  prediction: "What the reader should expect to see.",
  explanation: "A concise connection to the chapter.",
  controls: [
    { key: "walkers", label: "Walkers", type: "select", value: 64,
      options: [{ value: 16, label: "16" }, { value: 64, label: "64" }] },
    { key: "gamma", label: "Friction", type: "range", value: 1,
      min: 0.1, max: 3, step: 0.1 },
  ],
  async create({ params, seed, engine }) {
    // engine.defaults(): Promise<full RunConfig>
    // engine.create(fullConfig): Promise<BrowserGas>
    // BrowserGas uses the existing native API: snapshot(), await step(1), free().
    // Optional engine extensions are coordinated with the foundations owner.
    return {
      async step() { /* advance one bounded unit of work */ },
      snapshot() {
        return {
          step: 0, time: 0,
          charts: [{
            title: "Position", xLabel: "x₁", yLabel: "x₂",
            series: [{ name: "Walkers", points: [[0, 1], [1, 0]], style: "points" }],
            // style: "line" (default), "points", or "bars"; optional dashed/color.
            // Optional xDomain:[lo,hi], yDomain:[lo,hi], xScale/yScale:"log".
            // Optional segments:[[[x1,y1],[x2,y2]], ...].
            // A matrix plot may use matrix:[[...],...], rowLabels, columnLabels
            // instead of series. Optional colorDomain:[lo,hi].
          }],
          metrics: [{ label: "Alive", value: 64, unit: "walkers" }],
          message: "A short observation or measurement-stage label.",
          // Optional table: { columns:["Walker","Fitness"], rows:[[0,1.2]] }.
          // Optional done:true to stop animation at a finite experiment horizon.
        };
      },
      dispose() { /* free all WASM runs owned by the model */ },
    };
  },
}];
```

The host supplies each control's declared default, with the selected seed. Every
scientific control change resets the experiment. Use bounded histories (up to
400 samples), bounded ensembles, and deterministic seeded randomness. Tests can
import modules in Node, supplying a real WASM `engine` adapter. Implement useful
math helpers privately in a family module when they are specific to that family.

Shared `math.js` exports (implemented by the host owner):

- `rng(seed)`: function returning uniform [0,1), with `.normal()` for N(0,1).
- `linspace(a,b,n=101)`, `mean(xs)`, `variance(xs)`, `clamp(x,lo,hi)`.
- `histogram(xs,lo,hi,bins=32)`: `[[binCenter,density],...]` using total input
  sample count, so mass outside the plotted range is retained as missing mass.
- `normalPDF(x,mu=0,sigma=1)`, `covariance2(points)` -> `[[xx,xy],[xy,yy]]`.
- `line(name,points,options={})` and `scatter(name,points,options={})`.
- `positions(frame)` and `velocities(frame)` -> arrays of row arrays from the
  corresponding observation fields; absent velocities -> empty array.
- `pushBounded(array,value,limit=400)`: append and drop oldest if needed.

Owners: foundations.js (Part I and Rust/WASM diagnostic exposure),
convergence.js (Parts II–III), entropy.js (Part IV), and parent (shared host,
math/plots/worker, integration and document embedding). Keep changes in owned
files and coordinate any shared API extension before editing another file.
