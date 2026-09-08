import test from "node:test";
import assert from "node:assert/strict";
import * as T from "../web/lab/vendor/three.module.js";
import { CargoReadout } from "../web/lab/visuals/cargo-readout.js";

class Element {
  constructor() {
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.attributes = {};
    this.textContent = "";
  }
  append(...nodes) {
    this.children.push(...nodes);
  }
  setAttribute(key, value) {
    this.attributes[key] = value;
  }
  remove() {
    this.removed = true;
  }
}
function fixture(count = 1) {
  const doc = {
    head: new Element(),
    getElementById(id) {
      return this.head.children.find((node) => node.id === id);
    },
    createElement() {
      return new Element();
    },
  };
  const canvas = { ownerDocument: doc, parentElement: new Element() };
  const readout = new CargoReadout(canvas);
  readout.resize(1200, 800);
  const camera = new T.OrthographicCamera(-12, 12, 8, -8, 0.1, 100);
  camera.position.z = 20;
  camera.updateMatrixWorld();
  const frame = new T.Group();
  const layer = {
    active: Array(count).fill(true),
    inView: Array(count).fill(true),
  };
  const entries = Array.from({ length: count }, (_, body) => ({
    body,
    amount: 2.5,
    capacity: 5,
    delivered: 1.25,
    collected: 3.75,
    status: "Loading",
    loadingAmount: 0.5,
    anchor: new T.Vector3(
      (body % 5) * 4 - 8,
      Math.floor(body / 5) * 2.3 - 5,
      0,
    ),
  }));
  const update = (selected, followed, style = "futuristic") =>
    readout.update(entries, layer, camera, frame, selected, followed, style);
  return { doc, canvas, readout, entries, layer, camera, frame, update };
}
test("cargo chips show fractional held, picked and delivered amounts with static loading", () => {
  const { readout, entries, update } = fixture();
  update();
  const label = readout.labels[0];
  assert.equal(label.amount.textContent, "2.5 / 5");
  assert.equal(label.status.textContent, "Loading +0.5");
  assert.equal(label.total.textContent, "Picked 3.75 · Delivered 1.25");
  assert.equal(label.fill.style.transform, "scaleX(0.5)");
  entries[0].amount = 5;
  entries[0].status = "Full";
  update(undefined, undefined, "steampunk");
  assert.equal(label.amount.textContent, "5 / 5");
  assert.equal(label.status.textContent, "Full");
  assert.equal(label.fill.style.transform, "scaleX(1)");
  assert.equal(readout.root.dataset.style, "steampunk");
  assert.equal(readout.root.attributes["aria-live"], undefined);
});
test("label pool is capped, reused and resolves overlap in selected then followed order", () => {
  const { readout, entries, update } = fixture(20);
  update(19, 18);
  assert.equal(readout.labels.length, 16);
  assert.equal(readout.labels.filter((label) => !label.node.hidden).length, 16);
  assert.equal(readout.labels[0].body, 19);
  assert.equal(readout.labels[1].body, 18);
  const nodes = readout.labels.map((label) => label.node);
  for (const entry of entries) entry.anchor.set(0, 0, 0);
  update(6, 3);
  assert.equal(readout.labels[0].body, 6);
  assert.equal(readout.labels.filter((label) => !label.node.hidden).length, 1);
  assert.deepEqual(
    readout.labels.map((label) => label.node),
    nodes,
  );
  update(undefined, 3);
  assert.equal(readout.labels[0].body, 3);
  update();
  assert.equal(readout.labels[0].body, 0);
});
test("inactive, offscreen and absent cargo hide immediately", () => {
  const { readout, entries, layer, update } = fixture();
  update();
  assert.equal(readout.labels[0].node.hidden, false);
  layer.active[0] = false;
  update();
  assert.equal(readout.labels[0].node.hidden, true);
  layer.active[0] = true;
  layer.inView[0] = false;
  update();
  assert.equal(readout.labels[0].node.hidden, true);
  layer.inView[0] = true;
  entries[0].anchor.x = 100;
  update();
  assert.equal(readout.labels[0].node.hidden, true);
  entries[0].anchor.x = 0;
  update();
  assert.equal(readout.labels[0].node.hidden, false);
  entries.length = 0;
  update();
  assert.equal(readout.labels[0].node.hidden, true);
});
test("screen projection applies flight coordinate frame and camera movement without state changes", () => {
  const { readout, entries, frame, camera, update } = fixture();
  entries[0].anchor.set(2, 3, 0);
  update();
  const x = readout.labels[0].x,
    y = readout.labels[0].y;
  frame.rotation.x = Math.PI / 2;
  camera.up.set(0, 0, 1);
  camera.position.set(0, -20, 0);
  camera.lookAt(0, 0, 0);
  update();
  assert.equal(readout.labels[0].x, x);
  assert.equal(readout.labels[0].y, y);
  camera.position.x += 2;
  camera.lookAt(2, 0, 0);
  update();
  assert.equal(readout.labels[0].x, x - 100);
});
test("readout cleanup removes its overlay and shares one stylesheet across viewports", () => {
  const { readout, canvas, doc, update } = fixture();
  const second = new CargoReadout(canvas);
  assert.equal(doc.head.children.length, 1);
  update();
  readout.clear();
  assert.ok(readout.labels.every((label) => label.node.hidden));
  readout.dispose();
  second.dispose();
  assert.equal(readout.root.removed, true);
  assert.equal(second.root.removed, true);
});
