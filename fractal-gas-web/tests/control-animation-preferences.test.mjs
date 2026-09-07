import test from "node:test";
import assert from "node:assert/strict";
import {
  AnimationService,
  animationStorageKey,
} from "../web/lab/animations.js";

function setup(value, reduced = false) {
  const values = new Map(value == null ? [] : [[animationStorageKey, value]]);
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
  const media = new EventTarget();
  media.matches = reduced;
  const events = new EventTarget();
  const service = new AnimationService({ storage, media, events });
  return {
    service,
    values,
    storage,
    media,
    events,
    reduce(value) {
      media.matches = value;
      media.dispatchEvent(new Event("change"));
    },
    external(value, key = animationStorageKey, storageArea = storage) {
      events.dispatchEvent(
        Object.assign(new Event("storage"), {
          key,
          newValue: value,
          storageArea,
        }),
      );
    },
  };
}

test("defaults respect reduced motion and follow it until an explicit choice", () => {
  const normal = setup();
  assert.equal(normal.service.enabled, true);
  const reduced = setup(undefined, true);
  assert.equal(reduced.service.enabled, false);
  reduced.reduce(false);
  assert.equal(reduced.service.enabled, true);
  reduced.service.setEnabled(true);
  reduced.reduce(true);
  assert.equal(reduced.service.enabled, true);
  assert.equal(reduced.values.get(animationStorageKey), "on");
});

test("stored choices override OS preferences and invalid choices use OS default", () => {
  assert.equal(setup("on", true).service.enabled, true);
  assert.equal(setup("off", false).service.enabled, false);
  assert.equal(setup("invalid", true).service.enabled, false);
});

test("subscribers receive changes once with no initial callback and unsubscribe", () => {
  const { service, values } = setup();
  const changes = [];
  const unsubscribe = service.subscribe((enabled) => changes.push(enabled));
  assert.deepEqual(changes, []);
  service.setEnabled(false);
  service.setEnabled(false);
  assert.deepEqual(changes, [false]);
  assert.equal(values.get(animationStorageKey), "off");
  unsubscribe();
  service.setEnabled(true);
  assert.deepEqual(changes, [false]);
});

test("storage failures preserve working session controls and explicit choices", () => {
  const media = new EventTarget();
  media.matches = true;
  const service = new AnimationService({
    media,
    storage: {
      getItem() {
        throw new Error("blocked");
      },
      setItem() {
        throw new Error("full");
      },
    },
  });
  assert.equal(service.enabled, false);
  assert.doesNotThrow(() => service.setEnabled(true));
  media.dispatchEvent(new Event("change"));
  assert.equal(service.enabled, true);
});

test("other tabs synchronize choices, deletion and clear restore OS default", () => {
  const fixture = setup(undefined, true);
  fixture.external("on");
  assert.equal(fixture.service.enabled, true);
  fixture.reduce(true);
  assert.equal(fixture.service.enabled, true);
  fixture.external("off", "unrelated");
  fixture.external("off", animationStorageKey, {});
  assert.equal(fixture.service.enabled, true);
  fixture.external(null);
  assert.equal(fixture.service.enabled, false);
  fixture.reduce(false);
  assert.equal(fixture.service.enabled, true);
  fixture.external("off");
  fixture.external(null, null);
  assert.equal(fixture.service.enabled, true);
});

test("dispose removes preference listeners", () => {
  const fixture = setup();
  fixture.service.dispose();
  fixture.external("off");
  fixture.reduce(true);
  assert.equal(fixture.service.enabled, true);
});
