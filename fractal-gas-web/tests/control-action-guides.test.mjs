import test from "node:test";
import assert from "node:assert/strict";
import {
  ActionGuideService,
  actionGuideStorageKey,
} from "../web/lab/action-guides.js";

function fixture(value) {
  const data = new Map(value ? [[actionGuideStorageKey, value]] : []);
  const storage = {
    getItem: (key) => data.get(key),
    setItem: (key, value) => data.set(key, value),
  };
  const events = new EventTarget();
  const service = new ActionGuideService({ storage, events });
  return { service, data, events, storage };
}

test("action guides default off and persist explicit choices", () => {
  const { service, data } = fixture();
  assert.equal(service.enabled, false);
  service.setEnabled(true);
  assert.equal(data.get(actionGuideStorageKey), "on");
  assert.equal(fixture("on").service.enabled, true);
  assert.equal(fixture("invalid").service.enabled, false);
  service.setEnabled(false);
  assert.equal(data.get(actionGuideStorageKey), "off");
});
test("guide subscriptions have no initial callback and only notify changes", () => {
  const { service } = fixture();
  const values = [];
  const unsubscribe = service.subscribe((value) => values.push(value));
  assert.deepEqual(values, []);
  service.setEnabled(true);
  service.setEnabled(true);
  unsubscribe();
  service.setEnabled(false);
  assert.deepEqual(values, [true]);
});
test("guide choices synchronize other tabs and clear restores off", () => {
  const { service, storage, events } = fixture();
  const update = (key, newValue, storageArea = storage) =>
    events.dispatchEvent(
      Object.assign(new Event("storage"), { key, newValue, storageArea }),
    );
  update(actionGuideStorageKey, "on");
  assert.equal(service.enabled, true);
  update("unrelated", "off");
  update(actionGuideStorageKey, "off", {});
  assert.equal(service.enabled, true);
  update(null, null);
  assert.equal(service.enabled, false);
  service.dispose();
  update(actionGuideStorageKey, "on");
  assert.equal(service.enabled, false);
});
test("blocked storage leaves guide controls usable", () => {
  const service = new ActionGuideService({
    storage: {
      getItem() {
        throw new Error("blocked");
      },
      setItem() {
        throw new Error("blocked");
      },
    },
  });
  assert.equal(service.enabled, false);
  assert.doesNotThrow(() => service.setEnabled(true));
  assert.equal(service.enabled, true);
});
