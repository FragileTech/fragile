export const actionGuideStorageKey = "fragile.lab.action-guides";

export class ActionGuideService {
  constructor({ storage, events } = {}) {
    this.storage = storage;
    this.events = events;
    this.targets = new Set();
    this.enabled = false;
    try {
      this.enabled = storage?.getItem(actionGuideStorageKey) === "on";
    } catch {
      /* Optional storage must not block presentation controls. */
    }
    this.onStorage = (event) => {
      if (event.key !== actionGuideStorageKey && event.key !== null) return;
      if (event.storageArea && event.storageArea !== this.storage) return;
      this.update(event.key !== null && event.newValue === "on");
    };
    events?.addEventListener?.("storage", this.onStorage);
  }
  update(enabled) {
    if (this.enabled === enabled) return;
    this.enabled = enabled;
    for (const target of this.targets) target(enabled);
  }
  setEnabled(enabled) {
    const value = Boolean(enabled);
    try {
      this.storage?.setItem(actionGuideStorageKey, value ? "on" : "off");
    } catch {
      /* The current session still honors this choice. */
    }
    this.update(value);
  }
  subscribe(target) {
    this.targets.add(target);
    return () => this.targets.delete(target);
  }
  dispose() {
    this.events?.removeEventListener?.("storage", this.onStorage);
    this.targets.clear();
  }
}

let storage;
try {
  storage = globalThis.localStorage;
} catch {
  /* Storage access itself can fail in private browsing. */
}
export const labActionGuides = new ActionGuideService({
  storage,
  events: globalThis.window,
});
export function installActionGuideControls() {
  const inputs = [...document.querySelectorAll('input[name="action-guides"]')];
  const check = (enabled) => {
    for (const input of inputs) input.checked = enabled;
  };
  const change = (event) => labActionGuides.setEnabled(event.target.checked);
  check(labActionGuides.enabled);
  const unsubscribe = labActionGuides.subscribe(check);
  for (const input of inputs) input.addEventListener("change", change);
  return () => {
    unsubscribe();
    for (const input of inputs) input.removeEventListener("change", change);
  };
}
