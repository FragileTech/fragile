export const animationStorageKey = "fragile.lab.animations";

// Cosmetic motion is a presentation preference, never part of recorded physics.
export class AnimationService {
  constructor({ storage, media, events } = {}) {
    this.storage = storage;
    this.media = media;
    this.events = events;
    this.targets = new Set();
    let value;
    try {
      value = storage?.getItem(animationStorageKey);
    } catch {
      /* Storage is optional in private browsing. */
    }
    this.explicit = value === "on" || value === "off";
    this.enabled = this.explicit ? value === "on" : !media?.matches;
    this.onMedia = () => {
      if (!this.explicit) this.update(!this.media?.matches);
    };
    this.onStorage = (event) => {
      if (event.key !== animationStorageKey && event.key !== null) return;
      if (event.storageArea && event.storageArea !== this.storage) return;
      const value = event.key === null ? null : event.newValue;
      this.explicit = value === "on" || value === "off";
      this.update(this.explicit ? value === "on" : !this.media?.matches);
    };
    media?.addEventListener?.("change", this.onMedia);
    events?.addEventListener?.("storage", this.onStorage);
  }
  update(enabled) {
    if (this.enabled === enabled) return;
    this.enabled = enabled;
    for (const target of this.targets) target(enabled);
  }
  setEnabled(enabled) {
    this.explicit = true;
    const value = Boolean(enabled);
    try {
      this.storage?.setItem(animationStorageKey, value ? "on" : "off");
    } catch {
      /* A session choice still works when persistence is unavailable. */
    }
    this.update(value);
  }
  subscribe(target) {
    this.targets.add(target);
    return () => this.targets.delete(target);
  }
  dispose() {
    this.media?.removeEventListener?.("change", this.onMedia);
    this.events?.removeEventListener?.("storage", this.onStorage);
    this.targets.clear();
  }
}

let storage;
try {
  storage = globalThis.localStorage;
} catch {
  /* Access itself can be blocked by browser settings. */
}
export const labAnimations = new AnimationService({
  storage,
  media: globalThis.matchMedia?.("(prefers-reduced-motion: reduce)"),
  events: globalThis.window,
});
