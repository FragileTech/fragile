// UI configuration is a draft; consumers of a running session read active only.
export class WorkspaceState extends EventTarget {
  constructor() {
    super();
    this.mode = "inspect";
    this.timeline = "motion";
    this.readOnly = false;
  }
  commit(scene, settings) {
    this.active = structuredClone({ scene, settings });
    this.discard();
  }
  discard() {
    this.draft = structuredClone(this.active);
    this.changed();
  }
  changed() {
    this.dispatchEvent(new Event("change"));
  }
  get dirty() {
    return this.changes.length > 0;
  }
  get changes() {
    const changes = [];
    const walk = (a, b, prefix = "") => {
      for (const key of new Set([
        ...Object.keys(a || {}),
        ...Object.keys(b || {}),
      ])) {
        const before = a?.[key],
          after = b?.[key],
          path = prefix ? `${prefix}.${key}` : key;
        if (JSON.stringify(before) === JSON.stringify(after)) continue;
        if (
          before &&
          after &&
          typeof before === "object" &&
          typeof after === "object" &&
          ((!Array.isArray(before) && !Array.isArray(after)) ||
            (Array.isArray(before) &&
              Array.isArray(after) &&
              before.length === after.length))
        )
          walk(before, after, path);
        else changes.push({ path, before, after });
      }
    };
    walk(this.active, this.draft);
    return changes;
  }
}

export function readPreference(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(`lab.workspace.${key}`)) ?? fallback;
  } catch {
    return fallback;
  }
}
export function savePreference(key, value) {
  try {
    localStorage.setItem(`lab.workspace.${key}`, JSON.stringify(value));
  } catch {
    /* The workspace also works when preference storage is unavailable. */
  }
}
