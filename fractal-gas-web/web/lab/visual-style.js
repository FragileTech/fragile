import { preloadStyle } from "./visuals/assets.js";

export const visualStyles = Object.freeze(["futuristic", "steampunk"]);
export const styleStorageKey = "fragile.lab.visual-style";

export class StyleService {
  constructor({ preload = preloadStyle, storage } = {}) {
    this.preload = preload;
    this.storage = storage;
    this.current = "futuristic";
    this.ready = false;
    this.revision = 0;
    this.targets = new Set();
  }
  preferred() {
    try {
      const value = this.storage?.getItem(styleStorageKey);
      return visualStyles.includes(value) ? value : "futuristic";
    } catch {
      return "futuristic";
    }
  }
  subscribe(target) {
    this.targets.add(target);
    return () => this.targets.delete(target);
  }
  async change(style) {
    if (!visualStyles.includes(style))
      throw new Error(`Unknown visual style: ${style}`);
    const revision = ++this.revision;
    try {
      await this.preload(style);
    } catch (error) {
      if (revision !== this.revision) return false;
      throw error;
    }
    if (revision !== this.revision) return false;
    // Prepare every viewport before replacing any visible presentation.
    const transactions = [];
    try {
      for (const target of this.targets) transactions.push(target(style));
    } catch (error) {
      for (const transaction of transactions) transaction?.cancel?.();
      throw error;
    }
    for (const transaction of transactions) transaction?.commit?.();
    this.current = style;
    this.ready = true;
    try {
      this.storage?.setItem(styleStorageKey, style);
    } catch {
      /* Storage is optional. */
    }
    return true;
  }
}

let storage;
try {
  storage = globalThis.localStorage;
} catch {
  /* Private browsing. */
}
export const labStyle = new StyleService({ storage });
