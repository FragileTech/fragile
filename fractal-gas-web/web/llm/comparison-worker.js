import { computeComparison } from "./comparison-metrics.js";
let source,
  revision = 0;
const cache = new Map();
self.onmessage = ({ data }) => {
  try {
    if (data.source !== undefined) {
      source = data.source;
      revision = data.revision;
      cache.clear();
    }
    if (data.events) {
      source.events.push(...data.events);
      source.live = data.live;
      revision = data.revision;
      cache.clear();
    }
    if (!source) {
      postMessage({ id: data.id, revision, data: null });
      return;
    }
    const key = JSON.stringify([revision, data.filters, data.grades]);
    let result = cache.get(key);
    if (!result) {
      result = computeComparison(source, data.filters, data.grades);
      cache.set(key, result);
      if (cache.size > 4) cache.delete(cache.keys().next().value);
    }
    postMessage({ id: data.id, revision, data: result });
  } catch (e) {
    postMessage({ id: data.id, revision, error: e.message });
  }
};
