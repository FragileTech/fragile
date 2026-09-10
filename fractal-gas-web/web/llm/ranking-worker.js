import {
  analyzeRanking,
  fitRanking,
  trainingObservations,
} from "./ranking-statistics.js";
import { predictPair, RANKING_CRITERIA } from "./ranking-math.js";
let latest;
const fitsCache = new Map();
self.onmessage = ({ data }) => {
  try {
    if (data.type === "predict") {
      postMessage({
        id: data.id,
        value: latest
          ? predictPair(latest.fits[data.criterion], data.a, data.b, {
              draws: 2000,
              seed: data.seed,
            })
          : null,
      });
      return;
    }
    const session = data.session;
    const key = JSON.stringify([
      session.source_digest,
      session.plan.cohort,
      session.model,
      session.config.seed,
      data.type,
      RANKING_CRITERIA.map((k) => trainingObservations(session, k)),
    ]);
    let fits = fitsCache.get(key);
    if (!fits) {
      fits = fitRanking(
        session,
        data.type === "fit" ? { draws: 0, sensitivity: false } : {},
      );
      fitsCache.set(key, fits);
      if (fitsCache.size > 4) fitsCache.delete(fitsCache.keys().next().value);
    }
    const value =
      data.type === "fit" ? fits : analyzeRanking(session, { fits });
    if (data.type !== "fit") latest = value;
    postMessage({ id: data.id, value });
  } catch (e) {
    postMessage({ id: data.id, error: e.message });
  }
};
