// Limits are in bytes, using Numbers (bitwise arithmetic would wrap at 4 GiB).
export const MiB = 1024 ** 2;
export const GiB = 1024 ** 3;
export const PAGE = 65536;
export const MAIN_INITIAL = 512 * MiB;
export const PLAYBACK_LIMIT = 256 * MiB;
export const PLAYBACK_INITIAL = 64 * MiB;
export const RECORDING_LIMIT = 32 * MiB;
export const SHIM_INITIAL = 64 * MiB;
export function resourcePlan(params, resources, hardwareConcurrency = 4) {
  const requested = resources?.workers ?? params.nThreads ?? "auto";
  const memoryGiB = resources?.memoryGiB ?? 8;
  if (!Number.isInteger(params.n) || params.n < 1 || params.n > 1024)
    throw new Error("Walkers must be an integer between 1 and 1024");
  if (
    requested !== "auto" &&
    (!Number.isInteger(requested) || requested < 1 || requested > 20)
  )
    throw new Error("Workers must be Auto or an integer between 1 and 20");
  if (![1, 2, 4, 8].includes(memoryGiB))
    throw new Error("Engine memory limit must be 1, 2, 4, or 8 GiB");
  const workers = Math.min(
    params.n,
    requested === "auto"
      ? Math.min(20, Math.max(1, (hardwareConcurrency || 4) - 1))
      : requested,
  );
  const budgetBytes = memoryGiB * GiB;
  const farmWorkers = params.console === 2 ? workers : 0;
  const mainLimitBytes = Math.min(
    4 * GiB,
    budgetBytes - PLAYBACK_LIMIT - farmWorkers * SHIM_INITIAL,
  );
  if (mainLimitBytes < MAIN_INITIAL)
    throw new Error(
      "Engine memory limit is too small for these workers. Select more memory or fewer workers.",
    );
  const shimLimitBytes = farmWorkers
    ? Math.min(
        2 * GiB,
        Math.floor((budgetBytes - PLAYBACK_LIMIT - mainLimitBytes) / farmWorkers / PAGE) * PAGE,
      )
    : 0;
  return { workers, budgetBytes, mainLimitBytes, shimLimitBytes, farmWorkers, playbackLimitBytes: PLAYBACK_LIMIT };
}
