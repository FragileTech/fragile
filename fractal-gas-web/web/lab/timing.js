import { treePoseDim, treeWidth } from "./actions.js";
export function equalBytes(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// A late result never changes the action already committed to a physical tick.
export function acceptPlan(result, tick, revision, snapshot) {
  return (
    result.revision === revision &&
    result.target === tick &&
    equalBytes(result.root, snapshot)
  );
}

export function branchActions(tree, leaf) {
  const index = new Map(),
    width = treeWidth(tree);
  for (let i = 0; i < tree.meta.length / 5; i++) index.set(tree.meta[i * 5], i);
  const path = [],
    visited = new Set();
  while (leaf) {
    if (visited.has(leaf) || !index.has(leaf))
      throw new Error("Invalid recording ancestry");
    visited.add(leaf);
    const i = index.get(leaf),
      parent = tree.meta[i * 5 + 1];
    if (parent)
      path.push({
        frames: tree.meta[i * 5 + 3],
        action: tree.values.slice(
          i * width + 3 + treePoseDim(tree),
          (i + 1) * width,
        ),
      });
    leaf = parent;
  }
  return path.reverse();
}
