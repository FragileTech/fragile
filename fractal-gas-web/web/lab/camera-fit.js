export const ARENA_MARGIN = 1.05;

const defaultSize = [64, 44];

function sizeValue(size, index) {
  const value = size?.[index];
  return Number.isFinite(value) && value > 0 ? value : defaultSize[index];
}

export function arenaBounds(scene) {
  const fromSize = () => {
    const width = sizeValue(scene?.size, 0),
      height = sizeValue(scene?.size, 1);
    return {
      minX: 0,
      maxX: width,
      minY: 0,
      maxY: height,
      center: [width / 2, height / 2],
    };
  };
  const points = (Array.isArray(scene?.boundary) ? scene.boundary : []).filter(
    (point) =>
      Array.isArray(point) &&
      Number.isFinite(point[0]) &&
      Number.isFinite(point[1]),
  );
  if (points.length < 3) return fromSize();
  const xs = points.map(([x]) => x),
    ys = points.map(([, y]) => y),
    minX = Math.min(...xs),
    maxX = Math.max(...xs),
    minY = Math.min(...ys),
    maxY = Math.max(...ys);
  if (maxX <= minX || maxY <= minY) return fromSize();
  return {
    minX,
    maxX,
    minY,
    maxY,
    center: [(minX + maxX) / 2, (minY + maxY) / 2],
  };
}

export function arenaHalfSpan(bounds, aspect, margin = ARENA_MARGIN) {
  const ratio = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const padding =
    Number.isFinite(margin) && margin >= 1 ? margin : ARENA_MARGIN;
  return (
    (padding / 2) *
    Math.max(bounds.maxY - bounds.minY, (bounds.maxX - bounds.minX) / ratio)
  );
}
