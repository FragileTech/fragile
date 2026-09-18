// Pure display geometry for the 3D views. No three.js import: node-testable.
export const SPAN = 10;
export const coord = (x, low, high) =>
  ((x - low) / (high - low)) * 2 * SPAN - SPAN;

// Robust asinh scale anchored at the optimum side of the sampled values.
export function robustScale(values, direction = "minimize") {
  let count = 0;
  for (let i = 0; i < values.length; i++)
    if (Number.isFinite(values[i])) count++;
  if (!count) return null;
  const finite = new Float64Array(count);
  for (let i = 0, k = 0; i < values.length; i++)
    if (Number.isFinite(values[i])) finite[k++] = values[i];
  finite.sort();
  const maximize = direction === "maximize",
    base = maximize ? finite[count - 1] : finite[0],
    p = (q) => finite[Math.floor((count - 1) * q)];
  return {
    base,
    orient: maximize ? -1 : 1,
    scale: Math.max(1e-9, (p(0.95) - p(0.05)) / 3, Math.abs(base) * 1e-8),
  };
}
// asinh is odd, so the height is direction-independent; only the anchor moves.
export const heightOf = (v, s, heightScale) =>
  Math.asinh((v - s.base) / s.scale) * heightScale;
// 0 at the optimum side, 1 about three asinh units away.
export const shade = (v, s) => {
  const t = Math.asinh((s.orient * (v - s.base)) / s.scale) / 3;
  return Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0;
};
export const contourLevels = (s, count = 9) =>
  Array.from(
    { length: count },
    (_, i) => s.base + s.orient * s.scale * Math.sinh((i + 1) / 3),
  );

// Row-major grid (row = y, column = x). Triangles touching a nonfinite sample
// are skipped; their unused coordinates are zeroed before upload.
export function buildSurface(values, resolution, zOf, s) {
  const n = resolution,
    positions = new Float32Array(n * n * 3),
    t = new Float32Array(n * n),
    ok = new Uint8Array(n * n);
  let triangles = 0;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const k = j * n + i,
        z = zOf(values[k], k);
      ok[k] = Number.isFinite(values[k]) && Number.isFinite(z) ? 1 : 0;
      positions[k * 3] = (i / (n - 1)) * 2 * SPAN - SPAN;
      positions[k * 3 + 1] = (j / (n - 1)) * 2 * SPAN - SPAN;
      positions[k * 3 + 2] = ok[k] ? z : 0;
      t[k] = ok[k] && s ? shade(values[k], s) : 0;
    }
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  for (let j = 0; j < n - 1; j++)
    for (let i = 0; i < n - 1; i++) {
      const a = j * n + i,
        b = a + 1,
        c = a + n,
        d = c + 1;
      if (ok[a] && ok[b] && ok[c]) {
        indices[triangles++] = a;
        indices[triangles++] = b;
        indices[triangles++] = c;
      }
      if (ok[b] && ok[c] && ok[d]) {
        indices[triangles++] = b;
        indices[triangles++] = d;
        indices[triangles++] = c;
      }
    }
  return { positions, indices: indices.subarray(0, triangles), t };
}

// Marching triangles over the exact sampled values.
export function contourSegments(
  values,
  positions,
  indices,
  levels,
  lift = 0.012,
) {
  const lines = [];
  const hit = [0, 0, 0, 0, 0, 0];
  for (const target of levels)
    for (let t = 0; t < indices.length; t += 3) {
      let hits = 0;
      for (let e = 0; e < 3 && hits < 2; e++) {
        const a = indices[t + e],
          b = indices[t + ((e + 1) % 3)],
          va = values[a],
          vb = values[b];
        if ((va < target && vb >= target) || (vb < target && va >= target)) {
          const f = (target - va) / (vb - va);
          for (let k = 0; k < 3; k++)
            hit[hits * 3 + k] =
              positions[a * 3 + k] +
              f * (positions[b * 3 + k] - positions[a * 3 + k]) +
              (k === 2 ? lift : 0);
          hits++;
        }
      }
      if (hits === 2) lines.push(...hit);
    }
  return Float32Array.from(lines);
}

export function bestIndex(raw, eligible, direction = "minimize") {
  let best = -1;
  for (let i = 0; i < raw.length; i++) {
    if (!eligible[i] || !Number.isFinite(raw[i])) continue;
    if (
      best < 0 ||
      (direction === "minimize" ? raw[i] < raw[best] : raw[i] > raw[best])
    )
      best = i;
  }
  return best;
}
export function metricRange(metric, eligible) {
  let min = Infinity,
    max = -Infinity;
  for (let i = 0; i < metric.length; i++)
    if (eligible[i] && Number.isFinite(metric[i])) {
      if (metric[i] < min) min = metric[i];
      if (metric[i] > max) max = metric[i];
    }
  return [min, max];
}

// (walker, donor) slot pairs drawn during the last step. Donors frozen in an
// earlier frame have no current position and are only counted.
export function companionPairs(report, kind, n, out = null, only = -1) {
  let count = 0,
    skippedHistorical = 0;
  const push = (i, source) => {
    if (source.frame !== report.step - 1) skippedHistorical++;
    else if (source.slot !== i && source.slot < n) {
      if (!out || out.length < count + 2) {
        const next = new Uint32Array(Math.max(64, (out?.length || 0) * 2));
        if (out) next.set(out);
        out = next;
      }
      out[count++] = i;
      out[count++] = source.slot;
    }
  };
  if (report && kind === "distance") {
    const c = report.distance_companions;
    for (let i = 0; i < Math.min(n, c.rows); i++) {
      if (only >= 0 && i !== only) continue;
      for (let a = 0; a < c.count; a++) {
        const index = i * c.count + a;
        if (c.valid[index]) push(i, report.distance_sources[c.indices[index]]);
      }
    }
  } else if (report && kind === "cloning") {
    const plan = report.clone_plan;
    for (let i = 0; i < Math.min(n, plan.choices.length); i++) {
      if (only >= 0 && i !== only) continue;
      for (const donor of plan.choices[i].donors)
        push(i, plan.sources[donor.pool_index]);
    }
  }
  return { count: count / 2, skippedHistorical, out };
}

// Column-major uniform-scale translation, as InstancedMesh expects.
export function writeInstance(m, i, x, y, z, r) {
  const o = i * 16;
  m.fill(0, o, o + 16);
  m[o] = m[o + 5] = m[o + 10] = r;
  m[o + 12] = x;
  m[o + 13] = y;
  m[o + 14] = z;
  m[o + 15] = 1;
}

// Coarse readback right after a draw; used by browser smoke tests.
export function pixelStats(gl, background = [9, 13, 22]) {
  const w = gl.drawingBufferWidth,
    h = gl.drawingBufferHeight,
    pixels = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const colors = new Set(),
    stride = Math.max(1, Math.floor(Math.min(w, h) / 96));
  let samples = 0,
    foreground = 0;
  for (let y = 0; y < h; y += stride)
    for (let x = 0; x < w; x += stride) {
      const k = (y * w + x) * 4,
        r = pixels[k],
        g = pixels[k + 1],
        b = pixels[k + 2];
      colors.add((r << 16) | (g << 8) | b);
      samples++;
      if (
        Math.abs(r - background[0]) +
          Math.abs(g - background[1]) +
          Math.abs(b - background[2]) >
        12
      )
        foreground++;
    }
  return {
    distinctColors: colors.size,
    nonBackgroundFraction: samples ? foreground / samples : 0,
  };
}
