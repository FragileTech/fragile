// Power-law Gaussian noise without a DSP dependency. A radix-2 inverse FFT
// generates a padded stationary series with PSD proportional to 1/f^beta.
// DC uses the first nonzero frequency; ensemble variance (including DC) is 1.
// Crop to the requested horizon; never normalize individual realizations.
export class ColoredNoise {
  constructor(length, beta) {
    this.length = length;
    this.beta = beta;
    this.size = 2 ** Math.ceil(Math.log2(Math.max(2, length)));
    this.real = new Float64Array(this.size);
    this.imag = new Float64Array(this.size);
    this.scale = Float64Array.from(
      { length: this.size / 2 + 1 },
      (_, k) => Math.max(1, k) ** (-beta / 2),
    );
    let variance = this.scale[0] ** 2 + this.scale[this.size / 2] ** 2;
    for (let k = 1; k < this.size / 2; k++) variance += 2 * this.scale[k] ** 2;
    this.normalizer = Math.sqrt(variance);
  }
  fill(rng, out, offset = 0, stride = 1) {
    if (this.beta === 0 || this.length === 1) {
      for (let i = 0; i < this.length; i++)
        out[offset + i * stride] = rng.normal();
      return;
    }
    const n = this.size,
      re = this.real,
      im = this.imag;
    im.fill(0);
    re[0] = rng.normal() * this.scale[0];
    re[n / 2] = rng.normal() * this.scale[n / 2];
    for (let k = 1; k < n / 2; k++) {
      const a = (rng.normal() * this.scale[k]) / Math.SQRT2;
      const b = (rng.normal() * this.scale[k]) / Math.SQRT2;
      re[k] = re[n - k] = a;
      im[k] = b;
      im[n - k] = -b;
    }
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let width = 2; width <= n; width *= 2) {
      const angle = (2 * Math.PI) / width,
        wr = Math.cos(angle),
        wi = Math.sin(angle);
      for (let start = 0; start < n; start += width) {
        let ur = 1,
          ui = 0;
        for (let j = 0; j < width / 2; j++) {
          const a = start + j,
            b = a + width / 2;
          const vr = re[b] * ur - im[b] * ui,
            vi = re[b] * ui + im[b] * ur;
          re[b] = re[a] - vr;
          im[b] = im[a] - vi;
          re[a] += vr;
          im[a] += vi;
          const next = ur * wr - ui * wi;
          ui = ur * wi + ui * wr;
          ur = next;
        }
      }
    }
    for (let i = 0; i < this.length; i++)
      out[offset + i * stride] = re[i] / this.normalizer;
  }
}
