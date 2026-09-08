// Fixed simulation steps, bounded catch-up. A delayed tab never fast-forwards.
export class DriveClock {
  constructor(dt) {
    this.dt = dt * 1000;
    this.next = null;
  }
  reset() {
    this.next = null;
  }
  advance(now, step) {
    this.next ??= now;
    let count = 0;
    while (now >= this.next && count < 5) {
      if (step() === false) {
        this.reset();
        return { count, slow: false };
      }
      this.next += this.dt;
      count++;
    }
    const slow = now >= this.next;
    if (slow) this.next = now + this.dt;
    return { count, slow };
  }
}
