import { demos as foundations } from "./foundations.js";
import { demos as convergence } from "./convergence.js";
import { demos as entropy } from "./entropy.js";
import { demos as fractal } from "./fractal.js";
import { demos as qft } from "./partvi.js";

export const demos = [...foundations, ...convergence, ...entropy, ...fractal, ...qft];
export const metadata = demos.map(({ create, ...descriptor }) => descriptor);

export function parameters(demo, input = {}) {
  return Object.fromEntries(
    demo.controls.map((control) => {
      const value = input[control.key] ?? control.value;
      if (control.type === "select") {
        const option = control.options.find(
          (option) => String(option.value) === String(value),
        );
        if (!option) throw new Error("Invalid " + control.label);
        return [control.key, option.value];
      }
      const number = Number(value);
      if (
        !Number.isFinite(number) ||
        number < control.min ||
        number > control.max
      )
        throw new Error(
          control.label +
            " must be between " +
            control.min +
            " and " +
            control.max,
        );
      return [control.key, number];
    }),
  );
}
