import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { demos } from "../../web/euclidean-gas/lecture/catalog.js";

const base = process.env.LECTURE_BASE_URL || "http://127.0.0.1:8770";
const output = new URL("../../outputs/lecture-review/", import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
async function ready(id) {
  await page.waitForFunction(
    (id) =>
      document.querySelector("#kind")?.textContent.includes("/ " + id + " ·") &&
      document.querySelector("#status")?.textContent.startsWith("Ready") &&
      !document.querySelector("#step")?.disabled,
    id,
    { timeout: 30000 },
  );
  assert.equal(await page.locator("#error").isVisible(), false, id);
  assert.ok(await page.locator("#charts svg").count(), id + " needs plots");
  assert.equal(
    await page
      .locator("#charts")
      .evaluate((el) => /(?:NaN|Infinity)/.test(el.innerHTML)),
    false,
    id + " SVG finite",
  );
}
try {
  await page.goto(base + "/euclidean-gas/lecture.html?demo=I-01");
  for (const [index, demo] of demos.entries()) {
    if (index) await page.locator('[data-demo="' + demo.id + '"]').click();
    await ready(demo.id);
    const initial = await page.locator("#charts").innerHTML();
    await page.locator("#step").click();
    await page.waitForFunction(() => !document.querySelector("#step").disabled);
    assert.equal(
      await page.locator("#error").isVisible(),
      false,
      demo.id + " step error",
    );
    if (["I-08", "III-03", "IV-01", "IV-13", "IV-16"].includes(demo.id))
      await page.screenshot({
        path: new URL(`${demo.id}-revised.png`, output).pathname,
        fullPage: true,
      });
    await page.locator("#reset").click();
    await ready(demo.id);
    assert.equal(
      await page.locator("#charts").innerHTML(),
      initial,
      demo.id + " reset reproducibility",
    );
    const control = demo.controls.find(
      (control) => control.type === "select" && control.options.length > 1,
    );
    if (control) {
      const value = control.options.find(
        (option) => option.value !== control.value,
      ).value;
      if (!(await page.locator('[name="' + control.key + '"]').isVisible()))
        await page.locator(".advanced summary").click();
      await page
        .locator('[name="' + control.key + '"]')
        .selectOption(String(value));
      await ready(demo.id);
      await page.locator("#step").click();
      await page.waitForFunction(
        () => !document.querySelector("#step").disabled,
      );
      assert.equal(
        await page.locator("#error").isVisible(),
        false,
        demo.id + " control error",
      );
    }
    console.log(demo.id + " browser init / step / reset / controls passed");
  }
  await page.locator('[data-demo="I-05"]').click();
  await ready("I-05");
  await page.locator("#run").click();
  await page.waitForFunction(
    () =>
      Number(document.querySelector("#frame").textContent.split(" ")[0]) >= 3,
  );
  // The pause control must stay available while a worker step is pending.
  await page.locator("#run").click();
  await page.waitForFunction(() => !document.querySelector("#step").disabled);
  assert.equal(await page.locator("#run").textContent(), "Run");
  await page.screenshot({
    path: new URL("desktop.png", output).pathname,
    fullPage: true,
  });
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#save").click();
  const download = await downloadPromise;
  const saved = new URL("experiment.json", output).pathname;
  await download.saveAs(saved);
  const json = JSON.parse(await readFile(saved));
  assert.equal(json.id, "I-05");
  assert.ok(json.ticks >= 3);
  // Replay must rebuild a different state. Waiting on the old tick count alone
  // can finish before the asynchronous file read has even begun initialization.
  await page.locator("#reset").click();
  await ready("I-05");
  await page.locator("#load").setInputFiles(saved);
  await page.waitForFunction(
    (ticks) =>
      document.querySelector("#frame").textContent.startsWith(ticks + " ") &&
      document.querySelector("#load").value === "" &&
      !document.querySelector("#step").disabled,
    json.ticks,
  );
  await page.locator("#scrub").fill("0");
  assert.match(await page.locator("#frame").textContent(), /replay view/);
  await page.locator("#live").click();
  assert.match(await page.locator("#frame").textContent(), /latest/);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: new URL("mobile.png", output).pathname,
    fullPage: true,
  });
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
    "mobile overflow",
  );
  async function setScientificControl(id, key, value) {
    const input = page.locator(`[name="${key}"]`);
    if (!(await input.isVisible()))
      await page.locator(".advanced summary").click();
    if ((await input.evaluate((el) => el.tagName)) === "SELECT")
      await input.selectOption(String(value));
    else {
      await input.fill(String(value));
      await input.dispatchEvent("change");
    }
    await ready(id);
  }
  await page.goto(base + "/euclidean-gas/lecture.html?demo=I-04");
  await ready("I-04");
  await setScientificControl("I-04", "survivors", 0);
  assert.match(await page.locator("#message").textContent(), /extinc/i);
  assert.equal(await page.locator("#error").isVisible(), false);
  await setScientificControl("I-04", "survivors", 4);
  await page.locator("#step").click();
  await page.waitForFunction(() => !document.querySelector("#step").disabled);
  assert.equal(await page.locator("#error").isVisible(), false);
  for (const id of ["IV-01", "IV-16"]) {
    await page.goto(base + `/euclidean-gas/lecture.html?demo=${id}`);
    await ready(id);
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
      `${id}: numerical metrics fit a narrow screen`,
    );
    await page.screenshot({
      path: new URL(`${id}-mobile.png`, output).pathname,
      fullPage: true,
    });
  }
  await page.goto(
    base + "/fragile/euclidean-gas/lecture.html?demo=IV-16&embed=1",
  );
  await ready("IV-16");
  assert.equal(await page.locator("#catalog").isVisible(), false);
  await page.locator("#step").click();
  await page.waitForFunction(() => !document.querySelector("#step").disabled);
  assert.equal(await page.locator("#error").isVisible(), false);
  assert.deepEqual(errors, []);
  console.log(
    "Replay, exports, history, mobile, and project-prefix embed passed",
  );
} finally {
  await browser.close();
}
