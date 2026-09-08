// Test native controls independently of WebGL and WebAssembly startup.
import { chromium, firefox } from "playwright";
import assert from "node:assert/strict";

const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/";
for (const [name, type] of Object.entries({ chromium, firefox })) {
  const executablePath =
    process.env[`CONTROL_${name.toUpperCase()}_EXECUTABLE`];
  const browser = await type.launch({ headless: true, executablePath });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/lab/main.js", (route) =>
      route.fulfill({ contentType: "text/javascript", body: "" }),
    );
    await page.goto(base);
    await page.evaluate(async () => {
      const { initHelp } = await import("../help.js");
      initHelp();
      const count = document.querySelectorAll(".help-icon").length;
      initHelp();
      if (document.querySelectorAll(".help-icon").length !== count)
        throw new Error("Help initialization must be idempotent");
    });
    assert.equal(
      await page
        .locator("select button, button button, textarea button, input button")
        .count(),
      0,
    );
    assert.deepEqual(
      await page
        .locator("label[data-help]")
        .evaluateAll((labels) =>
          labels
            .filter((label) => label.control?.classList.contains("help-icon"))
            .map((label) => label.textContent),
        ),
      [],
    );

    // Resolve the actual checkbox id from the page rather than assuming it, and
    // skip the checkboxes that live inside collapsed panels.
    const checkLabel = page
      .locator("label.check[data-help]:has(input:not(:disabled)):visible")
      .first();
    const checkbox = checkLabel.locator('input[type="checkbox"]');
    const checked = await checkbox.isChecked();
    await checkLabel.click({ position: { x: 90, y: 7 } });
    assert.equal(await checkbox.isChecked(), !checked);
    await checkLabel.locator(".help-icon").click();
    assert.equal(await checkbox.isChecked(), !checked);
    assert.equal(await page.locator("#tooltip").isVisible(), true);
    await page.keyboard.press("Escape");
    assert.equal(await page.locator("#tooltip").isVisible(), false);

    const speed = page.locator("#motion-speed");
    const options = await speed.locator("option").count();
    await speed.focus();
    await speed.press("Home");
    await speed.press("ArrowDown");
    await speed.press("Enter");
    assert.equal(await speed.evaluate((el) => el.selectedIndex), 1);
    assert.equal(await speed.locator("option").count(), options);
    await speed.selectOption("2");
    assert.equal(await speed.inputValue(), "2");

    await page.locator("#view").evaluate((button) => {
      window.activations = 0;
      button.addEventListener("click", () => window.activations++);
    });
    await page.locator("#view").click();
    assert.equal(await page.evaluate(() => window.activations), 1);
    assert.deepEqual(errors, []);
    console.log(
      `${name}: native dropdown, label, checkbox, button and tooltip checks passed`,
    );
  } finally {
    await browser.close();
  }
}
