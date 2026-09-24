import assert from "node:assert/strict";
import { chromium } from "playwright";
const browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  await page.goto(process.env.ARCADE_TEST_URL || "http://127.0.0.1:8093/web/arcade.html");
  await page.waitForFunction(() => !document.getElementById("btn-start").disabled);
  await page.locator("#param-n").evaluate(el => { el.value = "4"; });
  for (const algorithm of (process.env.ARCADE_TEST_ALGORITHMS || "1,0,2,3").split(",").map(Number)) {
    await page.locator(`[data-algo="${algorithm}"]`).click();
    await page.waitForFunction(() => !document.getElementById("btn-start").disabled);
    await page.locator("#btn-start").click();
    await page.waitForFunction(() => Number(document.getElementById("stat-iteration").textContent) >= 8);
    await page.locator("#trajectory-best").click();
    await page.waitForFunction(() => document.getElementById("trajectory-status").textContent.includes("state 1 /"));
    assert.equal(await page.locator("#btn-pause").isDisabled(), true);
    const iteration = await page.locator("#stat-iteration").textContent();
    const root = await page.locator("#trajectory-screen").evaluate(c => c.toDataURL());
    const max = await page.locator("#trajectory-seek").getAttribute("max");
    assert.ok(Number(max) > 0);
    await page.locator("#trajectory-seek").evaluate(el => {
      el.value = el.max; el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.waitForFunction(() => {
      const slider = document.getElementById("trajectory-seek");
      return document.getElementById("trajectory-status").textContent.includes(`state ${Number(slider.max) + 1} /`);
    });
    const end = await page.locator("#trajectory-screen").evaluate(c => c.toDataURL());
    assert.notEqual(end, root);
    if (algorithm < 2) assert.equal(end, await page.locator("#screen").evaluate(c => c.toDataURL()));
    await page.locator("#trajectory-home").click();
    await page.waitForFunction(() => document.getElementById("trajectory-status").textContent.includes("state 1 /"));
    assert.equal(root, await page.locator("#trajectory-screen").evaluate(c => c.toDataURL()));
    await page.locator("#trajectory-play").click();
    await page.waitForFunction(() => Number(document.getElementById("trajectory-seek").value) > 0);
    if (await page.locator("#trajectory-play").textContent() === "Pause") await page.locator("#trajectory-play").click();
    assert.equal(iteration, await page.locator("#stat-iteration").textContent());
    await page.locator("#trajectory-walker").fill("0");
    await page.locator("#trajectory-walker").press("Tab");
    await page.waitForFunction(() => document.getElementById("trajectory-status").textContent.includes("Walker 0 · state"));
    if (algorithm === 1) {
      const live = await page.locator("#screen-panel").boundingBox();
      const replay = await page.locator("#trajectory-panel").boundingBox();
      assert.equal(live.y, replay.y, "player is beside the best walker");
      await page.screenshot({ path: "/tmp/arcade-trajectory.png", fullPage: true });
    }
    await page.locator("#btn-start").click();
    assert.equal(await page.locator("#trajectory-play").isDisabled(), true);
    await page.waitForFunction(old => Number(document.getElementById("stat-iteration").textContent) > Number(old), iteration);
    await page.locator("#btn-pause").click();
    await page.locator("#btn-reset").click();
    await page.waitForFunction(() => document.getElementById("status").textContent.startsWith("Reset"));
    assert.equal(await page.locator("#trajectory-play").isDisabled(), true);
    console.log(`Trajectory player: algorithm ${algorithm} passed`);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  assert.deepEqual(errors, []);
} finally { await browser.close(); }
