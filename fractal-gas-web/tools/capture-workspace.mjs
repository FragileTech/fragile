// Real browser captures of the default workspace, themes and responsive layout.
import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
const output = new URL(
  "../../docs/_static/control_lab/workspace/",
  import.meta.url,
);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1280, height: 720 },
    serviceWorkers: "block",
  });
  page.setDefaultTimeout(60000);
  await page.goto(process.env.CONTROL_TEST_URL || "http://127.0.0.1:8089/lab/");
  await page.waitForFunction(() => !document.querySelector("#run").disabled);
  await page.locator("#task-chooser").waitFor({ state: "visible" });
  await page.screenshot({
    animations: "disabled",
    path: new URL("tasks.png", output).pathname,
  });
  await page.locator("#chooser-explore").click();
  await page.locator("#dismiss-intro").click();
  await page.waitForFunction(
    () => document.querySelector("#style-status").textContent === "",
  );
  await page.screenshot({
    animations: "disabled",
    path: new URL("desktop.png", output).pathname,
  });
  await page.locator("#mode-drive").click();
  await page.screenshot({
    animations: "disabled",
    path: new URL("drive.png", output).pathname,
  });
  await page.locator("#mode-inspect").click();
  await page.locator("#tab-view").click();
  await page.locator('input[name="visual-style"][value="steampunk"]').check();
  await page.waitForFunction(
    () =>
      document.documentElement.dataset.visualStyle === "steampunk" &&
      document.querySelector("#style-status").textContent === "",
  );
  await page.screenshot({
    animations: "disabled",
    path: new URL("steampunk.png", output).pathname,
  });
  await page.locator('input[name="visual-style"][value="futuristic"]').check();
  await page.locator("#tab-setup").click();
  await page.setViewportSize({ width: 390, height: 844 });
  if (await page.locator("#close-inspector").isVisible())
    await page.locator("#close-inspector").click();
  await page.screenshot({
    animations: "disabled",
    path: new URL("mobile.png", output).pathname,
  });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.reload();
  await page.waitForFunction(() => !document.querySelector("#run").disabled);
  await page.waitForFunction(
    () => document.querySelector("#style-status").textContent === "",
  );
  await page.screenshot({
    animations: "disabled",
    path: new URL("mobile-reduced-motion.png", output).pathname,
  });
  await writeFile(
    new URL("capture-manifest.json", output),
    JSON.stringify(
      {
        source: "capture-workspace.mjs",
        viewports: [
          [1280, 720],
          [390, 844],
        ],
        captures: [
          "tasks",
          "desktop",
          "drive",
          "steampunk",
          "mobile",
          "mobile-reduced-motion",
        ],
      },
      null,
      2,
    ) + "\n",
  );
  console.log(
    "Captured workspace: desktop, Drive, task chooser, both themes, mobile and reduced motion.",
  );
} finally {
  await browser.close();
}
