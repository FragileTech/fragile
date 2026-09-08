import assert from "node:assert/strict";
import { chromium } from "playwright";

const base = process.env.VISUAL_COHESION_TEST_URL || "http://127.0.0.1:8099/";
const routes = ["", "lab/", "optimization/"];
const expected = {
  "--bg": "#15111b",
  "--panel": "#1d1825",
  "--line": "#3b3048",
  "--accent": "#d0a9e2",
  "--cyan": "#7ef5df",
};

const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--use-angle=swiftshader"],
});

try {
  for (const route of routes) {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    await page.goto(new URL(route, base), { waitUntil: "domcontentloaded" });

    const desktop = await page.evaluate((tokens) => {
      const root = getComputedStyle(document.documentElement);
      const header = document.querySelector(".app-header");
      const navItems = [
        ...document.querySelectorAll(".app-nav > a, .app-nav > span"),
      ];
      const navGeometry = navItems.map((item) => {
        const style = getComputedStyle(item);
        return {
          display: style.display,
          height: style.height,
          padding: `${style.paddingTop} ${style.paddingRight} ${style.paddingBottom} ${style.paddingLeft}`,
          borderRadius: style.borderRadius,
          borderStyle: style.borderStyle,
          textDecoration: style.textDecorationLine,
        };
      });
      const navLabels = navItems.map((item) => item.textContent.trim());
      const activeNav = document.querySelector(
        ".app-nav .is-active, .app-nav .nav-active",
      );
      const themeLink = [
        ...document.querySelectorAll('link[rel="stylesheet"]'),
      ].some((link) => new URL(link.href).pathname.endsWith("/theme.css"));
      return {
        tokens: Object.fromEntries(
          Object.keys(tokens).map((name) => [
            name,
            root.getPropertyValue(name).trim(),
          ]),
        ),
        themeLink,
        header: Boolean(header),
        navLabels,
        navGeometry,
        activeNav: activeNav
          ? {
              background: getComputedStyle(activeNav).backgroundColor,
              current: activeNav.getAttribute("aria-current"),
            }
          : null,
        overflow: document.documentElement.scrollWidth <= window.innerWidth,
      };
    }, expected);

    assert.equal(
      desktop.themeLink,
      true,
      `${route || "arcade/"} must load theme.css`,
    );
    assert.equal(
      desktop.header,
      true,
      `${route || "arcade/"} must use the shared header`,
    );
    assert.ok(
      desktop.navGeometry.length >= 3,
      `${route || "arcade/"} must expose the shared lab navigation`,
    );
    assert.deepEqual(
      desktop.navLabels.slice(0, 3),
      ["Arcade", "Control Lab", "Optimization Lab"],
      `${route || "arcade/"} must keep the cross-lab controls in a shared order`,
    );
    assert.equal(
      new Set(desktop.navGeometry.map((item) => JSON.stringify(item))).size,
      1,
      `${route || "arcade/"} navigation controls must share geometry`,
    );
    assert.ok(
      desktop.navGeometry.every(
        (item) =>
          item.display === "flex" &&
          item.borderStyle === "solid" &&
          item.textDecoration === "none",
      ),
      `${route || "arcade/"} navigation controls must render as outlined controls`,
    );
    assert.equal(
      desktop.activeNav?.current,
      "page",
      `${route || "arcade/"} active navigation control must identify the current page`,
    );
    assert.deepEqual(
      desktop.tokens,
      expected,
      `${route || "arcade/"} tokens differ`,
    );

    await page.setViewportSize({ width: 390, height: 844 });
    const mobile = await page.evaluate(() => {
      const navItems = [
        ...document.querySelectorAll(".app-nav > a, .app-nav > span"),
      ];
      return {
        overflow: document.documentElement.scrollWidth <= window.innerWidth,
        header: Boolean(document.querySelector(".app-header")),
        crossLabItems: navItems
          .slice(0, 3)
          .map((item) => getComputedStyle(item).display),
      };
    });
    assert.equal(
      mobile.header,
      true,
      `${route || "arcade/"} mobile header missing`,
    );
    assert.equal(
      mobile.overflow,
      true,
      `${route || "arcade/"} overflows at mobile width`,
    );
    assert.ok(
      mobile.crossLabItems.length === 3 &&
        mobile.crossLabItems.every((display) => display === "flex"),
      `${route || "arcade/"} mobile navigation controls must remain visible`,
    );

    await page.close();
  }

  const lab = await browser.newPage({ viewport: { width: 1024, height: 768 } });
  await lab.goto(new URL("lab/", base), { waitUntil: "domcontentloaded" });
  const styles = await lab.evaluate(() => {
    const root = document.documentElement;
    const futuristic = getComputedStyle(root).getPropertyValue("--bg").trim();
    root.dataset.visualStyle = "steampunk";
    const steampunk = getComputedStyle(root).getPropertyValue("--bg").trim();
    delete root.dataset.visualStyle;
    const restored = getComputedStyle(root).getPropertyValue("--bg").trim();
    return { futuristic, steampunk, restored };
  });
  assert.equal(styles.futuristic, expected["--bg"]);
  assert.equal(styles.steampunk, "#211a15");
  assert.equal(styles.restored, expected["--bg"]);
  await lab.close();
} finally {
  await browser.close();
}

console.log(
  "Visual cohesion browser checks passed for arcade, control lab, and optimization lab.",
);
