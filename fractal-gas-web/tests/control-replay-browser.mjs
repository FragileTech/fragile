import { prepareWorkspace, applyDraft, openFiles, closeFiles } from "./helpers/workspace-ui.mjs";
// End-to-end motion replay, extensible model kits, and graphics inspection.
import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
const base = process.env.CONTROL_TEST_URL || "http://127.0.0.1:8080/lab/";
const output =
    process.env.CONTROL_SCREENSHOTS || "/tmp/fractal-control-browser";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox"],
});
const page = await browser.newPage({ viewport: { width: 1536, height: 1050 } }),
    errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
});
const ready = () =>
    page.waitForFunction(() => !document.getElementById("run").disabled);
const tick = (n) =>
    page.waitForFunction(
        (n) =>
            document.getElementById("tick").textContent ===
            `TICK ${String(n).padStart(6, "0")}`,
        n,
    );
const seek = async (n) => {
    await page.locator("#motion-timeline").fill(String(n));
    await page.locator("#motion-timeline").dispatchEvent("input");
};
async function graphicsReady() {
    await page.waitForFunction(() => {
        const gl = document.getElementById("world").getContext("webgl2");
        if (gl.isContextLost()) {
            window.graphicsStableSince = undefined;
            return false;
        }
        window.graphicsStableSince ??= performance.now();
        return performance.now() - window.graphicsStableSince > 1000;
    });
}
try {
    await prepareWorkspace(page);
    await page.goto(base);
    await ready();
    // Keep the replay fixture at six frames after checking the new default.
    await page.waitForFunction(
      () => document.getElementById("tick").textContent === "TICK 000000",
    );
    await page.locator("#tab-controller").click();
    await page.locator("#frames").fill("6");
    await page.locator("#frames").press("Tab");
    await applyDraft(page);
    await ready();
    await page.locator("#step").click();
    await page.waitForFunction(
      () => document.getElementById("tick").textContent === "TICK 000006",
    );
    await tick(6);
    const recordedDead = await page.locator("#dead").textContent();
    await seek(0);
    await tick(0);
    assert.equal(await page.locator("#dead").textContent(), "—");
    await seek(6);
    await tick(6);
    assert.equal(await page.locator("#dead").textContent(), recordedDead);
    await page.locator("#motion-live").click();
    await page.locator("#focus").click();
    await page.locator("#clean").click();
    await graphicsReady();
    await page.screenshot({
        path: `${output}/rocket-detail.png`,
        fullPage: true,
    });
    await page.locator("#step").click();
    await tick(12);
    assert.equal(
        await page.locator("#motion-timeline").getAttribute("max"),
        "12",
    );
    await seek(3);
    await tick(3);
    assert.equal(await page.locator("#run-state").innerText(), "WORLD REPLAY");
    await page.locator("#motion-live").click();
    await tick(12);
    await seek(5);
    await page.locator("#motion-resume").click();
    await tick(5);
    await page.locator("#step").click();
    await tick(11);
    assert.match(
        await page.locator("#motion-position").innerText(),
        /7 \/ 7 frames/,
    );
    await seek(0);
    await tick(5);
    assert.match(
        await page.locator("#motion-segment").innerText(),
        /Initial world/,
    );
    await page.locator("#motion-speed").selectOption("0.25");
    await page.locator("#motion-play").click();
    await page.waitForFunction(() =>
        document
            .getElementById("motion-play")
            .textContent.includes("Play world"),
    );
    await tick(11);
    await page.locator("#timeline-decisions").click();
    await page.locator("#recording").selectOption("0");
    await applyDraft(page);
    await page.locator("#timeline-motion").click();
    await ready();
    await tick(0);
    await page.locator("#step").click();
    await tick(6);
    assert.match(
        await page.locator("#record-count").textContent(),
        /No decisions/,
    );
    await page.locator("#mode-drive").click();
    await page.locator("#run").click();
    await page.locator("#world").focus();
    await page.keyboard.down("w");
    await page.waitForFunction(
        () =>
            parseInt(document.getElementById("tick").textContent.slice(5)) >= 8,
    );
    await page.keyboard.up("w");
    await page.locator("#mode-inspect").click();
    assert(
        Number(await page.locator("#motion-timeline").getAttribute("max")) >= 8,
    );
    await openFiles(page);
    const download = page.waitForEvent("download");
    await page.locator("#export-run").click();
    const file = `${output}/motion-only.fgcrec`;
    await (await download).saveAs(file);
    const exported = await readFile(file);
    assert.equal(exported.readUInt32LE(0), 0x52434746);
    assert.equal(exported.readUInt32LE(4), 3);
    const choose = page.waitForEvent("filechooser");
    await page.locator("#import-run").click();
    await (await choose).setFiles(file);
    await closeFiles(page);
    await page.waitForFunction(() => !document.querySelector("main").inert && document.querySelector("#playback-status").textContent === "Replay");
    await tick(0);
    await seek(6);
    await tick(6);
    await page.locator("#motion-resume").click();
    await ready();
    // Mix types and introduce a new body entirely through declarative scene data.
    await page.locator("#mode-edit").click();
    await page.locator("#edit-json").click();
    const scene = JSON.parse(await page.locator("#scene-json").inputValue());
    scene.name = "Vehicle design range";
    scene.task = "navigation";
    scene.holes = [];
    scene.boundary = undefined;
    scene.bases = [];
    scene.gravity = [];
    scene.pickups = [];
    scene.tethers = [];
    scene.size = [32, 24];
    scene.agent_types.beacon = {
        extends: "drone",
        label: "Beacon courier",
        visual: {
            model: "kit",
            color: "#9f82ff",
            parts: [
                { shape: "box", size: [1, 0.45, 0.22], position: [0, 0, 0.3] },
                {
                    shape: "ring",
                    size: [0.5, 0.05],
                    position: [0, 0, 0.48],
                    emissive: true,
                },
            ],
        },
    };
    scene.bodies = [
        { agent_type: "kart", position: [8, 12], angle: 0.3 },
        { agent_type: "rocket", position: [12, 12], angle: 0.3 },
        { agent_type: "drone", position: [16, 12] },
        { agent_type: "beacon", position: [20, 12] },
    ];
    await page.locator("#scene-json").fill(JSON.stringify(scene));
    await page.locator("#apply-json").click();
    await applyDraft(page);
    await ready();
    assert.deepEqual(
        await page.locator("#agent-type option").evaluateAll((options) =>
            options.map((option) => option.value).sort(),
        ),
        Object.keys(scene.agent_types).sort(),
    );
    await graphicsReady();
    await page.screenshot({
        path: `${output}/vehicle-lineup.png`,
        fullPage: true,
    });
    await page.locator("#focus").click();
    await graphicsReady();
    await page.screenshot({
        path: `${output}/kart-detail.png`,
        fullPage: true,
    });
    await page.locator("#step").click();
    await tick(6);
    await seek(2);
    await tick(2);
    // Place a custom agent using the editor's type selector; the C++ compiler supplies controlled/mass defaults.
    await page.locator("#motion-live").click();
    await page.locator("#focus").click();
    await page.locator("#mode-edit").click();
    await page.locator("#tool").selectOption("bodies");
    await page.locator("#agent-type").selectOption("beacon");
    const canvas = await page.locator("#world").boundingBox();
    await page.mouse.click(
        canvas.x + canvas.width * 0.55,
        canvas.y + canvas.height * 0.6,
    );
    await page.locator("#edit-json").click();
    const edited = JSON.parse(await page.locator("#scene-json").inputValue());
    assert.equal(edited.bodies.length, 5);
    assert.equal(edited.bodies[4].agent_type, "beacon");
    await page
        .getByRole("button", { name: "Close JSON editor", exact: true })
        .click();
    await applyDraft(page);
    await page.setViewportSize({ width: 768, height: 1024 });
    await graphicsReady();
    await page.screenshot({
        path: `${output}/motion-tablet.png`,
        fullPage: true,
    });
    assert.equal(
        await page.evaluate(
            () => document.documentElement.scrollWidth > innerWidth,
        ),
        false,
    );
    assert.deepEqual(errors, []);
    console.log(
        "Browser passed: exact world seeking, playback, continuation, motion-only export/import, mixed agent types, declarative model kit, editor placement, detailed assets and responsive controls.",
    );
} finally {
    await browser.close();
}
