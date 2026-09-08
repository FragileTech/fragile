// Functional suites assert layout and rendering directly. Diagnostic screenshots
// are opt-in: software WebGL capture can time out and restore a stale viewport.
export async function captureScreenshot(page, options) {
  if (process.env.CONTROL_CAPTURE_SCREENSHOTS !== "1") return;
  await page.screenshot({
    ...options,
    fullPage: false,
    animations: "disabled",
    timeout: 30000,
  });
}
