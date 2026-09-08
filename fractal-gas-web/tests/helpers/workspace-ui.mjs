export async function prepareWorkspace(page) {
  await page.addInitScript(() => {
    localStorage.setItem('lab.workspace.onboarded','true');
    localStorage.setItem('lab.workspace.intro-dismissed','true');
  });
}
export async function applyDraft(page) {
  await page.waitForFunction(() => !document.body.dataset.loadingPreset);
  if (await page.locator('#pending-settings').isVisible()) {
    // An editor commit shares the same transaction as the settings footer.
    await page.locator(await page.locator('#editor').isVisible() ? '#apply-editor' : '#apply-configuration').click();
    try {
      await page.waitForFunction(() => !document.querySelector('main').inert && document.querySelector('#pending-settings').hidden);
    } catch (error) {
      console.error('Configuration did not apply:', await page.evaluate(() => ({status: document.querySelector('#status').textContent, pending: document.querySelector('#pending-diff').textContent, invalid: [...document.querySelectorAll('input:invalid')].map(e => ({id:e.id, value:e.value})), dialogs:[...document.querySelectorAll('dialog[open]')].map(e=>e.textContent)})));
      throw error;
    }
  }
}
export async function openFiles(page) {
  if (!await page.locator('#files-dialog').isVisible()) await page.locator('#save-menu').click();
}
export async function closeFiles(page) {
  if (await page.locator('#files-dialog').isVisible()) await page.locator('#files-dialog .dialog-close').click();
}
