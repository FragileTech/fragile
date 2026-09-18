import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
const base = process.env.LECTURE_BASE_URL || 'http://127.0.0.1:8770';
const browser = await chromium.launch({headless: true});
const page = await browser.newPage({viewport:{width:1440,height:1100}});
const errors=[]; page.on('pageerror',e=>errors.push(e.message));
const results=[];
async function ready() {
  await page.waitForFunction(()=>document.querySelector('#status')?.textContent.startsWith('Ready')&&!document.querySelector('#step').disabled,null,{timeout:180000});
  assert.equal(await page.locator('#error').isVisible(),false);
}
try {
  for(let i=1;i<=8;i++) {
    const id=`III-${String(i).padStart(2,'0')}`;
    await page.goto(`${base}/euclidean-gas/lecture.html?demo=${id}`);await ready();
    const before=await page.locator('#charts').innerHTML();
    const parsed=JSON.parse(await page.locator('.calculation-details pre').textContent());
    assert.equal(parsed.details.canonical_kernel,true);
    assert.equal(parsed.details.rooted_mean_field_reference.status,'available');
    assert.equal(typeof parsed.details.rooted_mean_field_reference.prediction_seed,'string');
    assert.equal(parsed.details.rooted_mean_field_reference.independent_root_draws,512);
    assert.ok(await page.locator('#charts svg').count()>0);
    await page.locator('#step').click();await page.waitForFunction(()=>!document.querySelector('#step').disabled,null,{timeout:180000});
    assert.equal(await page.locator('#error').isVisible(),false);
    await page.locator('#reset').click();await ready();
    assert.equal(await page.locator('#charts').innerHTML(),before,`${id} replay`);
    if(id==='III-04') {
      await mkdir('outputs/mean-field-review',{recursive:true});
      await page.screenshot({path:'outputs/mean-field-review/III-04.png',fullPage:true});
    }
    results.push({id,root_draws:512,rendered:true,replay:true});
    console.log(`${id}: Rust rendering and replay passed`);
  }
  assert.deepEqual(errors,[]);console.log(JSON.stringify(results,null,2));
} finally {await browser.close();}
