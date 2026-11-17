const { chromium } = require('playwright');
const BASE = 'http://127.0.0.1:8871/ui/';
const OUT = process.argv[2];
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 });
  const p = await ctx.newPage();
  await p.goto(BASE, { waitUntil: 'domcontentloaded' });
  await p.waitForSelector('#shep-tab-research-button', { timeout: 60000 });
  await p.waitForTimeout(2500);
  await p.click('#shep-dx-change-btn');
  await p.waitForTimeout(2500);
  await p.screenshot({ path: OUT + '/base_dx_drawer.png' });
  const r = await p.$$('#shep-dx-change button.shep-row');
  console.log('drawer rows', r.length);
  if (r.length > 2) { await r[2].click(); await p.waitForTimeout(2000); }
  await p.screenshot({ path: OUT + '/base_dx_drawer_sel.png' });
  await p.click('#shep-tab-research-button');
  await p.click('#shep-page-models-button');
  await p.waitForTimeout(2500);
  await p.click('#shep-mp-order label:has-text("By family")');
  await p.waitForTimeout(2500);
  await p.screenshot({ path: OUT + '/base_models_family.png' });
  const rows = await p.$$('#shep-models-research-rows tbody tr[data-key]');
  console.log('rows', rows.length);
  if (rows.length > 3) { await rows[3].click(); await p.waitForTimeout(2500); }
  await p.screenshot({ path: OUT + '/base_models_family_sel.png' });
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
