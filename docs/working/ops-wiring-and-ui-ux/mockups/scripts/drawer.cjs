// D04 / UI-07: the Diagnosis "Change model" drawer (Load for inference) with the
// same shared tree list; the list and the chosen model's details each scroll
// on their own and Load stays in reach. Wide: side by side; narrow: stacked.
const { chromium } = require('playwright');
const K = require('./common.cjs');
const T = require('./tree.cjs');
const URL = process.argv[2] || 'http://127.0.0.1:8871/ui/';
const SEL = 'gatv2/20261116-122738_demo-disease-extend_19ec67/model-01-0.3133.pt';
const SEL_SHA = '8f670b69d38a';
const L = '#shep-models-clinical';

const CSS = `
${T.TREE_CSS}
#shep-dx-change { display: flex !important; flex-direction: column; flex-wrap: nowrap !important; }
#shep-dx-change-bd { flex: 1 1 auto !important; min-height: 0; display: flex !important; flex-direction: column; flex-wrap: nowrap !important; }
${L} { flex: 1 1 auto !important; min-height: 0; display: flex !important; flex-direction: column; flex-wrap: nowrap !important; gap: 12px !important; }
${L} > .shep-model-head {
  flex: none !important; display: grid !important; align-items: end !important; gap: 8px 16px !important;
  grid-template-columns: minmax(0, 1fr) auto auto;
  grid-template-areas: "ws task link" "wsline wsline wsline" "search count count";
}
${L} > .shep-model-head > .form { grid-area: ws; min-width: 0 !important; }
${L} > .shep-model-head > #shep-dx-change-task { grid-area: task; }
${L} > .shep-model-head > #shep-dx-compare { grid-area: link; align-self: center; }
${L} > .shep-model-head > #shep-dx-change-ws { grid-area: wsline; margin-top: -4px !important; }
${L} > .shep-model-head > #mk-dsearch { grid-area: search; }
${L} > .shep-model-head > #shep-models-clinical-count { grid-area: count; align-self: center; width: auto !important; }
#shep-dx-compare .shep-dx-link { white-space: nowrap; }
/* The body: two regions, each its own scroll; the foot stays under them. */
${L} > .shep-model-body {
  flex: 1 1 auto !important; min-height: 0; overflow: hidden !important;
  display: grid !important; grid-template-columns: minmax(0, 60fr) minmax(0, 40fr); gap: 16px !important;
}
${L} > .shep-model-body > #shep-models-clinical-rows, ${L} > .shep-model-body > #shep-models-clinical-facts {
  min-height: 0; height: 100%; overflow: auto !important; overscroll-behavior: contain;
}
#shep-models-clinical-rows { scroll-padding-top: 36px; }
#shep-models-clinical-rows .shep-tbl-wrap { overflow: visible; margin: 0; }
#shep-models-clinical-rows .shep-tbl thead th { position: sticky !important; top: 0; z-index: 2; }
#shep-dx-change-note, #shep-dx-change-note .html-container { width: auto !important; min-width: 0 !important; max-width: 100%; flex: 1 1 320px !important; }
#shep-models-clinical-facts { padding-right: 4px; }
${L} > .shep-model-foot { flex: none !important; border-top: 1px solid var(--shep-border); margin: 0 -16px -12px !important; padding: 10px 16px !important; background: var(--shep-panel); }
.mk-search { display: flex; align-items: center; gap: 8px; min-width: 0; }
.mk-lbl { font-size: 14px; font-weight: 600; color: var(--shep-text); }
.mk-q-box { position: relative; flex: 1 1 auto; min-width: 0; display: flex; }
.mk-q-box input { width: 100%; height: 38px; box-sizing: border-box; padding: 0 36px 0 10px; font: inherit; font-size: 15px; color: var(--shep-text); border: 1px solid var(--shep-control-border); border-radius: 4px; background: #fff; }
.mk-q-box input::placeholder { color: var(--shep-muted); }
.mk-clear { position: absolute; right: 6px; top: 7px; width: 24px; height: 24px; border: 1px solid transparent; border-radius: 6px; background: none; color: var(--shep-text-2); font-size: 18px; line-height: 20px; padding: 0; }
.mk-countline { font-size: 14px; color: var(--shep-text-2); }
/* Narrow: the drawer is one screen tall when opened (it scrolls into view),
   the list over the details, each scrolling on its own, Load under them. */
@media (max-width: 1399px), (max-height: 759px) {
  #shep-dx-change { height: calc(100vh - 16px) !important; max-height: calc(100vh - 16px) !important; min-height: 640px; overflow: hidden !important; }
  #shep-dx-change-bd, ${L} { overflow: hidden !important; }
  ${L} > .shep-model-head { grid-template-columns: minmax(0, 1fr) auto; grid-template-areas: "ws ws" "task link" "wsline wsline" "search search" "count count"; }
  ${L} > .shep-model-body { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(160px, 52fr) minmax(160px, 48fr); gap: 0 !important; }
  ${L} > .shep-model-body > #shep-models-clinical-facts { border-top: 1px solid var(--shep-border); padding-top: 10px; }
  ${L} > .shep-model-body > #shep-models-clinical-rows { padding-bottom: 10px; }
}
`;

(async () => {
  const browser = await chromium.launch();
  const page = await K.open(browser, URL);
  await K.nav(page);
  await page.click('#shep-dx-change-btn');
  await page.waitForTimeout(2500);
  for (const r of await page.$$('#shep-dx-change button.shep-row')) { if ((await r.innerText()).includes(SEL_SHA)) { await r.click(); break; } }
  await page.waitForTimeout(3000);
  await K.style(page, 'mk-drawer', CSS);
  const expanded = new Set(['20261116-122738_demo-disease-extend_19ec67']);
  const tree = T.render({ selected: SEL, expanded, preset: 'clinical' });
  await page.evaluate((html) => {
    const head = document.querySelector('#shep-models-clinical > .shep-model-head');
    const s = document.createElement('div'); s.id = 'mk-dsearch'; s.className = 'mk-search';
    s.innerHTML = '<label for="mk-dq" class="mk-lbl">Search</label><span class="mk-q-box"><input id="mk-dq" type="search" autocomplete="off" placeholder="Name or fingerprint"><button type="button" class="mk-clear" aria-label="Clear search" style="visibility:hidden">×</button></span>';
    head.appendChild(s);
    document.querySelector('#shep-models-clinical-count .html-container').innerHTML = '<span class="mk-countline">2 families, 6 runs, 12 saved files (2 shown)</span>';
    document.querySelector('#shep-models-clinical-rows .html-container').innerHTML = html;
  }, tree.html);
  const res = {};
  for (const [w, h] of [[1850, 960], [760, 1000]]) {
    await K.resize(page, w, h);
    await page.evaluate(() => {
      const d = document.getElementById('shep-dx-change');
      if (innerWidth < 1400) { window.scrollTo(0, d.getBoundingClientRect().top + window.scrollY - 8); } else { window.scrollTo(0, 0); }
      const sel = document.querySelector('#shep-models-clinical-rows tr.shep-on');
      const reg = document.getElementById('shep-models-clinical-rows');
      // Located: the selection's run row just under the list's header.
      let run = sel; while (run && !run.classList.contains('mk-run')) { run = run.previousElementSibling; }
      if (sel && reg && run) { const r = sel.getBoundingClientRect(), g = reg.getBoundingClientRect(); if (r.bottom > g.bottom) { const head = reg.querySelector('thead').getBoundingClientRect().height; reg.scrollTop += run.getBoundingClientRect().top - g.top - head; } }
    });
    await page.mouse.move(w - 3, 3);
    await page.waitForTimeout(400);
    await K.shot(page, `D04_drawer_${w}x${h}`);
    res[w] = await K.check(page, { covered: ['#shep-models-clinical-load', '#shep-dx-change-cancel', '#mk-dq'] });
    res[w].regions = await page.evaluate(() => ['shep-models-clinical-rows', 'shep-models-clinical-facts'].map(id => { const e = document.getElementById(id); const r = e.getBoundingClientRect(); return { id, top: Math.round(r.top), h: e.clientHeight, sh: e.scrollHeight, scrolls: e.scrollHeight > e.clientHeight }; }));
    res[w].load = await page.evaluate(() => { const r = document.getElementById('shep-models-clinical-load').getBoundingClientRect(); return { top: Math.round(r.top), bottom: Math.round(r.bottom), inView: r.top >= 0 && r.bottom <= innerHeight }; });
    // The details region scrolled to its end: the list and Load do not move.
    const before = await page.evaluate(() => [document.getElementById('shep-models-clinical-rows').getBoundingClientRect().top, document.getElementById('shep-models-clinical-load').getBoundingClientRect().top]);
    await page.evaluate(() => { const f = document.getElementById('shep-models-clinical-facts'); f.scrollTop = f.scrollHeight; });
    await page.waitForTimeout(200);
    const after = await page.evaluate(() => [document.getElementById('shep-models-clinical-rows').getBoundingClientRect().top, document.getElementById('shep-models-clinical-load').getBoundingClientRect().top]);
    res[w].detailsScrolled = { listTop: [before[0], after[0]], loadTop: [before[1], after[1]] };
    await K.shot(page, `D04_drawer_${w}x${h}_details-scrolled`);
    await page.evaluate(() => { document.getElementById('shep-models-clinical-facts').scrollTop = 0; });
  }
  K.save('D04_drawer_check', res);
  console.log(JSON.stringify(res, null, 1).slice(0, 3500));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
