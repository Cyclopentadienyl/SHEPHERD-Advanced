// D04 / D05 on the real Models page: the tree list drawn into the page's own
// list region, with the real details panel (a real selection) beside it.
const { chromium } = require('playwright');
const K = require('./common.cjs');
const T = require('./tree.cjs');
const URL = process.argv[2] || 'http://127.0.0.1:8871/ui/';
const SEL = 'gatv2/20261116-122738_demo-disease-extend_19ec67/model-01-0.3133.pt';
const SEL_SHA = '8f670b69d38a';

const PAGE_CSS = `
${T.TREE_CSS}
#shep-page-models #shep-models-research { grid-template-columns: minmax(0, 53fr) minmax(0, 47fr) !important; }
@media (max-width: 1399px), (max-height: 759px) { #shep-page-models #shep-models-research { grid-template-columns: minmax(0, 1fr) !important; } }
.mk-search { display: flex; align-items: center; gap: 8px; flex: 1 1 260px; min-width: 0; max-width: 420px; margin-right: 16px; }
.mk-lbl { font-size: 14px; font-weight: 600; color: var(--shep-text); }
.mk-q-box { position: relative; flex: 1 1 auto; min-width: 0; display: flex; }
.mk-q-box input {
  width: 100%; height: 38px; box-sizing: border-box; padding: 0 36px 0 10px; font: inherit; font-size: 15px; color: var(--shep-text);
  border: 1px solid var(--shep-control-border); border-radius: 4px; background: #fff;
}
.mk-q-box input::placeholder { color: var(--shep-muted); }
.mk-q-box input:focus { outline: 2px solid var(--shep-focus-ring); outline-offset: -1px; }
.mk-q-box input::-webkit-search-cancel-button { display: none; }
.mk-clear { position: absolute; right: 6px; top: 7px; width: 24px; height: 24px; border: 1px solid transparent; border-radius: 6px; background: none; color: var(--shep-text-2); font-size: 18px; line-height: 20px; cursor: pointer; padding: 0; }
.mk-clear:hover { background: var(--shep-grey-50); border-color: var(--shep-grey-500); }
.mk-countline { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; font-size: 14px; color: var(--shep-text-2); }
.mk-note { margin-top: 6px; }
.mk-link { border: 0; background: none; padding: 0; min-height: 24px; color: var(--shep-accent-text); font: inherit; font-size: 14px; cursor: pointer; text-decoration: underline; text-underline-offset: 2px; }
.mk-pending { display: inline-flex; align-items: center; gap: 0; }
`;

async function inject(page, st) {
  const tree = T.render(st.tree);
  await page.evaluate(([st, tree]) => {
    // The search box, before the order control (once).
    const row = document.querySelector('#shep-mp-controls .shep-mp-order-row');
    let box = document.getElementById('mk-search');
    if (!box) {
      box = document.createElement('div'); box.id = 'mk-search'; box.className = 'mk-search';
      box.innerHTML = '<label for="mk-q" class="mk-lbl">Search</label><span class="mk-q-box"><input id="mk-q" type="search" autocomplete="off" placeholder="Name or fingerprint"><button type="button" class="mk-clear" aria-label="Clear search">×</button></span>';
      row.prepend(box);
    }
    const q = document.getElementById('mk-q'); q.value = st.query || '';
    box.querySelector('.mk-clear').style.visibility = st.query ? 'visible' : 'hidden';
    // The order control as the state says (a label class, as Gradio draws it).
    if (st.order) {
      document.querySelectorAll('#shep-mp-order label').forEach(l => {
        const on = l.textContent.trim() === st.order;
        l.classList.toggle('selected', on); const i = l.querySelector('input'); if (i) i.checked = on;
      });
    }
    document.querySelector('#shep-models-research-count .html-container').innerHTML = st.countHtml;
    document.querySelector('#shep-models-research-rows .html-container').innerHTML = tree.html;
  }, [st, tree]);
  return tree;
}

const count = (t) => `${t.shownFams} families, ${t.shownRuns} runs, ${t.shownFiles ? t.shownFiles + ' saved files shown' : 'files folded'}`;

async function locate(page) {
  await page.evaluate(() => { const r = document.querySelector('.mk-tree tr.shep-on'); if (r) r.scrollIntoView({ block: 'center' }); });
}

(async () => {
  const browser = await chromium.launch();
  const page = await K.open(browser, URL);
  await page.click('#shep-tab-research-button');
  await page.click('#shep-page-models-button');
  await page.waitForSelector('#shep-models-research-rows table', { timeout: 60000 });
  await page.waitForTimeout(1500);
  await K.nav(page);
  await K.style(page, 'mk-models', PAGE_CSS);
  await page.click('#shep-mp-order label:has-text("By family")');
  await page.waitForTimeout(2500);
  await page.mouse.move(5, 955);
  const res = {};
  const covered = ['#shep-models-research-load', '#mk-q'];
  const both = async (name, st, after) => {
    for (const [w, h] of [[1850, 960], [760, 1000]]) {
      await K.resize(page, w, h);
      await page.evaluate(() => window.scrollTo(0, 0));
      const t = await inject(page, st);
      if (after) { await after(w); }
      await page.waitForTimeout(250);
      await K.shot(page, `${name}_${w}x${h}`);
      res[`${name}_${w}`] = { ...(await K.check(page, { covered })), tree: { runs: t.shownRuns, files: t.shownFiles, fams: t.shownFams } };
    }
    await K.resize(page, 1850, 960);
  };

  // A. Default: by family, every run's files folded, nothing selected.
  let t = T.render({});
  await both('D04_models_default', { order: 'By family', query: '', tree: {},
    countHtml: `<span class="mk-countline">2 families, 8 runs, 16 saved files (folded)</span>` });

  // B. A file selected: its family and run unfolded and located; the other
  //    family folded by hand.
  // A fresh page, so the click below lands on the real list's row.
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#shep-tab-research-button', { timeout: 60000 });
  await page.waitForTimeout(2000);
  await page.click('#shep-tab-research-button');
  await page.click('#shep-page-models-button');
  await page.waitForSelector('#shep-models-research-rows table', { timeout: 60000 });
  await page.waitForTimeout(1500);
  await K.nav(page);
  await K.style(page, 'mk-models', PAGE_CSS);
  await page.click('#shep-mp-order label:has-text("By family")');
  await page.waitForTimeout(2500);
  const rows = await page.$$('#shep-models-research-rows tbody tr[data-key]');
  console.log('real rows', rows.length);
  for (const r of rows) { if ((await r.innerText()).includes(SEL_SHA)) { await r.click(); break; } }
  await page.waitForTimeout(3000);
  const folded = new Set(['20261116-122745_demo-base-b_ea55fd']);
  const expanded = new Set(['20261116-122738_demo-disease-extend_19ec67']);
  t = T.render({ selected: SEL, expanded, folded });
  await both('D04_models_run-expanded-file-selected', { order: 'By family', query: '', tree: { selected: SEL, expanded, folded },
    countHtml: `<span class="mk-countline">2 families, 8 runs, 16 saved files (2 shown)</span>` }, async (w) => { if (w < 1400) await locate(page); });

  // The real details of the selection, kept for the states below (D04: a
  // selection outside the filter keeps its details; today's page drops it).
  const kept = await page.evaluate(() => ({ detail: document.getElementById('shep-mp-detail').innerHTML,
    load: document.getElementById('shep-models-research-load').outerHTML }));
  const restore = async () => page.evaluate((k) => {
    document.getElementById('shep-mp-detail').innerHTML = k.detail;
    document.getElementById('shep-models-research-load').outerHTML = k.load;
  }, kept);

  // C. A search by name text, and one by a fingerprint prefix.
  t = T.render({ selected: SEL, expanded, query: 'disease' });
  await both('D04_models_search-name', { order: 'By family', query: 'disease', tree: { selected: SEL, expanded, query: 'disease' },
    countHtml: `<span class="mk-countline">${t.matches} runs match “disease”, shown with the runs they come from</span>` });
  t = T.render({ selected: SEL, query: '8f67' });
  await both('D04_models_search-fingerprint', { order: 'By family', query: '8f67', tree: { selected: SEL, query: '8f67' },
    countHtml: `<span class="mk-countline">1 saved file matches fingerprint “8f67…”, shown with the runs it comes from</span>` });

  // D. The selected model outside the Task filter (the real select set to Gene;
  //    the real selection is kept).
  await page.click('#shep-mp-task input');
  await page.click('ul[role=listbox] li:has-text("Gene")');
  await page.waitForSelector('ul[role=listbox]', { state: 'detached' }).catch(() => {});
  await page.waitForTimeout(2500);
  t = T.render({ selected: SEL, expanded, task: 'Gene' });
  await both('D04_models_selected-outside-filter', { order: 'By family', query: '', tree: { selected: SEL, expanded, task: 'Gene' },
    countHtml: `<span class="mk-countline">Gene: 1 of 8 runs, with the run it comes from</span>`
      + `<div class="shep-reason mk-note"><span><span class="shep-st shep-st-neutral"><span class="shep-st-shape" aria-hidden="true">○</span>Selected model outside this filter:</span> a Disease model, its details kept on the right. <button type="button" class="mk-link">Show all tasks</button></span></div>` }, restore);

  // E. D05: the order being applied (By family just chosen; the list still the
  //    previous order, newest first, until the new one is drawn).
  await page.click('#shep-mp-task input');
  await page.click('ul[role=listbox] li:has-text("All")');
  await page.waitForSelector('ul[role=listbox]', { state: 'detached' }).catch(() => {});
  await page.waitForTimeout(2500);
  await both('D05_models_order-pending', { order: 'By family', query: '', tree: { selected: SEL, flat: true },
    countHtml: `<span class="mk-countline" role="status"><span class="shep-st shep-st-info mk-pending"><span class="shep-spinner" aria-hidden="true"></span>Applying By family…</span></span>` }, restore);

  K.save('D04_D05_models_check', res);
  console.log(JSON.stringify(Object.fromEntries(Object.entries(res).map(([k, v]) => [k, { h: v.headerAndTabsH, hs: v.hScroll, doc: v.docScroll, minF: v.minFont, minC: v.minContrast, at: v.minContrastAt, sc: v.scrollers.map(s => s.id + ' ' + s.h + '/' + s.sh), cov: v.covered, tree: v.tree }])), null, 0));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
