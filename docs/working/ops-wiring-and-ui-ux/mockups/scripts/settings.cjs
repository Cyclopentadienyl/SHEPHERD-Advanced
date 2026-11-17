// D06 / D07: Settings › Data locations, a new panel beside the real Runtime
// settings panel, drawn into the real Settings page (its fonts, tokens and
// Gradio button classes). Six states, each at 1850x960 and 760x1000.
const { chromium } = require('playwright');
const K = require('./common.cjs');

const e = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const FILE = '~/.config/shepherd-advanced/deployment.json';
const LOCK = '🔒 ';

const LOCS = {
  ws: { name: 'Workspaces folder', env: 'SHEPHERD_WORKSPACES_ROOT', suggested: '~/SHEPHERD-data/workspaces',
    used: 'Datasets, runs and saved models. Needed by every Research page, model loading and GPU jobs.' },
  reg: { name: 'Results registry', env: 'SHEPHERD_REGISTRY', suggested: '~/SHEPHERD-data/registry',
    used: 'Cohort versions and registered reports. Needed by Test to measure a cohort and register.' },
  imp: { name: 'Import folder', env: 'SHEPHERD_INTAKE_DIR', suggested: '~/SHEPHERD-data/import',
    used: 'Case files you place here to import as a cohort version (Test › Import).' },
  ev: { name: 'Training evidence folders', env: 'SHEPHERD_TRAINING_EVIDENCE_DIRS', suggested: null, optional: true,
    used: 'Optional. More folders the training/test overlap check (R10) reads.' },
};
const SRC = { saved: 'Saved in Settings', env: 'Set when the service started (environment)', none: 'Not set' };
const mark = (kind, shape, word) => `<span class="shep-st shep-st-${kind}"><span class="shep-st-shape" aria-hidden="true">${shape}</span>${e(word)}</span>`;

function btn(label, { disabled = false, lock = false, primary = false, id = '' } = {}) {
  return `<button type="button" class="sm ${primary ? 'primary' : 'secondary'} __SV__ mk-btn"${id ? ` id="${id}"` : ''}${disabled ? ' disabled' : ''}>${lock ? LOCK : ''}${e(label)}</button>`;
}

// One location's row.
function row(key, r, ro) {
  const L = LOCS[key];
  const id = `mk-${key}`;
  const kv = [];
  // In effect
  let effect;
  if (r.effect === null) {
    effect = `<span>${mark('neutral', '○', L.optional ? 'None set' : 'Not set')}${L.optional ? '<span class="mk-wait">: the check reads the measured workspace only</span>' : ''}</span>`;
  } else if (r.redacted) {
    effect = `<span>Set <span class="mk-hidden">(path hidden)</span></span>`;
  } else {
    effect = (Array.isArray(r.effect) ? r.effect : [r.effect]).map(v => L.optional
      ? `<span class="mk-li"><code class="shep-full mk-path-v">${e(v)}</code>${btn('Remove', { disabled: ro, lock: ro }).replace('mk-btn', 'mk-btn mk-btn-s')}</span>`
      : `<code class="shep-full mk-path-v">${e(v)}</code>`).join('');
  }
  const src = r.effect === null ? '' : `<span class="mk-src">${e(SRC[r.src])}</span>`;
  kv.push(`<dt class="shep-k">In effect</dt><dd class="shep-v mk-eff"><span class="mk-eff-v">${effect}</span>${src}</dd>`);
  // Saved, when it differs or is pending
  if (r.saved) {
    const s = r.saved;
    const status = s.status === 'pending-job' ? mark('warn', '▲', 'Saved, applies when the running job ends')
      : s.status === 'override' ? `<span class="mk-note-t">Saved, applies when the start-up setting is removed</span>`
      : s.status === 'unknown' ? mark('neutral', '○', 'Unknown: the settings file cannot be read') : '';
    const val = s.value ? `<code class="shep-full mk-path-v">${e(s.value)}</code>` : '';
    const apply = s.status === 'pending-job' ? btn('Apply', { disabled: true }) : '';
    kv.push(`<dt class="shep-k">Saved</dt><dd class="shep-v mk-saved">${val}<span class="mk-saved-s">${status}</span>${apply}</dd>`);
  }
  // The path box with Check and Save (or Add for the list)
  const inp = r.input || {};
  const dis = ro || inp.disabled;
  const verb = L.optional ? 'Add' : 'Save';
  const lbl = L.optional ? 'Add folder' : (r.effect === null ? 'Set to' : 'Change to');
  kv.push(`<dt class="shep-k"><label for="${id}-in">${lbl}</label></dt><dd class="shep-v mk-edit">`
    + `<input id="${id}-in" class="mk-path" type="text" spellcheck="false" autocomplete="off" placeholder="Folder path on the server" value="${e(inp.value || '')}"${dis ? ' disabled' : ''}>`
    + btn('Check', { disabled: dis || !inp.value, lock: ro }) + btn(verb, { disabled: dis || inp.saveOff || !inp.value, lock: ro })
    + `</dd>`);
  if (inp.check) {
    const c = inp.check;
    kv.push(`<dt class="shep-k mk-k0"></dt><dd class="shep-v mk-check" role="status">${c.ok ? mark('ok', '●', c.word + ':') : mark('error', '✕', c.word + ':')}<span class="mk-check-t">${e(c.text)}</span></dd>`);
  }
  if (inp.note) {
    kv.push(`<dt class="shep-k mk-k0"></dt><dd class="shep-v mk-check"><span class="mk-note-t">${e(inp.note)}</span></dd>`);
  }
  // Create the suggested folder (an unset, required folder)
  if (r.effect === null && L.suggested && !r.noCreate) {
    kv.push(`<dt class="shep-k mk-k0"></dt><dd class="shep-v mk-create">${btn('Create suggested folder', { disabled: ro, lock: ro })}<code class="shep-full" title="/home/ops/SHEPHERD-data/${e(L.suggested.split('/').pop())}">${e(L.suggested)}</code></dd>`);
  }
  const adv = `<details class="shep-more mk-adv"${r.advOpen ? ' open' : ''}><summary>Advanced: source ›</summary>`
    + `<dl class="shep-kv mk-adv-kv"><dt class="shep-k">Start-up setting</dt><dd class="shep-v"><code>${e(L.env)}</code> ${r.src === 'env' ? 'set' : 'not set'}</dd>`
    + `<dt class="shep-k">Settings file</dt><dd class="shep-v"><code>${e(FILE)}</code>${r.fileWord ? ' ' + e(r.fileWord) : ''}</dd>`
    + `<dt class="shep-k">Order</dt><dd class="shep-v">start-up setting, then the saved value</dd></dl></details>`;
  return `<section class="mk-loc" aria-labelledby="${id}-h"><div class="mk-loc-name"><h3 id="${id}-h">${e(L.name)}</h3><p class="mk-used">${e(L.used)}</p></div>`
    + `<div class="mk-loc-body"><dl class="shep-kv mk-loc-kv">${kv.join('')}</dl>${adv}</div></section>`;
}

function panel(st) {
  const rows = ['ws', 'reg', 'imp', 'ev'].map(k => row(k, st.rows[k], st.readonly)).join('');
  return `<div id="mk-dl" class="shep-panel mk-dl" role="region" aria-labelledby="mk-dl-h">`
    + `<div class="shep-panel-hd"><h2 id="mk-dl-h">Data locations</h2><span class="shep-panel-right">where this service keeps its data</span></div>`
    + `<p class="shep-intro">These are folders on the server machine: this computer when SHEPHERD was started from its desktop.</p>`
    + (st.notice ? `<div class="mk-notice">${st.notice}</div>` : '')
    + `<div class="mk-rows">${rows}</div></div>`;
}

const CSS = `
#shep-tab-settings > .column { flex-direction: row !important; align-items: flex-start !important; gap: 16px !important; flex-wrap: nowrap !important; }
#shep-tab-settings > .column > #mk-dl { flex: 55 1 0; min-width: 0; }
#shep-tab-settings > .column > #shep-settings { flex: 45 1 0 !important; min-width: 0 !important; width: auto !important; max-width: none !important; }
@media (max-width: 1399px), (max-height: 759px) {
  #shep-tab-settings > .column { flex-direction: column !important; align-items: stretch !important; }
  #shep-tab-settings > .column > #mk-dl, #shep-tab-settings > .column > #shep-settings { flex: none !important; width: 100% !important; }
}
.mk-dl .shep-intro { padding: 10px 16px 2px; }
.mk-notice { padding: 10px 16px 0; }
.mk-notice .shep-reason { line-height: 1.45; }
.mk-rows { display: flex; flex-direction: column; padding: 4px 16px 6px; }
.mk-loc { display: grid; grid-template-columns: minmax(200px, 270px) minmax(0, 1fr); gap: 4px 24px; padding: 10px 0; border-top: 1px solid var(--shep-border); }
.mk-loc:first-child { border-top: 0; }
.mk-loc h3 { font-size: 15px; font-weight: 600; margin: 0 0 2px; color: var(--shep-text); }
.mk-used { margin: 0; font-size: 14px; color: var(--shep-text-2); line-height: 1.4; }
.mk-loc-body { min-width: 0; display: flex; flex-direction: column; gap: 4px; }
.mk-adv { font-size: 14px; }
.mk-adv-kv { margin: 2px 0 0; font-size: 14px; gap: 2px 12px; }
.mk-adv-kv > .shep-v { font-size: 14px; }
.mk-loc-kv { gap: 8px 14px; align-items: center; }
.mk-loc-kv > .shep-k { min-width: 6.5em !important; }
.mk-eff, .mk-saved { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 12px; }
.mk-eff-v { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.mk-path-v { font-size: 14px; color: var(--shep-text); }
.mk-src { font-size: 14px; color: var(--shep-text-2); margin-left: auto; text-align: right; }
.mk-wait { font-size: 14px; color: var(--shep-text-2); }
.mk-hidden { font-size: 14px; color: var(--shep-text-2); }
.mk-saved-s { font-size: 14px; }
.mk-note-t { font-size: 14px; color: var(--shep-text-2); }
.mk-edit { display: flex; gap: 8px; align-items: center; min-width: 0; }
.mk-path {
  flex: 1 1 auto; min-width: 0; height: 39px; box-sizing: border-box; padding: 0 10px; font: inherit; font-size: 15px;
  color: var(--shep-text); background: #fff; border: 1px solid var(--shep-control-border); border-radius: 4px;
}
.mk-path::placeholder { color: var(--shep-muted); }
.mk-path:focus { outline: 2px solid var(--shep-focus-ring); outline-offset: -1px; }
.mk-path:disabled { background: var(--shep-grey-50); color: var(--shep-text-2); border-color: var(--shep-grey-200); }
.gradio-container .mk-btn { height: 39px; min-width: 0; flex: none; padding: 0 14px; font-size: 15px; white-space: nowrap; }
.mk-li { display: flex; align-items: center; gap: 10px; }
.gradio-container .mk-btn.mk-btn-s { height: 28px; padding: 0 10px; font-size: 14px; }
.mk-check { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; font-size: 14px; margin-top: -2px; }
.mk-check .shep-st { font-size: 14px; }
.mk-check-t { color: var(--shep-text-2); }
.mk-create { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; }
.mk-k0 { min-height: 0; }
.mk-notice .shep-reason-grey, .mk-notice .shep-notice-red { display: block; }
.mk-replace { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin-top: 8px; }
@media (max-width: 1399px), (max-height: 759px) {
  .mk-loc { grid-template-columns: minmax(0, 1fr); }
  .mk-src { margin-left: 0; text-align: left; flex-basis: 100%; }
}
`;

const WS = '/home/ops/SHEPHERD-data/workspaces', REG = '/home/ops/SHEPHERD-data/registry', IMP = '/home/ops/SHEPHERD-data/import';

const STATES = {
  // 1. First start: nothing set anywhere; the dashboard works.
  first_start: { url: 'http://127.0.0.1:8872/ui/',
    notice: `<div class="shep-reason">${mark('neutral', '○', 'Nothing is set yet.')}<span>Each folder is needed only by what its row names; Settings and every other page open as usual.</span></div>`,
    rows: { ws: { effect: null, src: 'none', fileWord: '(not written yet)' }, reg: { effect: null, src: 'none' }, imp: { effect: null, src: 'none' }, ev: { effect: null, src: 'none' } } },
  // 2. Configured: every folder saved in Settings; one new path checked.
  configured: { url: 'http://127.0.0.1:8871/ui/',
    rows: { ws: { effect: WS, src: 'saved', advOpen: false }, reg: { effect: REG, src: 'saved' },
      imp: { effect: IMP, src: 'saved', input: { value: '/home/ops/SHEPHERD-data/import-2027', check: { ok: true, word: 'Usable', text: 'an existing folder outside the app folder; this service can read it' } } },
      ev: { effect: ['/mnt/lab/run-inputs-2026', '/mnt/lab/run-inputs-archive'], src: 'saved' } } },
  // 3. A row overridden at start: the start-up setting wins; a saved value waits.
  overridden: { url: 'http://127.0.0.1:8871/ui/',
    rows: { ws: { effect: WS, src: 'saved' },
      reg: { effect: '/mnt/lab/shepherd-registry', src: 'env', advOpen: true, saved: { value: REG, status: 'override' } },
      imp: { effect: IMP, src: 'saved' }, ev: { effect: null, src: 'none' } } },
  // 4. A change saved while a job runs.
  pending: { url: 'http://127.0.0.1:8871/ui/', gpuBusy: true,
    rows: { ws: { effect: WS, src: 'saved', saved: { value: '/data/shepherd/workspaces', status: 'pending-job' } },
      reg: { effect: REG, src: 'saved' }, imp: { effect: IMP, src: 'saved' }, ev: { effect: null, src: 'none' } } },
  // 5. The settings file cannot be read: nothing from it is used; one repair action.
  unreadable: { url: 'http://127.0.0.1:8871/ui/',
    notice: `<div class="shep-notice shep-notice-red" role="alert">${mark('error', '✕', 'Settings file cannot be read: ')}the folders saved in it are unknown and not used; Save is off until the file is replaced. Folders set when the service started still apply.`
      + `<div class="mk-replace">${btn('Replace the unreadable file (a copy is kept)')}<span class="mk-note-t">File: <code>${e(FILE)}</code>; the copy is kept beside it.</span></div></div>`,
    rows: { ws: { effect: '/mnt/lab/shepherd-workspaces', src: 'env', fileWord: '(cannot be read)', input: { saveOff: true, value: '' } },
      reg: { effect: null, src: 'none', noCreate: true, fileWord: '(cannot be read)', input: { saveOff: true } },
      imp: { effect: null, src: 'none', noCreate: true, fileWord: '(cannot be read)', input: { saveOff: true } },
      ev: { effect: null, src: 'none', fileWord: '(cannot be read)', input: { saveOff: true } } } },
  // 6. Read-only: values hidden, controls off with the visible reason (a real
  //    not-M1 deployment: its chip and Runtime settings are the app's own).
  readonly: { url: 'http://127.0.0.1:8873/ui/', readonly: true,
    notice: `<div class="shep-reason"><span>${LOCK}Operator controls only: deployment mode not set to M1. Folder paths are shown to operator controls only.</span></div>`,
    rows: { ws: { effect: 'x', redacted: true, src: 'env' }, reg: { effect: 'x', redacted: true, src: 'env' }, imp: { effect: null, src: 'none' }, ev: { effect: null, src: 'none' } } },
};

async function draw(page, st) {
  await page.click('#shep-tab-settings-button');
  await page.waitForSelector('#shep-settings', { state: 'visible' });
  await page.waitForTimeout(800);
  await K.nav(page);
  await K.style(page, 'mk-settings', CSS);
  await page.evaluate(([html, busy]) => {
    const sv = [...document.getElementById('runtime_restart').classList].find(c => c.startsWith('svelte-')) || '';
    const old = document.getElementById('mk-dl'); if (old) old.remove();
    const host = document.querySelector('#shep-tab-settings > .column');
    const t = document.createElement('div'); t.innerHTML = html.replace(/__SV__/g, sv);
    host.insertBefore(t.firstChild, document.getElementById('shep-settings'));
    if (busy) {
      const chips = [...document.querySelectorAll('#shep-chips .shep-chip')];
      const gpu = chips.find(c => /GPU/.test(c.textContent));
      if (gpu) { gpu.className = 'shep-chip shep-chip-busy'; const tx = gpu.querySelector('.shep-chip-text'); if (tx) tx.textContent = 'GPU job: training (job runner)'; }
    }
  }, [panel(st), !!st.gpuBusy]);
}

(async () => {
  const browser = await chromium.launch();
  const res = {};
  const only = process.argv[2] ? process.argv[2].split(',') : Object.keys(STATES);
  for (const name of only) {
    const st = STATES[name];
    const page = await K.open(browser, st.url);
    for (const [w, h] of [[1850, 960], [760, 1000]]) {
      await K.resize(page, w, h);
      await draw(page, st);
      await page.mouse.move(w - 3, h - 3);
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(300);
      await K.shot(page, `D06_settings_${name}_${w}x${h}`);
      res[`${name}_${w}`] = await K.check(page, { covered: ['#mk-ws-in', '#mk-reg-in'],
        contrast: [['path box border', '#mk-ws-in', 'border-top-color', 'rgb(255, 255, 255)'], ['row divider', '.mk-loc + .mk-loc', 'border-top-color', 'rgb(255, 255, 255)']] });
      res[`${name}_${w}`].panel = await page.evaluate(() => { const r = document.getElementById('mk-dl').getBoundingClientRect(); const s = document.getElementById('shep-settings').getBoundingClientRect(); return { dl: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)], runtime: [Math.round(s.left), Math.round(s.top), Math.round(s.width), Math.round(s.height)] }; });
    }
    await page.context().close();
  }
  // First start, a page that waits for a folder: Models says so in plain
  // words with one way on (today it names SHEPHERD_WORKSPACES_ROOT).
  if (only.includes('first_start')) {
    const page = await K.open(browser, STATES.first_start.url);
    await page.click('#shep-tab-research-button');
    await page.click('#shep-page-models-button');
    await page.waitForTimeout(2500);
    await K.nav(page);
    await K.style(page, 'mk-fs', `.mk-link { border: 0; background: none; padding: 0; min-height: 24px; color: var(--shep-accent-text); font: inherit; font-size: 15px; cursor: pointer; text-decoration: underline; text-underline-offset: 2px; }`);
    for (const [w, h] of [[1850, 960], [760, 1000]]) {
      await K.resize(page, w, h);
      const found = await page.evaluate(() => {
        const el = [...document.querySelectorAll('#shep-page-models .html-container *')].find(n => n.children.length === 0 && /SHEPHERD_WORKSPACES_ROOT is not set/.test(n.textContent))
          || document.getElementById('mk-fs-empty');
        if (!el) return false;
        el.id = 'mk-fs-empty';
        el.innerHTML = '<span class="shep-st shep-st-neutral"><span class="shep-st-shape" aria-hidden="true">○</span>No workspaces folder is set,</span> so no workspace or model is listed here yet. <button type="button" class="mk-link">Set one in Settings › Data locations</button>';
        return true;
      });
      await page.mouse.move(w - 3, h - 3);
      await page.waitForTimeout(300);
      await K.shot(page, `D06_settings_first_start_models-page_${w}x${h}`);
      res[`first_start_models_${w}`] = { replaced: found, ...(await K.check(page)) };
    }
    await page.context().close();
  }
  K.save('D06_settings_check', res);
  console.log(JSON.stringify(Object.fromEntries(Object.entries(res).map(([k, v]) => [k, { hs: v.hScroll, doc: v.docScroll, minF: v.minFont, minC: v.minContrast, at: v.minContrastAt, sc: v.scrollers.map(s => s.id + ' ' + s.h + '/' + s.sh), panel: v.panel, con: v.contrast }]))));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });

module.exports = { panel, STATES };
