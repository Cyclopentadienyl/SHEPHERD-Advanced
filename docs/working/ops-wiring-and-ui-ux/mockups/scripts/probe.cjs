const { chromium } = require('playwright');
const BASE = process.argv[2] || 'http://127.0.0.1:8871/ui/';
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1850, height: 960 }, deviceScaleFactor: 1 });
  const p = await ctx.newPage();
  await p.goto(BASE, { waitUntil: 'domcontentloaded' });
  await p.waitForSelector('#shep-tab-research-button', { timeout: 60000 });
  await p.waitForTimeout(2500);
  // a real text input (Patient ID) and its computed style
  const inp = await p.evaluate(() => {
    const out = {};
    const ta = document.querySelector('#shep-tab-clinical input[type=text], #shep-tab-clinical textarea');
    const pick = (el) => { const s = getComputedStyle(el); const k = ['font-size','font-family','padding','border','border-radius','background-color','color','height','box-shadow','line-height']; const o = {}; k.forEach(x => o[x] = s.getPropertyValue(x)); return o; };
    if (ta) { out.input = pick(ta); out.inputTag = ta.tagName; out.blockHTML = ta.closest('.block') ? ta.closest('.block').outerHTML.slice(0, 1500) : null; out.blockStyle = ta.closest('.block') ? pick(ta.closest('.block')) : null; }
    const vars = {};
    const cs = getComputedStyle(document.querySelector('.gradio-container'));
    ['--input-border-color','--input-background-fill','--input-radius','--input-padding','--input-shadow','--input-border-width','--input-text-size','--block-border-color','--block-radius','--button-secondary-background-fill','--button-secondary-border-color','--button-secondary-text-color','--button-small-padding','--button-large-padding','--button-large-radius','--button-small-radius','--font','--button-shadow','--input-border-color-focus','--input-shadow-focus','--layout-gap','--color-accent','--body-text-color'].forEach(v => vars[v] = cs.getPropertyValue(v));
    out.vars = vars;
    out.bodyFont = getComputedStyle(document.body).fontFamily;
    return out;
  });
  console.log(JSON.stringify(inp, null, 1));
  await p.click('#shep-tab-settings-button');
  await p.waitForTimeout(1500);
  const st = await p.evaluate(() => {
    const tab = document.getElementById('shep-tab-settings');
    const walk = (el, d) => { if (d > 4) return ''; let s = '  '.repeat(d) + el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (typeof el.className === 'string' && el.className ? '.' + el.className.trim().split(/\s+/).join('.') : '') + ' ' + Math.round(el.getBoundingClientRect().width) + 'x' + Math.round(el.getBoundingClientRect().height) + '\n'; for (const c of el.children) s += walk(c, d + 1); return s; };
    const btns = [...tab.querySelectorAll('button')].map(b => b.outerHTML.slice(0, 300));
    return walk(tab, 0) + '\n' + btns.join('\n');
  });
  console.log(st);
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
