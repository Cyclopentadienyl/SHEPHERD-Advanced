const { chromium } = require('playwright');
const K = require('./common.cjs');
const URL = 'http://127.0.0.1:8871/ui/';
(async () => {
  const b = await chromium.launch();
  const p = await K.open(b, URL);
  await p.click('#shep-tab-research-button');
  await p.click('#shep-page-models-button');
  await p.waitForTimeout(2500);
  const s = await p.evaluate(() => {
    const walk = (el, d) => { if (d > 7) return ''; const r = el.getBoundingClientRect(); let s = '  '.repeat(d) + el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (typeof el.className === 'string' && el.className ? '.' + el.className.trim().split(/\s+/).slice(0, 4).join('.') : '') + ` ${Math.round(r.left)},${Math.round(r.top)} ${Math.round(r.width)}x${Math.round(r.height)}` + '\n'; if (el.tagName === 'TABLE' || el.tagName === 'SVG') return s; for (const c of el.children) s += walk(c, d + 1); return s; };
    return walk(document.getElementById('shep-page-models'), 0);
  });
  console.log(s);
  await b.close();
})();
