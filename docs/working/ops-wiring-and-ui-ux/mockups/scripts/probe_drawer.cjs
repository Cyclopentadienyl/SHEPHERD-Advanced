const { chromium } = require('playwright');
const K = require('./common.cjs');
const URL = 'http://127.0.0.1:8871/ui/';
(async () => {
  const b = await chromium.launch();
  const p = await K.open(b, URL);
  await p.click('#shep-dx-change-btn');
  await p.waitForTimeout(2500);
  const rows = await p.$$('#shep-dx-change button.shep-row');
  for (const r of rows) { if ((await r.innerText()).includes('8f670b69d38a')) { await r.click(); break; } }
  await p.waitForTimeout(2500);
  const dump = async () => p.evaluate(() => {
    const walk = (el, d) => { if (d > 6) return ''; const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
      let out = '  '.repeat(d) + el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (typeof el.className === 'string' && el.className ? '.' + el.className.trim().split(/\s+/).filter(c => !c.startsWith('svelte')).slice(0, 4).join('.') : '') + ` ${Math.round(r.left)},${Math.round(r.top)} ${Math.round(r.width)}x${Math.round(r.height)} ov=${s.overflowY} sh=${el.scrollHeight}` + '\n';
      if (el.classList && (el.classList.contains('shep-list') || el.classList.contains('html-container'))) return out;
      for (const c of el.children) out += walk(c, d + 1); return out; };
    return walk(document.getElementById('shep-dx-change'), 0);
  });
  console.log(await dump());
  await K.resize(p, 760, 1000);
  console.log(await dump());
  await p.evaluate(() => document.getElementById('shep-dx-change').scrollIntoView());
  await p.screenshot({ path: K.MK + '/work/base_dx_drawer_760.png' });
  await b.close();
})();
