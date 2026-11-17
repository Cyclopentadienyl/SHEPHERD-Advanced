// D01 navigation mockups: normal, a tab hovered, a tab keyboard-focused (1850x960),
// and narrow (760x1000, 420x900), with the in-page self-check.
const { chromium } = require('playwright');
const K = require('./common.cjs');
const URL = process.argv[2] || 'http://127.0.0.1:8871/ui/';

const NAVCHECK = () => {
  const parse = (c) => { const m = c && c.match(/rgba?\(([^)]+)\)/); if (!m) return null; const p = m[1].split(',').map(s => parseFloat(s)); return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 }; };
  const lum = (c) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const ratio = (a, b) => { const la = lum(a), lb = lum(b); return +(((Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)).toFixed(2)); };
  const white = { r: 255, g: 255, b: 255 };
  const row = document.querySelector('#shep-research-nav > .tab-wrapper > .tab-container');
  const tabs = [...row.querySelectorAll(':scope > button[role="tab"]')].map(b => {
    const s = getComputedStyle(b); const r = b.getBoundingClientRect(); const bg = parse(s.backgroundColor);
    const back = bg && bg.a > 0.5 ? bg : white;
    return { id: b.id.replace('shep-page-', '').replace('-button', ''), h: Math.round(r.height), y: Math.round(r.top), x: Math.round(r.left), w: Math.round(r.width),
      selected: b.getAttribute('aria-selected') === 'true', hover: b.matches(':hover'), focus: b.matches(':focus-visible'),
      text: `${s.fontSize}/${s.fontWeight} ${s.color} ${ratio(parse(s.color), back)}:1`,
      bg: s.backgroundColor,
      line: parse(s.borderBottomColor).a > 0 ? `${s.borderBottomWidth} ${s.borderBottomColor} ${ratio(parse(s.borderBottomColor), back)}:1 on its fill, ${ratio(parse(s.borderBottomColor), white)}:1 on the bar` : 'none',
      outline: s.outlineStyle !== 'none' ? `${s.outlineWidth} ${s.outlineColor} ${ratio(parse(s.outlineColor), back)}:1 inside, ${ratio(parse(s.outlineColor), white)}:1 on the bar` : 'none' };
  });
  const captions = ['shep-page-datasets-button', 'shep-page-test-button'].map(id => {
    const b = document.getElementById(id); const s = getComputedStyle(b, '::before'); const r = b.getBoundingClientRect();
    const x = r.left + 12 + 30, y = r.top - 2 - 9; const top = document.elementsFromPoint(x, y)[0];
    return { text: s.content, font: `${s.fontSize}/${s.fontWeight}`, transform: s.textTransform, color: s.color, contrast: ratio(parse(s.color), white),
      pointerEvents: s.pointerEvents, centre: [Math.round(x), Math.round(y)], under: top ? (top.getAttribute('role') || top.tagName) : null,
      isTab: !!(top && top.closest && top.closest('[role=tab]')), cursor: top ? getComputedStyle(top).cursor : null };
  });
  const shell = document.getElementById('shep-shell').getBoundingClientRect();
  const r1 = document.querySelector('#shep-nav > .tab-wrapper').getBoundingClientRect();
  const r2 = document.querySelector('#shep-research-nav > .tab-wrapper').getBoundingClientRect();
  return { viewport: [innerWidth, innerHeight], row1H: Math.round(r1.height), row2H: Math.round(r2.height), headerAndTabsH: Math.round(r2.bottom - shell.top),
    tabRows: new Set(tabs.map(t => t.y)).size, wrappedClass: document.getElementById('shep-research-nav').classList.contains('shep-rnav-wrapped'),
    hScroll: document.documentElement.scrollWidth > document.documentElement.clientWidth, tabs, captions };
};

(async () => {
  const browser = await chromium.launch();
  const page = await K.open(browser, URL);
  await page.click('#shep-tab-research-button');
  await page.waitForSelector('#shep-page-training-button', { state: 'visible' });
  await page.click('#shep-page-training-button');
  await page.waitForTimeout(1200);
  await K.nav(page);
  await page.mouse.move(1840, 950);
  await page.evaluate(() => document.activeElement && document.activeElement.blur());
  await page.waitForTimeout(400);
  const res = {};
  await K.shot(page, 'D01_nav_1850x960_normal');
  res.normal = await page.evaluate(NAVCHECK);
  res.normal_page = await K.check(page);

  const ds = await page.$eval('#shep-page-datasets-button', b => { const r = b.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; });
  await page.mouse.move(ds[0], ds[1]);
  await page.waitForTimeout(400);
  await K.shot(page, 'D01_nav_1850x960_hover-datasets');
  res.hover = await page.evaluate(NAVCHECK);
  await page.mouse.move(1840, 950);

  // Keyboard: Training is selected; Tab from it reaches Fine-tune.
  await page.focus('#shep-page-training-button');
  await page.keyboard.press('Tab');
  await page.waitForTimeout(400);
  await K.shot(page, 'D01_nav_1850x960_focus-finetune');
  res.focus = await page.evaluate(NAVCHECK);
  res.focus.active = await page.evaluate(() => document.activeElement.id + ' focus-visible=' + document.activeElement.matches(':focus-visible'));
  await page.evaluate(() => document.activeElement && document.activeElement.blur());

  for (const [w, h] of [[760, 1000], [420, 900]]) {
    await K.resize(page, w, h);
    await page.mouse.move(w - 5, h - 5);
    await K.shot(page, `D01_nav_${w}x${h}_narrow`);
    res[`narrow_${w}`] = await page.evaluate(NAVCHECK);
  }
  // Back to wide (narrow -> wide must come back to one row).
  await K.resize(page, 1850, 960);
  res.back_wide = await page.evaluate(NAVCHECK);
  K.save('D01_nav_check', res);
  console.log(JSON.stringify({ normal: { h: res.normal.headerAndTabsH, r1: res.normal.row1H, r2: res.normal.row2H, rows: res.normal.tabRows, captions: res.normal.captions },
    hover: res.hover.tabs.find(t => t.hover), focus: res.focus.tabs.find(t => t.focus), active: res.focus.active,
    selected: res.normal.tabs.find(t => t.selected),
    n760: [res.narrow_760.headerAndTabsH, res.narrow_760.tabRows, res.narrow_760.wrappedClass, res.narrow_760.hScroll],
    n420: [res.narrow_420.headerAndTabsH, res.narrow_420.tabRows, res.narrow_420.wrappedClass, res.narrow_420.hScroll],
    back: [res.back_wide.headerAndTabsH, res.back_wide.tabRows], page: res.normal_page }, null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
