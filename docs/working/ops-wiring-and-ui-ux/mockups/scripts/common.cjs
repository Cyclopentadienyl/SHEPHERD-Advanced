// Shared pieces of the T01 mockups: the D01 navigation (option (e), refined),
// the shared cue tokens, opening the real app, and the in-page self-check.
const fs = require('fs');
const path = require('path');

const MK = path.resolve(__dirname, '..');
const OUT = path.join(MK, 'shots');
fs.mkdirSync(OUT, { recursive: true });

const C = '#shep-research-nav > .tab-wrapper > .tab-container';
const T = `${C} > button[role="tab"]`;
const DS = '#shep-page-datasets-button', TS = '#shep-page-test-button', MS = '#shep-page-models-button';
const LBL = `${DS}::before, ${TS}::before`;

// The cue tokens every tab level reads (one place; the values are existing
// colour tokens, no new hex).
const TOKENS = `
:root {
  --shep-row1-h: 46px;
  --shep-tab-h: 32px;
  --shep-tab-hover-bg: var(--shep-grey-50);
  --shep-tab-hover-line: var(--shep-grey-500);
  --shep-tab-selected-line: var(--shep-accent);
  --shep-tab-selected-text: var(--shep-accent-text);
  --shep-focus-ring: var(--shep-accent-600);
  --shep-group-caption: var(--shep-muted);
  --shep-control-border: var(--shep-grey-500);
}
/* UI-10's direction, shown so the mockups' controls meet 3:1: a field's own
   border (Gradio's is 0 px, #f9fafb, 1.05:1) becomes 1 px grey-500. */
.gradio-container {
  --input-border-width: 1px;
  --input-border-color: var(--shep-control-border);
  --input-border-color-hover: var(--shep-grey-600);
  --input-border-color-focus: var(--shep-focus-ring);
  --input-shadow: none;
  --button-secondary-border-color: var(--shep-control-border);
}
`;

// D01: option (e) as decided: captions above each group (sentence case,
// 14 px / 600, muted), 32 px tabs, row 1 at 46 px, hover, selected and focus
// from the tokens; groups wrap as whole groups.
const NAV_CSS = `
${TOKENS}
#shep-nav > .tab-wrapper > .tab-container { padding-top: 5px; padding-bottom: 5px; }
${C} { padding: 22px 16px 0 !important; row-gap: 22px; align-items: flex-end; min-height: 54px; }
${T} {
  height: var(--shep-tab-h); border-top: 0; border-radius: 6px 6px 0 0; padding: 0 12px;
  border-bottom: 3px solid transparent;
}
${T}:hover:not([aria-selected="true"]) {
  background: var(--shep-tab-hover-bg); border-bottom-color: var(--shep-tab-hover-line); color: var(--shep-text);
}
${T}[aria-selected="true"] {
  color: var(--shep-tab-selected-text); border-bottom-color: var(--shep-tab-selected-line); font-weight: 700;
}
${T}:focus-visible, #shep-nav > .tab-wrapper button[role="tab"]:focus-visible {
  outline: 2px solid var(--shep-focus-ring) !important; outline-offset: -2px; box-shadow: none;
}
#shep-nav > .tab-wrapper > .tab-container > button[role="tab"]:focus-visible { outline-offset: 2px; }
${DS} { margin-left: 0; }
${TS} { margin-left: 33px; }
${LBL} {
  left: 12px; right: auto; top: auto; bottom: calc(100% + 2px); height: 18px; line-height: 18px;
  margin: 0; padding: 0; border: 0; align-items: center;
  text-transform: none; letter-spacing: 0; font-size: 14px; font-weight: 600; color: var(--shep-group-caption);
  pointer-events: none;
}
${TS}::after {
  display: block !important; content: ""; position: absolute; left: -17px; right: auto; width: 1px; height: auto;
  top: -20px; bottom: 6px; background: var(--shep-border-2); animation: none; transform: none; z-index: 0;
}
#shep-research-nav.shep-rnav-wrapped > .tab-wrapper > .tab-container { row-gap: 0; }
#shep-research-nav.shep-rnav-wrapped > .tab-wrapper > .tab-container::after {
  display: block !important; content: ""; flex-basis: 100%; height: 22px; order: 1; position: static; background: none;
}
#shep-research-nav.shep-rnav-wrapped ${TS}, #shep-research-nav.shep-rnav-wrapped ${MS} { order: 2; }
#shep-research-nav.shep-rnav-wrapped ${TS} { margin-left: 0; }
#shep-research-nav.shep-rnav-wrapped ${TS}::after { display: none !important; }
`;

// The layout pass the shell would run (as it toggles shep-nav-wrapped for row 1).
const NAV_JS = `
(function () {
  function rnav() {
    var nav = document.getElementById('shep-research-nav');
    var row = nav && nav.querySelector(':scope > .tab-wrapper > .tab-container');
    if (!row) { return; }
    var tabs = row.querySelectorAll(':scope > button[role="tab"]'), need = 32 + 33;
    for (var i = 0; i < tabs.length; i++) { need += tabs[i].offsetWidth; }
    nav.classList.toggle('shep-rnav-wrapped', need > row.clientWidth);
  }
  window.__ui01 = rnav; window.addEventListener('resize', rnav); rnav();
})();`;

async function open(browser, url, w = 1850, h = 960) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#shep-tab-research-button', { timeout: 60000 });
  await page.waitForSelector('#shep-chips .shep-chips', { timeout: 60000 });
  await page.waitForTimeout(2000);
  return page;
}

async function style(page, id, css) {
  await page.evaluate(([id, css]) => {
    let el = document.getElementById(id);
    if (!el) { el = document.createElement('style'); el.id = id; document.head.appendChild(el); }
    el.textContent = css;
  }, [id, css]);
}

async function nav(page) {
  await style(page, 'mk-nav', NAV_CSS);
  await page.evaluate(NAV_JS);
  await page.evaluate(() => { if (window.ShepShell) { window.ShepShell.update(); } if (window.__ui01) { window.__ui01(); } });
}

async function resize(page, w, h) {
  await page.setViewportSize({ width: w, height: h });
  await page.waitForTimeout(600);
  await page.evaluate(() => { if (window.ShepShell) { window.ShepShell.update(); } if (window.__ui01) { window.__ui01(); } });
  await page.waitForTimeout(300);
}

async function shot(page, name, opts = {}) {
  const file = path.join(OUT, name + '.png');
  await page.screenshot({ path: file, ...opts });
  return file;
}

// The self-check run in the page: smallest visible text, horizontal page
// scroll, the scroll regions that can scroll, the header height, and what is
// on top of a given element's centre and corners.
const CHECK = (opts) => {
  const parse = (c) => { const m = c && c.match(/rgba?\(([^)]+)\)/); if (!m) return null; const p = m[1].split(',').map(s => parseFloat(s)); return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 }; };
  const lum = (c) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const ratio = (a, b) => { const la = lum(a), lb = lum(b); return +(((Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)).toFixed(2)); };
  const bgOf = (el) => { for (let e = el; e; e = e.parentElement) { const c = parse(getComputedStyle(e).backgroundColor); if (c && c.a > 0.5) return c; } return { r: 255, g: 255, b: 255, a: 1 }; };
  const vis = (el) => { const r = el.getBoundingClientRect(); if (r.width < 1 || r.height < 1) return false; const s = getComputedStyle(el); return s.visibility !== 'hidden' && s.display !== 'none' && +s.opacity > 0.05 && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth; };
  // Smallest text and lowest text contrast among visible text nodes.
  let minFont = 99, minFontAt = null, minContrast = 99, minContrastAt = null;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    if (!n.textContent.trim()) continue;
    const el = n.parentElement; if (!el || !vis(el)) continue;
    if (el.closest('.shep-vh, [aria-hidden="true"] svg, script, style')) continue;
    // The decorative " · " separator (`.shep-sep`, aria-hidden in the app) is not text that informs.
    if (el.closest('.shep-sep') && n.textContent.trim() === '·') continue;
    const s = getComputedStyle(el); const fs = parseFloat(s.fontSize);
    if (fs < minFont) { minFont = fs; minFontAt = n.textContent.trim().slice(0, 40); }
    const c = parse(s.color); if (c) { const cr = ratio(c, bgOf(el)); if (cr < minContrast) { minContrast = cr; minContrastAt = n.textContent.trim().slice(0, 40) + ' ' + s.color; } }
  }
  const scrollers = [...document.querySelectorAll('*')].filter(e => { const s = getComputedStyle(e); return /(auto|scroll)/.test(s.overflowY) && e.scrollHeight > e.clientHeight + 2 && e.clientHeight > 40 && vis(e); })
    .map(e => ({ id: e.id || (typeof e.className === 'string' ? '.' + e.className.trim().split(/\s+/).slice(0, 2).join('.') : e.tagName), h: e.clientHeight, sh: e.scrollHeight }));
  const nav2 = document.querySelector('#shep-research-nav > .tab-wrapper');
  const shell = document.getElementById('shep-shell');
  const n2 = nav2 && nav2.getBoundingClientRect();
  const out = {
    viewport: [innerWidth, innerHeight],
    hScroll: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    docScroll: document.documentElement.scrollHeight > innerHeight + 1,
    minFont, minFontAt, minContrast, minContrastAt, scrollers,
    headerAndTabsH: n2 && n2.height > 0 ? Math.round(n2.bottom - shell.getBoundingClientRect().top) : null,
  };
  if (opts && opts.covered) {
    out.covered = {};
    for (const sel of opts.covered) {
      const el = document.querySelector(sel); if (!el) { out.covered[sel] = 'absent'; continue; }
      const r = el.getBoundingClientRect();
      const pts = [[r.left + r.width / 2, r.top + r.height / 2], [r.left + 2, r.top + 2], [r.right - 2, r.bottom - 2]];
      out.covered[sel] = pts.map(([x, y]) => { const t = document.elementsFromPoint(x, y)[0]; return !t ? 'nothing' : (el.contains(t) || t.contains(el)) ? 'self' : (t.id || t.className || t.tagName); });
    }
  }
  if (opts && opts.contrast) {
    out.contrast = {};
    for (const [name, sel, prop, against] of opts.contrast) {
      const el = document.querySelector(sel); if (!el) { out.contrast[name] = 'absent'; continue; }
      const s = getComputedStyle(el, prop && prop.startsWith('::') ? prop : null);
      const what = prop && !prop.startsWith('::') ? s.getPropertyValue(prop) : s.color;
      const c = parse(what); const bg = against ? parse(against) : bgOf(el.parentElement || el);
      out.contrast[name] = c ? { color: what, against: against || 'parent bg', ratio: ratio(c, bg) } : what;
    }
  }
  return out;
};

async function check(page, opts) { return page.evaluate(CHECK, opts || {}); }

function save(name, obj) { fs.writeFileSync(path.join(OUT, name + '.json'), JSON.stringify(obj, null, 1)); }

module.exports = { MK, OUT, NAV_CSS, NAV_JS, TOKENS, open, style, nav, resize, shot, check, save };
