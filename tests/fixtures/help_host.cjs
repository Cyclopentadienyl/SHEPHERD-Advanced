// Drives the shared help bubble (src/webui/components/help.py's HELP_HEAD and
// HELP_CSS) in a real headless browser, on a page the test builds: a
// registered field (a Gradio-like ".block" with its title), a key-value row
// and a table row, each with its ⓘ. For each ⓘ, at each window size, it
// reports what a hover does before and after the open delay, what a click
// pins (the bubble's role and name, its text, any element its text made),
// the bubble's rect and the explained field's rect, measured here by the
// test's own rule (not the bubble's), and what document.elementsFromPoint
// finds at the field's centre.
//
// usage: node tests/fixtures/help_host.cjs <page.html>
// prints {"<width>x<height>": {"<which>": {...}}} as JSON;
// tests/unit/test_webui_help.py asserts on it. It needs Playwright (with
// its Chromium) where Node resolves it; the test skips without.
"use strict";
const { chromium } = require("playwright");

const SIZES = [[1850, 960], [420, 800]];
const TARGETS = {
  field: { button: "#fld button.shep-i", field: "#fld" },
  kv: { button: ".shep-kv button.shep-i", field: null },
  row: { button: "#tbl tr button.shep-i", field: "#tbl tr" },
};

(async () => {
  const file = process.argv[2];
  const browser = await chromium.launch();
  const out = {};
  try {
    for (const [width, height] of SIZES) {
      const page = await browser.newPage({ viewport: { width, height } });
      await page.goto("file://" + file);
      await page.waitForSelector(TARGETS.field.button);
      const size = {};
      for (const [which, t] of Object.entries(TARGETS)) {
        const button = page.locator(t.button);
        await button.scrollIntoViewIfNeeded();
        const hit = await button.boundingBox();
        await page.mouse.move(hit.x + hit.width / 2, hit.y + hit.height / 2);
        await page.waitForTimeout(80);
        const early = await page.evaluate(() => {
          const b = document.getElementById("shep-help-bubble");
          return !!b && !b.hidden;
        });
        await page.waitForTimeout(500);
        const hover = await page.evaluate(() => {
          const b = document.getElementById("shep-help-bubble");
          return b && !b.hidden ? b.getAttribute("role") : null;
        });
        await button.click();
        const pinned = await page.evaluate((fieldSel) => {
          const b = document.getElementById("shep-help-bubble");
          const box = (r) => ({ left: r.left, top: r.top, right: r.right, bottom: r.bottom });
          let field;
          if (fieldSel) {
            field = box(document.querySelector(fieldSel).getBoundingClientRect());
          } else {
            const k = document.querySelector(".shep-kv .shep-k").getBoundingClientRect();
            const v = document.querySelector(".shep-kv .shep-v").getBoundingClientRect();
            field = { left: Math.min(k.left, v.left), top: Math.min(k.top, v.top),
                      right: Math.max(k.right, v.right), bottom: Math.max(k.bottom, v.bottom) };
          }
          const r = b.getBoundingClientRect();
          const cx = (field.left + field.right) / 2, cy = (field.top + field.bottom) / 2;
          const atCentre = document.elementsFromPoint(cx, cy);
          return {
            hidden: b.hidden, role: b.getAttribute("role"),
            labelledby: b.getAttribute("aria-labelledby"),
            title: (document.getElementById(b.getAttribute("aria-labelledby") || "") || {}).textContent || null,
            text: b.querySelector(".shep-bubble-bd").textContent,
            made: b.querySelectorAll("img, script").length,
            pwned: window.__pwned === true,
            bubble: box(r), field,
            bubbleAtFieldCentre: atCentre.some((el) => b.contains(el)),
            width: r.width,
            viewport: { width: document.documentElement.clientWidth,
                        height: document.documentElement.clientHeight },
          };
        }, t.field);
        await page.keyboard.press("Escape");
        const closed = await page.evaluate(() => {
          const b = document.getElementById("shep-help-bubble");
          return b.hidden && b.getAttribute("role") === "tooltip";
        });
        await page.mouse.move(1, 1);
        await page.waitForTimeout(350);
        size[which] = { hit: { width: hit.width, height: hit.height }, early, hover, pinned, closed };
      }
      out[`${width}x${height}`] = size;
      await page.close();
    }
  } finally {
    await browser.close();
  }
  process.stdout.write(JSON.stringify(out));
})().catch((err) => { process.stderr.write(String(err && err.stack || err)); process.exit(1); });
