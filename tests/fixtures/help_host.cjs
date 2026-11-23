// Drives the shared help bubble (src/webui/components/help.py's HELP_HEAD and
// HELP_CSS) in a real headless browser, on a page the test builds: a
// registered field (a Gradio-like ".block" with its title), one drawn as a
// grid row ("display: contents" on its block, as Diagnosis' Options), a
// key-value row, a table row and a card's heading (a region marked
// "shep-help-region"), each with its ⓘ. For each ⓘ, at each window size, it
// reports what a hover does before and after the open delay (and whether its
// preview shows the "Esc to close" footer), what a click pins (the bubble's
// role and name, its text, its body's parts as the bubble built them, any
// element its text made, the footer), the bubble's rect and the explained
// field's rect, measured here by the test's own rule (not the bubble's), and
// what document.elementsFromPoint finds at the field's centre.
//
// usage: node tests/fixtures/help_host.cjs <page.html>
// prints {"<width>x<height>": {"<which>": {...}}} as JSON;
// tests/unit/test_webui_help.py asserts on it. It needs Playwright (with
// its Chromium) where Node resolves it; the test skips without.
"use strict";
const { chromium } = require("playwright");

const SIZES = [[1850, 960], [1280, 720]];
const TARGETS = {
  field: { button: "#fld button.shep-i", field: "#fld" },
  grid: { button: "#gfld button.shep-i", field: "#gfld > label" },
  kv: { button: ".shep-kv button.shep-i", field: null },
  row: { button: "#tbl tr button.shep-i", field: "#tbl tr" },
  region: { button: "#card .shep-card-title button.shep-i", field: "#card" },
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
        // The preview opens after the delay: waited for by what it shows.
        await page.waitForFunction(() => {
          const b = document.getElementById("shep-help-bubble");
          return !!b && !b.hidden;
        });
        const footerShown = (b) => {
          const ft = b.querySelector(".shep-bubble-ft");
          return !!ft && !ft.hidden && getComputedStyle(ft).display !== "none"
            && ft.getBoundingClientRect().height > 0;
        };
        const hover = await page.evaluate(() => {
          const b = document.getElementById("shep-help-bubble");
          return b && !b.hidden ? b.getAttribute("role") : null;
        });
        const hoverFooter = await page.evaluate(`(${footerShown})(document.getElementById("shep-help-bubble"))`);
        await button.click();
        await page.waitForFunction(() => document.getElementById("shep-help-bubble").getAttribute("role") === "dialog");
        const pinnedFooter = await page.evaluate(`(${footerShown})(document.getElementById("shep-help-bubble"))`);
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
            parts: Array.from(b.querySelector(".shep-bubble-bd").children).map((el) => ({
              tag: el.tagName.toLowerCase(), cls: el.className, text: el.textContent,
              weight: getComputedStyle(el).fontWeight,
              above: parseFloat(getComputedStyle(el).marginTop),
              itemsAbove: Array.from(el.querySelectorAll(":scope > li")).map((li) => parseFloat(getComputedStyle(li).marginTop)),
              items: Array.from(el.querySelectorAll(":scope > li")).map((li) => li.textContent),
              terms: Array.from(el.querySelectorAll("strong")).map((t) => ({
                text: t.textContent, weight: getComputedStyle(t).fontWeight })),
            })),
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
        await page.waitForFunction(() => document.getElementById("shep-help-bubble").hidden);
        size[which] = { hit: { width: hit.width, height: hit.height }, early, hover, hoverFooter,
                        pinnedFooter, pinned, closed };
      }
      out[`${width}x${height}`] = size;
      await page.close();
    }
  } finally {
    await browser.close();
  }
  process.stdout.write(JSON.stringify(out));
})().catch((err) => { process.stderr.write(String(err && err.stack || err)); process.exit(1); });
