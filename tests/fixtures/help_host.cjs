// Drives the shared help bubble (src/webui/components/help.py's HELP_HEAD and
// HELP_CSS) in a real headless browser, on a page the test builds: a
// registered field (a Gradio-like ".block" with its title), one drawn as a
// grid row ("display: contents" on its block, as Diagnosis' Options), a
// key-value row, a table row and a card's heading (a region marked
// "shep-help-region"), each with its ⓘ. For each ⓘ, at each window size, it
// reports what a hover does before and after the open delay (and whether its
// preview shows the Esc keycap beside its ×, and its close control's size),
// what a click pins (the bubble's role and name, its text, its body's parts as
// the bubble built them, any element its text made, its head and close
// control, `closeOf`), the bubble's rect and the explained field's rect,
// measured here by the test's own rule (not the bubble's), what
// document.elementsFromPoint finds at the field's centre, and how it closed
// (the field's by a click on its Esc keycap, the others by Esc; the table
// row's close control focused after a key press first, to read its focus
// ring). Then, at each size, a ⓘ whose title wraps and whose text is taller
// than the window ("long"): its head and its capped body.
//
// usage: node tests/fixtures/help_host.cjs <page.html>
// prints {"<width>x<height>": {"<which>": {...}}, "long": {"<width>x<height>": {...}}}
// as JSON; tests/unit/test_webui_help.py asserts on it. It needs Playwright
// (with its Chromium) where Node resolves it; the test skips without.
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
const LONG = "#long button.shep-i";
const KEY = "#shep-help-bubble .shep-bubble-x > .shep-kbd";

// The pinned bubble's head and body: its rows, its one close control (the keycap and the ×
// inside it, their boxes, the keycap's frame and size, the button's name and key), the title's
// lines, and the room under its body.
function closeOf() {
  const b = document.getElementById("shep-help-bubble");
  const q = (e) => { const r = e.getBoundingClientRect(); return { left: r.left, top: r.top, right: r.right, bottom: r.bottom }; };
  const hd = b.querySelector(".shep-bubble-hd"), x = hd.querySelector("button.shep-bubble-x");
  const title = hd.querySelector(".shep-bubble-title"), bd = b.querySelector(".shep-bubble-bd");
  const k = x && x.querySelector(".shep-kbd"), mark = x && x.querySelector(".shep-bubble-x-mark");
  const range = document.createRange(); range.selectNodeContents(title);
  const lines = [...range.getClientRects()].filter((r) => r.width > 0);
  const ks = k ? getComputedStyle(k) : null, bs = getComputedStyle(b), r = b.getBoundingClientRect();
  return {
    rows: [...b.children].map((c) => c.className),
    foot: !!b.querySelector(".shep-bubble-ft") || /to close/.test(b.textContent),
    label: x && x.getAttribute("aria-label"), keys: x && x.getAttribute("aria-keyshortcuts"),
    keyHidden: k ? k.getAttribute("aria-hidden") : null, keyText: k ? k.textContent : null,
    keySize: ks ? parseFloat(ks.fontSize) : null,
    keyFrame: ks ? [ks.borderTopStyle, parseFloat(ks.borderTopWidth), parseFloat(ks.borderBottomWidth)] : null,
    markText: mark ? mark.textContent : null,
    close: x && q(x), key: k && k.getClientRects().length ? q(k) : null, mark: mark ? q(mark) : null,
    head: q(hd), title: q(title), titleLines: new Set(lines.map((l) => Math.round(l.top))).size,
    firstLine: lines.length ? { top: lines[0].top, bottom: lines[0].bottom } : null,
    body: q(bd), bubble: q(b), innerBottom: r.bottom - parseFloat(bs.borderBottomWidth) - parseFloat(bs.paddingBottom),
    headGap: parseFloat(getComputedStyle(hd).marginBottom),
    capped: bd.scrollHeight - bd.clientHeight > 1, bodyHeight: bd.clientHeight,
  };
}

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
        // Whether the Esc keycap beside the × is drawn, and the close control's box.
        const keyShown = () => {
          const b = document.getElementById("shep-help-bubble"), x = b.querySelector(".shep-bubble-x");
          const k = x && x.querySelector(".shep-kbd"), r = x ? x.getBoundingClientRect() : null;
          return { key: !!k && !k.hidden && getComputedStyle(k).display !== "none" && k.getBoundingClientRect().height > 0,
                   close: r ? { width: r.width, height: r.height } : null };
        };
        const hover = await page.evaluate(() => {
          const b = document.getElementById("shep-help-bubble");
          return b && !b.hidden ? b.getAttribute("role") : null;
        });
        const hoverKey = await page.evaluate(keyShown);
        await button.click();
        await page.waitForFunction(() => document.getElementById("shep-help-bubble").getAttribute("role") === "dialog");
        const pinnedKey = await page.evaluate(keyShown);
        const head = await page.evaluate(closeOf);
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
        // The table row's close control focused after a key press (a keyboard's focus): its ring.
        let ring = null;
        if (which === "row") {
          await page.keyboard.press("a");
          ring = await page.evaluate(() => {
            const x = document.querySelector("#shep-help-bubble .shep-bubble-x");
            x.focus();
            const s = getComputedStyle(x);
            return { focused: document.activeElement === x, visible: x.matches(":focus-visible"),
                     style: s.outlineStyle, width: parseFloat(s.outlineWidth) };
          });
        }
        // The field's bubble closed by a click on its Esc keycap (part of the one close
        // control), the others by Esc.
        let closedBy = "Escape";
        if (which === "field" && await page.$(KEY)) {
          await page.click(KEY);
          closedBy = "keycap";
        } else {
          await page.keyboard.press("Escape");
        }
        const closed = await page.evaluate(() => {
          const b = document.getElementById("shep-help-bubble");
          return b.hidden && b.getAttribute("role") === "tooltip";
        });
        await page.mouse.move(1, 1);
        await page.waitForFunction(() => document.getElementById("shep-help-bubble").hidden);
        size[which] = { hit: { width: hit.width, height: hit.height }, early, hover, hoverKey,
                        pinnedKey, head, pinned, ring, closedBy, closed };
      }
      out[`${width}x${height}`] = size;
      // A ⓘ whose title wraps beside the close control and whose text is taller than the window.
      const long = page.locator(LONG);
      await long.scrollIntoViewIfNeeded();
      await long.click();
      await page.waitForFunction(() => document.getElementById("shep-help-bubble").getAttribute("role") === "dialog");
      (out.long = out.long || {})[`${width}x${height}`] = await page.evaluate(closeOf);
      await page.keyboard.press("Escape");
      await page.mouse.move(1, 1);
      await page.waitForFunction(() => document.getElementById("shep-help-bubble").hidden);
      await page.close();
    }
  } finally {
    await browser.close();
  }
  process.stdout.write(JSON.stringify(out));
})().catch((err) => { process.stderr.write(String(err && err.stack || err)); process.exit(1); });
