// Does the drawer list's sticky header cover a row focused from the keyboard?
// (The skill: nothing sticky covers a focused control.) Rebuilds the narrow
// drawer mockup state, scrolls a file row under the header, focuses it.
const { chromium } = require('playwright');
const K = require('./common.cjs');
const T = require('./tree.cjs');
const fs = require('fs');
const src = fs.readFileSync(__dirname + '/drawer.cjs', 'utf8');
const CSS = eval(src.match(/const CSS = (`[\s\S]*?`);/)[1].replace(/\$\{T\.TREE_CSS\}/, '${T.TREE_CSS}').replace(/\$\{L\}/g, '#shep-models-clinical'));
(async () => {
  const b = await chromium.launch();
  const p = await K.open(b, 'http://127.0.0.1:8871/ui/', 760, 1000);
  await K.nav(p);
  await p.click('#shep-dx-change-btn');
  await p.waitForTimeout(2500);
  await K.style(p, 'mk-drawer', CSS);
  const tree = T.render({ selected: 'x', expanded: new Set(['20261116-122806_demo-gene_367c36', '20261116-122738_demo-disease-extend_19ec67']), preset: 'clinical' });
  await p.evaluate((h) => { document.querySelector('#shep-models-clinical-rows .html-container').innerHTML = h; }, tree.html);
  await p.evaluate(() => window.scrollTo(0, document.getElementById('shep-dx-change').getBoundingClientRect().top + scrollY - 8));
  const out = [];
  for (const which of [0, 1, 2, 3]) {
    out.push(await p.evaluate((i) => {
      const reg = document.getElementById('shep-models-clinical-rows');
      const rows = [...reg.querySelectorAll('tr.mk-file')];
      const row = rows[i];
      reg.scrollTop = row.offsetTop + 10;   // the row now sits under (or above) the sticky header
      row.focus();
      const r = row.getBoundingClientRect();
      const top = document.elementsFromPoint(r.left + 60, r.top + r.height / 2)[0];
      const head = reg.querySelector('thead').getBoundingClientRect();
      return { row: row.innerText.replace(/\s+/g, ' ').trim(), focused: document.activeElement === row, rowTop: Math.round(r.top), headBottom: Math.round(head.bottom), onTop: row.contains(top) ? 'the row' : (top.tagName + ' ' + top.className) };
    }, which));
  }
  console.log(JSON.stringify(out, null, 1));
  K.save('D04_drawer_focus_check', out);
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
