// D04: the shared model list as a tree (family > run > saved file), drawn from
// the server's own family listing of the demo workspace (work/family.json,
// server_lists.list_family), so every name, fingerprint, parent and epoch is real.
const fs = require('fs');
const path = require('path');

const FAMILY = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'work', 'family.json'), 'utf8'));

const e = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const TASK = { null: 'Base', disease: 'Disease', gene: 'Gene', patients_like_me: 'Patients-like-me' };
const FROZEN_RUNS = /-frozen_/;   // the listing's own "frozen graph backbone" kind (model_kind_text)

function runWords(run) {
  // "20261116-122738_demo-disease-extend_19ec67" -> name, "2026-11-16 12:27"
  const m = run.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})\d{2}_(.+)_[0-9a-f]{6}$/);
  return { name: m[6], when: `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` };
}
function operation(r) {
  if (r.operation === 'new') return 'New run';
  if (r.operation === 'branch') return 'Branch';
  if (r.operation === 'extend') return 'Extend';
  if (r.operation === 'finetune') return FROZEN_RUNS.test(r.run) ? 'Fine-tune, frozen' : 'Fine-tune, full';
  return r.operation;
}
function fileWords(f) {
  if (f.file === 'last.pt') return 'Latest save';
  return f.validation && f.validation.metric === 'val_lp_accuracy' ? 'Best LP accuracy' : 'Best pooled MRR';
}

// Families in the server's order; runs in the server's order (a run's files
// together), each run with its depth and parent run.
function model() {
  const fams = [];
  const byFam = new Map();
  for (const row of FAMILY.family_rows) {
    if (!byFam.has(row.family_id)) { const f = { id: row.family_id, runs: [], byRun: new Map() }; byFam.set(row.family_id, f); fams.push(f); }
    const f = byFam.get(row.family_id);
    if (!f.byRun.has(row.run)) {
      const parentRun = row.parent ? row.parent.split('/')[1] : null;
      const r = { run: row.run, depth: row.depth, parentRun, task: TASK[row.task], op: operation(row), files: [], ...runWords(row.run) };
      f.byRun.set(row.run, r); f.runs.push(r);
    }
    const r = f.byRun.get(row.run);
    r.files.push({ key: row.key, words: fileWords(row), sha: row.sha12, epochs: row.completed_epochs, file: row.file });
  }
  for (const f of fams) {
    for (const r of f.runs) { r.epochs = Math.max(...r.files.map(x => x.epochs)); r.files.sort((a, b) => (a.file === 'last.pt' ? -1 : 1)); }
    f.root = f.runs.find(r => r.depth === 0);
    f.nfiles = f.runs.reduce((n, r) => n + r.files.length, 0);
  }
  return fams;
}
const FAMS = model();
const ALL_RUNS = FAMS.flatMap(f => f.runs);

function ancestors(fam, run) {
  const out = [];
  for (let r = run; r && r.parentRun; r = fam.byRun.get(r.parentRun)) { const p = fam.byRun.get(r.parentRun); if (p) out.push(p.run); }
  return out;
}

// opts: { selected: file key, expanded: Set(run), folded: Set(family id),
//         query: string, task: 'all'|'Disease'|..., preset: 'research'|'clinical',
//         flat: bool (newest first), served: file key }
function render(opts) {
  const o = { expanded: new Set(), folded: new Set(), query: '', task: 'all', preset: 'research', ...opts };
  const q = o.query.trim().toLowerCase();
  const isHex = /^[0-9a-f]{4,64}$/.test(q);
  const clinical = o.preset === 'clinical';
  const rows = [];
  let shownRuns = 0, shownFiles = 0, shownFams = 0, matches = 0;
  const hl = (text) => {
    if (!q || isHex) return e(text);
    const i = text.toLowerCase().indexOf(q); if (i < 0) return e(text);
    return e(text.slice(0, i)) + '<mark class="mk-hit">' + e(text.slice(i, i + q.length)) + '</mark>' + e(text.slice(i + q.length));
  };
  const hlSha = (sha) => (isHex && sha.startsWith(q)) ? `<mark class="mk-hit">${e(sha.slice(0, q.length))}</mark>${e(sha.slice(q.length))}` : e(sha);
  const toggle = (open, label) => `<button type="button" class="mk-tog" aria-expanded="${open}" aria-label="${e(label)}"><span aria-hidden="true">${open ? '▾' : '▸'}</span></button>`;

  for (const fam of (o.flat ? [{ flat: true, runs: [...ALL_RUNS].sort((a, b) => b.run.localeCompare(a.run)), byRun: new Map(ALL_RUNS.map(r => [r.run, r])) }] : FAMS)) {
    // Which runs and files this view keeps: task filter, then the search
    // (a match keeps itself and its ancestors; a run match keeps its run row).
    const keepRun = new Map(); // run -> 'match' | 'context'
    const keepFile = new Set();
    for (const r of fam.runs) {
      if (clinical && r.task === 'Base') continue;
      const taskOk = o.task === 'all' || r.task === o.task;
      let hit = taskOk;
      if (q && taskOk) {
        const nameHit = !isHex && r.name.toLowerCase().includes(q);
        const fileHits = isHex ? r.files.filter(f => f.sha.startsWith(q)) : [];
        hit = nameHit || fileHits.length > 0;
        fileHits.forEach(f => keepFile.add(f.key));
        if (hit) matches += nameHit ? 1 : fileHits.length;
      }
      if (hit) {
        keepRun.set(r.run, 'match');
        if (!o.flat) for (const a of ancestors(fam, r)) { if (!keepRun.has(a) && !(clinical && fam.byRun.get(a).task === 'Base')) keepRun.set(a, 'context'); }
      }
    }
    if (!keepRun.size) continue;
    shownFams++;
    const famRuns = fam.runs.filter(r => keepRun.has(r.run));
    const famFolded = o.folded.has(fam.id);
    if (!o.flat) {
      const nRuns = fam.runs.filter(r => !(clinical && r.task === 'Base')).length;
      const nFiles = fam.runs.filter(r => !(clinical && r.task === 'Base')).reduce((n, r) => n + r.files.length, 0);
      const n = (k, w) => `${k} ${w}${k === 1 ? '' : 's'}`;
      const counts = (q || o.task !== 'all') ? `${famRuns.length} of ${n(nRuns, 'run')} shown` : `${n(nRuns, 'run')}, ${n(nFiles, 'saved file')}`;
      rows.push(`<tr class="mk-fam"><th colspan="6" scope="rowgroup"><span class="mk-cell">${toggle(!famFolded, (famFolded ? 'Unfold' : 'Fold') + ' family from ' + fam.root.name)}<span class="mk-fam-t">Family from ${e(fam.root.name)}</span><span class="mk-fam-n">${counts}</span></span></th></tr>`);
      if (famFolded) continue;
    }
    for (const r of famRuns) {
      const state = keepRun.get(r.run);
      const filesShown = r.files.filter(f => keepFile.has(f.key) || (!q && o.expanded.has(r.run)) || (q && !isHex && state === 'match' && o.expanded.has(r.run)));
      const open = filesShown.length > 0;
      const depth = o.flat ? 0 : (clinical ? Math.max(0, r.depth - 1) : r.depth);
      const hasSel = r.files.some(f => f.key === o.selected);
      shownRuns++;
      rows.push(`<tr class="mk-run${state === 'context' ? ' mk-ctx' : ''}${hasSel && !open ? ' mk-has-sel' : ''}" style="--d:${depth}">`
        + `<td><span class="mk-cell">${toggle(open, (open ? 'Fold' : 'Unfold') + ' run ' + r.name)}<span class="mk-run-t"><b>${hl(r.name)}</b><span class="mk-run-sep"> · </span><span class="mk-run-w">GATv2 ${e(r.when)}</span></span></span></td>`
        + `<td>${e(r.task)}</td><td>${e(r.op)}</td><td class="shep-num">${r.epochs}</td><td class="shep-num">${r.files.length}</td><td></td></tr>`);
      for (const f of filesShown) {
        shownFiles++;
        const sel = f.key === o.selected, served = f.key === o.served;
        rows.push(`<tr class="mk-file${sel ? ' shep-on' : ''}" data-key="${e(f.key)}" tabindex="0" style="--d:${depth}"${sel ? ' aria-current="true"' : ''}>`
          + `<td><span class="mk-cell"><span class="mk-file-t">${e(f.words)}</span>${served ? '<span class="shep-st shep-st-ok mk-inuse"><span class="shep-st-shape" aria-hidden="true">●</span>In use</span>' : ''}</span></td>`
          + `<td></td><td></td><td class="shep-num">${f.epochs}</td><td></td><td><code class="shep-fpv">${hlSha(f.sha)}</code></td></tr>`);
      }
    }
  }
  const head = '<thead><tr><th scope="col" class="mk-c1">Model</th><th scope="col">Task</th><th scope="col">Operation</th><th scope="col" class="shep-num">Epochs</th><th scope="col" class="shep-num">Files</th><th scope="col">Fingerprint</th></tr></thead>';
  const table = `<div class="shep-tbl-wrap shep-mtbl-wrap mk-tree-wrap"><table class="shep-tbl shep-mtbl mk-tree" aria-label="Models">`
    + `<colgroup><col class="mk-w1"><col class="mk-w2"><col class="mk-w3"><col class="mk-w4"><col class="mk-w5"><col class="mk-w6"></colgroup>${head}<tbody>${rows.join('')}</tbody></table></div>`;
  return { html: table, shownRuns, shownFiles, shownFams, matches };
}

const TREE_CSS = `
.mk-tree { table-layout: fixed; }
.mk-tree col.mk-w1 { width: auto; } .mk-tree col.mk-w2 { width: 92px; } .mk-tree col.mk-w3 { width: 142px; }
.mk-tree col.mk-w4 { width: 74px; } .mk-tree col.mk-w5 { width: 58px; } .mk-tree col.mk-w6 { width: 128px; }
.mk-tree th, .mk-tree td { padding: 0 8px !important; height: 36px; vertical-align: middle !important; line-height: 1.3; }
.mk-tree thead th { height: 34px; }
.mk-tree .mk-cell { display: flex; align-items: center; gap: 6px; min-width: 0; padding-left: calc(var(--d, 0) * 22px); }
.mk-tree tr.mk-file .mk-cell { padding-left: calc(var(--d, 0) * 22px + 30px); }
.mk-tree button.mk-tog {
  flex: none; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center;
  border: 1px solid transparent; border-radius: 6px; background: none; color: var(--shep-text-2); font-size: 16px; cursor: pointer; padding: 0;
}
.mk-tree button.mk-tog:hover { background: var(--shep-grey-50); border-color: var(--shep-grey-500); }
.mk-tree button.mk-tog:focus-visible { outline: 2px solid var(--shep-focus-ring); outline-offset: 0; }
.mk-tree tr.mk-fam th { background: var(--shep-panel-2); font-size: 15px; font-weight: 600; color: var(--shep-text); border-bottom: 1px solid var(--shep-border); position: static; }
.mk-tree .mk-fam-n { margin-left: auto; font-weight: 400; font-size: 14px; color: var(--shep-text-2); }
.mk-tree tr.mk-run td { border-bottom: 1px solid #edf0f3; }
.mk-tree .mk-run-t { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mk-tree .mk-run-t b { font-weight: 600; }
.mk-tree .mk-run-w, .mk-tree .mk-run-sep { color: var(--shep-text-2); font-size: 14px; }
.mk-tree tr.mk-ctx td { color: var(--shep-text-2); }
.mk-tree tr.mk-ctx .mk-run-t b { font-weight: 500; }
.mk-tree tr.mk-file td { background: #fff; cursor: pointer; }
.mk-tree tr.mk-file:hover td { background: var(--shep-panel-2); }
.mk-tree tr.mk-file.shep-on td { background: var(--shep-accent-50); }
.mk-tree tr.mk-file.shep-on td:first-child { box-shadow: inset 4px 0 0 var(--shep-accent); }
.mk-tree tr.mk-file:focus-visible { outline: 2px solid var(--shep-focus-ring); outline-offset: -2px; }
.mk-tree .mk-file-t { font-size: 15px; }
.mk-tree mark.mk-hit { background: var(--shep-amber-100); color: inherit; border-radius: 3px; padding: 0 1px; box-shadow: inset 0 -2px 0 var(--shep-amber-800); }
.mk-tree .mk-inuse { margin-left: 8px; font-size: 14px; }
.mk-tree td.shep-num { padding-right: 14px !important; }
/* By the list's own width (D02: the container decides, not the window): under
   800 px a run's name keeps its line and its architecture and start go under it. */
.mk-tree-wrap { container-type: inline-size; }
@container (max-width: 880px) {
  .mk-tree .mk-run-t { white-space: normal; overflow: visible; }
  .mk-tree .mk-run-w { display: block; }
  .mk-tree .mk-run-sep { display: none; }
  .mk-tree tr.mk-run td { height: auto; padding-top: 6px !important; padding-bottom: 6px !important; }
  .mk-tree col.mk-w2 { width: 74px; } .mk-tree col.mk-w3 { width: 116px; } .mk-tree col.mk-w4 { width: 64px; }
  .mk-tree col.mk-w5 { width: 48px; } .mk-tree col.mk-w6 { width: 120px; }
  .mk-tree .mk-cell { padding-left: calc(var(--d, 0) * 16px); }
  .mk-tree tr.mk-file .mk-cell { padding-left: calc(var(--d, 0) * 16px + 30px); }
  .mk-tree .mk-run-t b { white-space: nowrap; }
}
`;

module.exports = { render, TREE_CSS, FAMS, ALL_RUNS, e };
