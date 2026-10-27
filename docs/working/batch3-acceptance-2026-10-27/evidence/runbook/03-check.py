"""只讀本次正式 CLI 的產物，複製證據；不執行模型、不修改 registry。"""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

E = Path('/home/user/batch3-acceptance-be0ee1a-01')
REPO = Path('/home/user/Desktop/SHEPHERD-Advanced')
CKPT = Path('/home/user/shepherd-data/workspaces/m5_webui_demo_01/runs/gat/20261024-173909_ui-live-stop-02_c1d9cf/checkpoints/last.pt')
VERSION = Path('/home/user/shepherd-data/registry/versions/m5-closeout-demo-01')
SHA = 'be0ee1a662db8d16a85533c0394244ffcb64ae29'
CKSHA = '568b109848430c92de8d13bd5e24d7836bc16fe32f17a088952be4b322a856bf'
COSHA = '1c2c6050951155df6ccad39e07579c0df83536a35779f7ce916bda9561bb96c3'

def check(ok, description):
    if not ok:
        raise SystemExit('FAIL：' + description)
    print('PASS：' + description, flush=True)

def save_new(name, data):
    with (E / name).open('xb') as f:
        f.write(data)

subprocess.run(['sha256sum', '--quiet', '-c', str(E / 'inputs.sha256')], check=True)
check(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == SHA, 'checkout 是受測版本')
subprocess.run(['git', '-C', str(REPO), 'diff', '--exit-code', '--quiet'], check=True)
subprocess.run(['git', '-C', str(REPO), 'diff', '--cached', '--exit-code', '--quiet'], check=True)
reports, outcomes = {}, {}
for mode, schema, batch in [('C', 1, 32), ('D', 2, 1)]:
    check((E / f'{mode}-exit.txt').read_text().split() == ['0', '0'], f'{mode}：量測和 tee 退出碼都是 0')
    o = json.loads((E / f'{mode}-outcome.json').read_bytes())
    check(o['complete'] is True and o['mode'] == mode and o['registration']['registered'] is True, f'{mode}：完成並登記')
    check(o['checkpoint_sha256'] == CKSHA and o['cohort']['manifest_sha256'] == COSHA, f'{mode}：輸入身分吻合')
    p = Path(o['report'])
    check(p == VERSION / 'reports' / o['run_id'] / 'report.json', f'{mode}：報告在原 registry')
    raw = p.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == o['report_sha256'], f'{mode}：報告摘要吻合')
    r = json.loads(raw)
    m = r['manifest']
    check(m['mode'] == mode and m['metric_schema_version'] == schema and m['batch_size'] == batch, f'{mode}：manifest 模式、schema、batch size')
    check(m['cuda_executed'] is True and m['software_revision'] == SHA and m['environment']['code']['dirty'] is False, f'{mode}：CUDA、受測版本及未修改程式')
    check('GB10' in m['environment']['facts']['gpu']['name'], f'{mode}：GPU 是 GB10')
    check(m['artifact_digests']['checkpoint'] == CKSHA and m['cohort_version']['manifest_sha256'] == COSHA and m['n_samples'] == 10, f'{mode}：同一 checkpoint、cohort 與 10 例')
    check(r['ranks']['file'] == 'ranks.json', f'{mode}：排名檔名')
    rank_bytes = (p.parent / 'ranks.json').read_bytes()
    check(hashlib.sha256(rank_bytes).hexdigest() == r['ranks']['sha256'], f'{mode}：排名檔摘要')
    (E / mode).mkdir()
    save_new(f'{mode}/report.json', raw)
    save_new(f'{mode}/ranks.json', rank_bytes)
    reports[mode], outcomes[mode] = r, o

d = reports['D']
counts = d['outcomes']
check(sum(counts[k] for k in ('ranked', 'not_reached', 'refused')) == counts['n_import_eligible'] == 10, 'D 三種結果合計為匯入合格的 10 例')
check(d['import_counts']['source_cases'] == 12 and d['import_counts']['n_excluded'] == 2, '來源 12 例、匯入排除 2 例')
check(d['served']['scoring_mode'] == 'gnn_only' and d['served']['settings']['eta'] == 1.0, '本次 D 為 gnn_only、η=1.0（沒有 SP 表）')
kept = [x for x in d['comparison']['kept_mode_c_reports'] if x['run_id'] == outcomes['C']['run_id']]
check(len(kept) == 1 and 'unreadable' not in kept[0] and kept[0]['n_compared'] == 10, 'D 指向這次 C 報告並完成 10 例對照')
print('C 報告與 D 內部 C 側的逐例核對：', kept[0], flush=True)
log = (E / 'D.log').read_text()
check('gnn_only' in log and 'η = 1.0' in log, 'D 主控台標籤包含模式和 η')
progress = re.findall(r'Mode D: (\d+) of (\d+) cases through the served path, [^\n]*?: (\{[^\n]+\})', log)
check(bool(progress) and progress[-1][:2] == ('10', '10'), 'D 最後進度為 10 / 10')
tally = ast.literal_eval(progress[-1][2])
check(tally == {k: counts[k] for k in ('ranked', 'not_reached', 'refused')}, '最後進度累計與報告吻合')

before = json.loads((E / 'ledger-before.json').read_bytes())['records']
ledger_raw = (CKPT.parent / 'evaluations.json').read_bytes()
records = json.loads(ledger_raw)['records']
check(records[:len(before)] == before and len(records) == len(before) + 2, '保留所有舊帳本紀錄，只新增兩筆')
new = records[len(before):]
for mode in ('C', 'D'):
    found = [r for r in new if r['scoring']['mode'] == mode]
    check(len(found) == 1 and found[0]['source_artifact_digest'] == outcomes[mode]['report_sha256'] and found[0]['checkpoint_digest'] == CKSHA and found[0]['cohort_digest'] == COSHA, f'新增 {mode} 紀錄正確指向本次報告及輸入')
check(len({r['scoring_semantics_digest'] for r in new}) == 2, 'C、D 的 scoring_semantics_digest 不同')
usage = (VERSION / 'reports/usage.jsonl').read_bytes()
check(usage.startswith((E / 'usage-before.jsonl').read_bytes()), '既有使用紀錄完整保留')
save_new('ledger-after.json', ledger_raw)
save_new('ledger-new-C-D.json', (json.dumps(new, ensure_ascii=False, indent=2) + '\n').encode())
save_new('usage-after.jsonl', usage)
print('\nPASS：GB10 C/D 功能短測通過；不代表模型品質、院方驗收或 SP 混合路徑的實機驗證。')
print('接著還原 checkout；不要把舊帳本備份覆蓋回去，新登記應保留。')
