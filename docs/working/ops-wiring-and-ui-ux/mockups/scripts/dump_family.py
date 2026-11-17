"""Dump the server's listing and family view of a workspace, as the Models page reads them.
usage: python dump_family.py <snapshot> <workspace-name> <out.json>"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

snap = Path(sys.argv[1])
sys.path.insert(0, str(snap))
from src.api.services.server_lists import list_checkpoints, list_family  # noqa: E402

name, out = sys.argv[2], Path(sys.argv[3])
listing = list_checkpoints(name)
fam = list_family(name, listing=listing)
rows = []
for row in fam.items:
    it = row.item
    rows.append({
        "key": it.key, "run": it.run, "file": it.filename, "sha12": (it.sha256 or "")[:12],
        "group": row.group, "family_id": row.family_id, "depth": row.depth,
        "parent": row.parent, "parent_state": row.parent_state, "context": row.context,
        "stage": it.learning_stage, "task": it.task, "operation": it.operation,
        "parent_sha12": (it.parent_sha256 or "")[:12], "parent_run": it.parent_run,
        "parent_changes": it.parent_changes, "kind": it.checkpoint_kind, "epoch": it.epoch,
        "completed_epochs": it.completed_epochs, "end": it.end,
        "frozen": (it.weight_transfer or {}).get("frozen_parts") if isinstance(it.weight_transfer, dict) else None,
        "weight_transfer_keys": sorted((it.weight_transfer or {}).keys()) if isinstance(it.weight_transfer, dict) else None,
        "validation": it.validation, "in_best_k": it.in_best_k, "selection": it.selection,
        "record": it.record,
    })
listing_order = [(i.key, (i.sha256 or "")[:12]) for i in listing.items]
out.write_text(json.dumps({"listing_order": listing_order, "family_rows": rows,
                           "weight_transfer_example": next((asdict(i).get("weight_transfer") for i in listing.items if i.weight_transfer), None)},
                          indent=1, default=str))
for r in rows:
    print(f'{r["family_id"]!s:45} d{r["depth"]} {r["stage"]:8} {str(r["task"]):8} {str(r["operation"]):10} {r["run"][:40]:40} {r["file"]:22} {r["sha12"]} parent={r["parent_sha12"]} {r["parent_state"]}')
