# External Data Directory

Place manually downloaded annotation files here before running the KG build pipeline.

## Required Files

| File | Source | Description |
|------|--------|-------------|
| `phenotype.hpoa` | [HPO Annotations](https://hpo.jax.org/data/annotations) | Phenotype-disease annotations (~250K rows) |
| `genes_to_phenotype.txt` | [HPO Annotations](https://hpo.jax.org/data/annotations) | Gene-phenotype and gene-disease links (~200K rows) |

Both files are available from the HPO project's annotation download page.

## After Downloading

```bash
# Verify files are in place
ls data/external/phenotype.hpoa data/external/genes_to_phenotype.txt

# Build the knowledge graph
python scripts/build_knowledge_graph.py \
    --workspace data/workspaces/<your_version_name> \
    --external-dir data/external \
    --generate-samples \
    --num-train <training_samples> \
    --num-val <validation_samples>

# Example:
#   python scripts/build_knowledge_graph.py \
#       --workspace data/workspaces/hpo_2026_v1 \
#       --external-dir data/external \
#       --generate-samples --num-train 80000 --num-val 15000
#
# --workspace:  Name it however you like (e.g. hpo_2026_v1, prod_may2026).
#               Each workspace is a self-contained folder with KG + model files.
# --num-train:  Number of simulated patient training samples.
#               Recommended: 50000-100000 (more = less overfitting, slower build).
# --num-val:    Number of validation samples. Recommended: 10%-20% of num-train.
```

## Notes

- Ontology files (HPO, MONDO) are downloaded automatically by `OntologyLoader` — you do not need to download them manually.
- These annotation files are updated periodically by the HPO project. Re-download and rebuild the KG when you want to incorporate new annotations.
- This directory is gitignored — data files are not version controlled.

---

## Currently Supported Data Sources

The KG build pipeline uses per-source parsers under `src/data_sources/`.
Each parser converts an external file format into a normalised
`List[Dict[str, Any]]` that `src/kg/builder.py` then assembles into the
graph. Adding a new data source means writing a new parser; the builder
and downstream pipeline do not need to change.

| Data Source | Parser | Status | Notes |
|-------------|--------|--------|-------|
| **HPO annotations** (`phenotype.hpoa`, `genes_to_phenotype.txt`) | `src/data_sources/hpo_annotations.py` | ✅ Active | Drives the current production pipeline (52K nodes / 494K edges from real HPO data) |
| **HPO / MONDO ontologies** | `src/ontology/loader.py` | ✅ Active | Auto-downloaded; no manual file placement needed |
| **PubMed** literature edges | `src/data_sources/pubmed.py` | ⏸️ Frozen | Implementation present but not wired into builder; literature integration is deferred (see `medical-kg-todo.md`) |
| **Ortholog** mappings (mouse, zebrafish) | `src/data_sources/ortholog.py` | ⏸️ Reserved | Interface defined; cross-species inference is a Phase 3 feature |
| **PrimeKG** (Harvard Dataverse `kg.csv`) | — | 🔜 Planned | Tracked in `docs/Repair/REPAIR_CHECKLIST.md` L206; integration requires extending `NodeType` / `EdgeType` enums and writing a new parser following the `hpo_annotations.py` pattern |
| **DisGeNET**, **ClinVar**, **Orphanet**, **GO**, **Reactome** | — | 📋 Backlog | Listed in `medical-kg-blueprint.md` L139-160 (priorities P0-P1); no parser yet |

### Adding a new data source

The recommended pattern is to follow `hpo_annotations.py`:

1. Create `src/data_sources/<source_name>.py` exposing a parser class.
2. Each `parse_*()` method returns `List[Dict[str, Any]]` matching the
   key schema expected by the relevant `KnowledgeGraphBuilder.add_*`
   method (see docstrings on those methods).
3. Wire it into `scripts/build_knowledge_graph.py` as another step.
4. Do **not** import `src/kg/*` from within the parser — it should stay
   format-translation-only. The data dict is the contract.

This keeps the parser ⇄ builder boundary clean (data coupling only)
and allows each new source to be added without touching the GNN
training / inference path.
