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
