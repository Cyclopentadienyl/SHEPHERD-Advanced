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
    --workspace data/workspaces/hpo_2026/ \
    --external-dir data/external/ \
    --generate-samples
```

## Notes

- Ontology files (HPO, MONDO) are downloaded automatically by `OntologyLoader` — you do not need to download them manually.
- These annotation files are updated periodically by the HPO project. Re-download and rebuild the KG when you want to incorporate new annotations.
- This directory is gitignored — data files are not version controlled.
