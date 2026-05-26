"""
Entity Linking (PLANNED — NOT YET IMPLEMENTED)
==============================================
Reserved for future entity normalisation between unstructured text mentions
(from PubMed abstracts, clinical notes, FHIR resources, etc.) and canonical
Knowledge Graph node IDs (HPO, MONDO, UMLS, MeSH).

Planned responsibilities:
  - Surface form -> canonical ID disambiguation
    (e.g., "Marfan syndrome" -> "MONDO:0007947")
  - Confidence scoring for ambiguous matches
  - Integration with src/data_sources/pubmed.py (currently frozen)
    and any future Pubtator / scispaCy adapter

Why this is a separate module rather than part of builder.py:
  Entity linking is a stateful, ML-driven concern (often uses sentence
  embeddings or a fine-tuned linker model). It does not belong in the
  rule-based KG construction layer.

Status: planned. Implementation deferred until NLP / literature pipeline
unfreezes (see medical-kg-todo.md and docs/Repair/REPAIR_CHECKLIST.md).
This file is intentionally empty until then; importing it yields an empty
namespace, and accessing any symbol will raise NameError.
"""
