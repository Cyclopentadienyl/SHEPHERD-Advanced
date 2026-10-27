"""
`matching_phenotypes`: what the graph annotates directly to each candidate.
==========================================================================
BACKLOG item 17, F6. The API declared the field and the real path never filled
it, while each candidate's explanation listed its matches under "Matching
Phenotypes"; the demo answer filled it with the first request phenotypes. These
tests pin the definition the field now states:

  - **exact direct annotation only**: a `phenotype_of_disease` edge from the
    very term to the disease. A phenotype linked only through a gene does not
    match, nor does an ancestor or descendant of an annotated term;
  - **the list as used**: unknown ids dropped, repeats removed, the request's
    spelling and order kept;
  - filled whether or not explanations are asked for, by the helper the
    explanation's section uses: the section lists the first ten of the field,
    in its order, and says how many more there are;
  - carried by the API and the WebUI's CSV, and empty in the demo answer.

Module: tests/unit/test_matching_phenotypes.py
"""
from __future__ import annotations

import csv
import io
import itertools
import re

import pytest

from src.core.types import (
    DataSource,
    Edge,
    EdgeType,
    Node,
    NodeID,
    NodeType,
    PatientPhenotypes,
)
from src.reasoning import find_matching_phenotypes

SEIZURE = "HP:0001250"
ATAXIA = "HP:0001251"
DELAY = "HP:0001263"
#: A descendant of SEIZURE (epileptic spasm).
SPASM = "HP:0011097"
#: An ancestor of SEIZURE (abnormality of the nervous system).
NERVOUS = "HP:0000707"
#: Well formed, and not a node of this graph.
UNKNOWN = "HP:9999999"

#: Annotated with SEIZURE and ATAXIA, and linked to DELAY through a gene.
D1 = NodeID(source=DataSource.MONDO, local_id="MONDO:0000001")
#: Annotated with ATAXIA, and linked to DELAY through the same gene.
D2 = NodeID(source=DataSource.MONDO, local_id="MONDO:0000002")
#: Linked to DELAY through the gene, and annotated with nothing.
D3 = NodeID(source=DataSource.MONDO, local_id="MONDO:0000003")


def _build_kg():
    from src.kg import KnowledgeGraph

    kg = KnowledgeGraph()
    names = {
        SEIZURE: "Seizure", ATAXIA: "Ataxia", DELAY: "Global developmental delay",
        SPASM: "Epileptic spasm", NERVOUS: "Abnormality of the nervous system",
    }
    for hpo_id, name in names.items():
        kg.add_node(Node(
            id=NodeID(source=DataSource.HPO, local_id=hpo_id),
            node_type=NodeType.PHENOTYPE, name=name, data_sources={DataSource.HPO},
        ))
    gene = NodeID(source=DataSource.DISGENET, local_id="SCN1A")
    kg.add_node(Node(id=gene, node_type=NodeType.GENE, name="SCN1A",
                     data_sources={DataSource.DISGENET}))
    for disease in (D1, D2, D3):
        kg.add_node(Node(id=disease, node_type=NodeType.DISEASE, name=disease.local_id,
                         data_sources={DataSource.MONDO}))

    def edge(source, target, edge_type):
        if isinstance(source, str):
            source = NodeID(source=DataSource.HPO, local_id=source)
        if isinstance(target, str):
            target = NodeID(source=DataSource.HPO, local_id=target)
        kg.add_edge(Edge(source_id=source, target_id=target, edge_type=edge_type, weight=0.9))

    edge(SEIZURE, D1, EdgeType.PHENOTYPE_OF_DISEASE)
    edge(ATAXIA, D1, EdgeType.PHENOTYPE_OF_DISEASE)
    edge(ATAXIA, D2, EdgeType.PHENOTYPE_OF_DISEASE)
    edge(gene, DELAY, EdgeType.GENE_HAS_PHENOTYPE)
    for disease in (D1, D2, D3):
        edge(gene, disease, EdgeType.GENE_ASSOCIATED_WITH_DISEASE)
    edge(SPASM, SEIZURE, EdgeType.IS_A)
    edge(SEIZURE, NERVOUS, EdgeType.IS_A)
    return kg


@pytest.fixture
def kg():
    return _build_kg()


@pytest.fixture
def pipeline(kg):
    from src.inference.pipeline import DiagnosisPipeline

    return DiagnosisPipeline(kg=kg)


def _run(pipeline, phenotypes, include_explanations):
    return pipeline.run(
        patient_input=PatientPhenotypes(patient_id="f6", phenotypes=list(phenotypes)),
        top_k=10,
        include_explanations=include_explanations,
    )


def _matches(result):
    return {str(c.disease_id): c.matching_phenotypes for c in result.candidates}


_ENTRY = re.compile(r"- .+ \((HP:\d{7})\)")


def _matching_section(explanation):
    """The ids under the explanation's "### Matching Phenotypes" heading, which
    lists them one per line as "- <name> (<id>)" until a blank line, and the
    last line when it is not an entry (the count of the rest), else None."""
    lines = explanation.splitlines()
    if "### Matching Phenotypes" not in lines:
        return [], None
    section = list(itertools.takewhile(
        bool, lines[lines.index("### Matching Phenotypes") + 1:]
    ))
    more = section.pop() if section and not _ENTRY.fullmatch(section[-1]) else None
    return [_ENTRY.fullmatch(line).group(1) for line in section], more


#: What a request for SEIZURE, ATAXIA and DELAY matches, by disease.
_EXPECTED = {str(D1): [SEIZURE, ATAXIA], str(D2): [ATAXIA], str(D3): []}


# ------------------------------------------------------------------ the definition
def test_only_a_direct_annotation_of_the_exact_term_matches(kg):
    assert find_matching_phenotypes(
        D1, [SEIZURE, ATAXIA, DELAY, SPASM, NERVOUS, UNKNOWN], kg
    ) == [SEIZURE, ATAXIA]
    assert find_matching_phenotypes(D3, [SEIZURE, ATAXIA, DELAY], kg) == []


@pytest.mark.parametrize("include_explanations", [False, True])
def test_every_candidate_carries_its_direct_matches(pipeline, include_explanations):
    """DELAY reaches all three diseases in two hops through the gene and matches
    none of them; D3 is a candidate with no match at all."""
    result = _run(pipeline, [SEIZURE, ATAXIA, DELAY], include_explanations)

    assert _matches(result) == _EXPECTED
    assert all((c.explanation is not None) is include_explanations
               for c in result.candidates)


def test_an_ancestor_or_descendant_of_an_annotated_term_does_not_match(pipeline):
    """D1 is annotated with SEIZURE. A case listing a descendant (epileptic
    spasm) and an ancestor of it reaches D1 through the ontology, and matches
    nothing there."""
    matches = _matches(_run(pipeline, [SPASM, NERVOUS], include_explanations=True))

    assert str(D1) in matches, "D1 must be a candidate for the test to mean anything"
    assert matches[str(D1)] == []


def test_the_matches_come_from_the_list_as_used(pipeline):
    """Unknown ids dropped, repeats removed, in the request's order and spelling:
    HPO ids as sent, not the graph's `hpo:` node ids."""
    result = _run(
        pipeline, [UNKNOWN, ATAXIA, SEIZURE, ATAXIA, SEIZURE], include_explanations=False
    )

    assert result.phenotype_input.used == (ATAXIA, SEIZURE)
    assert _matches(result)[str(D1)] == [ATAXIA, SEIZURE]
    assert _matches(result)[str(D2)] == [ATAXIA]


def test_the_explanation_lists_all_of_a_field_of_ten_or_fewer(pipeline):
    result = _run(pipeline, [DELAY, ATAXIA, SEIZURE, ATAXIA], include_explanations=True)

    assert result.candidates
    for candidate in result.candidates:
        assert _matching_section(candidate.explanation) == (
            candidate.matching_phenotypes, None
        ), candidate.disease_id
    assert _matches(result)[str(D1)] == [ATAXIA, SEIZURE]


#: A disease annotated directly with every phenotype it is built with.
D_MANY = NodeID(source=DataSource.MONDO, local_id="MONDO:0000004")


def _kg_annotating(phenotypes):
    from src.kg import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_node(Node(id=D_MANY, node_type=NodeType.DISEASE, name="many",
                     data_sources={DataSource.MONDO}))
    for hpo_id in phenotypes:
        node_id = NodeID(source=DataSource.HPO, local_id=hpo_id)
        kg.add_node(Node(id=node_id, node_type=NodeType.PHENOTYPE, name=f"Term {hpo_id}",
                         data_sources={DataSource.HPO}))
        kg.add_edge(Edge(source_id=node_id, target_id=D_MANY,
                         edge_type=EdgeType.PHENOTYPE_OF_DISEASE, weight=0.9))
    return kg


#: Twelve terms, requested in an order that is not the ids' own.
_TWELVE = [f"HP:{n:07d}" for n in range(3012, 3000, -1)]


@pytest.mark.parametrize("count, more", [
    (10, None), (11, "- …and 1 more"), (12, "- …and 2 more"),
])
def test_the_explanation_lists_the_first_ten_of_the_field_and_how_many_more(count, more):
    """A request carries up to 100 phenotypes and the field every match; the
    section lists the first ten in the field's order, then says how many it
    left out, rather than cutting the list without a word."""
    from src.inference.pipeline import DiagnosisPipeline

    requested = _TWELVE[:count]
    result = _run(DiagnosisPipeline(kg=_kg_annotating(requested)), requested,
                  include_explanations=True)

    (candidate,) = [c for c in result.candidates if c.disease_id == D_MANY]
    assert candidate.matching_phenotypes == requested
    assert _matching_section(candidate.explanation) == (requested[:10], more)


# ------------------------------------------------------------------ the API
@pytest.fixture
def client(monkeypatch, pipeline):
    """The real app, serving a real path-reasoning pipeline on this graph."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main as api_main

    monkeypatch.setattr(api_main.app_state, "pipeline", pipeline, raising=False)
    return TestClient(api_main.app)


def _post(client, **payload):
    response = client.post("/api/v1/diagnose", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("include_explanations", [False, True])
def test_the_api_carries_the_matches(client, include_explanations):
    body = _post(
        client,
        phenotypes=[SEIZURE, UNKNOWN, ATAXIA, SEIZURE, DELAY],
        include_explanations=include_explanations,
    )

    assert {c["disease_id"]: c["matching_phenotypes"] for c in body["candidates"]} == (
        _EXPECTED
    )


def test_the_field_states_its_definition():
    pytest.importorskip("fastapi")
    from src.api.routes.diagnose import (
        DemoCandidate,
        PathReasoningCandidate,
        ServedMixtureCandidate,
    )

    descriptions = {kind.model_fields["matching_phenotypes"].description
                    for kind in (ServedMixtureCandidate, PathReasoningCandidate, DemoCandidate)}
    assert len(descriptions) == 1, "every kind's candidate states one definition"
    (description,) = descriptions
    assert "directly" in description and "exact term" in description
    assert "not ontology ancestors or descendants, not gene-mediated" in description
    assert "An empty list does not mean no clinical overlap." in description


def test_the_demo_answer_claims_no_matches(monkeypatch):
    """No pipeline configured: the invented candidates were matched against
    nothing, so they carry no matches rather than the request's phenotypes."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main as api_main

    monkeypatch.setattr(api_main.app_state, "pipeline", None, raising=False)
    monkeypatch.setattr(api_main.app_state, "real_pipeline_requested", False, raising=False)
    monkeypatch.delenv("SHEPHERD_KG_PATH", raising=False)

    body = _post(TestClient(api_main.app), phenotypes=[SEIZURE, ATAXIA, DELAY])

    assert body["served_identity"] is None  # the demo answer
    assert body["candidates"]
    assert all(c["matching_phenotypes"] == [] for c in body["candidates"])


# ------------------------------------------------------------------ the CSV
def test_the_csv_column_carries_the_matches(client):
    pytest.importorskip("gradio")
    from src.webui.components import diagnosis_panel as dp

    body = _post(client, phenotypes=[SEIZURE, ATAXIA, DELAY], include_explanations=False)
    rows = list(csv.DictReader(io.StringIO(dp._build_results_csv(body))))

    assert {row["disease_id"]: row["matching_phenotypes"] for row in rows} == {
        D1.local_id: f"{SEIZURE}; {ATAXIA}",
        D2.local_id: ATAXIA,
        D3.local_id: "",
    }


# ------------------------------------------------------------------ the score's words
def test_the_written_explanation_names_a_ranking_score_not_a_confidence_band(client, pipeline):
    """The M5 UI recheck at `f9202b6` (2026-10-25): the table and the details
    header said "Score" while the Explanation segment and the report's Full
    explanation, the pipeline's own text, still read "Confidence: HIGH (score:
    0.806)". The real generator's text, as the API serves it and the WebUI shows
    and exports it, and the pipeline's summary, name the score for what it is."""
    pytest.importorskip("gradio")
    from src.reasoning import SCORE_MEANING
    from src.webui.components import diagnosis_panel as dp

    body = _post(client, phenotypes=[SEIZURE, ATAXIA, DELAY], include_explanations=True)
    assert body["candidates"]
    for candidate in body["candidates"]:
        explanation = candidate["explanation"]
        assert f"Score: {candidate['path_score']:.3f} ({SCORE_MEANING})" in explanation
        assert f"Ranking: {dp.kind_label(body)}" in explanation
        shown = dp.select_candidate(body, "explanation", str(candidate["rank"]))
        assert shown["details_md"]["value"] == explanation
    report = dp._build_results_report_md(body)
    assert "Full explanation" in report and SCORE_MEANING in report
    summary = _run(pipeline, [SEIZURE, ATAXIA, DELAY], True).summary_explanation
    assert "(Score: " in summary and SCORE_MEANING in summary
    for text in (report, summary):
        assert "Confidence" not in text and not re.search(r"\b(HIGH|MEDIUM|LOW)\b", text)
