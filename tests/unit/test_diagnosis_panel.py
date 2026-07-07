"""
Unit tests for the Diagnosis panel: tolerant HPO parsing and result export.
"""
import os

import pytest

# diagnosis_panel imports gradio at module load.
pytest.importorskip("gradio")
import gradio as gr  # noqa: E402

from src.webui.components import diagnosis_panel as dp  # noqa: E402


# ------------------------------------------------------------------- HPO parsing
def test_parse_strict_one_per_line():
    assert dp._parse_hpo_ids("HP:0001250 — Seizure\nHP:0001263 — Delay") == [
        "HP:0001250",
        "HP:0001263",
    ]


def test_parse_comma_and_space_separated():
    assert dp._parse_hpo_ids("HP:0001250, HP:0001263  HP:0002376") == [
        "HP:0001250",
        "HP:0001263",
        "HP:0002376",
    ]


def test_parse_tolerates_case_missing_colon_and_junk():
    text = "  hp:0001250 ;; HP_0001263 , seizure(HP:0002376)!!  hp 0004322"
    assert dp._parse_hpo_ids(text) == [
        "HP:0001250",
        "HP:0001263",
        "HP:0002376",
        "HP:0004322",
    ]


def test_parse_dedupes_preserving_order():
    assert dp._parse_hpo_ids("HP:0001263\nHP:0001250\nHP:0001263") == [
        "HP:0001263",
        "HP:0001250",
    ]


def test_parse_does_not_grab_prefix_of_longer_number():
    # An 8+ digit run must not yield a bogus 7-digit id.
    assert dp._parse_hpo_ids("HP:00012501234 HP:0009999") == ["HP:0009999"]


def test_parse_no_ids_returns_empty():
    assert dp._parse_hpo_ids("patient has seizures and developmental delay") == []
    assert dp._parse_hpo_ids("") == []


# ------------------------------------------------------------------- export
def _sample_result():
    return {
        "session_id": "sess_abc",
        "patient_id": "pt_01",
        "timestamp": "2026-07-07T16:00:00Z",
        "model_version": "1.0.0",
        "inference_time_ms": 123.4,
        "warnings": ["low phenotype count"],
        "_query_phenotypes": ["HP:0001250", "HP:0001263"],
        "candidates": [
            {
                "rank": 1,
                "disease_id": "mondo:MONDO:0011073",
                "disease_name": "Dravet syndrome",
                "confidence_score": 0.72,
                "gnn_score": 0.81,
                "sp_score": 0.55,
                "confidence_label": "Strong path support",
                "matching_phenotypes": ["HP:0001250"],
                "supporting_genes": ["SCN1A"],
                "explanation": "Because ...",
                "evidence_package": {
                    "mode": "direct_path",
                    "summary": "Direct 2-hop path",
                    "min_path_length": 2,
                    "direct_paths": [["hp:HP:0001250", "mondo:MONDO:0011073"]],
                    "analogies": [],
                },
            },
            {
                "rank": 2,
                "disease_id": "omim:OMIM:123",
                "disease_name": "Other disease",
                "confidence_score": 0.44,
                "confidence_label": "Analogy-based",
                "matching_phenotypes": [],
                "supporting_genes": [],
                "explanation": None,
                "evidence_package": {"mode": "analogy_based", "summary": "Analogy"},
            },
        ],
    }


def test_csv_has_row_per_candidate_with_key_fields():
    csv_text = dp._build_results_csv(_sample_result())
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("rank,disease_id,disease_name,confidence_score")
    assert len(lines) - 1 == 2  # header + 2 candidates
    assert "Dravet syndrome" in csv_text
    assert "mondo:MONDO:0011073" in csv_text
    assert "SCN1A" in csv_text


def test_report_includes_meta_and_all_candidates():
    md = dp._build_results_report_md(_sample_result())
    assert "# SHEPHERD-Advanced Diagnosis Report" in md
    assert "pt_01" in md  # patient id
    assert "HP:0001250" in md  # query phenotypes
    assert "Dravet syndrome" in md and "Other disease" in md  # every candidate
    assert "Full explanation" in md


def test_export_writes_named_files(tmp_path):
    csv_path = dp._export_csv(_sample_result())
    md_path = dp._export_report(_sample_result())
    assert os.path.exists(csv_path) and os.path.basename(csv_path) == "diagnosis_pt_01.csv"
    assert os.path.exists(md_path) and os.path.basename(md_path) == "diagnosis_report_pt_01.md"


def test_export_empty_state_raises():
    for fn in (dp._export_csv, dp._export_report):
        with pytest.raises(gr.Error):
            fn(None)
        with pytest.raises(gr.Error):
            fn({"candidates": []})
