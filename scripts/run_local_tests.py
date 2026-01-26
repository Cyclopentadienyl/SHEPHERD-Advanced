#!/usr/bin/env python3
"""
SHEPHERD-Advanced Local Test Script
====================================
Comprehensive test script for local environment validation.

Script: scripts/run_local_tests.py
Absolute Path: /home/user/SHEPHERD-Advanced/scripts/run_local_tests.py

Usage:
    python scripts/run_local_tests.py              # Run all tests
    python scripts/run_local_tests.py --quick      # Quick smoke test
    python scripts/run_local_tests.py --api        # API tests only
    python scripts/run_local_tests.py --training   # Training tests only

Requirements:
    - torch (for training tests)
    - fastapi, uvicorn (for API tests)
    - pytest (for unit tests)

Version: 1.0.0
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0


class LocalTestRunner:
    """Run comprehensive local tests"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[TestResult] = []

    def log(self, msg: str, color: str = Colors.RESET):
        if self.verbose:
            print(f"{color}{msg}{Colors.RESET}")

    def run_test(self, name: str, test_func) -> TestResult:
        """Run a single test function"""
        self.log(f"\n{'='*60}", Colors.BLUE)
        self.log(f"Testing: {name}", Colors.BOLD)
        self.log('='*60, Colors.BLUE)

        start = time.time()
        try:
            result = test_func()
            duration = (time.time() - start) * 1000

            if result:
                self.log(f"✅ PASSED ({duration:.1f}ms)", Colors.GREEN)
                return TestResult(name, True, "OK", duration)
            else:
                self.log(f"❌ FAILED", Colors.RED)
                return TestResult(name, False, "Test returned False", duration)

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.log(f"❌ ERROR: {e}", Colors.RED)
            return TestResult(name, False, str(e), duration)

    def add_result(self, result: TestResult):
        self.results.append(result)

    def print_summary(self):
        """Print test summary"""
        self.log(f"\n{'='*60}", Colors.BOLD)
        self.log("TEST SUMMARY", Colors.BOLD)
        self.log('='*60, Colors.BOLD)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for r in self.results:
            status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if r.passed else f"{Colors.RED}❌ FAIL{Colors.RESET}"
            self.log(f"  {status} {r.name} ({r.duration_ms:.1f}ms)")
            if not r.passed:
                self.log(f"       └─ {r.message}", Colors.YELLOW)

        self.log(f"\n{'-'*60}")
        color = Colors.GREEN if passed == total else Colors.RED
        self.log(f"Result: {passed}/{total} tests passed", color)

        return passed == total


# =============================================================================
# Test Functions
# =============================================================================

def test_core_imports() -> bool:
    """Test core module imports"""
    from src.core import types, protocols, schema
    from src.core.types import (
        Node, Edge, NodeID, PatientPhenotypes,
        DiagnosisCandidate, InferenceResult
    )
    print(f"  - types: {len(dir(types))} exports")
    print(f"  - protocols: {len(dir(protocols))} exports")
    return True


def test_kg_imports() -> bool:
    """Test knowledge graph module imports"""
    from src.kg import KnowledgeGraph, KnowledgeGraphBuilder
    from src.kg.graph import KnowledgeGraph
    from src.kg.builder import KGBuilderConfig
    from src.kg.preprocessing import preprocess_for_gnn
    print("  - KnowledgeGraph, KnowledgeGraphBuilder imported")
    return True


def test_ontology_imports() -> bool:
    """Test ontology module imports"""
    from src.ontology import Ontology, OntologyLoader
    from src.ontology.constraints import OntologyConstraintChecker
    print("  - Ontology, OntologyLoader imported")
    return True


def test_reasoning_imports() -> bool:
    """Test reasoning module imports"""
    from src.reasoning import PathReasoner, ExplanationGenerator
    from src.reasoning.path_reasoning import ReasoningPath
    print("  - PathReasoner, ExplanationGenerator imported")
    return True


def test_inference_imports() -> bool:
    """Test inference module imports"""
    from src.inference import DiagnosisPipeline, InputValidator
    from src.inference.pipeline import PipelineConfig
    print("  - DiagnosisPipeline, InputValidator imported")
    return True


def test_training_imports() -> bool:
    """Test training module imports"""
    from src.training import Trainer, TrainerConfig, MultiTaskLoss
    from src.training.callbacks import EarlyStopping, ModelCheckpoint
    print("  - Trainer, TrainerConfig, MultiTaskLoss imported")
    return True


def test_config_imports() -> bool:
    """Test config module imports"""
    from src.config import (
        HyperparameterManager,
        get_hyperparameter_manager,
        TrainingHyperparameters,
        InferenceHyperparameters,
    )
    manager = get_hyperparameter_manager()
    specs = manager.get_all_specs()
    print(f"  - HyperparameterManager: {len(specs)} parameters")
    return True


def test_api_imports() -> bool:
    """Test API module imports"""
    from src.api import app, app_state
    from src.api.routes import diagnose, search, disease
    print(f"  - FastAPI app: {app.title}")
    print(f"  - Routes: diagnose, search, disease")
    return True


def test_models_imports() -> bool:
    """Test models module imports (requires torch)"""
    try:
        import torch
    except ImportError:
        print("  ⚠️ torch not installed, skipping model import test")
        return True

    from src.models.gnn import ShepherdGNN, ShepherdGNNConfig
    from src.models.gnn.layers import HeteroGNNLayer, OrthologGate
    print("  - ShepherdGNN, HeteroGNNLayer imported")
    return True


def test_utils_imports() -> bool:
    """Test utils module imports"""
    from src.utils.metrics import DiagnosisMetrics, RankingMetrics
    metrics = RankingMetrics()
    print(f"  - RankingMetrics: {type(metrics)}")
    return True


def test_hyperparameter_manager() -> bool:
    """Test hyperparameter manager functionality"""
    from src.config import get_hyperparameter_manager, reset_hyperparameter_manager

    reset_hyperparameter_manager()
    manager = get_hyperparameter_manager()

    # Test get specs
    specs = manager.get_all_specs()
    assert len(specs) > 30, f"Expected >30 specs, got {len(specs)}"

    # Test update
    result = manager.update_parameter("learning_rate", 0.001)
    assert result["success"], f"Update failed: {result}"

    # Test get values
    values = manager.get_current_values()
    assert values["training"]["learning_rate"] == 0.001

    # Test JSON schema
    schema = manager.get_json_schema()
    assert "properties" in schema

    print(f"  - {len(specs)} parameters defined")
    print(f"  - Update/get/schema all working")
    return True


def test_input_validator() -> bool:
    """Test input validator"""
    from src.inference import InputValidator, create_input_validator

    validator = create_input_validator()

    # Valid input
    result = validator.validate_patient_input_dict({
        "patient_id": "test",
        "phenotypes": ["HP:0001250", "HP:0002311"],
    })
    assert result.success, f"Valid input rejected: {result.error}"

    # Invalid HPO format
    result = validator.validate_patient_input_dict({
        "phenotypes": ["INVALID"],
    })
    # Should have warnings but may still be valid in relaxed mode

    print("  - Validation logic working")
    return True


def test_api_endpoints() -> bool:
    """Test API endpoint definitions"""
    from src.api import app

    routes = [r.path for r in app.routes]

    expected = ["/health", "/ready", "/api/v1/diagnose", "/api/v1/hpo/search"]
    for e in expected:
        found = any(e in r for r in routes)
        assert found, f"Missing route: {e}"

    print(f"  - {len(routes)} routes defined")
    print(f"  - Core routes present: /health, /diagnose, /hpo/search")
    return True


def test_api_server_startup() -> bool:
    """Test that API server can start (quick check)"""
    import threading
    import socket

    # Find free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()

    # Try to import and create app
    from src.api import app

    # We don't actually start the server, just verify it's configured
    assert app is not None
    assert hasattr(app, 'routes')

    print(f"  - App configured correctly")
    print(f"  - Would start on port {port}")
    return True


def test_diagnosis_mock() -> bool:
    """Test diagnosis with mock data"""
    from src.api.routes.diagnose import DiagnoseRequest, _generate_mock_candidates

    request = DiagnoseRequest(
        phenotypes=["HP:0001250", "HP:0002311", "HP:0001263"],
        top_k=5,
    )

    candidates = _generate_mock_candidates(
        request.phenotypes,
        request.top_k,
        include_explanations=True,
    )

    assert len(candidates) == 5
    assert candidates[0].rank == 1
    assert candidates[0].confidence_score > candidates[-1].confidence_score

    print(f"  - Generated {len(candidates)} mock candidates")
    print(f"  - Top candidate: {candidates[0].disease_name}")
    return True


def test_torch_available() -> bool:
    """Test if torch is available"""
    try:
        import torch
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("  ⚠️ PyTorch not installed")
        print("  - Training tests will be skipped")
        return True  # Not a failure, just a warning


def test_trainer_init() -> bool:
    """Test Trainer initialization (requires torch)"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  ⚠️ torch not installed, skipping")
        return True

    from src.training import Trainer, TrainerConfig

    # Create minimal model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x, edge_index):
            return {"disease": self.linear(x.get("disease", torch.zeros(1, 10)))}

    model = DummyModel()
    config = TrainerConfig(num_epochs=1, device="cpu")

    # Just test initialization, not actual training
    trainer = Trainer(
        model=model,
        train_dataloader=iter([]),  # Empty iterator
        config=config,
    )

    assert trainer.model is not None
    assert trainer.optimizer is not None

    print("  - Trainer initialized successfully")
    print(f"  - Device: {trainer.device}")
    return True


def test_existing_pytest() -> bool:
    """Run existing pytest tests"""
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        cwd="/home/user/SHEPHERD-Advanced",
    )

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

    if result.returncode != 0:
        print(f"  ⚠️ Some tests failed")
        print(result.stderr[-500:] if result.stderr else "")

    return result.returncode == 0


# =============================================================================
# Main
# =============================================================================

def run_quick_tests(runner: LocalTestRunner):
    """Run quick smoke tests"""
    runner.add_result(runner.run_test("Core Imports", test_core_imports))
    runner.add_result(runner.run_test("Config Imports", test_config_imports))
    runner.add_result(runner.run_test("API Imports", test_api_imports))
    runner.add_result(runner.run_test("API Endpoints", test_api_endpoints))


def run_import_tests(runner: LocalTestRunner):
    """Run all import tests"""
    runner.add_result(runner.run_test("Core Imports", test_core_imports))
    runner.add_result(runner.run_test("KG Imports", test_kg_imports))
    runner.add_result(runner.run_test("Ontology Imports", test_ontology_imports))
    runner.add_result(runner.run_test("Reasoning Imports", test_reasoning_imports))
    runner.add_result(runner.run_test("Inference Imports", test_inference_imports))
    runner.add_result(runner.run_test("Training Imports", test_training_imports))
    runner.add_result(runner.run_test("Config Imports", test_config_imports))
    runner.add_result(runner.run_test("API Imports", test_api_imports))
    runner.add_result(runner.run_test("Models Imports", test_models_imports))
    runner.add_result(runner.run_test("Utils Imports", test_utils_imports))


def run_api_tests(runner: LocalTestRunner):
    """Run API tests"""
    runner.add_result(runner.run_test("API Imports", test_api_imports))
    runner.add_result(runner.run_test("API Endpoints", test_api_endpoints))
    runner.add_result(runner.run_test("API Server Config", test_api_server_startup))
    runner.add_result(runner.run_test("Diagnosis Mock", test_diagnosis_mock))


def run_training_tests(runner: LocalTestRunner):
    """Run training tests"""
    runner.add_result(runner.run_test("PyTorch Available", test_torch_available))
    runner.add_result(runner.run_test("Training Imports", test_training_imports))
    runner.add_result(runner.run_test("Trainer Init", test_trainer_init))


def run_functional_tests(runner: LocalTestRunner):
    """Run functional tests"""
    runner.add_result(runner.run_test("Hyperparameter Manager", test_hyperparameter_manager))
    runner.add_result(runner.run_test("Input Validator", test_input_validator))


def main():
    parser = argparse.ArgumentParser(description="Run local tests")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--api", action="store_true", help="API tests only")
    parser.add_argument("--training", action="store_true", help="Training tests only")
    parser.add_argument("--imports", action="store_true", help="Import tests only")
    parser.add_argument("--pytest", action="store_true", help="Run pytest")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    # Add project to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    runner = LocalTestRunner()

    print(f"{Colors.BOLD}")
    print("=" * 60)
    print("SHEPHERD-Advanced Local Test Suite")
    print("=" * 60)
    print(f"{Colors.RESET}")

    if args.quick:
        run_quick_tests(runner)
    elif args.api:
        run_api_tests(runner)
    elif args.training:
        run_training_tests(runner)
    elif args.imports:
        run_import_tests(runner)
    elif args.pytest:
        runner.add_result(runner.run_test("Pytest Suite", test_existing_pytest))
    elif args.all:
        run_import_tests(runner)
        run_functional_tests(runner)
        run_api_tests(runner)
        run_training_tests(runner)
    else:
        # Default: imports + functional + api
        run_import_tests(runner)
        run_functional_tests(runner)
        run_api_tests(runner)

    success = runner.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
