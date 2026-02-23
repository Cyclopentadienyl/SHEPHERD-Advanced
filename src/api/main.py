"""
SHEPHERD-Advanced API Service
=============================
FastAPI application for rare disease diagnosis.

Module: src/api/main.py
Absolute Path: /home/user/SHEPHERD-Advanced/src/api/main.py

Purpose:
    Main FastAPI application entry point providing:
    - REST API endpoints for diagnosis, HPO search, disease info
    - Health check and readiness probes
    - CORS and security middleware
    - OpenAPI documentation

Endpoints:
    POST /api/v1/diagnose     - Run diagnosis on patient phenotypes
    GET  /api/v1/hpo/search   - Search HPO terms
    GET  /api/v1/disease/{id} - Get disease information
    GET  /health              - Health check
    GET  /ready               - Readiness probe

Dependencies:
    - fastapi: Web framework
    - uvicorn: ASGI server
    - src.inference.pipeline: DiagnosisPipeline
    - src.ontology: HPO ontology access
    - src.kg: Knowledge graph access

Input:
    - HTTP requests with JSON payloads
    - Patient phenotypes as HPO term lists

Output:
    - JSON responses with diagnosis results
    - OpenAPI schema at /docs

Called by:
    - Frontend WebUI
    - External API clients
    - Health monitoring systems

Version: 1.0.0
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application State
# =============================================================================
class AppState:
    """Application state container"""

    def __init__(self):
        self.pipeline = None
        self.kg = None
        self.ontology = None
        self.is_ready = False
        self.start_time = None
        self.version = "1.0.0"
        self.model_version = "unknown"


app_state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting SHEPHERD-Advanced API...")
    app_state.start_time = datetime.now()

    try:
        # Attempt eager initialization if KG path is configured
        if os.environ.get("SHEPHERD_KG_PATH"):
            initialize_pipeline()
        else:
            logger.info(
                "No SHEPHERD_KG_PATH set. Pipeline will initialize "
                "lazily on first request if configured later."
            )

        app_state.is_ready = True
        logger.info("API service ready")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        app_state.is_ready = False

    yield

    # Shutdown
    logger.info("Shutting down SHEPHERD-Advanced API...")
    app_state.is_ready = False


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="SHEPHERD-Advanced API",
    description="Rare Disease Diagnosis Engine with Knowledge Graph Reasoning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Logging Middleware
# =============================================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} "
        f"- {process_time:.3f}s"
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response


# =============================================================================
# Exception Handlers
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path),
        },
    )


# =============================================================================
# Health & Readiness Endpoints
# =============================================================================
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Basic health check

    Returns service status and uptime.
    """
    uptime = None
    if app_state.start_time:
        uptime = (datetime.now() - app_state.start_time).total_seconds()

    return {
        "status": "healthy",
        "version": app_state.version,
        "uptime_seconds": uptime,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/ready", tags=["Health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe

    Returns whether the service is ready to accept requests.
    """
    if not app_state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )

    return {
        "ready": True,
        "pipeline_loaded": app_state.pipeline is not None,
        "kg_loaded": app_state.kg is not None,
        "ontology_loaded": app_state.ontology is not None,
    }


# =============================================================================
# Import and Register Routes
# =============================================================================
from src.api.routes import diagnose, search, disease, training, system

app.include_router(diagnose.router, prefix="/api/v1", tags=["Diagnosis"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(disease.router, prefix="/api/v1", tags=["Disease"])
app.include_router(training.router, prefix="/api/v1", tags=["Training"])
app.include_router(system.router, prefix="/api/v1", tags=["System"])


# =============================================================================
# Mount Gradio Dashboard
# =============================================================================
try:
    import gradio as gr
    from src.webui.app import create_gradio_app

    gradio_app = create_gradio_app()
    gr.mount_gradio_app(app, gradio_app, path="/ui")
    logger.info("Gradio dashboard mounted at /ui")
except ImportError as e:
    logger.warning(f"Gradio not available, dashboard disabled: {e}")
except Exception as e:
    logger.warning(f"Failed to mount Gradio dashboard: {e}")


# =============================================================================
# PWA Manifest (silences browser 404 for /manifest.json)
# =============================================================================
@app.get("/manifest.json", include_in_schema=False)
async def manifest() -> JSONResponse:
    """Minimal PWA manifest for the Gradio dashboard."""
    return JSONResponse({
        "name": "SHEPHERD-Advanced Dashboard",
        "short_name": "SHEPHERD",
        "start_url": "/ui",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#2563eb",
    })


# =============================================================================
# Chrome DevTools config (silences 404 when F12 is open)
# =============================================================================
@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools_config() -> JSONResponse:
    """Empty config to silence Chrome DevTools 404."""
    return JSONResponse({})


# =============================================================================
# Root Endpoint
# =============================================================================
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """API root - returns basic information"""
    return {
        "name": "SHEPHERD-Advanced API",
        "version": app_state.version,
        "description": "Rare Disease Diagnosis Engine",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# Utility Functions
# =============================================================================
def get_app_state() -> AppState:
    """Get application state (for dependency injection)"""
    return app_state


def initialize_pipeline(
    kg_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
):
    """
    Initialize diagnosis pipeline.

    Loads the knowledge graph and creates the DiagnosisPipeline with optional
    GNN model. Called lazily on first diagnosis request or explicitly at startup.

    Configuration is read from environment variables if not passed directly:
        SHEPHERD_KG_PATH: Path to KG JSON file
        SHEPHERD_CHECKPOINT_PATH: Path to model checkpoint
        SHEPHERD_DATA_DIR: Path to processed graph data directory
        SHEPHERD_DEVICE: Inference device (cpu/cuda)

    Args:
        kg_path: Path to KG JSON file (overrides env var)
        checkpoint_path: Path to trained model checkpoint (overrides env var)
        data_dir: Path to processed data directory (overrides env var)
        device: Inference device (overrides env var)
    """
    if app_state.pipeline is not None:
        return

    logger.info("Initializing diagnosis pipeline...")

    # Resolve paths from args or environment
    kg_path = kg_path or os.environ.get("SHEPHERD_KG_PATH")
    checkpoint_path = checkpoint_path or os.environ.get("SHEPHERD_CHECKPOINT_PATH")
    data_dir = data_dir or os.environ.get("SHEPHERD_DATA_DIR")
    device = device or os.environ.get("SHEPHERD_DEVICE")
    vector_index_path = os.environ.get("SHEPHERD_VECTOR_INDEX_PATH")

    if not kg_path:
        logger.warning(
            "No KG path configured. Set SHEPHERD_KG_PATH or pass kg_path. "
            "Pipeline will not be available."
        )
        return

    try:
        from src.kg.graph import KnowledgeGraph
        from src.inference.pipeline import create_diagnosis_pipeline

        # Step 1: Load knowledge graph
        kg_file = Path(kg_path)
        if not kg_file.exists():
            logger.error(f"KG file not found: {kg_file}")
            return

        logger.info(f"Loading knowledge graph from {kg_file}...")
        kg = KnowledgeGraph.load_json(str(kg_file))
        app_state.kg = kg
        logger.info(f"KG loaded: {kg.total_nodes} nodes, {kg.total_edges} edges")

        # Step 2: Create pipeline (with optional GNN and vector index)
        pipeline = create_diagnosis_pipeline(
            kg=kg,
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            device=device,
            vector_index_path=vector_index_path,
        )
        app_state.pipeline = pipeline

        config = pipeline.get_pipeline_config()
        app_state.model_version = config.get("version", "unknown")
        logger.info(
            f"Pipeline initialized: scoring_mode={config.get('scoring_mode')}, "
            f"gnn_ready={config.get('gnn_ready')}, "
            f"vector_index_ready={config.get('vector_index_ready')}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


def initialize_ontology():
    """Initialize HPO ontology"""
    if app_state.ontology is not None:
        return

    logger.info("Initializing ontology...")
    try:
        from src.ontology.loader import OntologyLoader

        # TODO: Load actual HPO ontology
        # loader = OntologyLoader()
        # app_state.ontology = loader.load_hpo()
        logger.info("Ontology initialization placeholder - awaiting data")
    except Exception as e:
        logger.error(f"Failed to initialize ontology: {e}")
        raise


# =============================================================================
# CLI Entry Point
# =============================================================================
def main():
    """Run the API server"""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
