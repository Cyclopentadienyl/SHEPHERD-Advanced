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
import time
from contextlib import asynccontextmanager
from datetime import datetime
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
        # Lazy initialization - components loaded on first request if not pre-loaded
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
from src.api.routes import diagnose, search, disease

app.include_router(diagnose.router, prefix="/api/v1", tags=["Diagnosis"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(disease.router, prefix="/api/v1", tags=["Disease"])


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


def initialize_pipeline():
    """
    Initialize diagnosis pipeline

    Called lazily on first diagnosis request or explicitly at startup.
    """
    if app_state.pipeline is not None:
        return

    logger.info("Initializing diagnosis pipeline...")
    try:
        from src.inference.pipeline import create_diagnosis_pipeline

        # TODO: Load actual KG and configure pipeline
        # app_state.pipeline = create_diagnosis_pipeline(kg=app_state.kg)
        logger.info("Pipeline initialization placeholder - awaiting KG data")
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
