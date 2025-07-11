#!/usr/bin/env python3
"""
Simple Local MCP Server (No Authentication)
==========================================
A simplified version of the MCP Federation System that runs locally without authentication.
This makes it easier to use for development and testing purposes.

Key Features:
- No authentication required
- Simple HTTP endpoints for MCP discovery and execution
- Local registry of available MCPs
- Easy to start and use

Usage:
    python simple_local_mcp_server.py --port 8080
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import logging

# Third-party imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Setup logging
logger = logging.getLogger("simple_mcp_server")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ----------------------------------------------------------------------------
# Pydantic models
# ----------------------------------------------------------------------------

class MCPInfo(BaseModel):
    """Metadata summary for an MCP."""
    name: str
    description: Optional[str] = None
    version: Optional[str] = None


class ExecuteRequest(BaseModel):
    """JSON payload for the `/execute` endpoint."""
    mcp_name: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}


class ExecuteResponse(BaseModel):
    """Success response for an MCP execution."""
    result: Any


class RegisterMCPRequest(BaseModel):
    """Schema for MCP registration submissions."""
    name: str
    description: Optional[str] = None
    version: Optional[str] = "1.0.0"
    source_repo: Optional[str] = None


# ----------------------------------------------------------------------------
# Registry helper (simplified)
# ----------------------------------------------------------------------------

from importlib import import_module
from types import ModuleType

_TOOLBOX_FACTORY_MAP = {
    "mcp_toolbox.data_processing_mcps": "create_data_processing_mcps",
    "mcp_toolbox.communication_mcps": "create_communication_mcps",
    "mcp_toolbox.web_information_mcps": "create_web_information_mcps",
}

_cached_registry: Optional[Dict[str, Dict[str, Any]]] = None
_dynamic_registry: Dict[str, Dict[str, Any]] = {}


def _load_toolbox_mcps() -> Dict[str, Dict[str, Any]]:
    """Load MCP instances from toolbox factory functions."""
    registry: Dict[str, Dict[str, Any]] = {}

    for module_path, factory_name in _TOOLBOX_FACTORY_MAP.items():
        try:
            module: ModuleType = import_module(module_path)
        except ModuleNotFoundError as exc:
            logger.warning("Toolbox module %s not found: %s", module_path, exc)
            continue

        factory = getattr(module, factory_name, None)
        if factory is None:
            logger.warning("Factory %s not found in %s", factory_name, module_path)
            continue

        try:
            mcps = factory()
        except Exception as exc:
            logger.error("Factory %s() in %s failed: %s", factory_name, module_path, exc)
            continue

        for mcp in mcps:
            try:
                name = getattr(mcp, "name")
                description = getattr(mcp, "description", "")
                registry[name] = {
                    "description": description,
                    "version": "1.0.0",
                    "callable": mcp.run,
                }
            except Exception as exc:
                logger.error("Failed to register MCP instance from %s: %s", module_path, exc)

    return registry


def get_local_mcp_registry() -> Dict[str, Dict[str, Any]]:
    """Get the local MCP registry with all available MCPs."""
    global _cached_registry
    if _cached_registry is not None:
        return _cached_registry

    registry: Dict[str, Dict[str, Any]] = {}

    # Load actual MCP objects from toolbox
    registry.update(_load_toolbox_mcps())

    # Load Pareto MCPs if available
    try:
        from alita_kgot_enhanced.alita_core.rag_mcp_engine import ParetoMCPRegistry
        for spec in ParetoMCPRegistry().mcps:
            registry.setdefault(
                spec.name,
                {
                    "description": spec.description,
                    "version": getattr(spec, "metadata", {}).get("version", "1.0.0"),
                    "callable": lambda *args, _name=spec.name, **kwargs: (
                        f"Execution for MCP '{_name}' is not yet implemented."
                    ),
                },
            )
    except (ModuleNotFoundError, ImportError):
        logger.info("ParetoMCPRegistry not available, skipping")

    # Include dynamically registered MCPs
    registry.update(_dynamic_registry)

    _cached_registry = registry
    return registry


# ----------------------------------------------------------------------------
# FastAPI application (no authentication)
# ----------------------------------------------------------------------------

app = FastAPI(
    title="Simple Local MCP Server",
    description="A simplified MCP server for local development (no authentication required)",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Root endpoint with server info."""
    return {
        "message": "Simple Local MCP Server",
        "version": "1.0.0",
        "endpoints": [
            "/discover - List available MCPs",
            "/execute - Execute an MCP",
            "/register - Register a new MCP",
            "/health - Health check"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    registry = get_local_mcp_registry()
    return {
        "status": "healthy",
        "mcps_available": len(registry),
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }


@app.get("/discover", response_model=List[MCPInfo])
async def discover() -> List[MCPInfo]:
    """List all MCPs available on this server."""
    registry = get_local_mcp_registry()
    logger.info("/discover called – %d MCPs found", len(registry))

    mcp_summaries: List[MCPInfo] = [
        MCPInfo(
            name=name,
            description=meta.get("description"),
            version=meta.get("version", "1.0.0"),
        )
        for name, meta in registry.items()
    ]
    return mcp_summaries


@app.post("/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest) -> ExecuteResponse:
    """Execute a local MCP and return its result."""
    registry = get_local_mcp_registry()

    if req.mcp_name not in registry:
        logger.error("Requested MCP '%s' not found", req.mcp_name)
        raise HTTPException(status_code=404, detail=f"MCP '{req.mcp_name}' not found")

    mcp_callable = registry[req.mcp_name]["callable"]

    try:
        logger.info("Executing MCP '%s'", req.mcp_name)
        result = mcp_callable(*req.args, **req.kwargs)
        return ExecuteResponse(result=result)
    except Exception as exc:
        logger.exception("Error executing MCP '%s'", req.mcp_name)
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(exc)}") from exc


@app.post("/register", status_code=201)
async def register_mcp(req: RegisterMCPRequest):
    """Register a new MCP (simplified, no quality/security checks)."""
    if req.name in _dynamic_registry:
        raise HTTPException(status_code=409, detail="MCP already registered")

    _dynamic_registry[req.name] = {
        "description": req.description,
        "version": req.version,
        "callable": lambda *args, _name=req.name, **kwargs: (
            f"Dynamic MCP '{_name}' registered but not yet implemented."
        ),
    }

    # Clear cache to force reload
    global _cached_registry
    _cached_registry = None

    logger.info("Successfully registered new MCP '%s'", req.name)
    return {"status": "registered", "name": req.name}


# ----------------------------------------------------------------------------
# Client helper functions
# ----------------------------------------------------------------------------

import requests

DEFAULT_TIMEOUT = 10


def simple_discover(base_url: str, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """Retrieve the MCP catalog from a simple local server."""
    url = f"{base_url.rstrip('/')}/discover"
    logger.debug("Fetching MCP catalog from %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def simple_execute(
    base_url: str,
    mcp_name: str,
    *,
    args: Optional[List[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    """Execute an MCP on a simple local server."""
    payload = {
        "mcp_name": mcp_name,
        "args": args or [],
        "kwargs": kwargs or {},
    }
    url = f"{base_url.rstrip('/')}/execute"

    logger.info("Executing MCP '%s' via %s", mcp_name, url)
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["result"]


# ----------------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Simple Local MCP Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"\n🚀 Starting Simple Local MCP Server...")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    print(f"📋 API docs available at: http://{args.host}:{args.port}/docs")
    print(f"🔍 Discover MCPs at: http://{args.host}:{args.port}/discover")
    print(f"❤️  Health check at: http://{args.host}:{args.port}/health")
    print("\n" + "="*50)

    uvicorn.run(
        "simple_local_mcp_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )