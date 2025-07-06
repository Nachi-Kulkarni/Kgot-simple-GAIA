"""
MCP Federation System (Task 44)
================================
This module implements the **initial skeleton** for a distributed MCP Federation.
It exposes HTTP endpoints that allow federated nodes to:
1. Discover available MCPs on a remote node.
2. Execute an MCP remotely in a secure, authenticated manner.

The design follows the *open MCP ecosystem* vision articulated in the RAG-MCP
and Alita papers, enabling **on-the-fly tool sharing** across organizations while
maintaining tractability.

Key Features Implemented
------------------------
• FastAPI-based server with `/discover` and `/execute` endpoints.
• Pydantic models for strong typing & automatic OpenAPI docs.
• Modular registry helper (`get_local_mcp_registry`) – will later integrate with
  existing MCP registries in the project.
• Simple synchronous execution path; streaming & async can be added later.
• Lightweight Python client helpers (`remote_discover`, `remote_execute`) for
  consuming the federation API from other nodes.

Security, governance, and federated-learning extensions are **planned** and will
be implemented in subsequent tasks.
"""

# Standard library imports
from __future__ import annotations
from typing import Any, Dict, List
import os

# Third-party imports
from fastapi import FastAPI, HTTPException, Depends, Header, status
from pydantic import BaseModel

# NOTE: The broader codebase uses Winston (Node.js) for logging. In Python we
# fall back to the built-in `logging` module but keep the same *logical* levels
# for consistency across services.
import logging

logger = logging.getLogger("mcp_federation")
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# Pydantic models (automatic validation & documentation)
# ----------------------------------------------------------------------------

class MCPInfo(BaseModel):
    """Metadata summary for an MCP exposed through federation."""

    name: str  # Unique identifier of the MCP
    description: str | None = None  # Human-readable summary
    version: str | None = None  # Semantic version string


class ExecuteRequest(BaseModel):
    """JSON payload for the `/execute` endpoint."""

    mcp_name: str  # Target MCP identifier
    args: List[Any] = []  # Positional arguments for the MCP
    kwargs: Dict[str, Any] = {}  # Keyword arguments for the MCP


class ExecuteResponse(BaseModel):
    """Success response for an MCP execution."""

    result: Any  # Arbitrary JSON-serialisable result


class PerformanceMetadata(BaseModel):
    """Minimal representation of MCP execution performance stats."""

    mcp_name: str
    success: bool
    latency_ms: float
    cost: float | None = None


class RegisterMCPRequest(BaseModel):
    """Schema for MCP registration submissions."""

    name: str
    description: str | None = None
    version: str | None = "1.0.0"
    source_repo: str | None = None  # URL of git/NPM/PyPI package


# ----------------------------------------------------------------------------
# Registry helper
# ----------------------------------------------------------------------------

from importlib import import_module
from types import ModuleType

_TOOLBOX_FACTORY_MAP = {
    "mcp_toolbox.data_processing_mcps": "create_data_processing_mcps",
    "mcp_toolbox.communication_mcps": "create_communication_mcps",
    "mcp_toolbox.web_information_mcps": "create_web_information_mcps",
}

_cached_registry: Dict[str, Dict[str, Any]] | None = None
_dynamic_registry: Dict[str, Dict[str, Any]] = {}


def _load_toolbox_mcps() -> Dict[str, Dict[str, Any]]:
    """Load MCP instances from toolbox factory functions and return registry mapping."""

    registry: Dict[str, Dict[str, Any]] = {}

    for module_path, factory_name in _TOOLBOX_FACTORY_MAP.items():
        try:
            module: ModuleType = import_module(module_path)  # type: ignore
        except ModuleNotFoundError as exc:
            logger.warning("Toolbox module %s not found: %s", module_path, exc)
            continue

        factory = getattr(module, factory_name, None)
        if factory is None:
            logger.warning("Factory %s not found in %s", factory_name, module_path)
            continue

        try:
            mcps = factory()  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001
            logger.error("Factory %s() in %s failed: %s", factory_name, module_path, exc)
            continue

        for mcp in mcps:
            try:
                name = getattr(mcp, "name")
                description = getattr(mcp, "description", "")
                registry[name] = {
                    "description": description,
                    "version": "1.0.0",
                    "callable": mcp.run,  # type: ignore[attr-defined]
                }
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to register MCP instance from %s: %s", module_path, exc)

    return registry


def get_local_mcp_registry() -> Dict[str, Dict[str, Any]]:
    """Combine ParetoMCPRegistry metadata with actual MCP instances from toolbox."""

    global _cached_registry  # pylint: disable=global-statement
    if _cached_registry is not None:
        return _cached_registry

    registry: Dict[str, Dict[str, Any]] = {}

    # Step 1: Load actual MCP objects from toolbox
    registry.update(_load_toolbox_mcps())

    # Step 2: Ensure all Pareto MCPs are at least discoverable (may be placeholders)
    try:
        from mcp_toolbox import data_processing_mcps  # type: ignore # noqa: F401
        from alita_kgot_enhanced.alita_core.rag_mcp_engine import ParetoMCPRegistry  # type: ignore
    except ModuleNotFoundError:
        ParetoMCPRegistry = None  # type: ignore

    if ParetoMCPRegistry is not None:
        for spec in ParetoMCPRegistry().mcps:
            registry.setdefault(
                spec.name,
                {
                    "description": spec.description,
                    "version": spec.metadata.get("version", "1.0.0") if hasattr(spec, "metadata") else "1.0.0",
                    "callable": lambda *args, _name=spec.name, **kwargs: (
                        f"Execution for MCP '{_name}' is not yet implemented in federation."
                    ),
                },
            )

    # Step 3: include dynamically registered MCPs
    registry.update(_dynamic_registry)

    _cached_registry = registry
    return registry


# ----------------------------------------------------------------------------
# Authentication dependency (token-based placeholder)
# ----------------------------------------------------------------------------

async def verify_token(authorization: str | None = Header(default=None)) -> None:
    """Very thin authentication layer.

    This placeholder simply checks that *some* token is provided when auth is
    enabled in the server's environment. Future work will integrate a proper
    RBAC system and OAuth2/JWT validation.
    """

    expected_token = os.getenv("MCP_FEDERATION_TOKEN")
    if expected_token and authorization != f"Bearer {expected_token}":
        logger.warning("Unauthorized access attempt")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ----------------------------------------------------------------------------
# FastAPI application
# ----------------------------------------------------------------------------

app = FastAPI(title="Alita-KGoT MCP Federation", version="0.1.0")


@app.get("/discover", response_model=List[MCPInfo], dependencies=[Depends(verify_token)])
async def discover() -> List[MCPInfo]:
    """List all MCPs available on this node.

    The endpoint returns metadata but **not** full code. Consumers can decide
    which MCPs to fetch/execute based on this information.
    """

    registry = get_local_mcp_registry()
    logger.debug("/discover called – %d MCPs found", len(registry))

    mcp_summaries: List[MCPInfo] = [
        MCPInfo(
            name=name,
            description=meta.get("description"),
            version=meta.get("version", "1.0.0"),
        )
        for name, meta in registry.items()
    ]
    return mcp_summaries


@app.post("/execute", response_model=ExecuteResponse, dependencies=[Depends(verify_token)])
async def execute(req: ExecuteRequest) -> ExecuteResponse:
    """Execute a local MCP and return its result.

    Parameters
    ----------
    req: ExecuteRequest
        The JSON body containing MCP identifier and its arguments.
    """

    registry = get_local_mcp_registry()

    if req.mcp_name not in registry:
        logger.error("Requested MCP '%s' not found", req.mcp_name)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="MCP not found")

    mcp_callable = registry[req.mcp_name]["callable"]

    # Execute and capture result
    try:
        logger.info("Executing MCP '%s' via federation", req.mcp_name)
        result = mcp_callable(*req.args, **req.kwargs)
        return ExecuteResponse(result=result)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Error executing MCP '%s'", req.mcp_name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


# In-memory storage (placeholder; to be swapped for persistent DB)
_performance_store: List[PerformanceMetadata] = []


@app.post("/metadata", status_code=204, dependencies=[Depends(verify_token)])
async def ingest_metadata(payload: List[PerformanceMetadata]) -> None:  # noqa: D401
    """Receive performance metadata from remote nodes.

    The data is appended to an in-memory list. Future work will aggregate and
    forward to `MCPPerformancePredictor` for federated learning.
    """

    _performance_store.extend(payload)
    logger.debug("Ingested %d performance records; store size now %d", len(payload), len(_performance_store))


@app.get("/metadata", response_model=List[PerformanceMetadata], dependencies=[Depends(verify_token)])
async def get_metadata() -> List[PerformanceMetadata]:
    """Expose collected performance metadata to federation peers."""

    return _performance_store[-500:]  # Return recent subset to limit payload


# ----------------------------------------------------------------------------
# Python client helpers
# ----------------------------------------------------------------------------

# NOTE: We keep dependencies minimal and rely on `requests` which is already in
# the global requirements.txt. If not present it should be added.

import requests  # noqa: E402  # pylint: disable=wrong-import-position

DEFAULT_TIMEOUT = 10  # seconds


def remote_discover(base_url: str, token: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """Retrieve the MCP catalog from a remote federation node."""

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{base_url.rstrip('/')}/discover"
    logger.debug("Fetching remote MCP catalog from %s", url)
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def remote_execute(
    base_url: str,
    mcp_name: str,
    *,
    args: List[Any] | None = None,
    kwargs: Dict[str, Any] | None = None,
    token: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    """Invoke an MCP on a remote federation node and return the result."""

    payload = {
        "mcp_name": mcp_name,
        "args": args or [],
        "kwargs": kwargs or {},
    }
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{base_url.rstrip('/')}/execute"

    logger.info("Executing remote MCP '%s' via %s", mcp_name, url)
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()["result"]


def push_performance_metadata(
    base_url: str,
    records: List[dict],
    token: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
    """Push local performance metadata to a remote node."""

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{base_url.rstrip('/')}/metadata"
    response = requests.post(url, json=records, headers=headers, timeout=timeout)
    response.raise_for_status()


# ----------------------------------------------------------------------------
# CLI entry-point (optional convenience)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn  # Lazy import since it's only needed here

    # Run a development server: `$ python mcp_federation.py --port 8000`
    import argparse

    parser = argparse.ArgumentParser(description="Run MCP Federation node")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run("alita-kgot-enhanced.federation.mcp_federation:app", host=args.host, port=args.port, reload=True)


def _run_quality_assurance(name: str) -> bool:
    """Placeholder invoking MCPQualityAssurance checks."""
    try:
        from alita_kgot_enhanced.quality.mcp_quality_framework import run_quality_checks  # type: ignore
    except ModuleNotFoundError:
        logger.warning("Quality framework not available – skipping QA for %s", name)
        return True  # Allow for now
    try:
        return run_quality_checks(name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Quality checks failed for %s: %s", name, exc)
        return False


def _run_security_checks(name: str) -> bool:
    """Placeholder invoking MCPSecurity checks."""
    try:
        from alita_kgot_enhanced.security.mcp_security_compliance import run_security_checks  # type: ignore
    except ModuleNotFoundError:
        logger.warning("Security compliance framework missing – skipping security check for %s", name)
        return True
    try:
        return run_security_checks(name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Security checks failed for %s: %s", name, exc)
        return False


@app.post("/register", status_code=201, dependencies=[Depends(verify_token)])
async def register_mcp(req: RegisterMCPRequest):
    """Endpoint for federated nodes to submit new MCP metadata.

    Applies automated quality and security checks before accepting.
    """

    if req.name in _dynamic_registry:
        raise HTTPException(status_code=409, detail="MCP already registered")

    if not _run_quality_assurance(req.name):
        raise HTTPException(status_code=400, detail="Quality assurance failed")

    if not _run_security_checks(req.name):
        raise HTTPException(status_code=400, detail="Security compliance failed")

    _dynamic_registry[req.name] = {
        "description": req.description,
        "version": req.version,
        "callable": lambda *args, _name=req.name, **kwargs: (
            f"Dynamic MCP '{_name}' not yet wired for execution."
        ),
    }

    logger.info("Successfully registered new MCP %s via federation", req.name)
    return {"status": "registered", "name": req.name} 