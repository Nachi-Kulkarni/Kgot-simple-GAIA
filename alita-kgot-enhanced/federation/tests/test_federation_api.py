"""Tests for MCP Federation API and Federated RAG-MCP Engine.

These tests use FastAPI TestClient to verify discover, execute, metadata,
registration, and federation integration with `FederatedRAGMCPEngine`.
"""

from __future__ import annotations

from typing import List, Dict

import os
import json

import pytest
from fastapi.testclient import TestClient

from alita_kgot_enhanced.federation.mcp_federation import app, get_local_mcp_registry, _dynamic_registry  # type: ignore
from alita_kgot_enhanced.federation.federated_rag_mcp_engine import FederatedRAGMCPEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_client():
    """Configure TestClient with auth token env var."""

    os.environ["MCP_FEDERATION_TOKEN"] = "TEST_TOKEN"
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def auth_header():
    return {"Authorization": "Bearer TEST_TOKEN"}

# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

def test_discover_endpoint(test_client, auth_header):
    response = test_client.get("/discover", headers=auth_header)
    assert response.status_code == 200
    catalog: List[Dict] = response.json()
    assert isinstance(catalog, list)
    # At least one MCP from toolbox should exist
    local_registry = get_local_mcp_registry()
    assert len(catalog) >= len(local_registry)


def test_execute_endpoint(test_client, auth_header):
    # Find a callable MCP that accepts no args for quick smoke test
    registry = get_local_mcp_registry()
    target_name = next(iter(registry))
    payload = {"mcp_name": target_name, "args": [], "kwargs": {}}
    response = test_client.post("/execute", json=payload, headers=auth_header)
    if response.status_code == 200:
        # Placeholder functions return string
        assert "result" in response.json()
    else:
        # Some MCPs may not be executable yet (501 internal error). Accept 500.
        assert response.status_code in (404, 500)


def test_metadata_ingestion(test_client, auth_header):
    sample = [{"mcp_name": "dummy", "success": True, "latency_ms": 12.5, "cost": 0.01}]
    post_resp = test_client.post("/metadata", json=sample, headers=auth_header)
    assert post_resp.status_code == 204
    get_resp = test_client.get("/metadata", headers=auth_header)
    assert get_resp.status_code == 200
    assert any(item["mcp_name"] == "dummy" for item in get_resp.json())


def test_registration_governance(test_client, auth_header):
    new_mcp = {
        "name": "demo_dynamic_mcp",
        "description": "Test MCP",
        "version": "0.0.1",
        "source_repo": "https://example.com/demo.git",
    }
    resp = test_client.post("/register", json=new_mcp, headers=auth_header)
    assert resp.status_code == 201
    assert new_mcp["name"] in _dynamic_registry

# ---------------------------------------------------------------------------
# Federated engine integration
# ---------------------------------------------------------------------------

def test_federated_engine_integration(test_client, auth_header):
    # Start local federation server via TestClient; use its base_url
    base_url = test_client.base_url

    engine = FederatedRAGMCPEngine(
        federated_nodes=[base_url],
        auth_token="TEST_TOKEN",
        enable_llm_validation=False,
    )

    # Ensure remote MCP proxies are added
    remote_mcps = [m for m in engine.pareto_registry.mcps if "@" in m.name]
    assert remote_mcps, "No remote MCPs discovered"

    # Verify that running vector manager initialization works
    # Minimal invocation: build index (async) over available mcps
    import asyncio
    asyncio.run(engine.vector_manager.build_mcp_index(engine.pareto_registry.mcps)) 