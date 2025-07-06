"""Federated RAG-MCP Engine Wrapper
================================
Adds federation-aware discovery to the existing `RAGMCPEngine` by
retrieving MCP catalogs from remote nodes via the `/discover` endpoint and
merging them into the local Pareto registry. This enables cross-organization
MCP sharing (Task 44 todo5).
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional

from alita_kgot_enhanced.alita_core.rag_mcp_engine import (
    RAGMCPEngine,
    MCPToolSpec,
    MCPCategory,
)

from alita_kgot_enhanced.federation.mcp_federation import remote_discover

logger = logging.getLogger("FederatedRAGMCPEngine")
logger.setLevel(logging.INFO)


class FederatedRAGMCPEngine(RAGMCPEngine):
    """Drop-in replacement for `RAGMCPEngine` with federation support."""

    def __init__(
        self,
        *,
        federated_nodes: Optional[List[str]] = None,
        auth_token: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Create engine and immediately fetch remote MCP catalogs.

        Parameters
        ----------
        federated_nodes: list[str] | None
            Base URLs (e.g. "https://acme.ai:8080") of remote federation nodes.
            If *None*, the constructor will look at environment variable
            `MCP_FEDERATION_NODES` which expects a comma-separated list.
        auth_token: str | None
            Bearer token for authenticating against remote nodes (optional).
        kwargs: dict
            Additional arguments forwarded to parent `RAGMCPEngine` constructor.
        """

        # Extract federation config from env fallback
        if federated_nodes is None:
            nodes_env = os.getenv("MCP_FEDERATION_NODES", "").strip()
            federated_nodes = [n.strip() for n in nodes_env.split(",") if n.strip()]

        self._federated_nodes: List[str] = federated_nodes or []
        self._federation_token = auth_token or os.getenv("MCP_FEDERATION_TOKEN")

        # Init base engine first (includes Pareto registry etc.)
        super().__init__(**kwargs)

        # Extend registry with remote MCPs
        self._extend_registry_with_federation()

        logger.info(
            "FederatedRAGMCPEngine initialized with %d remote nodes and %d total MCPs",
            len(self._federated_nodes),
            len(self.pareto_registry.mcps),
            extra={"operation": "FEDERATED_RAGMCP_INIT"},
        )

    # ------------------------------------------------------------------
    # Federation Helpers
    # ------------------------------------------------------------------

    def _extend_registry_with_federation(self) -> None:
        """Fetch MCP catalogs from remote nodes and merge into local registry."""

        if not self._federated_nodes:
            logger.info("No federated nodes configured – skipping remote discovery")
            return

        for base_url in self._federated_nodes:
            try:
                catalog = remote_discover(base_url, token=self._federation_token)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Remote discovery failed for %s: %s", base_url, exc)
                continue

            logger.info("Discovered %d MCPs from %s", len(catalog), base_url)

            for item in catalog:
                name: str = item.get("name")
                if not name:
                    continue

                # Use composite name to avoid clashes with local MCPs
                proxy_name = f"{name}@{base_url}"

                # Skip duplicates
                if any(mcp.name == proxy_name for mcp in self.pareto_registry.mcps):
                    continue

                spec = MCPToolSpec(
                    name=proxy_name,
                    description=item.get("description", "Remote MCP"),
                    capabilities=item.get("capabilities", []),  # Remote may include this later
                    category=MCPCategory.DEVELOPMENT,  # Fallback – could be improved
                    usage_frequency=0.05,  # Conservative default for unknown tools
                    reliability_score=0.80,
                    cost_efficiency=0.80,
                    metadata={
                        "remote": True,
                        "base_url": base_url,
                        "remote_name": name,
                        "version": item.get("version", "1.0.0"),
                    },
                )

                self.pareto_registry.mcps.append(spec)

        # Recalculate Pareto scores to include new tools
        for mcp in self.pareto_registry.mcps:
            mcp.pareto_score = self.pareto_registry._calculate_pareto_score(mcp)  # pylint: disable=protected-access 