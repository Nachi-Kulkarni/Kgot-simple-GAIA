"""Simple Federated RAG-MCP Engine
=================================
A simplified version of the Federated RAG-MCP Engine that works with the
simple local MCP server (no authentication required).

This version:
- Removes authentication requirements
- Uses the simple local server endpoints
- Provides easier setup for local development
- Maintains compatibility with the RAG-MCP Engine interface
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional, Dict, Any

try:
    from alita_kgot_enhanced.alita_core.rag_mcp_engine import (
        RAGMCPEngine,
        MCPToolSpec,
        MCPCategory,
    )
except ImportError:
    # Fallback for when the main engine is not available
    class RAGMCPEngine:
        def __init__(self, **kwargs):
            self.pareto_registry = type('MockRegistry', (), {'mcps': []})()
    
    class MCPToolSpec:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MCPCategory:
        DEVELOPMENT = "development"

from simple_local_mcp_server import simple_discover

logger = logging.getLogger("SimpleFederatedRAGMCPEngine")
logger.setLevel(logging.INFO)


class SimpleFederatedRAGMCPEngine(RAGMCPEngine):
    """Simplified federated RAG-MCP engine for local development."""

    def __init__(
        self,
        federation_nodes: Optional[List[str]] = None,
        local_servers: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Create engine and fetch MCP catalogs from simple local servers.

        Parameters
        ----------
        federation_nodes: Optional[List[str]]
            Base URLs of simple local MCP servers (e.g. "http://localhost:8080").
            Alias for local_servers for backward compatibility.
        local_servers: Optional[List[str]]
            Base URLs of simple local MCP servers (e.g. "http://localhost:8080").
            If None, will check environment variable `SIMPLE_MCP_SERVERS`.
        kwargs: dict
            Additional arguments forwarded to parent RAGMCPEngine constructor.
        """

        # Use federation_nodes if provided, otherwise fall back to local_servers
        servers = federation_nodes or local_servers
        
        # Extract server config from env fallback
        if servers is None:
            servers_env = os.getenv("SIMPLE_MCP_SERVERS", "http://127.0.0.1:8080").strip()
            servers = [s.strip() for s in servers_env.split(",") if s.strip()]

        self._local_servers: List[str] = servers or []
        self.federation_nodes: List[str] = self._local_servers  # For compatibility

        # Init base engine first
        super().__init__(**kwargs)

        # Extend registry with simple local servers
        self._extend_registry_with_simple_servers()

        logger.info(
            "SimpleFederatedRAGMCPEngine initialized with %d local servers and %d total MCPs",
            len(self._local_servers),
            len(getattr(self.pareto_registry, 'mcps', [])),
            extra={"operation": "SIMPLE_FEDERATED_RAGMCP_INIT"},
        )

    def _extend_registry_with_simple_servers(self) -> None:
        """Fetch MCP catalogs from simple local servers and merge into registry."""

        if not self._local_servers:
            logger.info("No local servers configured – skipping remote discovery")
            return

        for server_url in self._local_servers:
            try:
                catalog = simple_discover(server_url)
            except Exception as exc:
                logger.warning("Discovery failed for %s: %s", server_url, exc)
                continue

            logger.info("Discovered %d MCPs from %s", len(catalog), server_url)

            for item in catalog:
                name: str = item.get("name")
                if not name:
                    continue

                # Use composite name to avoid clashes
                proxy_name = f"{name}@{server_url.replace('http://', '').replace('https://', '')}"

                # Skip duplicates
                if hasattr(self.pareto_registry, 'mcps'):
                    if any(mcp.name == proxy_name for mcp in self.pareto_registry.mcps):
                        continue

                    spec = MCPToolSpec(
                        name=proxy_name,
                        description=item.get("description", "Local MCP"),
                        capabilities=item.get("capabilities", []),
                        category=MCPCategory.DEVELOPMENT,
                        usage_frequency=0.1,  # Default for local tools
                        reliability_score=0.9,  # Local tools are more reliable
                        cost_efficiency=0.95,  # Local execution is cheaper
                        metadata={
                            "local_server": True,
                            "server_url": server_url,
                            "original_name": name,
                            "version": item.get("version", "1.0.0"),
                        },
                    )

                    self.pareto_registry.mcps.append(spec)

        # Recalculate Pareto scores if the method exists
        if hasattr(self.pareto_registry, 'mcps') and hasattr(self.pareto_registry, '_calculate_pareto_score'):
            for mcp in self.pareto_registry.mcps:
                if hasattr(mcp, 'pareto_score'):
                    mcp.pareto_score = self.pareto_registry._calculate_pareto_score(mcp)

    def execute_remote_mcp(self, mcp_name: str, *args, **kwargs) -> Any:
        """Execute an MCP on a remote simple server."""
        # Find the MCP in our registry
        if not hasattr(self.pareto_registry, 'mcps'):
            raise ValueError("No MCP registry available")
        
        target_mcp = None
        for mcp in self.pareto_registry.mcps:
            if mcp.name == mcp_name:
                target_mcp = mcp
                break
        
        if not target_mcp:
            raise ValueError(f"MCP '{mcp_name}' not found in registry")
        
        # Check if it's a local server MCP
        if not target_mcp.metadata.get("local_server"):
            raise ValueError(f"MCP '{mcp_name}' is not a local server MCP")
        
        # Execute on the remote server
        from simple_local_mcp_server import simple_execute
        
        server_url = target_mcp.metadata["server_url"]
        original_name = target_mcp.metadata["original_name"]
        
        return simple_execute(
            server_url,
            original_name,
            args=list(args),
            kwargs=kwargs
        )

    def list_local_server_mcps(self) -> List[Dict[str, Any]]:
        """List all MCPs from local servers."""
        if not hasattr(self.pareto_registry, 'mcps'):
            return []
        
        local_mcps = []
        for mcp in self.pareto_registry.mcps:
            if mcp.metadata.get("local_server"):
                local_mcps.append({
                    "name": mcp.name,
                    "original_name": mcp.metadata["original_name"],
                    "description": mcp.description,
                    "server_url": mcp.metadata["server_url"],
                    "version": mcp.metadata.get("version", "1.0.0"),
                    "reliability_score": mcp.reliability_score,
                    "cost_efficiency": mcp.cost_efficiency,
                })
        
        return local_mcps

    def refresh_local_servers(self) -> None:
        """Refresh the MCP catalog from all configured local servers."""
        # Clear existing local server MCPs
        if hasattr(self.pareto_registry, 'mcps'):
            self.pareto_registry.mcps = [
                mcp for mcp in self.pareto_registry.mcps 
                if not mcp.metadata.get("local_server")
            ]
        
        # Re-discover from local servers
        self._extend_registry_with_simple_servers()
        
        logger.info("Refreshed local server MCPs")


# Convenience function for easy setup
def create_simple_federated_engine(
    local_servers: Optional[List[str]] = None,
    **kwargs
) -> SimpleFederatedRAGMCPEngine:
    """Create a SimpleFederatedRAGMCPEngine with sensible defaults.
    
    Args:
        local_servers: List of local server URLs. If None, uses environment
                      variable SIMPLE_MCP_SERVERS or defaults to localhost:8080
        **kwargs: Additional arguments for the engine
    
    Returns:
        Configured SimpleFederatedRAGMCPEngine instance
    """
    if local_servers is None:
        # Check environment or use default
        servers_env = os.getenv("SIMPLE_MCP_SERVERS", "")
        if servers_env:
            local_servers = [s.strip() for s in servers_env.split(",") if s.strip()]
        else:
            local_servers = ["http://127.0.0.1:8080"]
    
    return SimpleFederatedRAGMCPEngine(
        local_servers=local_servers,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    print("🚀 Simple Federated RAG-MCP Engine Example")
    print("=" * 45)
    
    # Create engine with default local server
    engine = create_simple_federated_engine()
    
    # List available MCPs from local servers
    local_mcps = engine.list_local_server_mcps()
    print(f"\n📋 Found {len(local_mcps)} MCPs from local servers:")
    
    for mcp in local_mcps:
        print(f"  - {mcp['name']} ({mcp['original_name']})")
        print(f"    Server: {mcp['server_url']}")
        print(f"    Description: {mcp['description']}")
        print()
    
    if local_mcps:
        print("💡 To execute an MCP, use:")
        print(f"   result = engine.execute_remote_mcp('{local_mcps[0]['name']}', *args, **kwargs)")
    else:
        print("⚠️  No MCPs found. Make sure a simple local MCP server is running at:")
        print("   http://127.0.0.1:8080")
        print("\n   Start one with:")
        print("   python simple_local_mcp_server.py --port 8080")