"""
MCP Marketplace Integration Package

Task 30 Implementation: Build MCP Marketplace Integration
- Connect to external MCP repositories following RAG-MCP extensibility principles
- Implement MCP certification and validation workflows
- Add community-driven MCP sharing and rating system
- Support automatic MCP updates and version management
- Smithery.ai registry integration for comprehensive MCP discovery (7,796+ servers)

@module MCPMarketplace
@author Enhanced Alita KGoT Team
@date 2025
"""

from .mcp_marketplace import (
    # Main marketplace orchestrator
    MCPMarketplaceCore,
    create_mcp_marketplace,
    
    # Core component classes
    MCPRepositoryConnector,
    MCPCertificationEngine,
    MCPCommunityPlatform,
    MCPVersionManager,
    MCPQualityValidator,
    
    # Configuration classes
    MCPMarketplaceConfig,
    QualityAssessmentConfig,
    
    # Data models
    MCPRepository,
    MCPPackage,
    MCPCommunityInteraction,
    QualityMetrics,
    
    # Enums
    RepositoryType,
    MCPCertificationLevel,
    MCPVersionStatus,
    CommunityInteractionType,
    
    # Example usage
    example_marketplace_usage
)

__version__ = "1.0.0"

# Smithery.ai integration metadata
SMITHERY_REGISTRY_URL = "https://smithery.ai"
SMITHERY_API_BASE = "https://registry.smithery.ai"
SMITHERY_SUPPORTED = True

def get_smithery_info():
    """
    Get information about Smithery.ai integration
    
    Returns:
        Dictionary with Smithery.ai integration details
    """
    return {
        "supported": SMITHERY_SUPPORTED,
        "registry_url": SMITHERY_REGISTRY_URL,
        "api_base": SMITHERY_API_BASE,
        "description": "Model Context Protocol Registry with 7,796+ servers and extensions",
        "features": [
            "Comprehensive MCP server discovery",
            "Usage statistics and popularity metrics", 
            "Deployment status tracking",
            "Security scan information",
            "Tool and connection metadata",
            "Remote and local server support"
        ],
        "authentication": {
            "required": False,
            "api_key_env": "SMITHERY_API_KEY",
            "public_access": True,
            "enhanced_access": "Set SMITHERY_API_KEY environment variable for full access"
        }
    }

async def quick_smithery_search(query: str, limit: int = 10):
    """
    Quick search function for Smithery.ai MCPs
    
    Args:
        query: Search query for MCPs
        limit: Maximum number of results to return
        
    Returns:
        List of MCPs from Smithery.ai matching the query
    """
    config = MCPMarketplaceConfig(
        supported_repositories=[RepositoryType.SMITHERY]
    )
    marketplace = create_mcp_marketplace(config)
    
    await marketplace.connect_to_repository(SMITHERY_REGISTRY_URL, RepositoryType.SMITHERY)
    
    results = await marketplace.search_mcps(
        query,
        filters={"repository_type": RepositoryType.SMITHERY}
    )
    
    return results['mcps'][:limit]

__all__ = [
    # Main classes
    "MCPMarketplaceCore",
    "create_mcp_marketplace",
    
    # Component classes
    "MCPRepositoryConnector", 
    "MCPCertificationEngine",
    "MCPCommunityPlatform",
    "MCPVersionManager",
    "MCPQualityValidator",
    
    # Configuration
    "MCPMarketplaceConfig",
    "QualityAssessmentConfig",
    
    # Data models
    "MCPRepository",
    "MCPPackage", 
    "MCPCommunityInteraction",
    "QualityMetrics",
    
    # Enums
    "RepositoryType",
    "MCPCertificationLevel",
    "MCPVersionStatus", 
    "CommunityInteractionType",
    
    # Examples and utilities
    "example_marketplace_usage",
    "quick_smithery_search",
    "get_smithery_info",
    
    # Smithery.ai constants
    "SMITHERY_REGISTRY_URL",
    "SMITHERY_API_BASE", 
    "SMITHERY_SUPPORTED"
] 