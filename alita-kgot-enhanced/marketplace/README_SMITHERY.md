# Smithery.ai Integration for MCP Marketplace

This document describes the Smithery.ai integration added to the MCP Marketplace system. Smithery.ai is a comprehensive registry for Model Context Protocol (MCP) servers and extensions with over **7,796 servers** available.

## 🌟 Features

- **Comprehensive MCP Discovery**: Access to 7,796+ MCP servers and extensions
- **Usage Statistics**: Real-time usage counts and popularity metrics
- **Deployment Status**: Track which MCPs are deployed and ready to use
- **Security Information**: Security scan results and validation status
- **Tool Metadata**: Detailed information about available tools and connections
- **Remote/Local Support**: Support for both remote and local MCP servers

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from marketplace import (
    create_mcp_marketplace, 
    MCPMarketplaceConfig, 
    RepositoryType,
    quick_smithery_search
)

# Quick search for MCPs
async def search_example():
    mcps = await quick_smithery_search("web search", limit=5)
    for mcp in mcps:
        print(f"{mcp['name']}: {mcp['description']}")

asyncio.run(search_example())
```

### Full Marketplace Integration

```python
import asyncio
from marketplace import create_mcp_marketplace, MCPMarketplaceConfig, RepositoryType

async def smithery_integration_example():
    # Create marketplace with Smithery.ai support
    config = MCPMarketplaceConfig(
        supported_repositories=[RepositoryType.SMITHERY],
        enable_community_features=True,
        enable_sequential_thinking=True
    )
    
    marketplace = create_mcp_marketplace(config)
    
    # Connect to Smithery.ai registry
    result = await marketplace.connect_to_repository(
        "https://smithery.ai",
        RepositoryType.SMITHERY
    )
    print(f"Connected to {result['repository']['name']}")
    
    # Search for MCPs
    search_results = await marketplace.search_mcps(
        "memory management",
        filters={"repository_type": RepositoryType.SMITHERY}
    )
    
    # Display results with Smithery metadata
    for mcp in search_results['mcps'][:5]:
        smithery_meta = mcp.get('metadata', {})
        print(f"\n{mcp['name']}")
        print(f"  Use Count: {smithery_meta.get('smithery_use_count', 0):,}")
        print(f"  Deployed: {smithery_meta.get('smithery_is_deployed', False)}")
        print(f"  Security: {smithery_meta.get('smithery_security', {}).get('scanPassed', 'Unknown')}")

asyncio.run(smithery_integration_example())
```

## 🔧 Configuration

### Authentication (Optional)

Smithery.ai provides public access, but you can get enhanced access with an API key:

```bash
export SMITHERY_API_KEY="your_api_key_here"
```

Get your API key from: [https://smithery.ai/account/api-keys](https://smithery.ai/account/api-keys)

### Repository Configuration

```python
config = MCPMarketplaceConfig(
    supported_repositories=[
        RepositoryType.SMITHERY,    # Smithery.ai registry
        RepositoryType.GIT_GITHUB,  # GitHub repositories
        RepositoryType.NPM_REGISTRY # npm packages
    ]
)
```

## 📊 Smithery Metadata

Each MCP from Smithery.ai includes rich metadata:

```python
smithery_meta = mcp.get('metadata', {})
print(f"Qualified Name: {smithery_meta.get('smithery_qualified_name')}")
print(f"Use Count: {smithery_meta.get('smithery_use_count', 0):,}")
print(f"Is Deployed: {smithery_meta.get('smithery_is_deployed', False)}")
print(f"Is Remote: {smithery_meta.get('smithery_is_remote', False)}")
print(f"Tools Available: {len(smithery_meta.get('smithery_tools', []))}")
print(f"Security Scan: {smithery_meta.get('smithery_security', {}).get('scanPassed')}")
```

## 🎯 Advanced Features

### Quality Assessment

```python
# Assess quality of Smithery MCPs
quality_result = await marketplace.assess_mcp_quality(mcp_id)
print(f"Quality Grade: {quality_result['quality_metrics']['quality_grade']}")
print(f"Overall Score: {quality_result['quality_metrics']['overall_score']:.2f}")
```

### Community Features

```python
# Rate and review Smithery MCPs
rating_result = await marketplace.community_platform.submit_rating(
    mcp_id=mcp_id,
    user_id="user123",
    rating=5,
    review_text="Excellent Smithery MCP! Works perfectly."
)
```

### Cross-Repository Search

```python
# Search across all repositories including Smithery
results = await marketplace.search_mcps(
    "database integration",
    filters={
        "min_rating": 4.0,
        "certification_level": ["STANDARD", "PREMIUM"]
    }
)

# Filter Smithery results
smithery_mcps = [
    mcp for mcp in results['mcps'] 
    if mcp.get('metadata', {}).get('smithery_qualified_name')
]
```

## 🛠️ Demo Script

Run the provided demo script to see all features in action:

```bash
# Full demonstration
python smithery_demo.py

# Quick search
python smithery_demo.py quick "web automation"
```

## 📈 Analytics

Get insights about Smithery MCPs in your marketplace:

```python
analytics = await marketplace.get_marketplace_analytics()
print(f"Total MCPs: {analytics['overview']['total_mcps']}")

# Repository breakdown
for repo_type, stats in analytics.get('repository_stats', {}).items():
    print(f"{repo_type}: {stats.get('mcp_count', 0)} MCPs")
```

## 🔍 Search Examples

### Popular MCPs
```python
# Search for popular tools
popular = await marketplace.search_mcps(
    "web search exa perplexity",
    filters={"repository_type": RepositoryType.SMITHERY}
)
```

### Memory Management
```python
# Find memory management MCPs
memory_mcps = await marketplace.search_mcps(
    "memory management mem0",
    filters={"repository_type": RepositoryType.SMITHERY}
)
```

### Browser Automation
```python
# Browser automation tools
browser_mcps = await marketplace.search_mcps(
    "browser automation playwright",
    filters={"repository_type": RepositoryType.SMITHERY}
)
```

## 🔒 Security

- All Smithery MCPs include security scan information
- Access security status via `smithery_security` metadata
- Use marketplace quality assessment for additional validation
- Community ratings provide crowdsourced quality indicators

## 🌐 API Reference

### Smithery API Endpoints Used

- **List Servers**: `GET https://registry.smithery.ai/servers`
- **Server Details**: `GET https://registry.smithery.ai/servers/{qualifiedName}`
- **Search**: Support for query parameters and filtering

### Error Handling

The integration includes robust error handling:
- Graceful fallback to public access if authentication fails
- Rate limiting respect with automatic delays
- Comprehensive logging for troubleshooting

## 📝 Integration Status

- ✅ **Repository Connection**: Full support for Smithery.ai registry
- ✅ **MCP Discovery**: Paginated discovery with metadata extraction
- ✅ **Search Functionality**: Semantic search with filtering
- ✅ **Quality Assessment**: Integration with marketplace quality system
- ✅ **Community Features**: Rating and review support
- ✅ **Analytics**: Cross-repository analytics including Smithery data
- ✅ **Sequential Thinking**: AI-powered analysis of discovery results

## 🤝 Contributing

The Smithery.ai integration is part of the comprehensive MCP Marketplace system. To extend or modify the integration:

1. Review the `MCPRepositoryConnector` class in `mcp_marketplace.py`
2. See `_discover_smithery_mcps()` method for core discovery logic
3. Check `_analyze_smithery_discovery()` for Sequential Thinking integration
4. Use the demo script for testing changes

## 📚 Resources

- [Smithery.ai Official Site](https://smithery.ai/)
- [Smithery.ai Documentation](https://smithery.ai/docs)
- [Registry API Docs](https://smithery.ai/docs/use/registry)
- [MCP Official Documentation](https://modelcontextprotocol.io/)

---

*This integration enables comprehensive MCP discovery by connecting to the largest registry of Model Context Protocol servers available today.* 