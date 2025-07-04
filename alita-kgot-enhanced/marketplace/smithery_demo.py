#!/usr/bin/env python3
"""
Smithery.ai Integration Demo for MCP Marketplace

This script demonstrates how to use the enhanced MCP Marketplace with Smithery.ai integration.
Smithery.ai is a comprehensive registry for Model Context Protocol servers and extensions.

Usage:
    python smithery_demo.py

Requirements:
    - Set SMITHERY_API_KEY environment variable for full access (optional)
    - Internet connection to access Smithery.ai registry
"""

import asyncio
import logging
import os
from pathlib import Path
import sys

# Add marketplace to path
sys.path.append(str(Path(__file__).parent))

from mcp_marketplace import (
    MCPMarketplaceConfig,
    MCPMarketplaceCore,
    RepositoryType,
    MCPCertificationLevel,
    create_mcp_marketplace
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger('SmitheryDemo')

async def demonstrate_smithery_integration():
    """
    Comprehensive demonstration of Smithery.ai integration
    """
    logger.info("🚀 Starting Smithery.ai Integration Demo")
    
    # Check for API key
    smithery_api_key = os.environ.get('SMITHERY_API_KEY')
    if smithery_api_key:
        logger.info("✅ Found Smithery API key - full access enabled")
    else:
        logger.info("⚠️  No Smithery API key found - using public access")
        logger.info("   Set SMITHERY_API_KEY environment variable for full access")
    
    try:
        # 1. Create marketplace with Smithery.ai support
        logger.info("\n📦 Creating MCP Marketplace with Smithery.ai support...")
        
        config = MCPMarketplaceConfig(
            supported_repositories=[RepositoryType.SMITHERY],
            enable_community_features=True,
            enable_sequential_thinking=True,
            enable_ai_recommendations=True
        )
        
        marketplace = create_mcp_marketplace(config)
        
        # 2. Connect to Smithery.ai registry
        logger.info("\n🔌 Connecting to Smithery.ai registry...")
        
        smithery_result = await marketplace.connect_to_repository(
            "https://smithery.ai",
            RepositoryType.SMITHERY
        )
        
        logger.info(f"✅ Connected to {smithery_result['repository']['name']}")
        logger.info(f"   Total MCPs discovered: {smithery_result['repository']['total_mcps']}")
        
        # 3. Search for popular MCPs
        logger.info("\n🔍 Searching for popular MCPs on Smithery.ai...")
        
        search_queries = [
            "web search exa perplexity",
            "memory management mem0",
            "browser automation playwright",
            "database postgresql sqlite",
            "github gitlab integration"
        ]
        
        all_discovered_mcps = []
        
        for query in search_queries:
            logger.info(f"   Searching: {query}")
            search_results = await marketplace.search_mcps(
                query,
                filters={"repository_type": RepositoryType.SMITHERY}
            )
            
            found_count = len(search_results['mcps'])
            logger.info(f"   Found {found_count} MCPs for '{query}'")
            
            # Add to discovered MCPs
            all_discovered_mcps.extend(search_results['mcps'][:3])  # Top 3 from each search
        
        # Remove duplicates
        unique_mcps = {mcp['id']: mcp for mcp in all_discovered_mcps}
        all_discovered_mcps = list(unique_mcps.values())
        
        logger.info(f"\n📊 Total unique MCPs discovered: {len(all_discovered_mcps)}")
        
        # 4. Display top Smithery MCPs with metadata
        logger.info("\n🏆 Top Smithery.ai MCPs:")
        
        for i, mcp in enumerate(all_discovered_mcps[:10], 1):
            smithery_meta = mcp.get('metadata', {})
            logger.info(f"\n{i}. {mcp['name']}")
            logger.info(f"   Qualified Name: {smithery_meta.get('smithery_qualified_name', 'N/A')}")
            logger.info(f"   Use Count: {smithery_meta.get('smithery_use_count', 0):,}")
            logger.info(f"   Deployed: {'✅' if smithery_meta.get('smithery_is_deployed') else '❌'}")
            logger.info(f"   Remote: {'✅' if smithery_meta.get('smithery_is_remote') else '❌'}")
            logger.info(f"   Tools: {len(smithery_meta.get('smithery_tools', []))}")
            
            # Security info
            security = smithery_meta.get('smithery_security', {})
            if security:
                scan_passed = security.get('scanPassed', 'Unknown')
                logger.info(f"   Security: {'✅ Passed' if scan_passed else '❌ Failed' if scan_passed is False else '⚠️ Unknown'}")
            
            logger.info(f"   Description: {mcp['description'][:100]}...")
        
        # 5. Quality assessment of top MCPs
        logger.info("\n🎯 Quality Assessment of Top MCPs...")
        
        top_mcps = all_discovered_mcps[:5]  # Assess top 5
        quality_results = {}
        
        for mcp in top_mcps:
            try:
                logger.info(f"   Assessing: {mcp['name']}")
                quality_result = await marketplace.assess_mcp_quality(mcp['id'])
                quality_results[mcp['id']] = quality_result
                
                grade = quality_result['quality_metrics']['quality_grade']
                score = quality_result['quality_metrics']['overall_score']
                logger.info(f"   Quality: {grade} grade ({score:.2f}/1.00)")
                
            except Exception as e:
                logger.warning(f"   Quality assessment failed: {str(e)}")
        
        # 6. Community interaction examples
        logger.info("\n👥 Community Interaction Examples...")
        
        # Rate some MCPs
        for i, mcp in enumerate(top_mcps[:3]):
            try:
                rating_result = await marketplace.community_platform.submit_rating(
                    mcp['id'],
                    f"demo_user_{i+1}",
                    5,
                    f"Excellent Smithery MCP! {mcp['name']} works perfectly."
                )
                logger.info(f"   ⭐ Rated {mcp['name']}: {rating_result['rating_id']}")
            except Exception as e:
                logger.warning(f"   Rating failed for {mcp['name']}: {str(e)}")
        
        # 7. Installation example
        logger.info("\n📥 Installation Example...")
        
        if top_mcps:
            sample_mcp = top_mcps[0]
            try:
                install_result = await marketplace.install_mcp(sample_mcp['id'])
                logger.info(f"   ✅ Installation of {sample_mcp['name']}: {install_result['action']}")
                
                # Check for updates
                update_check = await marketplace.version_manager.check_for_updates(sample_mcp['id'])
                updates_count = len(update_check.get('available_updates', []))
                logger.info(f"   📦 Updates available: {updates_count}")
                
            except Exception as e:
                logger.warning(f"   Installation failed: {str(e)}")
        
        # 8. Marketplace analytics
        logger.info("\n📈 Marketplace Analytics...")
        
        analytics = await marketplace.get_marketplace_analytics()
        logger.info(f"   Total MCPs: {analytics['overview']['total_mcps']}")
        logger.info(f"   Certified MCPs: {analytics['certification_stats']['certified']}")
        
        quality_analytics = await marketplace.get_marketplace_quality_analytics()
        logger.info(f"   Quality Distribution: {quality_analytics['grade_distribution']}")
        
        # 9. Smithery-specific insights
        logger.info("\n🔍 Smithery.ai Insights...")
        
        # Count deployment types
        deployed_count = sum(1 for mcp in all_discovered_mcps 
                           if mcp.get('metadata', {}).get('smithery_is_deployed', False))
        remote_count = sum(1 for mcp in all_discovered_mcps 
                          if mcp.get('metadata', {}).get('smithery_is_remote', False))
        
        logger.info(f"   Deployed MCPs: {deployed_count}/{len(all_discovered_mcps)}")
        logger.info(f"   Remote MCPs: {remote_count}/{len(all_discovered_mcps)}")
        
        # Usage statistics
        use_counts = [mcp.get('metadata', {}).get('smithery_use_count', 0) 
                     for mcp in all_discovered_mcps]
        if use_counts:
            avg_usage = sum(use_counts) / len(use_counts)
            max_usage = max(use_counts)
            logger.info(f"   Average Usage: {avg_usage:.1f}")
            logger.info(f"   Max Usage: {max_usage:,}")
        
        # Top tools available
        all_tools = []
        for mcp in all_discovered_mcps:
            tools = mcp.get('metadata', {}).get('smithery_tools', [])
            all_tools.extend(tools)
        
        logger.info(f"   Total Tools Discovered: {len(all_tools)}")
        
        logger.info("\n✅ Smithery.ai Integration Demo completed successfully!")
        
        return {
            'smithery_connection': smithery_result,
            'discovered_mcps': all_discovered_mcps,
            'quality_results': quality_results,
            'analytics': analytics,
            'quality_analytics': quality_analytics
        }
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        raise

async def quick_smithery_search(query: str = "web search"):
    """
    Quick search example for Smithery.ai MCPs
    """
    logger.info(f"🔍 Quick Smithery search for: '{query}'")
    
    # Simple marketplace setup
    config = MCPMarketplaceConfig(
        supported_repositories=[RepositoryType.SMITHERY]
    )
    marketplace = create_mcp_marketplace(config)
    
    # Connect and search
    await marketplace.connect_to_repository("https://smithery.ai", RepositoryType.SMITHERY)
    
    results = await marketplace.search_mcps(
        query,
        filters={"repository_type": RepositoryType.SMITHERY}
    )
    
    logger.info(f"Found {len(results['mcps'])} MCPs")
    
    for i, mcp in enumerate(results['mcps'][:5], 1):
        smithery_meta = mcp.get('metadata', {})
        logger.info(f"{i}. {mcp['name']} (Use Count: {smithery_meta.get('smithery_use_count', 0)})")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick search mode
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "web search"
        asyncio.run(quick_smithery_search(query))
    else:
        # Full demonstration
        asyncio.run(demonstrate_smithery_integration()) 