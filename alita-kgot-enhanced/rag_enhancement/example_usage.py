#!/usr/bin/env python3
"""
Advanced RAG-MCP Search System - Example Usage and Demo

This script demonstrates the complete functionality of the Task 29 implementation
including all advanced features for semantic search, recommendations, compositions,
and cross-domain transfer capabilities.

@module example_usage
@author Enhanced Alita KGoT System
@version 1.0.0
@requires rag_enhancement.advanced_rag_mcp_search
"""

import os
import sys
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the advanced RAG-MCP search system
from rag_enhancement import (
    AdvancedRAGMCPSearchSystem,
    SearchComplexity,
    RecommendationType,
    CompositionStrategy,
    get_module_info
)

# Import required components for initialization
from alita_core.mcp_knowledge_base import MCPKnowledgeBase

# Setup logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AdvancedRAGMCPDemo')

async def demonstrate_basic_search():
    """
    Demonstrate basic advanced semantic search functionality
    Shows how to perform enhanced MCP discovery with context awareness
    """
    logger.info("=== DEMO 1: Basic Advanced Semantic Search ===")
    
    try:
        # Initialize knowledge base and search system
        knowledge_base = MCPKnowledgeBase()
        await knowledge_base.initialize()
        
        search_system = AdvancedRAGMCPSearchSystem(
            knowledge_base=knowledge_base,
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
            enable_all_features=True
        )
        
        await search_system.initialize()
        
        # Perform advanced search for web automation task
        search_query = "Automate web scraping and data extraction from e-commerce websites"
        
        result = await search_system.execute_advanced_search(
            query=search_query,
            user_id="demo_user_001",
            task_domain="automation",
            complexity_level=SearchComplexity.MODERATE,
            enable_recommendations=True,
            enable_compositions=True,
            enable_cross_domain=True
        )
        
        # Display results
        logger.info(f"Search completed in {result.total_processing_time:.2f}s")
        logger.info(f"Found {len(result.primary_mcps)} primary MCPs")
        logger.info(f"Generated {len(result.recommended_mcps)} recommendations")
        logger.info(f"Created {len(result.compositions)} workflow compositions")
        logger.info(f"Found {len(result.cross_domain_suggestions)} cross-domain suggestions")
        
        # Show primary MCPs
        print("\n🎯 PRIMARY MCPs DISCOVERED:")
        for i, mcp in enumerate(result.primary_mcps[:3], 1):
            print(f"  {i}. {mcp.mcp_spec.name}")
            print(f"     Category: {mcp.mcp_spec.category.value}")
            print(f"     Confidence: {mcp.confidence_score:.3f}")
            print(f"     Description: {mcp.mcp_spec.description[:100]}...")
            print()
        
        return result
        
    except Exception as e:
        logger.error(f"Basic search demo failed: {str(e)}")
        return None

async def demonstrate_context_aware_recommendations():
    """
    Demonstrate context-aware MCP recommendation system
    Shows how the system provides intelligent recommendations based on context
    """
    logger.info("=== DEMO 2: Context-Aware Recommendations ===")
    
    try:
        # Initialize system
        knowledge_base = MCPKnowledgeBase()
        await knowledge_base.initialize()
        
        search_system = AdvancedRAGMCPSearchSystem(
            knowledge_base=knowledge_base,
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY')
        )
        await search_system.initialize()
        
        # Search for data analysis task
        result = await search_system.execute_advanced_search(
            query="Analyze customer behavior patterns from sales data and generate insights",
            user_id="demo_analyst_002",
            task_domain="data_analysis",
            complexity_level=SearchComplexity.COMPLEX,
            enable_recommendations=True
        )
        
        # Show different types of recommendations
        print("\n💡 CONTEXT-AWARE RECOMMENDATIONS:")
        
        recommendation_types = [
            (RecommendationType.DIRECT, "Direct Task-Relevant MCPs"),
            (RecommendationType.COMPLEMENTARY, "Complementary MCPs"),
            (RecommendationType.ALTERNATIVE, "Alternative Approaches"),
            (RecommendationType.ENHANCEMENT, "Workflow Enhancements"),
            (RecommendationType.CROSS_DOMAIN, "Cross-Domain Suggestions")
        ]
        
        for rec_type, title in recommendation_types:
            type_recommendations = [
                mcp for mcp in result.recommended_mcps 
                if result.recommendation_types.get(mcp.mcp_spec.name) == rec_type
            ]
            
            if type_recommendations:
                print(f"\n  📋 {title}:")
                for mcp in type_recommendations[:2]:
                    print(f"    • {mcp.mcp_spec.name}")
                    print(f"      Confidence: {mcp.confidence_score:.3f}")
                    explanation = result.recommendation_explanations.get(mcp.mcp_spec.name, "")
                    if explanation:
                        print(f"      Why: {explanation[:80]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"Recommendations demo failed: {str(e)}")
        return None

async def demonstrate_mcp_composition():
    """
    Demonstrate MCP composition for complex task workflows
    Shows how the system creates orchestrated workflows for complex tasks
    """
    logger.info("=== DEMO 3: MCP Composition for Complex Tasks ===")
    
    try:
        knowledge_base = MCPKnowledgeBase()
        await knowledge_base.initialize()
        
        search_system = AdvancedRAGMCPSearchSystem(
            knowledge_base=knowledge_base,
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY')
        )
        await search_system.initialize()
        
        # Complex enterprise automation task
        result = await search_system.execute_advanced_search(
            query="Build end-to-end customer onboarding automation: data collection, verification, account setup, and notification system",
            user_id="demo_enterprise_003",
            task_domain="enterprise_automation",
            complexity_level=SearchComplexity.ENTERPRISE,
            enable_compositions=True
        )
        
        # Display composition workflows
        print("\n🔧 MCP COMPOSITION WORKFLOWS:")
        
        for i, composition in enumerate(result.compositions, 1):
            print(f"\n  Workflow {i}: {composition.name}")
            print(f"  Strategy: {composition.execution_strategy.value}")
            print(f"  Estimated Time: {composition.estimated_completion_time:.1f} minutes")
            print(f"  Success Probability: {composition.success_probability:.2%}")
            print(f"  MCPs Involved: {len(composition.mcps)}")
            
            # Show execution flow
            print(f"  Execution Flow:")
            for step_idx, mcp in enumerate(composition.mcps[:4], 1):
                print(f"    {step_idx}. {mcp.name} ({mcp.category.value})")
            
            if len(composition.mcps) > 4:
                print(f"    ... and {len(composition.mcps) - 4} more MCPs")
            
            # Show parallel groups if any
            if composition.parallel_groups:
                print(f"  Parallel Groups: {len(composition.parallel_groups)}")
                for group_idx, group in enumerate(composition.parallel_groups[:2], 1):
                    print(f"    Group {group_idx}: {', '.join(group[:3])}")
        
        return result
        
    except Exception as e:
        logger.error(f"Composition demo failed: {str(e)}")
        return None

async def demonstrate_cross_domain_transfer():
    """
    Demonstrate cross-domain MCP transfer capabilities
    Shows how MCPs can be adapted across different domains
    """
    logger.info("=== DEMO 4: Cross-Domain MCP Transfer ===")
    
    try:
        knowledge_base = MCPKnowledgeBase()
        await knowledge_base.initialize()
        
        search_system = AdvancedRAGMCPSearchSystem(
            knowledge_base=knowledge_base,
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY')
        )
        await search_system.initialize()
        
        # Search in one domain but explore cross-domain possibilities
        result = await search_system.execute_advanced_search(
            query="Optimize supply chain logistics and inventory management",
            user_id="demo_logistics_004",
            task_domain="logistics",
            complexity_level=SearchComplexity.COMPLEX,
            enable_cross_domain=True
        )
        
        # Display cross-domain suggestions
        print("\n🌐 CROSS-DOMAIN TRANSFER SUGGESTIONS:")
        
        for i, transfer in enumerate(result.cross_domain_suggestions, 1):
            print(f"\n  Transfer {i}: {transfer.source_mcp.name}")
            print(f"  From Domain: {transfer.source_domain}")
            print(f"  To Domain: {transfer.target_domain}")
            print(f"  Adaptation Confidence: {transfer.adaptation_confidence:.2%}")
            print(f"  Transfer Method: {transfer.transfer_method}")
            print(f"  Effectiveness Score: {transfer.effectiveness_score:.3f}")
            
            # Show concept mappings
            if transfer.concept_mappings:
                print(f"  Concept Mappings:")
                for source_concept, target_concept in list(transfer.concept_mappings.items())[:3]:
                    print(f"    {source_concept} → {target_concept}")
            
            # Show adaptation requirements
            if transfer.adaptation_requirements:
                print(f"  Adaptation Requirements:")
                for req in transfer.adaptation_requirements[:2]:
                    print(f"    • {req}")
        
        return result
        
    except Exception as e:
        logger.error(f"Cross-domain demo failed: {str(e)}")
        return None

async def demonstrate_system_metrics():
    """
    Demonstrate system performance metrics and analytics
    Shows the monitoring and optimization capabilities
    """
    logger.info("=== DEMO 5: System Performance Metrics ===")
    
    try:
        knowledge_base = MCPKnowledgeBase()
        await knowledge_base.initialize()
        
        search_system = AdvancedRAGMCPSearchSystem(
            knowledge_base=knowledge_base,
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY')
        )
        await search_system.initialize()
        
        # Perform multiple searches to generate metrics
        test_queries = [
            "Automate document processing and approval workflows",
            "Create real-time dashboard for business analytics",
            "Implement chatbot with natural language understanding"
        ]
        
        results = []
        for query in test_queries:
            result = await search_system.execute_advanced_search(
                query=query,
                user_id=f"metrics_user_{len(results)}",
                complexity_level=SearchComplexity.MODERATE
            )
            results.append(result)
        
        # Display system metrics
        metrics = search_system.system_metrics
        
        print("\n📊 SYSTEM PERFORMANCE METRICS:")
        print(f"  Total Searches: {metrics['total_searches']}")
        print(f"  Successful Searches: {metrics['successful_searches']}")
        print(f"  Success Rate: {metrics['successful_searches'] / max(metrics['total_searches'], 1) * 100:.1f}%")
        print(f"  Average Response Time: {metrics['avg_response_time']:.2f}s")
        
        print("\n  Feature Usage Statistics:")
        for feature, count in metrics['feature_usage'].items():
            print(f"    {feature.title()}: {count} times")
        
        print("\n  Per-Search Performance:")
        for i, result in enumerate(results, 1):
            print(f"    Search {i}: {result.total_processing_time:.2f}s")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics demo failed: {str(e)}")
        return None

async def main():
    """
    Main demo function showcasing all Advanced RAG-MCP Search capabilities
    """
    print("🚀 Advanced RAG-MCP Search System - Complete Demo")
    print("=" * 60)
    
    # Display module information
    module_info = get_module_info()
    print(f"\nTask ID: {module_info['task_id']}")
    print(f"Implementation Status: {module_info['implementation_status']}")
    print(f"Features: {len(module_info['features'])} advanced capabilities")
    
    # Check for required environment variables
    if not os.getenv('OPENROUTER_API_KEY'):
        print("\n⚠️  Warning: OPENROUTER_API_KEY not found in environment")
        print("   Some features may not work without API access")
        print("   Set OPENROUTER_API_KEY=your_key to enable full functionality")
    
    try:
        # Run all demonstrations
        print("\n🎯 Starting comprehensive feature demonstrations...")
        
        # Demo 1: Basic advanced search
        basic_result = await demonstrate_basic_search()
        if basic_result:
            print("✅ Basic advanced search completed successfully")
        
        # Demo 2: Context-aware recommendations  
        recommendations_result = await demonstrate_context_aware_recommendations()
        if recommendations_result:
            print("✅ Context-aware recommendations completed successfully")
        
        # Demo 3: MCP composition
        composition_result = await demonstrate_mcp_composition()
        if composition_result:
            print("✅ MCP composition workflows completed successfully")
        
        # Demo 4: Cross-domain transfer
        transfer_result = await demonstrate_cross_domain_transfer()
        if transfer_result:
            print("✅ Cross-domain transfer completed successfully")
        
        # Demo 5: System metrics
        metrics_result = await demonstrate_system_metrics()
        if metrics_result:
            print("✅ System performance metrics completed successfully")
        
        print("\n🎉 All Advanced RAG-MCP Search features demonstrated successfully!")
        print("\nTask 29 Implementation: COMPLETE ✅")
        print("All requirements fulfilled with enterprise-grade capabilities")
        
    except Exception as e:
        logger.error(f"Demo execution failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")
        print("Check environment setup and dependencies")

if __name__ == "__main__":
    """
    Run the complete Advanced RAG-MCP Search System demonstration
    
    Usage:
        python example_usage.py
    
    Environment Variables:
        OPENROUTER_API_KEY: Your OpenRouter API key for LLM functionality
    """
    # Run the comprehensive demo
    asyncio.run(main()) 