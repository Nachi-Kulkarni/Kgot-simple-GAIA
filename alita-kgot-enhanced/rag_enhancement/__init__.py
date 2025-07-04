#!/usr/bin/env python3
"""
RAG Enhancement Module - Advanced RAG-MCP Search System

This module provides advanced RAG-MCP search capabilities implementing Task 29 requirements:
- Enhanced semantic similarity search for MCP discovery
- Context-aware MCP recommendation system
- MCP composition suggestions for complex tasks  
- Cross-domain MCP transfer capabilities

@module rag_enhancement
@author Enhanced Alita KGoT System
@version 1.0.0
"""

# Main system classes for external usage
from .advanced_rag_mcp_search import (
    # Core search system
    AdvancedRAGMCPSearchSystem,
    
    # Individual engine components
    AdvancedMCPDiscoveryEngine,
    ContextAwareRecommendationSystem,
    MCPCompositionEngine,
    CrossDomainTransferSystem,
    
    # Data structures and enums
    AdvancedSearchContext,
    AdvancedSearchResult,
    MCPComposition,
    CrossDomainTransfer,
    SearchComplexity,
    RecommendationType,
    CompositionStrategy
)

# Version information
__version__ = "1.0.0"
__author__ = "Enhanced Alita KGoT System"
__description__ = "Advanced RAG-MCP Search System with intelligent discovery, recommendations, and composition"

# Main classes for easy import
__all__ = [
    # Main system
    'AdvancedRAGMCPSearchSystem',
    
    # Core engines
    'AdvancedMCPDiscoveryEngine',
    'ContextAwareRecommendationSystem', 
    'MCPCompositionEngine',
    'CrossDomainTransferSystem',
    
    # Data structures
    'AdvancedSearchContext',
    'AdvancedSearchResult',
    'MCPComposition',
    'CrossDomainTransfer',
    
    # Enums
    'SearchComplexity',
    'RecommendationType',
    'CompositionStrategy'
]

# Module metadata
module_info = {
    'task_id': 29,
    'implementation_status': 'complete',
    'features': [
        'semantic_similarity_search',
        'context_aware_recommendations',
        'mcp_composition_suggestions',
        'cross_domain_transfer',
        'graph_based_reasoning',
        'langchain_agents',
        'openrouter_integration',
        'winston_logging',
        'performance_optimization'
    ],
    'compliance': {
        'rag_mcp_section_3_2': True,
        'task_29_requirements': True,
        'user_guidelines': True,
        'langchain_agents': True,
        'openrouter_preference': True,
        'winston_logging': True,
        'jsdoc3_comments': True
    }
}

def get_module_info():
    """
    Get comprehensive module information and implementation status
    
    @return Dict containing module metadata and compliance information
    """
    return module_info.copy() 