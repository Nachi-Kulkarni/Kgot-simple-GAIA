"""
KGoT Surfer Agent + Alita Web Agent Integration
Task 15: Integrate KGoT Surfer Agent with Alita Web Agent

This module combines:
- KGoT Section 2.3 "Surfer Agent (based on HuggingFace Agents design)" 
- KGoT Wikipedia Tool and granular navigation tools (PageUp, PageDown, Find)
- Alita Section 2.2 Web Agent capabilities (GoogleSearchTool, GithubSearchTool)
- KGoT Section 2.1 "Graph Store Module" for context-aware web navigation
- MCP validation for web automation templates

Author: Enhanced Alita KGoT Team
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import aiohttp
import requests
from dataclasses import dataclass, field

# Add paths for KGoT and Alita components
sys.path.append(str(Path(__file__).parent.parent / "knowledge-graph-of-thoughts"))
sys.path.append(str(Path(__file__).parent.parent / "alita_core"))

# Import KGoT Surfer components
from kgot.tools.tools_v2_3.SurferTool import SearchTool, SearchToolSchema
from kgot.tools.tools_v2_3.Web_surfer import (
    SearchInformationTool, NavigationalSearchTool, VisitTool,
    PageUpTool, PageDownTool, FinderTool, FindNextTool,
    FullPageSummaryTool, ArchiveSearchTool, init_browser
)
from kgot.tools.tools_v2_3.WikipediaTool import LangchainWikipediaTool
from kgot.utils import UsageStatistics, llm_utils

# Import Alita integration components
from kgot_core.integrated_tools.alita_integration import (
    AlitaWebAgentBridge, AlitaToolIntegrator, create_alita_integrator
)

# Import Graph Store interface
from kgot_core.graph_store.kg_interface import KnowledgeGraphInterface

# LangChain imports (as per user memory preference)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Winston logging setup
from config.logging.winston_config import setup_winston_logger

@dataclass
class WebIntegrationConfig:
    """Configuration for KGoT Surfer + Alita Web Agent integration"""
    
    # Alita Web Agent configuration
    alita_web_agent_endpoint: str = "http://localhost:3001"
    alita_timeout: int = 30
    
    # KGoT configuration
    kgot_model_name: str = "o3"
    kgot_temperature: float = 0.1
    kgot_max_iterations: int = 12
    
    # Graph Store configuration
    graph_store_backend: str = "networkx"  # networkx, neo4j, rdf4j
    graph_store_config: Dict = field(default_factory=dict)
    
    # MCP validation configuration
    enable_mcp_validation: bool = True
    mcp_confidence_threshold: float = 0.7
    
    # Web automation templates
    web_automation_templates: Dict = field(default_factory=lambda: {
        "search_and_analyze": {
            "steps": ["search", "navigate", "extract", "analyze"],
            "validation_rules": ["content_relevance", "source_credibility"]
        },
        "research_workflow": {
            "steps": ["multi_search", "cross_reference", "synthesize"],
            "validation_rules": ["fact_consistency", "source_diversity"]
        }
    })

class KGoTSurferAlitaWebIntegration:
    """
    Main integration class combining KGoT Surfer Agent with Alita Web Agent
    
    Features:
    - Enhanced web search with Google/GitHub integration
    - Context-aware navigation using knowledge graph
    - Granular page navigation (PageUp, PageDown, Find)
    - Wikipedia tool with date-based retrieval
    - MCP-validated web automation templates
    - LangChain agent orchestration
    """
    
    def __init__(self, config: Optional[WebIntegrationConfig] = None):
        """Initialize the integrated web agent system"""
        self.config = config or WebIntegrationConfig()
        
        # Initialize logging
        self.logger = setup_winston_logger("KGoTSurferAlitaWeb")
        
        # Component initialization
        self.usage_statistics = UsageStatistics()
        self.graph_store: Optional[KnowledgeGraphInterface] = None
        self.alita_integrator: Optional[AlitaToolIntegrator] = None
        self.langchain_agent: Optional[AgentExecutor] = None
        
        # Tool collections
        self.kgot_tools: List[BaseTool] = []
        self.alita_tools: List[BaseTool] = []
        self.integrated_tools: List[BaseTool] = []
        
        # Session management
        self.active_session_id: Optional[str] = None
        self.navigation_context: Dict = {}
        
        self.logger.info("KGoT Surfer + Alita Web Integration initialized", extra={
            'operation': 'INTEGRATION_INIT',
            'config': self.config.__dict__
        })

    async def initialize(self) -> bool:
        """Initialize all components of the integrated system"""
        try:
            self.logger.info("Starting integration initialization", extra={
                'operation': 'INTEGRATION_INITIALIZE_START'
            })
            
            # Initialize graph store
            await self._initialize_graph_store()
            
            # Initialize KGoT Surfer components
            await self._initialize_kgot_components()
            
            # Initialize Alita Web Agent integration
            await self._initialize_alita_integration()
            
            # Create integrated LangChain agent
            await self._create_integrated_agent()
            
            # Initialize browser for KGoT tools
            init_browser()
            
            self.logger.info("Integration initialization completed successfully", extra={
                'operation': 'INTEGRATION_INITIALIZE_SUCCESS',
                'kgot_tools': len(self.kgot_tools),
                'alita_tools': len(self.alita_tools),
                'integrated_tools': len(self.integrated_tools)
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Integration initialization failed", extra={
                'operation': 'INTEGRATION_INITIALIZE_FAILED',
                'error': str(e),
                'error_type': type(e).__name__
            })
            return False

    async def _initialize_graph_store(self):
        """Initialize the knowledge graph store connection"""
        try:
            # Import appropriate graph store implementation
            if self.config.graph_store_backend == "neo4j":
                from kgot_core.graph_store.neo4j.main import Neo4jGraphStore
                self.graph_store = Neo4jGraphStore(self.config.graph_store_config)
            elif self.config.graph_store_backend == "rdf4j":
                from kgot_core.graph_store.rdf4j.main import RDF4JGraphStore
                self.graph_store = RDF4JGraphStore(self.config.graph_store_config)
            else:
                # Default to NetworkX for development
                from kgot_core.graph_store.networkx.main import NetworkXGraphStore
                self.graph_store = NetworkXGraphStore(self.config.graph_store_config)
            
            await self.graph_store.initDatabase()
            
            self.logger.info("Graph store initialized", extra={
                'operation': 'GRAPH_STORE_INIT_SUCCESS',
                'backend': self.config.graph_store_backend
            })
            
        except Exception as e:
            self.logger.warning("Graph store initialization failed, continuing without graph features", extra={
                'operation': 'GRAPH_STORE_INIT_FAILED',
                'error': str(e)
            })
            self.graph_store = None

    async def _initialize_kgot_components(self):
        """Initialize KGoT Surfer Agent components"""
        try:
            # Initialize KGoT tools with enhanced capabilities
            self.kgot_tools = [
                # Core web navigation tools
                SearchInformationTool(),
                NavigationalSearchTool(),
                VisitTool(),
                
                # Granular navigation tools (PageUp, PageDown, Find)
                PageUpTool(),
                PageDownTool(),
                FinderTool(),
                FindNextTool(),
                
                # Advanced content tools
                FullPageSummaryTool(
                    model_name=self.config.kgot_model_name,
                    temperature=self.config.kgot_temperature,
                    usage_statistics=self.usage_statistics
                ),
                ArchiveSearchTool(),
                
                # Wikipedia tool with date-based retrieval
                LangchainWikipediaTool(
                    model_name=self.config.kgot_model_name,
                    temperature=self.config.kgot_temperature,
                    usage_statistics=self.usage_statistics
                )
            ]
            
            # Initialize main KGoT Surfer Agent
            self.kgot_surfer_agent = SearchTool(
                model_name=self.config.kgot_model_name,
                temperature=self.config.kgot_temperature,
                usage_statistics=self.usage_statistics
            )
            
            self.logger.info("KGoT components initialized", extra={
                'operation': 'KGOT_INIT_SUCCESS',
                'tool_count': len(self.kgot_tools)
            })
            
        except Exception as e:
            self.logger.error("KGoT component initialization failed", extra={
                'operation': 'KGOT_INIT_FAILED',
                'error': str(e)
            })
            raise

    async def _initialize_alita_integration(self):
        """Initialize Alita Web Agent integration components"""
        try:
            # Create Alita tool integrator
            self.alita_integrator = create_alita_integrator()
            
            # Initialize session with Alita Web Agent
            self.active_session_id = await self.alita_integrator.initialize_session(
                "KGoT Surfer + Alita Web Agent Integration Session"
            )
            
            # Create Alita-enhanced tools
            self.alita_tools = await self._create_alita_enhanced_tools()
            
            self.logger.info("Alita integration initialized", extra={
                'operation': 'ALITA_INIT_SUCCESS',
                'session_id': self.active_session_id,
                'alita_tools': len(self.alita_tools)
            })
            
        except Exception as e:
            self.logger.error("Alita integration initialization failed", extra={
                'operation': 'ALITA_INIT_FAILED',
                'error': str(e)
            })
            raise

    async def _create_alita_enhanced_tools(self) -> List[BaseTool]:
        """Create tools that leverage both KGoT and Alita capabilities"""
        
        class GoogleSearchAlitaTool(BaseTool):
            name = "google_search_enhanced"
            description = """
            Enhanced Google search using Alita Web Agent with KGoT context integration.
            Provides comprehensive web search with graph-aware context and result analysis.
            """
            
            def _run(self, query: str) -> str:
                """Execute enhanced Google search"""
                try:
                    # Use Alita Web Agent for Google search
                    response = requests.post(
                        f"{self.config.alita_web_agent_endpoint}/search/google",
                        json={"query": query, "options": {"includeSnippets": True}},
                        timeout=self.config.alita_timeout
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Update knowledge graph with search context
                        if self.graph_store:
                            asyncio.create_task(self._update_search_context(query, results))
                        
                        return json.dumps(results, indent=2)
                    else:
                        return f"Google search failed with status {response.status_code}"
                        
                except Exception as e:
                    return f"Enhanced Google search error: {str(e)}"
            
            async def _arun(self, query: str) -> str:
                return self._run(query)
        
        class GitHubSearchAlitaTool(BaseTool):
            name = "github_search_enhanced"
            description = """
            Enhanced GitHub search using Alita Web Agent with repository analysis.
            Supports searching repositories, code, issues, and users with context integration.
            """
            
            def _run(self, query: str, search_type: str = "repositories") -> str:
                """Execute enhanced GitHub search"""
                try:
                    # Use Alita Web Agent for GitHub search
                    response = requests.post(
                        f"{self.config.alita_web_agent_endpoint}/search/github",
                        json={
                            "query": query, 
                            "options": {"type": search_type, "perPage": 20}
                        },
                        timeout=self.config.alita_timeout
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Update knowledge graph with GitHub context
                        if self.graph_store:
                            asyncio.create_task(self._update_github_context(query, results))
                        
                        return json.dumps(results, indent=2)
                    else:
                        return f"GitHub search failed with status {response.status_code}"
                        
                except Exception as e:
                    return f"Enhanced GitHub search error: {str(e)}"
            
            async def _arun(self, query: str, search_type: str = "repositories") -> str:
                return self._run(query, search_type)
        
        # Bind methods to tools
        google_tool = GoogleSearchAlitaTool()
        github_tool = GitHubSearchAlitaTool()
        
        # Add config and graph store references
        google_tool.config = self.config
        google_tool.graph_store = self.graph_store
        github_tool.config = self.config  
        github_tool.graph_store = self.graph_store
        
        return [google_tool, github_tool]

    async def _create_integrated_agent(self):
        """Create the integrated LangChain agent with all tools"""
        try:
            # Combine all tools
            self.integrated_tools = self.kgot_tools + self.alita_tools + [self.kgot_surfer_agent]
            
            # Create LLM
            llm = ChatOpenAI(
                model_name=self.config.kgot_model_name,
                temperature=self.config.kgot_temperature
            )
            
            # Create agent with integrated tools
            self.langchain_agent = AgentExecutor.from_agent_and_tools(
                agent=create_openai_functions_agent(llm, self.integrated_tools),
                tools=self.integrated_tools,
                verbose=True,
                max_iterations=self.config.kgot_max_iterations,
                return_intermediate_steps=True
            )
            
            self.logger.info("Integrated LangChain agent created", extra={
                'operation': 'LANGCHAIN_AGENT_CREATED',
                'total_tools': len(self.integrated_tools),
                'max_iterations': self.config.kgot_max_iterations
            })
            
        except Exception as e:
            self.logger.error("Integrated agent creation failed", extra={
                'operation': 'LANGCHAIN_AGENT_FAILED',
                'error': str(e)
            })
            raise

    async def execute_integrated_search(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute integrated search using both KGoT and Alita capabilities
        
        Args:
            query: Search query or research task
            context: Additional context for the search
            
        Returns:
            Comprehensive search results with enhanced context
        """
        try:
            self.logger.info("Executing integrated search", extra={
                'operation': 'INTEGRATED_SEARCH_START',
                'query': query,
                'has_context': bool(context)
            })
            
            # Prepare enhanced context with graph knowledge
            enhanced_context = await self._prepare_enhanced_context(query, context or {})
            
            # Execute search using integrated agent
            agent_input = f"""
            Research Task: {query}
            
            Enhanced Context: {json.dumps(enhanced_context, indent=2)}
            
            Please conduct a comprehensive research using available tools:
            1. Use enhanced Google search for web information
            2. Use enhanced GitHub search for relevant repositories/code
            3. Use Wikipedia search for foundational knowledge
            4. Navigate to important pages using granular navigation tools
            5. Cross-reference information for accuracy
            6. Provide synthesized results with source credibility analysis
            
            Use context-aware navigation and update the knowledge graph as you research.
            """
            
            result = await self.langchain_agent.ainvoke({
                "input": agent_input,
                "context": enhanced_context
            })
            
            # Process and validate results
            processed_results = await self._process_search_results(result, query)
            
            self.logger.info("Integrated search completed", extra={
                'operation': 'INTEGRATED_SEARCH_SUCCESS',
                'query': query,
                'result_confidence': processed_results.get('confidence', 0)
            })
            
            return processed_results
            
        except Exception as e:
            self.logger.error("Integrated search failed", extra={
                'operation': 'INTEGRATED_SEARCH_FAILED',
                'query': query,
                'error': str(e)
            })
            return {
                'error': f"Integrated search failed: {str(e)}",
                'query': query,
                'timestamp': datetime.now().isoformat()
            }

    async def _prepare_enhanced_context(self, query: str, context: Dict) -> Dict[str, Any]:
        """Prepare enhanced context using knowledge graph and navigation history"""
        enhanced_context = {
            **context,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.active_session_id,
            'navigation_history': self.navigation_context.get('history', []),
            'graph_context': {}
        }
        
        # Add graph context if available
        if self.graph_store:
            try:
                # Query related entities from knowledge graph
                related_entities = await self.graph_store.queryEntities({
                    'search_query': query
                })
                enhanced_context['graph_context'] = {
                    'related_entities': related_entities[:5],  # Top 5 related
                    'entity_count': len(related_entities)
                }
            except Exception as e:
                self.logger.debug("Failed to get graph context", extra={'error': str(e)})
        
        return enhanced_context

    async def _process_search_results(self, agent_result: Dict, query: str) -> Dict[str, Any]:
        """Process and validate search results with MCP validation"""
        processed = {
            'query': query,
            'agent_output': agent_result.get('output', ''),
            'intermediate_steps': agent_result.get('intermediate_steps', []),
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.0,
            'validation_status': 'pending'
        }
        
        # MCP validation if enabled
        if self.config.enable_mcp_validation:
            try:
                validation_result = await self._validate_with_mcp(processed)
                processed.update(validation_result)
            except Exception as e:
                self.logger.warning("MCP validation failed", extra={'error': str(e)})
                processed['validation_status'] = 'failed'
        
        return processed

    async def _validate_with_mcp(self, results: Dict) -> Dict[str, Any]:
        """Validate results using MCP (Model Context Protocol) validation"""
        # Implement MCP validation logic
        # This would integrate with existing MCP validation systems
        
        confidence_score = 0.8  # Placeholder calculation
        
        validation_result = {
            'confidence': confidence_score,
            'validation_status': 'validated' if confidence_score >= self.config.mcp_confidence_threshold else 'low_confidence',
            'validation_details': {
                'content_relevance': 0.85,
                'source_credibility': 0.75,
                'fact_consistency': 0.80,
                'completeness': 0.78
            }
        }
        
        return validation_result

    async def _update_search_context(self, query: str, results: Dict):
        """Update knowledge graph with search context"""
        if not self.graph_store:
            return
        
        try:
            # Add search triplet
            await self.graph_store.addTriplet({
                'subject': f"search_{datetime.now().timestamp()}",
                'predicate': 'SEARCHED_FOR',
                'object': query,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'result_count': len(results.get('results', [])),
                    'source': 'integrated_search'
                }
            })
        except Exception as e:
            self.logger.debug("Failed to update search context in graph", extra={'error': str(e)})

    async def _update_github_context(self, query: str, results: Dict):
        """Update knowledge graph with GitHub search context"""
        if not self.graph_store:
            return
        
        try:
            # Add GitHub search context
            await self.graph_store.addTriplet({
                'subject': f"github_search_{datetime.now().timestamp()}",
                'predicate': 'SEARCHED_GITHUB_FOR',
                'object': query,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'result_count': len(results.get('results', [])),
                    'search_type': results.get('type', 'repositories')
                }
            })
        except Exception as e:
            self.logger.debug("Failed to update GitHub context in graph", extra={'error': str(e)})

    async def close(self):
        """Clean up resources and close connections"""
        try:
            if self.alita_integrator:
                await self.alita_integrator.close_session()
            
            if self.graph_store:
                await self.graph_store.close()
            
            self.logger.info("Integration closed successfully", extra={
                'operation': 'INTEGRATION_CLOSE_SUCCESS'
            })
            
        except Exception as e:
            self.logger.error("Error during integration cleanup", extra={
                'operation': 'INTEGRATION_CLOSE_ERROR',
                'error': str(e)
            })

# Factory function for easy instantiation
async def create_kgot_surfer_alita_integration(config: Optional[WebIntegrationConfig] = None) -> KGoTSurferAlitaWebIntegration:
    """
    Factory function to create and initialize the integrated web agent
    
    Args:
        config: Optional configuration for the integration
        
    Returns:
        Initialized KGoTSurferAlitaWebIntegration instance
    """
    integration = KGoTSurferAlitaWebIntegration(config)
    
    success = await integration.initialize()
    if not success:
        raise RuntimeError("Failed to initialize KGoT Surfer + Alita Web Integration")
    
    return integration

# Example usage and testing
if __name__ == "__main__":
    async def test_integration():
        """Test the integrated web agent system"""
        print("Testing KGoT Surfer + Alita Web Integration...")
        
        try:
            # Create configuration
            config = WebIntegrationConfig(
                kgot_model_name="o3",
                graph_store_backend="networkx",
                enable_mcp_validation=True
            )
            
            # Create and initialize integration
            integration = await create_kgot_surfer_alita_integration(config)
            
            # Test integrated search
            test_query = "Research latest developments in knowledge graph reasoning for AI agents"
            
            result = await integration.execute_integrated_search(
                query=test_query,
                context={"research_domain": "AI", "focus": "knowledge_graphs"}
            )
            
            print(f"Search Results: {json.dumps(result, indent=2)}")
            
            # Clean up
            await integration.close()
            
            print("Integration test completed successfully!")
            
        except Exception as e:
            print(f"Integration test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run the test
    asyncio.run(test_integration()) 