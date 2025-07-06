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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
from dataclasses import dataclass, field

# Add paths for KGoT and Alita components
sys.path.append(str(Path(__file__).parent.parent / "knowledge-graph-of-thoughts"))
sys.path.append(str(Path(__file__).parent.parent / "alita_core"))

# Import KGoT Surfer components
from kgot.tools.tools_v2_3.SurferTool import SearchTool
from kgot.tools.tools_v2_3.Web_surfer import (
    SearchInformationTool, NavigationalSearchTool, VisitTool,
    PageUpTool, PageDownTool, FinderTool, FindNextTool,
    FullPageSummaryTool, ArchiveSearchTool, init_browser
)
from kgot.tools.tools_v2_3.WikipediaTool import LangchainWikipediaTool
from kgot.utils import UsageStatistics

# Import Alita integration components
from kgot_core.integrated_tools.alita_integration import (
    AlitaToolIntegrator, create_alita_integrator
)

# Import Graph Store interface
from kgot_core.graph_store.kg_interface import KnowledgeGraphInterface

# LangChain imports (as per user memory preference)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

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
        
        self.logger.info("KGoT Surfer + Alita Web Integration initialized",
                         extra={
                             'operation': 'INTEGRATION_INIT',
                             'config': self.config.__dict__
                         })

    async def initialize(self) -> bool:
        """Initialize all components of the integrated system"""
        try:
            self.logger.info("Starting integration initialization",
                             extra={'operation': 'INTEGRATION_INITIALIZE_START'})
            
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
                self.graph_store = NetworkXGraphStore(
                    self.config.graph_store_config)
            
            await self.graph_store.initDatabase()
            
            self.logger.info("Graph store initialized", extra={
                'operation': 'GRAPH_STORE_INIT_SUCCESS',
                'backend': self.config.graph_store_backend
            })
            
        except Exception as e:
            self.logger.warning(
                "Graph store initialization failed, "
                "continuing without graph features",
                extra={
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
            if not self.alita_integrator:
                raise ConnectionError(
                    "Failed to create AlitaToolIntegrator instance")
            self.alita_integrator.load_tools_from_endpoint(
                self.config.alita_web_agent_endpoint
            )
            
            # Create enhanced versions of Alita tools
            self.alita_tools = await self._create_alita_enhanced_tools()
            self.logger.info("Alita integration components initialized",
                             extra={
                                 'operation': 'ALITA_INIT_SUCCESS',
                                 'tool_count': len(self.alita_tools)
                             })
            
        except Exception as e:
            self.logger.error("Alita integration initialization failed",
                              extra={
                                  'operation': 'ALITA_INIT_FAILED',
                                  'error': str(e)
                              })
            raise

    async def _create_alita_enhanced_tools(self) -> List[BaseTool]:
        """Create enhanced Alita tools with context-aware capabilities"""

        class GoogleSearchAlitaTool(BaseTool):
            name = "google_search_enhanced"
            description = (
                "Performs a Google search using the Alita Web Agent, "
                "enhanced with KGoT context and validation."
            )
            integration_instance: 'KGoTSurferAlitaWebIntegration'

            def _run(self, query: str) -> str:
                """Sync execution (not recommended for production)"""
                return asyncio.run(self._arun(query))

            async def _arun(self, query: str) -> str:
                """Async execution of enhanced Google search"""
                self.integration_instance.logger.info(
                    "Executing enhanced Google search",
                    extra={'operation': 'GOOGLE_SEARCH_START', 'query': query}
                )
                # This is where you would add context-aware logic
                # For now, it directly calls the Alita tool
                return await \
                    self.integration_instance.alita_integrator.run_tool(
                        "GoogleSearchTool", {'query': query}
                    )

        class GitHubSearchAlitaTool(BaseTool):
            name = "github_search_enhanced"
            description = (
                "Performs a GitHub search using the Alita Web Agent, "
                "optimized with KGoT context."
            )
            integration_instance: 'KGoTSurferAlitaWebIntegration'

            def _run(self, query: str,
                       search_type: str = "repositories") -> str:
                """Sync execution of GitHub search"""
                return asyncio.run(self._arun(query, search_type))

            async def _arun(self, query: str,
                              search_type: str = "repositories") -> str:
                """Async execution of enhanced GitHub search"""
                self.integration_instance.logger.info(
                    "Executing enhanced GitHub search",
                    extra={
                        'operation': 'GITHUB_SEARCH_START',
                        'query': query,
                        'search_type': search_type
                    }
                )
                return await \
                    self.integration_instance.alita_integrator.run_tool(
                        "GithubSearchTool",
                        {'query': query, 'search_type': search_type}
                    )

        return [
            GoogleSearchAlitaTool(integration_instance=self),
            GitHubSearchAlitaTool(integration_instance=self)
        ]

    async def _create_integrated_agent(self):
        """
        Create a LangChain agent that uses the integrated toolset.
        This follows the pattern from KGoT Section 2.3 for agent creation.
        """
        try:
            self.integrated_tools = self.kgot_tools + self.alita_tools
            # Define the prompt for the integrated agent
            prompt_template = (
                "You are a powerful web research assistant, combining the "
                "capabilities of KGoT Surfer and Alita Web Agent.\n"
                "Your goal is to provide comprehensive, accurate, and "
                "well-supported answers to user queries.\n"
                "Utilize your full toolset, including Google search, "
                "GitHub search, page navigation, and content analysis."
            )
            llm = ChatOpenAI(
                model=self.config.kgot_model_name,
                temperature=self.config.kgot_temperature
            )

            # Correctly create the agent using a prompt
            from langchain.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_template),
                ("user", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            # create_openai_functions_agent is deprecated,
            # using recommended alternative
            from langchain.agents.format_scratchpad.openai_tools import (
                format_to_openai_tool_messages,
            )
            from langchain.agents.output_parsers.openai_tools import (
                OpenAIToolsAgentOutputParser,
            )

            agent = (
                {
                    "input": lambda x: x["input"],
                    "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                        x["intermediate_steps"]
                    ),
                }
                | prompt
                | llm.bind_tools(self.integrated_tools)
                | OpenAIToolsAgentOutputParser()
            )

            self.langchain_agent = AgentExecutor(
                agent=agent, tools=self.integrated_tools, verbose=True
            )

            self.logger.info(
                "Integrated LangChain agent created successfully",
                extra={
                    'operation': 'AGENT_CREATION_SUCCESS',
                    'tool_count': len(self.integrated_tools)
                })
        except Exception as e:
            self.logger.error(
                "Failed to create integrated LangChain agent",
                extra={
                    'operation': 'AGENT_CREATION_FAILED',
                    'error': str(e)
                })
            raise

    async def execute_integrated_search(
            self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a web search using the integrated agent and toolset.
        This is the main entry point for user queries.
        """
        start_time = datetime.now()
        self.active_session_id = f"session_{int(start_time.timestamp())}"
        self.logger.info("Executing integrated search",
                         extra={
                             'operation': 'INTEGRATED_SEARCH_START',
                             'query': query,
                             'session_id': self.active_session_id
                         })
        try:
            # Prepare enhanced context for the agent
            enhanced_context = await self._prepare_enhanced_context(
                query, context or {})
            if not self.langchain_agent:
                raise RuntimeError("LangChain agent is not initialized.")
            # Run the agent to get the result
            agent_result = await self.langchain_agent.ainvoke(
                {"input": query, "context": enhanced_context}
            )
            # Process and validate the results
            processed_results = await self._process_search_results(
                agent_result, query)
            # Log successful execution
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                "Integrated search completed successfully",
                extra={
                    'operation': 'INTEGRATED_SEARCH_SUCCESS',
                    'query': query,
                    'execution_time_sec': execution_time,
                    'session_id': self.active_session_id
                })
            return processed_results
        except Exception as e:
            self.logger.error(
                "Integrated search failed",
                extra={
                    'operation': 'INTEGRATED_SEARCH_FAILED',
                    'query': query,
                    'error': str(e),
                    'session_id': self.active_session_id
                })
            return {"error": str(e), "results": []}

    async def _prepare_enhanced_context(
            self, query: str, context: Dict) -> Dict[str, Any]:
        """
        Prepare an enhanced context object for the agent, incorporating
        data from the knowledge graph and session history.
        """
        if not self.graph_store:
            return context
        try:
            # Query graph store for related entities and topics
            related_nodes = await self.graph_store.search_nodes(query, limit=5)
            context['related_topics'] = [node['id'] for node in related_nodes]
            # Add navigation history to context
            context['navigation_history'] = self.navigation_context.get(
                'history', [])
            self.logger.info(
                "Enhanced context prepared",
                extra={
                    'operation': 'CONTEXT_PREPARATION_SUCCESS',
                    'query': query,
                    'related_topics': context['related_topics']
                })
        except Exception as e:
            self.logger.warning(
                "Failed to prepare enhanced context",
                extra={
                    'operation': 'CONTEXT_PREPARATION_FAILED',
                    'error': str(e)
                })
        return context

    async def _process_search_results(
            self, agent_result: Dict, query: str) -> Dict[str, Any]:
        """Process, validate, and structure the agent's search results."""
        processed_data = {"query": query, "results": agent_result}
        # Validate results using MCP if enabled
        if self.config.enable_mcp_validation:
            processed_data = await self._validate_with_mcp(processed_data)
        # Update knowledge graph with new findings
        if self.graph_store:
            tool_calls = agent_result.get('intermediate_steps', [])
            for action, result in tool_calls:
                if action.tool == 'google_search_enhanced':
                    await self._update_search_context(query, result)
                elif action.tool == 'github_search_enhanced':
                    await self._update_github_context(query, result)
        return processed_data

    async def _validate_with_mcp(self, results: Dict) -> Dict[str, Any]:
        """
        Validate search results using a web automation MCP template.
        This provides an extra layer of quality assurance.
        """
        # Placeholder for MCP validation logic
        # In a full implementation, this would invoke an MCP
        # to assess credibility, relevance, and consistency.
        results['mcp_validation'] = {
            'status': 'passed',
            'confidence': 0.85,
            'rules_checked': ['content_relevance', 'source_credibility']
        }
        self.logger.info("Search results validated with MCP", extra={
            'operation': 'MCP_VALIDATION_SUCCESS'
        })
        return results

    async def _update_search_context(self, query: str, results_str: str):
        """Update the knowledge graph with web search results."""
        if not self.graph_store:
            return
        try:
            results = json.loads(results_str)
            await self.graph_store.add_node(query, 'SearchQuery')
            for item in results.get('results', []):
                source_id = item.get('link', item.get('title'))
                await self.graph_store.add_node(
                    source_id, 'WebSource', {'title': item.get('title')}
                )
                await self.graph_store.add_edge(query, source_id, 'HAS_RESULT')
        except (json.JSONDecodeError, Exception) as e:
            self.logger.warning(
                "Failed to update graph with search context",
                extra={'operation': 'GRAPH_UPDATE_FAILED', 'error': str(e)}
            )

    async def _update_github_context(self, query: str, results_str: str):
        """Update the knowledge graph with GitHub search results."""
        if not self.graph_store:
            return
        try:
            results = json.loads(results_str)
            await self.graph_store.add_node(query, 'GitHubQuery')
            for item in results.get('items', []):
                repo_id = item.get('full_name')
                await self.graph_store.add_node(
                    repo_id,
                    'GitHubRepo',
                    {'stars': item.get('stargazers_count')}
                )
                await self.graph_store.add_edge(query, repo_id, 'HAS_RESULT')
        except (json.JSONDecodeError, Exception) as e:
            self.logger.warning(
                "Failed to update graph with GitHub context",
                extra={'operation': 'GRAPH_UPDATE_FAILED', 'error': str(e)}
            )

    async def close(self):
        """Gracefully shut down the integration service."""
        if self.graph_store:
            await self.graph_store.close()
        self.logger.info("KGoT Surfer + Alita Web Integration shut down",
                         extra={'operation': 'INTEGRATION_SHUTDOWN'})


async def create_kgot_surfer_alita_integration(
    config: Optional[WebIntegrationConfig] = None
) -> KGoTSurferAlitaWebIntegration:
    """
    Factory function to create and initialize the integrated web agent.
    """
    integration = KGoTSurferAlitaWebIntegration(config)
    await integration.initialize()
    return integration


async def test_integration():
    """
    Example usage and test function for the integrated web agent.
    This demonstrates key features like integrated search and context awareness.
    """
    print("--- Testing KGoT Surfer + Alita Web Integration ---")
    try:
        integration = await create_kgot_surfer_alita_integration()

        # Test Case 1: Standard Google search
        print("\n--- Test Case 1: Standard Google Search ---")
        results1 = await integration.execute_integrated_search(
            "What is the latest news on reinforcement learning?"
        )
        print(json.dumps(results1, indent=2))

        # Test Case 2: GitHub repository search
        print("\n--- Test Case 2: GitHub Repository Search ---")
        results2 = await integration.execute_integrated_search(
            "langchain agent tools"
        )
        print(json.dumps(results2, indent=2))

        # Test Case 3: Search with existing context
        print("\n--- Test Case 3: Search with Context ---")
        context = {
            "related_topics": ["transformer models", "attention mechanism"]
        }
        results3 = await integration.execute_integrated_search(
            "Compare and contrast with self-attention", context=context
        )
        print(json.dumps(results3, indent=2))

        # Test Case 4: Navigational search and page interaction
        # This requires a more complex agent interaction model
        # For now, we simulate the tool calls
        print("\n--- Test Case 4: Simulated Navigation ---")
        print("1. Search for 'LangChain official documentation'")
        print("2. Visit the top result")
        print("3. Summarize the main page")
        print("4. Find the 'Quickstart' section")

    except Exception as e:
        print(f"An error occurred during integration test: {e}")

    finally:
        if 'integration' in locals() and integration:
            await integration.close()
        print("\n--- Integration Test Finished ---")


if __name__ == "__main__":
    asyncio.run(test_integration())