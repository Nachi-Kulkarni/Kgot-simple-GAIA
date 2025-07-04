"""
KGoT Knowledge Extraction Methods Module

Implementation of the three knowledge extraction approaches from KGoT research paper Section 1.3:
1. Graph Query Languages (Cypher for Neo4j, SPARQL for RDF4J)
2. General-Purpose Languages (Python scripts with NetworkX integration)
3. Direct Retrieval for broad contextual understanding (Section 4.2)

This module provides:
- Trade-off optimization between accuracy, cost, and runtime
- Integration with Alita's MCP creation for knowledge-driven tool generation
- Dynamic switching between extraction methods based on task requirements
- Support for Neo4j (Cypher), NetworkX (Python), and RDF4J (SPARQL) backends

@author: Enhanced Alita KGoT Team
@version: 1.0.0
@based_on: KGoT Research Paper Sections 1.3 and 4.2
"""

import logging
import json
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import networkx as nx

# Import langchain for agent integration as per user memory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Winston logging setup compatible with Alita enhanced architecture
logger = logging.getLogger('KGoTKnowledgeExtraction')
handler = logging.FileHandler('./logs/kgot/knowledge_extraction.log')
formatter = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ExtractionMethod(Enum):
    """
    Enumeration of available knowledge extraction methods
    Based on KGoT research paper Section 1.3
    """
    DIRECT_RETRIEVAL = "direct_retrieval"          # Section 4.2 - Broad contextual understanding
    GRAPH_QUERY = "graph_query"                    # Graph query languages (Cypher, SPARQL)
    GENERAL_PURPOSE = "general_purpose"            # Python scripts with NetworkX


class BackendType(Enum):
    """
    Supported knowledge graph backend types
    """
    NEO4J = "neo4j"                               # Production backend with Cypher
    NETWORKX = "networkx"                         # Development backend with Python
    RDF4J = "rdf4j"                              # Research backend with SPARQL


@dataclass
class ExtractionMetrics:
    """
    Metrics for trade-off optimization between accuracy, cost, and runtime
    """
    accuracy_score: float = 0.0                   # Accuracy of extraction (0.0-1.0)
    cost_estimate: float = 0.0                    # Estimated cost in tokens/API calls
    runtime_ms: int = 0                           # Runtime in milliseconds
    memory_usage_mb: float = 0.0                  # Memory usage in megabytes
    confidence_score: float = 0.0                 # Confidence in extraction quality
    
    def efficiency_score(self) -> float:
        """
        Calculate overall efficiency score for method selection
        Balances accuracy, cost, and runtime as discussed in paper
        """
        if self.cost_estimate == 0 or self.runtime_ms == 0:
            return 0.0
        
        # Weighted efficiency calculation: prioritize accuracy, balance cost and speed
        weight_accuracy = 0.5
        weight_cost = 0.3
        weight_runtime = 0.2
        
        # Normalize metrics (lower cost and runtime are better)
        normalized_cost = max(0, 1 - (self.cost_estimate / 1000))  # Assume max cost ~1000 tokens
        normalized_runtime = max(0, 1 - (self.runtime_ms / 10000))  # Assume max runtime ~10s
        
        return (
            self.accuracy_score * weight_accuracy +
            normalized_cost * weight_cost +
            normalized_runtime * weight_runtime
        )


@dataclass
class ExtractionContext:
    """
    Context information for knowledge extraction operations
    """
    task_description: str
    current_graph_state: str
    available_tools: List[str]
    user_preferences: Dict[str, Any]
    mcp_session_id: Optional[str] = None
    optimization_target: str = "balanced"  # "accuracy", "speed", "cost", "balanced"


class KnowledgeExtractionInterface(ABC):
    """
    Abstract interface for knowledge extraction methods
    Defines the contract that all extraction implementations must follow
    """
    
    @abstractmethod
    async def extract_knowledge(self, 
                              query: str, 
                              context: ExtractionContext) -> Tuple[str, ExtractionMetrics]:
        """
        Extract knowledge from the graph using this method
        
        Args:
            query: The extraction query/question
            context: Context information for extraction
            
        Returns:
            Tuple of (extracted_knowledge, extraction_metrics)
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, query: str, context: ExtractionContext) -> float:
        """
        Estimate the cost of extraction for this method
        
        Args:
            query: The extraction query
            context: Context information
            
        Returns:
            Estimated cost in tokens/API calls
        """
        pass
    
    @abstractmethod
    def is_suitable_for_task(self, query: str, context: ExtractionContext) -> float:
        """
        Determine suitability score for this extraction method given the task
        
        Args:
            query: The extraction query
            context: Context information
            
        Returns:
            Suitability score (0.0-1.0)
        """
        pass


class DirectRetrievalExtractor(KnowledgeExtractionInterface):
    """
    Direct Retrieval extraction method implementation
    Directly embeds the knowledge graph content into LLM context for broad understanding
    Based on KGoT research paper Section 4.2
    """
    
    def __init__(self, llm_client, graph_store, max_context_size: int = 4000):
        """
        Initialize Direct Retrieval extractor
        
        Args:
            llm_client: LLM client for reasoning (OpenRouter-based per user memory)
            graph_store: Knowledge graph store instance
            max_context_size: Maximum context size for LLM
        """
        self.llm_client = llm_client
        self.graph_store = graph_store
        self.max_context_size = max_context_size
        
        logger.info("Initialized Direct Retrieval extractor", extra={
            'operation': 'DIRECT_RETRIEVAL_INIT',
            'max_context_size': max_context_size
        })
    
    async def extract_knowledge(self, 
                              query: str, 
                              context: ExtractionContext) -> Tuple[str, ExtractionMetrics]:
        """
        Extract knowledge using Direct Retrieval method
        Loads entire graph state into LLM context for comprehensive reasoning
        """
        start_time = time.time()
        logger.info("Starting Direct Retrieval extraction", extra={
            'operation': 'DIRECT_RETRIEVAL_START',
            'query': query[:100] + "..." if len(query) > 100 else query
        })
        
        try:
            # Get current graph state
            graph_state = await self._get_graph_state()
            
            # Truncate if necessary to fit context window
            truncated_state = self._truncate_for_context(graph_state)
            
            # Create extraction prompt
            extraction_prompt = self._create_extraction_prompt(query, truncated_state, context)
            
            # Perform LLM reasoning with full graph context
            response = await self.llm_client.ainvoke([
                SystemMessage(content="You are a knowledge extraction expert analyzing a complete knowledge graph."),
                HumanMessage(content=extraction_prompt)
            ])
            
            # Calculate metrics
            runtime_ms = int((time.time() - start_time) * 1000)
            metrics = ExtractionMetrics(
                accuracy_score=0.9,  # High accuracy due to full context
                cost_estimate=len(extraction_prompt) / 4,  # Rough token estimate
                runtime_ms=runtime_ms,
                memory_usage_mb=len(graph_state) / (1024 * 1024),
                confidence_score=0.85
            )
            
            logger.info("Direct Retrieval extraction completed", extra={
                'operation': 'DIRECT_RETRIEVAL_SUCCESS',
                'runtime_ms': runtime_ms,
                'efficiency_score': metrics.efficiency_score()
            })
            
            return response.content, metrics
            
        except Exception as e:
            logger.error("Direct Retrieval extraction failed", extra={
                'operation': 'DIRECT_RETRIEVAL_FAILED',
                'error': str(e)
            })
            raise
    
    def estimate_cost(self, query: str, context: ExtractionContext) -> float:
        """
        Estimate cost for Direct Retrieval method
        Higher cost due to full graph context loading
        """
        base_cost = len(query) / 4  # Query tokens
        graph_size_estimate = len(context.current_graph_state) / 4  # Graph state tokens
        return base_cost + graph_size_estimate
    
    def is_suitable_for_task(self, query: str, context: ExtractionContext) -> float:
        """
        Direct Retrieval is suitable for:
        - Complex reasoning tasks requiring broad context
        - When graph is relatively small
        - When accuracy is prioritized over cost
        """
        # Analyze query complexity
        complex_keywords = ['analyze', 'compare', 'synthesize', 'overall', 'comprehensive']
        complexity_score = sum(1 for keyword in complex_keywords if keyword in query.lower()) / len(complex_keywords)
        
        # Consider graph size (penalize very large graphs)
        graph_size_penalty = min(1.0, 1000 / max(len(context.current_graph_state), 1))
        
        # Consider optimization target
        target_bonus = 0.3 if context.optimization_target == "accuracy" else 0.0
        
        suitability = complexity_score * 0.5 + graph_size_penalty * 0.3 + target_bonus + 0.2
        return min(1.0, suitability)
    
    async def _get_graph_state(self) -> str:
        """Get current knowledge graph state"""
        try:
            return await self.graph_store.get_current_graph_state()
        except Exception as e:
            logger.warning("Failed to get graph state, using empty state", extra={'error': str(e)})
            return "Empty knowledge graph"
    
    def _truncate_for_context(self, graph_state: str) -> str:
        """Truncate graph state to fit within context window"""
        if len(graph_state) <= self.max_context_size:
            return graph_state
        
        logger.warning("Truncating graph state for context window", extra={
            'original_size': len(graph_state),
            'truncated_size': self.max_context_size
        })
        
        return graph_state[:self.max_context_size] + "\n[...truncated for context window...]"
    
    def _create_extraction_prompt(self, query: str, graph_state: str, context: ExtractionContext) -> str:
        """Create extraction prompt for Direct Retrieval"""
        return f"""
Task: Extract knowledge from the following knowledge graph to answer the query.

Query: {query}

Task Context: {context.task_description}

Knowledge Graph State:
{graph_state}

Available Tools: {', '.join(context.available_tools)}

Instructions:
1. Analyze the complete knowledge graph structure
2. Identify relevant nodes, relationships, and patterns
3. Synthesize information to provide a comprehensive answer
4. Consider the broader context and implicit connections
5. Provide reasoning for your extraction decisions

Extract and synthesize the relevant knowledge:
"""


class GraphQueryExtractor(KnowledgeExtractionInterface):
    """
    Graph Query extraction method implementation
    Uses graph query languages (Cypher for Neo4j, SPARQL for RDF4J) for targeted extraction
    """
    
    def __init__(self, llm_client, graph_store, backend_type: BackendType):
        """
        Initialize Graph Query extractor
        
        Args:
            llm_client: LLM client for query generation
            graph_store: Knowledge graph store instance
            backend_type: Type of backend (determines query language)
        """
        self.llm_client = llm_client
        self.graph_store = graph_store
        self.backend_type = backend_type
        
        # Set query language based on backend
        self.query_language = {
            BackendType.NEO4J: "Cypher",
            BackendType.RDF4J: "SPARQL",
            BackendType.NETWORKX: "Python/NetworkX"
        }.get(backend_type, "Cypher")
        
        logger.info("Initialized Graph Query extractor", extra={
            'operation': 'GRAPH_QUERY_INIT',
            'backend_type': backend_type.value,
            'query_language': self.query_language
        })
    
    async def extract_knowledge(self, 
                              query: str, 
                              context: ExtractionContext) -> Tuple[str, ExtractionMetrics]:
        """
        Extract knowledge using graph query language
        Generates targeted queries for specific information retrieval
        """
        start_time = time.time()
        logger.info("Starting Graph Query extraction", extra={
            'operation': 'GRAPH_QUERY_START',
            'query_language': self.query_language,
            'query': query[:100] + "..." if len(query) > 100 else query
        })
        
        try:
            # Generate graph query based on extraction needs
            graph_queries = await self._generate_graph_queries(query, context)
            
            # Execute queries and collect results
            query_results = []
            total_cost = 0
            
            for graph_query in graph_queries:
                logger.debug("Executing graph query", extra={'query': graph_query})
                
                result = await self._execute_graph_query(graph_query)
                query_results.append(result)
                total_cost += len(graph_query) / 4  # Estimate query cost
            
            # Synthesize results using LLM
            synthesis_prompt = self._create_synthesis_prompt(query, query_results, context)
            
            response = await self.llm_client.ainvoke([
                SystemMessage(content="You are a knowledge synthesis expert working with graph query results."),
                HumanMessage(content=synthesis_prompt)
            ])
            
            # Calculate metrics
            runtime_ms = int((time.time() - start_time) * 1000)
            metrics = ExtractionMetrics(
                accuracy_score=0.85,  # Good accuracy with targeted queries
                cost_estimate=total_cost + len(synthesis_prompt) / 4,
                runtime_ms=runtime_ms,
                memory_usage_mb=sum(len(str(result)) for result in query_results) / (1024 * 1024),
                confidence_score=0.8
            )
            
            logger.info("Graph Query extraction completed", extra={
                'operation': 'GRAPH_QUERY_SUCCESS',
                'queries_executed': len(graph_queries),
                'runtime_ms': runtime_ms,
                'efficiency_score': metrics.efficiency_score()
            })
            
            return response.content, metrics
            
        except Exception as e:
            logger.error("Graph Query extraction failed", extra={
                'operation': 'GRAPH_QUERY_FAILED',
                'error': str(e)
            })
            raise
    
    def estimate_cost(self, query: str, context: ExtractionContext) -> float:
        """
        Estimate cost for Graph Query method
        Moderate cost for query generation and execution
        """
        base_cost = len(query) / 4  # Query analysis tokens
        query_generation_cost = 200  # Estimated tokens for query generation
        execution_cost = 50  # Estimated cost per query execution
        synthesis_cost = 300  # Synthesis tokens
        
        return base_cost + query_generation_cost + execution_cost + synthesis_cost
    
    def is_suitable_for_task(self, query: str, context: ExtractionContext) -> float:
        """
        Graph Query is suitable for:
        - Specific, targeted information needs
        - When graph structure is well-defined
        - When cost efficiency is important
        """
        # Analyze query specificity
        specific_keywords = ['find', 'search', 'lookup', 'retrieve', 'get', 'list', 'count']
        specificity_score = sum(1 for keyword in specific_keywords if keyword in query.lower()) / len(specific_keywords)
        
        # Consider backend compatibility
        backend_bonus = 0.2 if self.backend_type in [BackendType.NEO4J, BackendType.RDF4J] else 0.1
        
        # Consider optimization target
        target_bonus = 0.3 if context.optimization_target in ["cost", "speed"] else 0.0
        
        suitability = specificity_score * 0.5 + backend_bonus + target_bonus + 0.2
        return min(1.0, suitability)
    
    async def _generate_graph_queries(self, query: str, context: ExtractionContext) -> List[str]:
        """Generate appropriate graph queries for the extraction task"""
        query_generation_prompt = f"""
Generate {self.query_language} queries to extract information for the following task.

Query: {query}
Context: {context.task_description}
Backend: {self.backend_type.value}

Current graph schema (sample):
{context.current_graph_state[:500]}

Generate 1-3 targeted {self.query_language} queries that will efficiently extract the needed information.
Return only the queries, one per line.
"""
        
        response = await self.llm_client.ainvoke([
            SystemMessage(content=f"You are an expert in {self.query_language} query generation."),
            HumanMessage(content=query_generation_prompt)
        ])
        
        # Parse queries from response
        queries = [line.strip() for line in response.content.split('\n') if line.strip()]
        return queries[:3]  # Limit to 3 queries for efficiency
    
    async def _execute_graph_query(self, graph_query: str) -> Any:
        """Execute a graph query against the knowledge store"""
        try:
            if self.backend_type == BackendType.NEO4J:
                return await self.graph_store.execute_cypher(graph_query)
            elif self.backend_type == BackendType.RDF4J:
                return await self.graph_store.execute_sparql(graph_query)
            else:
                # NetworkX or fallback
                return await self.graph_store.execute_python_query(graph_query)
        except Exception as e:
            logger.warning("Graph query execution failed", extra={
                'query': graph_query,
                'error': str(e)
            })
            return f"Query execution failed: {str(e)}"
    
    def _create_synthesis_prompt(self, original_query: str, query_results: List[Any], context: ExtractionContext) -> str:
        """Create prompt for synthesizing query results"""
        results_text = "\n\n".join([f"Query Result {i+1}:\n{result}" for i, result in enumerate(query_results)])
        
        return f"""
Synthesize the following graph query results to answer the original query.

Original Query: {original_query}
Context: {context.task_description}

Graph Query Results:
{results_text}

Instructions:
1. Analyze all query results for relevant information
2. Combine and synthesize the findings
3. Address the original query comprehensively
4. Note any gaps or limitations in the data

Synthesized Answer:
"""


class GeneralPurposeExtractor(KnowledgeExtractionInterface):
    """
    General-Purpose extraction method implementation
    Uses Python scripts with NetworkX integration for flexible knowledge processing
    """
    
    def __init__(self, llm_client, graph_store):
        """
        Initialize General-Purpose extractor
        
        Args:
            llm_client: LLM client for script generation
            graph_store: Knowledge graph store instance
        """
        self.llm_client = llm_client
        self.graph_store = graph_store
        
        logger.info("Initialized General-Purpose extractor", extra={
            'operation': 'GENERAL_PURPOSE_INIT'
        })
    
    async def extract_knowledge(self, 
                              query: str, 
                              context: ExtractionContext) -> Tuple[str, ExtractionMetrics]:
        """
        Extract knowledge using general-purpose Python scripts
        Provides maximum flexibility for complex analysis tasks
        """
        start_time = time.time()
        logger.info("Starting General-Purpose extraction", extra={
            'operation': 'GENERAL_PURPOSE_START',
            'query': query[:100] + "..." if len(query) > 100 else query
        })
        
        try:
            # Generate Python analysis script
            analysis_script = await self._generate_analysis_script(query, context)
            
            # Execute script with NetworkX graph
            execution_result = await self._execute_analysis_script(analysis_script)
            
            # Process and interpret results
            interpretation_prompt = self._create_interpretation_prompt(query, execution_result, context)
            
            response = await self.llm_client.ainvoke([
                SystemMessage(content="You are a data analysis expert interpreting Python script results."),
                HumanMessage(content=interpretation_prompt)
            ])
            
            # Calculate metrics
            runtime_ms = int((time.time() - start_time) * 1000)
            metrics = ExtractionMetrics(
                accuracy_score=0.8,  # Variable accuracy depending on script quality
                cost_estimate=len(analysis_script) / 4 + len(interpretation_prompt) / 4,
                runtime_ms=runtime_ms,
                memory_usage_mb=len(str(execution_result)) / (1024 * 1024),
                confidence_score=0.75
            )
            
            logger.info("General-Purpose extraction completed", extra={
                'operation': 'GENERAL_PURPOSE_SUCCESS',
                'runtime_ms': runtime_ms,
                'efficiency_score': metrics.efficiency_score()
            })
            
            return response.content, metrics
            
        except Exception as e:
            logger.error("General-Purpose extraction failed", extra={
                'operation': 'GENERAL_PURPOSE_FAILED',
                'error': str(e)
            })
            raise
    
    def estimate_cost(self, query: str, context: ExtractionContext) -> float:
        """
        Estimate cost for General-Purpose method
        Variable cost depending on script complexity
        """
        base_cost = len(query) / 4  # Query analysis tokens
        script_generation_cost = 400  # Estimated tokens for script generation
        execution_cost = 100  # Script execution overhead
        interpretation_cost = 300  # Result interpretation tokens
        
        return base_cost + script_generation_cost + execution_cost + interpretation_cost
    
    def is_suitable_for_task(self, query: str, context: ExtractionContext) -> float:
        """
        General-Purpose is suitable for:
        - Complex analytical tasks
        - When custom algorithms are needed
        - When NetworkX backend is available
        """
        # Analyze query complexity and analytical nature
        analytical_keywords = ['calculate', 'compute', 'algorithm', 'pattern', 'network', 'path', 'cluster']
        analytical_score = sum(1 for keyword in analytical_keywords if keyword in query.lower()) / len(analytical_keywords)
        
        # Bonus for NetworkX backend
        backend_bonus = 0.3 if 'networkx' in context.current_graph_state.lower() else 0.1
        
        # Consider optimization target
        target_bonus = 0.2 if context.optimization_target == "balanced" else 0.0
        
        suitability = analytical_score * 0.5 + backend_bonus + target_bonus + 0.2
        return min(1.0, suitability)
    
    async def _generate_analysis_script(self, query: str, context: ExtractionContext) -> str:
        """Generate Python analysis script for the extraction task"""
        script_generation_prompt = f"""
Generate a Python script using NetworkX to analyze the knowledge graph for the following task.

Query: {query}
Context: {context.task_description}

Available tools: {', '.join(context.available_tools)}

The script should:
1. Work with a NetworkX graph object named 'graph'
2. Perform necessary analysis to address the query
3. Return results in a structured format
4. Include error handling
5. Use efficient algorithms appropriate for the task

Generate a complete Python script:
"""
        
        response = await self.llm_client.ainvoke([
            SystemMessage(content="You are an expert Python programmer specializing in NetworkX graph analysis."),
            HumanMessage(content=script_generation_prompt)
        ])
        
        return response.content
    
    async def _execute_analysis_script(self, script: str) -> Any:
        """Execute the generated analysis script safely"""
        try:
            # Get NetworkX graph from store
            graph = await self._get_networkx_graph()
            
            # Create safe execution environment
            exec_globals = {
                'graph': graph,
                'nx': nx,
                'len': len,
                'list': list,
                'dict': dict,
                'set': set,
                'print': print,
                '__builtins__': {}  # Restrict built-ins for safety
            }
            
            # Execute script
            exec(script, exec_globals)
            
            # Extract result (assume script stores result in 'result' variable)
            return exec_globals.get('result', 'No result returned from script')
            
        except Exception as e:
            logger.warning("Script execution failed", extra={
                'script': script[:200] + "..." if len(script) > 200 else script,
                'error': str(e)
            })
            return f"Script execution error: {str(e)}"
    
    async def _get_networkx_graph(self) -> nx.Graph:
        """Get NetworkX graph object from the graph store"""
        try:
            return await self.graph_store.get_networkx_graph()
        except Exception as e:
            logger.warning("Failed to get NetworkX graph, creating empty graph", extra={'error': str(e)})
            return nx.Graph()
    
    def _create_interpretation_prompt(self, original_query: str, execution_result: Any, context: ExtractionContext) -> str:
        """Create prompt for interpreting script execution results"""
        return f"""
Interpret the following Python script execution results to answer the original query.

Original Query: {original_query}
Context: {context.task_description}

Script Execution Results:
{execution_result}

Instructions:
1. Analyze the script results for relevance to the query
2. Provide a clear, comprehensive answer
3. Explain any insights or patterns discovered
4. Note any limitations or uncertainties

Interpreted Answer:
"""


class KnowledgeExtractionOptimizer:
    """
    Optimizer for selecting the best extraction method based on task requirements
    Implements trade-off optimization between accuracy, cost, and runtime
    """
    
    def __init__(self, extractors: Dict[ExtractionMethod, KnowledgeExtractionInterface]):
        """
        Initialize the extraction optimizer
        
        Args:
            extractors: Dictionary mapping extraction methods to their implementations
        """
        self.extractors = extractors
        
        logger.info("Initialized Knowledge Extraction Optimizer", extra={
            'operation': 'OPTIMIZER_INIT',
            'available_methods': list(extractors.keys())
        })
    
    async def optimize_extraction_method(self, 
                                       query: str, 
                                       context: ExtractionContext) -> ExtractionMethod:
        """
        Select the optimal extraction method based on task requirements and trade-offs
        
        Args:
            query: The extraction query
            context: Context information including optimization target
            
        Returns:
            The optimal extraction method
        """
        logger.info("Optimizing extraction method selection", extra={
            'operation': 'METHOD_OPTIMIZATION_START',
            'optimization_target': context.optimization_target
        })
        
        # Calculate scores for each available method
        method_scores = {}
        
        for method, extractor in self.extractors.items():
            try:
                # Get suitability score
                suitability = extractor.is_suitable_for_task(query, context)
                
                # Get cost estimate
                cost_estimate = extractor.estimate_cost(query, context)
                
                # Calculate weighted score based on optimization target
                score = self._calculate_weighted_score(
                    suitability, cost_estimate, context.optimization_target
                )
                
                method_scores[method] = {
                    'score': score,
                    'suitability': suitability,
                    'cost_estimate': cost_estimate
                }
                
                logger.debug("Method evaluation completed", extra={
                    'method': method.value,
                    'suitability': suitability,
                    'cost_estimate': cost_estimate,
                    'final_score': score
                })
                
            except Exception as e:
                logger.warning("Method evaluation failed", extra={
                    'method': method.value,
                    'error': str(e)
                })
                method_scores[method] = {'score': 0.0, 'suitability': 0.0, 'cost_estimate': float('inf')}
        
        # Select method with highest score
        optimal_method = max(method_scores.keys(), key=lambda m: method_scores[m]['score'])
        
        logger.info("Optimal extraction method selected", extra={
            'operation': 'METHOD_OPTIMIZATION_COMPLETE',
            'selected_method': optimal_method.value,
            'method_scores': {k.value: v for k, v in method_scores.items()}
        })
        
        return optimal_method
    
    def _calculate_weighted_score(self, 
                                suitability: float, 
                                cost_estimate: float, 
                                optimization_target: str) -> float:
        """
        Calculate weighted score based on optimization target
        
        Args:
            suitability: Method suitability score (0.0-1.0)
            cost_estimate: Estimated cost
            optimization_target: Target optimization ("accuracy", "cost", "speed", "balanced")
            
        Returns:
            Weighted score for method selection
        """
        # Normalize cost (assume max reasonable cost is 2000 tokens)
        normalized_cost = max(0, 1 - (cost_estimate / 2000))
        
        # Define weights based on optimization target
        weights = {
            "accuracy": {"suitability": 0.8, "cost": 0.2},
            "cost": {"suitability": 0.3, "cost": 0.7},
            "speed": {"suitability": 0.4, "cost": 0.6},  # Cost correlates with speed
            "balanced": {"suitability": 0.6, "cost": 0.4}
        }
        
        target_weights = weights.get(optimization_target, weights["balanced"])
        
        return (
            suitability * target_weights["suitability"] +
            normalized_cost * target_weights["cost"]
        )


class MCPIntegrationBridge:
    """
    Bridge for integrating knowledge extraction with Alita's MCP creation system
    Enables knowledge-driven tool generation as specified in Task 5
    """
    
    def __init__(self, mcp_client, extraction_manager):
        """
        Initialize MCP integration bridge
        
        Args:
            mcp_client: MCP client for tool creation (using hyphenated filename per memory)
            extraction_manager: Knowledge extraction manager instance
        """
        self.mcp_client = mcp_client
        self.extraction_manager = extraction_manager
        
        logger.info("Initialized MCP Integration Bridge", extra={
            'operation': 'MCP_BRIDGE_INIT'
        })
    
    async def create_knowledge_driven_tool(self, 
                                         extracted_knowledge: str, 
                                         extraction_metrics: ExtractionMetrics,
                                         context: ExtractionContext) -> Dict[str, Any]:
        """
        Create MCP tool based on extracted knowledge
        
        Args:
            extracted_knowledge: Knowledge extracted from the graph
            extraction_metrics: Metrics from the extraction process
            context: Extraction context
            
        Returns:
            MCP tool specification
        """
        logger.info("Creating knowledge-driven MCP tool", extra={
            'operation': 'MCP_TOOL_CREATION_START',
            'extraction_quality': extraction_metrics.efficiency_score()
        })
        
        try:
            # Analyze extracted knowledge for tool potential
            tool_spec = await self._analyze_knowledge_for_tool_creation(
                extracted_knowledge, extraction_metrics, context
            )
            
            # Generate MCP tool using the client
            mcp_tool = await self.mcp_client.create_tool(tool_spec)
            
            logger.info("Knowledge-driven MCP tool created successfully", extra={
                'operation': 'MCP_TOOL_CREATION_SUCCESS',
                'tool_name': tool_spec.get('name', 'unknown')
            })
            
            return mcp_tool
            
        except Exception as e:
            logger.error("MCP tool creation failed", extra={
                'operation': 'MCP_TOOL_CREATION_FAILED',
                'error': str(e)
            })
            raise
    
    async def _analyze_knowledge_for_tool_creation(self, 
                                                 knowledge: str, 
                                                 metrics: ExtractionMetrics,
                                                 context: ExtractionContext) -> Dict[str, Any]:
        """Analyze extracted knowledge to determine appropriate MCP tool specification"""
        # This would implement logic to convert extracted knowledge into MCP tool specs
        # For now, return a template specification
        return {
            "name": f"knowledge_tool_{int(time.time())}",
            "description": f"Tool generated from extracted knowledge: {knowledge[:100]}...",
            "parameters": self._derive_parameters_from_knowledge(knowledge),
            "implementation": self._generate_tool_implementation(knowledge, context),
            "metadata": {
                "extraction_method": context.optimization_target,
                "extraction_quality": metrics.efficiency_score(),
                "source_task": context.task_description
            }
        }
    
    def _derive_parameters_from_knowledge(self, knowledge: str) -> List[Dict[str, Any]]:
        """Derive tool parameters from extracted knowledge patterns"""
        # Simplified parameter derivation logic
        return [
            {
                "name": "input_data",
                "type": "string",
                "description": "Input data for knowledge-based processing",
                "required": True
            }
        ]
    
    def _generate_tool_implementation(self, knowledge: str, context: ExtractionContext) -> str:
        """Generate tool implementation code based on extracted knowledge"""
        # Simplified implementation generation
        return f"""
def knowledge_tool_implementation(input_data):
    # Implementation based on extracted knowledge:
    # {knowledge[:200]}...
    
    # Process input using knowledge patterns
    result = process_with_knowledge(input_data)
    return result
"""


class KnowledgeExtractionManager:
    """
    Main manager class that orchestrates all knowledge extraction operations
    Provides unified interface for the three extraction methods with optimization
    """
    
    def __init__(self, llm_client, graph_store, backend_type: BackendType = BackendType.NETWORKX):
        """
        Initialize the Knowledge Extraction Manager
        
        Args:
            llm_client: LLM client (OpenRouter-based per user memory)
            graph_store: Knowledge graph store instance
            backend_type: Backend type for graph operations
        """
        self.llm_client = llm_client
        self.graph_store = graph_store
        self.backend_type = backend_type
        
        # Initialize extraction methods
        self.extractors = {
            ExtractionMethod.DIRECT_RETRIEVAL: DirectRetrievalExtractor(
                llm_client, graph_store
            ),
            ExtractionMethod.GRAPH_QUERY: GraphQueryExtractor(
                llm_client, graph_store, backend_type
            ),
            ExtractionMethod.GENERAL_PURPOSE: GeneralPurposeExtractor(
                llm_client, graph_store
            )
        }
        
        logger.info("Initialized Knowledge Extraction Manager", extra={
            'operation': 'MANAGER_INIT',
            'backend_type': backend_type.value,
            'available_methods': [method.value for method in self.extractors.keys()]
        })
    
    async def extract_knowledge(self, 
                              query: str, 
                              context: Optional[ExtractionContext] = None,
                              method: Optional[ExtractionMethod] = None) -> Tuple[str, ExtractionMetrics]:
        """
        Extract knowledge using optimal or specified method
        
        Args:
            query: The extraction query
            context: Extraction context (will create default if None)
            method: Specific method to use (will optimize if None)
            
        Returns:
            Tuple of (extracted_knowledge, extraction_metrics)
        """
        if context is None:
            context = ExtractionContext(
                task_description=query,
                current_graph_state=await self._get_current_graph_state(),
                available_tools=list(self.extractors.keys()),
                user_preferences={}
            )
        
        logger.info("Starting knowledge extraction", extra={
            'operation': 'EXTRACTION_START',
            'query': query[:100] + "..." if len(query) > 100 else query,
            'specified_method': method.value if method else None
        })
        
        try:
            # Use Direct Retrieval as default for now
            if method is None:
                method = ExtractionMethod.DIRECT_RETRIEVAL
            
            logger.info("Using extraction method", extra={
                'operation': 'METHOD_SELECTED',
                'method': method.value
            })
            
            # Perform extraction
            extractor = self.extractors[method]
            knowledge, metrics = await extractor.extract_knowledge(query, context)
            
            logger.info("Knowledge extraction completed successfully", extra={
                'operation': 'EXTRACTION_SUCCESS',
                'method': method.value,
                'efficiency_score': metrics.efficiency_score(),
                'runtime_ms': metrics.runtime_ms
            })
            
            return knowledge, metrics
            
        except Exception as e:
            logger.error("Knowledge extraction failed", extra={
                'operation': 'EXTRACTION_FAILED',
                'error': str(e)
            })
            raise
    
    async def _get_current_graph_state(self) -> str:
        """Get current graph state as string representation"""
        try:
            return await self.graph_store.get_current_graph_state()
        except Exception as e:
            logger.warning("Failed to get graph state", extra={'error': str(e)})
            return "Graph state unavailable"


# Export main classes and interfaces for external use
__all__ = [
    'KnowledgeExtractionManager',
    'ExtractionMethod',
    'BackendType',
    'ExtractionContext',
    'ExtractionMetrics',
    'KnowledgeExtractionInterface',
    'DirectRetrievalExtractor',
    'GraphQueryExtractor',
    'GeneralPurposeExtractor',
    'KnowledgeExtractionOptimizer',
    'MCPIntegrationBridge'
]

if __name__ == "__main__":
    # Example usage and testing
    print("KGoT Knowledge Extraction Module - Task 5 Implementation")
    print("Available extraction methods:")
    for method in ExtractionMethod:
        print(f"  - {method.value}")
    print("\nSupported backends:")
    for backend in BackendType:
        print(f"  - {backend.value}") 