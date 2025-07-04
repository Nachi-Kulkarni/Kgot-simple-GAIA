#!/usr/bin/env python3
"""
KGoT Advanced Query Processing Module

This module implements KGoT Section 1.3 functionality for optimal query processing
across multiple graph database backends (Neo4j, RDF4J, NetworkX).

Features:
- Graph query languages with Cypher for Neo4j and SPARQL for RDF4J
- General-purpose languages with Python scripts and NetworkX
- Direct Retrieval for broad contextual understanding
- Optimal query selection between different approaches
- Performance monitoring and caching
- Comprehensive error handling and fallback mechanisms

Author: Alita-KGoT Enhanced System
License: BSD-style license
"""

import logging
import re
import time
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Import existing KG interfaces if available
try:
    from ..knowledge_graph.kg_interface import KnowledgeGraphInterface
    from ..knowledge_graph.neo4j.main import KnowledgeGraph as Neo4jKG
    from ..knowledge_graph.rdf4j.main import KnowledgeGraph as RDF4jKG
    from ..knowledge_graph.networkX.main import KnowledgeGraph as NetworkXKG
except ImportError:
    # Fallback for development/testing
    logging.warning("Could not import existing KG interfaces, using abstract interfaces")


class QueryType(Enum):
    """Enumeration of different query types for optimal backend selection"""
    GRAPH_TRAVERSAL = "graph_traversal"
    PATTERN_MATCHING = "pattern_matching"
    AGGREGATION = "aggregation"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    SEMANTIC_SEARCH = "semantic_search"
    DIRECT_RETRIEVAL = "direct_retrieval"
    HYBRID = "hybrid"


class BackendType(Enum):
    """Enumeration of supported graph database backends"""
    NEO4J = "neo4j"
    RDF4J = "rdf4j"
    NETWORKX = "networkx"
    AUTO = "auto"


@dataclass
class QueryAnalysisResult:
    """
    Result of query analysis containing type classification and recommendations
    
    Attributes:
        query_type: Detected type of the query
        complexity_score: Complexity rating (1-10)
        recommended_backend: Suggested backend for execution
        confidence: Confidence in the recommendation (0-1)
        features: List of detected query features
        estimated_cost: Estimated execution cost
        requires_translation: Whether query needs translation
    """
    query_type: QueryType
    complexity_score: int
    recommended_backend: BackendType
    confidence: float
    features: List[str]
    estimated_cost: float
    requires_translation: bool = False
    fallback_backends: List[BackendType] = None

    def __post_init__(self):
        if self.fallback_backends is None:
            self.fallback_backends = []


@dataclass
class QueryExecutionResult:
    """
    Result of query execution with performance metrics
    
    Attributes:
        success: Whether execution was successful
        result: Query execution result
        execution_time: Time taken for execution (seconds)
        backend_used: Backend that executed the query
        error: Error message if execution failed
        metadata: Additional execution metadata
    """
    success: bool
    result: Any
    execution_time: float
    backend_used: BackendType
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdvancedQueryProcessor:
    """
    Advanced Query Processing System for KGoT
    
    Implements KGoT Section 1.3 functionality including:
    - Graph query languages (Cypher, SPARQL)
    - General-purpose languages (Python/NetworkX)
    - Direct Retrieval for broad contextual understanding
    - Optimal query selection and execution
    """

    def __init__(self, 
                 neo4j_config: Optional[Dict] = None,
                 rdf4j_config: Optional[Dict] = None,
                 enable_caching: bool = True,
                 cache_size: int = 1000,
                 performance_monitoring: bool = True,
                 logger_name: str = "AdvancedQueryProcessor"):
        """
        Initialize the Advanced Query Processor
        
        Args:
            neo4j_config: Configuration for Neo4j connection
            rdf4j_config: Configuration for RDF4J connection
            enable_caching: Whether to enable query result caching
            cache_size: Maximum number of cached results
            performance_monitoring: Whether to collect performance metrics
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.info("Initializing Advanced Query Processor")
        
        # Initialize backend configurations
        self.neo4j_config = neo4j_config or {}
        self.rdf4j_config = rdf4j_config or {}
        
        # Initialize backend instances
        self.backends: Dict[BackendType, Any] = {}
        self._initialize_backends()
        
        # Initialize caching system
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.query_cache: Dict[str, QueryExecutionResult] = {}
        self.cache_lock = threading.Lock()
        
        # Initialize performance monitoring
        self.performance_monitoring = performance_monitoring
        self.performance_metrics: Dict[str, List[float]] = {
            'neo4j_times': [],
            'rdf4j_times': [],
            'networkx_times': [],
            'translation_times': [],
            'analysis_times': []
        }
        
        # Initialize component instances
        self.query_analyzer = QueryAnalyzer(self.logger)
        self.query_translator = QueryTranslator(self.logger)
        self.backend_selector = BackendSelector(self.logger, self.performance_metrics)
        self.direct_retriever = DirectRetriever(self.logger, self.backends)
        
        self.logger.info("Advanced Query Processor initialized successfully")

    def _initialize_backends(self) -> None:
        """Initialize available graph database backends"""
        try:
            # Initialize Neo4j backend
            if self.neo4j_config:
                self.backends[BackendType.NEO4J] = Neo4jKG(
                    neo4j_uri=self.neo4j_config.get('uri', 'bolt://localhost:7687'),
                    neo4j_user=self.neo4j_config.get('user', 'neo4j'),
                    neo4j_pwd=self.neo4j_config.get('password', 'password')
                )
                self.logger.info("Neo4j backend initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Neo4j backend: {e}")

        try:
            # Initialize RDF4J backend
            if self.rdf4j_config:
                self.backends[BackendType.RDF4J] = RDF4jKG(
                    rdf4j_read_endpoint=self.rdf4j_config.get('read_endpoint'),
                    rdf4j_write_endpoint=self.rdf4j_config.get('write_endpoint')
                )
                self.logger.info("RDF4J backend initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize RDF4J backend: {e}")

        try:
            # Initialize NetworkX backend (always available)
            self.backends[BackendType.NETWORKX] = NetworkXKG()
            self.logger.info("NetworkX backend initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize NetworkX backend: {e}")

    def process_query(self, 
                     query: str, 
                     preferred_backend: BackendType = BackendType.AUTO,
                     enable_fallback: bool = True,
                     context: Optional[Dict] = None) -> QueryExecutionResult:
        """
        Process a query using optimal backend selection and execution
        
        Args:
            query: The query to process
            preferred_backend: Preferred backend (AUTO for optimal selection)
            enable_fallback: Whether to use fallback backends on failure
            context: Additional context for query processing
            
        Returns:
            QueryExecutionResult containing execution results and metadata
        """
        start_time = time.time()
        context = context or {}
        
        self.logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Check cache first
            if self.enable_caching:
                cached_result = self._get_cached_result(query, preferred_backend)
                if cached_result:
                    self.logger.info("Returning cached result")
                    return cached_result
            
            # Analyze query to determine optimal approach
            analysis_start = time.time()
            analysis_result = self.query_analyzer.analyze_query(query, context)
            analysis_time = time.time() - analysis_start
            
            if self.performance_monitoring:
                self.performance_metrics['analysis_times'].append(analysis_time)
            
            self.logger.info(f"Query analysis completed: type={analysis_result.query_type}, "
                           f"complexity={analysis_result.complexity_score}, "
                           f"recommended_backend={analysis_result.recommended_backend}")
            
            # Select backend based on preference and analysis
            selected_backend = (preferred_backend if preferred_backend != BackendType.AUTO 
                              else analysis_result.recommended_backend)
            
            # Execute query with selected backend
            result = self._execute_with_backend(query, selected_backend, analysis_result, context)
            
            # Handle fallback if execution failed and fallback is enabled
            if not result.success and enable_fallback:
                result = self._handle_fallback_execution(query, selected_backend, 
                                                       analysis_result, context)
            
            # Cache successful results
            if result.success and self.enable_caching:
                self._cache_result(query, preferred_backend, result)
            
            # Record performance metrics
            total_time = time.time() - start_time
            result.metadata['total_processing_time'] = total_time
            result.metadata['analysis_result'] = analysis_result
            
            self.logger.info(f"Query processing completed in {total_time:.3f}s, "
                           f"success={result.success}, backend={result.backend_used}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return QueryExecutionResult(
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                backend_used=BackendType.AUTO,
                error=str(e),
                metadata={'error_type': type(e).__name__}
            )

    def _execute_with_backend(self, query: str, backend_type: BackendType, 
                            analysis_result: QueryAnalysisResult, 
                            context: Dict) -> QueryExecutionResult:
        """
        Execute query with specified backend
        
        Args:
            query: Query to execute
            backend_type: Selected backend
            analysis_result: Query analysis results
            context: Additional context
            
        Returns:
            Query execution result
        """
        start_time = time.time()
        
        # Check if backend is available
        if backend_type not in self.backends:
            return QueryExecutionResult(
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                backend_used=backend_type,
                error=f"Backend {backend_type} not available"
            )
        
        backend_instance = self.backends[backend_type]
        
        try:
            # Translate query if needed
            final_query = query
            if analysis_result.requires_translation:
                source_language, _ = self.query_analyzer._detect_query_language(query)
                translated_query, translation_success = self.query_translator.translate_query(
                    query, source_language, backend_type
                )
                if translation_success:
                    final_query = translated_query
                    self.logger.debug(f"Query translated for {backend_type}")
                else:
                    self.logger.warning(f"Query translation failed for {backend_type}")
            
            # Execute query
            if backend_type == BackendType.NETWORKX:
                result, success, error = backend_instance.get_query(final_query)
            else:
                result, success, error = backend_instance.get_query(final_query)
            
            execution_time = time.time() - start_time
            
            # Record performance metrics
            if self.performance_monitoring:
                metric_key = f"{backend_type.value}_times"
                self.performance_metrics[metric_key].append(execution_time)
                # Keep only last 100 measurements
                if len(self.performance_metrics[metric_key]) > 100:
                    self.performance_metrics[metric_key] = self.performance_metrics[metric_key][-100:]
            
            return QueryExecutionResult(
                success=success,
                result=result,
                execution_time=execution_time,
                backend_used=backend_type,
                error=str(error) if error else None,
                metadata={
                    'translated': analysis_result.requires_translation,
                    'original_query': query,
                    'final_query': final_query
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error executing query on {backend_type}: {e}")
            
            return QueryExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                backend_used=backend_type,
                error=str(e),
                metadata={'exception_type': type(e).__name__}
            )

    def _handle_fallback_execution(self, query: str, failed_backend: BackendType,
                                 analysis_result: QueryAnalysisResult, 
                                 context: Dict) -> QueryExecutionResult:
        """
        Handle fallback execution when primary backend fails
        
        Args:
            query: Original query
            failed_backend: Backend that failed
            analysis_result: Query analysis results
            context: Additional context
            
        Returns:
            Result from fallback execution
        """
        self.logger.info(f"Attempting fallback execution after {failed_backend} failure")
        
        # Try fallback backends in order
        for fallback_backend in analysis_result.fallback_backends:
            if fallback_backend in self.backends and fallback_backend != failed_backend:
                self.logger.info(f"Trying fallback backend: {fallback_backend}")
                
                result = self._execute_with_backend(query, fallback_backend, analysis_result, context)
                
                if result.success:
                    self.logger.info(f"Fallback execution successful with {fallback_backend}")
                    result.metadata['fallback_used'] = True
                    result.metadata['original_backend'] = failed_backend
                    return result
                else:
                    self.logger.warning(f"Fallback backend {fallback_backend} also failed")
        
        # If all fallbacks failed, try direct retrieval as last resort
        self.logger.info("All backends failed, attempting direct retrieval")
        return self.direct_retriever.perform_direct_retrieval(query, context)

    def _get_cached_result(self, query: str, backend_type: BackendType) -> Optional[QueryExecutionResult]:
        """Get cached result if available"""
        if not self.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(query, backend_type)
        with self.cache_lock:
            return self.query_cache.get(cache_key)

    def _cache_result(self, query: str, backend_type: BackendType, result: QueryExecutionResult):
        """Cache query result"""
        if not self.enable_caching:
            return
        
        cache_key = self._generate_cache_key(query, backend_type)
        with self.cache_lock:
            # Implement LRU eviction
            if len(self.query_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
            
            self.query_cache[cache_key] = result

    def _generate_cache_key(self, query: str, backend_type: BackendType) -> str:
        """Generate cache key for query and backend combination"""
        content = f"{query}:{backend_type.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for all backends
        
        Returns:
            Dictionary containing performance statistics
        """
        stats = {}
        
        for metric_name, times in self.performance_metrics.items():
            if times:
                stats[metric_name] = {
                    'count': len(times),
                    'average': statistics.mean(times),
                    'median': statistics.median(times),
                    'min': min(times),
                    'max': max(times),
                    'std_dev': statistics.stdev(times) if len(times) > 1 else 0
                }
            else:
                stats[metric_name] = {'count': 0}
        
        return stats

    def clear_cache(self):
        """Clear the query cache"""
        with self.cache_lock:
            self.query_cache.clear()
        self.logger.info("Query cache cleared")

    def get_available_backends(self) -> List[BackendType]:
        """Get list of available backends"""
        return list(self.backends.keys())


class QueryAnalyzer:
    """
    Analyzes queries to determine type, complexity, and optimal backend selection
    
    This class implements sophisticated query analysis to classify queries and
    recommend the most appropriate backend for execution based on query patterns,
    complexity, and performance characteristics.
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize the Query Analyzer
        
        Args:
            logger: Logger instance for logging analysis operations
        """
        self.logger = logger
        
        # Define query pattern matching rules
        self.cypher_patterns = [
            r'MATCH\s+\(',           # Cypher MATCH clause
            r'CREATE\s+\(',          # Cypher CREATE clause
            r'MERGE\s+\(',           # Cypher MERGE clause
            r'WITH\s+',              # Cypher WITH clause
            r'RETURN\s+',            # Cypher RETURN clause
            r'WHERE\s+',             # Cypher WHERE clause
            r'\]->\(',               # Cypher relationship pattern
            r'\]-\[',                # Cypher relationship pattern
        ]
        
        self.sparql_patterns = [
            r'SELECT\s+\?',          # SPARQL SELECT query
            r'CONSTRUCT\s+{',        # SPARQL CONSTRUCT query
            r'ASK\s+{',              # SPARQL ASK query
            r'DESCRIBE\s+',          # SPARQL DESCRIBE query
            r'INSERT\s+DATA\s+{',    # SPARQL INSERT
            r'DELETE\s+DATA\s+{',    # SPARQL DELETE
            r'PREFIX\s+\w+:',        # SPARQL PREFIX
            r'WHERE\s+{',            # SPARQL WHERE clause
        ]
        
        self.networkx_patterns = [
            r'nx\.',                 # NetworkX function calls
            r'self\.G\.',            # NetworkX graph reference
            r'nodes\(',              # NetworkX nodes method
            r'edges\(',              # NetworkX edges method
            r'add_node\(',           # NetworkX add_node method
            r'add_edge\(',           # NetworkX add_edge method
            r'shortest_path\(',      # NetworkX algorithms
            r'connected_components\(',# NetworkX algorithms
        ]
        
        # Define complexity indicators
        self.complexity_indicators = {
            'high': [
                r'OPTIONAL\s+MATCH',     # Complex Cypher operations
                r'COLLECT\s*\(',         # Aggregation functions
                r'UNION\s+',             # Set operations
                r'WITH\s+.*ORDER\s+BY',  # Sorting operations
                r'shortest_path\(',      # Path algorithms
                r'allShortestPaths\(',   # Path algorithms
                r'FILTER\s*\(',          # Complex filtering
                r'GROUP\s+BY',           # Grouping operations
            ],
            'medium': [
                r'WHERE\s+.*AND\s+.*OR', # Multiple conditions
                r'LIMIT\s+\d+',          # Result limiting
                r'ORDER\s+BY',           # Ordering
                r'COUNT\s*\(',           # Simple aggregation
                r'SUM\s*\(',             # Simple aggregation
            ],
            'low': [
                r'MATCH\s+\(\w+\)',      # Simple node matching
                r'RETURN\s+\w+',         # Simple return
                r'CREATE\s+\(\w+\)',     # Simple creation
            ]
        }

    def analyze_query(self, query: str, context: Dict = None) -> QueryAnalysisResult:
        """
        Analyze a query to determine its characteristics and optimal backend
        
        Args:
            query: The query string to analyze
            context: Additional context information
            
        Returns:
            QueryAnalysisResult containing analysis results and recommendations
        """
        self.logger.debug(f"Analyzing query: {query[:100]}...")
        context = context or {}
        
        # Detect query language and type
        query_type, language_confidence = self._detect_query_language(query)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(query)
        
        # Determine query type based on patterns
        semantic_type = self._classify_query_type(query)
        
        # Select optimal backend
        recommended_backend, confidence = self._recommend_backend(
            query, query_type, complexity_score, semantic_type, context
        )
        
        # Detect query features
        features = self._extract_query_features(query)
        
        # Estimate execution cost
        estimated_cost = self._estimate_execution_cost(query, complexity_score, recommended_backend)
        
        # Determine if translation is needed
        requires_translation = self._requires_translation(query, recommended_backend)
        
        # Generate fallback options
        fallback_backends = self._generate_fallback_options(recommended_backend, query_type)
        
        result = QueryAnalysisResult(
            query_type=semantic_type,
            complexity_score=complexity_score,
            recommended_backend=recommended_backend,
            confidence=confidence,
            features=features,
            estimated_cost=estimated_cost,
            requires_translation=requires_translation,
            fallback_backends=fallback_backends
        )
        
        self.logger.debug(f"Query analysis complete: {result}")
        return result

    def _detect_query_language(self, query: str) -> Tuple[str, float]:
        """
        Detect the language of the query (Cypher, SPARQL, or Python)
        
        Args:
            query: Query string to analyze
            
        Returns:
            Tuple of (detected_language, confidence_score)
        """
        query_upper = query.upper()
        
        # Count pattern matches for each language
        cypher_matches = sum(1 for pattern in self.cypher_patterns 
                           if re.search(pattern, query_upper, re.IGNORECASE))
        sparql_matches = sum(1 for pattern in self.sparql_patterns 
                           if re.search(pattern, query_upper, re.IGNORECASE))
        networkx_matches = sum(1 for pattern in self.networkx_patterns 
                             if re.search(pattern, query, re.IGNORECASE))
        
        total_matches = cypher_matches + sparql_matches + networkx_matches
        
        if total_matches == 0:
            return "unknown", 0.0
        
        # Determine most likely language
        if cypher_matches >= sparql_matches and cypher_matches >= networkx_matches:
            confidence = cypher_matches / total_matches
            return "cypher", confidence
        elif sparql_matches >= networkx_matches:
            confidence = sparql_matches / total_matches
            return "sparql", confidence
        else:
            confidence = networkx_matches / total_matches
            return "networkx", confidence

    def _calculate_complexity(self, query: str) -> int:
        """
        Calculate query complexity score (1-10)
        
        Args:
            query: Query string to analyze
            
        Returns:
            Complexity score from 1 (simple) to 10 (very complex)
        """
        complexity_score = 1
        
        # Check for high complexity indicators
        for pattern in self.complexity_indicators['high']:
            if re.search(pattern, query, re.IGNORECASE):
                complexity_score += 3
        
        # Check for medium complexity indicators
        for pattern in self.complexity_indicators['medium']:
            if re.search(pattern, query, re.IGNORECASE):
                complexity_score += 2
        
        # Check for low complexity indicators
        for pattern in self.complexity_indicators['low']:
            if re.search(pattern, query, re.IGNORECASE):
                complexity_score += 1
        
        # Additional complexity factors
        if len(query) > 500:
            complexity_score += 1
        if query.count('(') > 5:  # Many nested operations
            complexity_score += 1
        if len(re.findall(r'\bJOIN\b|\bUNION\b|\bMERGE\b', query, re.IGNORECASE)) > 0:
            complexity_score += 2
        
        return min(complexity_score, 10)

    def _classify_query_type(self, query: str) -> QueryType:
        """
        Classify the semantic type of the query
        
        Args:
            query: Query string to classify
            
        Returns:
            QueryType enum value
        """
        query_upper = query.upper()
        
        # Pattern matching for different query types
        if re.search(r'PATH|SHORTEST|TRAVERSE|WALK', query_upper):
            return QueryType.GRAPH_TRAVERSAL
        elif re.search(r'MATCH.*WHERE|FILTER|PATTERN', query_upper):
            return QueryType.PATTERN_MATCHING
        elif re.search(r'COUNT|SUM|AVG|MIN|MAX|COLLECT|GROUP', query_upper):
            return QueryType.AGGREGATION
        elif re.search(r'DEGREE|CENTRALITY|CLUSTER|COMPONENT', query_upper):
            return QueryType.STRUCTURAL_ANALYSIS
        elif re.search(r'SEARCH|FIND|CONTAINS|LIKE|SIMILAR', query_upper):
            return QueryType.SEMANTIC_SEARCH
        elif re.search(r'ALL|EVERYTHING|COMPLETE|FULL', query_upper):
            return QueryType.DIRECT_RETRIEVAL
        else:
            return QueryType.PATTERN_MATCHING  # Default type

    def _recommend_backend(self, query: str, language: str, complexity: int, 
                          query_type: QueryType, context: Dict) -> Tuple[BackendType, float]:
        """
        Recommend the optimal backend for query execution
        
        Args:
            query: Query string
            language: Detected query language
            complexity: Complexity score
            query_type: Semantic query type
            context: Additional context
            
        Returns:
            Tuple of (recommended_backend, confidence)
        """
        # Language-based recommendations
        if language == "cypher":
            return BackendType.NEO4J, 0.9
        elif language == "sparql":
            return BackendType.RDF4J, 0.9
        elif language == "networkx":
            return BackendType.NETWORKX, 0.9
        
        # Type-based recommendations for unknown language
        if query_type == QueryType.GRAPH_TRAVERSAL:
            if complexity > 7:
                return BackendType.NEO4J, 0.8  # Neo4j excels at complex traversals
            else:
                return BackendType.NETWORKX, 0.7  # NetworkX good for simple traversals
        elif query_type == QueryType.SEMANTIC_SEARCH:
            return BackendType.RDF4J, 0.8  # RDF4J good for semantic queries
        elif query_type == QueryType.STRUCTURAL_ANALYSIS:
            return BackendType.NETWORKX, 0.8  # NetworkX has many analysis algorithms
        elif query_type == QueryType.AGGREGATION:
            return BackendType.NEO4J, 0.7  # Neo4j good for aggregations
        else:
            return BackendType.NETWORKX, 0.6  # Default fallback

    def _extract_query_features(self, query: str) -> List[str]:
        """Extract features from the query for analysis"""
        features = []
        
        if re.search(r'MATCH', query, re.IGNORECASE):
            features.append('pattern_matching')
        if re.search(r'CREATE|INSERT', query, re.IGNORECASE):
            features.append('data_modification')
        if re.search(r'DELETE|REMOVE', query, re.IGNORECASE):
            features.append('data_deletion')
        if re.search(r'ORDER\s+BY|LIMIT', query, re.IGNORECASE):
            features.append('result_ordering')
        if re.search(r'COUNT|SUM|AVG', query, re.IGNORECASE):
            features.append('aggregation')
        if re.search(r'WHERE.*AND.*OR', query, re.IGNORECASE):
            features.append('complex_filtering')
        
        return features

    def _estimate_execution_cost(self, query: str, complexity: int, backend: BackendType) -> float:
        """Estimate the execution cost of the query"""
        base_cost = complexity * 0.1
        
        # Backend-specific cost adjustments
        if backend == BackendType.NEO4J:
            base_cost *= 1.2  # Slightly higher cost for Neo4j operations
        elif backend == BackendType.RDF4J:
            base_cost *= 1.3  # Higher cost for SPARQL operations
        elif backend == BackendType.NETWORKX:
            base_cost *= 1.0  # Base cost for NetworkX
        
        return base_cost

    def _requires_translation(self, query: str, backend: BackendType) -> bool:
        """Determine if query needs translation for the target backend"""
        language, _ = self._detect_query_language(query)
        
        if language == "cypher" and backend != BackendType.NEO4J:
            return True
        elif language == "sparql" and backend != BackendType.RDF4J:
            return True
        elif language == "networkx" and backend != BackendType.NETWORKX:
            return True
        
        return False

    def _generate_fallback_options(self, primary_backend: BackendType, 
                                 query_type: QueryType) -> List[BackendType]:
        """Generate fallback backend options"""
        all_backends = [BackendType.NEO4J, BackendType.RDF4J, BackendType.NETWORKX]
        fallbacks = [b for b in all_backends if b != primary_backend]
        
        # Prioritize fallbacks based on query type
        if query_type == QueryType.GRAPH_TRAVERSAL:
            fallbacks.sort(key=lambda x: 0 if x == BackendType.NEO4J else 1)
        elif query_type == QueryType.SEMANTIC_SEARCH:
            fallbacks.sort(key=lambda x: 0 if x == BackendType.RDF4J else 1)
        elif query_type == QueryType.STRUCTURAL_ANALYSIS:
            fallbacks.sort(key=lambda x: 0 if x == BackendType.NETWORKX else 1)
        
        return fallbacks[:2]  # Return top 2 fallback options 


class QueryTranslator:
    """
    Translates queries between different graph query languages
    
    Supports translation between:
    - Cypher (Neo4j)
    - SPARQL (RDF4J)
    - Python/NetworkX code
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize the Query Translator
        
        Args:
            logger: Logger instance for logging translation operations
        """
        self.logger = logger
        
        # Define translation patterns and templates
        self.cypher_to_sparql_mappings = {
            'MATCH': 'SELECT',
            'RETURN': 'WHERE',
            'WHERE': 'FILTER',
            'CREATE': 'INSERT DATA',
            'DELETE': 'DELETE DATA'
        }
        
        self.sparql_to_cypher_mappings = {v: k for k, v in self.cypher_to_sparql_mappings.items()}

    def translate_query(self, query: str, source_language: str, 
                       target_backend: BackendType) -> Tuple[str, bool]:
        """
        Translate query from source language to target backend language
        
        Args:
            query: Original query string
            source_language: Source query language (cypher, sparql, networkx)
            target_backend: Target backend type
            
        Returns:
            Tuple of (translated_query, success_flag)
        """
        self.logger.debug(f"Translating {source_language} query to {target_backend}")
        
        try:
            if source_language == "cypher" and target_backend == BackendType.RDF4J:
                return self._cypher_to_sparql(query)
            elif source_language == "cypher" and target_backend == BackendType.NETWORKX:
                return self._cypher_to_networkx(query)
            elif source_language == "sparql" and target_backend == BackendType.NEO4J:
                return self._sparql_to_cypher(query)
            elif source_language == "sparql" and target_backend == BackendType.NETWORKX:
                return self._sparql_to_networkx(query)
            elif source_language == "networkx" and target_backend == BackendType.NEO4J:
                return self._networkx_to_cypher(query)
            elif source_language == "networkx" and target_backend == BackendType.RDF4J:
                return self._networkx_to_sparql(query)
            else:
                # No translation needed or unsupported
                return query, True
                
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return query, False

    def _cypher_to_sparql(self, cypher_query: str) -> Tuple[str, bool]:
        """
        Convert Cypher query to SPARQL
        
        Args:
            cypher_query: Cypher query string
            
        Returns:
            Tuple of (sparql_query, success_flag)
        """
        try:
            # Basic Cypher to SPARQL translation
            sparql_query = cypher_query
            
            # Handle simple MATCH patterns
            match_pattern = r'MATCH\s*\((\w+):(\w+)\)'
            sparql_query = re.sub(match_pattern, 
                                r'SELECT ?s WHERE { ?s rdf:type :\2 }', 
                                sparql_query, flags=re.IGNORECASE)
            
            # Handle RETURN clauses
            sparql_query = re.sub(r'RETURN\s+(\w+)', 
                                r'SELECT ?\1 WHERE { ?\1 ?p ?o }', 
                                sparql_query, flags=re.IGNORECASE)
            
            # Add necessary prefixes
            prefixes = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            """
            sparql_query = prefixes + sparql_query
            
            self.logger.debug(f"Translated Cypher to SPARQL: {sparql_query}")
            return sparql_query, True
            
        except Exception as e:
            self.logger.error(f"Cypher to SPARQL translation failed: {e}")
            return cypher_query, False

    def _cypher_to_networkx(self, cypher_query: str) -> Tuple[str, bool]:
        """
        Convert Cypher query to NetworkX Python code
        
        Args:
            cypher_query: Cypher query string
            
        Returns:
            Tuple of (networkx_code, success_flag)
        """
        try:
            networkx_code = "# Translated from Cypher\n"
            
            # Handle MATCH patterns for nodes
            if re.search(r'MATCH\s*\((\w+):(\w+)\)', cypher_query, re.IGNORECASE):
                match = re.search(r'MATCH\s*\((\w+):(\w+)\)', cypher_query, re.IGNORECASE)
                var_name = match.group(1)
                label = match.group(2)
                networkx_code += f"""
result = []
for node in self.G.nodes(data=True):
    if node[1].get('label') == '{label}':
        result.append(node)
"""
            
            # Handle simple RETURN statements
            if re.search(r'RETURN\s+(\w+)', cypher_query, re.IGNORECASE):
                networkx_code += "\n# Return result as requested\n"
            
            # Handle relationships
            if re.search(r'\[(\w+):(\w+)\]', cypher_query, re.IGNORECASE):
                match = re.search(r'\[(\w+):(\w+)\]', cypher_query, re.IGNORECASE)
                rel_var = match.group(1)
                rel_type = match.group(2)
                networkx_code += f"""
# Filter relationships by type: {rel_type}
filtered_edges = []
for edge in self.G.edges(data=True):
    if edge[2].get('relationship') == '{rel_type}':
        filtered_edges.append(edge)
result = filtered_edges
"""
            
            # Default result if no specific pattern matched
            if "result = " not in networkx_code:
                networkx_code += "\nresult = list(self.G.nodes())\n"
            
            self.logger.debug(f"Translated Cypher to NetworkX: {networkx_code}")
            return networkx_code, True
            
        except Exception as e:
            self.logger.error(f"Cypher to NetworkX translation failed: {e}")
            return cypher_query, False

    def _sparql_to_cypher(self, sparql_query: str) -> Tuple[str, bool]:
        """
        Convert SPARQL query to Cypher
        
        Args:
            sparql_query: SPARQL query string
            
        Returns:
            Tuple of (cypher_query, success_flag)
        """
        try:
            cypher_query = sparql_query
            
            # Handle SELECT patterns
            select_pattern = r'SELECT\s+\?(\w+)\s+WHERE\s*\{\s*\?(\w+)\s+rdf:type\s+:(\w+)'
            cypher_query = re.sub(select_pattern, 
                                r'MATCH (\1:\3) RETURN \1', 
                                cypher_query, flags=re.IGNORECASE)
            
            # Handle basic triple patterns
            triple_pattern = r'\?(\w+)\s+\?(\w+)\s+\?(\w+)'
            cypher_query = re.sub(triple_pattern, 
                                r'MATCH (\1)-[:\2]->(\3)', 
                                cypher_query, flags=re.IGNORECASE)
            
            # Remove SPARQL prefixes and formatting
            cypher_query = re.sub(r'PREFIX\s+\w+:\s*<[^>]+>', '', cypher_query, flags=re.IGNORECASE)
            cypher_query = cypher_query.strip()
            
            self.logger.debug(f"Translated SPARQL to Cypher: {cypher_query}")
            return cypher_query, True
            
        except Exception as e:
            self.logger.error(f"SPARQL to Cypher translation failed: {e}")
            return sparql_query, False

    def _sparql_to_networkx(self, sparql_query: str) -> Tuple[str, bool]:
        """
        Convert SPARQL query to NetworkX Python code
        
        Args:
            sparql_query: SPARQL query string
            
        Returns:
            Tuple of (networkx_code, success_flag)
        """
        try:
            networkx_code = "# Translated from SPARQL\n"
            
            # Handle SELECT queries
            if re.search(r'SELECT\s+\?(\w+)', sparql_query, re.IGNORECASE):
                match = re.search(r'SELECT\s+\?(\w+)', sparql_query, re.IGNORECASE)
                var_name = match.group(1)
                networkx_code += f"""
result = []
for node in self.G.nodes(data=True):
    result.append({{'{var_name}': node[0], 'data': node[1]}})
"""
            
            # Handle type patterns
            if re.search(r'rdf:type\s+:(\w+)', sparql_query, re.IGNORECASE):
                match = re.search(r'rdf:type\s+:(\w+)', sparql_query, re.IGNORECASE)
                type_name = match.group(1)
                networkx_code += f"""
# Filter by type: {type_name}
filtered_result = []
for item in result:
    if item.get('data', {{}}).get('type') == '{type_name}':
        filtered_result.append(item)
result = filtered_result
"""
            
            # Default result if no specific pattern matched
            if "result = " not in networkx_code:
                networkx_code += "\nresult = list(self.G.nodes(data=True))\n"
            
            self.logger.debug(f"Translated SPARQL to NetworkX: {networkx_code}")
            return networkx_code, True
            
        except Exception as e:
            self.logger.error(f"SPARQL to NetworkX translation failed: {e}")
            return sparql_query, False

    def _networkx_to_cypher(self, networkx_code: str) -> Tuple[str, bool]:
        """
        Convert NetworkX Python code to Cypher (basic patterns only)
        
        Args:
            networkx_code: NetworkX Python code
            
        Returns:
            Tuple of (cypher_query, success_flag)
        """
        try:
            cypher_query = ""
            
            # Handle node queries
            if "self.G.nodes()" in networkx_code:
                cypher_query = "MATCH (n) RETURN n"
            elif "self.G.edges()" in networkx_code:
                cypher_query = "MATCH (a)-[r]->(b) RETURN a, r, b"
            elif "add_node" in networkx_code:
                # Extract node creation patterns
                node_pattern = r'add_node\([\'"](\w+)[\'"]'
                match = re.search(node_pattern, networkx_code)
                if match:
                    node_name = match.group(1)
                    cypher_query = f"CREATE (n:{node_name}) RETURN n"
            elif "add_edge" in networkx_code:
                # Extract edge creation patterns
                edge_pattern = r'add_edge\([\'"](\w+)[\'"],\s*[\'"](\w+)[\'"]'
                match = re.search(edge_pattern, networkx_code)
                if match:
                    from_node = match.group(1)
                    to_node = match.group(2)
                    cypher_query = f"MATCH (a), (b) WHERE a.name = '{from_node}' AND b.name = '{to_node}' CREATE (a)-[r:CONNECTED]->(b) RETURN r"
            
            # Default fallback
            if not cypher_query:
                cypher_query = "MATCH (n) RETURN n LIMIT 100"
            
            self.logger.debug(f"Translated NetworkX to Cypher: {cypher_query}")
            return cypher_query, True
            
        except Exception as e:
            self.logger.error(f"NetworkX to Cypher translation failed: {e}")
            return networkx_code, False

    def _networkx_to_sparql(self, networkx_code: str) -> Tuple[str, bool]:
        """
        Convert NetworkX Python code to SPARQL (basic patterns only)
        
        Args:
            networkx_code: NetworkX Python code
            
        Returns:
            Tuple of (sparql_query, success_flag)
        """
        try:
            sparql_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
            
            # Handle node queries
            if "self.G.nodes()" in networkx_code:
                sparql_query += "SELECT ?s WHERE { ?s ?p ?o }"
            elif "self.G.edges()" in networkx_code:
                sparql_query += "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
            elif "add_node" in networkx_code:
                # Extract node creation patterns
                node_pattern = r'add_node\([\'"](\w+)[\'"]'
                match = re.search(node_pattern, networkx_code)
                if match:
                    node_name = match.group(1)
                    sparql_query += f"INSERT DATA {{ :{node_name} rdf:type :Node }}"
            
            # Default fallback
            if "SELECT" not in sparql_query and "INSERT" not in sparql_query:
                sparql_query += "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100"
            
            self.logger.debug(f"Translated NetworkX to SPARQL: {sparql_query}")
            return sparql_query, True
            
        except Exception as e:
            self.logger.error(f"NetworkX to SPARQL translation failed: {e}")
            return networkx_code, False 


class BackendSelector:
    """
    Selects optimal backend based on performance metrics and query characteristics
    
    This class monitors performance across different backends and makes intelligent
    decisions about which backend to use for optimal query execution.
    """

    def __init__(self, logger: logging.Logger, performance_metrics: Dict[str, List[float]]):
        """
        Initialize the Backend Selector
        
        Args:
            logger: Logger instance for logging selection decisions
            performance_metrics: Dictionary containing performance metrics for each backend
        """
        self.logger = logger
        self.performance_metrics = performance_metrics

    def select_optimal_backend(self, analysis_result: QueryAnalysisResult, 
                             available_backends: List[BackendType],
                             context: Dict = None) -> BackendType:
        """
        Select the optimal backend based on analysis results and performance history
        
        Args:
            analysis_result: Query analysis results
            available_backends: List of available backends
            context: Additional context for selection
            
        Returns:
            Selected backend type
        """
        context = context or {}
        
        # Start with the recommended backend from analysis
        recommended = analysis_result.recommended_backend
        
        # Check if recommended backend is available
        if recommended in available_backends:
            # Check performance history
            avg_performance = self._get_average_performance(recommended)
            
            # If performance is good or no history available, use recommended
            if avg_performance is None or avg_performance < 2.0:  # Less than 2 seconds average
                self.logger.debug(f"Selected recommended backend: {recommended}")
                return recommended
        
        # Fallback to performance-based selection
        best_backend = self._select_by_performance(available_backends, analysis_result)
        self.logger.debug(f"Selected performance-optimized backend: {best_backend}")
        return best_backend

    def _get_average_performance(self, backend: BackendType) -> Optional[float]:
        """Get average performance for a backend"""
        metric_key = f"{backend.value}_times"
        if metric_key in self.performance_metrics and self.performance_metrics[metric_key]:
            return statistics.mean(self.performance_metrics[metric_key][-10:])  # Last 10 executions
        return None

    def _select_by_performance(self, available_backends: List[BackendType], 
                             analysis_result: QueryAnalysisResult) -> BackendType:
        """Select backend based on performance metrics"""
        best_backend = available_backends[0]  # Default
        best_performance = float('inf')
        
        for backend in available_backends:
            avg_perf = self._get_average_performance(backend)
            if avg_perf is not None and avg_perf < best_performance:
                best_performance = avg_perf
                best_backend = backend
        
        return best_backend


class DirectRetriever:
    """
    Implements Direct Retrieval for broad contextual understanding
    
    This class provides comprehensive data retrieval by combining results from
    multiple backends and implementing advanced retrieval strategies.
    """

    def __init__(self, logger: logging.Logger, backends: Dict[BackendType, Any]):
        """
        Initialize the Direct Retriever
        
        Args:
            logger: Logger instance for logging retrieval operations
            backends: Dictionary of available backends
        """
        self.logger = logger
        self.backends = backends

    def perform_direct_retrieval(self, query: str, context: Dict = None) -> QueryExecutionResult:
        """
        Perform direct retrieval across multiple backends for comprehensive results
        
        Args:
            query: Query string for retrieval
            context: Additional context for retrieval
            
        Returns:
            Combined results from multiple backends
        """
        context = context or {}
        start_time = time.time()
        
        self.logger.info("Performing direct retrieval across all available backends")
        
        # Execute query on all available backends concurrently
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.backends)) as executor:
            future_to_backend = {}
            
            for backend_type, backend_instance in self.backends.items():
                # Adapt query for each backend
                adapted_query = self._adapt_query_for_backend(query, backend_type)
                future = executor.submit(self._execute_on_backend, 
                                       adapted_query, backend_instance, backend_type)
                future_to_backend[future] = backend_type
            
            # Collect results
            for future in as_completed(future_to_backend):
                backend_type = future_to_backend[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results[backend_type] = result
                    self.logger.debug(f"Retrieved results from {backend_type}")
                except Exception as e:
                    self.logger.warning(f"Failed to retrieve from {backend_type}: {e}")
                    results[backend_type] = None
        
        # Combine and merge results
        combined_result = self._combine_results(results, query)
        execution_time = time.time() - start_time
        
        self.logger.info(f"Direct retrieval completed in {execution_time:.3f}s")
        
        return QueryExecutionResult(
            success=True,
            result=combined_result,
            execution_time=execution_time,
            backend_used=BackendType.AUTO,
            metadata={'retrieval_type': 'direct', 'backends_used': list(results.keys())}
        )

    def _adapt_query_for_backend(self, query: str, backend_type: BackendType) -> str:
        """Adapt query for specific backend"""
        if backend_type == BackendType.NEO4J:
            # Ensure Cypher-compatible query
            if not any(keyword in query.upper() for keyword in ['MATCH', 'CREATE', 'MERGE']):
                return f"MATCH (n) WHERE n.name CONTAINS '{query}' RETURN n LIMIT 100"
        elif backend_type == BackendType.RDF4J:
            # Ensure SPARQL-compatible query
            if not query.strip().upper().startswith(('SELECT', 'CONSTRUCT', 'ASK')):
                return f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?s ?p ?o WHERE {{ 
                    ?s ?p ?o . 
                    FILTER(CONTAINS(STR(?o), "{query}"))
                }} LIMIT 100
                """
        elif backend_type == BackendType.NETWORKX:
            # Ensure NetworkX-compatible code
            if 'result =' not in query:
                return f"""
# Search for nodes containing: {query}
result = []
search_term = "{query}".lower()
for node, data in self.G.nodes(data=True):
    if search_term in str(node).lower() or any(search_term in str(v).lower() for v in data.values()):
        result.append((node, data))
"""
        
        return query

    def _execute_on_backend(self, query: str, backend_instance: Any, 
                          backend_type: BackendType) -> Any:
        """Execute query on specific backend"""
        try:
            if backend_type == BackendType.NETWORKX:
                return backend_instance.get_query(query)
            else:
                return backend_instance.get_query(query)
        except Exception as e:
            self.logger.error(f"Execution failed on {backend_type}: {e}")
            raise

    def _combine_results(self, results: Dict[BackendType, Any], original_query: str) -> Dict:
        """Combine results from multiple backends"""
        combined = {
            'original_query': original_query,
            'backend_results': {},
            'unified_results': [],
            'summary': {
                'total_backends': len(results),
                'successful_backends': 0,
                'failed_backends': 0
            }
        }
        
        for backend_type, result in results.items():
            if result and result[1]:  # Check if successful
                combined['backend_results'][backend_type.value] = {
                    'success': True,
                    'data': result[0],
                    'error': result[2]
                }
                combined['summary']['successful_backends'] += 1
                
                # Add to unified results
                if result[0]:
                    combined['unified_results'].append({
                        'source': backend_type.value,
                        'data': result[0]
                    })
            else:
                combined['backend_results'][backend_type.value] = {
                    'success': False,
                    'data': None,
                    'error': result[2] if result else "Backend unavailable"
                }
                combined['summary']['failed_backends'] += 1
        
        return combined


# Example usage and testing functions
def create_example_processor() -> AdvancedQueryProcessor:
    """
    Create an example Advanced Query Processor instance for testing
    
    Returns:
        Configured AdvancedQueryProcessor instance
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configurations (would be loaded from config files in production)
    neo4j_config = {
        'uri': 'bolt://localhost:7687',
        'user': 'neo4j',
        'password': 'password'
    }
    
    rdf4j_config = {
        'read_endpoint': 'http://localhost:8080/rdf4j-server/repositories/test',
        'write_endpoint': 'http://localhost:8080/rdf4j-server/repositories/test/statements'
    }
    
    # Create processor instance
    processor = AdvancedQueryProcessor(
        neo4j_config=neo4j_config,
        rdf4j_config=rdf4j_config,
        enable_caching=True,
        performance_monitoring=True
    )
    
    return processor


if __name__ == "__main__":
    # Example usage
    processor = create_example_processor()
    
    # Example queries for testing
    test_queries = [
        "MATCH (n:Person) RETURN n LIMIT 10",  # Cypher query
        "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",  # SPARQL query
        "result = list(self.G.nodes())",  # NetworkX query
        "Find all connected components"  # Natural language query
    ]
    
    for query in test_queries:
        print(f"\nProcessing query: {query}")
        result = processor.process_query(query)
        print(f"Success: {result.success}")
        print(f"Backend used: {result.backend_used}")
        print(f"Execution time: {result.execution_time:.3f}s")
        if result.error:
            print(f"Error: {result.error}")

    # Print performance statistics
    print("\nPerformance Statistics:")
    stats = processor.get_performance_stats()
    for metric, data in stats.items():
        if data['count'] > 0:
            print(f"{metric}: avg={data['average']:.3f}s, count={data['count']}") 