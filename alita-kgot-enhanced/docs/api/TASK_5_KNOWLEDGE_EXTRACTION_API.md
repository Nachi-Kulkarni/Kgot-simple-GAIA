# Task 5: Knowledge Extraction Methods - API Reference

## 📖 **Overview**

Complete API reference for the KGoT Knowledge Extraction Methods implementation, providing comprehensive documentation for all classes, methods, and interfaces.

**Module**: `kgot_core.knowledge_extraction`  
**Python Version**: 3.8+  
**Dependencies**: asyncio, typing, dataclasses, abc

---

## 🏗️ **Core Classes**

### **KnowledgeExtractionManager**

The main orchestrator class for all knowledge extraction operations.

```python
class KnowledgeExtractionManager:
    """
    Central manager for knowledge extraction with dynamic optimization.
    
    Manages three extraction methods and provides automatic method selection
    based on task requirements and performance optimization.
    """
```

#### **Constructor**

```python
def __init__(self, 
             backend_type: str,
             backend_config: Dict[str, Any] = None,
             optimization_weights: OptimizationWeights = None,
             logger: logging.Logger = None):
    """
    Initialize the Knowledge Extraction Manager.
    
    Args:
        backend_type (str): Backend type ('neo4j', 'networkx', 'rdf4j')
        backend_config (Dict[str, Any], optional): Backend-specific configuration
        optimization_weights (OptimizationWeights, optional): Optimization weights
        logger (logging.Logger, optional): Custom logger instance
        
    Raises:
        ValueError: If backend_type is not supported
        ConnectionError: If backend connection fails
    """
```

#### **Main Methods**

##### **extract_knowledge()**

```python
async def extract_knowledge(self,
                          query: str,
                          context: ExtractionContext,
                          preferred_method: Optional[str] = None) -> ExtractionResult:
    """
    Extract knowledge using automatic method selection.
    
    Args:
        query (str): Query string or extraction specification
        context (ExtractionContext): Task context and requirements
        preferred_method (Optional[str]): Override automatic selection
        
    Returns:
        ExtractionResult: Comprehensive extraction results with metrics
        
    Raises:
        ExtractionError: If extraction fails
        ValidationError: If query or context is invalid
        
    Example:
        >>> manager = KnowledgeExtractionManager('neo4j')
        >>> context = ExtractionContext(accuracy_threshold=0.9)
        >>> result = await manager.extract_knowledge(
        ...     "Find entities related to 'artificial intelligence'",
        ...     context
        ... )
        >>> print(f"Found {len(result.entities)} entities")
    """
```

##### **extract_with_method()**

```python
async def extract_with_method(self,
                             method: str,
                             query: str,
                             parameters: Dict[str, Any] = None,
                             context: ExtractionContext = None) -> ExtractionResult:
    """
    Extract knowledge using a specific method.
    
    Args:
        method (str): Extraction method ('graph_query', 'general_purpose', 'direct_retrieval')
        query (str): Method-specific query
        parameters (Dict[str, Any], optional): Method parameters
        context (ExtractionContext, optional): Extraction context
        
    Returns:
        ExtractionResult: Method-specific extraction results
        
    Example:
        >>> # Graph query extraction
        >>> result = await manager.extract_with_method(
        ...     'graph_query',
        ...     'MATCH (n:Entity)-[r]->(m) RETURN n, r, m LIMIT 100'
        ... )
        
        >>> # General purpose extraction
        >>> result = await manager.extract_with_method(
        ...     'general_purpose',
        ...     query='community_detection',
        ...     parameters={'algorithm': 'louvain'}
        ... )
    """
```

##### **generate_mcp_tools()**

```python
async def generate_mcp_tools(self,
                           knowledge_patterns: List[Dict[str, Any]],
                           tool_types: List[str] = None) -> List[MCPToolSpec]:
    """
    Generate MCP tools from extracted knowledge patterns.
    
    Args:
        knowledge_patterns (List[Dict[str, Any]]): Extracted knowledge patterns
        tool_types (List[str], optional): Types of tools to generate
        
    Returns:
        List[MCPToolSpec]: Generated MCP tool specifications
        
    Example:
        >>> patterns = result.knowledge_patterns
        >>> tools = await manager.generate_mcp_tools(
        ...     patterns,
        ...     tool_types=['entity_analysis', 'relationship_mapping']
        ... )
        >>> print(f"Generated {len(tools)} MCP tools")
    """
```

---

## 🔧 **Extraction Method Classes**

### **BaseExtractor**

Abstract base class for all extraction methods.

```python
class BaseExtractor(ABC):
    """Abstract base class for knowledge extraction methods."""
    
    @abstractmethod
    async def extract(self, query: str, parameters: Dict[str, Any] = None) -> ExtractionResult:
        """Extract knowledge using this method."""
        pass
    
    @abstractmethod
    def validate_query(self, query: str) -> bool:
        """Validate query format for this method."""
        pass
    
    @abstractmethod
    def estimate_cost(self, query: str, parameters: Dict[str, Any] = None) -> float:
        """Estimate computational cost for this extraction."""
        pass
```

### **GraphQueryExtractor**

Handles structured query languages (Cypher, SPARQL).

```python
class GraphQueryExtractor(BaseExtractor):
    """
    Graph query-based extraction using Cypher (Neo4j) or SPARQL (RDF4J).
    
    Supports:
    - Cypher queries for Neo4j
    - SPARQL queries for RDF4J
    - Query optimization and caching
    - Result validation and processing
    """
```

#### **Methods**

```python
async def extract(self, query: str, parameters: Dict[str, Any] = None) -> ExtractionResult:
    """
    Execute graph query and return structured results.
    
    Args:
        query (str): Cypher or SPARQL query
        parameters (Dict[str, Any], optional): Query parameters
        
    Returns:
        ExtractionResult: Query results with metadata
        
    Example:
        >>> extractor = GraphQueryExtractor(neo4j_config)
        >>> result = await extractor.extract(
        ...     "MATCH (n:Entity) WHERE n.type = $entity_type RETURN n",
        ...     parameters={'entity_type': 'Person'}
        ... )
    """

def optimize_query(self, query: str) -> str:
    """
    Optimize query for better performance.
    
    Args:
        query (str): Original query
        
    Returns:
        str: Optimized query
    """

def validate_cypher_query(self, query: str) -> bool:
    """Validate Cypher query syntax."""
    
def validate_sparql_query(self, query: str) -> bool:
    """Validate SPARQL query syntax."""
```

### **GeneralPurposeExtractor**

NetworkX-based analysis with custom algorithms.

```python
class GeneralPurposeExtractor(BaseExtractor):
    """
    General-purpose extraction using NetworkX and custom algorithms.
    
    Capabilities:
    - Community detection algorithms
    - Centrality analysis
    - Path analysis and shortest paths
    - Custom algorithm implementation
    - Statistical graph analysis
    """
```

#### **Methods**

```python
async def extract(self, query: str, parameters: Dict[str, Any] = None) -> ExtractionResult:
    """
    Perform general-purpose graph analysis.
    
    Args:
        query (str): Analysis type or custom algorithm name
        parameters (Dict[str, Any], optional): Algorithm parameters
        
    Returns:
        ExtractionResult: Analysis results
        
    Example:
        >>> extractor = GeneralPurposeExtractor(graph)
        >>> result = await extractor.extract(
        ...     'community_detection',
        ...     parameters={'algorithm': 'louvain', 'resolution': 1.0}
        ... )
    """

async def detect_communities(self, algorithm: str = 'louvain', **kwargs) -> Dict[str, Any]:
    """Detect communities using specified algorithm."""

async def calculate_centrality(self, centrality_type: str, **kwargs) -> Dict[str, float]:
    """Calculate centrality measures for nodes."""

async def analyze_paths(self, source: str, target: str = None, **kwargs) -> Dict[str, Any]:
    """Analyze paths between nodes."""

def run_custom_algorithm(self, algorithm_func: Callable, **kwargs) -> Any:
    """Execute custom analysis algorithm."""
```

### **DirectRetrievalExtractor**

Fast, direct access for simple queries.

```python
class DirectRetrievalExtractor(BaseExtractor):
    """
    Direct retrieval for fast, simple operations.
    
    Operations:
    - Node lookup by ID
    - Immediate neighbor retrieval
    - Simple property filtering
    - Basic relationship traversal
    """
```

#### **Methods**

```python
async def extract(self, query: str, parameters: Dict[str, Any] = None) -> ExtractionResult:
    """
    Perform direct retrieval operations.
    
    Args:
        query (str): Retrieval operation type
        parameters (Dict[str, Any], optional): Operation parameters
        
    Returns:
        ExtractionResult: Direct retrieval results
        
    Example:
        >>> extractor = DirectRetrievalExtractor(backend)
        >>> result = await extractor.extract(
        ...     'get_nodes',
        ...     parameters={'node_ids': ['n1', 'n2'], 'include_neighbors': True}
        ... )
    """

async def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve single node by ID."""

async def get_neighbors(self, node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
    """Get neighboring nodes."""

async def filter_nodes(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter nodes by properties."""

async def get_relationships(self, source_id: str, target_id: str = None) -> List[Dict[str, Any]]:
    """Get relationships between nodes."""
```

---

## ⚖️ **Optimization Classes**

### **ExtractionMetrics**

Performance metrics tracking and calculation.

```python
@dataclass
class ExtractionMetrics:
    """
    Performance metrics for extraction operations.
    
    Attributes:
        accuracy (float): Accuracy percentage (0.0-1.0)
        cost (float): Computational cost units
        runtime (float): Execution time in seconds
        memory_usage (float): Memory usage in MB
        success_rate (float): Success rate (0.0-1.0)
        timestamp (datetime): Measurement timestamp
    """
    accuracy: float
    cost: float
    runtime: float
    memory_usage: float = 0.0
    success_rate: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_efficiency_score(self, weights: OptimizationWeights) -> float:
        """
        Calculate weighted efficiency score.
        
        Args:
            weights (OptimizationWeights): Optimization weights
            
        Returns:
            float: Efficiency score (0.0-1.0)
        """
```

### **OptimizationWeights**

Configuration for trade-off optimization.

```python
@dataclass
class OptimizationWeights:
    """
    Weights for multi-objective optimization.
    
    Attributes:
        accuracy (float): Weight for accuracy (default: 0.4)
        cost (float): Weight for cost (default: 0.3)
        runtime (float): Weight for runtime (default: 0.3)
    """
    accuracy: float = 0.4
    cost: float = 0.3
    runtime: float = 0.3
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.accuracy + self.cost + self.runtime
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
```

### **ExtractionContext**

Task context and requirements.

```python
@dataclass
class ExtractionContext:
    """
    Context and requirements for extraction tasks.
    
    Attributes:
        accuracy_threshold (float): Minimum required accuracy
        max_cost (float): Maximum allowed cost
        max_runtime (float): Maximum allowed runtime in seconds
        preferred_method (Optional[str]): Preferred extraction method
        cache_results (bool): Whether to cache results
        parallel_execution (bool): Allow parallel execution
    """
    accuracy_threshold: float = 0.8
    max_cost: float = 100.0
    max_runtime: float = 60.0
    preferred_method: Optional[str] = None
    cache_results: bool = True
    parallel_execution: bool = True
```

### **KnowledgeExtractionOptimizer**

Optimization engine for method selection.

```python
class KnowledgeExtractionOptimizer:
    """
    Optimizer for dynamic method selection and performance tuning.
    
    Features:
    - Method selection based on historical performance
    - Dynamic weight adjustment
    - Performance prediction
    - Resource optimization
    """
    
    def select_optimal_method(self, 
                            context: ExtractionContext,
                            available_methods: List[str]) -> str:
        """
        Select optimal extraction method for given context.
        
        Args:
            context (ExtractionContext): Task requirements
            available_methods (List[str]): Available extraction methods
            
        Returns:
            str: Selected method name
        """
    
    def predict_performance(self, 
                          method: str, 
                          query: str, 
                          context: ExtractionContext) -> ExtractionMetrics:
        """
        Predict performance metrics for method and query.
        
        Args:
            method (str): Extraction method
            query (str): Query string
            context (ExtractionContext): Task context
            
        Returns:
            ExtractionMetrics: Predicted performance metrics
        """
    
    def update_performance_history(self, 
                                 method: str, 
                                 metrics: ExtractionMetrics):
        """Update historical performance data."""
```

---

## 🔌 **MCP Integration**

### **MCPIntegrationBridge**

Bridge for MCP tool generation.

```python
class MCPIntegrationBridge:
    """
    Bridge for MCP tool generation based on extracted knowledge.
    
    Capabilities:
    - Pattern analysis for tool opportunities
    - Dynamic tool specification generation
    - Tool implementation scaffolding
    - Integration with MCP infrastructure
    """
    
    async def generate_tools_from_knowledge(self, 
                                          knowledge_patterns: List[Dict[str, Any]],
                                          tool_types: List[str] = None) -> List[MCPToolSpec]:
        """
        Generate MCP tools from knowledge patterns.
        
        Args:
            knowledge_patterns (List[Dict[str, Any]]): Extracted patterns
            tool_types (List[str], optional): Types of tools to generate
            
        Returns:
            List[MCPToolSpec]: Generated tool specifications
        """
    
    def analyze_patterns(self, patterns: List[Dict[str, Any]]) -> List[ToolOpportunity]:
        """Analyze patterns for tool generation opportunities."""
    
    def generate_tool_spec(self, opportunity: ToolOpportunity) -> MCPToolSpec:
        """Generate MCP tool specification from opportunity."""
    
    def create_tool_implementation(self, spec: MCPToolSpec) -> str:
        """Create tool implementation code."""
```

### **MCPToolSpec**

MCP tool specification format.

```python
@dataclass
class MCPToolSpec:
    """
    Specification for generated MCP tools.
    
    Attributes:
        name (str): Tool name
        description (str): Tool description
        parameters (Dict[str, Any]): Tool parameters schema
        implementation (str): Tool implementation code
        capabilities (List[str]): Tool capabilities
        dependencies (List[str]): Required dependencies
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    implementation: str
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
```

---

## 📊 **Result Classes**

### **ExtractionResult**

Comprehensive extraction results.

```python
@dataclass
class ExtractionResult:
    """
    Comprehensive results from knowledge extraction.
    
    Attributes:
        success (bool): Whether extraction succeeded
        method_used (str): Method that was used
        entities (List[Dict[str, Any]]): Extracted entities
        relationships (List[Dict[str, Any]]): Extracted relationships
        knowledge_patterns (List[Dict[str, Any]]): Identified patterns
        metrics (ExtractionMetrics): Performance metrics
        metadata (Dict[str, Any]): Additional metadata
        error_message (Optional[str]): Error message if failed
    """
    success: bool
    method_used: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_patterns: List[Dict[str, Any]] = field(default_factory=list)
    metrics: ExtractionMetrics = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
    
    def get_summary(self) -> str:
        """Get human-readable summary of results."""
```

---

## 🛡️ **Exception Classes**

### **Custom Exceptions**

```python
class ExtractionError(Exception):
    """Base exception for knowledge extraction errors."""
    pass

class QueryExecutionError(ExtractionError):
    """Error during query execution."""
    pass

class BackendConnectionError(ExtractionError):
    """Backend connection issues."""
    pass

class OptimizationError(ExtractionError):
    """Optimization process errors."""
    pass

class ValidationError(ExtractionError):
    """Query or parameter validation errors."""
    pass

class MCPIntegrationError(ExtractionError):
    """MCP integration errors."""
    pass
```

---

## 🔧 **Configuration Classes**

### **BackendConfig**

Backend-specific configuration.

```python
@dataclass
class BackendConfig:
    """Configuration for backend connections."""
    connection_string: str
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    timeout: int = 30
    max_connections: int = 10
    ssl_enabled: bool = False
    additional_params: Dict[str, Any] = field(default_factory=dict)
```

### **CacheConfig**

Caching configuration.

```python
@dataclass
class CacheConfig:
    """Configuration for result caching."""
    enabled: bool = True
    ttl: int = 3600  # Time-to-live in seconds
    max_size: int = 1000  # Maximum cache entries
    storage_type: str = 'memory'  # 'memory' or 'redis'
    compression: bool = True
```

---

## 📚 **Usage Examples**

### **Basic Extraction**

```python
import asyncio
from kgot_core.knowledge_extraction import (
    KnowledgeExtractionManager,
    ExtractionContext,
    OptimizationWeights
)

async def basic_extraction():
    # Initialize manager
    manager = KnowledgeExtractionManager(
        backend_type='neo4j',
        optimization_weights=OptimizationWeights(
            accuracy=0.5,
            cost=0.2,
            runtime=0.3
        )
    )
    
    # Create extraction context
    context = ExtractionContext(
        accuracy_threshold=0.9,
        max_cost=50.0,
        max_runtime=10.0
    )
    
    # Extract knowledge
    result = await manager.extract_knowledge(
        query="Find entities related to artificial intelligence",
        context=context
    )
    
    # Process results
    print(f"Success: {result.success}")
    print(f"Method used: {result.method_used}")
    print(f"Entities found: {len(result.entities)}")
    print(f"Accuracy: {result.metrics.accuracy:.2%}")
    print(f"Runtime: {result.metrics.runtime:.2f}s")

# Run example
asyncio.run(basic_extraction())
```

### **Method-Specific Extraction**

```python
async def method_specific_extraction():
    manager = KnowledgeExtractionManager('neo4j')
    
    # Graph query extraction
    graph_result = await manager.extract_with_method(
        method='graph_query',
        query="""
        MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity)
        WHERE n.category = 'Technology'
        RETURN n, r, m
        ORDER BY r.strength DESC
        LIMIT 100
        """
    )
    
    # General purpose extraction
    analysis_result = await manager.extract_with_method(
        method='general_purpose',
        query='community_detection',
        parameters={
            'algorithm': 'louvain',
            'resolution': 1.0,
            'random_state': 42
        }
    )
    
    # Direct retrieval
    direct_result = await manager.extract_with_method(
        method='direct_retrieval',
        query='get_nodes',
        parameters={
            'node_ids': ['entity_1', 'entity_2', 'entity_3'],
            'include_neighbors': True,
            'neighbor_depth': 2
        }
    )
    
    return graph_result, analysis_result, direct_result
```

### **MCP Tool Generation**

```python
async def generate_mcp_tools():
    manager = KnowledgeExtractionManager('neo4j')
    
    # Extract knowledge to identify patterns
    result = await manager.extract_knowledge(
        query="Analyze knowledge patterns in the graph",
        context=ExtractionContext(accuracy_threshold=0.95)
    )
    
    # Generate MCP tools from patterns
    tools = await manager.generate_mcp_tools(
        knowledge_patterns=result.knowledge_patterns,
        tool_types=[
            'entity_analysis',
            'relationship_mapping',
            'pattern_detection',
            'community_analysis'
        ]
    )
    
    # Display generated tools
    for tool in tools:
        print(f"Generated tool: {tool.name}")
        print(f"Description: {tool.description}")
        print(f"Capabilities: {', '.join(tool.capabilities)}")
        print("---")
    
    return tools
```

### **Performance Optimization**

```python
async def performance_optimization():
    # Create optimizer with custom weights
    weights = OptimizationWeights(
        accuracy=0.6,  # Prioritize accuracy
        cost=0.2,      # Lower cost priority
        runtime=0.2    # Lower runtime priority
    )
    
    manager = KnowledgeExtractionManager(
        backend_type='neo4j',
        optimization_weights=weights
    )
    
    # Multiple extractions with performance tracking
    queries = [
        "Find technology entities",
        "Analyze research patterns",
        "Detect entity clusters",
        "Map relationship networks"
    ]
    
    results = []
    for query in queries:
        context = ExtractionContext(
            accuracy_threshold=0.9,
            max_cost=100.0,
            max_runtime=30.0
        )
        
        result = await manager.extract_knowledge(query, context)
        results.append(result)
        
        print(f"Query: {query}")
        print(f"Method: {result.method_used}")
        print(f"Efficiency: {result.metrics.calculate_efficiency_score(weights):.3f}")
        print("---")
    
    return results
```

---

## 🔍 **Integration Examples**

### **Backend Integration**

```python
# Neo4j backend
neo4j_config = BackendConfig(
    connection_string="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="knowledge_graph"
)

neo4j_manager = KnowledgeExtractionManager(
    backend_type='neo4j',
    backend_config=neo4j_config
)

# NetworkX backend
networkx_manager = KnowledgeExtractionManager(
    backend_type='networkx',
    backend_config={'graph_file': 'knowledge_graph.graphml'}
)

# RDF4J backend
rdf4j_config = BackendConfig(
    connection_string="http://localhost:8080/rdf4j-server",
    database="knowledge_repository"
)

rdf4j_manager = KnowledgeExtractionManager(
    backend_type='rdf4j',
    backend_config=rdf4j_config
)
```

### **Caching Integration**

```python
# Enable caching
cache_config = CacheConfig(
    enabled=True,
    ttl=7200,  # 2 hours
    max_size=5000,
    storage_type='redis',
    compression=True
)

manager = KnowledgeExtractionManager(
    backend_type='neo4j',
    cache_config=cache_config
)
```

---

## 📋 **API Reference Summary**

### **Main Classes**
- `KnowledgeExtractionManager`: Main orchestrator
- `GraphQueryExtractor`: Query-based extraction
- `GeneralPurposeExtractor`: Algorithm-based extraction  
- `DirectRetrievalExtractor`: Fast direct retrieval
- `KnowledgeExtractionOptimizer`: Performance optimizer
- `MCPIntegrationBridge`: MCP tool generation

### **Data Classes**
- `ExtractionMetrics`: Performance metrics
- `OptimizationWeights`: Optimization configuration
- `ExtractionContext`: Task requirements
- `ExtractionResult`: Comprehensive results
- `MCPToolSpec`: MCP tool specifications
- `BackendConfig`: Backend configuration

### **Exception Classes**
- `ExtractionError`: Base exception
- `QueryExecutionError`: Query execution errors
- `BackendConnectionError`: Connection issues
- `OptimizationError`: Optimization errors
- `ValidationError`: Validation errors
- `MCPIntegrationError`: MCP integration errors

**Status**: ✅ **API Complete** - Comprehensive API documentation for production use.
