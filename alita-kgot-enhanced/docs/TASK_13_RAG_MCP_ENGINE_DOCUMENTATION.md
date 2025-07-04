# RAG-MCP Engine Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Quick Start Guide](#quick-start-guide)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Integration Guide](#integration-guide)
8. [Performance & Analytics](#performance--analytics)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)
11. [Development & Contributing](#development--contributing)

---

## Overview

The **RAG-MCP Engine** is a state-of-the-art implementation of the RAG-MCP (Retrieval-Augmented Generation for Model Control Protocol) framework, specifically designed for the Alita-KGoT Enhanced system. It implements the complete RAG-MCP Section 3.2 framework architecture with Section 3.3 three-step pipeline integration.

### Key Features

- 🎯 **RAG-MCP Section 3.2 Framework**: Complete implementation with external vector indexing
- 🔄 **Three-Step Pipeline**: Query Encoding → Vector Search & Validation → MCP Selection
- 🤖 **LangChain Integration**: Intelligent agents for MCP analysis and optimization
- 🌐 **OpenRouter Support**: Claude-3-Sonnet validation and ada-002 embeddings
- 🔗 **Alita Integration**: Seamless bridge to existing MCP Brainstorming workflow
- ⚡ **Production Ready**: Comprehensive logging, caching, and error handling

### Research Foundation

Based on the RAG-MCP research paper implementation:
- **Section 3.2**: RAG-MCP Framework complete architecture
- **Section 3.3**: Three-step semantic search pipeline
- **Section 4**: Pareto MCP retrieval based on experimental findings
- **Section 2.3.1**: Integration with Alita MCP Brainstorming workflow

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG-MCP Engine                          │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ LangChain     │  │ RAGMCPEngine    │  │ Integration   │ │
│  │ Agents        │  │ (Orchestrator)  │  │ Bridge        │ │
│  └───────────────┘  └─────────────────┘  └───────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ Vector Index  │  │ RAGMCPValidator │  │ Pareto MCP    │ │
│  │ Manager       │  │                 │  │ Registry      │ │
│  └───────────────┘  └─────────────────┘  └───────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ OpenRouter    │  │ FAISS Vector    │  │ Winston       │ │
│  │ Embeddings    │  │ Search          │  │ Logging       │ │
│  └───────────────┘  └─────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query 
    ↓
Step 1: Query Encoding (OpenRouter Embeddings)
    ↓
Step 2: Vector Search (FAISS/Fallback Similarity)
    ↓
Step 3: MCP Selection (LLM Validation + Pareto Optimization)
    ↓
Final MCP Recommendations
    ↓
Integration with Alita MCP Brainstorming
```

### Core Classes

1. **RAGMCPEngine**: Main orchestrator implementing the complete pipeline
2. **VectorIndexManager**: Handles embedding generation and similarity search
3. **RAGMCPValidator**: Provides relevance validation and scoring
4. **ParetoMCPRegistry**: Manages high-value MCP specifications
5. **LangChain Agents**: Intelligent analysis tools for MCP optimization

---

## Installation & Setup

### Prerequisites

```bash
# Python dependencies
pip install langchain
pip install langchain-openai
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install numpy
pip install aiohttp
pip install python-dotenv

# Optional for enhanced performance
pip install sentence-transformers
pip install chromadb
```

### Environment Configuration

Create a `.env` file in your project root:

```env
# Required for OpenRouter integration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional configurations
APP_URL=http://localhost:3000
MCP_BRAINSTORMING_ENDPOINT=http://localhost:8001/api/mcp-brainstorming
RAG_MCP_CACHE_DIR=./cache/rag_mcp
RAG_MCP_LOG_LEVEL=INFO
```

### File Structure

```
alita-kgot-enhanced/
├── alita_core/
│   ├── rag_mcp_engine.py          # Main implementation
│   ├── environment_manager.py      # Existing component
│   └── code_runner.py              # Existing component
├── config/
│   └── logging/
│       └── winston_config.js       # Logging configuration
├── cache/
│   └── rag_mcp/                    # Cache directory (auto-created)
└── docs/
    └── RAG_MCP_ENGINE_DOCUMENTATION.md  # This file
```

---

## Quick Start Guide

### Basic Usage

```python
#!/usr/bin/env python3
import asyncio
from alita_core.rag_mcp_engine import RAGMCPEngine

async def basic_example():
    """Basic RAG-MCP Engine usage example"""
    
    # Initialize the engine
    engine = RAGMCPEngine(
        similarity_threshold=0.7,
        top_k_candidates=5,
        enable_llm_validation=True
    )
    
    # Initialize (builds vector indices)
    success = await engine.initialize()
    if not success:
        print("Failed to initialize RAG-MCP Engine")
        return
    
    # Execute pipeline for a user query
    result = await engine.execute_rag_mcp_pipeline(
        "I need to scrape product data from e-commerce websites and analyze it"
    )
    
    if result.success:
        print(f"Found {len(result.retrieved_mcps)} relevant MCPs:")
        for mcp_result in result.retrieved_mcps:
            mcp = mcp_result.mcp_spec
            print(f"- {mcp.name}: {mcp.description}")
            print(f"  Confidence: {mcp_result.selection_confidence:.2f}")
            print(f"  Pareto Score: {mcp.pareto_score:.2f}")
    else:
        print(f"Pipeline failed: {result.error_message}")

# Run the example
asyncio.run(basic_example())
```

### Quick Recommendations

```python
async def quick_recommendations():
    """Get quick MCP recommendations"""
    
    engine = RAGMCPEngine()
    await engine.initialize()
    
    recommendations = await engine.get_mcp_recommendations(
        "Automate email notifications and calendar scheduling",
        max_recommendations=3
    )
    
    for rec in recommendations:
        print(f"{rec['name']}: {rec['description']}")
        print(f"Category: {rec['category']}")
        print(f"Confidence: {rec['confidence_score']}")
        print("---")

asyncio.run(quick_recommendations())
```

---

## API Reference

### RAGMCPEngine Class

#### Constructor

```python
RAGMCPEngine(
    openrouter_api_key: Optional[str] = None,
    vector_dimensions: int = 1536,
    similarity_threshold: float = 0.7,
    top_k_candidates: int = 5,
    enable_llm_validation: bool = True,
    cache_directory: Optional[str] = None,
    mcp_brainstorming_endpoint: Optional[str] = None
)
```

**Parameters:**
- `openrouter_api_key`: OpenRouter API key (defaults to env var)
- `vector_dimensions`: Embedding vector dimensions (1536 for ada-002)
- `similarity_threshold`: Minimum similarity score for retrieval (0.0-1.0)
- `top_k_candidates`: Number of top candidates to retrieve
- `enable_llm_validation`: Enable LLM-based validation using Claude-3-Sonnet
- `cache_directory`: Directory for caching indices and results
- `mcp_brainstorming_endpoint`: Endpoint for Alita MCP Brainstorming integration

#### Core Methods

##### `initialize() -> bool`

Initialize the RAG-MCP engine and build vector indices.

```python
engine = RAGMCPEngine()
success = await engine.initialize()
```

**Returns:** `bool` - Success status

##### `execute_rag_mcp_pipeline(user_query: str, options: Optional[Dict[str, Any]] = None) -> RAGMCPPipelineResult`

Execute the complete three-step RAG-MCP pipeline.

```python
result = await engine.execute_rag_mcp_pipeline(
    "Analyze CSV data and create visualizations",
    options={
        'top_k': 5,
        'enable_agent_analysis': True,
        'filter_category': 'DATA_PROCESSING'
    }
)
```

**Parameters:**
- `user_query`: User's task description
- `options`: Pipeline configuration options

**Pipeline Options:**
- `top_k`: Number of top results to return
- `enable_agent_analysis`: Enable LangChain agent analysis
- `filter_category`: Filter by MCP category (WEB_INFORMATION, DATA_PROCESSING, etc.)

**Returns:** `RAGMCPPipelineResult` containing:
- `query`: Processed query object
- `retrieved_mcps`: List of validated MCP results
- `pipeline_metadata`: Timing and performance data
- `success`: Pipeline success status
- `error_message`: Error details if failed

##### `get_mcp_recommendations(query: str, max_recommendations: int = 3) -> List[Dict[str, Any]]`

Get simplified MCP recommendations for quick queries.

```python
recommendations = await engine.get_mcp_recommendations(
    "Search Wikipedia and process text data",
    max_recommendations=3
)
```

**Returns:** List of recommendation dictionaries with:
- `name`: MCP name
- `description`: MCP description
- `category`: MCP category
- `capabilities`: List of capabilities
- `confidence_score`: Selection confidence (0.0-1.0)
- `pareto_score`: Pareto efficiency score
- `reliability_score`: Reliability metric
- `recommendation_reason`: Explanation of recommendation

##### `get_performance_analytics() -> Dict[str, Any]`

Get detailed performance analytics for the pipeline.

```python
analytics = engine.get_performance_analytics()
print(f"Average pipeline time: {analytics['total_pipeline_time']['average_ms']:.2f}ms")
```

**Returns:** Analytics dictionary with timing metrics and system status

### Data Structures

#### MCPToolSpec

```python
@dataclass
class MCPToolSpec:
    name: str                           # MCP name
    description: str                    # MCP description  
    category: MCPCategory              # Category enum
    capabilities: List[str]            # List of capabilities
    usage_frequency: float             # Usage frequency (0.0-1.0)
    reliability_score: float           # Reliability score (0.0-1.0)
    cost_efficiency: float             # Cost efficiency (0.0-1.0)
    pareto_score: float               # Pareto optimization score
    embedding: Optional[np.ndarray]    # Vector embedding
    indexed_at: Optional[datetime]     # Indexing timestamp
```

#### MCPRetrievalResult

```python
@dataclass
class MCPRetrievalResult:
    mcp_spec: MCPToolSpec             # MCP specification
    similarity_score: float           # Semantic similarity score
    relevance_score: float            # Calculated relevance score
    validation_score: float           # LLM validation score
    selection_confidence: float       # Overall selection confidence
    ranking_factors: Dict[str, Any]   # Ranking transparency data
```

#### RAGMCPPipelineResult

```python
@dataclass
class RAGMCPPipelineResult:
    query: RAGMCPQuery                    # Processed query
    retrieved_mcps: List[MCPRetrievalResult]  # Retrieved MCPs
    pipeline_metadata: Dict[str, Any]     # Pipeline execution data
    processing_time_ms: float            # Total processing time
    success: bool                        # Success status
    error_message: Optional[str]         # Error message if failed
```

---

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key for embeddings and LLM | None | Yes |
| `APP_URL` | Application URL for OpenRouter headers | `http://localhost:3000` | No |
| `MCP_BRAINSTORMING_ENDPOINT` | Alita MCP Brainstorming endpoint | `http://localhost:8001/api/mcp-brainstorming` | No |
| `RAG_MCP_CACHE_DIR` | Cache directory for indices | `./cache/rag_mcp` | No |
| `RAG_MCP_LOG_LEVEL` | Logging level | `INFO` | No |

### Engine Configuration Examples

```python
# High-performance configuration
engine = RAGMCPEngine(
    vector_dimensions=1536,
    similarity_threshold=0.8,          # Higher threshold for precision
    top_k_candidates=3,                # Fewer, higher-quality results
    enable_llm_validation=True,        # Enable Claude-3-Sonnet validation
    cache_directory="./cache/rag_mcp"
)

# Development configuration
engine = RAGMCPEngine(
    similarity_threshold=0.6,          # Lower threshold for recall
    top_k_candidates=10,               # More candidates for testing
    enable_llm_validation=False        # Disable LLM for faster testing
)

# Integration-focused configuration
engine = RAGMCPEngine(
    mcp_brainstorming_endpoint="http://localhost:8001/api/mcp-brainstorming",
    cache_directory="/shared/cache"    # Shared cache for multiple instances
)
```

### Pareto MCP Registry Configuration

The system includes pre-configured high-value MCPs based on RAG-MCP research findings:

```python
# Web Information MCPs
- brave_search: Web search with high reliability (0.85 Pareto score)
- wikipedia_tool: Knowledge retrieval (0.82 Pareto score)
- web_scraper: Advanced scraping (0.78 Pareto score)

# Data Processing MCPs  
- csv_analyzer: Data analysis (0.88 Pareto score)
- image_processor: Image processing (0.75 Pareto score)
- text_processor: Text processing (0.73 Pareto score)

# Communication MCPs
- email_automation: Email management (0.80 Pareto score)
- calendar_integration: Schedule management (0.76 Pareto score)
- api_integrator: API connections (0.74 Pareto score)

# Development MCPs
- python_executor: Code execution (0.90 Pareto score)
- docker_manager: Container management (0.84 Pareto score)
- git_operations: Version control (0.79 Pareto score)
```

---

## Integration Guide

### Alita MCP Brainstorming Integration

The RAG-MCP Engine seamlessly integrates with the existing Alita Section 2.3.1 MCP Brainstorming workflow:

#### Setup Integration

```python
# Enable integration during initialization
engine = RAGMCPEngine(
    mcp_brainstorming_endpoint="http://localhost:8001/api/mcp-brainstorming"
)

await engine.initialize()  # Automatically creates integration session
```

#### Integration Flow

1. **Session Creation**: Engine creates session with MCP brainstorming system
2. **Pipeline Execution**: RAG-MCP pipeline runs independently
3. **Result Notification**: Results are automatically sent to brainstorming system
4. **Workflow Continuation**: Existing workflow can use RAG-MCP recommendations

#### Manual Integration

```python
# Get RAG-MCP results
result = await engine.execute_rag_mcp_pipeline(user_query)

# Send to existing MCP brainstorming system
import aiohttp

async with aiohttp.ClientSession() as session:
    async with session.post(
        "http://localhost:8001/api/mcp-brainstorming/rag-results",
        json={
            'query': user_query,
            'mcps': [
                {
                    'name': mcp.mcp_spec.name,
                    'confidence': mcp.selection_confidence,
                    'pareto_score': mcp.mcp_spec.pareto_score
                }
                for mcp in result.retrieved_mcps
            ]
        }
    ) as response:
        print(f"Integration status: {response.status}")
```

### Custom MCP Registry

Add custom MCPs to the registry:

```python
from alita_core.rag_mcp_engine import ParetoMCPRegistry, MCPToolSpec, MCPCategory

# Create custom MCP
custom_mcp = MCPToolSpec(
    name="custom_analytics_tool",
    description="Advanced analytics tool for business intelligence",
    category=MCPCategory.DATA_PROCESSING,
    capabilities=["data_visualization", "statistical_analysis", "reporting"],
    usage_frequency=0.75,
    reliability_score=0.85,
    cost_efficiency=0.80,
    pareto_score=0.83
)

# Add to registry
registry = ParetoMCPRegistry()
registry.add_mcp(custom_mcp)

# Use custom registry with engine
engine = RAGMCPEngine()
engine.pareto_registry = registry
```

### REST API Wrapper

Create a REST API wrapper for the RAG-MCP Engine:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RAG-MCP Engine API")
engine = None

class QueryRequest(BaseModel):
    query: str
    max_recommendations: int = 5
    enable_agent_analysis: bool = True

@app.on_event("startup")
async def startup_event():
    global engine
    engine = RAGMCPEngine()
    await engine.initialize()

@app.post("/recommendations")
async def get_recommendations(request: QueryRequest):
    try:
        result = await engine.execute_rag_mcp_pipeline(
            request.query,
            options={
                'top_k': request.max_recommendations,
                'enable_agent_analysis': request.enable_agent_analysis
            }
        )
        
        if result.success:
            return {
                'recommendations': [
                    {
                        'name': mcp.mcp_spec.name,
                        'description': mcp.mcp_spec.description,
                        'confidence': mcp.selection_confidence,
                        'category': mcp.mcp_spec.category.value
                    }
                    for mcp in result.retrieved_mcps
                ],
                'processing_time_ms': result.processing_time_ms
            }
        else:
            raise HTTPException(status_code=500, detail=result.error_message)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_analytics():
    return engine.get_performance_analytics()
```

---

## Performance & Analytics

### Performance Metrics

The engine tracks comprehensive performance metrics:

```python
analytics = engine.get_performance_analytics()

# Timing metrics (all in milliseconds)
print(f"Query encoding: {analytics['query_encoding_time']['average_ms']:.2f}ms")
print(f"Vector search: {analytics['vector_search_time']['average_ms']:.2f}ms") 
print(f"Validation: {analytics['validation_time']['average_ms']:.2f}ms")
print(f"Total pipeline: {analytics['total_pipeline_time']['average_ms']:.2f}ms")

# System status
print(f"LLM validation: {analytics['integration_status']['llm_validation_enabled']}")
print(f"Agent analysis: {analytics['integration_status']['agent_analysis_enabled']}")
print(f"MCP brainstorming: {analytics['integration_status']['mcp_brainstorming_connected']}")
```

### Performance Optimization

#### Vector Search Optimization

```python
# Use FAISS for large-scale deployments
engine = RAGMCPEngine(vector_dimensions=1536)  # Enables FAISS if available

# Adjust similarity threshold for performance vs. quality trade-off
engine = RAGMCPEngine(similarity_threshold=0.8)  # Higher = faster, fewer results

# Reduce top_k for faster responses
engine = RAGMCPEngine(top_k_candidates=3)  # Fewer candidates = faster validation
```

#### Caching Strategies

```python
# Enable persistent caching
engine = RAGMCPEngine(cache_directory="/persistent/cache")

# Warm up cache on startup
await engine.initialize()  # Builds and caches vector indices

# Monitor cache performance
analytics = engine.get_performance_analytics()
cache_hits = analytics.get('cache_hits', 0)
cache_misses = analytics.get('cache_misses', 0)
cache_ratio = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
print(f"Cache hit ratio: {cache_ratio:.2%}")
```

#### Memory Management

```python
# Monitor memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory usage: {get_memory_usage():.2f} MB")

# Clear cache periodically
engine.vector_manager.fallback_index.clear()
engine.validator.validation_cache.clear()
```

### Benchmarking

```python
import time
import statistics

async def benchmark_pipeline(engine, queries, iterations=5):
    """Benchmark pipeline performance"""
    
    all_times = []
    
    for query in queries:
        query_times = []
        
        for _ in range(iterations):
            start_time = time.time()
            result = await engine.execute_rag_mcp_pipeline(query)
            end_time = time.time()
            
            if result.success:
                query_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        if query_times:
            avg_time = statistics.mean(query_times)
            std_time = statistics.stdev(query_times) if len(query_times) > 1 else 0
            all_times.extend(query_times)
            
            print(f"Query: {query[:50]}...")
            print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
    
    if all_times:
        overall_avg = statistics.mean(all_times)
        overall_std = statistics.stdev(all_times) if len(all_times) > 1 else 0
        print(f"\nOverall Performance:")
        print(f"  Average: {overall_avg:.2f}ms ± {overall_std:.2f}ms")
        print(f"  Min: {min(all_times):.2f}ms")
        print(f"  Max: {max(all_times):.2f}ms")

# Run benchmark
test_queries = [
    "Scrape e-commerce product data",
    "Analyze CSV with statistical insights", 
    "Automate email notifications",
    "Execute Python code securely",
    "Search Wikipedia for information"
]

engine = RAGMCPEngine()
await engine.initialize()
await benchmark_pipeline(engine, test_queries)
```

---

## Troubleshooting

### Common Issues

#### 1. Initialization Failures

**Problem**: Engine fails to initialize

```python
success = await engine.initialize()
if not success:
    print("Initialization failed!")
```

**Solutions**:
- Check OpenRouter API key: `echo $OPENROUTER_API_KEY`
- Verify network connectivity to OpenRouter
- Check cache directory permissions
- Enable debug logging for detailed error info

#### 2. Embedding Generation Failures

**Problem**: Vector embeddings fail to generate

**Solutions**:
- Engine automatically falls back to mock embeddings if OpenRouter fails
- Check OpenRouter API limits and billing status
- Verify internet connectivity

#### 3. Performance Issues

**Problem**: Slow query processing

**Solutions**:
- Install FAISS for faster vector search: `pip install faiss-cpu`
- Increase similarity threshold to reduce candidates
- Disable LLM validation for testing: `enable_llm_validation=False`
- Use persistent caching

### Debug Mode

Enable comprehensive debugging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create engine with verbose logging
engine = RAGMCPEngine()

# Check pipeline metadata for debugging info
result = await engine.execute_rag_mcp_pipeline("test query")
if result.pipeline_metadata:
    print("Pipeline timing:", result.pipeline_metadata.get('timing'))
    print("Search stats:", result.pipeline_metadata.get('search_stats'))
```

---

## Advanced Usage

### Custom Validation Logic

Implement custom MCP validation:

```python
from alita_core.rag_mcp_engine import RAGMCPValidator

class CustomRAGMCPValidator(RAGMCPValidator):
    async def _calculate_relevance_score(self, mcp_spec, query, similarity_score):
        # Custom relevance calculation
        base_score = await super()._calculate_relevance_score(mcp_spec, query, similarity_score)
        
        # Add custom business logic
        if "enterprise" in query.lower() and mcp_spec.reliability_score > 0.9:
            base_score *= 1.2  # Boost enterprise-grade tools
        
        return min(base_score, 1.0)

# Use custom validator
engine = RAGMCPEngine()
engine.validator = CustomRAGMCPValidator(llm_client=engine.llm_client)
```

### Batch Processing

Process multiple queries efficiently:

```python
async def batch_process_queries(engine, queries, batch_size=5):
    """Process multiple queries in batches"""
    
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        
        # Process batch concurrently
        batch_tasks = [
            engine.execute_rag_mcp_pipeline(query)
            for query in batch
        ]
        
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(queries) + batch_size - 1)//batch_size}")
    
    return results
```

---

## Development & Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd alita-kgot-enhanced

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio black isort mypy
```

### Running Tests

```python
# Test file: tests/test_rag_mcp_engine.py
import pytest
import asyncio
from alita_core.rag_mcp_engine import RAGMCPEngine

@pytest.mark.asyncio
async def test_engine_initialization():
    engine = RAGMCPEngine(enable_llm_validation=False)  # Disable for testing
    success = await engine.initialize()
    assert success, "Engine should initialize successfully"

@pytest.mark.asyncio 
async def test_pipeline_execution():
    engine = RAGMCPEngine(enable_llm_validation=False)
    await engine.initialize()
    
    result = await engine.execute_rag_mcp_pipeline("test query")
    assert result.success, "Pipeline should execute successfully"
    assert len(result.retrieved_mcps) > 0, "Should retrieve at least one MCP"

# Run tests: pytest tests/test_rag_mcp_engine.py -v
```

### Contributing Guidelines

1. **Code Quality**: All code must pass `black`, `isort`, and `mypy` checks
2. **Testing**: All new features must include comprehensive tests
3. **Documentation**: Update this documentation for any API changes
4. **Performance**: Ensure changes don't degrade performance significantly
5. **Logging**: Add appropriate logging for all new functionality

---

## Conclusion

The RAG-MCP Engine provides a comprehensive, production-ready implementation of the RAG-MCP framework for intelligent MCP discovery and selection. With its three-step pipeline, LangChain integration, and seamless Alita system integration, it represents the state-of-the-art in automated MCP recommendation systems.

**Version**: 1.0.0  
**Last Updated**: 2024  
**Authors**: Alita-KGoT Enhanced Development Team 