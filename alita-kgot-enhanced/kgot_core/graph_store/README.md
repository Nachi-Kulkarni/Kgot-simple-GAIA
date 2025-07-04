# KGoT Graph Store Module

Complete Knowledge Graph storage implementation based on the KGoT research paper Section 2.1.

## Overview

The KGoT Graph Store Module provides the knowledge representation layer for the Alita AI system. It implements a flexible, multi-backend graph storage system that supports triplet-based knowledge representation (subject, predicate, object) with integrated MCP validation metrics.

## Architecture

### Core Components

1. **Knowledge Graph Interface** (`kg_interface.js`)
   - Abstract base class defining the contract for all implementations
   - Event-driven architecture for integration with KGoT Controller
   - MCP validation metrics tracking
   - Snapshot and persistence capabilities

2. **NetworkX Implementation** (`networkx_implementation.js`)
   - In-memory graph storage using JavaScript Maps and Sets
   - Lightweight development and testing backend
   - JavaScript-based query execution (similar to Python exec)
   - Fast operations without external dependencies

3. **Neo4j Implementation** (`neo4j_implementation.js`)
   - Production-ready graph database with Cypher query support
   - ACID transactions and enterprise scalability
   - Connection pooling and error handling
   - APOC export functionality for snapshots

4. **Knowledge Graph Factory** (`knowledge_graph_factory.js`)
   - Factory pattern for backend management
   - Automatic backend selection and fallback
   - Environment-aware configuration
   - Dependency validation

## Features

### Knowledge Representation
- **Triplet Structure**: (subject, predicate, object) knowledge representation
- **Entity Management**: Add, query, and update entities with typed properties
- **Relationship Handling**: Complex relationship management between entities
- **Metadata Support**: Rich metadata and properties for entities and relationships

### Backend Support
- **NetworkX**: In-memory backend for development and testing
- **Neo4j**: Production Cypher-based backend
- **RDF4J**: SPARQL-compatible backend (planned)

### Integration Features
- **MCP Validation Metrics**: Integrated confidence scoring and validation tracking
- **Event System**: Event-driven architecture for real-time integration
- **Snapshot Export**: JSON export for debugging and visualization
- **Query Execution**: Backend-specific query languages (JavaScript, Cypher, SPARQL)

### Logging and Monitoring
- **Winston Integration**: Comprehensive logging with configurable levels
- **Metrics Tracking**: Performance and usage metrics
- **Error Handling**: Robust error handling and recovery

## Installation

### Basic Setup
```bash
# Install required dependencies
npm install winston

# For Neo4j backend support
npm install neo4j-driver
```

### Environment Configuration
```bash
# Neo4j configuration (optional)
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

# Logging level
export LOG_LEVEL=info
```

## Usage

### Quick Start

```javascript
const { createDevelopmentGraph } = require('./alita-kgot-enhanced/kgot_core/graph_store');

async function example() {
  // Create a development graph
  const graph = await createDevelopmentGraph();
  
  // Initialize the database
  await graph.initDatabase();
  
  // Add entities
  await graph.addEntity({
    id: 'user_001',
    type: 'User',
    properties: { name: 'Alice', role: 'researcher' }
  });
  
  await graph.addEntity({
    id: 'project_001',
    type: 'Project',
    properties: { name: 'AI Research', status: 'active' }
  });
  
  // Add relationship triplet
  await graph.addTriplet({
    subject: 'user_001',
    predicate: 'worksOn',
    object: 'project_001',
    metadata: { role: 'lead', startDate: '2024-01-01' }
  });
  
  // Query entities
  const users = await graph.queryEntities({ type: 'User' });
  console.log('Users:', users.result);
  
  // Query relationships
  const relationships = await graph.queryRelationships({ subject: 'user_001' });
  console.log('User relationships:', relationships.result);
  
  // Export snapshot
  const snapshotPath = await graph.exportSnapshot();
  console.log('Snapshot saved to:', snapshotPath);
  
  // Close the graph
  await graph.close();
}
```

### Factory Usage

```javascript
const { KnowledgeGraphFactory } = require('./alita-kgot-enhanced/kgot_core/graph_store');

async function factoryExample() {
  const factory = new KnowledgeGraphFactory();
  
  // Check available backends
  const backends = factory.getAvailableBackends();
  console.log('Available backends:', backends);
  
  // Get recommendation
  const recommended = factory.getRecommendedBackend('production');
  console.log('Recommended for production:', recommended);
  
  // Create specific backend
  const graph = await factory.createKnowledgeGraph('networkx', {
    logLevel: 'debug'
  });
}
```

### Neo4j Production Usage

```javascript
const { createProductionGraph } = require('./alita-kgot-enhanced/kgot_core/graph_store');

async function productionExample() {
  const graph = await createProductionGraph({
    uri: 'bolt://localhost:7687',
    user: 'neo4j',
    password: 'password'
  });
  
  await graph.initDatabase();
  
  // Execute Cypher query
  const result = await graph.executeQuery(
    'MATCH (n:User) RETURN n.name as name, n.role as role'
  );
  
  console.log('Users from Neo4j:', result.result);
}
```

### Custom Queries

#### NetworkX JavaScript Queries
```javascript
// Find connected entities
const query = `
  const userId = 'user_001';
  const connected = [];
  
  for (const [key, edge] of edges) {
    if (edge.source === userId) {
      const target = getNode(edge.target);
      connected.push({ id: edge.target, entity: target, relationship: edge.relationship });
    }
  }
  
  result = connected;
`;

const result = await graph.executeQuery(query);
```

#### Neo4j Cypher Queries
```javascript
// Find user projects and tasks
const cypherQuery = `
  MATCH (u:User {id: $userId})-[r1]->(p:Project)-[r2]->(t:Task)
  RETURN u.name as user, p.name as project, t.title as task, 
         type(r1) as userRelation, type(r2) as projectRelation
`;

const result = await graph.executeQuery(cypherQuery, { userId: 'user_001' });
```

## API Reference

### KnowledgeGraphInterface

Base abstract class that defines the interface for all implementations.

#### Core Methods

```javascript
// Database management
await graph.initDatabase(snapshotIndex, snapshotSubdir)
await graph.close()
await graph.testConnection()

// Entity operations
await graph.addEntity({ id, type, properties, metadata })
await graph.queryEntities({ type, properties, limit })

// Triplet operations
await graph.addTriplet({ subject, predicate, object, metadata })
await graph.queryRelationships({ subject, predicate, object })

// Query execution
await graph.executeQuery(query, params)
await graph.executeWriteQuery(query, params)

// State and export
await graph.getCurrentGraphState()
await graph.exportSnapshot()

// MCP metrics
graph.getMCPMetrics()
graph.updateMCPMetrics(updates)
```

#### Events

```javascript
graph.on('database:initialized', () => console.log('Database ready'));
graph.on('triplet:added', (triplet) => console.log('New triplet:', triplet));
graph.on('entity:added', (entity) => console.log('New entity:', entity));
graph.on('graph:modified', () => console.log('Graph changed'));
```

### Factory Methods

```javascript
const factory = new KnowledgeGraphFactory(config);

// Create graphs
await factory.createKnowledgeGraph(backend, options)
await factory.createProductionKnowledgeGraph(options)
await factory.createDevelopmentKnowledgeGraph(options)

// Information
factory.getAvailableBackends()
factory.isBackendAvailable(backendType)
factory.getRecommendedBackend(useCase)
```

### Convenience Functions

```javascript
// Quick creation
await createKnowledgeGraph(backend, options)
await createProductionGraph(options)
await createDevelopmentGraph(options)

// Information
getAvailableBackends()
getRecommendedBackend(useCase)
```

## Implementation Status

✅ **COMPLETE** - The KGoT Graph Store Module has been fully implemented and tested as per Task 3 of the 5-Phase Implementation Plan.

### Current Status
- **Implementation**: 100% complete based on KGoT research paper Section 2.1
- **Testing**: All 13 integration tests passing successfully
- **Backends**: NetworkX (ready), Neo4j (ready, requires neo4j-driver)
- **Integration**: Ready for KGoT Controller integration
- **Documentation**: Complete with examples and troubleshooting

### Test Results Summary
```bash
# Latest test run (2025-06-27)
✅ Backend availability and selection: PASSED
✅ Development graph creation: PASSED
✅ Database initialization: PASSED
✅ Entity management (3 entities): PASSED
✅ Triplet management (3 relationships): PASSED
✅ Entity querying: PASSED
✅ Relationship querying: PASSED
✅ Custom JavaScript queries: PASSED
✅ Graph state retrieval: PASSED
✅ MCP metrics tracking: PASSED
✅ Snapshot export functionality: PASSED
✅ Connection testing: PASSED
✅ Resource cleanup: PASSED

Total: 13/13 tests passed
```

## Testing

Run the integration tests to verify the implementation:

```bash
# Navigate to graph store directory
cd alita-kgot-enhanced/kgot_core/graph_store

# Run integration tests
node test_integration.js
```

The test suite covers:
- Backend availability and selection
- Graph initialization and configuration
- Entity and triplet management
- Querying and relationship traversal
- Custom query execution
- Snapshot export functionality
- MCP metrics integration
- Connection testing and cleanup

## File Structure

```
alita-kgot-enhanced/kgot_core/graph_store/
├── README.md                    # This documentation
├── index.js                     # Main module entry point
├── kg_interface.js              # Abstract base interface
├── networkx_implementation.js   # NetworkX backend
├── neo4j_implementation.js      # Neo4j backend
├── knowledge_graph_factory.js   # Factory for backend management
├── test_integration.js          # Integration test suite
└── _snapshots/                  # Snapshot storage directory
```

## Integration with KGoT Controller

The Graph Store Module is designed to integrate seamlessly with the KGoT Controller's iterative graph construction process:

```javascript
// Example KGoT Controller integration
const { createProductionGraph } = require('./graph_store');

class KGoTController {
  async initialize() {
    this.knowledgeGraph = await createProductionGraph();
    
    // Listen for graph events
    this.knowledgeGraph.on('graph:modified', () => {
      this.onGraphModified();
    });
    
    await this.knowledgeGraph.initDatabase();
  }
  
  async addKnowledge(subject, predicate, object, confidence) {
    await this.knowledgeGraph.addTriplet({
      subject, predicate, object,
      metadata: { confidence, timestamp: new Date() }
    });
    
    // Update MCP metrics
    this.knowledgeGraph.updateMCPMetrics({
      confidenceScores: [confidence]
    });
  }
}
```

## Performance Considerations

### NetworkX Backend
- **Memory Usage**: Stores entire graph in memory - suitable for development/testing
- **Query Performance**: Fast for small to medium graphs (< 100K nodes)
- **Persistence**: Requires periodic snapshots for data persistence

### Neo4j Backend
- **Scalability**: Handles large graphs (millions of nodes/relationships)
- **Query Performance**: Optimized Cypher execution with indexes
- **Persistence**: ACID transactions with automatic persistence
- **Memory**: More efficient memory usage for large datasets

## Configuration

### Environment Variables
```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Logging
LOG_LEVEL=info

# Development settings
NODE_ENV=development
```

### Factory Configuration
```javascript
const factory = new KnowledgeGraphFactory({
  defaultBackend: 'networkx',
  enableFallback: true,
  loggerConfig: {
    level: 'debug',
    format: winston.format.simple()
  }
});
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Fails**
   - Check if Neo4j is running: `docker ps` or service status
   - Verify connection parameters in environment variables
   - Test with NetworkX fallback for development

2. **Missing Dependencies**
   - Install required packages: `npm install neo4j-driver winston`
   - Check module resolution paths

3. **Snapshot Directory Issues**
   - Ensure write permissions for `./alita-kgot-enhanced/logs/` directory
   - Check disk space for snapshot exports

4. **Memory Issues with NetworkX**
   - Monitor graph size and node/edge count
   - Consider Neo4j backend for larger datasets
   - Implement periodic cleanup for development

### Logging

Enable debug logging for troubleshooting:
```bash
export LOG_LEVEL=debug
node your_application.js
```

Log files are stored in:
- Graph Store: `./alita-kgot-enhanced/logs/kgot/graph_store.log`
- Factory: `./alita-kgot-enhanced/logs/kgot/kg_factory.log`

## Future Enhancements

1. **RDF4J Backend**: SPARQL-compatible implementation for semantic web integration
2. **Redis Backend**: Distributed in-memory backend for horizontal scaling
3. **GraphQL Interface**: Modern query interface for web applications
4. **Batch Operations**: Optimized bulk insert and update operations
5. **Schema Validation**: Entity and relationship schema enforcement
6. **Full-text Search**: Integration with Elasticsearch for content search
7. **Graph Analytics**: Built-in graph analysis and centrality metrics

## Contributing

1. Follow JSDoc3 commenting standards
2. Use Winston for all logging operations
3. Include integration tests for new features
4. Maintain backward compatibility with existing interfaces
5. Document all public APIs with examples

## License

Part of the Alita Enhanced KGoT implementation based on the Knowledge Graph of Thoughts research paper. 