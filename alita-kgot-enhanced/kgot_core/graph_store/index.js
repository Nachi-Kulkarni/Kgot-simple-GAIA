/**
 * KGoT Graph Store Module - Main Entry Point
 * 
 * Based on KGoT research paper Section 2.1 implementation
 * Provides complete Knowledge Graph storage functionality for Alita AI system
 * 
 * This module implements the graph store layer that manages knowledge representation
 * using triplet structures (subject, predicate, object) and supports multiple backends:
 * - NetworkX: In-memory development backend
 * - Neo4j: Production Cypher-based backend  
 * - RDF4J: SPARQL-compatible research backend (future)
 * 
 * Features:
 * - Abstract Knowledge Graph Interface
 * - Multiple backend implementations
 * - Factory pattern for backend management
 * - MCP validation metrics integration
 * - Snapshot and persistence capabilities
 * - Event-driven architecture for KGoT Controller integration
 * 
 * @module KGoTGraphStore
 * @version 1.0.0
 * @author Alita Enhanced KGoT Team
 */

// Core interfaces and abstractions
const KnowledgeGraphInterface = require('./kg_interface');

// Backend implementations
// Winston logger configuration reference
const winston = require('winston');

/**
 * Initialize the Graph Store Module with logging
 */
const logger = winston.loggers.get('KGoTGraphStore') || winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { component: 'KGoTGraphStore' },
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ 
      filename: './alita-kgot-enhanced/logs/kgot/graph_store.log' 
    })
  ]
});

const NetworkXKnowledgeGraph = require('./networkx_implementation');

// Try to load Neo4j implementation - may fail if neo4j-driver is not installed
let Neo4jKnowledgeGraph = null;
try {
  Neo4jKnowledgeGraph = require('./neo4j_implementation');
} catch (error) {
  logger.warn('Neo4j implementation not available - neo4j-driver dependency missing', {
    operation: 'NEO4J_IMPORT_FAILED',
    error: error.message
  });
}

// Factory and management
const { KnowledgeGraphFactory, KnowledgeGraphFactoryStatic } = require('./knowledge_graph_factory');

logger.info('KGoT Graph Store Module loaded', {
  operation: 'MODULE_INIT',
  version: '1.0.0',
  availableBackends: ['networkx', 'neo4j'],
  implementation: 'based_on_kgot_research_paper'
});

console.log('🔧 [KGoT Graph Store] Module initialization started');
console.log('📊 [KGoT Graph Store] Version: 1.0.0');
console.log('🏗️ [KGoT Graph Store] Implementation: KGoT Research Paper Section 2.1');
console.log('🔌 [KGoT Graph Store] Available backends: NetworkX, Neo4j');
console.log('📁 [KGoT Graph Store] Log file: ./alita-kgot-enhanced/logs/kgot/graph_store.log');
console.log('✅ [KGoT Graph Store] Module initialization completed');

/**
 * Default factory instance for convenience
 */
const defaultFactory = new KnowledgeGraphFactory({
  defaultBackend: 'networkx',
  loggerConfig: {
    level: process.env.LOG_LEVEL || 'info'
  }
});

/**
 * Convenience functions for common use cases
 */

/**
 * Create a knowledge graph instance using the default factory
 * 
 * @param {string} backend - Backend type ('networkx', 'neo4j')
 * @param {Object} options - Implementation-specific options
 * @returns {Promise<KnowledgeGraphInterface>} Knowledge graph instance
 * 
 * @example
 * // Create development graph
 * const devGraph = await createKnowledgeGraph('networkx');
 * 
 * // Create production graph
 * const prodGraph = await createKnowledgeGraph('neo4j', {
 *   uri: 'bolt://localhost:7687',
 *   user: 'neo4j',
 *   password: 'password'
 * });
 */
async function createKnowledgeGraph(backend = 'networkx', options = {}) {
  console.log(`🚀 [KGoT Graph Store] Creating knowledge graph with backend: ${backend}`);
  console.log(`⚙️ [KGoT Graph Store] Options:`, JSON.stringify(options, null, 2));
  
  try {
    const graph = await defaultFactory.createKnowledgeGraph(backend, options);
    console.log(`✅ [KGoT Graph Store] Knowledge graph created successfully with ${backend} backend`);
    return graph;
  } catch (error) {
    console.error(`❌ [KGoT Graph Store] Failed to create knowledge graph with ${backend} backend:`, error.message);
    logger.error('Failed to create knowledge graph', {
      operation: 'CREATE_KG_FAILED',
      backend,
      error: error.message
    });
    throw error;
  }
}

/**
 * Create a production-ready knowledge graph
 * Automatically selects best available production backend
 * 
 * @param {Object} options - Configuration options
 * @returns {Promise<KnowledgeGraphInterface>} Production knowledge graph
 * 
 * @example
 * const productionGraph = await createProductionGraph({
 *   neo4j: {
 *     uri: process.env.NEO4J_URI,
 *     user: process.env.NEO4J_USER,
 *     password: process.env.NEO4J_PASSWORD
 *   }
 * });
 */
async function createProductionGraph(options = {}) {
  console.log('🏭 [KGoT Graph Store] Creating production knowledge graph');
  console.log('🔧 [KGoT Graph Store] Production options:', JSON.stringify(options, null, 2));
  
  try {
    const graph = await defaultFactory.createProductionKnowledgeGraph(options);
    console.log('✅ [KGoT Graph Store] Production knowledge graph created successfully');
    return graph;
  } catch (error) {
    console.error('❌ [KGoT Graph Store] Failed to create production knowledge graph:', error.message);
    logger.error('Failed to create production knowledge graph', {
      operation: 'CREATE_PRODUCTION_KG_FAILED',
      error: error.message
    });
    throw error;
  }
}

/**
 * Create a development knowledge graph optimized for fast iteration
 * 
 * @param {Object} options - Configuration options
 * @returns {Promise<KnowledgeGraphInterface>} Development knowledge graph
 * 
 * @example
 * const devGraph = await createDevelopmentGraph({
 *   logLevel: 'debug',
 *   enableSnapshots: true
 * });
 */
async function createDevelopmentGraph(options = {}) {
  console.log('🛠️ [KGoT Graph Store] Creating development knowledge graph');
  console.log('⚙️ [KGoT Graph Store] Development options:', JSON.stringify(options, null, 2));
  
  try {
    const graph = await defaultFactory.createDevelopmentKnowledgeGraph(options);
    console.log('✅ [KGoT Graph Store] Development knowledge graph created successfully');
    return graph;
  } catch (error) {
    console.error('❌ [KGoT Graph Store] Failed to create development knowledge graph:', error.message);
    logger.error('Failed to create development knowledge graph', {
      operation: 'CREATE_DEVELOPMENT_KG_FAILED',
      error: error.message
    });
    throw error;
  }
}

/**
 * Get information about available backends
 * 
 * @returns {Object} Backend availability and information
 * 
 * @example
 * const backends = getAvailableBackends();
 * console.log(backends);
 * // {
 * //   networkx: { name: 'NetworkX', available: true, production: false },
 * //   neo4j: { name: 'Neo4j', available: false, production: true }
 * // }
 */
function getAvailableBackends() {
  console.log('📋 [KGoT Graph Store] Retrieving available backends');
  const backends = defaultFactory.getAvailableBackends();
  console.log('📊 [KGoT Graph Store] Available backends:', JSON.stringify(backends, null, 2));
  return backends;
}

/**
 * Get recommended backend for specific use case
 * 
 * @param {string} useCase - Use case ('development', 'production', 'testing', 'research')
 * @returns {string} Recommended backend type
 * 
 * @example
 * const prodBackend = getRecommendedBackend('production'); // 'neo4j' or 'networkx'
 * const devBackend = getRecommendedBackend('development'); // 'networkx'
 */
function getRecommendedBackend(useCase) {
  console.log(`🎯 [KGoT Graph Store] Getting recommended backend for use case: ${useCase}`);
  const backend = defaultFactory.getRecommendedBackend(useCase);
  console.log(`💡 [KGoT Graph Store] Recommended backend for ${useCase}: ${backend}`);
  return backend;
}

/**
 * Module exports
 * Provides both low-level components and high-level convenience functions
 */
module.exports = {
  // Core interfaces and implementations
  KnowledgeGraphInterface,
  NetworkXKnowledgeGraph,
  Neo4jKnowledgeGraph,
  
  // Factory and management
  KnowledgeGraphFactory,
  KnowledgeGraphFactoryStatic,
  defaultFactory,
  
  // Convenience functions
  createKnowledgeGraph,
  createProductionGraph,
  createDevelopmentGraph,
  getAvailableBackends,
  getRecommendedBackend,
  
  // Version and metadata
  version: '1.0.0',
  implementation: 'kgot_research_paper_section_2_1',
  supportedBackends: ['networkx', 'neo4j'],
  
  // Logger for module-level operations
  logger
};