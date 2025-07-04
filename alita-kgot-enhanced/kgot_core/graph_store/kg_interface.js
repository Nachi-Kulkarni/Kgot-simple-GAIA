/**
 * Knowledge Graph Interface - Abstract Base Class
 * 
 * Based on KGoT research paper Section 2.1 - Knowledge Graph Store Module
 * This interface defines the contract for all knowledge graph implementations
 * supporting triplet structure (subject, predicate, object) and multimodal data
 * 
 * Supported Backends:
 * - Neo4j (Production) with Cypher queries
 * - NetworkX (Development) with Python-like execution
 * - RDF4J (Research) with SPARQL queries
 * 
 * Features:
 * - Graph initialization and state management
 * - Triplet-based knowledge representation
 * - Query interface (Cypher/SPARQL/NetworkX)
 * - Persistence and serialization
 * - MCP validation metrics integration
 * - Snapshot functionality for debugging
 * 
 * @module KnowledgeGraphInterface
 */

const EventEmitter = require('events');
const { loggers } = require('../../config/logging/winston_config');

/**
 * Abstract Knowledge Graph Interface
 * Defines the contract for all graph store implementations
 */
class KnowledgeGraphInterface extends EventEmitter {
  /**
   * Initialize the Knowledge Graph Interface
   * @param {Object} options - Configuration options
   * @param {string} options.backend - Backend type (neo4j, networkx, rdf4j)
   * @param {string} options.loggerName - Logger identifier
   * @param {Object} options.connectionConfig - Backend-specific connection config
   */
  constructor(options = {}) {
    super();
    
    this.options = {
      backend: options.backend || 'networkx',
      loggerName: options.loggerName || 'KnowledgeGraph',
      snapshotEnabled: options.snapshotEnabled !== false,
      mcpValidation: options.mcpValidation !== false,
      maxRetries: options.maxRetries || 3,
      timeout: options.timeout || 30000,
      ...options
    };

    // Initialize logger
    this.logger = loggers[this.options.loggerName] || loggers.kgotController;
    
    // Graph state tracking
    this.currentSnapshotId = 0;
    this.currentFolderName = '';
    this.isInitialized = false;
    this.connectionStatus = 'disconnected';
    
    // MCP validation metrics tracking
    this.mcpMetrics = {
      totalNodes: 0,
      totalEdges: 0,
      validationErrors: 0,
      lastValidation: null,
      confidenceScores: []
    };

    this.logger.info('Knowledge Graph Interface initialized', {
      operation: 'KG_INTERFACE_INIT',
      backend: this.options.backend,
      options: this.options
    });
  }

  /**
   * Initialize the database/graph store
   * Must be implemented by concrete classes
   * 
   * @param {number} snapshotIndex - Index for snapshot organization
   * @param {string} snapshotSubdir - Subdirectory for snapshots
   * @param {...*} args - Additional backend-specific arguments
   * @returns {Promise<void>}
   */
  async initDatabase(snapshotIndex = 0, snapshotSubdir = '', ...args) {
    throw new Error('initDatabase must be implemented by concrete classes');
  }

  /**
   * Get current graph state as string representation
   * Returns all nodes and relationships in a readable format
   * 
   * @param {...*} args - Backend-specific arguments
   * @returns {Promise<string>} Current graph state
   */
  async getCurrentGraphState(...args) {
    throw new Error('getCurrentGraphState must be implemented by concrete classes');
  }

  /**
   * Execute a read query on the graph
   * Supports Cypher (Neo4j), SPARQL (RDF4J), or Python code (NetworkX)
   * 
   * @param {string} query - Query string in appropriate language
   * @param {...*} args - Additional query parameters
   * @returns {Promise<{result: *, success: boolean, error: Error|null}>}
   */
  async executeQuery(query, ...args) {
    throw new Error('executeQuery must be implemented by concrete classes');
  }

  /**
   * Execute a write query on the graph
   * Modifies graph structure or content
   * 
   * @param {string} query - Write query string
   * @param {...*} args - Additional query parameters
   * @returns {Promise<{success: boolean, error: Error|null}>}
   */
  async executeWriteQuery(query, ...args) {
    throw new Error('executeWriteQuery must be implemented by concrete classes');
  }

  /**
   * Execute multiple read queries in batch
   * Optimized for bulk operations
   * 
   * @param {string[]} queries - Array of query strings
   * @param {...*} args - Additional parameters
   * @returns {Promise<Array<{result: *, success: boolean, error: Error|null}>>}
   */
  async executeBatchQueries(queries, ...args) {
    if (!Array.isArray(queries)) {
      queries = [queries];
    }

    const results = [];
    for (const query of queries) {
      const result = await this.executeQuery(query, ...args);
      results.push(result);
    }

    return results;
  }

  /**
   * Execute multiple write queries in batch
   * Optimized for bulk modifications
   * 
   * @param {string[]} queries - Array of write query strings
   * @param {...*} args - Additional parameters
   * @returns {Promise<Array<{success: boolean, error: Error|null}>>}
   */
  async executeBatchWriteQueries(queries, ...args) {
    if (!Array.isArray(queries)) {
      queries = [queries];
    }

    const results = [];
    for (const query of queries) {
      const result = await this.executeWriteQuery(query, ...args);
      results.push(result);
    }

    return results;
  }

  /**
   * Add a triplet (subject, predicate, object) to the knowledge graph
   * Core method for knowledge representation
   * 
   * @param {Object} triplet - The triplet to add
   * @param {string} triplet.subject - Subject entity
   * @param {string} triplet.predicate - Relationship/property
   * @param {string|Object} triplet.object - Object entity or value
   * @param {Object} triplet.metadata - Additional metadata
   * @returns {Promise<{success: boolean, error: Error|null}>}
   */
  async addTriplet(triplet) {
    this.logger.info('Adding triplet to knowledge graph', {
      operation: 'KG_ADD_TRIPLET',
      triplet,
      backend: this.options.backend
    });

    // Update MCP metrics
    if (this.options.mcpValidation) {
      this.mcpMetrics.totalEdges++;
      this.emit('triplet:added', triplet);
    }

    // Must be implemented by concrete classes
    throw new Error('addTriplet must be implemented by concrete classes');
  }

  /**
   * Add an entity node to the knowledge graph
   * 
   * @param {Object} entity - Entity to add
   * @param {string} entity.id - Unique identifier
   * @param {string} entity.type - Entity type/label
   * @param {Object} entity.properties - Entity properties
   * @param {Object} entity.metadata - Additional metadata
   * @returns {Promise<{success: boolean, error: Error|null}>}
   */
  async addEntity(entity) {
    this.logger.info('Adding entity to knowledge graph', {
      operation: 'KG_ADD_ENTITY',
      entity,
      backend: this.options.backend
    });

    // Update MCP metrics
    if (this.options.mcpValidation) {
      this.mcpMetrics.totalNodes++;
      this.emit('entity:added', entity);
    }

    // Must be implemented by concrete classes
    throw new Error('addEntity must be implemented by concrete classes');
  }

  /**
   * Query entities by type or properties
   * 
   * @param {Object} criteria - Query criteria
   * @param {string} criteria.type - Entity type to filter
   * @param {Object} criteria.properties - Properties to match
   * @param {number} criteria.limit - Maximum results
   * @returns {Promise<{result: Array, success: boolean, error: Error|null}>}
   */
  async queryEntities(criteria = {}) {
    // Must be implemented by concrete classes
    throw new Error('queryEntities must be implemented by concrete classes');
  }

  /**
   * Query relationships between entities
   * 
   * @param {Object} criteria - Query criteria
   * @param {string} criteria.subject - Source entity
   * @param {string} criteria.predicate - Relationship type
   * @param {string} criteria.object - Target entity
   * @returns {Promise<{result: Array, success: boolean, error: Error|null}>}
   */
  async queryRelationships(criteria = {}) {
    // Must be implemented by concrete classes
    throw new Error('queryRelationships must be implemented by concrete classes');
  }

  /**
   * Export current graph state to snapshot
   * For debugging and visualization
   * 
   * @returns {Promise<{success: boolean, snapshotPath: string, error: Error|null}>}
   */
  async exportSnapshot() {
    if (!this.options.snapshotEnabled) {
      return { success: false, snapshotPath: null, error: new Error('Snapshots disabled') };
    }

    try {
      const snapshotPath = await this._exportDatabase();
      
      this.logger.info('Graph snapshot exported', {
        operation: 'KG_SNAPSHOT_EXPORT',
        snapshotPath,
        snapshotId: this.currentSnapshotId,
        metrics: this.mcpMetrics
      });

      this.emit('snapshot:exported', { path: snapshotPath, id: this.currentSnapshotId });
      return { success: true, snapshotPath, error: null };

    } catch (error) {
      this.logger.error('Failed to export graph snapshot', {
        operation: 'KG_SNAPSHOT_EXPORT_FAILED',
        error: error.message,
        stack: error.stack
      });

      return { success: false, snapshotPath: null, error };
    }
  }

  /**
   * Validate graph structure and content
   * Integrates with MCP validation metrics
   * 
   * @returns {Promise<{isValid: boolean, errors: Array, metrics: Object}>}
   */
  async validateGraph() {
    const validationStart = Date.now();
    
    try {
      this.logger.info('Starting graph validation', {
        operation: 'KG_VALIDATION_START',
        metrics: this.mcpMetrics
      });

      const errors = [];
      
      // Basic structure validation
      const graphState = await this.getCurrentGraphState();
      if (!graphState || graphState.trim().length === 0) {
        errors.push('Graph is empty or unreadable');
      }

      // Update MCP metrics
      if (this.options.mcpValidation) {
        this.mcpMetrics.lastValidation = new Date();
        this.mcpMetrics.validationErrors = errors.length;
        
        // Calculate confidence score based on validation
        const confidence = Math.max(0, 1 - (errors.length * 0.1));
        this.mcpMetrics.confidenceScores.push(confidence);
        
        // Keep only last 10 confidence scores
        if (this.mcpMetrics.confidenceScores.length > 10) {
          this.mcpMetrics.confidenceScores = this.mcpMetrics.confidenceScores.slice(-10);
        }
      }

      const validationTime = Date.now() - validationStart;
      const isValid = errors.length === 0;

      this.logger.info('Graph validation completed', {
        operation: 'KG_VALIDATION_COMPLETE',
        isValid,
        errorCount: errors.length,
        validationTime,
        metrics: this.mcpMetrics
      });

      this.emit('graph:validated', { isValid, errors, metrics: this.mcpMetrics });

      return {
        isValid,
        errors,
        metrics: {
          ...this.mcpMetrics,
          validationTime
        }
      };

    } catch (error) {
      this.logger.error('Graph validation failed', {
        operation: 'KG_VALIDATION_FAILED',
        error: error.message,
        stack: error.stack
      });

      return {
        isValid: false,
        errors: [`Validation failed: ${error.message}`],
        metrics: this.mcpMetrics
      };
    }
  }

  /**
   * Get MCP validation metrics
   * 
   * @returns {Object} Current MCP metrics
   */
  getMCPMetrics() {
    return { ...this.mcpMetrics };
  }

  /**
   * Reset MCP validation metrics
   */
  resetMCPMetrics() {
    this.mcpMetrics = {
      totalNodes: 0,
      totalEdges: 0,
      validationErrors: 0,
      lastValidation: null,
      confidenceScores: []
    };

    this.logger.info('MCP metrics reset', {
      operation: 'KG_MCP_METRICS_RESET'
    });
  }

  /**
   * Test connection to the graph backend
   * 
   * @returns {Promise<{connected: boolean, error: Error|null}>}
   */
  async testConnection() {
    // Must be implemented by concrete classes
    throw new Error('testConnection must be implemented by concrete classes');
  }

  /**
   * Close connection to the graph backend
   * 
   * @returns {Promise<void>}
   */
  async close() {
    this.connectionStatus = 'disconnected';
    this.isInitialized = false;
    
    this.logger.info('Knowledge Graph connection closed', {
      operation: 'KG_CONNECTION_CLOSED',
      backend: this.options.backend
    });

    this.emit('connection:closed');
  }

  // Protected methods for concrete implementations

  /**
   * Export database to snapshot (implementation-specific)
   * Must be implemented by concrete classes
   * 
   * @protected
   * @returns {Promise<string>} Snapshot file path
   */
  async _exportDatabase() {
    throw new Error('_exportDatabase must be implemented by concrete classes');
  }

  /**
   * Create snapshot folder structure
   * 
   * @protected
   * @param {number} index - Snapshot index
   * @param {string} subdir - Subdirectory name
   * @returns {string} Folder path
   */
  _createSnapshotFolder(index, subdir = '') {
    const fs = require('fs');
    const path = require('path');

    let folderName = '';
    if (subdir !== '') {
      folderName = `${subdir}/`;
    }
    folderName += `snapshot_${index}`;
    this.currentFolderName = folderName;

    const folderPath = path.join('./alita-kgot-enhanced/kgot_core/graph_store/_snapshots', folderName);
    
    if (!fs.existsSync(folderPath)) {
      fs.mkdirSync(folderPath, { recursive: true });
    }

    return folderPath;
  }

  /**
   * Generate unique identifier
   * 
   * @protected
   * @returns {string} Unique identifier
   */
  _generateId() {
    return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

module.exports = KnowledgeGraphInterface; 