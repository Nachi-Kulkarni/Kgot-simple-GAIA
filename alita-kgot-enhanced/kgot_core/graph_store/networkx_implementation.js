/**
 * NetworkX Implementation for Knowledge Graph Store
 * 
 * Based on KGoT research paper NetworkX backend implementation
 * Provides lightweight, in-memory graph storage for development and testing
 * Uses JavaScript objects to simulate NetworkX DiGraph functionality
 * 
 * Features:
 * - In-memory directed graph representation
 * - JavaScript-based query execution (similar to Python exec)
 * - JSON snapshot export/import
 * - Fast operations without external dependencies
 * - Triplet-based knowledge representation
 * - MCP validation metrics integration
 * 
 * @module NetworkXImplementation
 */

const KnowledgeGraphInterface = require('./kg_interface');
const fs = require('fs').promises;
const path = require('path');

/**
 * NetworkX-style Knowledge Graph Implementation
 * Simulates NetworkX DiGraph using JavaScript Map and Set structures
 */
class NetworkXKnowledgeGraph extends KnowledgeGraphInterface {
  /**
   * Initialize NetworkX Knowledge Graph
   * @param {Object} options - Configuration options
   */
  constructor(options = {}) {
    super({
      backend: 'networkx',
      loggerName: 'NetworkXKG',
      ...options
    });

    // Graph structure using Maps for efficient lookups
    // Similar to NetworkX DiGraph structure
    this.nodes = new Map(); // node_id -> {label, properties}
    this.edges = new Map(); // edge_key -> {source, target, relationship, properties}
    this.adjacencyList = new Map(); // source_id -> Set(target_ids)
    this.reverseAdjacencyList = new Map(); // target_id -> Set(source_ids)
    
    // Snapshot directory management
    this.snapshotDir = './alita-kgot-enhanced/kgot_core/graph_store/_snapshots';
    
    this.logger.info('NetworkX Knowledge Graph initialized', {
      operation: 'NETWORKX_KG_INIT',
      snapshotDir: this.snapshotDir
    });
  }

  /**
   * Initialize the database by clearing all nodes and edges
   * Creates snapshot folder structure
   * 
   * @param {number} snapshotIndex - Index for snapshot organization
   * @param {string} snapshotSubdir - Subdirectory for snapshots
   * @returns {Promise<void>}
   */
  async initDatabase(snapshotIndex = 0, snapshotSubdir = '') {
    try {
      // Clear all graph data
      this.nodes.clear();
      this.edges.clear();
      this.adjacencyList.clear();
      this.reverseAdjacencyList.clear();
      
      // Reset MCP metrics
      this.resetMCPMetrics();
      
      // Create snapshot folder
      this._createSnapshotFolder(snapshotIndex, snapshotSubdir);
      this.currentSnapshotId = 0;
      this.isInitialized = true;
      this.connectionStatus = 'connected';

      this.logger.info('NetworkX database initialized', {
        operation: 'NETWORKX_DB_INIT',
        snapshotIndex,
        snapshotSubdir,
        folderName: this.currentFolderName
      });

      this.emit('database:initialized');

    } catch (error) {
      this.logger.error('Failed to initialize NetworkX database', {
        operation: 'NETWORKX_DB_INIT_FAILED',
        error: error.message,
        stack: error.stack
      });
      throw error;
    }
  }

  /**
   * Get current graph state as formatted string
   * Returns all nodes and relationships in a readable format
   * 
   * @returns {Promise<string>} Current graph state
   */
  async getCurrentGraphState() {
    try {
      let output = "This is the current state of the NetworkX Graph.\n";

      // Group nodes by label
      const nodesByLabel = new Map();
      for (const [nodeId, nodeData] of this.nodes) {
        const label = nodeData.label || 'UnknownLabel';
        if (!nodesByLabel.has(label)) {
          nodesByLabel.set(label, []);
        }
        nodesByLabel.get(label).push({ id: nodeId, ...nodeData });
      }

      // Output nodes
      output += "Existing Nodes:\n";
      if (nodesByLabel.size === 0) {
        output += "\tNo nodes found\n";
      } else {
        for (const [label, nodes] of nodesByLabel) {
          output += `\tLabel: ${label}\n \t\t[`;
          for (const node of nodes) {
            const properties = { ...node };
            delete properties.id;
            delete properties.label;
            output += `{id:${node.id}, properties:${JSON.stringify(properties)}}, `;
          }
          output = output.slice(0, -2); // Remove last comma and space
          output += "]\n";
        }
      }

      // Group edges by relationship
      const edgesByRelationship = new Map();
      for (const [edgeKey, edgeData] of this.edges) {
        const relationship = edgeData.relationship || 'UnknownRelationship';
        if (!edgesByRelationship.has(relationship)) {
          edgesByRelationship.set(relationship, []);
        }
        edgesByRelationship.get(relationship).push(edgeData);
      }

      // Output relationships
      output += "Existing Relationships:\n";
      if (edgesByRelationship.size === 0) {
        output += "\tNo relationships found\n";
      } else {
        for (const [relationship, edges] of edgesByRelationship) {
          output += `\tLabel: ${relationship}\n \t\t[`;
          for (const edge of edges) {
            const properties = { ...edge.properties };
            output += `{source: {id: ${edge.source}}, target: {id: ${edge.target}}, properties: ${JSON.stringify(properties)}}, `;
          }
          output = output.slice(0, -2); // Remove last comma and space
          output += "]\n";
        }
      }

      this.logger.debug('Graph state retrieved', {
        operation: 'NETWORKX_GET_STATE',
        nodeCount: this.nodes.size,
        edgeCount: this.edges.size
      });

      return output;

    } catch (error) {
      this.logger.error('Failed to get graph state', {
        operation: 'NETWORKX_GET_STATE_FAILED',
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Execute a read query on the graph
   * Executes JavaScript code with graph context (similar to Python exec)
   * 
   * @param {string} query - JavaScript code to execute
   * @returns {Promise<{result: *, success: boolean, error: Error|null}>}
   */
  async executeQuery(query) {
    if (!query || typeof query !== 'string') {
      return { result: null, success: false, error: new Error('Query must be a non-empty string') };
    }

    try {
      this.logger.debug('Executing NetworkX query', {
        operation: 'NETWORKX_EXECUTE_QUERY',
        queryLength: query.length
      });

      // Create execution context with graph access
      const context = {
        nodes: this.nodes,
        edges: this.edges,
        adjacencyList: this.adjacencyList,
        reverseAdjacencyList: this.reverseAdjacencyList,
        result: null,
        // Utility functions for graph operations
        getNode: (id) => this.nodes.get(id),
        getEdge: (source, target) => {
          for (const [key, edge] of this.edges) {
            if (edge.source === source && edge.target === target) {
              return edge;
            }
          }
          return null;
        },
        getNodesByLabel: (label) => {
          const result = [];
          for (const [id, node] of this.nodes) {
            if (node.label === label) {
              result.push({ id, ...node });
            }
          }
          return result;
        },
        getEdgesByRelationship: (relationship) => {
          const result = [];
          for (const [key, edge] of this.edges) {
            if (edge.relationship === relationship) {
              result.push(edge);
            }
          }
          return result;
        },
        neighbors: (nodeId) => {
          return Array.from(this.adjacencyList.get(nodeId) || []);
        },
        predecessors: (nodeId) => {
          return Array.from(this.reverseAdjacencyList.get(nodeId) || []);
        }
      };

      // Execute the query code in the context
      const wrappedQuery = `
        (function() {
          const { nodes, edges, adjacencyList, reverseAdjacencyList, getNode, getEdge, 
                  getNodesByLabel, getEdgesByRelationship, neighbors, predecessors } = this;
          let result = null;
          
          ${query}
          
          return result;
        }).call(context)
      `;

      const result = eval(wrappedQuery);

      this.logger.debug('NetworkX query executed successfully', {
        operation: 'NETWORKX_QUERY_SUCCESS',
        hasResult: result !== null
      });

      return { result, success: true, error: null };

    } catch (error) {
      this.logger.error('NetworkX query execution failed', {
        operation: 'NETWORKX_QUERY_FAILED',
        error: error.message,
        query: query.substring(0, 200) // Log first 200 chars of query
      });

      return { result: null, success: false, error };
    }
  }

  /**
   * Execute a write query on the graph
   * Modifies graph structure using JavaScript code execution
   * 
   * @param {string} query - JavaScript code to execute for modification
   * @returns {Promise<{success: boolean, error: Error|null}>}
   */
  async executeWriteQuery(query) {
    if (!query || typeof query !== 'string') {
      return { success: false, error: new Error('Query must be a non-empty string') };
    }

    try {
      this.logger.debug('Executing NetworkX write query', {
        operation: 'NETWORKX_EXECUTE_WRITE_QUERY',
        queryLength: query.length
      });

      // Create write context with modification functions
      const context = {
        nodes: this.nodes,
        edges: this.edges,
        adjacencyList: this.adjacencyList,
        reverseAdjacencyList: this.reverseAdjacencyList,
        // Write operations
        addNode: (id, label, properties = {}) => {
          this.nodes.set(id, { label, ...properties });
          if (!this.adjacencyList.has(id)) {
            this.adjacencyList.set(id, new Set());
          }
          if (!this.reverseAdjacencyList.has(id)) {
            this.reverseAdjacencyList.set(id, new Set());
          }
          this.mcpMetrics.totalNodes++;
        },
        addEdge: (source, target, relationship, properties = {}) => {
          const edgeKey = `${source}->${target}`;
          this.edges.set(edgeKey, { source, target, relationship, properties });
          
          // Update adjacency lists
          if (!this.adjacencyList.has(source)) {
            this.adjacencyList.set(source, new Set());
          }
          if (!this.reverseAdjacencyList.has(target)) {
            this.reverseAdjacencyList.set(target, new Set());
          }
          this.adjacencyList.get(source).add(target);
          this.reverseAdjacencyList.get(target).add(source);
          this.mcpMetrics.totalEdges++;
        },
        removeNode: (id) => {
          this.nodes.delete(id);
          this.adjacencyList.delete(id);
          this.reverseAdjacencyList.delete(id);
          // Remove edges involving this node
          for (const [key, edge] of this.edges) {
            if (edge.source === id || edge.target === id) {
              this.edges.delete(key);
            }
          }
        },
        removeEdge: (source, target) => {
          const edgeKey = `${source}->${target}`;
          this.edges.delete(edgeKey);
          this.adjacencyList.get(source)?.delete(target);
          this.reverseAdjacencyList.get(target)?.delete(source);
        },
        updateNodeProperties: (id, properties) => {
          const node = this.nodes.get(id);
          if (node) {
            this.nodes.set(id, { ...node, ...properties });
          }
        },
        updateEdgeProperties: (source, target, properties) => {
          const edgeKey = `${source}->${target}`;
          const edge = this.edges.get(edgeKey);
          if (edge) {
            this.edges.set(edgeKey, { ...edge, properties: { ...edge.properties, ...properties } });
          }
        }
      };

      // Execute the write query
      const wrappedQuery = `
        (function() {
          const { nodes, edges, adjacencyList, reverseAdjacencyList, addNode, addEdge, 
                  removeNode, removeEdge, updateNodeProperties, updateEdgeProperties } = this;
          
          ${query}
        }).call(context)
      `;

      eval(wrappedQuery);

      this.logger.debug('NetworkX write query executed successfully', {
        operation: 'NETWORKX_WRITE_QUERY_SUCCESS',
        nodeCount: this.nodes.size,
        edgeCount: this.edges.size
      });

      this.emit('graph:modified');
      return { success: true, error: null };

    } catch (error) {
      this.logger.error('NetworkX write query execution failed', {
        operation: 'NETWORKX_WRITE_QUERY_FAILED',
        error: error.message,
        query: query.substring(0, 200)
      });

      return { success: false, error };
    }
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
    try {
      const { subject, predicate, object, metadata = {} } = triplet;
      
      if (!subject || !predicate || !object) {
        throw new Error('Triplet must have subject, predicate, and object');
      }

      // Add subject node if it doesn't exist
      if (!this.nodes.has(subject)) {
        this.nodes.set(subject, { 
          label: metadata.subjectType || 'Entity',
          ...metadata.subjectProperties || {}
        });
        this.adjacencyList.set(subject, new Set());
        this.reverseAdjacencyList.set(subject, new Set());
        this.mcpMetrics.totalNodes++;
      }

      // Add object node if it's an entity (not a literal value)
      if (typeof object === 'string' && !object.startsWith('"')) {
        if (!this.nodes.has(object)) {
          this.nodes.set(object, { 
            label: metadata.objectType || 'Entity',
            ...metadata.objectProperties || {}
          });
          this.adjacencyList.set(object, new Set());
          this.reverseAdjacencyList.set(object, new Set());
          this.mcpMetrics.totalNodes++;
        }
      }

      // Add edge representing the triplet
      const edgeKey = `${subject}-${predicate}->${object}`;
      this.edges.set(edgeKey, {
        source: subject,
        target: object,
        relationship: predicate,
        properties: {
          ...metadata,
          timestamp: new Date(),
          tripletId: this._generateId()
        }
      });

      // Update adjacency lists
      this.adjacencyList.get(subject).add(object);
      if (this.reverseAdjacencyList.has(object)) {
        this.reverseAdjacencyList.get(object).add(subject);
      }
      
      this.mcpMetrics.totalEdges++;

      this.logger.info('Triplet added successfully', {
        operation: 'NETWORKX_ADD_TRIPLET',
        triplet,
        edgeKey,
        metrics: this.mcpMetrics
      });

      this.emit('triplet:added', triplet);
      return { success: true, error: null };

    } catch (error) {
      this.logger.error('Failed to add triplet', {
        operation: 'NETWORKX_ADD_TRIPLET_FAILED',
        triplet,
        error: error.message
      });

      return { success: false, error };
    }
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
    try {
      const { id, type, properties = {}, metadata = {} } = entity;
      
      if (!id || !type) {
        throw new Error('Entity must have id and type');
      }

      this.nodes.set(id, {
        label: type,
        ...properties,
        ...metadata,
        timestamp: new Date(),
        entityId: this._generateId()
      });

      // Initialize adjacency lists if not exists
      if (!this.adjacencyList.has(id)) {
        this.adjacencyList.set(id, new Set());
      }
      if (!this.reverseAdjacencyList.has(id)) {
        this.reverseAdjacencyList.set(id, new Set());
      }

      this.mcpMetrics.totalNodes++;

      this.logger.info('Entity added successfully', {
        operation: 'NETWORKX_ADD_ENTITY',
        entity,
        metrics: this.mcpMetrics
      });

      this.emit('entity:added', entity);
      return { success: true, error: null };

    } catch (error) {
      this.logger.error('Failed to add entity', {
        operation: 'NETWORKX_ADD_ENTITY_FAILED',
        entity,
        error: error.message
      });

      return { success: false, error };
    }
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
    try {
      const { type, properties, limit } = criteria;
      const results = [];

      for (const [id, node] of this.nodes) {
        let matches = true;

        // Check type filter
        if (type && node.label !== type) {
          matches = false;
        }

        // Check property filters
        if (properties && matches) {
          for (const [key, value] of Object.entries(properties)) {
            if (node[key] !== value) {
              matches = false;
              break;
            }
          }
        }

        if (matches) {
          results.push({ id, ...node });
          
          // Apply limit if specified
          if (limit && results.length >= limit) {
            break;
          }
        }
      }

      this.logger.debug('Entity query completed', {
        operation: 'NETWORKX_QUERY_ENTITIES',
        criteria,
        resultCount: results.length
      });

      return { result: results, success: true, error: null };

    } catch (error) {
      this.logger.error('Entity query failed', {
        operation: 'NETWORKX_QUERY_ENTITIES_FAILED',
        criteria,
        error: error.message
      });

      return { result: [], success: false, error };
    }
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
    try {
      const { subject, predicate, object } = criteria;
      const results = [];

      for (const [key, edge] of this.edges) {
        let matches = true;

        // Check subject filter
        if (subject && edge.source !== subject) {
          matches = false;
        }

        // Check predicate filter
        if (predicate && edge.relationship !== predicate) {
          matches = false;
        }

        // Check object filter
        if (object && edge.target !== object) {
          matches = false;
        }

        if (matches) {
          results.push(edge);
        }
      }

      this.logger.debug('Relationship query completed', {
        operation: 'NETWORKX_QUERY_RELATIONSHIPS',
        criteria,
        resultCount: results.length
      });

      return { result: results, success: true, error: null };

    } catch (error) {
      this.logger.error('Relationship query failed', {
        operation: 'NETWORKX_QUERY_RELATIONSHIPS_FAILED',
        criteria,
        error: error.message
      });

      return { result: [], success: false, error };
    }
  }

  /**
   * Export current graph state to JSON snapshot
   * 
   * @protected
   * @returns {Promise<string>} Snapshot file path
   */
  async _exportDatabase() {
    const exportFile = `nx_snapshot_${this.currentSnapshotId}.json`;
    const exportPath = path.join(this.snapshotDir, this.currentFolderName, exportFile);

    // Ensure directory exists
    const exportDir = path.dirname(exportPath);
    await fs.mkdir(exportDir, { recursive: true });

    const data = [];

    // Export nodes
    for (const [id, node] of this.nodes) {
      data.push({
        type: 'node',
        id,
        labels: node.label,
        properties: { ...node }
      });
    }

    // Export edges
    for (const [key, edge] of this.edges) {
      data.push({
        type: 'relationship',
        label: edge.relationship,
        properties: edge.properties,
        start: { id: edge.source },
        end: { id: edge.target }
      });
    }

    // Write to file
    const jsonData = data.map(item => JSON.stringify(item)).join('\n');
    await fs.writeFile(exportPath, jsonData, 'utf8');

    this.currentSnapshotId++;

    this.logger.info('NetworkX snapshot exported', {
      operation: 'NETWORKX_SNAPSHOT_EXPORT',
      exportPath,
      nodeCount: this.nodes.size,
      edgeCount: this.edges.size
    });

    return exportPath;
  }

  /**
   * Test connection to the graph backend
   * 
   * @returns {Promise<{connected: boolean, error: Error|null}>}
   */
  async testConnection() {
    try {
      // NetworkX is in-memory, so connection test is just checking initialization
      const isConnected = this.isInitialized;
      
      this.logger.debug('NetworkX connection test', {
        operation: 'NETWORKX_CONNECTION_TEST',
        connected: isConnected
      });

      return { connected: isConnected, error: null };

    } catch (error) {
      this.logger.error('NetworkX connection test failed', {
        operation: 'NETWORKX_CONNECTION_TEST_FAILED',
        error: error.message
      });

      return { connected: false, error };
    }
  }
}

module.exports = NetworkXKnowledgeGraph;