/**
 * Neo4j Implementation for Knowledge Graph Store
 * 
 * Based on KGoT research paper Neo4j backend implementation
 * Provides production-ready graph database storage with Cypher query language
 * Supports ACID transactions, scalability, and enterprise features
 * 
 * Features:
 * - Neo4j graph database integration
 * - Cypher query language support
 * - Transaction management
 * - APOC export functionality for snapshots
 * - Connection pooling and error handling
 * - Triplet-based knowledge representation
 * - MCP validation metrics integration
 * 
 * @module Neo4jImplementation
 */

const KnowledgeGraphInterface = require('./kg_interface');
const neo4j = require('neo4j-driver');
const fs = require('fs').promises;
const path = require('path');

/**
 * Neo4j Knowledge Graph Implementation
 * Provides production-ready graph storage with Cypher queries
 */
class Neo4jKnowledgeGraph extends KnowledgeGraphInterface {
  /**
   * Initialize Neo4j Knowledge Graph
   * @param {Object} options - Configuration options
   * @param {string} options.uri - Neo4j connection URI
   * @param {string} options.user - Neo4j username
   * @param {string} options.password - Neo4j password
   * @param {Object} options.config - Additional Neo4j driver config
   */
  constructor(options = {}) {
    super({
      backend: 'neo4j',
      loggerName: 'Neo4jKG',
      ...options
    });

    // Neo4j connection configuration
    this.connectionConfig = {
      uri: options.uri || process.env.NEO4J_URI || 'bolt://localhost:7687',
      user: options.user || process.env.NEO4J_USER || 'neo4j',
      password: options.password || process.env.NEO4J_PASSWORD || 'password',
      config: {
        maxConnectionLifetime: 3 * 60 * 60 * 1000, // 3 hours
        maxConnectionPoolSize: 50,
        connectionAcquisitionTimeout: 2 * 60 * 1000, // 2 minutes
        disableLosslessIntegers: true,
        ...options.config
      }
    };

    // Neo4j driver and session management
    this.driver = null;
    this.session = null;
    
    // Snapshot directory management
    this.snapshotDir = './alita-kgot-enhanced/kgot_core/graph_store/_snapshots';
    
    this.logger.info('Neo4j Knowledge Graph initialized', {
      operation: 'NEO4J_KG_INIT',
      uri: this.connectionConfig.uri,
      snapshotDir: this.snapshotDir
    });
  }

  /**
   * Initialize the database connection and clear existing data
   * Creates snapshot folder structure
   * 
   * @param {number} snapshotIndex - Index for snapshot organization
   * @param {string} snapshotSubdir - Subdirectory for snapshots
   * @returns {Promise<void>}
   */
  async initDatabase(snapshotIndex = 0, snapshotSubdir = '') {
    try {
      // Initialize Neo4j driver if not already done
      if (!this.driver) {
        await this._initializeDriver();
      }

      // Test connection
      await this._testConnection();

      // Clear all existing data
      const session = this.driver.session();
      try {
        await session.run('MATCH (n) DETACH DELETE n');
        this.logger.info('Cleared all existing Neo4j data');
      } finally {
        await session.close();
      }

      // Reset MCP metrics
      this.resetMCPMetrics();
      
      // Create snapshot folder
      this._createSnapshotFolder(snapshotIndex, snapshotSubdir);
      this.currentSnapshotId = 0;
      this.isInitialized = true;
      this.connectionStatus = 'connected';

      this.logger.info('Neo4j database initialized', {
        operation: 'NEO4J_DB_INIT',
        snapshotIndex,
        snapshotSubdir,
        folderName: this.currentFolderName
      });

      this.emit('database:initialized');

    } catch (error) {
      this.logger.error('Failed to initialize Neo4j database', {
        operation: 'NEO4J_DB_INIT_FAILED',
        error: error.message,
        stack: error.stack
      });
      this.connectionStatus = 'error';
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
      const session = this.driver.session();
      let output = "This is the current state of the Neo4j database.\n";

      try {
        // Get all nodes grouped by labels
        const nodesQuery = `
          MATCH (n)
          WITH labels(n) AS labels, collect({properties: properties(n), id: elementId(n)}) AS nodes
          RETURN {labels: labels, nodes: nodes} AS groupedNodes
        `;
        const nodesResult = await session.run(nodesQuery);
        const nodes = nodesResult.records.map(record => record.get('groupedNodes'));

        // Get all relationships grouped by type
        const relsQuery = `
          MATCH (n)-[r]->(m)
          WITH type(r) as labels, collect({
              properties: properties(r),
              source: labels(n),
              target: labels(m),
              source_id: elementId(n),
              target_id: elementId(m)
          }) as rels
          RETURN {labels: labels, rels: rels} AS groupedRels
        `;
        const relsResult = await session.run(relsQuery);
        const rels = relsResult.records.map(record => record.get('groupedRels'));

        // Format nodes output
        output += "Nodes:\n";
        if (nodes.length === 0) {
          output += "  No nodes found\n";
        } else {
          for (const group of nodes) {
            const label = (group.labels && group.labels.length > 0) ? group.labels[0] : '';
            output += `  Label: ${label}\n`;
            for (const node of group.nodes) {
              const nodeId = node.id.split(':')[2]; // Extract numeric ID
              const properties = node.properties;
              output += `    {neo4j_id:${nodeId}, properties:${JSON.stringify(properties)}}\n`;
            }
          }
        }

        // Format relationships output
        output += "Relationships:\n";
        if (rels.length === 0) {
          output += "  No relationships found\n";
        } else {
          for (const group of rels) {
            const label = group.labels;
            output += `  Label: ${label}\n`;
            for (const rel of group.rels) {
              const sourceLabel = (rel.source && rel.source.length > 0) ? rel.source[0] : '';
              const sourceId = rel.source_id.split(':')[2];
              const targetLabel = (rel.target && rel.target.length > 0) ? rel.target[0] : '';
              const targetId = rel.target_id.split(':')[2];
              const relProperties = rel.properties;
              output += `    {source: {neo4j_id: ${sourceId}, label: ${sourceLabel}}, target: {neo4j_id: ${targetId}, label: ${targetLabel}}, properties: ${JSON.stringify(relProperties)}}\n`;
            }
          }
        }

        this.logger.debug('Neo4j graph state retrieved', {
          operation: 'NEO4J_GET_STATE',
          nodeCount: nodes.reduce((sum, group) => sum + group.nodes.length, 0),
          relationshipCount: rels.reduce((sum, group) => sum + group.rels.length, 0)
        });

        return output;

      } finally {
        await session.close();
      }

    } catch (error) {
      this.logger.error('Failed to get Neo4j graph state', {
        operation: 'NEO4J_GET_STATE_FAILED',
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Execute a read query on the graph
   * Executes Cypher queries for data retrieval
   * 
   * @param {string} query - Cypher query string
   * @param {Object} params - Query parameters
   * @returns {Promise<{result: *, success: boolean, error: Error|null}>}
   */
  async executeQuery(query, params = {}) {
    if (!query || typeof query !== 'string') {
      return { result: null, success: false, error: new Error('Query must be a non-empty string') };
    }

    const session = this.driver.session();
    try {
      this.logger.debug('Executing Neo4j read query', {
        operation: 'NEO4J_EXECUTE_QUERY',
        queryLength: query.length,
        hasParams: Object.keys(params).length > 0
      });

      const result = await session.run(query, params);
      const records = result.records.map(record => {
        const recordObj = {};
        record.keys.forEach(key => {
          recordObj[key] = record.get(key);
        });
        return recordObj;
      });

      this.logger.debug('Neo4j read query executed successfully', {
        operation: 'NEO4J_QUERY_SUCCESS',
        recordCount: records.length
      });

      return { result: records, success: true, error: null };

    } catch (error) {
      this.logger.error('Neo4j read query execution failed', {
        operation: 'NEO4J_QUERY_FAILED',
        error: error.message,
        query: query.substring(0, 200)
      });

      return { result: null, success: false, error };

    } finally {
      await session.close();
    }
  }

  /**
   * Execute a write query on the graph
   * Executes Cypher queries for data modification
   * 
   * @param {string} query - Cypher write query string
   * @param {Object} params - Query parameters
   * @returns {Promise<{success: boolean, error: Error|null}>}
   */
  async executeWriteQuery(query, params = {}) {
    if (!query || typeof query !== 'string') {
      return { success: false, error: new Error('Query must be a non-empty string') };
    }

    const session = this.driver.session();
    try {
      this.logger.debug('Executing Neo4j write query', {
        operation: 'NEO4J_EXECUTE_WRITE_QUERY',
        queryLength: query.length,
        hasParams: Object.keys(params).length > 0
      });

      await session.run(query, params);

      this.logger.debug('Neo4j write query executed successfully', {
        operation: 'NEO4J_WRITE_QUERY_SUCCESS'
      });

      this.emit('graph:modified');
      return { success: true, error: null };

    } catch (error) {
      this.logger.error('Neo4j write query execution failed', {
        operation: 'NEO4J_WRITE_QUERY_FAILED',
        error: error.message,
        query: query.substring(0, 200)
      });

      return { success: false, error };

    } finally {
      await session.close();
    }
  }

  /**
   * Add a triplet (subject, predicate, object) to the knowledge graph
   * Core method for knowledge representation using Cypher
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

      const session = this.driver.session();
      try {
        // Create Cypher query to add the triplet
        const cypherQuery = `
          MERGE (s:Entity {id: $subject})
          ON CREATE SET s.label = $subjectType, s += $subjectProperties, s.timestamp = datetime()
          MERGE (o:Entity {id: $object})
          ON CREATE SET o.label = $objectType, o += $objectProperties, o.timestamp = datetime()
          CREATE (s)-[r:\`${predicate}\`]->(o)
          SET r += $relationshipProperties, r.timestamp = datetime(), r.tripletId = $tripletId
          RETURN s, r, o
        `;

        const params = {
          subject,
          object,
          subjectType: metadata.subjectType || 'Entity',
          objectType: metadata.objectType || 'Entity',
          subjectProperties: metadata.subjectProperties || {},
          objectProperties: metadata.objectProperties || {},
          relationshipProperties: {
            ...metadata,
            timestamp: new Date().toISOString(),
            tripletId: this._generateId()
          }
        };

        await session.run(cypherQuery, params);

        this.mcpMetrics.totalEdges++;
        
        this.logger.info('Triplet added successfully to Neo4j', {
          operation: 'NEO4J_ADD_TRIPLET',
          triplet,
          metrics: this.mcpMetrics
        });

        this.emit('triplet:added', triplet);
        return { success: true, error: null };

      } finally {
        await session.close();
      }

    } catch (error) {
      this.logger.error('Failed to add triplet to Neo4j', {
        operation: 'NEO4J_ADD_TRIPLET_FAILED',
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

      const session = this.driver.session();
      try {
        const cypherQuery = `
          MERGE (e:\`${type}\` {id: $id})
          ON CREATE SET e += $properties, e += $metadata, e.timestamp = datetime(), e.entityId = $entityId
          ON MATCH SET e += $properties, e += $metadata
          RETURN e
        `;

        const params = {
          id,
          properties,
          metadata,
          entityId: this._generateId()
        };

        await session.run(cypherQuery, params);

        this.mcpMetrics.totalNodes++;

        this.logger.info('Entity added successfully to Neo4j', {
          operation: 'NEO4J_ADD_ENTITY',
          entity,
          metrics: this.mcpMetrics
        });

        this.emit('entity:added', entity);
        return { success: true, error: null };

      } finally {
        await session.close();
      }

    } catch (error) {
      this.logger.error('Failed to add entity to Neo4j', {
        operation: 'NEO4J_ADD_ENTITY_FAILED',
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
      const session = this.driver.session();

      try {
        let cypherQuery = 'MATCH (e';
        const params = {};

        // Add type filter
        if (type) {
          cypherQuery += `:${type}`;
        }

        cypherQuery += ')';

        // Add property filters
        if (properties && Object.keys(properties).length > 0) {
          const whereConditions = Object.keys(properties).map((key, index) => {
            params[`prop${index}`] = properties[key];
            return `e.${key} = $prop${index}`;
          });
          cypherQuery += ` WHERE ${whereConditions.join(' AND ')}`;
        }

        cypherQuery += ' RETURN e, elementId(e) as id';

        // Add limit
        if (limit && limit > 0) {
          cypherQuery += ` LIMIT ${parseInt(limit)}`;
        }

        const result = await session.run(cypherQuery, params);
        const entities = result.records.map(record => ({
          id: record.get('id'),
          ...record.get('e').properties
        }));

        this.logger.debug('Neo4j entity query completed', {
          operation: 'NEO4J_QUERY_ENTITIES',
          criteria,
          resultCount: entities.length
        });

        return { result: entities, success: true, error: null };

      } finally {
        await session.close();
      }

    } catch (error) {
      this.logger.error('Neo4j entity query failed', {
        operation: 'NEO4J_QUERY_ENTITIES_FAILED',
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
      const session = this.driver.session();

      try {
        let cypherQuery = 'MATCH (s)-[r';
        const params = {};

        // Add predicate filter
        if (predicate) {
          cypherQuery += `:${predicate}`;
        }

        cypherQuery += ']->(o)';

        // Add filters
        const whereConditions = [];
        if (subject) {
          whereConditions.push('s.id = $subject');
          params.subject = subject;
        }
        if (object) {
          whereConditions.push('o.id = $object');
          params.object = object;
        }

        if (whereConditions.length > 0) {
          cypherQuery += ` WHERE ${whereConditions.join(' AND ')}`;
        }

        cypherQuery += ' RETURN s, r, o, type(r) as relationshipType';

        const result = await session.run(cypherQuery, params);
        const relationships = result.records.map(record => ({
          source: record.get('s').properties.id,
          target: record.get('o').properties.id,
          relationship: record.get('relationshipType'),
          properties: record.get('r').properties
        }));

        this.logger.debug('Neo4j relationship query completed', {
          operation: 'NEO4J_QUERY_RELATIONSHIPS',
          criteria,
          resultCount: relationships.length
        });

        return { result: relationships, success: true, error: null };

      } finally {
        await session.close();
      }

    } catch (error) {
      this.logger.error('Neo4j relationship query failed', {
        operation: 'NEO4J_QUERY_RELATIONSHIPS_FAILED',
        criteria,
        error: error.message
      });

      return { result: [], success: false, error };
    }
  }

  /**
   * Export current graph state to JSON snapshot using APOC
   * 
   * @protected
   * @returns {Promise<string>} Snapshot file path
   */
  async _exportDatabase() {
    const exportFile = `neo4j_snapshot_${this.currentSnapshotId}.json`;
    const exportPath = path.join(this.snapshotDir, this.currentFolderName, exportFile);

    // Ensure directory exists
    const exportDir = path.dirname(exportPath);
    await fs.mkdir(exportDir, { recursive: true });

    const session = this.driver.session();
    try {
      // Try to use APOC export if available, otherwise use manual export
      try {
        const apocQuery = `
          MATCH (n)
          MATCH (k)-[r]->()
          WITH collect(DISTINCT n) as nodes, collect(DISTINCT r) as rels
          CALL apoc.export.json.data(nodes, rels, '${exportFile}', {})
          YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
          RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
        `;
        
        await session.run(apocQuery);
        
      } catch (apocError) {
        // Fallback to manual export if APOC is not available
        this.logger.warn('APOC not available, using manual export', {
          operation: 'NEO4J_EXPORT_FALLBACK',
          error: apocError.message
        });

        // Manual export
        const nodesResult = await session.run('MATCH (n) RETURN n, elementId(n) as id');
        const relsResult = await session.run('MATCH (s)-[r]->(t) RETURN s, r, t, elementId(s) as sid, elementId(t) as tid');

        const exportData = [];

        // Export nodes
        nodesResult.records.forEach(record => {
          const node = record.get('n');
          exportData.push({
            type: 'node',
            id: record.get('id'),
            labels: node.labels,
            properties: node.properties
          });
        });

        // Export relationships
        relsResult.records.forEach(record => {
          const rel = record.get('r');
          exportData.push({
            type: 'relationship',
            label: rel.type,
            properties: rel.properties,
            start: { id: record.get('sid') },
            end: { id: record.get('tid') }
          });
        });

        // Write to file
        const jsonData = exportData.map(item => JSON.stringify(item)).join('\n');
        await fs.writeFile(exportPath, jsonData, 'utf8');
      }

      this.currentSnapshotId++;

      this.logger.info('Neo4j snapshot exported', {
        operation: 'NEO4J_SNAPSHOT_EXPORT',
        exportPath
      });

      return exportPath;

    } finally {
      await session.close();
    }
  }

  /**
   * Test connection to Neo4j database
   * 
   * @returns {Promise<{connected: boolean, error: Error|null}>}
   */
  async testConnection() {
    try {
      if (!this.driver) {
        await this._initializeDriver();
      }

      const session = this.driver.session();
      try {
        await session.run('RETURN 1');
        this.connectionStatus = 'connected';
        
        this.logger.debug('Neo4j connection test successful', {
          operation: 'NEO4J_CONNECTION_TEST',
          connected: true
        });

        return { connected: true, error: null };

      } finally {
        await session.close();
      }

    } catch (error) {
      this.connectionStatus = 'error';
      this.logger.error('Neo4j connection test failed', {
        operation: 'NEO4J_CONNECTION_TEST_FAILED',
        error: error.message
      });

      return { connected: false, error };
    }
  }

  /**
   * Close Neo4j driver connection
   * 
   * @returns {Promise<void>}
   */
  async close() {
    if (this.driver) {
      await this.driver.close();
      this.driver = null;
    }
    
    await super.close();
  }

  // Private methods

  /**
   * Initialize Neo4j driver
   * 
   * @private
   * @returns {Promise<void>}
   */
  async _initializeDriver() {
    try {
      this.driver = neo4j.driver(
        this.connectionConfig.uri,
        neo4j.auth.basic(this.connectionConfig.user, this.connectionConfig.password),
        this.connectionConfig.config
      );

      this.logger.info('Neo4j driver initialized', {
        operation: 'NEO4J_DRIVER_INIT',
        uri: this.connectionConfig.uri
      });

    } catch (error) {
      this.logger.error('Failed to initialize Neo4j driver', {
        operation: 'NEO4J_DRIVER_INIT_FAILED',
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Test Neo4j connection during initialization
   * 
   * @private
   * @returns {Promise<void>}
   */
  async _testConnection() {
    const session = this.driver.session();
    try {
      await session.run('RETURN 1');
      this.logger.info('Neo4j connection test successful');
    } catch (error) {
      this.logger.error('Neo4j connection failed', {
        operation: 'NEO4J_CONNECTION_FAILED',
        error: error.message
      });
      throw new Error(`Failed to connect to Neo4j: ${error.message}`);
    } finally {
      await session.close();
    }
  }
}

module.exports = Neo4jKnowledgeGraph; 