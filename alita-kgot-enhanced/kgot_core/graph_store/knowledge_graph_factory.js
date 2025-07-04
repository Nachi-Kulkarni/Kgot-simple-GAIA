/**
 * Knowledge Graph Factory
 * 
 * Factory pattern implementation for managing different knowledge graph backends
 * Provides unified interface for creating and managing graph implementations
 * Based on KGoT research paper architecture with support for multiple backends
 * 
 * Features:
 * - Dynamic backend selection (NetworkX, Neo4j, RDF4J)
 * - Configuration-based instantiation
 * - Environment-aware defaults
 * - Connection validation and fallback
 * - Metrics and logging integration
 * 
 * @module KnowledgeGraphFactory
 */

const winston = require('winston');

// Import available implementations
const NetworkXKnowledgeGraph = require('./networkx_implementation');

// Try to load Neo4j implementation - may fail if neo4j-driver is not installed
let Neo4jKnowledgeGraph = null;
try {
  Neo4jKnowledgeGraph = require('./neo4j_implementation');
} catch (error) {
  // Neo4j driver not available - will be handled in factory
}

/**
 * Knowledge Graph Factory Class
 * Manages creation and lifecycle of different graph implementations
 */
class KnowledgeGraphFactory {
  /**
   * Initialize the factory with configuration
   * @param {Object} config - Factory configuration
   * @param {string} config.defaultBackend - Default backend to use
   * @param {Object} config.loggerConfig - Winston logger configuration
   */
  constructor(config = {}) {
    this.config = {
      defaultBackend: config.defaultBackend || 'networkx',
      loggerConfig: config.loggerConfig || {},
      ...config
    };

    // Initialize logger
    this.logger = winston.loggers.get('KGFactory') || winston.createLogger({
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
      ),
      defaultMeta: { component: 'KnowledgeGraphFactory' },
      transports: [
        new winston.transports.Console(),
        new winston.transports.File({ 
          filename: './alita-kgot-enhanced/logs/kgot/kg_factory.log' 
        })
      ],
      ...this.config.loggerConfig
    });

    // Available backend types
    this.availableBackends = {
      networkx: {
        class: NetworkXKnowledgeGraph,
        name: 'NetworkX',
        description: 'In-memory graph for development and testing',
        production: false,
        dependencies: []
      }
    };

    // Add Neo4j backend if available
    if (Neo4jKnowledgeGraph) {
      this.availableBackends.neo4j = {
        class: Neo4jKnowledgeGraph,
        name: 'Neo4j',
        description: 'Production-ready graph database with Cypher',
        production: true,
        dependencies: ['neo4j-driver']
      };
    }
    
    // Note: RDF4J implementation can be added here when needed
    // if (RDF4jKnowledgeGraph) {
    //   this.availableBackends.rdf4j = {
    //     class: RDF4jKnowledgeGraph,
    //     name: 'RDF4J',
    //     description: 'SPARQL-compatible RDF store',
    //     production: true,
    //     dependencies: ['rdf4j-client']
    //   };
    // }

    this.logger.info('Knowledge Graph Factory initialized', {
      operation: 'KG_FACTORY_INIT',
      defaultBackend: this.config.defaultBackend,
      availableBackends: Object.keys(this.availableBackends)
    });
  }

  /**
   * Create a knowledge graph instance
   * 
   * @param {string|Object} backend - Backend type or configuration object
   * @param {Object} options - Implementation-specific options
   * @returns {Promise<KnowledgeGraphInterface>} Knowledge graph instance
   */
  async createKnowledgeGraph(backend = null, options = {}) {
    try {
      // Determine backend to use
      const backendType = this._resolveBackend(backend);
      const backendConfig = this.availableBackends[backendType];

      if (!backendConfig) {
        throw new Error(`Unknown backend type: ${backendType}`);
      }

      this.logger.info('Creating knowledge graph instance', {
        operation: 'KG_CREATE',
        backend: backendType,
        backendName: backendConfig.name
      });

      // Validate dependencies
      await this._validateDependencies(backendType);

      // Create instance with backend-specific options
      const implementationOptions = {
        ...this._getDefaultOptions(backendType),
        ...options,
        backend: backendType
      };

      const instance = new backendConfig.class(implementationOptions);

      // Test connection if applicable
      if (typeof instance.testConnection === 'function') {
        const connectionResult = await instance.testConnection();
        if (!connectionResult.connected) {
          this.logger.warn('Backend connection failed, consider fallback', {
            operation: 'KG_CONNECTION_WARNING',
            backend: backendType,
            error: connectionResult.error?.message
          });
          
          // Attempt fallback to NetworkX if production backend fails
          if (backendType !== 'networkx' && this.config.enableFallback !== false) {
            this.logger.info('Attempting fallback to NetworkX backend');
            return await this.createKnowledgeGraph('networkx', options);
          }
        }
      }

      this.logger.info('Knowledge graph instance created successfully', {
        operation: 'KG_CREATE_SUCCESS',
        backend: backendType,
        instanceId: instance.constructor.name
      });

      return instance;

    } catch (error) {
      this.logger.error('Failed to create knowledge graph instance', {
        operation: 'KG_CREATE_FAILED',
        backend: backend,
        error: error.message,
        stack: error.stack
      });
      throw error;
    }
  }

  /**
   * Create a production knowledge graph instance
   * Automatically selects the best production backend available
   * 
   * @param {Object} options - Implementation options
   * @returns {Promise<KnowledgeGraphInterface>} Production KG instance
   */
  async createProductionKnowledgeGraph(options = {}) {
    try {
      // Try Neo4j first for production
      if (this._isBackendAvailable('neo4j')) {
        this.logger.info('Creating production KG with Neo4j backend');
        return await this.createKnowledgeGraph('neo4j', options);
      }

      // Fallback to NetworkX with warning
      this.logger.warn('Neo4j not available, falling back to NetworkX for production use', {
        operation: 'PRODUCTION_FALLBACK'
      });
      
      return await this.createKnowledgeGraph('networkx', {
        ...options,
        productionMode: true
      });

    } catch (error) {
      this.logger.error('Failed to create production knowledge graph', {
        operation: 'PRODUCTION_KG_CREATE_FAILED',
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Create a development knowledge graph instance
   * Optimized for fast iteration and testing
   * 
   * @param {Object} options - Implementation options
   * @returns {Promise<KnowledgeGraphInterface>} Development KG instance
   */
  async createDevelopmentKnowledgeGraph(options = {}) {
    try {
      this.logger.info('Creating development KG with NetworkX backend');
      
      return await this.createKnowledgeGraph('networkx', {
        ...options,
        developmentMode: true
      });

    } catch (error) {
      this.logger.error('Failed to create development knowledge graph', {
        operation: 'DEVELOPMENT_KG_CREATE_FAILED',
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Get information about available backends
   * 
   * @returns {Object} Backend information
   */
  getAvailableBackends() {
    const backends = {};
    
    for (const [type, config] of Object.entries(this.availableBackends)) {
      backends[type] = {
        name: config.name,
        description: config.description,
        production: config.production,
        available: this._isBackendAvailable(type),
        dependencies: config.dependencies
      };
    }

    return backends;
  }

  /**
   * Check if a specific backend is available
   * 
   * @param {string} backendType - Backend type to check
   * @returns {boolean} True if backend is available
   */
  isBackendAvailable(backendType) {
    return this._isBackendAvailable(backendType);
  }

  /**
   * Get recommended backend based on use case
   * 
   * @param {string} useCase - Use case ('development', 'production', 'testing')
   * @returns {string} Recommended backend type
   */
  getRecommendedBackend(useCase = 'development') {
    switch (useCase.toLowerCase()) {
      case 'production':
        return this._isBackendAvailable('neo4j') ? 'neo4j' : 'networkx';
      
      case 'development':
      case 'testing':
        return 'networkx';
      
      case 'research':
        // For SPARQL research, would recommend RDF4J when available
        return this._isBackendAvailable('neo4j') ? 'neo4j' : 'networkx';
      
      default:
        return this.config.defaultBackend;
    }
  }

  // Private methods

  /**
   * Resolve backend type from input
   * 
   * @private
   * @param {string|Object|null} backend - Backend specification
   * @returns {string} Resolved backend type
   */
  _resolveBackend(backend) {
    if (!backend) {
      return this.config.defaultBackend;
    }

    if (typeof backend === 'string') {
      return backend.toLowerCase();
    }

    if (typeof backend === 'object' && backend.type) {
      return backend.type.toLowerCase();
    }

    return this.config.defaultBackend;
  }

  /**
   * Get default options for a backend type
   * 
   * @private
   * @param {string} backendType - Backend type
   * @returns {Object} Default options
   */
  _getDefaultOptions(backendType) {
    const defaults = {
      networkx: {
        // NetworkX is in-memory, no external dependencies
      },
      neo4j: {
        uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
        user: process.env.NEO4J_USER || 'neo4j',
        password: process.env.NEO4J_PASSWORD || 'password'
      }
    };

    return defaults[backendType] || {};
  }

  /**
   * Check if backend dependencies are available
   * 
   * @private
   * @param {string} backendType - Backend type
   * @returns {Promise<void>}
   */
  async _validateDependencies(backendType) {
    const backendConfig = this.availableBackends[backendType];
    
    if (!backendConfig.dependencies || backendConfig.dependencies.length === 0) {
      return; // No dependencies to check
    }

    for (const dependency of backendConfig.dependencies) {
      try {
        require.resolve(dependency);
      } catch (error) {
        throw new Error(`Missing dependency for ${backendType} backend: ${dependency}`);
      }
    }
  }

  /**
   * Check if a backend is available
   * 
   * @private
   * @param {string} backendType - Backend type
   * @returns {boolean} True if available
   */
  _isBackendAvailable(backendType) {
    const backendConfig = this.availableBackends[backendType];
    
    if (!backendConfig) {
      return false;
    }

    // Check if dependencies are available
    if (backendConfig.dependencies && backendConfig.dependencies.length > 0) {
      for (const dependency of backendConfig.dependencies) {
        try {
          require.resolve(dependency);
        } catch (error) {
          return false;
        }
      }
    }

    return true;
  }
}

/**
 * Static factory methods for convenience
 */
class KnowledgeGraphFactoryStatic {
  static factory = null;

  /**
   * Get default factory instance
   * 
   * @param {Object} config - Factory configuration
   * @returns {KnowledgeGraphFactory} Factory instance
   */
  static getInstance(config = {}) {
    if (!this.factory) {
      this.factory = new KnowledgeGraphFactory(config);
    }
    return this.factory;
  }

  /**
   * Create knowledge graph using default factory
   * 
   * @param {string} backend - Backend type
   * @param {Object} options - Implementation options
   * @returns {Promise<KnowledgeGraphInterface>} KG instance
   */
  static async create(backend, options = {}) {
    const factory = this.getInstance();
    return await factory.createKnowledgeGraph(backend, options);
  }

  /**
   * Create production knowledge graph using default factory
   * 
   * @param {Object} options - Implementation options
   * @returns {Promise<KnowledgeGraphInterface>} Production KG instance
   */
  static async createProduction(options = {}) {
    const factory = this.getInstance();
    return await factory.createProductionKnowledgeGraph(options);
  }

  /**
   * Create development knowledge graph using default factory
   * 
   * @param {Object} options - Implementation options
   * @returns {Promise<KnowledgeGraphInterface>} Development KG instance
   */
  static async createDevelopment(options = {}) {
    const factory = this.getInstance();
    return await factory.createDevelopmentKnowledgeGraph(options);
  }
}

module.exports = {
  KnowledgeGraphFactory,
  KnowledgeGraphFactoryStatic
}; 