/**
 * KGoT Integrated Tools Manager
 * 
 * Main entry point for the KGoT Integrated Tools service.
 * Provides HTTP API endpoints for tool execution and management.
 * 
 * @module IntegratedToolsManager
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const { KGoTToolBridge } = require('./kgot_tool_bridge');

// Import logging configuration
const { loggers } = require('../../config/logging/winston_config');
const logger = loggers.integratedTools;

/**
 * Integrated Tools Manager Class
 */
class IntegratedToolsManager {
  constructor(options = {}) {
    this.options = {
      port: options.port || process.env.PORT || 3005,
      host: options.host || '0.0.0.0',
      enableCors: options.enableCors !== false,
      enableRateLimit: options.enableRateLimit !== false,
      ...options
    };

    this.app = express();
    this.server = null;
    this.toolBridge = null;
    this.isReady = false;

    logger.info('Initializing Integrated Tools Manager', {
      operation: 'TOOLS_MANAGER_INIT',
      options: this.options
    });

    this.initializeServer();
  }

  /**
   * Initialize the Express server and middleware
   */
  async initializeServer() {
    try {
      // Security middleware
      this.app.use(helmet());
      this.app.use(compression());

      // CORS configuration
      if (this.options.enableCors) {
        this.app.use(cors({
          origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
          credentials: true
        }));
      }

      // Rate limiting
      if (this.options.enableRateLimit) {
        const limiter = rateLimit({
          windowMs: 15 * 60 * 1000, // 15 minutes
          max: 100, // limit each IP to 100 requests per windowMs
          message: 'Too many requests from this IP'
        });
        this.app.use(limiter);
      }

      // Body parsing middleware
      this.app.use(express.json({ limit: '10mb' }));
      this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

      // Initialize tool bridge
      await this.initializeToolBridge();

      // Setup routes
      this.setupRoutes();

      // Error handling middleware
      this.setupErrorHandling();

      logger.logOperation('info', 'SERVER_INIT_SUCCESS', 'Integrated Tools Manager server initialized');

    } catch (error) {
      logger.logError('SERVER_INIT_FAILED', error);
      throw error;
    }
  }

  /**
   * Initialize the KGoT Tool Bridge
   */
  async initializeToolBridge() {
    try {
      this.toolBridge = new KGoTToolBridge({
        enableAlitaIntegration: true,
        enableErrorManagement: true
      });

      // Wait for tool bridge to be ready
      await new Promise((resolve, reject) => {
        this.toolBridge.once('ready', resolve);
        this.toolBridge.once('error', reject);
      });

      this.isReady = true;
      logger.logOperation('info', 'TOOL_BRIDGE_READY', 'KGoT Tool Bridge initialized successfully');

    } catch (error) {
      logger.logError('TOOL_BRIDGE_INIT_FAILED', error);
      throw error;
    }
  }

  /**
   * Setup API routes
   */
  setupRoutes() {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        service: 'kgot-integrated-tools',
        ready: this.isReady,
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
      });
    });

    // Status endpoint
    this.app.get('/status', (req, res) => {
      res.json({
        service: 'KGoT Integrated Tools Manager',
        version: '1.0.0',
        ready: this.isReady,
        toolBridge: this.toolBridge ? 'initialized' : 'not initialized',
        environment: process.env.NODE_ENV || 'development',
        timestamp: new Date().toISOString()
      });
    });

    // Tools listing endpoint
    this.app.get('/tools', async (req, res) => {
      try {
        if (!this.isReady || !this.toolBridge) {
          return res.status(503).json({ error: 'Service not ready' });
        }

        const tools = await this.toolBridge.getAvailableTools();
        res.json({
          success: true,
          tools: tools,
          count: tools.length
        });

      } catch (error) {
        logger.logError('TOOLS_LIST_FAILED', error);
        res.status(500).json({
          success: false,
          error: 'Failed to retrieve tools list'
        });
      }
    });

    // Tool execution endpoint
    this.app.post('/execute', async (req, res) => {
      try {
        if (!this.isReady || !this.toolBridge) {
          return res.status(503).json({ error: 'Service not ready' });
        }

        const { toolName, parameters, sessionId } = req.body;

        if (!toolName) {
          return res.status(400).json({
            success: false,
            error: 'Tool name is required'
          });
        }

        const result = await this.toolBridge.executeTool(toolName, parameters, sessionId);
        
        res.json({
          success: true,
          result: result,
          toolName: toolName,
          sessionId: sessionId
        });

      } catch (error) {
        logger.logError('TOOL_EXECUTION_FAILED', error, { 
          toolName: req.body.toolName,
          sessionId: req.body.sessionId 
        });
        
        res.status(500).json({
          success: false,
          error: 'Tool execution failed',
          details: error.message
        });
      }
    });

    // Session management endpoints
    this.app.post('/session/create', async (req, res) => {
      try {
        const sessionId = await this.toolBridge.createSession(req.body.options);
        res.json({
          success: true,
          sessionId: sessionId
        });
      } catch (error) {
        logger.logError('SESSION_CREATE_FAILED', error);
        res.status(500).json({
          success: false,
          error: 'Failed to create session'
        });
      }
    });

    this.app.delete('/session/:sessionId', async (req, res) => {
      try {
        await this.toolBridge.closeSession(req.params.sessionId);
        res.json({
          success: true,
          message: 'Session closed successfully'
        });
      } catch (error) {
        logger.logError('SESSION_CLOSE_FAILED', error);
        res.status(500).json({
          success: false,
          error: 'Failed to close session'
        });
      }
    });
  }

  /**
   * Setup error handling middleware
   */
  setupErrorHandling() {
    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        success: false,
        error: 'Endpoint not found',
        path: req.originalUrl
      });
    });

    // Global error handler
    this.app.use((error, req, res, next) => {
      logger.logError('UNHANDLED_REQUEST_ERROR', error, {
        method: req.method,
        url: req.url,
        body: req.body
      });

      res.status(500).json({
        success: false,
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
      });
    });
  }

  /**
   * Start the server
   */
  async start() {
    try {
      this.server = this.app.listen(this.options.port, this.options.host, () => {
        logger.logOperation('info', 'SERVER_STARTED', `Integrated Tools Manager listening on ${this.options.host}:${this.options.port}`);
      });

      // Graceful shutdown handling
      process.on('SIGTERM', () => this.shutdown());
      process.on('SIGINT', () => this.shutdown());

    } catch (error) {
      logger.logError('SERVER_START_FAILED', error);
      throw error;
    }
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    logger.logOperation('info', 'SERVER_SHUTDOWN', 'Shutting down Integrated Tools Manager');

    if (this.server) {
      this.server.close(() => {
        logger.logOperation('info', 'SERVER_CLOSED', 'Server closed successfully');
        process.exit(0);
      });
    }
  }
}

// Start the server if this file is run directly
if (require.main === module) {
  const manager = new IntegratedToolsManager();
  manager.start().catch(error => {
    console.error('Failed to start Integrated Tools Manager:', error);
    process.exit(1);
  });
}

module.exports = { IntegratedToolsManager };