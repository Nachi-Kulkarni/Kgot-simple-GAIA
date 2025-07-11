/**
 * KGoT Controller Entry Point
 * 
 * This file serves as the main entry point for the Knowledge Graph of Thoughts (KGoT) Controller.
 * It initializes and starts the KGoT system with proper error handling and graceful shutdown.
 * 
 * @module KGoTControllerEntry
 */

// Load environment variables
require('dotenv').config();

const { KGoTController } = require('./kgot_controller');
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');

// Import logging configuration
const { loggers } = require('../../config/logging/winston_config');
const logger = loggers.kgotController;

/**
 * Main application class for KGoT Controller
 */
class KGoTApp {
  constructor() {
    console.log('🔧 [KGoT Controller] Initializing KGoT Controller application');
    
    this.app = express();
    this.port = process.env.KGOT_CONTROLLER_PORT || process.env.KGOT_PORT || 3001;
    this.kgotController = null;
    this.server = null;
    
    console.log(`🌐 [KGoT Controller] Port configured: ${this.port}`);
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
    
    console.log('✅ [KGoT Controller] Application initialization completed');
  }

  /**
   * Setup Express middleware
   */
  setupMiddleware() {
    console.log('🛡️ [KGoT Controller] Setting up security and middleware');
    
    this.app.use(helmet());
    console.log('🔒 [KGoT Controller] Helmet security middleware enabled');
    
    this.app.use(compression());
    console.log('📦 [KGoT Controller] Compression middleware enabled');
    
    this.app.use(cors());
    console.log('🌐 [KGoT Controller] CORS middleware enabled');
    
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
    console.log('📝 [KGoT Controller] JSON and URL-encoded parsers enabled (10MB limit)');
  }

  /**
   * Setup API routes
   */
  setupRoutes() {
    console.log('🛣️ [KGoT Controller] Setting up API routes');
    
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      console.log('💓 [KGoT Controller] Health check requested');
      res.json({
        status: 'healthy',
        service: 'KGoT Controller',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        version: '1.0.0'
      });
    });

    // KGoT processing endpoint
    this.app.post('/process', async (req, res) => {
      console.log('🧠 [KGoT Controller] Processing task request received');
      
      try {
        const { task, context = {} } = req.body;
        
        if (!task) {
          console.log('❌ [KGoT Controller] Task processing failed: Missing task');
          return res.status(400).json({
            error: 'Task is required',
            code: 'MISSING_TASK'
          });
        }

        console.log(`🎯 [KGoT Controller] Processing task: ${JSON.stringify(task).substring(0, 100)}...`);
        console.log(`📋 [KGoT Controller] Context keys: ${Object.keys(context).join(', ')}`);
        
        logger.info('Processing KGoT task', { task, context });
        
        const result = await this.kgotController.processTask(task, context);
        
        console.log('✅ [KGoT Controller] Task processing completed successfully');
        
        res.json({
          success: true,
          result,
          timestamp: new Date().toISOString()
        });
        
      } catch (error) {
        console.error('❌ [KGoT Controller] Task processing error:', error.message);
        logger.error('Error processing KGoT task', { error: error.message, stack: error.stack });
        
        res.status(500).json({
          error: 'Internal server error',
          message: error.message,
          code: 'PROCESSING_ERROR'
        });
      }
    });

    // Get KGoT status
    this.app.get('/status', (req, res) => {
      console.log('📊 [KGoT Controller] Status request received');
      
      const status = {
        isActive: this.kgotController?.isActive || false,
        currentIteration: this.kgotController?.currentIteration || 0,
        graphSize: this.kgotController?.knowledgeGraph?.size || 0,
        taskState: this.kgotController?.taskState?.size || 0
      };
      
      console.log('📈 [KGoT Controller] Current status:', JSON.stringify(status, null, 2));
      res.json(status);
    });

    // Get knowledge graph state
    this.app.get('/graph', (req, res) => {
      console.log('🕸️ [KGoT Controller] Knowledge graph state request received');
      
      if (!this.kgotController) {
        console.log('❌ [KGoT Controller] Graph request failed: Controller not initialized');
        return res.status(503).json({
          error: 'KGoT Controller not initialized'
        });
      }
      
      const graphState = this.kgotController.analyzeGraphState();
      console.log('📊 [KGoT Controller] Graph state retrieved successfully');
      res.json(graphState);
    });

    // Reset KGoT state
    this.app.post('/reset', async (req, res) => {
      console.log('🔄 [KGoT Controller] Reset request received');
      
      try {
        if (this.kgotController) {
          await this.kgotController.reset();
          console.log('✅ [KGoT Controller] Controller reset completed successfully');
        } else {
          console.log('⚠️ [KGoT Controller] Reset requested but controller not initialized');
        }
        
        res.json({
          success: true,
          message: 'KGoT Controller reset successfully'
        });
        
      } catch (error) {
        console.error('❌ [KGoT Controller] Reset failed:', error.message);
        logger.error('Error resetting KGoT Controller', { error: error.message });
        
        res.status(500).json({
          error: 'Reset failed',
          message: error.message
        });
      }
    });
    
    console.log('✅ [KGoT Controller] All API routes configured');
  }

  /**
   * Setup error handling middleware
   */
  setupErrorHandling() {
    // 404 handler
    this.app.use((req, res) => {
      res.status(404).json({
        error: 'Not found',
        path: req.path,
        method: req.method
      });
    });

    // Global error handler
    this.app.use((error, req, res, next) => {
      logger.error('Unhandled error', { 
        error: error.message, 
        stack: error.stack,
        path: req.path,
        method: req.method
      });
      
      res.status(500).json({
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
      });
    });
  }

  /**
   * Initialize and start the KGoT Controller
   */
  async start() {
    try {
      console.log('🚀 [KGoT Controller] Starting KGoT Controller application');
      logger.info('Starting KGoT Controller application');
      
      // Initialize KGoT Controller
      const config = {
        maxIterations: parseInt(process.env.KGOT_MAX_ITERATIONS) || 10,
        maxRetries: parseInt(process.env.KGOT_MAX_RETRIES) || 3,
        votingThreshold: parseFloat(process.env.KGOT_VOTING_THRESHOLD) || 0.6,
        validationEnabled: process.env.KGOT_VALIDATION_ENABLED !== 'false'
      };
      
      console.log('⚙️ [KGoT Controller] Configuration:', JSON.stringify(config, null, 2));
      
      this.kgotController = new KGoTController(config);
      
      console.log('🔧 [KGoT Controller] Initializing controller...');
      // Wait for controller initialization
      await this.kgotController.initializeAsync();
      console.log('✅ [KGoT Controller] Controller initialization completed');
      
      // Start HTTP server
      this.server = this.app.listen(this.port, () => {
        logger.info(`KGoT Controller server started on port ${this.port}`);
        console.log(`🚀 KGoT Controller running on http://localhost:${this.port}`);
        console.log(`📊 Health check: http://localhost:${this.port}/health`);
        console.log(`🧠 Status endpoint: http://localhost:${this.port}/status`);
      });
      
      // Setup graceful shutdown
      this.setupGracefulShutdown();
      
    } catch (error) {
      console.error('❌ [KGoT Controller] Failed to start application:', error.message);
      logger.error('Failed to start KGoT Controller', { error: error.message, stack: error.stack });
      process.exit(1);
    }
  }

  /**
   * Setup graceful shutdown handlers
   */
  setupGracefulShutdown() {
    console.log('🛡️ [KGoT Controller] Setting up graceful shutdown handlers');
    
    const shutdown = async (signal) => {
      logger.info(`Received ${signal}, shutting down gracefully`);
      console.log(`\n🛑 [KGoT Controller] Received ${signal}, shutting down gracefully...`);
      
      if (this.server) {
        this.server.close(() => {
          logger.info('HTTP server closed');
          console.log('✅ [KGoT Controller] HTTP server closed');
          process.exit(0);
        });
      } else {
        console.log('✅ [KGoT Controller] No server to close, exiting');
        process.exit(0);
      }
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
    
    process.on('uncaughtException', (error) => {
      logger.error('Uncaught exception', { error: error.message, stack: error.stack });
      console.error('❌ Uncaught exception:', error);
      process.exit(1);
    });
    
    process.on('unhandledRejection', (reason, promise) => {
      logger.error('Unhandled rejection', { reason, promise });
      console.error('❌ Unhandled rejection:', reason);
      process.exit(1);
    });
  }
}

// Start the application if this file is run directly
if (require.main === module) {
  const app = new KGoTApp();
  app.start().catch((error) => {
    console.error('❌ Failed to start application:', error);
    process.exit(1);
  });
}

module.exports = { KGoTApp };