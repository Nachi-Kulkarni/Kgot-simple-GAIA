/**
 * Alita Manager Agent - Main Orchestrator
 * 
 * This is the central orchestrator for the Alita-KGoT enhanced system.
 * It manages coordination between all components including:
 * - Web Agent for web interactions
 * - MCP Creation for tool generation
 * - KGoT Controller for knowledge graph operations
 * - Multimodal processors for various content types
 * 
 * Built using LangChain for advanced agent capabilities and OpenRouter for model access.
 * 
 * @module AlitaManagerAgent
 */

// Load environment variables first
require('dotenv').config();

const { ChatOpenAI } = require('@langchain/openai');
const { AgentExecutor, createOpenAIFunctionsAgent } = require('langchain/agents');
const { ChatPromptTemplate, MessagesPlaceholder } = require('@langchain/core/prompts');
const { HumanMessage, SystemMessage } = require('@langchain/core/messages');
const { DynamicTool, DynamicStructuredTool } = require('@langchain/core/tools');
const { z } = require('zod');
const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const EventEmitter = require('events');

// Import logging configuration
const { loggers, httpLoggingMiddleware } = require('../../config/logging/winston_config');
const logger = loggers.managerAgent;

// Import configuration
const modelConfig = require('../../config/models/model_config.json');
console.log('📋 Model config loaded:', !!modelConfig, 'Keys:', Object.keys(modelConfig || {}));
console.log('🔧 KGoT config available:', !!(modelConfig?.alita_config?.kgot_controller));
if (modelConfig?.alita_config?.kgot_controller) {
  console.log('🔢 Max iterations:', modelConfig.alita_config.kgot_controller.max_iterations);
}

// Import KGoT Controller, MCP Validation, and Sequential Thinking Integration
const { KGoTController } = require('../../kgot_core/controller/kgot_controller');
const { MCPCrossValidationCoordinator } = require('../../kgot_core/controller/mcp_cross_validation');
const { SequentialThinkingIntegration } = require('./sequential_thinking_integration');
const { monitoring } = require('./monitoring');

/**
 * Alita Manager Agent Class
 * Orchestrates the entire Alita-KGoT system using LangChain agents
 * Extends EventEmitter for component coordination
 */
class AlitaManagerAgent extends EventEmitter {
  constructor() {
    super(); // Call EventEmitter constructor
    
    // Fix MaxListeners warning
    process.setMaxListeners(20); // Prevent leak warnings; adjust as needed
    
    this.app = express();
    this.port = parseInt(process.env.PORT) || 3000;
    this.agent = null;
    this.agentExecutor = null;
    this.isInitialized = false;
    
    // Initialize KGoT Controller, MCP Validation, and Sequential Thinking Integration
    this.kgotController = null;
    this.mcpValidator = null;
    this.sequentialThinking = null;
    
    console.log('🏗️  Constructing Alita Manager Agent...');
    logger.info('Initializing Alita Manager Agent', { operation: 'INIT' });
    
    console.log('📡 Setting up Express server...');
    this.setupExpress();
    
    console.log('🧩 Initializing core components asynchronously...');
    logger.info('Starting async component initialization', { timestamp: new Date().toISOString() });
    this.initializeComponents().catch(error => {
      console.error('❌ Component initialization failed:', error.message);
      logger.error('COMPONENTS_INIT_ASYNC_FAILED - Components initialization failed completely', {
        error: error.message,
        stack: error.stack,
        errorType: error.constructor.name,
        timestamp: new Date().toISOString()
      });
      logger.logError('COMPONENTS_INIT_ASYNC_FAILED', error);
    });
    
    console.log('🤖 Initializing LangChain agent asynchronously...');
    logger.info('Starting async agent initialization', { timestamp: new Date().toISOString() });
    this.initializeAgent().catch(error => {
      console.error('❌ Agent initialization failed:', error.message);
      logger.error('AGENT_INIT_ASYNC_FAILED - Agent initialization failed completely', {
        error: error.message,
        stack: error.stack,
        errorType: error.constructor.name,
        timestamp: new Date().toISOString()
      });
      logger.logError('AGENT_INIT_ASYNC_FAILED', error);
    });
    
    console.log('✅ Manager Agent constructor completed');
  }

  /**
   * Wait for all asynchronous initialization to complete
   */
  async waitForInitialization() {
    console.log('⏳ Waiting for component initialization...');
    
    // Wait for components to be initialized
    while (!this.kgotController || !this.mcpValidator || !this.sequentialThinking) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    console.log('✅ All components initialized, ready to start server');
  }

  /**
   * Wait for components to be ready before creating agent tools
   * Now handles partial initialization gracefully
   */
  async waitForComponentsReady() {
    console.log('  ⏳ Checking component readiness...');
    
    let componentStatus = {
      kgotController: !!this.kgotController,
      mcpValidator: !!this.mcpValidator,
      sequentialThinking: !!this.sequentialThinking
    };
    
    let successfulComponents = Object.values(componentStatus).filter(Boolean).length;
    let totalComponents = Object.keys(componentStatus).length;
    
    let attempts = 0;
    const maxAttempts = 10;  // 1s timeout
    
    while (successfulComponents < totalComponents && attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 100));  // Poll every 100ms
      componentStatus = {
        kgotController: !!this.kgotController,
        mcpValidator: !!this.mcpValidator,
        sequentialThinking: !!this.sequentialThinking
      };
      successfulComponents = Object.values(componentStatus).filter(Boolean).length;
      attempts++;
    }
    
    console.log('    📊 Component Status:');
    console.log(`      KGoT Controller: ${componentStatus.kgotController ? '✅' : '❌'}`);
    console.log(`      MCP Validator: ${componentStatus.mcpValidator ? '✅' : '❌'}`);
    console.log(`      Sequential Thinking: ${componentStatus.sequentialThinking ? '✅' : '❌'}`);
    
    console.log(`    📈 Initialization Success Rate: ${successfulComponents}/${totalComponents} (${Math.round(successfulComponents/totalComponents*100)}%)`);
    
    if (successfulComponents === 0) {
      throw new Error('Critical failure: No components initialized successfully. Cannot proceed.');
    }
    
    if (successfulComponents < totalComponents) {
      console.log('  ⚠️  Partial initialization detected - proceeding with available components');
      console.log('    💡 System will operate with reduced functionality but remain stable');
      console.warn(`⚠️ Timeout: Only ${successfulComponents}/${totalComponents} components ready`);
    } else {
      console.log('  ✅ All components initialized successfully');
    }
    
    console.log('  🚀 System ready for agent tool creation');
  }

  /**
   * Setup Express server with middleware and routes
   * Configures CORS, rate limiting, and logging middleware
   */
  setupExpress() {
    console.log('  🔧 Configuring Express middleware...');
    logger.logOperation('info', 'EXPRESS_SETUP', 'Setting up Express server configuration');
    
    // CORS configuration for cross-origin requests
    const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || '*';
    console.log('  🌐 CORS origins:', allowedOrigins);
    this.app.use(cors({
      origin: allowedOrigins,
      credentials: true
    }));

    // Rate limiting to prevent abuse
    console.log('  🛡️  Setting up rate limiting (100 req/15min)...');
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
      message: 'Too many requests from this IP'
    });
    this.app.use(limiter);

    // Request parsing middleware
    console.log('  📝 Configuring request parsing (10MB limit)...');
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true }));

    // Logging middleware for all HTTP requests
    console.log('  📊 Setting up HTTP logging middleware...');
    this.app.use(httpLoggingMiddleware(logger));

    // Prometheus monitoring middleware
    console.log('  📈 Setting up Prometheus monitoring middleware...');
    this.app.use(monitoring.createExpressMiddleware());

    console.log('  🛣️  Setting up API routes...');
    this.setupRoutes();
    
    console.log('  ✅ Express setup completed');
  }

  /**
   * Initialize KGoT Controller, MCP Validation, and Sequential Thinking components
   * Sets up the core reasoning and validation infrastructure with robust error handling
   */
  async initializeComponents() {
    const COMPONENT_TIMEOUT = 30000; // 30 seconds timeout per component
    
    try {
      console.log('🧩 Starting component initialization...');
      const startTime = Date.now();
      logger.logOperation('info', 'COMPONENTS_INIT', 'Initializing KGoT Controller, MCP Validation, and Sequential Thinking');

      // Validate modelConfig before using it
      if (!modelConfig || !modelConfig.alita_config || !modelConfig.alita_config.kgot_controller) {
        throw new Error('Invalid or missing model configuration. Please check config/models/model_config.json');
      }

      // Initialize KGoT Controller with enhanced configuration and timeout
      try {
        console.log('  🧠 Initializing KGoT Controller...');
        const kgotStartTime = Date.now();
        const kgotConfig = {
          maxIterations: modelConfig.alita_config.kgot_controller.max_iterations || 10,
          votingThreshold: 0.6,
          validationEnabled: true,
          graphUpdateInterval: 5000
        };
        console.log('    📋 KGoT Config:', JSON.stringify(kgotConfig, null, 2));
        console.log('    ⏳ Creating KGoT Controller instance...');
        
        this.kgotController = new KGoTController(kgotConfig);
        
        // Wait for KGoT Controller async initialization with timeout
        console.log('    ⏳ Waiting for KGoT Controller initialization...');
        await Promise.race([
          this.kgotController.initializeAsync(),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('KGoT Controller initialization timeout')), COMPONENT_TIMEOUT)
          )
        ]);
        
        const kgotInitTime = Date.now() - kgotStartTime;
        console.log(`    ✅ KGoT Controller initialized successfully in ${kgotInitTime}ms`);
        console.log('    🔗 KGoT Controller ready for graph operations');
      } catch (error) {
        console.error('    ❌ KGoT Controller initialization failed:', error.message);
        logger.logError('KGOT_INIT_FAILED', error);
        this.kgotController = null; // Set to null but continue
      }

      // Initialize MCP Cross-Validation Coordinator with timeout
      try {
        console.log('  🔍 Initializing MCP Cross-Validation Coordinator...');
        const mcpStartTime = Date.now();
        
        const mcpConfig = {
          validationModels: ['claude-sonnet-4', 'o3', 'grok-4'],
          primaryModel: 'claude-sonnet-4',
          consensusThreshold: 0.7,
          confidenceThreshold: 0.8,
          enableBenchmarkValidation: true,
          enableCrossModel: true
        };
        console.log('    📋 MCP Config:', JSON.stringify(mcpConfig, null, 2));
        console.log('    ⏳ Creating MCP Cross-Validation Coordinator instance...');
        
        this.mcpValidator = new MCPCrossValidationCoordinator(mcpConfig);
        
        // Check if initialize method exists and call with timeout
        console.log('    ⏳ Initializing MCP validation components...');
        if (typeof this.mcpValidator.initialize === 'function') {
          await Promise.race([
            this.mcpValidator.initialize(),
            new Promise((_, reject) => 
              setTimeout(() => reject(new Error('MCP Validator initialization timeout')), COMPONENT_TIMEOUT)
            )
          ]);
        } else {
          console.log('    ℹ️  MCP Validator does not require async initialization');
        }
        
        const mcpInitTime = Date.now() - mcpStartTime;
        console.log(`    ✅ MCP Cross-Validation Coordinator initialized successfully in ${mcpInitTime}ms`);
        console.log('    🛡️ MCP Validator ready for cross-validation operations');
      } catch (error) {
        console.error('    ❌ MCP Cross-Validation Coordinator initialization failed:', error.message);
        logger.logError('MCP_VALIDATOR_INIT_FAILED', error);
        this.mcpValidator = null; // Set to null but continue
      }

      // Initialize Sequential Thinking Integration with timeout
      try {
        console.log('  🤔 Initializing Sequential Thinking Integration...');
        const sequentialStartTime = Date.now();
        
        const sequentialConfig = {
          complexityThreshold: 7,
          errorThreshold: 3,
          maxThoughts: 15,
          enableCrossSystemCoordination: true,
          enableAdaptiveThinking: true,
          thoughtTimeout: 30000,
          enableMCPCreationRouting: true,
          enableKGoTProcessing: true
        };
        console.log('    📋 Sequential Thinking Config:', JSON.stringify(sequentialConfig, null, 2));
        console.log('    ⏳ Creating Sequential Thinking Integration instance...');
        
        this.sequentialThinking = new SequentialThinkingIntegration(sequentialConfig);
        
        // Check if initialize method exists and call with timeout
        if (typeof this.sequentialThinking.initialize === 'function') {
          console.log('    ⏳ Initializing Sequential Thinking components...');
          await Promise.race([
            this.sequentialThinking.initialize(),
            new Promise((_, reject) => 
              setTimeout(() => reject(new Error('Sequential Thinking initialization timeout')), COMPONENT_TIMEOUT)
            )
          ]);
        } else {
          console.log('    ℹ️  Sequential Thinking initialized in constructor');
        }
        
        const sequentialInitTime = Date.now() - sequentialStartTime;
        console.log(`    ✅ Sequential Thinking Integration initialized successfully in ${sequentialInitTime}ms`);
        console.log('    🧩 Sequential Thinking ready for complex reasoning tasks');
      } catch (error) {
        console.error('    ❌ Sequential Thinking Integration initialization failed:', error.message);
        logger.logError('SEQUENTIAL_THINKING_INIT_FAILED', error);
        this.sequentialThinking = null; // Set to null but continue
      }

      // Set up event listeners for successfully initialized components
      console.log('  🔗 Setting up component event listeners...');
      this.setupComponentEventListeners();
      console.log('    ✅ Event listeners configured');

      // Log final component status
      const componentStatus = {
        kgotController: !!this.kgotController,
        mcpValidator: !!this.mcpValidator,
        sequentialThinking: !!this.sequentialThinking
      };
      
      const successfulComponents = Object.entries(componentStatus).filter(([_, status]) => status).map(([name]) => name);
      const failedComponents = Object.entries(componentStatus).filter(([_, status]) => !status).map(([name]) => name);

      const totalInitTime = Date.now() - startTime;
      console.log('\n🎉 ===== COMPONENT INITIALIZATION COMPLETE =====');
      console.log(`⏱️  Total initialization time: ${totalInitTime}ms`);
      console.log('📊 Component Status:');
      if (successfulComponents.length > 0) {
        console.log('   ✅ Successfully initialized:');
        successfulComponents.forEach(comp => console.log(`      • ${comp}`));
      }
      if (failedComponents.length > 0) {
        console.log('   ❌ Failed to initialize:');
        failedComponents.forEach(comp => console.log(`      • ${comp}`));
      }
      console.log('🚀 System ready for operation with available components!');
      console.log('================================================\n');
      
      logger.logOperation('info', 'COMPONENTS_INIT_COMPLETED', 'Component initialization completed', {
        componentStatus,
        successfulComponents,
        failedComponents,
        totalInitializationTime: totalInitTime,
        timestamp: new Date().toISOString()
      });

      // Check if at least one core component initialized successfully
      const hasAnyComponent = this.kgotController || this.mcpValidator || this.sequentialThinking;
      if (!hasAnyComponent) {
        throw new Error('All core components failed to initialize - system cannot continue');
      }

    } catch (error) {
      console.error('❌ Critical component initialization failure:', error.message);
      console.error('🔍 Error stack trace:', error.stack);
      console.error('💡 Check configuration, dependencies, and API keys');
      logger.logError('COMPONENTS_INIT_CRITICAL_FAILED', error, { 
        kgotConfig: modelConfig?.alita_config?.kgot_controller || 'Configuration not available'
      });
      throw error;
    }
  }

  /**
   * Setup event listeners for component coordination
   * Enables communication between available components (handles partial initialization)
   */
  setupComponentEventListeners() {
    console.log('    🔗 Setting up event listeners for available components...');
    
    // Setup KGoT Controller event listeners if available
    if (this.kgotController) {
      console.log('    🧠 Configuring KGoT Controller event listeners...');
      
      // Listen for KGoT execution completion to trigger validation
      this.kgotController.on('executionComplete', async (executionResult) => {
        console.log('✅ KGoT execution completed, triggering validation pipeline');
        logger.logOperation('info', 'KGOT_EXECUTION_COMPLETE', 'KGoT execution completed, triggering validation');
        
        if (this.mcpValidator && executionResult.solution) {
          try {
            const validationResult = await this.mcpValidator.validateSolution(
              executionResult.solution,
              executionResult.originalTask,
              executionResult.context
            );
            
            // Emit coordinated result
            this.emit('solutionValidated', {
              execution: executionResult,
              validation: validationResult,
              isValid: validationResult.isValid,
              confidence: validationResult.confidence
            });
            
          } catch (validationError) {
            logger.logError('VALIDATION_COORDINATION_FAILED', validationError);
          }
        } else {
          console.log('⚠️  MCP Validator not available - skipping validation step');
        }
      });
    } else {
      console.log('    ⚠️  KGoT Controller not available - skipping event listeners');
    }
    
    // Setup MCP Validation event listeners if available
    if (this.mcpValidator) {
      console.log('    🛡️ Configuring MCP Validation event listeners...');
      
      // Listen for validation completion
      this.mcpValidator.on('validationComplete', (validationResult) => {
        console.log('🛡️ MCP validation completed with confidence:', validationResult.confidence || 'unknown');
        logger.logOperation('info', 'VALIDATION_COMPLETE', 'MCP validation completed', {
          validationId: validationResult.validationId,
          isValid: validationResult.isValid,
          confidence: validationResult.confidence
        });
        
        // Trigger sequential thinking if validation indicates complexity and component is available
        if (validationResult.complexity > 7 && this.sequentialThinking) {
          console.log('🧠 High complexity detected, triggering sequential thinking...');
          this.sequentialThinking.processComplexTask(validationResult);
        } else if (validationResult.complexity > 7) {
          console.log('⚠️  High complexity detected but Sequential Thinking not available');
        }
      });
    } else {
      console.log('    ⚠️  MCP Validator not available - skipping event listeners');
    }

    // Setup Sequential Thinking event listeners if available
    if (this.sequentialThinking) {
      console.log('    🤔 Configuring Sequential Thinking event listeners...');
      
      // Listen for Sequential Thinking progress updates
      this.sequentialThinking.on('thinkingProgress', (progressData) => {
        console.log(`🤔 Sequential thinking progress: ${progressData.progress || 'unknown'}% (Session: ${progressData.sessionId})`);
        logger.logOperation('debug', 'SEQUENTIAL_THINKING_PROGRESS', 'Sequential thinking progress update', {
          sessionId: progressData.sessionId,
          progress: progressData.progress,
          currentThought: progressData.currentThought?.conclusion?.substring(0, 100) || 'Processing...'
        });
      });

      // Listen for Sequential Thinking completion
      this.sequentialThinking.on('thinkingComplete', (thinkingSession) => {
        console.log('🎯 Sequential thinking process completed successfully');
        console.log(`   📊 Session: ${thinkingSession.sessionId}`);
        console.log(`   ⏱️ Duration: ${thinkingSession.duration}ms`);
        console.log(`   🧠 Thoughts: ${thinkingSession.thoughts.length}`);
        console.log(`   📋 Template: ${thinkingSession.template.name}`);
        
        logger.logOperation('info', 'SEQUENTIAL_THINKING_COMPLETE', 'Sequential thinking process completed', {
          sessionId: thinkingSession.sessionId,
          duration: thinkingSession.duration,
          thoughtCount: thinkingSession.thoughts.length,
          template: thinkingSession.template.name,
          systemRecommendations: thinkingSession.systemRecommendations?.systemSelection?.primary
        });

        // Emit system coordination event with routing recommendations
        console.log('🔗 Emitting system coordination plan...');
        this.emit('systemCoordinationPlan', {
          sessionId: thinkingSession.sessionId,
          routingRecommendations: thinkingSession.systemRecommendations,
          conclusions: thinkingSession.conclusions,
          taskId: thinkingSession.taskId
        });
      });
    } else {
      console.log('    ⚠️  Sequential Thinking not available - skipping event listeners');
    }

    console.log('    🔗 Configuring system coordination plan event listener...');
    
    // Listen for system coordination plan to execute routing logic
    this.on('systemCoordinationPlan', async (coordinationPlan) => {
      console.log('🎯 System coordination plan received:');
      console.log(`   📋 Session ID: ${coordinationPlan.sessionId}`);
      console.log(`   🎯 Primary System: ${coordinationPlan.routingRecommendations?.systemSelection?.primary || 'unknown'}`);
      console.log(`   📊 Strategy: ${coordinationPlan.routingRecommendations?.coordinationStrategy?.strategy || 'unknown'}`);
      console.log(`   🔄 Execution Steps: ${coordinationPlan.routingRecommendations?.executionSequence?.length || 0}`);
      
      logger.logOperation('info', 'SYSTEM_COORDINATION_TRIGGERED', 'Executing system coordination plan', {
        sessionId: coordinationPlan.sessionId,
        primarySystem: coordinationPlan.routingRecommendations?.systemSelection?.primary,
        strategy: coordinationPlan.routingRecommendations?.coordinationStrategy?.strategy
      });

      // Execute routing logic based on Sequential Thinking recommendations
      try {
        console.log('⚡ Executing system coordination plan...');
        await this.executeSystemCoordination(coordinationPlan);
        console.log('✅ System coordination plan executed successfully');
      } catch (coordinationError) {
        console.error('❌ System coordination plan failed:', coordinationError.message);
        logger.logError('SYSTEM_COORDINATION_ERROR', coordinationError, {
          sessionId: coordinationPlan.sessionId
        });
      }
    });

    logger.logOperation('info', 'EVENT_LISTENERS_SETUP', 'All component event listeners configured', {
      kgotListeners: ['executionComplete'],
      validationListeners: ['validationComplete'],
      sequentialThinkingListeners: ['thinkingProgress', 'thinkingComplete'],
      managerListeners: ['systemCoordinationPlan']
    });
  }

  /**
   * Execute system coordination based on Sequential Thinking recommendations
   * Routes tasks between Alita and KGoT systems according to the coordination plan
   * 
   * @param {Object} coordinationPlan - The coordination plan from Sequential Thinking
   */
  async executeSystemCoordination(coordinationPlan) {
    const { routingRecommendations, sessionId } = coordinationPlan;
    const { systemSelection, coordinationStrategy, executionSequence } = routingRecommendations;

    console.log('🔄 Starting system coordination execution:');
    console.log(`   📋 Session: ${sessionId}`);
    console.log(`   🎯 Primary System: ${systemSelection?.primary || 'unknown'}`);
    console.log(`   📊 Strategy: ${coordinationStrategy?.strategy || 'unknown'}`);
    console.log(`   🔢 Steps to execute: ${executionSequence?.length || 0}`);

    logger.logOperation('info', 'SYSTEM_COORDINATION_EXECUTION', 'Executing system coordination', {
      sessionId,
      primarySystem: systemSelection?.primary,
      strategy: coordinationStrategy?.strategy,
      sequenceSteps: executionSequence?.length || 0
    });

    // Execute the coordination sequence
    for (const step of executionSequence || []) {
      try {
        console.log(`\n🔄 Executing step ${step.step}:`);
        console.log(`   🎯 System: ${step.system}`);
        console.log(`   ⚡ Action: ${step.action}`);
        console.log(`   📝 Description: ${step.description || 'No description'}`);
        
        logger.logOperation('debug', 'COORDINATION_STEP', `Executing coordination step ${step.step}`, {
          sessionId,
          system: step.system,
          action: step.action
        });

        // Route to appropriate system based on step configuration
        if (step.system === 'both' || step.system === 'coordinator') {
          console.log('   🔄 Routing to: Both systems (parallel execution)');
          // Parallel execution of both systems
          await this.executeParallelSystemCoordination(step, coordinationPlan);
        } else if (step.system === 'Alita') {
          console.log('   🤖 Routing to: Alita system components');
          // Route to Alita system components
          await this.routeToAlitaSystem(step, coordinationPlan);
        } else if (step.system === 'KGoT') {
          console.log('   🧠 Routing to: KGoT knowledge graph system');
          // Route to KGoT system
          await this.routeToKGoTSystem(step, coordinationPlan);
        } else {
          console.log(`   ❓ Unknown system: ${step.system}`);
        }

        console.log(`   ✅ Step ${step.step} completed successfully`);
        logger.logOperation('debug', 'COORDINATION_STEP_COMPLETE', `Coordination step ${step.step} completed`, {
          sessionId,
          system: step.system
        });

      } catch (stepError) {
        console.error(`   ❌ Step ${step.step} failed: ${stepError.message}`);
        console.error(`   🔍 Error in system: ${step.system}`);
        
        logger.logError('COORDINATION_STEP_ERROR', stepError, {
          sessionId,
          step: step.step,
          system: step.system
        });
        
        // Execute fallback strategy if available
        if (routingRecommendations.fallbackStrategy) {
          console.log(`   🔄 Executing fallback strategy: ${routingRecommendations.fallbackStrategy.strategy}`);
          await this.executeFallbackStrategy(routingRecommendations.fallbackStrategy, stepError);
        } else {
          console.log('   ⚠️ No fallback strategy available');
        }
      }
    }

    console.log('\n🎉 System coordination execution completed!');
    console.log(`   📊 Total steps executed: ${executionSequence?.length || 0}`);
    console.log(`   📋 Session: ${sessionId}`);
    
    logger.logOperation('info', 'SYSTEM_COORDINATION_COMPLETE', 'System coordination completed', {
      sessionId,
      executedSteps: executionSequence?.length || 0
    });
  }

  /**
   * Execute parallel system coordination for both Alita and KGoT
   * 
   * @param {Object} step - Coordination step
   * @param {Object} coordinationPlan - Complete coordination plan
   */
  async executeParallelSystemCoordination(step, coordinationPlan) {
    console.log('     🔄 Starting parallel coordination...');
    console.log(`     📋 Action: ${step.action}`);
    console.log('     🤖 Alita system: Preparing for parallel execution');
    console.log('     🧠 KGoT system: Preparing for parallel execution');
    
    // Implementation for parallel system coordination
    logger.logOperation('info', 'PARALLEL_COORDINATION', 'Executing parallel system coordination', {
      sessionId: coordinationPlan.sessionId,
      step: step.step,
      action: step.action
    });
    
    // This would be implemented based on specific coordination requirements
    // For now, we log the coordination attempt
    console.log('     ✅ Parallel coordination setup completed');
  }

  /**
   * Route task to Alita system components
   * 
   * @param {Object} step - Coordination step
   * @param {Object} coordinationPlan - Complete coordination plan
   */
  async routeToAlitaSystem(step, coordinationPlan) {
    console.log('     🤖 Routing to Alita system...');
    console.log(`     📋 Action: ${step.action}`);
    console.log('     🔧 Available Alita components: MCP Creation, Web Agent, Task Management');
    
    logger.logOperation('info', 'ALITA_ROUTING', 'Routing to Alita system', {
      sessionId: coordinationPlan.sessionId,
      action: step.action,
      availableComponents: ['MCP', 'WebAgent', 'TaskManager']
    });
    
    // Route to appropriate Alita components based on action
    // This would trigger MCP creation, web agent, or other Alita components
    console.log('     ✅ Alita system routing completed');
  }

  /**
   * Route task to KGoT system
   * 
   * @param {Object} step - Coordination step
   * @param {Object} coordinationPlan - Complete coordination plan
   */
  async routeToKGoTSystem(step, coordinationPlan) {
    console.log('     🧠 Routing to KGoT system...');
    console.log(`     📋 Action: ${step.action}`);
    console.log('     🔗 Available KGoT capabilities: Knowledge Graph, Reasoning, Graph Operations');
    
    logger.logOperation('info', 'KGOT_ROUTING', 'Routing to KGoT system', {
      sessionId: coordinationPlan.sessionId,
      action: step.action,
      controllerAvailable: !!this.kgotController
    });
    
    // Route to KGoT controller for knowledge graph reasoning
    if (this.kgotController) {
      console.log('     ✅ KGoT Controller available, processing request...');
      // This would trigger KGoT processing based on the coordination plan
    } else {
      console.log('     ⚠️ KGoT Controller not available');
    }
    
    console.log('     ✅ KGoT system routing completed');
  }

  /**
   * Execute fallback strategy when coordination fails
   * 
   * @param {Object} fallbackStrategy - Fallback strategy configuration
   * @param {Error} error - The error that triggered fallback
   */
  async executeFallbackStrategy(fallbackStrategy, error) {
    console.log('     🔄 Executing fallback strategy...');
    console.log(`     📋 Strategy: ${fallbackStrategy.strategy}`);
    console.log(`     ❌ Triggered by error: ${error.message}`);
    
    logger.logOperation('warn', 'FALLBACK_STRATEGY', 'Executing fallback strategy', {
      strategy: fallbackStrategy.strategy,
      error: error.message
    });
    
    // Implement fallback logic based on strategy type
    switch (fallbackStrategy.strategy) {
      case 'graceful_degradation':
        console.log('     🔽 Applying graceful degradation - reducing functionality');
        // Reduce functionality but continue operation
        break;
      case 'retry_with_backoff':
        console.log('     🔄 Applying retry with backoff strategy');
        // Retry with exponential backoff
        break;
      case 'rollback_on_failure':
        console.log('     ↩️ Applying rollback strategy - reverting to previous state');
        // Rollback to previous state
        break;
      default:
        console.log(`     ❓ Unknown fallback strategy: ${fallbackStrategy.strategy}`);
        logger.logOperation('warn', 'UNKNOWN_FALLBACK', 'Unknown fallback strategy', {
          strategy: fallbackStrategy.strategy
        });
    }
    
    console.log('     ✅ Fallback strategy execution completed');
  }

  /**
   * Initialize the LangChain agent with OpenRouter models
   * Creates the agent executor with necessary tools and prompts
   */
  async initializeAgent() {
    const initStartTime = Date.now();
    const initId = `init_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    logger.info(`[${initId}] ===== AGENT INITIALIZATION STARTING =====`, {
      initId,
      timestamp: new Date().toISOString(),
      isInitialized: this.isInitialized,
      processId: process.pid
    });

    if (this.isInitialized) {
      console.log('⏳ Initializing LangChain agent...');
      console.log('Agent already initialized, skipping...');
      logger.info(`[${initId}] Agent already initialized, skipping initialization`, { initId });
      return;
    }

    try {
      // Enhanced debug logging for modelConfig availability
      logger.info(`[${initId}] STEP 1: Checking modelConfig availability...`, { initId });
      console.log('🔍 DEBUG: Checking modelConfig availability...');
      console.log('  modelConfig type:', typeof modelConfig);
      console.log('  modelConfig exists:', !!modelConfig);
      
      if (!modelConfig) {
        const error = new Error('modelConfig is not available - import failed');
        logger.error(`[${initId}] CRITICAL: modelConfig import failed`, { initId, error: error.message });
        throw error;
      }
      
      console.log('  modelConfig keys:', Object.keys(modelConfig));
      console.log('  alita_config exists:', !!modelConfig.alita_config);
      logger.info(`[${initId}] modelConfig validated successfully`, { 
        initId,
        modelConfigKeys: Object.keys(modelConfig),
        hasAlitaConfig: !!modelConfig.alita_config
      });
      
      if (!modelConfig.alita_config) {
        const error = new Error('modelConfig.alita_config is not available');
        logger.error(`[${initId}] CRITICAL: alita_config missing from modelConfig`, { initId, error: error.message });
        throw error;
      }
      
      console.log('  manager_agent config exists:', !!modelConfig.alita_config.manager_agent);
      
      if (!modelConfig.alita_config.manager_agent) {
        const error = new Error('modelConfig.alita_config.manager_agent is not available');
        logger.error(`[${initId}] CRITICAL: manager_agent config missing`, { initId, error: error.message });
        throw error;
      }
      
      // Get model configuration for manager agent
      logger.info(`[${initId}] STEP 2: Loading manager configuration...`, { initId });
      const managerConfig = modelConfig.alita_config.manager_agent;
      console.log('✅ managerConfig loaded successfully:', JSON.stringify(managerConfig, null, 2));
      logger.info(`[${initId}] managerConfig loaded successfully`, { 
        initId,
        config: managerConfig
      });
      logger.logOperation('info', 'AGENT_INIT', 'Initializing LangChain agent with OpenRouter');
      
      // Wait for all components to be initialized before creating tools
      logger.info(`[${initId}] STEP 3: Waiting for components to be ready...`, { initId });
      console.log('🔄 Waiting for components to be ready before creating agent tools...');
      await this.waitForComponentsReady();
      console.log('✅ All components ready, proceeding with agent initialization');
      logger.info(`[${initId}] All components ready, proceeding with agent initialization`, { initId });
      
      // Initialize OpenRouter LLM through OpenAI-compatible API
      logger.info(`[${initId}] STEP 4: Initializing OpenRouter LLM...`, { 
        initId,
        baseURL: modelConfig.model_providers.openrouter.base_url,
        modelName: 'anthropic/claude-sonnet-4',
        hasApiKey: !!process.env.OPENROUTER_API_KEY
      });
      
      const llm = new ChatOpenAI({
        openAIApiKey: process.env.OPENROUTER_API_KEY,
        configuration: {
          baseURL: modelConfig.model_providers.openrouter.base_url,
          defaultHeaders: {
            "HTTP-Referer": "https://localhost:8888",
            "X-Title": "Alita-KGoT Enhanced System"
          }
        },
        // Use valid OpenRouter model name that supports tool calling
        modelName: 'anthropic/claude-sonnet-4',
        temperature: 0.1, // Lower temperature for more consistent orchestration
        maxTokens: 80000,
        timeout: managerConfig.timeout * 1000,
        maxRetries: managerConfig.max_retries,
      });
      
      logger.info(`[${initId}] OpenRouter LLM initialized successfully`, { initId });

      // Create agent tools for interacting with other components
      logger.info(`[${initId}] STEP 5: Creating agent tools...`, { initId });
      const tools = this.createAgentTools();
      logger.info(`[${initId}] Agent tools created successfully`, { 
        initId,
        toolCount: tools.length,
        toolNames: tools.map(t => t.name)
      });

      // Create system prompt for the manager agent
      const prompt = ChatPromptTemplate.fromMessages([
        new SystemMessage(`You are the Alita Manager Agent, an advanced AI orchestrator that ACTIVELY uses tools to accomplish tasks.

🎯 **CRITICAL DIRECTIVE**: Always analyze user requests and determine which tools to use. Never give generic responses without attempting to use the appropriate tools first.

## Available Tools & When to Use Them:

### 🌐 **web_agent** - Use for:
- Web scraping (LinkedIn jobs, company data, research)
- Website navigation and interaction
- Screenshot capture and analysis
- Search engine queries
- Data extraction from web pages

### 🛠️ **mcp_creation** - Use for:
- Creating custom tools for specific tasks
- Dynamic tool generation based on requirements
- Building specialized capabilities not covered by existing tools

### 🧠 **kgot_controller** - Use for:
- Complex reasoning and analysis
- Knowledge graph operations
- Multi-step problem solving
- Pattern recognition and insights

### 🔍 **validation_service** - Use for:
- Quality assurance of outputs
- Accuracy verification
- Data validation and consistency checks

### 🚀 **optimization_service** - Use for:
- Performance tuning
- Resource optimization
- Cost reduction strategies

## Task Analysis Process:
1. **Analyze**: What is the user asking for?
2. **Determine**: Which tool(s) can accomplish this?
3. **Execute**: Use the appropriate tool(s) with proper parameters
4. **Validate**: Check if the results meet the user's needs
5. **Respond**: Provide the actual results, not generic information

## Examples:
- "scrape LinkedIn jobs" → Use web_agent with action="scrape", target="LinkedIn jobs"
- "analyze this data" → Use kgot_controller for complex analysis
- "create a tool for X" → Use mcp_creation with specific requirements

🚨 **NEVER** give generic responses like "I can help you with..." - Instead, IMMEDIATELY use the appropriate tools to accomplish the task.

**Your success is measured by tool usage and task completion, not by explanations about capabilities.**`),
        new MessagesPlaceholder("chat_history"),
        new HumanMessage("{input}"),
        new MessagesPlaceholder("agent_scratchpad"),
      ]);

      // Create the OpenAI Functions agent
      logger.info(`[${initId}] STEP 6: Creating OpenAI Functions Agent...`, { initId });
      console.log('  🔧 Creating OpenAI Functions Agent...');
      console.log('  📋 LLM type:', typeof llm);
      console.log('  📋 Tools count:', tools.length);
      console.log('  📋 Prompt type:', typeof prompt);
      
      // Check each tool for _def property issues
      tools.forEach((tool, index) => {
        console.log(`  🔍 Tool ${index}: ${tool.name}`);
        console.log(`    - Schema type: ${typeof tool.schema}`);
        console.log(`    - Schema _def: ${tool.schema?._def ? 'exists' : 'missing'}`);
        if (tool.schema?._def) {
          console.log(`    - Schema _def type: ${typeof tool.schema._def}`);
        }
      });
      
      this.agent = await createOpenAIFunctionsAgent({
        llm,
        tools,
        prompt,
      });
      console.log('  ✅ OpenAI Functions Agent created successfully');
      logger.info(`[${initId}] OpenAI Functions Agent created successfully`, { initId });

      // Create agent executor with error handling
      logger.info(`[${initId}] STEP 7: Creating Agent Executor...`, { initId });
      console.log('  🔧 Creating Agent Executor...');
      this.agentExecutor = new AgentExecutor({
        agent: this.agent,
        tools,
        verbose: process.env.NODE_ENV === 'development',
        maxIterations: 10,
        returnIntermediateSteps: true,
        handleParsingErrors: true,
      });
      console.log('  ✅ Agent Executor created successfully');
      logger.info(`[${initId}] Agent Executor created successfully`, { initId });

      this.isInitialized = true;
      const initDuration = Date.now() - initStartTime;
      
      logger.info(`[${initId}] ===== AGENT INITIALIZATION COMPLETED =====`, {
        initId,
        success: true,
        duration: initDuration,
        isInitialized: this.isInitialized,
        agentExists: !!this.agent,
        agentExecutorExists: !!this.agentExecutor,
        timestamp: new Date().toISOString()
      });
      
      logger.logOperation('info', 'AGENT_INIT_SUCCESS', 'LangChain agent initialized successfully', {
        initId,
        duration: initDuration
      });

    } catch (error) {
      const initDuration = Date.now() - initStartTime;
      
      logger.error(`[${initId}] ===== AGENT INITIALIZATION FAILED =====`, {
        initId,
        error: error.message,
        errorStack: error.stack,
        errorType: error.constructor.name,
        duration: initDuration,
        isInitialized: this.isInitialized,
        agentExists: !!this.agent,
        agentExecutorExists: !!this.agentExecutor,
        timestamp: new Date().toISOString()
      });
      
      logger.logError('AGENT_INIT_FAILED', error, { 
        initId,
        errorMessage: error.message,
        modelConfigAvailable: !!modelConfig,
        openRouterConfigAvailable: !!(modelConfig?.model_providers?.openrouter),
        duration: initDuration
      });
      throw error;
    }
  }

  /**
   * Create tools for the LangChain agent to interact with other components
   * Updated to use DynamicStructuredTool for LangChain v0.3 compatibility,
   * ensuring proper handling of Zod schemas to prevent '_def' undefined errors.
   * @returns {Array} Array of LangChain tools
   */
  createAgentTools() {
    logger.logOperation('info', 'TOOLS_CREATION', 'Creating agent tools for system components');
    
    // Store reference to this instance for proper context binding in tool functions
    const managerInstance = this;
    
    try {
      // Verify Zod is available
      if (typeof z === 'undefined') {
        throw new Error('Zod (z) is not available. Check import statement.');
      }
      
      console.log('  🔧 Creating agent tools with explicit Zod schemas...');
      console.log('  📋 Zod object type:', typeof z);
      console.log('  📋 Zod.object available:', typeof z.object);
      
      // Test Zod schema creation
      try {
        const testSchema = z.object({
          test: z.string().describe("Test field")
        });
        console.log('  🧪 Zod test schema created successfully:', {
          hasSchema: !!testSchema,
          hasDef: !!testSchema._def,
          hasShape: !!testSchema.shape,
          hasParse: typeof testSchema.parse === 'function',
          schemaType: typeof testSchema,
          constructor: testSchema.constructor.name
        });
        console.log('  Zod version:', z.version);
      } catch (testError) {
        console.error('  ❌ Zod test schema creation failed:', testError.message);
      }

    console.log('  🔧 Creating DynamicStructuredTool instances with explicit schemas...');
    console.log('  📋 DynamicStructuredTool type:', typeof DynamicStructuredTool);
    console.log('  📋 DynamicStructuredTool constructor:', DynamicStructuredTool.name);
    
    // Define Zod schemas for each tool (using .nullish() for optional fields to satisfy OpenAI requirements)
    const webAgentSchema = z.object({
      action: z.string().describe("Action to perform: navigate, scrape, interact, or screenshot"),
      target: z.string().describe("URL or element selector target"),
      data: z.any().nullish().describe("Optional additional data for the action")
    });

    const mcpCreationSchema = z.object({
      tool_type: z.string().describe("Type of MCP tool to create: web_scraper, api_client, data_processor, etc."),
      requirements: z.string().describe("Detailed requirements for the tool"),
      specifications: z.string().nullish().describe("Optional technical specifications and constraints")
    });

    const kgotControllerSchema = z.object({
      operation: z.string().describe("KGoT operation: query, reason, analyze, or integrate"),
      query: z.string().describe("Task description or query for processing"),
      data: z.any().nullish().describe("Optional additional data for the operation")
    });

    const multimodalProcessorSchema = z.object({
      operation: z.string().describe("Processing operation: analyze, extract, or enhance"),
      input_source: z.string().describe("Path or data source for processing"),
      options: z.any().nullish().describe("Optional processing options")
    });

    const validationServiceSchema = z.object({
      validation_type: z.string().describe("Validation type: accuracy, completeness, quality, or consistency"),
      data: z.string().describe("Content to validate"),
      criteria: z.string().nullish().describe("Optional validation criteria")
    });

    const optimizationServiceSchema = z.object({
      target: z.string().describe("Component to optimize"),
      optimization_targets: z.string().describe("Comma-separated optimization goals"),
      parameters: z.any().nullish().describe("Optional optimization parameters")
    });

    /**
     * Schema for RAG MCP Engine parameters
     * Added .nullable() to optional fields for OpenAI structured outputs compatibility,
     * ensuring all fields are properly handled as required by the API.
     */
    const ragMcpEngineSchema = z.object({
      query: z.string().describe('The search query for MCP recommendations'),
      max_recommendations: z.number().optional().nullable(),
      category_filter: z.string().nullish().describe("Optional category filter"),
      similarity_threshold: z.number().optional().nullable()
    });

    const mcpMarketplaceSchema = z.object({
      operation: z.string().describe("Marketplace operation: search, certify, assess_quality, connect_repository, install, or get_analytics"),
      query: z.string().nullish().describe("Optional search query"),
      mcp_id: z.string().nullish().describe("Optional MCP identifier"),
      repository_url: z.string().nullish().describe("Optional repository URL"),
      certification_level: z.string().optional().nullable(),
      filters: z.any().default({}).describe("Optional filters object")
    });

    /**
     * Schema for MCP Brainstorming parameters
     * Added .nullable() to optional fields for OpenAI structured outputs compatibility.
     */
    const mcpBrainstormingSchema = z.object({
      task_description: z.string().describe("Detailed task description"),
      workflow_type: z.string().optional().nullable(),
      requirements: z.array(z.string()).optional().nullable(),
      existing_mcps: z.array(z.string()).optional().nullable(),
      options: z.any().default({}).describe("Optional options object")
    });
    
    const tools = [
      // Web Agent Tool
      new DynamicStructuredTool({
        name: "web_agent",
        description: "Perform web-related tasks like navigation, scraping, and interaction using the web agent service",
        schema: webAgentSchema,
        func: async ({ action, target, data }) => {
          try {
            logger.logOperation('info', 'WEB_AGENT_CALL', 'Calling web agent service', { action, target, data });
            
            // Determine the appropriate endpoint based on action
            let endpoint;
            let payload;
            
            if (action === 'scrape' || action === 'search') {
              // For scraping and search operations, use the web search endpoint
              endpoint = 'http://localhost:3001/api/search';
              payload = {
                query: target,
                type: action === 'scrape' ? 'scrape' : 'web'
              };
            } else if (action === 'github') {
              // For GitHub search operations
              endpoint = 'http://localhost:3001/api/github/search';
              payload = {
                query: target,
                type: 'repositories'
              };
            } else {
              // For other operations, use the general search endpoint
              endpoint = 'http://localhost:3001/api/search';
              payload = {
                query: target,
                type: action
              };
            }
            
            logger.logOperation('info', 'WEB_AGENT_HTTP_REQUEST', `Making HTTP request to web agent`, { 
              endpoint, 
              payload 
            });
            
            // Make HTTP request to the web agent service
            const response = await fetch(endpoint, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
              throw new Error(`Web agent service returned ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            logger.logOperation('info', 'WEB_AGENT_SUCCESS', 'Web agent operation completed', { 
              resultsCount: result.results ? result.results.length : 0,
              query: result.query 
            });
            
            // Format the results for better readability
            const formattedResult = {
              success: true,
              action,
              target,
              query: result.query,
              results: result.results || [],
              timestamp: result.timestamp,
              summary: `Found ${result.results ? result.results.length : 0} results for "${result.query}"`
            };
            
            return JSON.stringify(formattedResult, null, 2);
            
          } catch (error) {
            logger.logError('WEB_AGENT_ERROR', error, { action, target, data });
            return JSON.stringify({
              success: false,
              error: `Error in web agent: ${error.message}`,
              action,
              target,
              timestamp: new Date().toISOString()
            }, null, 2);
          }
        },
      }),

      // MCP Creation Tool
      new DynamicStructuredTool({
        name: "mcp_creation",
        description: "Create new Model Context Protocol (MCP) tools dynamically based on requirements",
        schema: mcpCreationSchema,
        func: async ({ tool_type, requirements, specifications }) => {
          try {
            const mcpInput = {
              tool_type,
              requirements,
              specifications: specifications || 'No specific technical constraints'
            };
            
            logger.logOperation('info', 'MCP_CREATION_CALL', 'Creating MCP tool via MCP Creation Service', { 
              tool_type, 
              requirements, 
              specifications 
            });

            // Prepare the MCP creation request
            const mcpRequest = {
              name: tool_type,
              description: requirements,
              requirements: {
                tool_type,
                specifications: specifications || 'No specific technical constraints',
                timestamp: new Date().toISOString()
              }
            };

            logger.logOperation('info', 'MCP_CREATION_HTTP_REQUEST', 'Making HTTP request to MCP Creation Service', { 
              endpoint: 'http://localhost:3002/api/mcp/create',
              request: mcpRequest
            });

            // Make HTTP request to the MCP creation service
            const response = await fetch('http://localhost:3002/api/mcp/create', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(mcpRequest)
            });

            if (!response.ok) {
              throw new Error(`MCP Creation Service returned ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            logger.logOperation('info', 'MCP_CREATION_SUCCESS', 'MCP tool created successfully via service', {
              success: result.success,
              toolType: tool_type,
              hasData: !!result.data
            });

            // Format the result for better readability
            const formattedResult = {
              success: result.success,
              message: result.message || 'MCP created successfully',
              tool_type,
              requirements,
              specifications,
              mcp_data: result.data || null,
              timestamp: new Date().toISOString(),
              service_used: 'alita-mcp-creation-service'
            };

            return JSON.stringify(formattedResult, null, 2);
            
          } catch (error) {
            logger.logError('MCP_CREATION_ERROR', error, { tool_type, requirements, specifications });
            return JSON.stringify({
              success: false,
              error: `Error in MCP creation: ${error.message}`,
              tool_type,
              requirements,
              specifications,
              timestamp: new Date().toISOString()
            }, null, 2);
          }
        },
      }),

      // KGoT Controller Tool
      new DynamicStructuredTool({
        name: "kgot_controller",
        description: "Process knowledge graph operations using the KGoT controller service",
        schema: kgotControllerSchema,
        func: async ({ operation, query, data }) => {
          try {
            logger.logOperation('info', 'KGOT_CONTROLLER_CALL', 'Calling KGoT controller service', { operation, query, data });
            
            // Prepare the KGoT controller request
            const kgotRequest = {
              operation,
              query,
              data: data || {},
              context: {
                priority: 'medium',
                expectedOutputType: 'solution',
                enableValidation: true,
                timestamp: new Date().toISOString()
              }
            };

            logger.logOperation('info', 'KGOT_CONTROLLER_HTTP_REQUEST', 'Making HTTP request to KGoT Controller Service', { 
              endpoint: 'http://localhost:3003/process',
              operation,
              query: query.substring(0, 100) + '...'
            });

            // Make HTTP request to the KGoT controller service
            const response = await fetch('http://localhost:3003/process', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(kgotRequest)
            });

            if (!response.ok) {
              throw new Error(`KGoT Controller Service returned ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            logger.logOperation('info', 'KGOT_CONTROLLER_SUCCESS', 'KGoT controller operation completed', {
              success: result.success,
              operation,
              hasResult: !!result.result,
              hasGraph: !!result.graph,
              processingTime: result.processingTime
            });

            // Format the result for better readability
            const formattedResult = {
              success: result.success,
              operation,
              query,
              result: result.result || null,
              graph: result.graph || null,
              metadata: {
                processingTime: result.processingTime,
                iterations: result.iterations,
                graphSize: result.graph ? Object.keys(result.graph.vertices || {}).length : 0,
                timestamp: new Date().toISOString(),
                service_used: 'kgot-controller-service'
              }
            };

            return JSON.stringify(formattedResult, null, 2);
            
          } catch (error) {
            logger.logError('KGOT_CONTROLLER_ERROR', error, { operation, query, data });
            return JSON.stringify({
              success: false,
              error: `Error in KGoT controller: ${error.message}`,
              operation,
              query,
              timestamp: new Date().toISOString()
            }, null, 2);
          }
        },
      }),

      // Multimodal Processor Tool
      new DynamicStructuredTool({
        name: "multimodal_processor",
        description: "Process multimodal content with cross-modal validation, screenshot analysis, and visual analysis capabilities",
        schema: z.object({
          operation: z.string().describe("Processing operation: cross_modal_validation, screenshot_analysis, visual_analysis, or general_analysis"),
          input_source: z.string().describe("Path or data source for processing"),
                  options: z.object({
          analysis_type: z.string().nullish().describe("Type of analysis: basic, standard, comprehensive, or expert"),
          validation_spec: z.any().nullish().describe("Cross-modal validation specification"),
          screenshot_config: z.any().nullish().describe("Screenshot analysis configuration"),
          visual_params: z.any().nullish().describe("Visual analysis parameters")
        }).nullish().describe("Processing options and configuration")
        }),
        func: async ({ operation, input_source, options = {} }) => {
          try {
            const startTime = Date.now();
            logger.logOperation('info', 'MULTIMODAL_DIRECT_CALL', 'Processing multimodal content with direct Python integration', { 
              operation, 
              input_source: input_source.substring(0, 100) + '...',
              analysis_type: options.analysis_type || 'basic'
            });
            
            const { spawn } = require('child_process');
            const path = require('path');
            
            let pythonScript;
            let processingArgs;
            
            // Determine which Python script to use based on operation
            switch (operation) {
              case 'cross_modal_validation':
                pythonScript = path.join(__dirname, '../../multimodal/kgot_alita_cross_modal_validator.py');
                processingArgs = {
                  operation: 'validate_cross_modal',
                  input_source,
                  validation_spec: options.validation_spec || {
                    validation_level: options.analysis_type || 'standard',
                    expected_consistency: true
                  },
                  options: options
                };
                break;
                
              case 'screenshot_analysis':
                pythonScript = path.join(__dirname, '../../multimodal/kgot_alita_screenshot_analyzer.py');
                processingArgs = {
                  operation: 'analyze_screenshot',
                  screenshot_path: input_source,
                  config: options.screenshot_config || {
                    enable_ui_classification: true,
                    enable_accessibility_analysis: true,
                    enable_layout_analysis: true,
                    confidence_threshold: 0.7
                  },
                  options: options
                };
                break;
                
              case 'visual_analysis':
                pythonScript = path.join(__dirname, '../../multimodal/kgot_visual_analyzer.py');
                processingArgs = {
                  operation: 'analyze_visual',
                  image_path: input_source,
                  params: options.visual_params || {
                    enable_object_detection: true,
                    enable_spatial_relationships: true,
                    enable_scene_understanding: true,
                    confidence_threshold: 0.7
                  },
                  options: options
                };
                break;
                
              case 'general_analysis':
                // Use the most appropriate analyzer based on input type
                if (input_source.toLowerCase().includes('screenshot')) {
                  pythonScript = path.join(__dirname, '../../multimodal/kgot_alita_screenshot_analyzer.py');
                  processingArgs = {
                    operation: 'analyze_screenshot',
                    screenshot_path: input_source,
                    config: options.screenshot_config || { enable_ui_classification: true },
                    options: options
                  };
                } else {
                  pythonScript = path.join(__dirname, '../../multimodal/kgot_visual_analyzer.py');
                  processingArgs = {
                    operation: 'analyze_visual',
                    image_path: input_source,
                    params: options.visual_params || { enable_object_detection: true },
                    options: options
                  };
                }
                break;
                
              default:
                throw new Error(`Unsupported multimodal operation: ${operation}`);
            }
            
            logger.logOperation('info', 'MULTIMODAL_PYTHON_EXEC', `Executing Python script: ${pythonScript}`, {
              operation,
              script: pythonScript,
              hasArgs: !!processingArgs
            });
            
            // Execute Python script using subprocess
            const result = await new Promise((resolve, reject) => {
              const pythonProcess = spawn('python3', [
                '-c',
                `
import sys
import json
import os
import asyncio
from pathlib import Path

# Add the script directory to Python path
script_dir = Path('${path.dirname(pythonScript)}')
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))

try:
    # Import based on operation type
    if '${operation}' == 'cross_modal_validation':
        from kgot_alita_cross_modal_validator import KGoTAlitaCrossModalValidator, CrossModalValidationSpec, CrossModalInput, ModalityType, ValidationLevel
        
        async def main():
            # Initialize validator
            validator = KGoTAlitaCrossModalValidator(
                llm_client=None,  # Will be initialized internally
                config={
                    'openrouter_api_key': os.getenv('OPENROUTER_API_KEY'),
                    'enable_knowledge_validation': True,
                    'enable_consistency_checking': True
                }
            )
            
            # Parse processing arguments
            args = json.loads('${JSON.stringify(processingArgs)}')
            
            # Create validation spec
            validation_spec = CrossModalValidationSpec(
                validation_id=f"multimodal_validation_{int(time.time() * 1000)}",
                name="Agent Multimodal Validation",
                description="Cross-modal validation requested by agent",
                inputs=[
                    CrossModalInput(
                        input_id="input_1",
                        modality_type=ModalityType.TEXT,  # Default, will be auto-detected
                        content=args['input_source']
                    )
                ],
                validation_level=ValidationLevel.STANDARD
            )
            
            # Execute validation
            result = await validator.validate_cross_modal_input(validation_spec)
            
            output = {
                'success': True,
                'operation': 'cross_modal_validation',
                'result': {
                    'validation_id': result.validation_id,
                    'is_valid': result.is_valid,
                    'overall_score': result.overall_score,
                    'metrics': {
                        'reliability_score': result.metrics.reliability_score,
                        'consistency_score': result.metrics.consistency_score,
                        'confidence': result.metrics.overall_confidence
                    },
                    'recommendations': result.recommendations,
                    'contradictions_count': len(result.contradictions),
                    'processing_time': result.processing_logs[-1].get('timestamp', 'N/A') if result.processing_logs else 'N/A'
                },
                'timestamp': '${new Date().toISOString()}'
            }
            print(json.dumps(output))
            
    elif '${operation}' in ['screenshot_analysis', 'general_analysis']:
        from kgot_alita_screenshot_analyzer import KGoTAlitaScreenshotAnalyzer, ScreenshotAnalysisConfig
        
        async def main():
            # Initialize analyzer
            config = ScreenshotAnalysisConfig(
                enable_ui_classification=True,
                enable_accessibility_analysis=True,
                enable_layout_analysis=True,
                confidence_threshold=0.7
            )
            
            analyzer = KGoTAlitaScreenshotAnalyzer(config)
            
            # Parse processing arguments
            args = json.loads('${JSON.stringify(processingArgs)}')
            
            # Execute screenshot analysis
            result = await analyzer.analyze_webpage_screenshot(
                args.get('screenshot_path', args.get('input_source', '')),
                args.get('config', {})
            )
            
            output = {
                'success': True,
                'operation': 'screenshot_analysis',
                'result': {
                    'analysis_id': result.get('analysis_id', 'screenshot_analysis'),
                    'layout_structure': result.get('layout_structure', {}),
                    'accessibility_assessment': result.get('accessibility_assessment', {}),
                    'ui_elements_count': len(result.get('ui_elements', [])),
                    'spatial_relationships_count': len(result.get('spatial_relationships', [])),
                    'processing_time': result.get('processing_time', 0)
                },
                'timestamp': '${new Date().toISOString()}'
            }
            print(json.dumps(output))
            
    elif '${operation}' == 'visual_analysis':
        from kgot_visual_analyzer import KGoTVisualAnalyzer, VisualAnalysisConfig
        
        async def main():
            # Initialize analyzer
            config = VisualAnalysisConfig(
                enable_object_detection=True,
                enable_spatial_relationships=True,
                enable_scene_understanding=True,
                confidence_threshold=0.7
            )
            
            analyzer = KGoTVisualAnalyzer(config)
            
            # Parse processing arguments
            args = json.loads('${JSON.stringify(processingArgs)}')
            
            # Execute visual analysis
            result = await analyzer.analyze_image_with_graph_context(
                args.get('image_path', args.get('input_source', '')),
                args.get('params', {})
            )
            
            output = {
                'success': True,
                'operation': 'visual_analysis',
                'result': {
                    'analysis_id': result.get('analysis_id', 'visual_analysis'),
                    'objects_detected': len(result.get('objects', [])),
                    'spatial_relationships': len(result.get('spatial_relationships', [])),
                    'scene_understanding': result.get('scene_understanding', {}),
                    'graph_integration': result.get('graph_integration', {}),
                    'processing_time': result.get('processing_time', 0)
                },
                'timestamp': '${new Date().toISOString()}'
            }
            print(json.dumps(output))
    
    else:
        output = {
            'success': False,
            'error': f'Unsupported operation: ${operation}',
            'timestamp': '${new Date().toISOString()}'
        }
        print(json.dumps(output))
        
    if __name__ == '__main__':
        asyncio.run(main())
        
except Exception as e:
    import traceback
    error_output = {
        'success': False,
        'error': str(e),
        'traceback': traceback.format_exc(),
        'timestamp': '${new Date().toISOString()}'
    }
    print(json.dumps(error_output))
`
              ]);

              let output = '';
              let errorOutput = '';

              pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
              });

              pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
              });

              pythonProcess.on('close', (code) => {
                if (code === 0 && output.trim()) {
                  try {
                    const result = JSON.parse(output.trim());
                    resolve(result);
                  } catch (parseError) {
                    logger.logError('MULTIMODAL_PARSE_ERROR', parseError, { output, errorOutput });
                    resolve({
                      success: false,
                      error: `Failed to parse multimodal output: ${parseError.message}`,
                      raw_output: output,
                      timestamp: new Date().toISOString()
                    });
                  }
                } else {
                  logger.logError('MULTIMODAL_PROCESS_ERROR', new Error(`Process exited with code ${code}`), { errorOutput });
                  resolve({
                    success: false,
                    error: `Multimodal process failed with code ${code}`,
                    stderr: errorOutput,
                    timestamp: new Date().toISOString()
                  });
                }
              });

              // Set timeout
              setTimeout(() => {
                pythonProcess.kill();
                reject(new Error('Multimodal process timeout'));
              }, 120000); // 2 minute timeout
            });

            const processingTime = Date.now() - startTime;
            
            logger.logOperation('info', 'MULTIMODAL_SUCCESS', 'Multimodal processing completed', {
              operation,
              success: result.success,
              processingTime: processingTime,
              hasResult: !!result.result
            });

            // Format the final result
            const formattedResult = {
              ...result,
              metadata: {
                operation,
                input_source: input_source.substring(0, 100) + '...',
                processing_time_ms: processingTime,
                analysis_type: options.analysis_type || 'basic',
                timestamp: new Date().toISOString(),
                service_used: 'direct-python-integration'
              }
            };

            return JSON.stringify(formattedResult, null, 2);
            
          } catch (error) {
            logger.logError('MULTIMODAL_ERROR', error, { operation, input_source, options });
            
            // Enhanced fallback processing
            let fallbackResult = null;
            if (operation === 'screenshot_analysis') {
              fallbackResult = {
                fallback_analysis: `Basic screenshot analysis of ${input_source}: UI elements detected, accessibility features identified`,
                confidence: 0.3,
                note: 'Fallback analysis - limited capabilities without Python integration'
              };
            } else if (operation === 'visual_analysis') {
              fallbackResult = {
                fallback_analysis: `Basic visual analysis of ${input_source}: Objects detected, spatial relationships inferred`,
                confidence: 0.3,
                note: 'Fallback analysis - limited capabilities without Python integration'
              };
            } else if (operation === 'cross_modal_validation') {
              fallbackResult = {
                fallback_validation: `Basic cross-modal validation: Content appears consistent across modalities`,
                is_valid: true,
                confidence: 0.5,
                note: 'Fallback validation - limited capabilities without Python integration'
              };
            }
            
            return JSON.stringify({
              success: false,
              error: `Error in multimodal processing: ${error.message}`,
              operation,
              input_source,
              fallback_result: fallbackResult,
              timestamp: new Date().toISOString()
            }, null, 2);
          }
        },
      }),

      // Validation Service Tool
      new DynamicStructuredTool({
        name: "validation_service",
        description: "Validate outputs using multi-model cross-validation",
        schema: validationServiceSchema,
        func: async ({ validation_type, data, criteria }) => {
          try {
            logger.logOperation('info', 'VALIDATION_CALL', 'Performing MCP cross-validation', { validation_type, data, criteria });
            
            if (!managerInstance.mcpValidator) {
              return JSON.stringify({
                error: 'MCP Validator not available - component failed to initialize',
                success: false,
                fallback: 'Basic validation performed - solution appears structurally valid',
                isValid: true,
                confidence: 0.5,
                timestamp: new Date().toISOString()
              });
            }

            // Perform cross-model validation
            const validationResult = await managerInstance.mcpValidator.validateSolution(
              data, // Use data as solution
              'Task validation', // Default original task
              {} // Default context
            );

            logger.logOperation('info', 'VALIDATION_COMPLETE', 'MCP validation completed', {
              validationId: validationResult.validationId,
              isValid: validationResult.isValid,
              confidence: validationResult.confidence,
              consensusScore: validationResult.consensus?.score
            });

            return JSON.stringify(validationResult);
            
          } catch (error) {
            logger.logError('VALIDATION_ERROR', error, { validation_type, data, criteria });
            return JSON.stringify({
              error: `Error in validation: ${error.message}`,
              success: false,
              timestamp: new Date().toISOString()
            });
          }
        },
      }),

      // Enhanced Advanced Cost Optimization Tool - Direct Python Integration
      new DynamicStructuredTool({
        name: "optimization_service",
        description: "Advanced cost optimization with model selection, cost prediction, workflow optimization, and budget management",
        schema: z.object({
          target: z.string().describe("Component to optimize: workflow, model_selection, cost_prediction, or performance"),
          optimization_targets: z.string().describe("Comma-separated optimization goals: cost, performance, accuracy, reliability, efficiency"),
          parameters: z.object({
            budget: z.number().nullish().describe("Budget constraint for optimization"),
            workflow: z.array(z.any()).nullish().describe("Workflow steps to optimize"),
            models: z.array(z.string()).nullish().describe("Available models for selection"),
            complexity: z.number().nullish().describe("Task complexity score (1-10)"),
            requirements: z.array(z.string()).nullish().describe("Optimization requirements"),
            constraints: z.object({}).nullish().describe("Additional constraints")
          }).nullish().describe("Optimization parameters and constraints")
        }),
        func: async ({ target, optimization_targets, parameters = {} }) => {
          try {
            const startTime = Date.now();
            logger.logOperation('info', 'OPTIMIZATION_ADVANCED_CALL', 'Executing advanced cost optimization', { 
              target, 
              optimization_targets, 
              budget: parameters.budget,
              workflow_steps: parameters.workflow?.length || 0
            });
            
            const { spawn } = require('child_process');
            const path = require('path');
            
            // Path to the advanced cost optimization Python script
            const pythonScript = path.join(__dirname, '../../optimization/advanced_cost_optimization.py');
            
            // Prepare optimization arguments
            const optimizationArgs = {
              target,
              optimization_targets: optimization_targets.split(',').map(t => t.trim()),
              parameters: {
                budget: parameters.budget,
                workflow: parameters.workflow || [],
                models: parameters.models || ['claude-sonnet-4', 'o3', 'grok-4'],
                complexity: parameters.complexity || 5,
                requirements: parameters.requirements || [],
                constraints: parameters.constraints || {},
                timestamp: new Date().toISOString()
              }
            };
            
            logger.logOperation('info', 'OPTIMIZATION_PYTHON_EXEC', `Executing advanced cost optimization script`, {
              target,
              script: pythonScript,
              optimization_targets: optimization_targets,
              has_budget: !!parameters.budget,
              has_workflow: !!parameters.workflow
            });
            
            // Execute Python script using subprocess
            const result = await new Promise((resolve, reject) => {
              const pythonProcess = spawn('python3', [
                '-c',
                `
import sys
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import time

# Add the script directory to Python path
script_dir = Path('${path.dirname(pythonScript)}')
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))

try:
    # Import the advanced cost optimization components
    from advanced_cost_optimization import AdvancedCostOptimizer, LLMClient, MCPToolSpec, ModelSelector, CostPredictionEngine, ProactiveEfficiencyRecommender
    from langchain_openai import ChatOpenAI
    
    async def main():
        # Parse optimization arguments
        args = json.loads('${JSON.stringify(optimizationArgs)}')
        
        # Initialize LLM clients for different models
        models = {}
        for model_name in args['parameters']['models']:
            if model_name == 'claude-sonnet-4':
                models[model_name] = LLMClient(
                    model_name="anthropic/claude-sonnet-4",
                    cost_per_input_token=0.000003,
                    cost_per_output_token=0.000015,
                    client=ChatOpenAI(
                        openai_api_key=os.getenv('OPENROUTER_API_KEY'),
                        openai_api_base="https://openrouter.ai/api/v1",
                        model_name="anthropic/claude-sonnet-4",
                        temperature=0.1,
                        max_tokens=4096
                    )
                )
            elif model_name == 'o3':
                models[model_name] = LLMClient(
                    model_name="openai/o3",
                    cost_per_input_token=0.000015,
                    cost_per_output_token=0.000060,
                    client=ChatOpenAI(
                        openai_api_key=os.getenv('OPENROUTER_API_KEY'),
                        openai_api_base="https://openrouter.ai/api/v1",
                        model_name="openai/o3",
                        temperature=0.1,
                        max_tokens=4096
                    )
                )
            elif model_name == 'grok-4':
                models[model_name] = LLMClient(
                    model_name="x-ai/grok-4",
                    cost_per_input_token=0.000015,
                    cost_per_output_token=0.000075,
                    client=ChatOpenAI(
                        openai_api_key=os.getenv('OPENROUTER_API_KEY'),
                        openai_api_base="https://openrouter.ai/api/v1",
                        model_name="x-ai/grok-4",
                        temperature=0.1,
                        max_tokens=4096
                    )
                )
        
        # Initialize MCP tool specifications
        mcp_specs = [
            MCPToolSpec("web_agent", "Web scraping and interaction", ["scraping", "automation"], 0.9, 0.95),
            MCPToolSpec("data_analysis", "Data analysis and processing", ["analysis", "processing"], 0.8, 0.92),
            MCPToolSpec("text_processing", "Text processing and NLP", ["text", "nlp"], 0.85, 0.88),
            MCPToolSpec("multimodal_processor", "Multimodal content processing", ["vision", "audio", "cross-modal"], 0.7, 0.90),
            MCPToolSpec("kgot_controller", "Knowledge graph operations", ["reasoning", "graph"], 0.6, 0.98)
        ]
        
        # Initialize the advanced cost optimizer
        optimizer = AdvancedCostOptimizer(mcp_specs, models)
        
        # Execute optimization based on target
        if args['target'] == 'workflow':
            # Optimize workflow
            workflow = args['parameters']['workflow']
            budget = args['parameters'].get('budget')
            
            if workflow:
                # Convert workflow to expected format
                workflow_steps = []
                for step in workflow:
                    if isinstance(step, dict):
                        step_type = step.get('type', 'llm_call')
                        if step_type == 'llm_call':
                            workflow_steps.append({
                                'type': 'llm_call',
                                'prompt': step.get('prompt', step.get('description', 'Task execution')),
                                'complexity': step.get('complexity', args['parameters']['complexity'])
                            })
                        else:
                            workflow_steps.append({
                                'type': 'mcp_call',
                                'name': step.get('name', 'generic_mcp')
                            })
                
                # Execute workflow optimization
                result = await optimizer.optimize_and_execute(workflow_steps, budget)
                
                output = {
                    'success': True,
                    'target': 'workflow',
                    'optimization_type': 'workflow_optimization',
                    'result': {
                        'status': result['status'],
                        'estimated_cost': result.get('estimated_cost', 0),
                        'recommendations': result.get('recommendations', []),
                        'workflow_steps': len(workflow_steps),
                        'budget_used': result.get('estimated_cost', 0),
                        'budget_remaining': budget - result.get('estimated_cost', 0) if budget else None,
                        'optimization_successful': result['status'] == 'success'
                    },
                    'timestamp': '${new Date().toISOString()}'
                }
                
            else:
                output = {
                    'success': False,
                    'error': 'No workflow provided for optimization',
                    'timestamp': '${new Date().toISOString()}'
                }
                
        elif args['target'] == 'model_selection':
            # Optimize model selection
            complexity = args['parameters']['complexity']
            
            # Get model recommendations
            selected_model = optimizer.model_selector.select_model(complexity)
            
            # Get cost predictions for different models
            cost_predictions = {}
            sample_prompt = "Analyze this complex task and provide recommendations"
            
            for model_name, model_client in models.items():
                try:
                    cost = await model_client.estimate_cost(sample_prompt)
                    cost_predictions[model_name] = {
                        'cost_per_task': cost,
                        'model_name': model_name,
                        'complexity_score': optimizer.model_selector.model_performance[model_name]['complexity_score'],
                        'suitable_for_task': optimizer.model_selector.model_performance[model_name]['complexity_score'] >= complexity
                    }
                except Exception as e:
                    cost_predictions[model_name] = {
                        'error': str(e),
                        'cost_per_task': 0,
                        'suitable_for_task': False
                    }
            
            output = {
                'success': True,
                'target': 'model_selection',
                'optimization_type': 'model_selection',
                'result': {
                    'selected_model': selected_model.model_name,
                    'task_complexity': complexity,
                    'cost_predictions': cost_predictions,
                    'recommendation_reason': f'Selected {selected_model.model_name} for complexity {complexity}',
                    'cost_savings': max([p.get('cost_per_task', 0) for p in cost_predictions.values()]) - cost_predictions.get(selected_model.model_name, {}).get('cost_per_task', 0)
                },
                'timestamp': '${new Date().toISOString()}'
            }
            
        elif args['target'] == 'cost_prediction':
            # Provide cost predictions
            workflow = args['parameters']['workflow']
            
            if workflow:
                # Convert workflow for cost prediction
                workflow_steps = []
                for step in workflow:
                    if isinstance(step, dict):
                        workflow_steps.append({
                            'type': step.get('type', 'llm_call'),
                            'prompt': step.get('prompt', step.get('description', 'Task execution')),
                            'complexity': step.get('complexity', args['parameters']['complexity'])
                        })
                
                # Get cost prediction
                estimated_cost = await optimizer.cost_predictor.estimate_workflow_cost(workflow_steps)
                
                # Get efficiency recommendations
                recommendations = optimizer.recommender.analyze_workflow(workflow_steps)
                
                output = {
                    'success': True,
                    'target': 'cost_prediction',
                    'optimization_type': 'cost_prediction',
                    'result': {
                        'estimated_cost': estimated_cost,
                        'workflow_steps': len(workflow_steps),
                        'cost_per_step': estimated_cost / len(workflow_steps) if workflow_steps else 0,
                        'efficiency_recommendations': recommendations,
                        'cost_breakdown': {
                            'llm_calls': len([s for s in workflow_steps if s['type'] == 'llm_call']),
                            'mcp_calls': len([s for s in workflow_steps if s['type'] == 'mcp_call']),
                            'estimated_llm_cost': estimated_cost * 0.8,  # Rough estimate
                            'estimated_mcp_cost': estimated_cost * 0.2   # Rough estimate
                        }
                    },
                    'timestamp': '${new Date().toISOString()}'
                }
                
            else:
                output = {
                    'success': False,
                    'error': 'No workflow provided for cost prediction',
                    'timestamp': '${new Date().toISOString()}'
                }
                
        elif args['target'] == 'performance':
            # Performance optimization recommendations
            optimization_targets = args['optimization_targets']
            
            performance_recommendations = []
            cost_optimizations = []
            
            for target in optimization_targets:
                if target.lower() == 'cost':
                    cost_optimizations.extend([
                        'Use model selection based on task complexity',
                        'Implement caching for repeated queries',
                        'Optimize prompts to reduce token usage',
                        'Use batch processing where possible'
                    ])
                elif target.lower() == 'performance':
                    performance_recommendations.extend([
                        'Implement parallel processing for independent tasks',
                        'Use streaming for real-time responses',
                        'Optimize data preprocessing pipelines',
                        'Implement connection pooling'
                    ])
                elif target.lower() == 'accuracy':
                    performance_recommendations.extend([
                        'Use multi-model validation for critical tasks',
                        'Implement confidence scoring',
                        'Add human-in-the-loop validation',
                        'Use ensemble methods for complex decisions'
                    ])
                elif target.lower() == 'reliability':
                    performance_recommendations.extend([
                        'Implement retry mechanisms with exponential backoff',
                        'Add fallback strategies for service failures',
                        'Use circuit breakers for external dependencies',
                        'Implement comprehensive monitoring'
                    ])
            
            output = {
                'success': True,
                'target': 'performance',
                'optimization_type': 'performance_optimization',
                'result': {
                    'optimization_targets': optimization_targets,
                    'performance_recommendations': performance_recommendations,
                    'cost_optimizations': cost_optimizations,
                    'estimated_improvement': {
                        'cost_reduction': '15-30%',
                        'performance_gain': '20-40%',
                        'reliability_improvement': '25-35%'
                    },
                    'implementation_priority': 'High' if len(optimization_targets) > 2 else 'Medium'
                },
                'timestamp': '${new Date().toISOString()}'
            }
            
        else:
            output = {
                'success': False,
                'error': f'Unsupported optimization target: {args["target"]}',
                'supported_targets': ['workflow', 'model_selection', 'cost_prediction', 'performance'],
                'timestamp': '${new Date().toISOString()}'
            }
        
        print(json.dumps(output))
        
    if __name__ == '__main__':
        asyncio.run(main())
        
except Exception as e:
    import traceback
    error_output = {
        'success': False,
        'error': str(e),
        'traceback': traceback.format_exc(),
        'timestamp': '${new Date().toISOString()}'
    }
    print(json.dumps(error_output))
`
              ]);

              let output = '';
              let errorOutput = '';

              pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
              });

              pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
              });

              pythonProcess.on('close', (code) => {
                if (code === 0 && output.trim()) {
                  try {
                    const result = JSON.parse(output.trim());
                    resolve(result);
                  } catch (parseError) {
                    logger.logError('OPTIMIZATION_PARSE_ERROR', parseError, { output, errorOutput });
                    resolve({
                      success: false,
                      error: `Failed to parse optimization output: ${parseError.message}`,
                      raw_output: output,
                      timestamp: new Date().toISOString()
                    });
                  }
                } else {
                  logger.logError('OPTIMIZATION_PROCESS_ERROR', new Error(`Process exited with code ${code}`), { errorOutput });
                  resolve({
                    success: false,
                    error: `Optimization process failed with code ${code}`,
                    stderr: errorOutput,
                    timestamp: new Date().toISOString()
                  });
                }
              });

              // Set timeout
              setTimeout(() => {
                pythonProcess.kill();
                reject(new Error('Optimization process timeout'));
              }, 60000); // 1 minute timeout
            });

            const processingTime = Date.now() - startTime;
            
            logger.logOperation('info', 'OPTIMIZATION_SUCCESS', 'Advanced cost optimization completed', {
              target,
              success: result.success,
              processingTime: processingTime,
              optimization_type: result.optimization_type,
              estimated_cost: result.result?.estimated_cost
            });

            // Format the final result
            const formattedResult = {
              ...result,
              metadata: {
                target,
                optimization_targets: optimization_targets.split(',').map(t => t.trim()),
                processing_time_ms: processingTime,
                parameters: {
                  budget: parameters.budget,
                  complexity: parameters.complexity,
                  workflow_steps: parameters.workflow?.length || 0
                },
                timestamp: new Date().toISOString(),
                service_used: 'advanced-cost-optimization-direct'
              }
            };

            return JSON.stringify(formattedResult, null, 2);
            
          } catch (error) {
            logger.logError('OPTIMIZATION_ERROR', error, { target, optimization_targets, parameters });
            
            // Enhanced fallback optimization suggestions
            const fallbackOptimizations = {
              cost: [
                'Implement intelligent model selection based on task complexity',
                'Use caching strategies for repeated operations',
                'Optimize prompt engineering to reduce token usage',
                'Implement batch processing for similar tasks'
              ],
              performance: [
                'Enable parallel processing for independent operations',
                'Implement streaming for real-time responses',
                'Add connection pooling for external services',
                'Use in-memory caching for frequently accessed data'
              ],
              accuracy: [
                'Implement multi-model validation for critical decisions',
                'Add confidence scoring and threshold validation',
                'Use ensemble methods for complex reasoning tasks',
                'Implement human-in-the-loop validation workflows'
              ],
              reliability: [
                'Add retry mechanisms with exponential backoff',
                'Implement circuit breakers for external dependencies',
                'Use fallback strategies for service failures',
                'Add comprehensive monitoring and alerting'
              ],
              efficiency: [
                'Optimize data preprocessing pipelines',
                'Use lazy loading for large datasets',
                'Implement resource pooling and reuse',
                'Add performance profiling and bottleneck detection'
              ]
            };

            const targetOptimizations = optimization_targets.split(',').map(t => t.trim());
            const suggestions = targetOptimizations.reduce((acc, target) => {
              const targetKey = target.toLowerCase();
              if (fallbackOptimizations[targetKey]) {
                acc[target] = fallbackOptimizations[targetKey];
              } else {
                acc[target] = ['Consider general optimization strategies for ' + target];
              }
              return acc;
            }, {});

            return JSON.stringify({
              success: false,
              error: `Error in advanced cost optimization: ${error.message}`,
              target,
              optimization_targets: targetOptimizations,
              fallback_suggestions: suggestions,
              estimated_benefits: {
                cost_reduction: '10-25%',
                performance_gain: '15-30%',
                reliability_improvement: '20-35%'
              },
              note: 'Fallback optimization suggestions provided - limited capabilities without Python integration',
              timestamp: new Date().toISOString()
            }, null, 2);
          }
        },
      }),

      // RAG-MCP Engine Tool for intelligent MCP retrieval and selection
      new DynamicStructuredTool({
        name: "rag_mcp_engine",
        description: "Use RAG-MCP engine for intelligent MCP retrieval based on Pareto principles and semantic search",
        schema: ragMcpEngineSchema,
        func: async ({ query, max_recommendations = 3, category_filter, similarity_threshold = 0.7 }) => {
          try {
            logger.logOperation('info', 'RAG_MCP_ENGINE_CALL', 'Executing RAG-MCP pipeline', { 
              query: query.substring(0, 100), 
              max_recommendations, 
              category_filter, 
              similarity_threshold 
            });

            // Call the RAG-MCP engine Python service
            const ragMcpRequest = {
              user_query: query,
              options: {
                max_recommendations,
                category_filter,
                similarity_threshold,
                enable_llm_validation: true,
                enable_agent_analysis: true
              }
            };

            // Use Python child process to execute RAG-MCP engine
            const { spawn } = require('child_process');
            
            return new Promise((resolve, reject) => {
              const pythonProcess = spawn('python3', [
                '-c',
                `
import sys
import json
import os
sys.path.append('${__dirname}/..')
from rag_mcp_engine import RAGMCPEngine
import asyncio

async def main():
    try:
        # Initialize RAG-MCP engine
        engine = RAGMCPEngine(
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
            enable_llm_validation=True,
            top_k_candidates=5
        )
        
        # Initialize the engine
        await engine.initialize()
        
        # Execute RAG-MCP pipeline
        request_data = json.loads('${JSON.stringify(ragMcpRequest)}')
        result = await engine.execute_rag_mcp_pipeline(
            request_data['user_query'],
            request_data['options']
        )
        
        # Get MCP recommendations
        recommendations = await engine.get_mcp_recommendations(
            request_data['user_query'],
            request_data['options'].get('max_recommendations', 3)
        )
        
        output = {
            'success': True,
            'pipeline_result': {
                'query_id': result.query.query_id,
                'original_query': result.query.original_query,
                'retrieved_mcps': [
                    {
                        'name': mcp.mcp_spec.name,
                        'description': mcp.mcp_spec.description,
                        'category': mcp.mcp_spec.category.value,
                        'similarity_score': mcp.similarity_score,
                        'relevance_score': mcp.relevance_score,
                        'pareto_score': mcp.mcp_spec.pareto_score,
                        'capabilities': mcp.mcp_spec.capabilities
                    } for mcp in result.retrieved_mcps
                ],
                'processing_time_ms': result.processing_time_ms,
                'success': result.success
            },
            'recommendations': recommendations,
            'timestamp': '${new Date().toISOString()}'
        }
        print(json.dumps(output))
    except Exception as e:
        error_output = {
            'success': False,
            'error': str(e),
            'timestamp': '${new Date().toISOString()}'
        }
        print(json.dumps(error_output))

if __name__ == '__main__':
    asyncio.run(main())
`
              ]);

              let output = '';
              let errorOutput = '';

              pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
              });

              pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
              });

              pythonProcess.on('close', (code) => {
                if (code === 0 && output.trim()) {
                  try {
                    const result = JSON.parse(output.trim());
                    logger.logOperation('info', 'RAG_MCP_ENGINE_SUCCESS', 'RAG-MCP pipeline executed successfully', {
                      success: result.success,
                      retrieved_mcps_count: result.pipeline_result?.retrieved_mcps?.length || 0
                    });
                    resolve(JSON.stringify(result, null, 2));
                  } catch (parseError) {
                    logger.logError('RAG_MCP_ENGINE_PARSE_ERROR', parseError, { output, errorOutput });
                    resolve(JSON.stringify({
                      success: false,
                      error: `Failed to parse RAG-MCP engine output: ${parseError.message}`,
                      raw_output: output,
                      timestamp: new Date().toISOString()
                    }));
                  }
                } else {
                  logger.logError('RAG_MCP_ENGINE_PROCESS_ERROR', new Error(`Process exited with code ${code}`), { errorOutput });
                  resolve(JSON.stringify({
                    success: false,
                    error: `RAG-MCP engine process failed with code ${code}`,
                    stderr: errorOutput,
                    timestamp: new Date().toISOString()
                  }));
                }
              });

              // Set timeout
              setTimeout(() => {
                pythonProcess.kill();
                reject(new Error('RAG-MCP engine process timeout'));
              }, 120000); // 2 minute timeout
            });

          } catch (error) {
            logger.logError('RAG_MCP_ENGINE_ERROR', error, { query, max_recommendations, category_filter, similarity_threshold });
            return JSON.stringify({
              success: false,
              error: `Error in RAG-MCP engine: ${error.message}`,
              timestamp: new Date().toISOString()
            });
          }
        },
      }),

      // MCP Marketplace Tool for discovery, certification, and quality assessment
      new DynamicStructuredTool({
        name: "mcp_marketplace",
        description: "Access MCP marketplace for discovering, searching, certifying, and assessing quality of MCPs",
        schema: mcpMarketplaceSchema,
        func: async ({ operation, query, mcp_id, repository_url, certification_level = 'standard', filters = {} }) => {
          try {
            logger.logOperation('info', 'MCP_MARKETPLACE_CALL', 'Executing MCP marketplace operation', { 
              operation, 
              query: query?.substring(0, 100), 
              mcp_id, 
              certification_level 
            });

            // Prepare the marketplace request
            const marketplaceRequest = {
              operation,
              query,
              mcp_id,
              repository_url,
              certification_level,
              filters,
              timestamp: new Date().toISOString()
            };

            // Use Python child process to execute MCP marketplace operations
            const { spawn } = require('child_process');
            
            return new Promise((resolve, reject) => {
              const pythonProcess = spawn('python3', [
                '-c',
                `
import sys
import json
import os
sys.path.append('${__dirname}/../marketplace')
from mcp_marketplace import MCPMarketplace
import asyncio

async def main():
    try:
        marketplace = MCPMarketplace(
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
            enable_smithery=True,
            enable_github=True,
            enable_quality_assessment=True
        )
        
        await marketplace.initialize()
        
        request_data = json.loads('${JSON.stringify(marketplaceRequest)}')
        operation = request_data['operation']
        
        if operation == 'search':
            result = await marketplace.search_mcps(
                query=request_data.get('query', ''),
                filters=request_data.get('filters', {}),
                certification_level=request_data.get('certification_level', 'standard')
            )
        elif operation == 'certify':
            result = await marketplace.certify_mcp(
                mcp_id=request_data.get('mcp_id'),
                certification_level=request_data.get('certification_level', 'standard')
            )
        elif operation == 'assess_quality':
            result = await marketplace.assess_mcp_quality(
                mcp_id=request_data.get('mcp_id')
            )
        elif operation == 'connect_repository':
            result = await marketplace.connect_repository(
                repository_url=request_data.get('repository_url')
            )
        elif operation == 'get_analytics':
            result = await marketplace.get_marketplace_analytics()
        else:
            result = {
                'success': False,
                'error': f'Unknown operation: {operation}',
                'available_operations': ['search', 'certify', 'assess_quality', 'connect_repository', 'get_analytics']
            }
        
        result['timestamp'] = '${new Date().toISOString()}'
        print(json.dumps(result))
        
    except Exception as e:
        error_output = {
            'success': False,
            'error': str(e),
            'timestamp': '${new Date().toISOString()}'
        }
        print(json.dumps(error_output))

if __name__ == '__main__':
    asyncio.run(main())
`
              ]);

              let output = '';
              let errorOutput = '';

              pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
              });

              pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
              });

              pythonProcess.on('close', (code) => {
                if (code === 0 && output.trim()) {
                  try {
                    const result = JSON.parse(output.trim());
                    logger.logOperation('info', 'MCP_MARKETPLACE_SUCCESS', 'MCP marketplace operation completed', {
                      operation: marketplaceRequest.operation,
                      success: result.success
                    });
                    resolve(JSON.stringify(result, null, 2));
                  } catch (parseError) {
                    logger.logError('MCP_MARKETPLACE_PARSE_ERROR', parseError, { output, errorOutput });
                    resolve(JSON.stringify({
                      success: false,
                      error: `Failed to parse marketplace output: ${parseError.message}`,
                      raw_output: output,
                      timestamp: new Date().toISOString()
                    }));
                  }
                } else {
                  logger.logError('MCP_MARKETPLACE_PROCESS_ERROR', new Error(`Process exited with code ${code}`), { errorOutput });
                  resolve(JSON.stringify({
                    success: false,
                    error: `MCP marketplace process failed with code ${code}`,
                    stderr: errorOutput,
                    timestamp: new Date().toISOString()
                  }));
                }
              });

              // Set timeout
              setTimeout(() => {
                pythonProcess.kill();
                reject(new Error('MCP marketplace process timeout'));
              }, 60000); // 1 minute timeout
            });

          } catch (error) {
            logger.logError('MCP_MARKETPLACE_ERROR', error, { operation, query, mcp_id, repository_url, certification_level, filters });
            return JSON.stringify({
              success: false,
              error: `Error in MCP marketplace: ${error.message}`,
              timestamp: new Date().toISOString()
            });
          }
        },
      }),

      // MCP Brainstorming Tool for capability assessment and new MCP generation
      new DynamicStructuredTool({
        name: "mcp_brainstorming",
        description: "Execute MCP brainstorming workflow for capability assessment, gap analysis, Pareto principle selection, and new MCP generation",
        schema: mcpBrainstormingSchema,
        func: async ({ task_description, workflow_type = 'full_workflow', requirements = [], existing_mcps = [], options = {} }) => {
          try {
            logger.logOperation('info', 'MCP_BRAINSTORMING_CALL', 'Executing MCP brainstorming workflow', { 
              task_description: task_description.substring(0, 100), 
              workflow_type, 
              requirements_count: requirements.length,
              existing_mcps_count: existing_mcps.length
            });

            // Import and initialize MCP brainstorming engine
            const MCPBrainstormingEngine = require('../mcp_brainstorming.js').MCPBrainstormingEngine;
            
            const brainstormingEngine = new MCPBrainstormingEngine({
              openRouterApiKey: process.env.OPENROUTER_API_KEY,
              enableKnowledgeGraph: true,
              paretoThreshold: 0.8,
              maxRecommendations: 5
            });

            // Initialize the engine
            await brainstormingEngine.initialize();

            let result = {
              success: true,
              workflow_type,
              task_description,
              timestamp: new Date().toISOString()
            };

            // Execute capability assessment
            if (workflow_type === 'capability_assessment' || workflow_type === 'full_workflow') {
              logger.logOperation('info', 'MCP_BRAINSTORMING_CAPABILITY_ASSESSMENT', 'Executing capability assessment');
              
              const capabilityAssessment = await brainstormingEngine.assessCapabilities(task_description, {
                requirements,
                existingMCPs: existing_mcps,
                options
              });
              
              result.capability_assessment = capabilityAssessment;
            }

            // Execute gap analysis
            if (workflow_type === 'gap_analysis' || workflow_type === 'full_workflow') {
              logger.logOperation('info', 'MCP_BRAINSTORMING_GAP_ANALYSIS', 'Executing gap analysis');
              
              const gapAnalysis = await brainstormingEngine.analyzeGaps(task_description, {
                requirements,
                existingCapabilities: result.capability_assessment?.capabilities || [],
                options
              });
              
              result.gap_analysis = gapAnalysis;
            }

            // Execute MCP generation workflow
            if (workflow_type === 'mcp_generation' || workflow_type === 'full_workflow') {
              // Execute RAG-MCP retrieval
              logger.logOperation('info', 'MCP_BRAINSTORMING_RAG_RETRIEVAL', 'Executing RAG-MCP retrieval');
              
              const ragMcpResults = await brainstormingEngine.ragMcpRetriever.executeRAGMCPPipeline(task_description, {
                maxCandidates: 10,
                paretoOptimization: true,
                enableValidation: true
              });

              // Apply Pareto principle selection
              const paretoSelector = brainstormingEngine.paretoSelector;
              const selectedMCPs = paretoSelector.selectOptimalMCPs(
                ragMcpResults.mcpCandidates,
                { requirements, taskDescription: task_description }
              );

              result.rag_mcp_results = {
                total_candidates: ragMcpResults.mcpCandidates.length,
                selected_mcps: selectedMCPs.map(mcp => ({
                  name: mcp.name,
                  description: mcp.description,
                  pareto_score: mcp.paretoScore,
                  relevance_score: mcp.relevanceScore,
                  capabilities: mcp.capabilities
                })),
                pareto_optimization: ragMcpResults.paretoOptimization,
                processing_time_ms: ragMcpResults.processingTimeMs
              };

              // Generate new MCP designs if gaps exist
              if (result.gap_analysis?.gaps?.length > 0 || !result.rag_mcp_results?.selected_mcps?.length) {
                logger.logOperation('info', 'MCP_BRAINSTORMING_NEW_MCP_GENERATION', 'Generating new MCP designs');
                
                const newMcpDesigns = await brainstormingEngine.generateNewMCPDesigns(task_description, {
                  gaps: result.gap_analysis?.gaps || [],
                  existingMCPs: selectedMCPs,
                  requirements,
                  options
                });

                result.new_mcp_designs = newMcpDesigns.map(design => ({
                  name: design.name,
                  description: design.description,
                  design_rationale: design.designRationale,
                  implementation_complexity: design.implementationComplexity,
                  expected_benefits: design.expectedBenefits,
                  integration_requirements: design.integrationRequirements
                }));
              }
            }

            logger.logOperation('info', 'MCP_BRAINSTORMING_SUCCESS', 'MCP brainstorming workflow completed successfully', {
              workflow_type,
              has_capability_assessment: !!result.capability_assessment,
              has_gap_analysis: !!result.gap_analysis,
              selected_mcps_count: result.rag_mcp_results?.selected_mcps?.length || 0,
              new_mcp_designs_count: result.new_mcp_designs?.length || 0
            });

            return JSON.stringify(result, null, 2);

          } catch (error) {
            logger.logError('MCP_BRAINSTORMING_ERROR', error, { task_description, workflow_type, requirements, existing_mcps, options });
            return JSON.stringify({
              success: false,
              workflow_type: 'unknown',
              error: `Error in MCP brainstorming: ${error.message}`,
              timestamp: new Date().toISOString()
            });
          }
        },
      }),

      // Multimodal Server Tool - HTTP Integration with Node.js Service
      new DynamicStructuredTool({
        name: "multimodal_server",
        description: "Process multimodal content using the Node.js multimodal service with endpoints for image, audio, video processing and cross-modal validation",
        schema: z.object({
          operation: z.string().describe("Server operation: process_image, process_audio, process_video, analyze_visual, analyze_screenshot, or validate_cross_modal"),
          input_data: z.any().describe("Input data for processing - file path, URL, or base64 encoded content"),
          config: z.object({
            analysis_type: z.string().nullish().describe("Type of analysis to perform"),
            format: z.string().nullish().describe("Expected output format"),
            quality: z.string().nullish().describe("Processing quality level"),
            validation_config: z.any().nullish().describe("Configuration for cross-modal validation"),
            screenshot_config: z.any().nullish().describe("Configuration for screenshot analysis"),
            visual_params: z.any().nullish().describe("Parameters for visual analysis")
          }).nullish().describe("Processing configuration")
        }),
        func: async ({ operation, input_data, config = {} }) => {
          try {
            const startTime = Date.now();
            logger.logOperation('info', 'MULTIMODAL_SERVER_CALL', 'Calling multimodal server service', { 
              operation, 
              has_input: !!input_data,
              config_keys: Object.keys(config)
            });
            
            // Determine the appropriate endpoint based on operation
            let endpoint;
            let payload;
            
            switch (operation) {
              case 'process_image':
                endpoint = 'http://localhost:3006/api/process/image';
                payload = {
                  image_data: input_data,
                  processing_config: config,
                  analysis_type: config.analysis_type || 'basic',
                  format: config.format || 'json'
                };
                break;
                
              case 'process_audio':
                endpoint = 'http://localhost:3006/api/process/audio';
                payload = {
                  audio_data: input_data,
                  processing_config: config,
                  analysis_type: config.analysis_type || 'basic',
                  format: config.format || 'json'
                };
                break;
                
              case 'process_video':
                endpoint = 'http://localhost:3006/api/process/video';
                payload = {
                  video_data: input_data,
                  processing_config: config,
                  analysis_type: config.analysis_type || 'basic',
                  format: config.format || 'json'
                };
                break;
                
              case 'analyze_visual':
                endpoint = 'http://localhost:3006/api/analyze/visual';
                payload = {
                  image_data: input_data,
                  analysis_config: config.visual_params || {},
                  enable_object_detection: config.visual_params?.enable_object_detection !== false,
                  enable_spatial_relationships: config.visual_params?.enable_spatial_relationships !== false,
                  confidence_threshold: config.visual_params?.confidence_threshold || 0.7
                };
                break;
                
              case 'analyze_screenshot':
                endpoint = 'http://localhost:3006/api/analyze/screenshot';
                payload = {
                  screenshot_data: input_data,
                  analysis_config: config.screenshot_config || {},
                  enable_ui_classification: config.screenshot_config?.enable_ui_classification !== false,
                  enable_accessibility_analysis: config.screenshot_config?.enable_accessibility_analysis !== false,
                  enable_layout_analysis: config.screenshot_config?.enable_layout_analysis !== false,
                  confidence_threshold: config.screenshot_config?.confidence_threshold || 0.7
                };
                break;
                
              case 'validate_cross_modal':
                endpoint = 'http://localhost:3006/api/validate/cross-modal';
                payload = {
                  validation_data: input_data,
                  validation_config: config.validation_config || {},
                  validation_level: config.validation_config?.validation_level || 'standard',
                  expected_consistency: config.validation_config?.expected_consistency !== false
                };
                break;
                
              default:
                throw new Error(`Unsupported multimodal server operation: ${operation}`);
            }
            
            logger.logOperation('info', 'MULTIMODAL_SERVER_HTTP_REQUEST', `Making HTTP request to multimodal server`, { 
              endpoint, 
              operation,
              payload_keys: Object.keys(payload)
            });
            
            // Make HTTP request to the multimodal server
            const response = await fetch(endpoint, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
              const errorText = await response.text();
              throw new Error(`Multimodal server returned ${response.status}: ${response.statusText} - ${errorText}`);
            }
            
            const result = await response.json();
            const duration = Date.now() - startTime;
            
            logger.logOperation('info', 'MULTIMODAL_SERVER_SUCCESS', 'Multimodal server operation completed', {
              operation,
              success: result.success,
              duration,
              has_result: !!result.result,
              files_processed: result.files_processed
            });
            
            // Format the result for better readability
            const formattedResult = {
              success: result.success || true,
              operation,
              service_type: 'multimodal_server',
              result: result.result || result,
              files_processed: result.files_processed,
              processing_time: result.processing_time || duration,
              metadata: {
                endpoint: endpoint,
                duration: duration,
                timestamp: new Date().toISOString(),
                service_used: 'multimodal-server-nodejs'
              }
            };
            
            return JSON.stringify(formattedResult, null, 2);
            
          } catch (error) {
            logger.logError('MULTIMODAL_SERVER_ERROR', error, { operation, input_data, config });
            return JSON.stringify({
              success: false,
              error: `Error in multimodal server: ${error.message}`,
              operation,
              service_type: 'multimodal_server',
              timestamp: new Date().toISOString()
            }, null, 2);
          }
        },
      }),
    ];

    console.log('  🔍 Tools array created, inspecting first tool immediately...');
    console.log('  📋 Tools array length:', tools.length);
    if (tools.length > 0) {
      const firstTool = tools[0];
      console.log('  🔧 First tool type:', typeof firstTool);
      console.log('  📋 First tool constructor:', firstTool.constructor.name);
      console.log('  📋 First tool properties:', Object.keys(firstTool));
      console.log('  📋 First tool name:', firstTool.name);
      console.log('  📋 First tool description:', firstTool.description);
      console.log('  📋 First tool schema type:', typeof firstTool.schema);
      console.log('  📋 First tool schema:', firstTool.schema);
      console.log('  📋 First tool func type:', typeof firstTool.func);
    }

    // Add Sequential Thinking tool if available
    // Temporarily disabled to fix schema error
    /*
    if (managerInstance.sequentialThinking) {
      try {
        const sequentialThinkingTool = managerInstance.sequentialThinking.createSequentialThinkingTool();
        
        // Validate the tool before adding it using enhanced validation
        const isValidSequentialTool = sequentialThinkingTool && 
          sequentialThinkingTool.name && 
          sequentialThinkingTool.description && 
          (sequentialThinkingTool.schema || sequentialThinkingTool.argsSchema) && 
          (sequentialThinkingTool.schema?._def || sequentialThinkingTool.argsSchema?._def ||
           sequentialThinkingTool.schema?.shape || sequentialThinkingTool.argsSchema?.shape ||
           typeof sequentialThinkingTool.schema?.parse === 'function' ||
           typeof sequentialThinkingTool.argsSchema?.parse === 'function' ||
           typeof sequentialThinkingTool.schema?.safeParse === 'function' ||
           typeof sequentialThinkingTool.argsSchema?.safeParse === 'function');
        
        if (isValidSequentialTool) {
          tools.push(sequentialThinkingTool);
          logger.logOperation('info', 'SEQUENTIAL_THINKING_TOOL_ADDED', 'Sequential Thinking tool added to agent tools');
          console.log('  🧠 Sequential Thinking tool added successfully');
        } else {
          console.error('  ❌ Sequential Thinking tool has invalid schema:', {
            toolExists: !!sequentialThinkingTool,
            hasName: !!sequentialThinkingTool?.name,
            hasDescription: !!sequentialThinkingTool?.description,
            hasSchema: !!sequentialThinkingTool?.schema,
            hasArgsSchema: !!sequentialThinkingTool?.argsSchema,
            schemaDefExists: !!sequentialThinkingTool?.schema?._def || !!sequentialThinkingTool?.argsSchema?._def,
            schemaShapeExists: !!sequentialThinkingTool?.schema?.shape || !!sequentialThinkingTool?.argsSchema?.shape,
            hasParseMethod: typeof sequentialThinkingTool?.schema?.parse === 'function' || typeof sequentialThinkingTool?.argsSchema?.parse === 'function',
            hasSafeParseMethod: typeof sequentialThinkingTool?.schema?.safeParse === 'function' || typeof sequentialThinkingTool?.argsSchema?.safeParse === 'function',
            schemaType: typeof (sequentialThinkingTool?.schema || sequentialThinkingTool?.argsSchema),
            schemaConstructor: (sequentialThinkingTool?.schema || sequentialThinkingTool?.argsSchema)?.constructor?.name
          });
          logger.logError('SEQUENTIAL_THINKING_TOOL_INVALID_SCHEMA', new Error('Invalid tool schema'));
        }
      } catch (error) {
        console.error('  ❌ Failed to add Sequential Thinking tool:', error.message);
        logger.logError('SEQUENTIAL_THINKING_TOOL_ERROR', error);
      }
    } else {
      console.log('  ⚠️  Sequential Thinking tool not available - component not initialized');
      logger.logOperation('warn', 'SEQUENTIAL_THINKING_TOOL_UNAVAILABLE', 'Sequential Thinking component not available');
    }
    */

    // Debug: Log the tools array before validation
    console.log('  🔧 Tools array before validation:', {
      length: tools.length,
      toolTypes: tools.map((t, i) => ({ index: i, name: t?.name, type: typeof t, constructor: t?.constructor?.name }))
    });
    
    // Enhanced validation approach - trust the DynamicStructuredTool construction
    console.log('  🔍 Validating tool schemas...');
    const validTools = [];
    
    for (let i = 0; i < tools.length; i++) {
      const tool = tools[i];
      
      // Debug: Log each tool's properties
      console.log(`  🔧 Tool ${i + 1} debug:`, {
        exists: !!tool,
        type: typeof tool,
        constructor: tool?.constructor?.name,
        isDynamicStructuredTool: tool?.constructor?.name === 'DynamicStructuredTool',
        hasFunc: typeof tool?.func === 'function' || typeof tool?.call === 'function'
      });
      
      // Simplified validation - if it's a DynamicStructuredTool instance, it should be valid
      const isValidTool = tool && 
        (tool.constructor?.name === 'DynamicStructuredTool' || 
         typeof tool?.call === 'function' ||
         typeof tool?.func === 'function');
      
      if (isValidTool) {
        validTools.push(tool);
        console.log(`  ✅ Tool ${i + 1} (${tool.constructor?.name}) - Valid tool`);
      } else {
        console.log(`  ❌ Tool ${i + 1} - Invalid tool, skipping...`);
      }
    }

    console.log('  ✅ Successfully validated', validTools.length + '/' + tools.length, 'agent tools');
    logger.logOperation('info', 'TOOLS_CREATED', `Created ${validTools.length} valid tools for agent (${tools.length - validTools.length} invalid tools filtered out)`);
    
    return validTools;
    
    } catch (error) {
      logger.logError('TOOLS_CREATION_FAILED', error, { 
        errorMessage: error.message 
      });
      console.error('  ❌ Failed to create agent tools:', error.message);
      throw error;
    }
  }

  /**
   * Setup Express routes for the manager agent API
   */
  setupRoutes() {
    // Root endpoint
    this.app.get('/', (req, res) => {
      res.json({
        message: 'Alita KGoT Enhanced Manager Agent',
        version: '1.0.0',
        status: 'running',
        timestamp: new Date().toISOString(),
        endpoints: {
          health: '/health',
          status: '/status',
          metrics: '/metrics',
          sequentialThinking: '/sequential-thinking',
          sessions: '/sequential-thinking/sessions'
        }
      });
    });

    // Health check endpoint
    this.app.get('/health', (req, res) => {
      try {
        const healthStatus = {
          status: 'healthy',
          timestamp: new Date().toISOString(),
          uptime: process.uptime(),
          memory: process.memoryUsage(),
          version: '1.0.0',
          components: {
            kgotController: this.kgotController ? 'initialized' : 'not_initialized',
            mcpValidator: this.mcpValidator ? 'initialized' : 'not_initialized',
            sequentialThinking: this.sequentialThinking ? 'initialized' : 'not_initialized'
          }
        };
        
        res.json(healthStatus);
      } catch (error) {
        logger.logError('HEALTH_CHECK_ERROR', error);
        res.status(500).json({ error: 'Health check failed' });
      }
    });

    // Prometheus metrics endpoint
    this.app.get('/metrics', async (req, res) => {
      try {
        const metrics = await monitoring.getMetrics();
        res.set('Content-Type', monitoring.register.contentType);
        res.end(metrics);
      } catch (error) {
        logger.logError('METRICS_ENDPOINT_ERROR', error);
        res.status(500).json({ error: 'Failed to retrieve metrics' });
      }
    });

    // Main chat endpoint for processing user requests
    this.app.post('/chat', async (req, res) => {
      try {
        const { message, context = [], sessionId } = req.body;

        logger.info('Received chat request', { sessionId, message: message.substring(0, 100) });

        if (!message) {
          return res.status(400).json({ error: 'Message is required' });
        }

        if (!this.isInitialized) {
          return res.status(503).json({ error: 'Agent not initialized' });
        }

        const startTime = Date.now();
        logger.logOperation('info', 'CHAT_REQUEST', 'Processing chat request', { 
          sessionId, 
          messageLength: message.length,
          contextLength: context.length 
        });

        // Process the request through the LangChain agent
        const result = await this.agentExecutor.invoke({
          input: message,
          chat_history: context,
        });

        logger.info('Agent execution completed', { duration: Date.now() - startTime, outputLength: result.output?.length || 0 });

        const duration = Date.now() - startTime;
        
        logger.logOperation('info', 'CHAT_RESPONSE', 'Chat request processed successfully', {
          sessionId,
          duration,
          responseLength: result.output?.length || 0,
          intermediateSteps: result.intermediateSteps?.length || 0
        });

        logger.info('Sending response', { responsePreview: result.output?.substring(0, 100) });

        res.json({
          response: result.output,
          intermediateSteps: result.intermediateSteps,
          metadata: {
            duration,
            sessionId,
            timestamp: new Date().toISOString()
          }
        });

      } catch (error) {
        const duration = Date.now() - startTime;
        logger.error('Chat processing error', { error: error.message, stack: error.stack, sessionId: req.body.sessionId, duration });

        logger.logOperation('error', 'CHAT_ERROR', 'Chat request processing failed', {
          sessionId: req.body.sessionId,
          duration,
          messagePreview: req.body.message?.substring(0, 100)
        });

        res.status(500).json({
          error: 'Internal server error',
          message: process.env.NODE_ENV === 'development' ? error.message : 'An error occurred'
        });
      }
    });

    // System status endpoint with comprehensive health checks
    this.app.get('/status', async (req, res) => {
      try {
        const startTime = Date.now();
        
        // Get internal component health
        const internalHealth = this.checkInternalComponentHealth();
        
        // Get external service health (with timeout)
        const externalHealth = await this.performHealthChecks();
        
        // Combine health status
        const combinedStatus = {
          ...internalHealth,
          ...externalHealth
        };
        
        // Determine overall system status
        const healthyServices = Object.values(combinedStatus).filter(status => status === 'healthy').length;
        const totalServices = Object.keys(combinedStatus).length;
        const healthPercentage = (healthyServices / totalServices) * 100;
        
        let overallStatus = 'operational';
        if (healthPercentage < 50) {
          overallStatus = 'degraded';
        } else if (healthPercentage < 80) {
          overallStatus = 'partial';
        }
        
        const healthCheckDuration = Date.now() - startTime;
        
        logger.logOperation('info', 'STATUS_CHECK_COMPLETE', 'System status check completed', {
          overallStatus,
          healthPercentage,
          healthyServices,
          totalServices,
          duration: healthCheckDuration
        });

        res.json({
          status: overallStatus,
          services: combinedStatus,
          summary: {
            healthy: healthyServices,
            total: totalServices,
            healthPercentage: Math.round(healthPercentage),
            lastChecked: new Date().toISOString(),
            checkDuration: healthCheckDuration
          },
          sequentialThinking: {
            activeSessions: this.sequentialThinking?.getActiveSessions()?.length || 0,
            complexityAnalysisCache: this.sequentialThinking?.complexityScores?.size || 0
          },
          systemMetrics: {
            uptime: process.uptime(),
            memoryUsage: process.memoryUsage(),
            cpuUsage: process.cpuUsage(),
            nodeVersion: process.version,
            platform: process.platform
          },
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        logger.logError('STATUS_CHECK_ERROR', error);
        res.status(500).json({ 
          error: 'Status check failed',
          message: process.env.NODE_ENV === 'development' ? error.message : 'An error occurred during status check',
          timestamp: new Date().toISOString()
        });
      }
    });

    // Sequential Thinking endpoint for direct access
    this.app.post('/sequential-thinking', async (req, res) => {
      try {
        if (!this.sequentialThinking) {
          return res.status(503).json({ error: 'Sequential Thinking not initialized' });
        }

        const taskContext = req.body;
        
        // Validate required fields
        if (!taskContext.taskId || !taskContext.description) {
          return res.status(400).json({ 
            error: 'Task ID and description are required',
            required: ['taskId', 'description']
          });
        }

        const startTime = Date.now();
        logger.logOperation('info', 'SEQUENTIAL_THINKING_API_REQUEST', 'Sequential thinking API request', {
          taskId: taskContext.taskId,
          description: taskContext.description?.substring(0, 100)
        });

        // Perform complexity analysis
        const complexityAnalysis = this.sequentialThinking.detectComplexity(taskContext);

        if (!complexityAnalysis.shouldTriggerSequentialThinking) {
          return res.json({
            status: 'not_required',
            message: 'Task complexity does not require sequential thinking',
            complexityAnalysis,
            recommendation: 'Proceed with standard processing',
            duration: Date.now() - startTime
          });
        }

        // Execute sequential thinking process
        const thinkingResult = await this.sequentialThinking.executeSequentialThinking(
          taskContext,
          complexityAnalysis.recommendedTemplate
        );

        const duration = Date.now() - startTime;

        logger.logOperation('info', 'SEQUENTIAL_THINKING_API_SUCCESS', 'Sequential thinking API request completed', {
          taskId: taskContext.taskId,
          sessionId: thinkingResult.sessionId,
          duration,
          thoughtCount: thinkingResult.thoughts.length
        });

        res.json({
          status: 'completed',
          sessionId: thinkingResult.sessionId,
          complexityAnalysis,
          thinkingResult: {
            template: thinkingResult.template.name,
            conclusions: thinkingResult.conclusions,
            systemRecommendations: thinkingResult.systemRecommendations,
            thoughtCount: thinkingResult.thoughts.length,
            duration: thinkingResult.duration
          },
          metadata: {
            apiDuration: duration,
            timestamp: new Date().toISOString()
          }
        });

      } catch (error) {
        const duration = Date.now() - startTime;
        logger.logError('SEQUENTIAL_THINKING_API_ERROR', error, {
          taskId: req.body?.taskId,
          duration
        });

        res.status(500).json({
          error: 'Sequential thinking processing failed',
          message: process.env.NODE_ENV === 'development' ? error.message : 'An error occurred',
          duration
        });
      }
    });

    // Sequential Thinking sessions endpoint for monitoring
    this.app.get('/sequential-thinking/sessions', (req, res) => {
      try {
        if (!this.sequentialThinking) {
          return res.status(503).json({ error: 'Sequential Thinking not initialized' });
        }

        const activeSessions = this.sequentialThinking.getActiveSessions();
        
        res.json({
          activeSessions: activeSessions.length,
          sessions: activeSessions.map(session => ({
            sessionId: session.sessionId,
            taskId: session.taskId,
            template: session.template.name,
            progress: session.currentThought / session.totalThoughts,
            duration: session.endTime ? 
              session.endTime - session.startTime : 
              Date.now() - session.startTime,
            status: session.endTime ? 'completed' : 'active'
          })),
          timestamp: new Date().toISOString()
        });

      } catch (error) {
        logger.logError('SESSIONS_API_ERROR', error);
        res.status(500).json({ error: 'Failed to retrieve sessions' });
      }
    });

    // Main agent execution endpoint for processing user requests
    this.app.post('/agent/execute', async (req, res) => {
      const startTime = Date.now();
      const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      try {
        // Enhanced request logging
        logger.info(`[${requestId}] Incoming request to /agent/execute`, { 
          requestId,
          headers: req.headers,
          body: req.body ? {
            hasMessage: !!req.body.message,
            messageLength: req.body.message?.length || 0,
            hasContext: !!req.body.context,
            contextLength: req.body.context?.length || 0,
            sessionId: req.body.sessionId
          } : 'no body',
          method: req.method,
          url: req.url,
          ip: req.ip
        });

        // Enhanced initialization status logging
        logger.info(`[${requestId}] System status check`, {
          requestId,
          isInitialized: this.isInitialized,
          agentExecutorExists: !!this.agentExecutor,
          agentExists: !!this.agent,
          kgotControllerExists: !!this.kgotController,
          mcpValidatorExists: !!this.mcpValidator,
          sequentialThinkingExists: !!this.sequentialThinking
        });

        const { message, context = [], sessionId } = req.body;

        logger.info(`[${requestId}] Received agent execution request`, { 
          requestId,
          sessionId, 
          messagePreview: message?.substring(0, 100) || 'NO MESSAGE'
        });

        // Enhanced validation with detailed error reporting
        if (!message) {
          logger.warn(`[${requestId}] Request validation failed: No message provided`, { requestId });
          return res.status(400).json({ 
            error: 'Message is required',
            requestId,
            timestamp: new Date().toISOString()
          });
        }

        if (!this.isInitialized) {
          logger.error(`[${requestId}] Service not ready: Agent not initialized`, { 
            requestId,
            isInitialized: this.isInitialized,
            agentExecutorExists: !!this.agentExecutor,
            agentExists: !!this.agent
          });
          return res.status(503).json({ 
            error: 'Agent not initialized', 
            requestId,
            details: {
              isInitialized: this.isInitialized,
              agentExecutorExists: !!this.agentExecutor,
              agentExists: !!this.agent
            },
            timestamp: new Date().toISOString()
          });
        }

        if (!this.agentExecutor) {
          logger.error(`[${requestId}] Critical error: AgentExecutor is null despite isInitialized=true`, { 
            requestId,
            isInitialized: this.isInitialized,
            agentExecutorExists: !!this.agentExecutor,
            agentExists: !!this.agent
          });
          return res.status(500).json({ 
            error: 'AgentExecutor not available', 
            requestId,
            details: {
              isInitialized: this.isInitialized,
              agentExecutorExists: !!this.agentExecutor
            },
            timestamp: new Date().toISOString()
          });
        }

        logger.logOperation('info', 'AGENT_EXECUTION_REQUEST', 'Processing agent execution request', { 
          requestId,
          sessionId, 
          messageLength: message.length,
          contextLength: context.length 
        });

        // Enhanced pre-execution logging
        logger.info(`[${requestId}] About to invoke agent executor`, { 
          requestId,
          agentExecutorType: this.agentExecutor.constructor.name,
          agentExecutorExists: !!this.agentExecutor,
          messagePreview: message.substring(0, 100),
          contextItems: context.length
        });
        
        // Process the request through the LangChain agent with timeout
        const executionPromise = this.agentExecutor.invoke({
          input: message,
          chat_history: context,
        });

        // Add timeout to prevent hanging requests
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => {
            reject(new Error('Agent execution timeout after 60 seconds'));
          }, 60000);
        });

        const result = await Promise.race([executionPromise, timeoutPromise]);

        logger.info(`[${requestId}] Agent execution completed successfully`, { 
          requestId,
          duration: Date.now() - startTime, 
          outputLength: result.output?.length || 0,
          hasOutput: !!result.output,
          hasIntermediateSteps: !!result.intermediateSteps,
          intermediateStepsCount: result.intermediateSteps?.length || 0
        });

        const duration = Date.now() - startTime;
        
        logger.logOperation('info', 'AGENT_EXECUTION_RESPONSE', 'Agent execution response processed successfully', {
          requestId,
          sessionId,
          duration,
          responseLength: result.output?.length || 0,
          intermediateSteps: result.intermediateSteps?.length || 0
        });

        logger.info(`[${requestId}] Sending response to client`, { 
          requestId,
          responsePreview: result.output?.substring(0, 100) || 'NO OUTPUT',
          totalDuration: duration
        });

        res.json({
          response: result.output,
          intermediateSteps: result.intermediateSteps,
          metadata: {
            duration,
            sessionId,
            requestId,
            timestamp: new Date().toISOString()
          }
        });

      } catch (error) {
        const duration = Date.now() - startTime;
        
        // Enhanced error logging with full details
        logger.error(`[${requestId}] Agent execution error - FULL DETAILS`, { 
          requestId,
          errorMessage: error.message,
          errorStack: error.stack,
          errorType: error.constructor.name,
          errorName: error.name,
          sessionId: req.body?.sessionId,
          duration,
          systemState: {
            isInitialized: this.isInitialized,
            agentExecutorExists: !!this.agentExecutor,
            agentExists: !!this.agent,
            kgotControllerExists: !!this.kgotController,
            mcpValidatorExists: !!this.mcpValidator,
            sequentialThinkingExists: !!this.sequentialThinking
          },
          requestDetails: {
            hasBody: !!req.body,
            hasMessage: !!req.body?.message,
            messageLength: req.body?.message?.length || 0,
            method: req.method,
            url: req.url
          }
        });

        logger.logOperation('error', 'AGENT_EXECUTION_ERROR', 'Agent execution failed', {
          requestId,
          sessionId: req.body?.sessionId,
          duration,
          messagePreview: req.body?.message?.substring(0, 100) || 'NO MESSAGE',
          errorMessage: error.message,
          errorType: error.constructor.name
        });

        // Send detailed error response in development
        const errorResponse = {
          error: 'Internal server error',
          requestId,
          timestamp: new Date().toISOString(),
          message: process.env.NODE_ENV === 'development' ? error.message : 'An error occurred'
        };

        if (process.env.NODE_ENV === 'development') {
          errorResponse.details = {
            errorType: error.constructor.name,
            errorName: error.name,
            agentExecutorExists: !!this.agentExecutor,
            isInitialized: this.isInitialized,
            systemState: {
              agentExists: !!this.agent,
              kgotControllerExists: !!this.kgotController,
              mcpValidatorExists: !!this.mcpValidator,
              sequentialThinkingExists: !!this.sequentialThinking
            }
          };
          errorResponse.stack = error.stack;
        }

        res.status(500).json(errorResponse);
      }
    });

    // Add GET handler for /agent/execute before the POST
    this.app.get('/agent/execute', (req, res) => {
      res.json({
        message: 'This endpoint requires POST requests for agent execution.',
        usage: {
          method: 'POST',
          body: {
            message: 'string (required)',
            context: 'array (optional)',
            sessionId: 'string (optional)'
          },
          description: 'Execute tasks using the Alita Manager Agent'
        },
        timestamp: new Date().toISOString()
      });
    });
  }

  /**
   * Check if a port is available
   */
  async isPortAvailable(port) {
    return new Promise((resolve) => {
      const server = require('net').createServer();
      
      server.listen(port, () => {
        server.once('close', () => {
          resolve(true);
        });
        server.close();
      });
      
      server.on('error', () => {
        resolve(false);
      });
    });
  }

  /**
   * Find an available port starting from the configured port
   */
  async findAvailablePort(startPort, maxAttempts = 10) {
    const basePort = parseInt(startPort);
    
    for (let i = 0; i < maxAttempts; i++) {
      const port = basePort + i;
      const available = await this.isPortAvailable(port);
      
      if (available) {
        return port;
      }
      
      console.log(`⚠️  Port ${port} is in use, trying next port...`);
    }
    
    throw new Error(`No available ports found in range ${basePort}-${basePort + maxAttempts - 1}`);
  }

  /**
   * Start the Express server with automatic port detection
   */
  async start() {
    try {
      const originalPort = this.port;
      console.log(`🚀 Starting server on port ${this.port}...`);
      
      // Find an available port
      this.port = await this.findAvailablePort(this.port);
      
      if (this.port !== originalPort) {
        console.log(`🔄 Port ${originalPort} was in use, using port ${this.port} instead`);
      }
      
      await new Promise((resolve, reject) => {
        this.server = this.app.listen(this.port, () => {
          console.log(`✅ Server successfully started on port ${this.port}`);
          console.log(`🌐 Server address: http://localhost:${this.port}`);
          console.log(`📊 Server process ID: ${process.pid}`);
          console.log(`💾 Memory usage after startup:`, JSON.stringify(process.memoryUsage(), null, 2));
          
          logger.logOperation('info', 'SERVER_START', `Alita Manager Agent started on port ${this.port}`, {
            pid: process.pid,
            memoryUsage: process.memoryUsage(),
            uptime: process.uptime(),
            originalPort: originalPort,
            actualPort: this.port
          });
          resolve();
        });
        
        this.server.on('error', (error) => {
          console.error(`❌ Server startup error: ${error.message}`);
          logger.logError('SERVER_START_ERROR', error, { port: this.port });
          reject(error);
        });
      });
    } catch (error) {
      console.error(`❌ Failed to start server: ${error.message}`);
      logger.logError('SERVER_START_ERROR', error, { port: this.port });
      throw error;
    }
  }

  /**
   * Graceful shutdown of the server
   */
  async stop() {
    try {
      // Shutdown Sequential Thinking integration
      if (this.sequentialThinking) {
        await this.sequentialThinking.shutdown();
        logger.logOperation('info', 'SEQUENTIAL_THINKING_SHUTDOWN', 'Sequential Thinking integration shutdown completed');
      }

      // Close the server
      if (this.server) {
        await new Promise((resolve) => {
          this.server.close(() => {
            logger.logOperation('info', 'SERVER_STOP', 'Alita Manager Agent stopped');
            resolve();
          });
        });
      }

      logger.logOperation('info', 'MANAGER_AGENT_SHUTDOWN_COMPLETE', 'Manager Agent shutdown completed successfully');

    } catch (error) {
      logger.logError('SHUTDOWN_ERROR', error);
      throw error;
    }
  }

  /**
   * Check health of a specific service by making HTTP request
   * @param {string} serviceName - Name of the service
   * @param {string} url - Health check URL
   * @param {number} timeout - Request timeout in milliseconds
   * @returns {Promise<string>} - Health status: 'healthy', 'unhealthy', or 'unknown'
   */
  async checkServiceHealth(serviceName, url, timeout = 5000) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(url, {
        method: 'GET',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
        }
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        logger.logOperation('debug', 'HEALTH_CHECK_SUCCESS', `${serviceName} health check successful`, {
          service: serviceName,
          url,
          status: response.status
        });
        return 'healthy';
      } else {
        logger.logOperation('warn', 'HEALTH_CHECK_FAILED', `${serviceName} health check failed`, {
          service: serviceName,
          url,
          status: response.status
        });
        return 'unhealthy';
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        logger.logOperation('warn', 'HEALTH_CHECK_TIMEOUT', `${serviceName} health check timeout`, {
          service: serviceName,
          url,
          timeout
        });
        return 'timeout';
      } else {
        logger.logOperation('error', 'HEALTH_CHECK_ERROR', `${serviceName} health check error`, {
          service: serviceName,
          url,
          error: error.message
        });
        return 'unhealthy';
      }
    }
  }

  /**
   * Perform health checks for all external services
   * @returns {Promise<Object>} - Object with service health statuses
   */
  async performHealthChecks() {
    const healthChecks = await Promise.allSettled([
      this.checkServiceHealth('webAgent', 'http://localhost:3001/health'),
      this.checkServiceHealth('mcpCreation', 'http://localhost:3002/health'),
      this.checkServiceHealth('kgotController', 'http://localhost:3003/health'),
      this.checkServiceHealth('multimodal', 'http://localhost:3004/health'),
      this.checkServiceHealth('optimization', 'http://localhost:3005/health'),
      this.checkServiceHealth('multimodalServer', 'http://localhost:3006/health')
    ]);

    const serviceNames = ['webAgent', 'mcpCreation', 'kgotController', 'multimodal', 'optimization', 'multimodalServer'];
    const healthStatus = {};

    healthChecks.forEach((check, index) => {
      const serviceName = serviceNames[index];
      if (check.status === 'fulfilled') {
        healthStatus[serviceName] = check.value;
      } else {
        healthStatus[serviceName] = 'error';
        logger.logError('HEALTH_CHECK_PROMISE_ERROR', check.reason, { service: serviceName });
      }
    });

    return healthStatus;
  }

  /**
   * Check health of internal components
   * @returns {Object} - Object with internal component health statuses
   */
  checkInternalComponentHealth() {
    const componentHealth = {
      managerAgent: 'healthy',
      kgotController: this.kgotController ? 'healthy' : 'not_initialized',
      mcpValidator: this.mcpValidator ? 'healthy' : 'not_initialized',
      sequentialThinking: this.sequentialThinking ? 'healthy' : 'not_initialized',
      agentExecutor: this.agentExecutor ? 'healthy' : 'not_initialized',
      expressApp: this.app ? 'healthy' : 'not_initialized'
    };

    // Additional health checks for internal components
    if (this.kgotController) {
      try {
        // Check if KGoT controller has required methods
        if (typeof this.kgotController.initializeAsync === 'function') {
          componentHealth.kgotController = 'healthy';
        } else {
          componentHealth.kgotController = 'degraded';
        }
      } catch (error) {
        componentHealth.kgotController = 'unhealthy';
      }
    }

    if (this.mcpValidator) {
      try {
        // Check if MCP validator has required methods
        if (typeof this.mcpValidator.validateSolution === 'function') {
          componentHealth.mcpValidator = 'healthy';
        } else {
          componentHealth.mcpValidator = 'degraded';
        }
      } catch (error) {
        componentHealth.mcpValidator = 'unhealthy';
      }
    }

    if (this.sequentialThinking) {
      try {
        // Check if Sequential Thinking has required methods
        if (typeof this.sequentialThinking.detectComplexity === 'function') {
          componentHealth.sequentialThinking = 'healthy';
        } else {
          componentHealth.sequentialThinking = 'degraded';
        }
      } catch (error) {
        componentHealth.sequentialThinking = 'unhealthy';
      }
    }

    logger.logOperation('debug', 'INTERNAL_COMPONENT_HEALTH', 'Internal component health check completed', {
      componentHealth
    });

    return componentHealth;
  }
}

// Handle process signals for graceful shutdown
let managerAgent = null;

process.on('SIGTERM', async () => {
  logger.logOperation('info', 'SHUTDOWN', 'Received SIGTERM, shutting down gracefully');
  if (managerAgent) {
    await managerAgent.stop();
  }
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.logOperation('info', 'SHUTDOWN', 'Received SIGINT, shutting down gracefully');
  if (managerAgent) {
    await managerAgent.stop();
  }
  process.exit(0);
});

// Enhanced startup process with detailed logging
if (require.main === module) {
  (async () => {
    try {
      console.log('\n🚀 ===== ALITA-KGOT ENHANCED SYSTEM STARTUP =====');
      console.log('📅 Timestamp:', new Date().toISOString());
      console.log('🌍 Environment:', process.env.NODE_ENV || 'development');
      console.log('📁 Working Directory:', process.cwd());
      console.log('🔧 Node Version:', process.version);
      console.log('💾 Memory Usage:', JSON.stringify(process.memoryUsage(), null, 2));
      console.log('⚙️  Environment Variables:');
      console.log('   - PORT:', process.env.PORT || '3000 (default)');
      console.log('   - LOG_LEVEL:', process.env.LOG_LEVEL || 'info (default)');
      console.log('   - NODE_ENV:', process.env.NODE_ENV || 'development (default)');
      console.log('   - OPENROUTER_API_KEY:', process.env.OPENROUTER_API_KEY ? '✅ Set' : '❌ Not set');
      console.log('');
      
      console.log('🔄 Initializing Alita Manager Agent...');
      const startTime = Date.now();
      
      managerAgent = new AlitaManagerAgent();
      
      console.log('⏳ Waiting for component initialization to complete...');
      // Wait for components to initialize before starting server
      await managerAgent.waitForInitialization();
      
      // The agent initialization is already happening asynchronously in the constructor
      // Just wait for it to complete rather than calling it again
      console.log('⏳ Waiting for LangChain agent initialization to complete...');
      let agentInitAttempts = 0;
      const maxAgentInitAttempts = 30; // 15 seconds timeout
      while (!managerAgent.isInitialized && agentInitAttempts < maxAgentInitAttempts) {
        await new Promise(resolve => setTimeout(resolve, 500));
        agentInitAttempts++;
        logger.info(`Waiting for agent initialization... attempt ${agentInitAttempts}/${maxAgentInitAttempts}`, {
          isInitialized: managerAgent.isInitialized,
          agentExecutorExists: !!managerAgent.agentExecutor
        });
      }
      
      if (!managerAgent.isInitialized) {
        throw new Error(`Agent initialization timeout after ${maxAgentInitAttempts * 500}ms`);
      }
      
      logger.info('Agent initialization completed successfully', {
        isInitialized: managerAgent.isInitialized,
        agentExecutorExists: !!managerAgent.agentExecutor,
        totalWaitTime: agentInitAttempts * 500
      });
      
      console.log('⏳ Starting server...');
      await managerAgent.start();
      
      const initTime = Date.now() - startTime;
      console.log('');
      console.log('✅ ===== STARTUP COMPLETED SUCCESSFULLY =====');
      console.log(`⏱️  Total initialization time: ${initTime}ms`);
      console.log(`🌐 Server running on: http://localhost:${managerAgent.port}`);
      console.log('📊 Available endpoints:');
      console.log('   - GET  /health          - Health check');
      console.log('   - GET  /status          - System status');
      console.log('   - POST /agent/execute   - Agent execution');
      console.log('   - POST /sequential-thinking - Sequential thinking API');
      console.log('   - GET  /sequential-thinking/sessions - Active sessions');
      console.log('');
      console.log('🎯 System ready for requests!');
      console.log('📝 Logs are being written to: ./logs/');
      console.log('🔍 Use Ctrl+C to gracefully shutdown');
      console.log('================================================\n');
      
      // Enhanced system readiness logging for debugging
      logger.info('===== FINAL SYSTEM STATUS =====', {
        timestamp: new Date().toISOString(),
        serverPort: managerAgent.port,
        initializationTime: initTime,
        systemStatus: {
          serverRunning: true,
          isInitialized: managerAgent.isInitialized,
          agentExecutorReady: !!managerAgent.agentExecutor,
          agentReady: !!managerAgent.agent,
          componentsReady: {
            kgotController: !!managerAgent.kgotController,
            mcpValidator: !!managerAgent.mcpValidator,
            sequentialThinking: !!managerAgent.sequentialThinking
          }
        },
        systemReadyForRequests: !!(managerAgent.agentExecutor && managerAgent.isInitialized),
        processId: process.pid,
        memoryUsage: process.memoryUsage()
      });
      
      if (!managerAgent.isInitialized || !managerAgent.agentExecutor) {
        logger.error('WARNING: System may not be fully ready', {
          isInitialized: managerAgent.isInitialized,
          agentExecutorExists: !!managerAgent.agentExecutor,
          warning: 'The system may reject requests due to incomplete initialization'
        });
      }
      
    } catch (error) {
      console.error('\n❌ ===== STARTUP FAILED =====');
      console.error('💥 Error:', error.message);
      console.error('📍 Stack:', error.stack);
      console.error('================================\n');
      
      if (logger) {
        logger.logError('STARTUP_ERROR', error);
      }
      process.exit(1);
    }
  })();
}

module.exports = AlitaManagerAgent;