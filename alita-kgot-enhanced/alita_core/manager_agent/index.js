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

const { ChatOpenAI } = require('@langchain/openai');
const { AgentExecutor, createOpenAIFunctionsAgent } = require('langchain/agents');
const { ChatPromptTemplate, MessagesPlaceholder } = require('@langchain/core/prompts');
const { HumanMessage, SystemMessage } = require('@langchain/core/messages');
const { DynamicTool } = require('@langchain/core/tools');
const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const EventEmitter = require('events');

// Import logging configuration
const { loggers, httpLoggingMiddleware } = require('../../config/logging/winston_config');
const logger = loggers.managerAgent;

// Import configuration
const modelConfig = require('../../config/models/model_config.json');

// Import KGoT Controller, MCP Validation, and Sequential Thinking Integration
const { KGoTController } = require('../../kgot_core/controller/kgot_controller');
const { MCPCrossValidationCoordinator } = require('../../kgot_core/controller/mcp_cross_validation');
const { SequentialThinkingIntegration } = require('./sequential_thinking_integration');

/**
 * Alita Manager Agent Class
 * Orchestrates the entire Alita-KGoT system using LangChain agents
 * Extends EventEmitter for component coordination
 */
class AlitaManagerAgent extends EventEmitter {
  constructor() {
    super(); // Call EventEmitter constructor
    this.app = express();
    this.port = process.env.PORT || 3000;
    this.agent = null;
    this.agentExecutor = null;
    this.isInitialized = false;
    
    // Initialize KGoT Controller, MCP Validation, and Sequential Thinking Integration
    this.kgotController = null;
    this.mcpValidator = null;
    this.sequentialThinking = null;
    
    logger.info('Initializing Alita Manager Agent', { operation: 'INIT' });
    this.setupExpress();
    this.initializeComponents().catch(error => {
      logger.logError('COMPONENTS_INIT_ASYNC_FAILED', error);
    });
    this.initializeAgent().catch(error => {
      logger.logError('AGENT_INIT_ASYNC_FAILED', error);
    });
  }

  /**
   * Setup Express server with middleware and routes
   * Configures CORS, rate limiting, and logging middleware
   */
  setupExpress() {
    logger.logOperation('info', 'EXPRESS_SETUP', 'Setting up Express server configuration');
    
    // CORS configuration for cross-origin requests
    this.app.use(cors({
      origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
      credentials: true
    }));

    // Rate limiting to prevent abuse
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
      message: 'Too many requests from this IP'
    });
    this.app.use(limiter);

    // Request parsing middleware
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true }));

    // Logging middleware for all HTTP requests
    this.app.use(httpLoggingMiddleware(logger));

    this.setupRoutes();
  }

  /**
   * Initialize KGoT Controller, MCP Validation, and Sequential Thinking components
   * Sets up the core reasoning and validation infrastructure
   */
  async initializeComponents() {
    try {
      logger.logOperation('info', 'COMPONENTS_INIT', 'Initializing KGoT Controller, MCP Validation, and Sequential Thinking');

      // Initialize KGoT Controller with enhanced configuration
      this.kgotController = new KGoTController({
        maxIterations: modelConfig.alita_config.kgot_controller.max_iterations,
        votingThreshold: 0.6,
        validationEnabled: true,
        graphUpdateInterval: 5000
      });

      // Initialize MCP Cross-Validation Coordinator with Claude-4-Sonnet as primary
      this.mcpValidator = new MCPCrossValidationCoordinator({
        validationModels: ['claude-4-sonnet', 'o3', 'gemini-2.5-pro'],
        primaryModel: 'claude-4-sonnet', // Prioritize Claude-4-Sonnet for MCP validation
        consensusThreshold: 0.7,
        confidenceThreshold: 0.8,
        enableBenchmarkValidation: true,
        enableCrossModel: true
      });

      // Initialize Sequential Thinking Integration with optimized configuration
      this.sequentialThinking = new SequentialThinkingIntegration({
        complexityThreshold: 7, // Trigger for task complexity score > 7
        errorThreshold: 3, // Trigger for error count > 3
        maxThoughts: 15, // Maximum thought steps in reasoning process
        enableCrossSystemCoordination: true, // Enable Alita-KGoT coordination
        enableAdaptiveThinking: true, // Allow dynamic thought process adjustment
        thoughtTimeout: 30000, // 30 second timeout per thought step
        enableMCPCreationRouting: true, // Route to MCP creation when needed
        enableKGoTProcessing: true // Route to KGoT processing when needed
      });

      // Set up event listeners for coordination
      this.setupComponentEventListeners();

      logger.logOperation('info', 'COMPONENTS_INIT_SUCCESS', 'All core components initialized successfully', {
        kgotController: !!this.kgotController,
        mcpValidator: !!this.mcpValidator,
        sequentialThinking: !!this.sequentialThinking
      });

    } catch (error) {
      logger.logError('COMPONENTS_INIT_FAILED', error, { 
        kgotConfig: modelConfig.alita_config.kgot_controller 
      });
      throw error;
    }
  }

  /**
   * Setup event listeners for component coordination
   * Enables communication between KGoT Controller, MCP Validation, and Sequential Thinking
   */
  setupComponentEventListeners() {
    // Listen for KGoT execution completion to trigger validation
    this.kgotController.on('executionComplete', async (executionResult) => {
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
      }
    });

    // Listen for validation completion
    this.mcpValidator.on('validationComplete', (validationResult) => {
      logger.logOperation('info', 'VALIDATION_COMPLETE', 'MCP validation completed', {
        validationId: validationResult.validationId,
        isValid: validationResult.isValid,
        confidence: validationResult.confidence
      });
    });

    // Listen for Sequential Thinking progress updates
    this.sequentialThinking.on('thinkingProgress', (progressData) => {
      logger.logOperation('debug', 'SEQUENTIAL_THINKING_PROGRESS', 'Sequential thinking progress update', {
        sessionId: progressData.sessionId,
        progress: progressData.progress,
        currentThought: progressData.currentThought?.conclusion?.substring(0, 100) || 'Processing...'
      });
    });

    // Listen for Sequential Thinking completion
    this.sequentialThinking.on('thinkingComplete', (thinkingSession) => {
      logger.logOperation('info', 'SEQUENTIAL_THINKING_COMPLETE', 'Sequential thinking process completed', {
        sessionId: thinkingSession.sessionId,
        duration: thinkingSession.duration,
        thoughtCount: thinkingSession.thoughts.length,
        template: thinkingSession.template.name,
        systemRecommendations: thinkingSession.systemRecommendations?.systemSelection?.primary
      });

      // Emit system coordination event with routing recommendations
      this.emit('systemCoordinationPlan', {
        sessionId: thinkingSession.sessionId,
        routingRecommendations: thinkingSession.systemRecommendations,
        conclusions: thinkingSession.conclusions,
        taskId: thinkingSession.taskId
      });
    });

    // Listen for system coordination plan to execute routing logic
    this.on('systemCoordinationPlan', async (coordinationPlan) => {
      logger.logOperation('info', 'SYSTEM_COORDINATION_TRIGGERED', 'Executing system coordination plan', {
        sessionId: coordinationPlan.sessionId,
        primarySystem: coordinationPlan.routingRecommendations?.systemSelection?.primary,
        strategy: coordinationPlan.routingRecommendations?.coordinationStrategy?.strategy
      });

      // Execute routing logic based on Sequential Thinking recommendations
      try {
        await this.executeSystemCoordination(coordinationPlan);
      } catch (coordinationError) {
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

    logger.logOperation('info', 'SYSTEM_COORDINATION_EXECUTION', 'Executing system coordination', {
      sessionId,
      primarySystem: systemSelection.primary,
      strategy: coordinationStrategy.strategy,
      sequenceSteps: executionSequence.length
    });

    // Execute the coordination sequence
    for (const step of executionSequence) {
      try {
        logger.logOperation('debug', 'COORDINATION_STEP', `Executing coordination step ${step.step}`, {
          sessionId,
          system: step.system,
          action: step.action
        });

        // Route to appropriate system based on step configuration
        if (step.system === 'both' || step.system === 'coordinator') {
          // Parallel execution of both systems
          await this.executeParallelSystemCoordination(step, coordinationPlan);
        } else if (step.system === 'Alita') {
          // Route to Alita system components
          await this.routeToAlitaSystem(step, coordinationPlan);
        } else if (step.system === 'KGoT') {
          // Route to KGoT system
          await this.routeToKGoTSystem(step, coordinationPlan);
        }

        logger.logOperation('debug', 'COORDINATION_STEP_COMPLETE', `Coordination step ${step.step} completed`, {
          sessionId,
          system: step.system
        });

      } catch (stepError) {
        logger.logError('COORDINATION_STEP_ERROR', stepError, {
          sessionId,
          step: step.step,
          system: step.system
        });
        
        // Execute fallback strategy if available
        if (routingRecommendations.fallbackStrategy) {
          await this.executeFallbackStrategy(routingRecommendations.fallbackStrategy, stepError);
        }
      }
    }

    logger.logOperation('info', 'SYSTEM_COORDINATION_COMPLETE', 'System coordination completed', {
      sessionId,
      executedSteps: executionSequence.length
    });
  }

  /**
   * Execute parallel system coordination for both Alita and KGoT
   * 
   * @param {Object} step - Coordination step
   * @param {Object} coordinationPlan - Complete coordination plan
   */
  async executeParallelSystemCoordination(step, coordinationPlan) {
    // Implementation for parallel system coordination
    logger.logOperation('info', 'PARALLEL_COORDINATION', 'Executing parallel system coordination', {
      sessionId: coordinationPlan.sessionId,
      step: step.step
    });
    
    // This would be implemented based on specific coordination requirements
    // For now, we log the coordination attempt
  }

  /**
   * Route task to Alita system components
   * 
   * @param {Object} step - Coordination step
   * @param {Object} coordinationPlan - Complete coordination plan
   */
  async routeToAlitaSystem(step, coordinationPlan) {
    logger.logOperation('info', 'ALITA_ROUTING', 'Routing to Alita system', {
      sessionId: coordinationPlan.sessionId,
      action: step.action
    });
    
    // Route to appropriate Alita components based on action
    // This would trigger MCP creation, web agent, or other Alita components
  }

  /**
   * Route task to KGoT system
   * 
   * @param {Object} step - Coordination step
   * @param {Object} coordinationPlan - Complete coordination plan
   */
  async routeToKGoTSystem(step, coordinationPlan) {
    logger.logOperation('info', 'KGOT_ROUTING', 'Routing to KGoT system', {
      sessionId: coordinationPlan.sessionId,
      action: step.action
    });
    
    // Route to KGoT controller for knowledge graph reasoning
    if (this.kgotController) {
      // This would trigger KGoT processing based on the coordination plan
    }
  }

  /**
   * Execute fallback strategy when coordination fails
   * 
   * @param {Object} fallbackStrategy - Fallback strategy configuration
   * @param {Error} error - The error that triggered fallback
   */
  async executeFallbackStrategy(fallbackStrategy, error) {
    logger.logOperation('warn', 'FALLBACK_STRATEGY', 'Executing fallback strategy', {
      strategy: fallbackStrategy.strategy,
      error: error.message
    });
    
    // Implement fallback logic based on strategy type
    switch (fallbackStrategy.strategy) {
      case 'graceful_degradation':
        // Reduce functionality but continue operation
        break;
      case 'retry_with_backoff':
        // Retry with exponential backoff
        break;
      case 'rollback_on_failure':
        // Rollback to previous state
        break;
      default:
        logger.logOperation('warn', 'UNKNOWN_FALLBACK', 'Unknown fallback strategy', {
          strategy: fallbackStrategy.strategy
        });
    }
  }

  /**
   * Initialize the LangChain agent with OpenRouter models
   * Creates the agent executor with necessary tools and prompts
   */
  async initializeAgent() {
    try {
      logger.logOperation('info', 'AGENT_INIT', 'Initializing LangChain agent with OpenRouter');

      // Get model configuration for manager agent
      const managerConfig = modelConfig.alita_config.manager_agent;
      
      // Initialize OpenRouter LLM through OpenAI-compatible API
      const llm = new ChatOpenAI({
        openAIApiKey: process.env.OPENROUTER_API_KEY,
        configuration: {
          baseURL: modelConfig.model_providers.openrouter.base_url,
        },
        modelName: modelConfig.model_providers.openrouter.models[managerConfig.primary_model].model_id,
        temperature: 0.1, // Lower temperature for more consistent orchestration
        maxTokens: 4000,
        timeout: managerConfig.timeout * 1000,
        maxRetries: managerConfig.max_retries,
      });

      // Create agent tools for interacting with other components
      const tools = this.createAgentTools();

      // Create system prompt for the manager agent
      const prompt = ChatPromptTemplate.fromMessages([
        new SystemMessage(`You are the Alita Manager Agent, the central orchestrator of an advanced AI system that combines:

1. **Alita Architecture**: AI assistant capabilities with web agents and MCP creation
2. **Knowledge Graph of Thoughts (KGoT)**: Advanced reasoning using knowledge graphs
3. **Multimodal Processing**: Vision, audio, and text processing capabilities
4. **Validation & Optimization**: Quality assurance and performance optimization

Your responsibilities:
- Analyze incoming requests and determine the best approach
- Coordinate between different agents and services
- Manage task decomposition and workflow orchestration
- Ensure efficient resource utilization and cost optimization
- Handle error recovery and fallback strategies
- Maintain conversation context and state

Available capabilities:
- Web agent for browsing, scraping, and web interactions
- MCP creation for dynamic tool generation (powered by Claude-4-Sonnet for optimal results)
- KGoT controller for knowledge graph operations (dual-LLM: Gemini-2.5-Pro + O3)
- Multimodal processors for various content types
- Validation services for quality assurance (Claude-4-Sonnet primary validation)
- Optimization services for performance tuning

Model Specializations:
- Claude-4-Sonnet: Primary for MCP creation, brainstorming, and validation tasks
- O3: Advanced reasoning, complex problem solving, and tool execution
- Gemini-2.5-Pro: Knowledge graph operations, multimodal processing, large context tasks

Always prioritize:
1. User experience and task completion
2. Cost-effective model usage
3. Error handling and graceful degradation
4. Clear communication and transparency
5. Efficient resource utilization

Use the available tools to accomplish tasks efficiently while maintaining high quality standards.`),
        new MessagesPlaceholder("chat_history"),
        new HumanMessage("{input}"),
        new MessagesPlaceholder("agent_scratchpad"),
      ]);

      // Create the OpenAI Functions agent
      this.agent = await createOpenAIFunctionsAgent({
        llm,
        tools,
        prompt,
      });

      // Create agent executor with error handling
      this.agentExecutor = new AgentExecutor({
        agent: this.agent,
        tools,
        verbose: process.env.NODE_ENV === 'development',
        maxIterations: 10,
        returnIntermediateSteps: true,
        handleParsingErrors: true,
      });

      this.isInitialized = true;
      logger.logOperation('info', 'AGENT_INIT_SUCCESS', 'LangChain agent initialized successfully');

    } catch (error) {
      logger.logError('AGENT_INIT_FAILED', error, { 
        modelConfig: managerConfig,
        openRouterConfig: modelConfig.model_providers.openrouter 
      });
      throw error;
    }
  }

  /**
   * Create tools for the LangChain agent to interact with other components
   * @returns {Array} Array of LangChain tools
   */
  createAgentTools() {
    logger.logOperation('info', 'TOOLS_CREATION', 'Creating agent tools for system components');

    const tools = [
      // Web Agent Tool
      new DynamicTool({
        name: "web_agent",
        description: "Interact with web pages, perform searches, scrape content, and handle web-based tasks. Use this for any web-related operations.",
        func: async (input) => {
          try {
            logger.logOperation('info', 'WEB_AGENT_CALL', 'Calling web agent', { input });
            // TODO: Implement actual web agent API call
            return `Web agent would process: ${input}`;
          } catch (error) {
            logger.logError('WEB_AGENT_ERROR', error, { input });
            return `Error in web agent: ${error.message}`;
          }
        },
      }),

      // MCP Creation Tool - Using Claude-4-Sonnet for optimal MCP generation
      new DynamicTool({
        name: "mcp_creation",
        description: "Create new Model Context Protocol (MCP) tools dynamically based on requirements using Claude-4-Sonnet for optimal reasoning and code generation. Use this when you need to generate custom tools or brainstorm MCP implementations.",
        func: async (input) => {
          try {
            logger.logOperation('info', 'MCP_CREATION_CALL', 'Creating MCP tool with Claude-4-Sonnet', { input });
            
            // Parse input for MCP creation
            let mcpInput;
            try {
              mcpInput = typeof input === 'string' ? JSON.parse(input) : input;
            } catch (parseError) {
              mcpInput = { 
                description: input,
                requirements: ['standard MCP protocol compliance']
              };
            }

            // Initialize Claude-4-Sonnet specifically for MCP tasks
            const claudeMcpLLM = new ChatOpenAI({
              openAIApiKey: process.env.OPENROUTER_API_KEY,
              configuration: {
                baseURL: modelConfig.model_providers.openrouter.base_url,
              },
              modelName: modelConfig.model_providers.openrouter.models['claude-4-sonnet'].model_id,
              temperature: 0.1, // Lower temperature for consistent MCP generation
              maxTokens: 4000,
              timeout: modelConfig.alita_config.mcp_creation.timeout * 1000,
              maxRetries: modelConfig.alita_config.mcp_creation.max_retries,
            });

            // Create MCP generation prompt
            const mcpPrompt = `You are an expert MCP (Model Context Protocol) tool creator using Claude-4-Sonnet capabilities.

Task: Create an MCP tool based on the following requirements:
${JSON.stringify(mcpInput, null, 2)}

Requirements:
1. Follow MCP protocol specifications exactly
2. Provide complete, functional code
3. Include proper error handling and logging
4. Use TypeScript for type safety
5. Include comprehensive JSDoc documentation
6. Follow best practices for tool design

Generate a complete MCP tool implementation with:
- Tool definition and schema
- Implementation logic
- Error handling
- Usage examples
- Integration guidelines

Focus on creating production-ready, efficient code that follows the user's specifications.`;

            // Generate MCP tool using Claude-4-Sonnet
            const mcpResponse = await claudeMcpLLM.invoke([
              { role: 'system', content: 'You are a specialized MCP tool generator focusing on creating high-quality, production-ready tools.' },
              { role: 'user', content: mcpPrompt }
            ]);

            const result = {
              tool_generated: true,
              model_used: 'claude-4-sonnet',
              input: mcpInput,
              mcp_implementation: mcpResponse.content,
              timestamp: new Date().toISOString(),
              generation_metadata: {
                temperature: 0.1,
                max_tokens: 4000,
                model_optimized_for: 'mcp_creation'
              }
            };

            logger.logOperation('info', 'MCP_CREATION_SUCCESS', 'MCP tool created successfully with Claude-4-Sonnet', {
              toolType: mcpInput.description,
              responseLength: mcpResponse.content?.length || 0
            });

            return JSON.stringify(result);
            
          } catch (error) {
            logger.logError('MCP_CREATION_ERROR', error, { input });
            return JSON.stringify({
              error: `Error in MCP creation: ${error.message}`,
              success: false,
              model_attempted: 'claude-4-sonnet',
              timestamp: new Date().toISOString()
            });
          }
        },
      }),

      // KGoT Controller Tool
      new DynamicTool({
        name: "kgot_controller",
        description: "Execute knowledge graph of thoughts reasoning for complex problem solving with dual-LLM architecture. Provides enhanced reasoning through iterative graph construction and tool integration.",
        func: async (input) => {
          try {
            logger.logOperation('info', 'KGOT_CONTROLLER_CALL', 'Calling KGoT controller with integrated reasoning', { input });
            
            if (!this.kgotController) {
              throw new Error('KGoT Controller not initialized');
            }

            // Parse input if it's a string
            let taskInput;
            try {
              taskInput = typeof input === 'string' ? JSON.parse(input) : input;
            } catch (parseError) {
              // If parsing fails, treat input as plain task string
              taskInput = { task: input };
            }
            
            // Execute KGoT reasoning with full workflow
            const executionResult = await this.kgotController.executeTask(
              taskInput.task || taskInput,
              {
                context: taskInput.context || {},
                priority: taskInput.priority || 'medium',
                expectedOutputType: taskInput.expectedOutputType || 'solution',
                enableValidation: true
              }
            );

            logger.logOperation('info', 'KGOT_EXECUTION_COMPLETE', 'KGoT reasoning completed', {
              iterations: executionResult.iterations,
              graphSize: executionResult.graph ? Object.keys(executionResult.graph.vertices).length : 0,
              hasValidation: !!executionResult.validation
            });

            return JSON.stringify(executionResult);
            
          } catch (error) {
            logger.logError('KGOT_CONTROLLER_ERROR', error, { input });
            return JSON.stringify({
              error: `Error in KGoT reasoning: ${error.message}`,
              success: false,
              timestamp: new Date().toISOString()
            });
          }
        },
      }),

      // Multimodal Processing Tool
      new DynamicTool({
        name: "multimodal_processor",
        description: "Process images, audio, and other non-text content. Use this for vision, audio analysis, or document processing tasks.",
        func: async (input) => {
          try {
            logger.logOperation('info', 'MULTIMODAL_CALL', 'Processing multimodal content', { input });
            // TODO: Implement actual multimodal processor API call
            return `Multimodal processor would handle: ${input}`;
          } catch (error) {
            logger.logError('MULTIMODAL_ERROR', error, { input });
            return `Error in multimodal processing: ${error.message}`;
          }
        },
      }),

      // MCP Cross-Validation Tool
      new DynamicTool({
        name: "validation_service",
        description: "Validate outputs using multi-model cross-validation with consensus-based quality assessment. Provides comprehensive quality assurance through multiple LLM perspectives.",
        func: async (input) => {
          try {
            logger.logOperation('info', 'VALIDATION_CALL', 'Performing MCP cross-validation', { input });
            
            if (!this.mcpValidator) {
              throw new Error('MCP Validator not initialized');
            }

            // Parse input for validation
            let validationInput;
            try {
              validationInput = typeof input === 'string' ? JSON.parse(input) : input;
            } catch (parseError) {
              validationInput = { solution: input };
            }

            // Perform cross-model validation
            const validationResult = await this.mcpValidator.validateSolution(
              validationInput.solution || validationInput,
              validationInput.originalTask || 'Task validation',
              validationInput.context || {}
            );

            logger.logOperation('info', 'VALIDATION_COMPLETE', 'MCP validation completed', {
              validationId: validationResult.validationId,
              isValid: validationResult.isValid,
              confidence: validationResult.confidence,
              consensusScore: validationResult.consensus?.score
            });

            return JSON.stringify(validationResult);
            
          } catch (error) {
            logger.logError('VALIDATION_ERROR', error, { input });
            return JSON.stringify({
              error: `Error in validation: ${error.message}`,
              success: false,
              timestamp: new Date().toISOString()
            });
          }
        },
      }),

      // Optimization Tool
      new DynamicTool({
        name: "optimization_service",
        description: "Optimize performance, reduce costs, and improve efficiency. Use this for performance tuning and resource optimization.",
        func: async (input) => {
          try {
            logger.logOperation('info', 'OPTIMIZATION_CALL', 'Optimizing performance', { input });
            // TODO: Implement actual optimization service API call
            return `Optimization service would optimize: ${input}`;
          } catch (error) {
            logger.logError('OPTIMIZATION_ERROR', error, { input });
            return `Error in optimization: ${error.message}`;
          }
        },
      }),
    ];

    // Add Sequential Thinking tool if available
    if (this.sequentialThinking) {
      tools.push(this.sequentialThinking.createSequentialThinkingTool());
      logger.logOperation('info', 'SEQUENTIAL_THINKING_TOOL_ADDED', 'Sequential Thinking tool added to agent tools');
    }

    logger.logOperation('info', 'TOOLS_CREATED', `Created ${tools.length} tools for agent`);
    return tools;
  }

  /**
   * Setup Express routes for the manager agent API
   */
  setupRoutes() {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({ 
        status: 'healthy', 
        initialized: this.isInitialized,
        timestamp: new Date().toISOString()
      });
    });

    // Main chat endpoint for processing user requests
    this.app.post('/chat', async (req, res) => {
      try {
        const { message, context = [], sessionId } = req.body;

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

        const duration = Date.now() - startTime;
        
        logger.logOperation('info', 'CHAT_RESPONSE', 'Chat request processed successfully', {
          sessionId,
          duration,
          responseLength: result.output?.length || 0,
          intermediateSteps: result.intermediateSteps?.length || 0
        });

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
        logger.logError('CHAT_ERROR', error, { 
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

    // System status endpoint
    this.app.get('/status', async (req, res) => {
      try {
        // Check status of all connected services
        const status = {
          managerAgent: 'healthy',
          webAgent: 'unknown', // TODO: Implement health checks
          mcpCreation: 'unknown',
          kgotController: this.kgotController ? 'healthy' : 'not_initialized',
          multimodal: 'unknown',
          validation: this.mcpValidator ? 'healthy' : 'not_initialized',
          optimization: 'unknown',
          sequentialThinking: this.sequentialThinking ? 'healthy' : 'not_initialized',
        };

        res.json({
          status: 'operational',
          services: status,
          sequentialThinking: {
            activeSessions: this.sequentialThinking?.getActiveSessions()?.length || 0,
            complexityAnalysisCache: this.sequentialThinking?.complexityScores?.size || 0
          },
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        logger.logError('STATUS_CHECK_ERROR', error);
        res.status(500).json({ error: 'Status check failed' });
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
  }

  /**
   * Start the Express server
   */
  async start() {
    try {
      await new Promise((resolve) => {
        this.server = this.app.listen(this.port, () => {
          logger.logOperation('info', 'SERVER_START', `Alita Manager Agent started on port ${this.port}`);
          resolve();
        });
      });
    } catch (error) {
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

// Start the server if this file is run directly
if (require.main === module) {
  (async () => {
    try {
      managerAgent = new AlitaManagerAgent();
      await managerAgent.start();
    } catch (error) {
      logger.logError('STARTUP_ERROR', error);
      process.exit(1);
    }
  })();
}

module.exports = AlitaManagerAgent;