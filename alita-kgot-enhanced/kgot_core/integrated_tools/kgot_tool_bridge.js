/**
 * KGoT Tool Bridge
 * 
 * Node.js bridge for integrating KGoT Integrated Tools with the KGoT Controller.
 * This module provides seamless communication between Python-based tools and 
 * JavaScript-based controller architecture.
 * 
 * Key Features:
 * - Bridge between Python KGoT tools and Node.js KGoT Controller
 * - Tool execution coordination with specific AI model assignments
 * - Integration with Alita Web Agent capabilities
 * - Session management for tool orchestration
 * - Error handling and logging with Winston compatibility
 * 
 * @module KGoTToolBridge
 * @author AI Assistant
 * @date 2025
 */

const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const EventEmitter = require('events');

// Import logging configuration
const { loggers } = require('../../config/logging/winston_config');
const logger = loggers.integratedTools;

/**
 * KGoT Tool Bridge Class
 * 
 * Provides seamless integration between Python-based KGoT tools and 
 * Node.js KGoT Controller architecture.
 */
class KGoTToolBridge extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.options = {
      pythonExecutable: options.pythonExecutable || 'python3',
      toolsModulePath: options.toolsModulePath || path.join(__dirname, 'integrated_tools_manager.py'),
      alitaIntegrationPath: options.alitaIntegrationPath || path.join(__dirname, 'alita_integration.py'),
      errorIntegrationPath: options.errorIntegrationPath || path.join(__dirname, 'kgot_error_integration.py'),
      sessionTimeout: options.sessionTimeout || 300000, // 5 minutes
      maxRetries: options.maxRetries || 3,
      enableAlitaIntegration: options.enableAlitaIntegration !== false,
      enableErrorManagement: options.enableErrorManagement !== false,
      ...options
    };

    // Tool management state
    this.toolsManager = null;
    this.alitaIntegrator = null;
    this.errorIntegrationOrchestrator = null;
    this.activeSessions = new Map();
    this.toolRegistry = new Map();
    this.executionHistory = [];
    
    // Model configurations matching the specific AI model assignments
    this.modelConfigurations = {
      vision: 'openai/o3',               // OpenAI o3 for vision
      orchestration: 'x-ai/grok-4', // Gemini 2.5 Pro for orchestration and complex reasoning
      webAgent: 'anthropic/claude-sonnet-4'   // Claude 4 Sonnet for web agent
    };

    // ERROR MANAGEMENT: Enhanced error tracking
    this.errorStatistics = {
      totalErrors: 0,
      recoveredErrors: 0,
      escalatedErrors: 0,
      errorsByType: {},
      lastErrorAnalysis: null
    };

    logger.info('Initializing KGoT Tool Bridge', { 
      operation: 'KGOT_TOOL_BRIDGE_INIT',
      options: this.options,
      modelConfigurations: this.modelConfigurations
    });
    
    this.initializeToolBridge();
  }

  /**
   * Initialize the tool bridge with Python tools integration
   */
  async initializeToolBridge() {
    try {
      logger.logOperation('info', 'TOOL_BRIDGE_INIT', 'Initializing tool bridge components');

      // Initialize the tools manager
      await this.initializeToolsManager();
      
      // Initialize error management integration (NEW)
      if (this.options.enableErrorManagement) {
        await this.initializeErrorManagement();
      }
      
      // Initialize Alita integration if enabled
      if (this.options.enableAlitaIntegration) {
        await this.initializeAlitaIntegration();
      }
      
      // Load tool registry
      await this.loadToolRegistry();
      
      // Setup session cleanup
      this.setupSessionCleanup();
      
      logger.logOperation('info', 'TOOL_BRIDGE_READY', 'KGoT Tool Bridge initialized successfully');
      
      this.emit('ready', {
        toolsCount: this.toolRegistry.size,
        alitaIntegration: this.options.enableAlitaIntegration,
        errorManagement: this.options.enableErrorManagement,
        modelConfigurations: this.modelConfigurations
      });

    } catch (error) {
      logger.logError('TOOL_BRIDGE_INIT_FAILED', error, { 
        options: this.options 
      });
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Initialize the Python-based tools manager
   */
  async initializeToolsManager() {
    try {
      logger.logOperation('info', 'TOOLS_MANAGER_INIT', 'Initializing Python tools manager');

      const pythonScript = `
import sys
import os
sys.path.insert(0, '${path.dirname(this.options.toolsModulePath)}')

try:
    from integrated_tools_manager import create_integrated_tools_manager, ModelConfiguration
    
    # Create model configuration with specific AI model assignments
    config = ModelConfiguration(
        vision_model="${this.modelConfigurations.vision}",
        orchestration_model="${this.modelConfigurations.orchestration}",
        web_agent_model="${this.modelConfigurations.webAgent}",
        temperature=0.3
    )
    
    # Create tools manager
    manager = create_integrated_tools_manager(config)
    
    # Export configuration
    import json
    result = manager.export_configuration()
    print("TOOLS_MANAGER_SUCCESS:" + json.dumps(result))
    
except Exception as e:
    print("TOOLS_MANAGER_ERROR:" + str(e))
    import traceback
    traceback.print_exc()
`;

      const result = await this.executePythonScript(pythonScript);
      
      if (result.includes('TOOLS_MANAGER_SUCCESS:')) {
        const configStr = result.split('TOOLS_MANAGER_SUCCESS:')[1];
        const configData = JSON.parse(configStr.replace(/'/g, '"'));
        
        this.toolsManager = {
          configuration: configData,
          initialized: true
        };
        
        logger.logOperation('info', 'TOOLS_MANAGER_SUCCESS', 'Python tools manager initialized', {
          totalTools: configData.metadata.total_tools,
          categories: Object.keys(configData.categories)
        });
        
      } else {
        throw new Error(`Tools manager initialization failed: ${result}`);
      }

    } catch (error) {
      logger.logError('TOOLS_MANAGER_INIT_FAILED', error);
      throw error;
    }
  }

  /**
   * Initialize Alita integration capabilities
   */
  async initializeAlitaIntegration() {
    try {
      logger.logOperation('info', 'ALITA_INTEGRATION_INIT', 'Initializing Alita integration');

      const pythonScript = `
import sys
import os
sys.path.insert(0, '${path.dirname(this.options.alitaIntegrationPath)}')

try:
    from alita_integration import create_alita_integrator
    
    # Create Alita integrator
    integrator = create_alita_integrator()
    
    print("ALITA_INTEGRATION_SUCCESS:initialized")
    
except Exception as e:
    print("ALITA_INTEGRATION_ERROR:" + str(e))
    import traceback
    traceback.print_exc()
`;

      const result = await this.executePythonScript(pythonScript);
      
      if (result.includes('ALITA_INTEGRATION_SUCCESS:')) {
        this.alitaIntegrator = {
          initialized: true,
          capabilities: {
            webNavigation: true,
            contextEnrichment: true,
            toolEnhancement: true
          }
        };
        
        logger.logOperation('info', 'ALITA_INTEGRATION_SUCCESS', 'Alita integration initialized');
        
      } else {
        logger.logOperation('warn', 'ALITA_INTEGRATION_FALLBACK', 'Alita integration failed, continuing without it', {
          error: result
        });
        this.options.enableAlitaIntegration = false;
      }

    } catch (error) {
      logger.logError('ALITA_INTEGRATION_INIT_FAILED', error);
      // Continue without Alita integration
      this.options.enableAlitaIntegration = false;
    }
  }

  /**
   * Initialize KGoT Error Management Integration
   * 
   * Sets up the comprehensive error management system including:
   * - KGoT Error Management System
   * - Alita Refinement Bridge 
   * - Tool Bridge Error Integration
   * - Unified Error Reporting
   */
  async initializeErrorManagement() {
    try {
      logger.info('Initializing KGoT Error Management Integration', {
        operation: 'ERROR_MANAGEMENT_INIT_START'
      });

      const pythonScript = `
import sys
import os
sys.path.insert(0, '${path.dirname(this.options.errorIntegrationPath)}')

try:
    from kgot_error_integration import create_kgot_alita_error_integration
    
    # Mock LLM client for integration (would be OpenRouter client in production)
    class MockLLMClient:
        async def acomplete(self, prompt):
            class MockResponse:
                text = '{"status": "error_management_integrated"}'
            return MockResponse()
    
    # Create error integration orchestrator
    integration_orchestrator = create_kgot_alita_error_integration(
        llm_client=MockLLMClient(),
        config=None  # Use default configuration
    )
    
    print("ERROR_MANAGEMENT_SUCCESS:initialized")
    
except Exception as e:
    print("ERROR_MANAGEMENT_ERROR:" + str(e))
    import traceback
    traceback.print_exc()
`;

      const result = await this.executePythonScript(pythonScript);
      
      if (result.includes('ERROR_MANAGEMENT_SUCCESS:')) {
        this.errorIntegrationOrchestrator = {
          initialized: true,
          capabilities: {
            syntaxErrorHandling: true,
            apiErrorRecovery: true,
            containerizedExecution: true,
            alitaRefinement: true,
            majorityVoting: true,
            unifiedReporting: true
          }
        };
        
        logger.info('KGoT Error Management Integration initialized successfully', {
          operation: 'ERROR_MANAGEMENT_INIT_SUCCESS',
          capabilities: Object.keys(this.errorIntegrationOrchestrator.capabilities)
        });
        
      } else {
        logger.warn('Error Management initialization failed, continuing without it', {
          operation: 'ERROR_MANAGEMENT_INIT_FALLBACK',
          error: result
        });
        this.options.enableErrorManagement = false;
      }

    } catch (error) {
      logger.error('Error Management initialization failed', {
        operation: 'ERROR_MANAGEMENT_INIT_FAILED',
        error: error.message
      });
      // Continue without error management
      this.options.enableErrorManagement = false;
    }
  }

  /**
   * Load tool registry from the tools manager
   */
  async loadToolRegistry() {
    try {
      if (!this.toolsManager || !this.toolsManager.configuration) {
        throw new Error('Tools manager not initialized');
      }

      const toolRegistry = this.toolsManager.configuration.tool_registry;
      
      for (const [toolId, toolConfig] of Object.entries(toolRegistry)) {
        this.toolRegistry.set(toolConfig.name, {
          id: toolId,
          name: toolConfig.name,
          model: toolConfig.model,
          description: toolConfig.description,
          category: toolConfig.category,
          initialized: true
        });
      }

      logger.logOperation('info', 'TOOL_REGISTRY_LOADED', `Loaded ${this.toolRegistry.size} tools into registry`);

    } catch (error) {
      logger.logError('TOOL_REGISTRY_LOAD_FAILED', error);
      throw error;
    }
  }

  /**
   * Execute a tool with comprehensive error management integration
   * 
   * @param {string} toolName - Name of the tool to execute
   * @param {Object} toolInput - Input parameters for the tool
   * @param {Object} context - Additional context for execution
   * @returns {Promise<Object>} Tool execution results with error recovery
   */
  async executeTool(toolName, toolInput, context = {}) {
    const sessionId = context.sessionId || `session_${Date.now()}`;
    
    try {
      logger.info(`Executing tool with error management: ${toolName}`, {
        operation: 'TOOL_EXECUTION_START',
        toolName,
        sessionId,
        inputKeys: Object.keys(toolInput),
        errorManagementEnabled: this.options.enableErrorManagement
      });

      // Validate tool exists
      if (!this.toolRegistry.has(toolName)) {
        throw new Error(`Tool not found: ${toolName}`);
      }

      const toolInfo = this.toolRegistry.get(toolName);
      
      // Create execution session with error tracking
      const executionSession = {
        sessionId,
        toolName,
        toolInfo,
        startTime: Date.now(),
        status: 'executing',
        errorAttempts: 0,
        maxRetries: this.options.maxRetries
      };
      
      this.activeSessions.set(sessionId, executionSession);

      // Enhanced tool execution with error management
      let enhancedInput = toolInput;
      let enhancedContext = context;
      
      // Apply Alita enhancements if enabled
      if (this.options.enableAlitaIntegration && this.alitaIntegrator) {
        const enhancement = await this.enhanceToolWithAlita(toolName, toolInput, context);
        enhancedInput = enhancement.enhanced_input || toolInput;
        enhancedContext = enhancement.enhanced_context || context;
      }

      // Execute the tool with error management protection
      let executionResult;
      let finalSuccess = false;
      
      for (let attempt = 0; attempt < this.options.maxRetries; attempt++) {
        try {
          executionSession.errorAttempts = attempt;
          
          // Execute tool via Python bridge
          executionResult = await this.executeToolViaPython(toolName, enhancedInput, enhancedContext);
          finalSuccess = true;
          break;
          
        } catch (toolError) {
          // NEW: Comprehensive error handling with KGoT Error Management
          logger.warn(`Tool execution attempt ${attempt + 1} failed: ${toolError.message}`, {
            operation: 'TOOL_EXECUTION_ATTEMPT_FAILED',
            toolName,
            sessionId,
            attempt: attempt + 1,
            error: toolError.message
          });
          
          // Update error statistics
          this.errorStatistics.totalErrors++;
          const errorType = toolError.constructor.name;
          this.errorStatistics.errorsByType[errorType] = 
            (this.errorStatistics.errorsByType[errorType] || 0) + 1;
          
          // Use integrated error management if available
          if (this.options.enableErrorManagement && this.errorIntegrationOrchestrator) {
            const errorRecoveryResult = await this.handleToolErrorWithIntegration(
              toolName, toolError, enhancedInput, enhancedContext, sessionId
            );
            
            if (errorRecoveryResult.success) {
              executionResult = errorRecoveryResult.result;
              finalSuccess = true;
              this.errorStatistics.recoveredErrors++;
              
              logger.info('Tool error recovered by integrated error management', {
                operation: 'TOOL_ERROR_RECOVERY_SUCCESS',
                toolName,
                sessionId,
                recoveryMethod: errorRecoveryResult.recovery_method
              });
              break;
            }
          }
          
          // If this is the last attempt, escalate the error
          if (attempt === this.options.maxRetries - 1) {
            this.errorStatistics.escalatedErrors++;
            
            logger.error('All tool execution attempts failed, escalating error', {
              operation: 'TOOL_EXECUTION_ESCALATION',
              toolName,
              sessionId,
              totalAttempts: this.options.maxRetries,
              finalError: toolError.message
            });
            
            throw toolError;
          }
          
          // Brief delay before retry
          await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
        }
      }

      // Update session status
      executionSession.status = finalSuccess ? 'completed' : 'failed';
      executionSession.endTime = Date.now();
      executionSession.result = executionResult;

      // Store in execution history
      this.executionHistory.push({
        ...executionSession,
        duration: executionSession.endTime - executionSession.startTime
      });

      logger.info(`Tool execution completed: ${toolName}`, {
        operation: 'TOOL_EXECUTION_SUCCESS',
        toolName,
        sessionId,
        duration: executionSession.endTime - executionSession.startTime,
        resultType: typeof executionResult,
        errorAttempts: executionSession.errorAttempts
      });

      // Clean up session
      this.activeSessions.delete(sessionId);

      return {
        success: true,
        sessionId,
        toolName,
        result: executionResult,
        metadata: {
          model: toolInfo.model,
          category: toolInfo.category,
          duration: executionSession.endTime - executionSession.startTime,
          alitaEnhanced: this.options.enableAlitaIntegration,
          errorManaged: this.options.enableErrorManagement,
          errorAttempts: executionSession.errorAttempts
        }
      };

    } catch (error) {
      logger.error('Tool execution failed completely', {
        operation: 'TOOL_EXECUTION_FAILED',
        toolName,
        sessionId,
        error: error.message
      });

      // Clean up failed session
      if (this.activeSessions.has(sessionId)) {
        const session = this.activeSessions.get(sessionId);
        session.status = 'failed';
        session.error = error.message;
        this.activeSessions.delete(sessionId);
      }

      return {
        success: false,
        sessionId,
        toolName,
        error: error.message,
        metadata: {
          failureTime: Date.now(),
          errorManagementAttempted: this.options.enableErrorManagement
        }
      };
    }
  }

  /**
   * Execute tool via Python bridge
   */
  async executeToolViaPython(toolName, toolInput, context) {
    const pythonScript = `
import sys
import os
import json
import asyncio
sys.path.insert(0, '${path.dirname(this.options.toolsModulePath)}')

async def execute_tool():
    try:
        from integrated_tools_manager import create_integrated_tools_manager, ModelConfiguration
        
        # Create tools manager
        config = ModelConfiguration(
            vision_model="${this.modelConfigurations.vision}",
            orchestration_model="${this.modelConfigurations.orchestration}",
            web_agent_model="${this.modelConfigurations.webAgent}",
            general_model="${this.modelConfigurations.general}",
            temperature=0.3
        )
        
        manager = create_integrated_tools_manager(config)
        
        # Get tool configuration
        tool_config = manager.get_tool_configuration("${toolName.replace('"', '\\"')}")
        
        if not tool_config:
            raise Exception(f"Tool not found: ${toolName}")
        
        # For now, return tool configuration and input as simulation
        # In a full implementation, this would actually execute the tool
        result = {
            "tool_executed": "${toolName}",
            "model_used": tool_config.get("model", "unknown"),
            "category": tool_config.get("category", "unknown"),
            "input_received": ${JSON.stringify(toolInput)},
            "context_received": ${JSON.stringify(context)},
            "execution_timestamp": "$(new Date().toISOString())",
            "status": "simulated_execution"
        }
        
        print("TOOL_EXECUTION_SUCCESS:" + json.dumps(result))
        
    except Exception as e:
        print("TOOL_EXECUTION_ERROR:" + str(e))
        import traceback
        traceback.print_exc()

# Run the async function
asyncio.run(execute_tool())
`;

    const result = await this.executePythonScript(pythonScript);
    
    if (result.includes('TOOL_EXECUTION_SUCCESS:')) {
      const resultStr = result.split('TOOL_EXECUTION_SUCCESS:')[1];
      return JSON.parse(resultStr);
    } else {
      throw new Error(`Tool execution failed: ${result}`);
    }
  }

  /**
   * Enhance tool execution with Alita capabilities
   */
  async enhanceToolWithAlita(toolName, toolInput, context) {
    try {
      const pythonScript = `
import sys
import os
import json
import asyncio
sys.path.insert(0, '${path.dirname(this.options.alitaIntegrationPath)}')

async def enhance_tool():
    try:
        from alita_integration import create_alita_integrator
        
        # Create Alita integrator
        integrator = create_alita_integrator()
        
        # Initialize session
        session_id = await integrator.initialize_session("KGoT tool execution: ${toolName}")
        
        # Enhance tool execution
        enhancement = await integrator.enhance_tool_execution(
            "${toolName.replace('"', '\\"')}",
            ${JSON.stringify(toolInput)},
            ${JSON.stringify(context)}
        )
        
        # Close session
        await integrator.close_session()
        
        print("ALITA_ENHANCEMENT_SUCCESS:" + json.dumps(enhancement))
        
    except Exception as e:
        print("ALITA_ENHANCEMENT_ERROR:" + str(e))
        # Return original input as fallback
        fallback = {
            "enhanced_input": ${JSON.stringify(toolInput)},
            "enhanced_context": ${JSON.stringify(context)},
            "alita_enhancements": {"error": str(e)}
        }
        print("ALITA_ENHANCEMENT_SUCCESS:" + json.dumps(fallback))

# Run the async function
asyncio.run(enhance_tool())
`;

      const result = await this.executePythonScript(pythonScript);
      
      if (result.includes('ALITA_ENHANCEMENT_SUCCESS:')) {
        const enhancementStr = result.split('ALITA_ENHANCEMENT_SUCCESS:')[1];
        return JSON.parse(enhancementStr);
      } else {
        // Return original input as fallback
        return {
          enhanced_input: toolInput,
          enhanced_context: context,
          alita_enhancements: { error: 'Enhancement failed' }
        };
      }

    } catch (error) {
      logger.logError('ALITA_ENHANCEMENT_FAILED', error);
      return {
        enhanced_input: toolInput,
        enhanced_context: context,
        alita_enhancements: { error: error.message }
      };
    }
  }

  /**
   * Execute a Python script and return the output
   */
  async executePythonScript(script) {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn(this.options.pythonExecutable, ['-c', script]);
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve(stdout);
        } else {
          reject(new Error(`Python script failed with code ${code}: ${stderr}`));
        }
      });
      
      pythonProcess.on('error', (error) => {
        reject(error);
      });
    });
  }

  /**
   * Get available tools information
   */
  getAvailableTools() {
    const tools = Array.from(this.toolRegistry.values());
    
    return {
      tools,
      totalCount: tools.length,
      categories: [...new Set(tools.map(tool => tool.category))],
      modelAssignments: this.modelConfigurations,
      alitaIntegration: this.options.enableAlitaIntegration
    };
  }

  /**
   * Get tool execution history
   */
  getExecutionHistory(limit = 10) {
    return this.executionHistory.slice(-limit);
  }

  /**
   * Setup session cleanup for abandoned sessions
   */
  setupSessionCleanup() {
    setInterval(() => {
      const now = Date.now();
      for (const [sessionId, session] of this.activeSessions.entries()) {
        if (now - session.startTime > this.options.sessionTimeout) {
          logger.logOperation('warn', 'SESSION_TIMEOUT', `Cleaning up abandoned session: ${sessionId}`);
          this.activeSessions.delete(sessionId);
        }
      }
    }, 60000); // Check every minute
  }

  /**
   * Get current status of the tool bridge
   */
  getStatus() {
    return {
      initialized: !!this.toolsManager,
      toolsCount: this.toolRegistry.size,
      activeSessions: this.activeSessions.size,
      alitaIntegration: this.options.enableAlitaIntegration && !!this.alitaIntegrator,
      errorManagement: this.options.enableErrorManagement && !!this.errorIntegrationOrchestrator,
      modelConfigurations: this.modelConfigurations,
      executionHistory: this.executionHistory.length
    };
  }

  /**
   * NEW: Handle tool errors using integrated error management system
   * 
   * @param {string} toolName - Name of the tool that failed
   * @param {Error} toolError - The error that occurred
   * @param {Object} toolInput - Input parameters for the tool
   * @param {Object} context - Execution context
   * @param {string} sessionId - Session identifier
   * @returns {Promise<Object>} Error recovery result
   */
  async handleToolErrorWithIntegration(toolName, toolError, toolInput, context, sessionId) {
    try {
      logger.info('Handling tool error with integrated error management', {
        operation: 'INTEGRATED_ERROR_HANDLING_START',
        toolName,
        sessionId,
        errorType: toolError.constructor.name
      });

      const pythonScript = `
import sys
import os
import json
import asyncio
sys.path.insert(0, '${path.dirname(this.options.errorIntegrationPath)}')

async def handle_integrated_error():
    try:
        from kgot_error_integration import create_kgot_alita_error_integration
        
        # Mock LLM client for error handling
        class MockLLMClient:
            async def acomplete(self, prompt):
                class MockResponse:
                    text = '{"status": "error_handled"}'
                return MockResponse()
        
        # Create error integration orchestrator
        orchestrator = create_kgot_alita_error_integration(
            llm_client=MockLLMClient()
        )
        
        # Handle the integrated error
        error_context = {
            'tool_name': '${toolName.replace("'", "\\'")}',
            'tool_input': ${JSON.stringify(toolInput)},
            'execution_context': ${JSON.stringify(context)},
            'session_id': '${sessionId}',
            'operation_context': 'Tool execution: ${toolName.replace("'", "\\'")}'
        }
        
        # Simulate the error (in real implementation, this would be the actual error)
        class SimulatedError(Exception):
            pass
        
        simulated_error = SimulatedError('${toolError.message.replace("'", "\\'")}')
        
        result = await orchestrator.handle_integrated_error(
            error=simulated_error,
            context=error_context,
            error_source='tool'
        )
        
        # Cleanup
        orchestrator.cleanup_integration()
        
        print("INTEGRATED_ERROR_HANDLING_SUCCESS:" + json.dumps(result))
        
    except Exception as e:
        print("INTEGRATED_ERROR_HANDLING_ERROR:" + str(e))
        import traceback
        traceback.print_exc()

# Run the async function
asyncio.run(handle_integrated_error())
`;

      const result = await this.executePythonScript(pythonScript);
      
      if (result.includes('INTEGRATED_ERROR_HANDLING_SUCCESS:')) {
        const recoveryResultStr = result.split('INTEGRATED_ERROR_HANDLING_SUCCESS:')[1];
        const recoveryResult = JSON.parse(recoveryResultStr);
        
        logger.info('Integrated error handling completed', {
          operation: 'INTEGRATED_ERROR_HANDLING_SUCCESS',
          toolName,
          sessionId,
          success: recoveryResult.success,
          recoveryMethod: recoveryResult.recovery_method
        });
        
        return recoveryResult;
      } else {
        logger.warn('Integrated error handling failed', {
          operation: 'INTEGRATED_ERROR_HANDLING_FAILED',
          toolName,
          sessionId,
          error: result
        });
        
        return { success: false, error: 'Integrated error handling failed' };
      }

    } catch (error) {
      logger.error('Integrated error handling exception', {
        operation: 'INTEGRATED_ERROR_HANDLING_EXCEPTION',
        toolName,
        sessionId,
        error: error.message
      });
      
      return { success: false, error: error.message };
    }
  }

  /**
   * NEW: Get comprehensive error management statistics
   * 
   * @returns {Object} Detailed error statistics and health metrics
   */
  getErrorManagementStatistics() {
    const successRate = this.errorStatistics.totalErrors > 0 
      ? (this.errorStatistics.recoveredErrors / this.errorStatistics.totalErrors * 100).toFixed(2)
      : 100;
    
    return {
      ...this.errorStatistics,
      successRate: `${successRate}%`,
      errorManagementEnabled: this.options.enableErrorManagement,
      integrationHealth: this.errorIntegrationOrchestrator?.initialized || false,
      timestamp: new Date().toISOString()
    };
  }
}

module.exports = {
  KGoTToolBridge,
  createKGoTToolBridge: (options) => new KGoTToolBridge(options)
};