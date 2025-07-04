/**
 * Sequential Thinking MCP Integration for Alita Manager Agent
 * 
 * This module integrates the Sequential Thinking MCP as the core reasoning engine
 * for the LangChain-based manager agent. It provides sophisticated complexity detection,
 * systematic routing logic, and intelligent decision trees for coordinating between
 * Alita MCP creation and KGoT knowledge processing.
 * 
 * Key Features:
 * - Complexity detection triggers (task complexity >7, errors >3, cross-system coordination)
 * - Thought process templates for common scenarios
 * - Systematic routing between Alita and KGoT systems
 * - Intelligent decision trees for multi-system coordination
 * - Integration with existing LangChain agent architecture
 * 
 * @module SequentialThinkingIntegration
 */

const { DynamicTool } = require('@langchain/core/tools');
const { ChatPromptTemplate } = require('@langchain/core/prompts');
const EventEmitter = require('events');

// Import logging configuration
const { loggers } = require('../../config/logging/winston_config');
const logger = loggers.managerAgent;

/**
 * Sequential Thinking MCP Integration Class
 * Provides core reasoning capabilities for complex task coordination
 * Extends EventEmitter for seamless integration with Manager Agent
 */
class SequentialThinkingIntegration extends EventEmitter {
  constructor(options = {}) {
    super();
    
    /**
     * Configuration for sequential thinking integration
     */
    this.config = {
      // Complexity detection thresholds
      complexityThreshold: options.complexityThreshold || 7,
      errorThreshold: options.errorThreshold || 3,
      maxThoughts: options.maxThoughts || 15,
      
      // Reasoning engine configuration
      enableCrossSystemCoordination: options.enableCrossSystemCoordination || true,
      enableAdaptiveThinking: options.enableAdaptiveThinking || true,
      thoughtTimeout: options.thoughtTimeout || 30000, // 30 seconds
      
      // Integration settings
      enableMCPCreationRouting: options.enableMCPCreationRouting || true,
      enableKGoTProcessing: options.enableKGoTProcessing || true,
      ...options
    };

    /**
     * State tracking for sequential thinking sessions
     */
    this.activeThinkingSessions = new Map();
    this.complexityScores = new Map();
    this.systemCoordinationHistory = [];
    
    /**
     * Thought process templates for different scenarios
     */
    this.thoughtTemplates = this.initializeThoughtTemplates();
    
    /**
     * Decision trees for system routing
     */
    this.decisionTrees = this.initializeDecisionTrees();

    logger.logOperation('info', 'SEQUENTIAL_THINKING_INIT', 'Sequential Thinking MCP Integration initialized', {
      config: this.config
    });
  }

  /**
   * Initialize thought process templates for common scenarios
   * These templates guide the reasoning process for different types of complex tasks
   * 
   * @returns {Object} Collection of thought process templates
   */
  initializeThoughtTemplates() {
    logger.logOperation('debug', 'TEMPLATE_INIT', 'Initializing thought process templates');

    return {
      /**
       * Task decomposition template for breaking down complex requests
       */
      taskDecomposition: {
        name: 'Task Decomposition',
        description: 'Break down complex tasks into manageable subtasks',
        triggerConditions: ['complexity_score > 7', 'multiple_domains', 'unclear_requirements'],
        thoughtSteps: [
          'Analyze the overall task complexity and scope',
          'Identify key domains and technologies involved',
          'Break down into logical subtasks with clear dependencies',
          'Prioritize subtasks based on dependencies and criticality',
          'Validate decomposition against original requirements',
          'Create execution plan with resource allocation'
        ],
        expectedOutputs: ['subtask_list', 'dependency_graph', 'execution_plan']
      },

      /**
       * Error resolution template for systematic problem-solving
       */
      errorResolution: {
        name: 'Error Resolution',
        description: 'Systematically resolve multiple or complex errors',
        triggerConditions: ['error_count > 3', 'cascading_failures', 'system_integration_errors'],
        thoughtSteps: [
          'Categorize and prioritize all detected errors',
          'Identify root causes and error propagation patterns',
          'Determine which errors are symptoms vs. causes',
          'Plan resolution sequence to minimize cascading effects',
          'Identify required tools and system components for resolution',
          'Validate resolution plan against system constraints'
        ],
        expectedOutputs: ['error_hierarchy', 'resolution_sequence', 'tool_requirements']
      },

      /**
       * System coordination template for multi-system scenarios
       */
      systemCoordination: {
        name: 'System Coordination',
        description: 'Coordinate complex operations across Alita and KGoT systems',
        triggerConditions: ['cross_system_required', 'data_synchronization', 'workflow_orchestration'],
        thoughtSteps: [
          'Map required capabilities to appropriate systems (Alita vs KGoT)',
          'Identify data flow and synchronization points',
          'Plan inter-system communication and coordination',
          'Design fallback strategies for system failures',
          'Optimize workflow for parallel vs. sequential execution',
          'Validate coordination plan against system limitations'
        ],
        expectedOutputs: ['system_mapping', 'coordination_plan', 'fallback_strategies']
      },

      /**
       * Knowledge processing template for complex reasoning tasks
       */
      knowledgeProcessing: {
        name: 'Knowledge Processing',
        description: 'Process and integrate knowledge from multiple sources',
        triggerConditions: ['knowledge_integration', 'complex_reasoning', 'multi_source_analysis'],
        thoughtSteps: [
          'Identify and categorize available knowledge sources',
          'Analyze knowledge quality and reliability',
          'Design integration strategy for disparate knowledge types',
          'Plan reasoning workflow for knowledge synthesis',
          'Validate integrated knowledge for consistency',
          'Generate actionable insights from processed knowledge'
        ],
        expectedOutputs: ['knowledge_map', 'integration_strategy', 'reasoning_workflow']
      }
    };
  }

  /**
   * Initialize decision trees for systematic routing and coordination
   * These trees guide the system selection and coordination logic
   * 
   * @returns {Object} Collection of decision trees
   */
  initializeDecisionTrees() {
    logger.logOperation('debug', 'DECISION_TREE_INIT', 'Initializing decision trees for system routing');

    return {
      /**
       * System selection decision tree
       */
      systemSelection: {
        name: 'System Selection Decision Tree',
        description: 'Determine which systems (Alita, KGoT, or both) to use for a task',
        
        /**
         * Evaluate task requirements and return system selection
         * @param {Object} taskContext - Context about the task
         * @returns {Object} System selection decision
         */
        evaluate: (taskContext) => {
          const {
            requiresMCPCreation,
            requiresKnowledgeGraph,
            requiresWebInteraction,
            requiresCodeGeneration,
            requiresComplexReasoning,
            dataComplexity,
            interactionType
          } = taskContext;

          // Decision logic based on task characteristics
          if (requiresKnowledgeGraph && requiresComplexReasoning) {
            return {
              primary: 'KGoT',
              secondary: requiresMCPCreation ? 'Alita' : null,
              coordination: 'sequential',
              reasoning: 'KGoT handles complex reasoning, Alita provides tool creation if needed'
            };
          }

          if (requiresMCPCreation && (requiresWebInteraction || requiresCodeGeneration)) {
            return {
              primary: 'Alita',
              secondary: requiresKnowledgeGraph ? 'KGoT' : null,
              coordination: 'parallel',
              reasoning: 'Alita handles tool creation and web interaction, KGoT for knowledge processing'
            };
          }

          if (dataComplexity === 'high' || interactionType === 'multi_domain') {
            return {
              primary: 'both',
              secondary: null,
              coordination: 'integrated',
              reasoning: 'Complex multi-domain task requires integrated approach'
            };
          }

          // Default to Alita for most tasks
          return {
            primary: 'Alita',
            secondary: null,
            coordination: 'single',
            reasoning: 'Standard task handled by Alita system'
          };
        }
      },

      /**
       * Coordination strategy decision tree
       */
      coordinationStrategy: {
        name: 'Coordination Strategy Decision Tree',
        description: 'Determine how to coordinate between systems',
        
        /**
         * Evaluate coordination requirements
         * @param {Object} coordinationContext - Context about coordination needs
         * @returns {Object} Coordination strategy
         */
        evaluate: (coordinationContext) => {
          const {
            systemSelection,
            taskComplexity,
            timeConstraints,
            resourceAvailability,
            errorTolerance
          } = coordinationContext;

          if (systemSelection.primary === 'both') {
            return {
              strategy: 'integrated_workflow',
              execution: 'parallel_with_synchronization',
              monitoring: 'real_time',
              fallback: 'graceful_degradation'
            };
          }

          if (systemSelection.secondary && taskComplexity > 8) {
            return {
              strategy: 'sequential_coordination',
              execution: 'primary_then_secondary',
              monitoring: 'checkpoint_based',
              fallback: 'rollback_on_failure'
            };
          }

          return {
            strategy: 'single_system',
            execution: 'direct',
            monitoring: 'standard',
            fallback: 'retry_with_backoff'
          };
        }
      }
    };
  }

  /**
   * Detect task complexity and determine if sequential thinking is needed
   * Uses multiple criteria to assess complexity and trigger appropriate reasoning
   * 
   * @param {Object} taskContext - Information about the current task
   * @returns {Object} Complexity analysis with recommendations
   */
  detectComplexity(taskContext) {
    logger.logOperation('info', 'COMPLEXITY_DETECTION', 'Analyzing task complexity', {
      taskId: taskContext.taskId
    });

    const {
      description,
      requirements = [],
      errors = [],
      systemsInvolved = [],
      dataTypes = [],
      interactions = [],
      timeline,
      dependencies = []
    } = taskContext;

    /**
     * Calculate complexity score based on multiple factors
     */
    let complexityScore = 0;
    const complexityFactors = [];

    // Factor 1: Multiple system involvement
    if (systemsInvolved.length > 1) {
      complexityScore += 3;
      complexityFactors.push(`Multi-system coordination (${systemsInvolved.length} systems)`);
    }

    // Factor 2: Error count threshold
    if (errors.length > this.config.errorThreshold) {
      complexityScore += 4;
      complexityFactors.push(`High error count (${errors.length} errors)`);
    }

    // Factor 3: Data type diversity
    if (dataTypes.length > 3) {
      complexityScore += 2;
      complexityFactors.push(`Multiple data types (${dataTypes.length} types)`);
    }

    // Factor 4: Complex interactions
    const complexInteractions = interactions.filter(i => 
      i.type === 'bidirectional' || i.complexity === 'high'
    );
    if (complexInteractions.length > 0) {
      complexityScore += 3;
      complexityFactors.push(`Complex interactions (${complexInteractions.length})`);
    }

    // Factor 5: Requirement complexity
    const complexRequirements = requirements.filter(r => 
      r.priority === 'critical' || r.complexity === 'high'
    );
    if (complexRequirements.length > 2) {
      complexityScore += 2;
      complexityFactors.push(`Complex requirements (${complexRequirements.length})`);
    }

    // Factor 6: Dependency complexity
    if (dependencies.length > 5) {
      complexityScore += 2;
      complexityFactors.push(`High dependency count (${dependencies.length})`);
    }

    // Factor 7: Time constraints
    if (timeline && timeline.urgency === 'high') {
      complexityScore += 1;
      complexityFactors.push('High urgency timeline');
    }

    /**
     * Determine trigger conditions
     */
    const triggerConditions = {
      complexityThreshold: complexityScore > this.config.complexityThreshold,
      errorThreshold: errors.length > this.config.errorThreshold,
      crossSystemRequired: systemsInvolved.length > 1,
      multiDomainTask: dataTypes.length > 3,
      cascadingErrors: errors.some(e => e.type === 'cascading'),
      integrationRequired: interactions.some(i => i.type === 'integration')
    };

    const shouldTrigger = Object.values(triggerConditions).some(condition => condition);

    /**
     * Generate complexity analysis result
     */
    const analysisResult = {
      taskId: taskContext.taskId,
      complexityScore,
      complexityFactors,
      triggerConditions,
      shouldTriggerSequentialThinking: shouldTrigger,
      recommendedTemplate: this.selectThoughtTemplate(taskContext, triggerConditions),
      systemRecommendation: this.decisionTrees.systemSelection.evaluate(taskContext)
    };

    // Cache complexity score for future reference
    this.complexityScores.set(taskContext.taskId, analysisResult);

    logger.logOperation('info', 'COMPLEXITY_ANALYSIS_COMPLETE', 'Task complexity analysis completed', {
      taskId: taskContext.taskId,
      complexityScore,
      shouldTrigger,
      recommendedTemplate: analysisResult.recommendedTemplate?.name
    });

    return analysisResult;
  }

  /**
   * Select appropriate thought process template based on task characteristics
   * 
   * @param {Object} taskContext - Task context information
   * @param {Object} triggerConditions - Detected trigger conditions
   * @returns {Object|null} Selected thought template or null if none match
   */
  selectThoughtTemplate(taskContext, triggerConditions) {
    logger.logOperation('debug', 'TEMPLATE_SELECTION', 'Selecting thought process template', {
      taskId: taskContext.taskId,
      triggerConditions
    });

    // Prioritize templates based on trigger conditions
    if (triggerConditions.cascadingErrors || triggerConditions.errorThreshold) {
      return this.thoughtTemplates.errorResolution;
    }

    if (triggerConditions.crossSystemRequired || triggerConditions.integrationRequired) {
      return this.thoughtTemplates.systemCoordination;
    }

    if (triggerConditions.complexityThreshold && triggerConditions.multiDomainTask) {
      return this.thoughtTemplates.knowledgeProcessing;
    }

    if (triggerConditions.complexityThreshold) {
      return this.thoughtTemplates.taskDecomposition;
    }

    return null;
  }

  /**
   * Execute sequential thinking process for complex tasks
   * Coordinates with the Sequential Thinking MCP tool for step-by-step reasoning
   * 
   * @param {Object} taskContext - Complete task context
   * @param {Object} template - Selected thought process template
   * @returns {Promise<Object>} Sequential thinking results
   */
  async executeSequentialThinking(taskContext, template) {
    const sessionId = `thinking_${taskContext.taskId}_${Date.now()}`;
    
    logger.logOperation('info', 'SEQUENTIAL_THINKING_START', 'Starting sequential thinking process', {
      sessionId,
      taskId: taskContext.taskId,
      template: template.name
    });

    try {
      // Initialize thinking session
      const thinkingSession = {
        sessionId,
        taskId: taskContext.taskId,
        template,
        startTime: Date.now(),
        currentThought: 0,
        totalThoughts: template.thoughtSteps.length,
        thoughts: [],
        conclusions: {},
        systemRecommendations: {}
      };

      this.activeThinkingSessions.set(sessionId, thinkingSession);

      /**
       * Execute each thought step systematically
       */
      for (let i = 0; i < template.thoughtSteps.length; i++) {
        const thoughtStep = template.thoughtSteps[i];
        
        logger.logOperation('debug', 'THOUGHT_STEP_EXECUTION', `Executing thought step ${i + 1}`, {
          sessionId,
          step: thoughtStep
        });

        // Create thought context for this step
        const thoughtContext = {
          sessionId,
          thoughtNumber: i + 1,
          totalThoughts: template.thoughtSteps.length,
          currentStep: thoughtStep,
          taskContext,
          previousThoughts: thinkingSession.thoughts,
          template
        };

        // Execute individual thought step
        const thoughtResult = await this.executeThoughtStep(thoughtContext);
        
        // Store thought result
        thinkingSession.thoughts.push(thoughtResult);
        thinkingSession.currentThought = i + 1;

        // Check if we need to adjust the thinking process
        if (thoughtResult.needsMoreThoughts) {
          template.thoughtSteps.push(...thoughtResult.additionalSteps);
          thinkingSession.totalThoughts = template.thoughtSteps.length;
        }

        // Emit progress update
        this.emit('thinkingProgress', {
          sessionId,
          progress: (i + 1) / template.thoughtSteps.length,
          currentThought: thoughtResult
        });
      }

      /**
       * Synthesize final conclusions and recommendations
       */
      const finalConclusions = await this.synthesizeConclusions(thinkingSession);
      thinkingSession.conclusions = finalConclusions;

      // Generate system routing recommendations
      const routingRecommendations = this.generateRoutingRecommendations(
        taskContext, 
        thinkingSession
      );
      thinkingSession.systemRecommendations = routingRecommendations;

      // Mark session as complete
      thinkingSession.endTime = Date.now();
      thinkingSession.duration = thinkingSession.endTime - thinkingSession.startTime;

      logger.logOperation('info', 'SEQUENTIAL_THINKING_COMPLETE', 'Sequential thinking process completed', {
        sessionId,
        duration: thinkingSession.duration,
        thoughtCount: thinkingSession.thoughts.length,
        conclusions: Object.keys(finalConclusions)
      });

      // Emit completion event
      this.emit('thinkingComplete', thinkingSession);

      return thinkingSession;

    } catch (error) {
      logger.logError('SEQUENTIAL_THINKING_ERROR', error, {
        sessionId,
        taskId: taskContext.taskId
      });

      // Clean up failed session
      this.activeThinkingSessions.delete(sessionId);
      
      throw error;
    }
  }

  /**
   * Execute individual thought step using Sequential Thinking MCP
   * This method interfaces with the Sequential Thinking MCP tool
   * 
   * @param {Object} thoughtContext - Context for the current thought step
   * @returns {Promise<Object>} Thought step result
   */
  async executeThoughtStep(thoughtContext) {
    const {
      sessionId,
      thoughtNumber,
      totalThoughts,
      currentStep,
      taskContext,
      previousThoughts
    } = thoughtContext;

    logger.logOperation('debug', 'THOUGHT_STEP_START', 'Executing individual thought step', {
      sessionId,
      thoughtNumber,
      step: currentStep
    });

    try {
      // Prepare thought input for Sequential Thinking MCP
      const thoughtInput = {
        thought: this.formulateThought(currentStep, taskContext, previousThoughts),
        nextThoughtNeeded: thoughtNumber < totalThoughts,
        thoughtNumber,
        totalThoughts,
        isRevision: false,
        needsMoreThoughts: false
      };

      // Note: In a real implementation, this would call the Sequential Thinking MCP tool
      // For now, we'll simulate the structured thinking process
      const thoughtResult = await this.simulateSequentialThinking(thoughtInput, thoughtContext);

      logger.logOperation('debug', 'THOUGHT_STEP_COMPLETE', 'Thought step completed', {
        sessionId,
        thoughtNumber,
        hasConclusion: !!thoughtResult.conclusion
      });

      return thoughtResult;

    } catch (error) {
      logger.logError('THOUGHT_STEP_ERROR', error, {
        sessionId,
        thoughtNumber,
        step: currentStep
      });
      throw error;
    }
  }

  /**
   * Formulate thought content for Sequential Thinking MCP
   * Converts task context and step into structured thought
   * 
   * @param {string} step - Current thought step
   * @param {Object} taskContext - Task context
   * @param {Array} previousThoughts - Previous thought results
   * @returns {string} Formulated thought content
   */
  formulateThought(step, taskContext, previousThoughts) {
    const previousInsights = previousThoughts
      .map(t => t.conclusion)
      .filter(c => c)
      .join(' ');

    return `Analyzing: ${step}

Task Context:
- Description: ${taskContext.description}
- Systems Involved: ${taskContext.systemsInvolved?.join(', ') || 'None specified'}
- Error Count: ${taskContext.errors?.length || 0}
- Requirements: ${taskContext.requirements?.length || 0} requirements

Previous Insights: ${previousInsights || 'None yet'}

Current Analysis: ${step}`;
  }

  /**
   * Simulate Sequential Thinking MCP behavior
   * This would be replaced with actual MCP tool call in production
   * 
   * @param {Object} thoughtInput - Input for thinking process
   * @param {Object} thoughtContext - Context for the thought
   * @returns {Promise<Object>} Simulated thought result
   */
  async simulateSequentialThinking(thoughtInput, thoughtContext) {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 100));

    const { currentStep, taskContext } = thoughtContext;
    
    // Generate contextual conclusion based on the step
    let conclusion = '';
    let recommendations = [];

    if (currentStep.includes('complexity')) {
      conclusion = `Task complexity analysis: Score ${taskContext.complexityScore || 'TBD'}/10. ` +
                  `${taskContext.systemsInvolved?.length > 1 ? 'Multi-system coordination required.' : 'Single system sufficient.'}`;
      recommendations.push('Consider phased approach for high complexity');
    } else if (currentStep.includes('error')) {
      conclusion = `Error analysis: ${taskContext.errors?.length || 0} errors detected. ` +
                  `${taskContext.errors?.some(e => e.type === 'cascading') ? 'Cascading errors present.' : 'Independent errors.'}`;
      recommendations.push('Prioritize root cause resolution');
    } else if (currentStep.includes('system')) {
      conclusion = `System coordination: ${taskContext.systemsInvolved?.join(' + ') || 'Single system'}. ` +
                  `Integration complexity: ${taskContext.systemsInvolved?.length > 1 ? 'High' : 'Low'}`;
      recommendations.push('Implement proper inter-system communication');
    } else {
      conclusion = `Step analysis: ${currentStep}. Context evaluated successfully.`;
      recommendations.push('Proceed with standard approach');
    }

    return {
      thought: thoughtInput.thought,
      conclusion,
      recommendations,
      confidence: 0.8,
      nextThoughtNeeded: thoughtInput.nextThoughtNeeded,
      needsMoreThoughts: false,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Synthesize conclusions from all thought steps
   * Combines insights from individual steps into actionable conclusions
   * 
   * @param {Object} thinkingSession - Complete thinking session
   * @returns {Promise<Object>} Synthesized conclusions
   */
  async synthesizeConclusions(thinkingSession) {
    logger.logOperation('info', 'CONCLUSION_SYNTHESIS', 'Synthesizing conclusions from thought process', {
      sessionId: thinkingSession.sessionId,
      thoughtCount: thinkingSession.thoughts.length
    });

    const conclusions = thinkingSession.thoughts.map(t => t.conclusion).filter(c => c);
    const allRecommendations = thinkingSession.thoughts
      .flatMap(t => t.recommendations || [])
      .filter(r => r);

    // Synthesize key insights
    const keyInsights = {
      complexityAssessment: conclusions.find(c => c.includes('complexity')) || 'Standard complexity',
      errorAnalysis: conclusions.find(c => c.includes('error')) || 'No significant errors',
      systemCoordination: conclusions.find(c => c.includes('system')) || 'Single system operation',
      recommendedActions: [...new Set(allRecommendations)] // Remove duplicates
    };

    // Determine overall approach
    const overallApproach = this.determineOverallApproach(keyInsights, thinkingSession);

    return {
      keyInsights,
      overallApproach,
      actionPlan: this.generateActionPlan(keyInsights, thinkingSession),
      riskAssessment: this.assessRisks(keyInsights, thinkingSession),
      successCriteria: this.defineSuccessCriteria(keyInsights, thinkingSession)
    };
  }

  /**
   * Determine overall approach based on synthesized insights
   * 
   * @param {Object} keyInsights - Key insights from thinking process
   * @param {Object} thinkingSession - Complete thinking session
   * @returns {Object} Overall approach recommendation
   */
  determineOverallApproach(keyInsights, thinkingSession) {
    const { taskContext } = thinkingSession;
    
    if (keyInsights.systemCoordination.includes('Multi-system')) {
      return {
        strategy: 'coordinated_multi_system',
        primary: 'coordination',
        description: 'Coordinate between Alita and KGoT systems with proper synchronization'
      };
    }

    if (keyInsights.complexityAssessment.includes('High') || 
        keyInsights.errorAnalysis.includes('Cascading')) {
      return {
        strategy: 'phased_execution',
        primary: 'risk_mitigation',
        description: 'Execute in phases with careful error handling and validation'
      };
    }

    return {
      strategy: 'direct_execution',
      primary: 'efficiency',
      description: 'Direct execution with standard monitoring and error handling'
    };
  }

  /**
   * Generate actionable plan based on conclusions
   * 
   * @param {Object} keyInsights - Key insights from thinking process
   * @param {Object} thinkingSession - Complete thinking session
   * @returns {Array} Action plan steps
   */
  generateActionPlan(keyInsights, thinkingSession) {
    const actionPlan = [];

    // Add actions based on insights
    if (keyInsights.systemCoordination.includes('Multi-system')) {
      actionPlan.push({
        step: 1,
        action: 'Initialize system coordination',
        description: 'Set up communication between Alita and KGoT systems',
        system: 'both',
        priority: 'high'
      });
    }

    if (keyInsights.errorAnalysis.includes('error')) {
      actionPlan.push({
        step: actionPlan.length + 1,
        action: 'Resolve detected errors',
        description: 'Address errors before proceeding with main task',
        system: 'diagnostics',
        priority: 'critical'
      });
    }

    // Add main execution step
    actionPlan.push({
      step: actionPlan.length + 1,
      action: 'Execute main task',
      description: 'Perform the primary task operation',
      system: 'primary',
      priority: 'high'
    });

    return actionPlan;
  }

  /**
   * Assess risks based on thinking process results
   * 
   * @param {Object} keyInsights - Key insights
   * @param {Object} thinkingSession - Thinking session
   * @returns {Object} Risk assessment
   */
  assessRisks(keyInsights, thinkingSession) {
    const risks = [];

    if (keyInsights.systemCoordination.includes('Multi-system')) {
      risks.push({
        type: 'coordination_failure',
        severity: 'medium',
        description: 'Systems may fail to coordinate properly',
        mitigation: 'Implement robust error handling and fallback mechanisms'
      });
    }

    if (keyInsights.errorAnalysis.includes('Cascading')) {
      risks.push({
        type: 'error_propagation',
        severity: 'high',
        description: 'Errors may cascade across systems',
        mitigation: 'Isolate systems and implement circuit breakers'
      });
    }

    return {
      overallRisk: risks.length > 0 ? 'medium' : 'low',
      identifiedRisks: risks,
      mitigationPlan: risks.map(r => r.mitigation)
    };
  }

  /**
   * Define success criteria for the task
   * 
   * @param {Object} keyInsights - Key insights
   * @param {Object} thinkingSession - Thinking session
   * @returns {Array} Success criteria
   */
  defineSuccessCriteria(keyInsights, thinkingSession) {
    return [
      'Task completed without critical errors',
      'All system components functioning properly',
      'Expected outputs generated successfully',
      'Performance within acceptable parameters',
      'Proper coordination between systems (if applicable)'
    ];
  }

  /**
   * Generate routing recommendations for system coordination
   * Determines how to route tasks between Alita and KGoT systems
   * 
   * @param {Object} taskContext - Original task context
   * @param {Object} thinkingSession - Completed thinking session
   * @returns {Object} Routing recommendations
   */
  generateRoutingRecommendations(taskContext, thinkingSession) {
    logger.logOperation('info', 'ROUTING_RECOMMENDATIONS', 'Generating system routing recommendations', {
      sessionId: thinkingSession.sessionId,
      taskId: taskContext.taskId
    });

    const { conclusions } = thinkingSession;
    const systemSelection = this.decisionTrees.systemSelection.evaluate(taskContext);
    
    // Determine coordination strategy
    const coordinationContext = {
      systemSelection,
      taskComplexity: taskContext.complexityScore || 5,
      timeConstraints: taskContext.timeline?.urgency || 'normal',
      resourceAvailability: 'normal',
      errorTolerance: taskContext.errors?.length > 0 ? 'low' : 'medium'
    };

    const coordinationStrategy = this.decisionTrees.coordinationStrategy.evaluate(coordinationContext);

    // Generate specific routing plan
    const routingPlan = {
      systemSelection,
      coordinationStrategy,
      executionSequence: this.generateExecutionSequence(systemSelection, coordinationStrategy),
      monitoringPlan: this.generateMonitoringPlan(systemSelection, coordinationStrategy),
      fallbackStrategy: this.generateFallbackStrategy(systemSelection, coordinationStrategy)
    };

    logger.logOperation('info', 'ROUTING_PLAN_GENERATED', 'System routing plan generated', {
      sessionId: thinkingSession.sessionId,
      primarySystem: systemSelection.primary,
      coordinationStrategy: coordinationStrategy.strategy
    });

    return routingPlan;
  }

  /**
   * Generate execution sequence for system coordination
   * 
   * @param {Object} systemSelection - Selected systems
   * @param {Object} coordinationStrategy - Coordination strategy
   * @returns {Array} Execution sequence steps
   */
  generateExecutionSequence(systemSelection, coordinationStrategy) {
    const sequence = [];

    if (systemSelection.primary === 'both') {
      sequence.push({
        step: 1,
        system: 'coordinator',
        action: 'initialize_both_systems',
        description: 'Initialize both Alita and KGoT systems'
      });
      sequence.push({
        step: 2,
        system: 'both',
        action: 'parallel_execution',
        description: 'Execute tasks in parallel with synchronization'
      });
    } else {
      sequence.push({
        step: 1,
        system: systemSelection.primary,
        action: 'primary_execution',
        description: `Execute primary task using ${systemSelection.primary} system`
      });
      
      if (systemSelection.secondary) {
        sequence.push({
          step: 2,
          system: systemSelection.secondary,
          action: 'secondary_execution',
          description: `Execute secondary task using ${systemSelection.secondary} system`
        });
      }
    }

    return sequence;
  }

  /**
   * Generate monitoring plan for system coordination
   * 
   * @param {Object} systemSelection - Selected systems
   * @param {Object} coordinationStrategy - Coordination strategy
   * @returns {Object} Monitoring plan
   */
  generateMonitoringPlan(systemSelection, coordinationStrategy) {
    return {
      monitoringType: coordinationStrategy.monitoring,
      checkpoints: systemSelection.primary === 'both' ? 
        ['initialization', 'synchronization', 'completion'] :
        ['start', 'progress', 'completion'],
      alerting: {
        errorThreshold: 3,
        performanceThreshold: '90th_percentile',
        coordinationFailureDetection: true
      }
    };
  }

  /**
   * Generate fallback strategy for system failures
   * 
   * @param {Object} systemSelection - Selected systems
   * @param {Object} coordinationStrategy - Coordination strategy
   * @returns {Object} Fallback strategy
   */
  generateFallbackStrategy(systemSelection, coordinationStrategy) {
    return {
      strategy: coordinationStrategy.fallback,
      triggers: ['system_failure', 'coordination_timeout', 'error_threshold_exceeded'],
      actions: {
        system_failure: systemSelection.secondary ? 
          'switch_to_secondary_system' : 'retry_with_backoff',
        coordination_timeout: 'fallback_to_sequential_execution',
        error_threshold_exceeded: 'abort_and_report'
      }
    };
  }

  /**
   * Create LangChain tool for Sequential Thinking integration
   * This tool can be used by the Manager Agent to trigger sequential thinking
   * 
   * @returns {DynamicTool} LangChain tool for sequential thinking
   */
  createSequentialThinkingTool() {
    logger.logOperation('info', 'TOOL_CREATION', 'Creating Sequential Thinking LangChain tool');

    return new DynamicTool({
      name: 'sequential_thinking',
      description: `Engage Sequential Thinking MCP for complex task analysis and system coordination.
        
        Use this tool when:
        - Task complexity score > 7
        - Multiple errors detected (> 3)
        - Cross-system coordination required
        - Complex reasoning needed
        - Multi-domain task processing
        
        Input should be a JSON object with:
        - taskId: Unique identifier for the task
        - description: Task description
        - requirements: Array of requirements
        - errors: Array of detected errors
        - systemsInvolved: Array of systems needed
        - dataTypes: Array of data types involved
        - interactions: Array of system interactions
        - timeline: Timeline information
        - dependencies: Array of task dependencies`,
        
      func: async (input) => {
        try {
          const taskContext = JSON.parse(input);
          
          // Validate input
          if (!taskContext.taskId || !taskContext.description) {
            throw new Error('Task ID and description are required');
          }

          logger.logOperation('info', 'SEQUENTIAL_THINKING_TOOL_INVOKED', 'Sequential thinking tool invoked', {
            taskId: taskContext.taskId
          });

          // Detect complexity and determine if sequential thinking is needed
          const complexityAnalysis = this.detectComplexity(taskContext);

          if (!complexityAnalysis.shouldTriggerSequentialThinking) {
            return JSON.stringify({
              status: 'not_required',
              message: 'Task complexity does not require sequential thinking',
              complexityScore: complexityAnalysis.complexityScore,
              recommendation: 'Proceed with standard processing'
            });
          }

          // Execute sequential thinking process
          const thinkingResult = await this.executeSequentialThinking(
            taskContext, 
            complexityAnalysis.recommendedTemplate
          );

          // Return structured result
          return JSON.stringify({
            status: 'completed',
            sessionId: thinkingResult.sessionId,
            complexityScore: complexityAnalysis.complexityScore,
            template: complexityAnalysis.recommendedTemplate.name,
            conclusions: thinkingResult.conclusions,
            routingRecommendations: thinkingResult.systemRecommendations,
            duration: thinkingResult.duration,
            thoughtCount: thinkingResult.thoughts.length
          });

        } catch (error) {
          logger.logError('SEQUENTIAL_THINKING_TOOL_ERROR', error);
          return JSON.stringify({
            status: 'error',
            error: error.message,
            recommendation: 'Fallback to standard processing'
          });
        }
      }
    });
  }

  /**
   * Get active thinking sessions
   * 
   * @returns {Array} Array of active thinking sessions
   */
  getActiveSessions() {
    return Array.from(this.activeThinkingSessions.values());
  }

  /**
   * Get complexity analysis for a task
   * 
   * @param {string} taskId - Task identifier
   * @returns {Object|null} Complexity analysis or null if not found
   */
  getComplexityAnalysis(taskId) {
    return this.complexityScores.get(taskId) || null;
  }

  /**
   * Clean up completed thinking sessions
   * Removes sessions older than specified time
   * 
   * @param {number} maxAge - Maximum age in milliseconds (default: 1 hour)
   */
  cleanupSessions(maxAge = 3600000) {
    const now = Date.now();
    const sessionsToRemove = [];

    for (const [sessionId, session] of this.activeThinkingSessions.entries()) {
      if (session.endTime && (now - session.endTime) > maxAge) {
        sessionsToRemove.push(sessionId);
      }
    }

    sessionsToRemove.forEach(sessionId => {
      this.activeThinkingSessions.delete(sessionId);
    });

    if (sessionsToRemove.length > 0) {
      logger.logOperation('info', 'SESSION_CLEANUP', `Cleaned up ${sessionsToRemove.length} old thinking sessions`);
    }
  }

  /**
   * Shutdown sequential thinking integration
   * Clean up resources and close connections
   */
  async shutdown() {
    logger.logOperation('info', 'SEQUENTIAL_THINKING_SHUTDOWN', 'Shutting down Sequential Thinking integration');
    
    // Clean up all active sessions
    this.activeThinkingSessions.clear();
    this.complexityScores.clear();
    this.systemCoordinationHistory = [];

    // Remove all event listeners
    this.removeAllListeners();

    logger.logOperation('info', 'SEQUENTIAL_THINKING_SHUTDOWN_COMPLETE', 'Sequential Thinking integration shutdown complete');
  }
}

module.exports = {
  SequentialThinkingIntegration
}; 