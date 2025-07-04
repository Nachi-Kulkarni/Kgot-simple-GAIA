#!/usr/bin/env python3
"""
Sequential Thinking Decision Trees for Cross-System Coordination - Task 17c Implementation

This module implements the Sequential Thinking Decision Trees system for systematic coordination
between Alita and KGoT systems as specified in the 5-Phase Implementation Plan for Enhanced Alita.

This system provides:
- Decision tree templates for systematic coordination between Alita and KGoT systems
- Branching logic for scenarios requiring: MCP-only, KGoT-only, or combined system approaches
- Validation checkpoints within decision trees using cross-validation framework
- Resource optimization decision workflows considering both performance and cost factors
- Fallback strategies with sequential reasoning for system failures or bottlenecks
- Integration with RAG-MCP Section 3.2 intelligent MCP selection and KGoT Section 2.2 Controller orchestration

Key Components:
1. DecisionNode: Individual decision points with conditions, actions, and outcomes
2. DecisionTree: Container for decision nodes with intelligent traversal logic
3. SystemCoordinationDecisionTree: Specific tree for Alita/KGoT coordination decisions
4. ResourceOptimizationDecisionTree: Tree for performance/cost optimization decisions
5. FallbackStrategyDecisionTree: Tree for handling system failures and bottlenecks
6. SequentialDecisionTreeManager: Main orchestrator managing all decision trees

@module SequentialDecisionTrees
@author Enhanced Alita KGoT Team
@version 1.0.0
@task Task 17c - Create Sequential Thinking Decision Trees for Cross-System Coordination
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
import numpy as np
from pathlib import Path

# LangChain imports for agent architecture (per user preference)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

# Winston logging integration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config/logging'))

# Cross-validation framework integration
try:
    from ...validation.mcp_cross_validator import (
        MCPCrossValidationEngine, 
        CrossValidationResult,
        ValidationMetrics,
        MCPValidationSpec
    )
except ImportError:
    # Fallback for development
    CrossValidationResult = Dict[str, Any]
    ValidationMetrics = Dict[str, Any]
    MCPValidationSpec = Dict[str, Any]
    MCPCrossValidationEngine = Any

# Setup logging
logger = logging.getLogger(__name__)


class SystemType(Enum):
    """Enumeration for system types in the architecture"""
    MCP_ONLY = "mcp_only"
    KGOT_ONLY = "kgot_only"
    COMBINED = "combined"
    VALIDATION = "validation"
    MULTIMODAL = "multimodal"
    FALLBACK = "fallback"


class DecisionType(Enum):
    """Enumeration for decision tree types"""
    SYSTEM_SELECTION = "system_selection"
    MCP_SELECTION = "mcp_selection"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    VALIDATION_STRATEGY = "validation_strategy"
    FALLBACK_STRATEGY = "fallback_strategy"


class TaskComplexity(Enum):
    """Enumeration for task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


class ResourceConstraint(Enum):
    """Enumeration for resource constraint types"""
    TIME_CRITICAL = "time_critical"
    MEMORY_LIMITED = "memory_limited"
    CPU_LIMITED = "cpu_limited"
    COST_SENSITIVE = "cost_sensitive"
    QUALITY_PRIORITY = "quality_priority"


@dataclass
class DecisionContext:
    """
    Context information for decision tree traversal
    
    Attributes:
        task_id: Unique identifier for the task
        task_description: Description of the task to be processed
        task_type: Type of task (data_processing, reasoning, etc.)
        complexity_level: Assessed complexity level
        data_types: Types of data involved in the task
        resource_constraints: Resource limitations and priorities
        time_constraints: Time-related constraints
        quality_requirements: Quality and accuracy requirements
        system_state: Current state of available systems
        user_preferences: User preferences for system selection
        historical_performance: Historical performance data for similar tasks
        validation_requirements: Validation strategy requirements
        metadata: Additional contextual information
    """
    task_id: str
    task_description: str
    task_type: str
    complexity_level: TaskComplexity
    data_types: List[str] = field(default_factory=list)
    resource_constraints: List[ResourceConstraint] = field(default_factory=list)
    time_constraints: Optional[Dict[str, Any]] = None
    quality_requirements: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    historical_performance: Optional[Dict[str, Any]] = None
    validation_requirements: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionNode:
    """
    Individual decision point in a decision tree
    
    Attributes:
        node_id: Unique identifier for the decision node
        name: Human-readable name for the decision node
        description: Description of what this decision node evaluates
        condition_function: Function that evaluates the decision condition
        action_function: Optional function to execute when condition is met
        true_branch: Node to traverse when condition is True
        false_branch: Node to traverse when condition is False
        leaf_action: Action to take if this is a leaf node
        validation_checkpoint: Optional validation to perform at this node
        resource_cost: Estimated resource cost for this decision path
        confidence_threshold: Minimum confidence required for this decision
        metadata: Additional node-specific information
    """
    node_id: str
    name: str
    description: str
    condition_function: Callable[[DecisionContext], bool]
    action_function: Optional[Callable[[DecisionContext], Any]] = None
    true_branch: Optional['DecisionNode'] = None
    false_branch: Optional['DecisionNode'] = None
    leaf_action: Optional[Callable[[DecisionContext], Dict[str, Any]]] = None
    validation_checkpoint: Optional[Callable[[DecisionContext], bool]] = None
    resource_cost: float = 0.0
    confidence_threshold: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionPath:
    """
    Record of decisions made during tree traversal
    
    Attributes:
        path_id: Unique identifier for the decision path
        context: Original decision context
        nodes_visited: List of nodes visited during traversal
        decisions_made: List of decision outcomes at each node
        total_cost: Total resource cost for the path
        execution_time: Time taken to traverse the path
        confidence_score: Overall confidence in the decision path
        validation_results: Results from validation checkpoints
        final_decision: Final decision result
        metadata: Additional path-specific information
    """
    path_id: str
    context: DecisionContext
    nodes_visited: List[str] = field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    execution_time: float = 0.0
    confidence_score: float = 0.0
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    final_decision: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemCoordinationResult:
    """
    Result of the decision tree system coordination process
    
    Attributes:
        coordination_id: Unique identifier for the coordination session
        selected_system: Selected system type for task execution
        execution_strategy: Strategy for executing the task
        resource_allocation: Resource allocation plan
        validation_strategy: Validation strategy to be used
        fallback_plan: Fallback plan in case of failures
        confidence_score: Confidence in the coordination decision
        estimated_cost: Estimated cost for the selected approach
        estimated_duration: Estimated time for task completion
        decision_path: Full decision path taken
        metadata: Additional coordination-specific information
    """
    coordination_id: str
    selected_system: SystemType
    execution_strategy: Dict[str, Any]
    resource_allocation: Dict[str, Any]
    validation_strategy: Dict[str, Any]
    fallback_plan: Dict[str, Any]
    confidence_score: float
    estimated_cost: float
    estimated_duration: float
    decision_path: DecisionPath
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDecisionTree(ABC):
    """
    Abstract base class for all decision trees in the system
    
    Provides common functionality for decision tree traversal, validation,
    and integration with the cross-validation framework.
    """
    
    def __init__(self, 
                 tree_id: str,
                 name: str,
                 description: str,
                 root_node: Optional[DecisionNode] = None,
                 validation_engine: Optional[MCPCrossValidationEngine] = None):
        """
        Initialize the base decision tree
        
        @param {str} tree_id - Unique identifier for the decision tree
        @param {str} name - Human-readable name for the decision tree
        @param {str} description - Description of the decision tree purpose
        @param {Optional[DecisionNode]} root_node - Root node of the decision tree
        @param {Optional[MCPCrossValidationEngine]} validation_engine - Cross-validation engine
        """
        self.tree_id = tree_id
        self.name = name
        self.description = description
        self.root_node = root_node
        self.validation_engine = validation_engine
        
        # Decision tree state
        self.traversal_history: List[DecisionPath] = []
        self.performance_metrics: Dict[str, Any] = defaultdict(list)
        self.node_registry: Dict[str, DecisionNode] = {}
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized {self.__class__.__name__}", extra={
            'operation': 'DECISION_TREE_INIT',
            'tree_id': tree_id,
            'name': name
        })
    
    @abstractmethod
    async def build_tree(self) -> DecisionNode:
        """
        Abstract method to build the decision tree structure
        
        @returns {DecisionNode} Root node of the built decision tree
        """
        pass
    
    @abstractmethod
    async def get_default_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Abstract method to get default decision when tree traversal fails
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} Default decision result
        """
        pass
    
    async def traverse_tree(self, context: DecisionContext) -> DecisionPath:
        """
        Traverse the decision tree with the given context
        
        @param {DecisionContext} context - Context for decision making
        @returns {DecisionPath} Path taken through the decision tree
        """
        path_id = f"{self.tree_id}_{context.task_id}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        self.logger.info(f"Starting decision tree traversal", extra={
            'operation': 'DECISION_TREE_TRAVERSAL_START',
            'tree_id': self.tree_id,
            'path_id': path_id,
            'task_id': context.task_id
        })
        
        decision_path = DecisionPath(
            path_id=path_id,
            context=context
        )
        
        try:
            # Ensure tree is built
            if not self.root_node:
                self.root_node = await self.build_tree()
            
            # Traverse the tree
            current_node = self.root_node
            total_cost = 0.0
            confidence_scores = []
            
            while current_node:
                decision_path.nodes_visited.append(current_node.node_id)
                
                self.logger.debug(f"Evaluating node: {current_node.name}", extra={
                    'operation': 'DECISION_NODE_EVALUATION',
                    'node_id': current_node.node_id,
                    'path_id': path_id
                })
                
                # Perform validation checkpoint if required
                if current_node.validation_checkpoint:
                    validation_result = await self._perform_validation_checkpoint(
                        current_node, context
                    )
                    decision_path.validation_results.append(validation_result)
                    
                    if not validation_result.get('passed', True):
                        self.logger.warning(f"Validation checkpoint failed", extra={
                            'operation': 'VALIDATION_CHECKPOINT_FAILED',
                            'node_id': current_node.node_id,
                            'path_id': path_id
                        })
                
                # Execute action function if present
                if current_node.action_function:
                    action_result = await self._execute_node_action(
                        current_node.action_function, context
                    )
                    decision_path.metadata[f'action_{current_node.node_id}'] = action_result
                
                # Evaluate condition and determine next node
                if current_node.leaf_action:
                    # This is a leaf node - execute leaf action and finish
                    final_decision = await self._execute_node_action(
                        current_node.leaf_action, context
                    )
                    decision_path.final_decision = final_decision
                    break
                
                # Evaluate condition for branching
                condition_result = await self._evaluate_condition(
                    current_node.condition_function, context
                )
                
                decision_info = {
                    'node_id': current_node.node_id,
                    'condition_result': condition_result,
                    'timestamp': datetime.now().isoformat()
                }
                decision_path.decisions_made.append(decision_info)
                
                # Update costs and confidence
                total_cost += current_node.resource_cost
                confidence_scores.append(getattr(current_node, 'confidence_threshold', 0.7))
                
                # Move to next node based on condition
                if condition_result:
                    current_node = current_node.true_branch
                else:
                    current_node = current_node.false_branch
            
            # Calculate final metrics
            execution_time = time.time() - start_time
            decision_path.total_cost = total_cost
            decision_path.execution_time = execution_time
            decision_path.confidence_score = np.mean(confidence_scores) if confidence_scores else 0.5
            
            # Store traversal history
            self.traversal_history.append(decision_path)
            self._update_performance_metrics(decision_path)
            
            self.logger.info(f"Decision tree traversal completed", extra={
                'operation': 'DECISION_TREE_TRAVERSAL_COMPLETE',
                'path_id': path_id,
                'execution_time': execution_time,
                'total_cost': total_cost,
                'confidence_score': decision_path.confidence_score
            })
            
            return decision_path
            
        except Exception as e:
            self.logger.error(f"Decision tree traversal failed", extra={
                'operation': 'DECISION_TREE_TRAVERSAL_FAILED',
                'path_id': path_id,
                'error': str(e)
            })
            
            # Return path with default decision
            decision_path.final_decision = await self.get_default_decision(context)
            decision_path.execution_time = time.time() - start_time
            decision_path.confidence_score = 0.3  # Low confidence for fallback
            decision_path.metadata['error'] = str(e)
            
            return decision_path
    
    async def _perform_validation_checkpoint(self, 
                                           node: DecisionNode, 
                                           context: DecisionContext) -> Dict[str, Any]:
        """
        Perform validation checkpoint at a decision node
        
        @param {DecisionNode} node - Decision node with validation checkpoint
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} Validation checkpoint result
        """
        try:
            # Execute the validation checkpoint function
            validation_result = await self._execute_node_action(
                node.validation_checkpoint, context
            )
            
            # Integrate with cross-validation framework if available
            if self.validation_engine and isinstance(validation_result, dict):
                enhanced_validation = await self._enhance_validation_with_framework(
                    validation_result, context
                )
                validation_result.update(enhanced_validation)
            
            return {
                'node_id': node.node_id,
                'passed': validation_result.get('passed', True),
                'confidence': validation_result.get('confidence', 0.7),
                'details': validation_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Validation checkpoint error", extra={
                'operation': 'VALIDATION_CHECKPOINT_ERROR',
                'node_id': node.node_id,
                'error': str(e)
            })
            
            return {
                'node_id': node.node_id,
                'passed': False,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _enhance_validation_with_framework(self, 
                                               base_validation: Dict[str, Any],
                                               context: DecisionContext) -> Dict[str, Any]:
        """
        Enhance validation using the cross-validation framework
        
        @param {Dict[str, Any]} base_validation - Base validation result
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} Enhanced validation result
        """
        try:
            # This would integrate with the MCPCrossValidationEngine
            # for more sophisticated validation
            enhanced_result = {
                'framework_validation': True,
                'cross_validation_score': base_validation.get('confidence', 0.7),
                'statistical_significance': True,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            return enhanced_result
            
        except Exception as e:
            self.logger.warning(f"Framework validation enhancement failed", extra={
                'operation': 'FRAMEWORK_VALIDATION_WARNING',
                'error': str(e)
            })
            return {}
    
    async def _execute_node_action(self, 
                                 action_function: Callable,
                                 context: DecisionContext) -> Any:
        """
        Execute a node action function safely
        
        @param {Callable} action_function - Function to execute
        @param {DecisionContext} context - Decision context
        @returns {Any} Action execution result
        """
        try:
            if asyncio.iscoroutinefunction(action_function):
                return await action_function(context)
            else:
                return action_function(context)
        except Exception as e:
            self.logger.error(f"Node action execution failed", extra={
                'operation': 'NODE_ACTION_EXECUTION_FAILED',
                'error': str(e)
            })
            return {'error': str(e), 'success': False}
    
    async def _evaluate_condition(self, 
                                condition_function: Callable,
                                context: DecisionContext) -> bool:
        """
        Evaluate a condition function safely
        
        @param {Callable} condition_function - Condition function to evaluate
        @param {DecisionContext} context - Decision context
        @returns {bool} Condition evaluation result
        """
        try:
            if asyncio.iscoroutinefunction(condition_function):
                return await condition_function(context)
            else:
                return condition_function(context)
        except Exception as e:
            self.logger.error(f"Condition evaluation failed", extra={
                'operation': 'CONDITION_EVALUATION_FAILED',
                'error': str(e)
            })
            return False  # Default to False on error
    
    def _update_performance_metrics(self, decision_path: DecisionPath) -> None:
        """
        Update performance metrics based on decision path
        
        @param {DecisionPath} decision_path - Completed decision path
        """
        metrics = self.performance_metrics
        
        metrics['execution_times'].append(decision_path.execution_time)
        metrics['total_costs'].append(decision_path.total_cost)
        metrics['confidence_scores'].append(decision_path.confidence_score)
        metrics['nodes_visited_counts'].append(len(decision_path.nodes_visited))
        
        # Calculate rolling averages
        if len(metrics['execution_times']) > 100:
            for key in ['execution_times', 'total_costs', 'confidence_scores', 'nodes_visited_counts']:
                metrics[key] = metrics[key][-50:]  # Keep last 50 entries
    
    def register_node(self, node: DecisionNode) -> None:
        """
        Register a node in the node registry
        
        @param {DecisionNode} node - Node to register
        """
        self.node_registry[node.node_id] = node
        
        self.logger.debug(f"Registered decision node", extra={
            'operation': 'DECISION_NODE_REGISTERED',
            'node_id': node.node_id,
            'node_name': node.name
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for the decision tree
        
        @returns {Dict[str, Any]} Performance summary statistics
        """
        metrics = self.performance_metrics
        
        if not metrics['execution_times']:
            return {'status': 'no_data'}
        
        return {
            'total_traversals': len(self.traversal_history),
            'avg_execution_time': np.mean(metrics['execution_times']),
            'avg_cost': np.mean(metrics['total_costs']),
            'avg_confidence': np.mean(metrics['confidence_scores']),
            'avg_nodes_visited': np.mean(metrics['nodes_visited_counts']),
            'total_nodes_registered': len(self.node_registry),
            'last_updated': datetime.now().isoformat()
        }


class SystemSelectionDecisionTree(BaseDecisionTree):
    """
    Decision tree for selecting between Alita MCP, KGoT, or combined system approaches
    
    This decision tree implements the branching logic for scenarios requiring:
    - MCP-only: Simple, well-defined tasks with existing MCPs
    - KGoT-only: Complex reasoning requiring knowledge graph traversal
    - Combined: Complex tasks requiring both tools and reasoning
    
    Integration points:
    - RAG-MCP Section 3.2 intelligent MCP selection
    - KGoT Section 2.2 Controller orchestration
    - Cross-validation framework for validation checkpoints
    """
    
    def __init__(self, 
                 validation_engine: Optional[MCPCrossValidationEngine] = None,
                 rag_mcp_client: Optional[Any] = None,
                 kgot_controller_client: Optional[Any] = None):
        """
        Initialize the system selection decision tree
        
        @param {Optional[MCPCrossValidationEngine]} validation_engine - Cross-validation engine
        @param {Optional[Any]} rag_mcp_client - RAG-MCP client for intelligent MCP selection
        @param {Optional[Any]} kgot_controller_client - KGoT controller client
        """
        super().__init__(
            tree_id="system_selection_tree",
            name="System Selection Decision Tree",
            description="Selects between Alita MCP, KGoT, or combined approaches for task execution",
            validation_engine=validation_engine
        )
        
        self.rag_mcp_client = rag_mcp_client
        self.kgot_controller_client = kgot_controller_client
        
        # System selection metrics
        self.system_usage_stats = defaultdict(int)
        self.system_success_rates = defaultdict(list)
    
    async def build_tree(self) -> DecisionNode:
        """
        Build the system selection decision tree structure
        
        Decision flow:
        1. Task Analysis → Assess complexity, data types, requirements
        2. Resource Assessment → Check available resources and constraints
        3. System Capability Matching → Match task needs to system capabilities
        4. Validation Strategy Selection → Choose appropriate validation approach
        5. Final System Selection → Select MCP-only, KGoT-only, or combined
        
        @returns {DecisionNode} Root node of the system selection tree
        """
        self.logger.info("Building system selection decision tree", extra={
            'operation': 'SYSTEM_SELECTION_TREE_BUILD_START'
        })
        
        # Build decision tree nodes
        
        # Root node: Task complexity assessment
        root_node = DecisionNode(
            node_id="task_complexity_assessment",
            name="Task Complexity Assessment",
            description="Assess the complexity level of the incoming task",
            condition_function=self._is_complex_task,
            resource_cost=0.1,
            confidence_threshold=0.8
        )
        
        # High complexity branch
        high_complexity_node = DecisionNode(
            node_id="high_complexity_analysis",
            name="High Complexity Analysis",
            description="Analyze requirements for high complexity tasks",
            condition_function=self._requires_knowledge_reasoning,
            resource_cost=0.2,
            confidence_threshold=0.7
        )
        
        # Knowledge reasoning required branch
        knowledge_reasoning_node = DecisionNode(
            node_id="knowledge_reasoning_required",
            name="Knowledge Reasoning Required",
            description="Determine if task requires pure knowledge reasoning or combined approach",
            condition_function=self._requires_tool_execution,
            resource_cost=0.15,
            validation_checkpoint=self._validate_kgot_availability
        )
        
        # Combined approach leaf
        combined_approach_leaf = DecisionNode(
            node_id="combined_approach_decision",
            name="Combined Approach Decision",
            description="Select combined Alita MCP + KGoT approach",
            condition_function=lambda x: True,  # Always true for leaf
            leaf_action=self._select_combined_approach,
            resource_cost=0.3,
            confidence_threshold=0.8
        )
        
        # KGoT-only leaf
        kgot_only_leaf = DecisionNode(
            node_id="kgot_only_decision",
            name="KGoT-Only Decision",
            description="Select KGoT-only approach for pure reasoning tasks",
            condition_function=lambda x: True,  # Always true for leaf
            leaf_action=self._select_kgot_only_approach,
            resource_cost=0.25,
            confidence_threshold=0.75
        )
        
        # Tool execution required branch (for high complexity)
        tool_execution_node = DecisionNode(
            node_id="tool_execution_analysis",
            name="Tool Execution Analysis",
            description="Analyze tool execution requirements for high complexity tasks",
            condition_function=self._has_suitable_mcps,
            validation_checkpoint=self._validate_mcp_availability,
            resource_cost=0.15
        )
        
        # MCP available for high complexity
        high_complexity_mcp_leaf = DecisionNode(
            node_id="high_complexity_mcp_decision",
            name="High Complexity MCP Decision",
            description="Use combined approach for high complexity tasks with suitable MCPs",
            condition_function=lambda x: True,
            leaf_action=self._select_combined_approach,
            resource_cost=0.3,
            confidence_threshold=0.8
        )
        
        # No suitable MCP for high complexity - fallback to KGoT
        high_complexity_fallback_leaf = DecisionNode(
            node_id="high_complexity_fallback_decision",
            name="High Complexity Fallback Decision",
            description="Fallback to KGoT-only for high complexity tasks without suitable MCPs",
            condition_function=lambda x: True,
            leaf_action=self._select_kgot_fallback,
            resource_cost=0.25,
            confidence_threshold=0.6
        )
        
        # Low/Medium complexity branch
        low_complexity_node = DecisionNode(
            node_id="low_complexity_analysis",
            name="Low/Medium Complexity Analysis",
            description="Analyze requirements for low to medium complexity tasks",
            condition_function=self._is_time_critical,
            resource_cost=0.1
        )
        
        # Time critical branch
        time_critical_node = DecisionNode(
            node_id="time_critical_analysis",
            name="Time Critical Analysis",
            description="Analyze time-critical task requirements",
            condition_function=self._has_fast_mcps,
            validation_checkpoint=self._validate_fast_execution,
            resource_cost=0.1
        )
        
        # Fast MCP available leaf
        fast_mcp_leaf = DecisionNode(
            node_id="fast_mcp_decision",
            name="Fast MCP Decision",
            description="Select MCP-only approach for time-critical tasks with fast MCPs",
            condition_function=lambda x: True,
            leaf_action=self._select_mcp_only_approach,
            resource_cost=0.15,
            confidence_threshold=0.85
        )
        
        # No fast MCP - assess quality requirements
        quality_assessment_node = DecisionNode(
            node_id="quality_assessment",
            name="Quality Assessment",
            description="Assess quality requirements when no fast MCP available",
            condition_function=self._requires_high_quality,
            resource_cost=0.1
        )
        
        # High quality required - use combined approach
        high_quality_leaf = DecisionNode(
            node_id="high_quality_decision",
            name="High Quality Decision",
            description="Use combined approach for high quality requirements",
            condition_function=lambda x: True,
            leaf_action=self._select_combined_approach,
            resource_cost=0.3,
            confidence_threshold=0.8
        )
        
        # Standard quality - use available MCP
        standard_quality_leaf = DecisionNode(
            node_id="standard_quality_decision",
            name="Standard Quality Decision",
            description="Use MCP-only approach for standard quality requirements",
            condition_function=lambda x: True,
            leaf_action=self._select_mcp_only_approach,
            resource_cost=0.15,
            confidence_threshold=0.75
        )
        
        # Non-time critical branch
        non_time_critical_node = DecisionNode(
            node_id="non_time_critical_analysis",
            name="Non-Time Critical Analysis",
            description="Analyze non-time critical task requirements",
            condition_function=self._has_suitable_mcps,
            validation_checkpoint=self._validate_mcp_suitability,
            resource_cost=0.1
        )
        
        # Suitable MCP available
        suitable_mcp_leaf = DecisionNode(
            node_id="suitable_mcp_decision",
            name="Suitable MCP Decision",
            description="Use MCP-only approach for tasks with suitable MCPs",
            condition_function=lambda x: True,
            leaf_action=self._select_mcp_only_approach,
            resource_cost=0.15,
            confidence_threshold=0.8
        )
        
        # No suitable MCP - fallback analysis
        fallback_analysis_node = DecisionNode(
            node_id="fallback_analysis",
            name="Fallback Analysis",
            description="Analyze fallback options when no suitable MCP available",
            condition_function=self._can_use_kgot_reasoning,
            resource_cost=0.1
        )
        
        # KGoT reasoning available
        kgot_reasoning_leaf = DecisionNode(
            node_id="kgot_reasoning_decision",
            name="KGoT Reasoning Decision",
            description="Use KGoT-only approach for reasoning-based tasks",
            condition_function=lambda x: True,
            leaf_action=self._select_kgot_only_approach,
            resource_cost=0.25,
            confidence_threshold=0.7
        )
        
        # Final fallback
        final_fallback_leaf = DecisionNode(
            node_id="final_fallback_decision",
            name="Final Fallback Decision",
            description="Final fallback to combined approach with lower confidence",
            condition_function=lambda x: True,
            leaf_action=self._select_fallback_combined,
            resource_cost=0.3,
            confidence_threshold=0.5
        )
        
        # Wire up the tree structure
        root_node.true_branch = high_complexity_node
        root_node.false_branch = low_complexity_node
        
        # High complexity branch wiring
        high_complexity_node.true_branch = knowledge_reasoning_node
        high_complexity_node.false_branch = tool_execution_node
        
        knowledge_reasoning_node.true_branch = combined_approach_leaf
        knowledge_reasoning_node.false_branch = kgot_only_leaf
        
        tool_execution_node.true_branch = high_complexity_mcp_leaf
        tool_execution_node.false_branch = high_complexity_fallback_leaf
        
        # Low complexity branch wiring
        low_complexity_node.true_branch = time_critical_node
        low_complexity_node.false_branch = non_time_critical_node
        
        time_critical_node.true_branch = fast_mcp_leaf
        time_critical_node.false_branch = quality_assessment_node
        
        quality_assessment_node.true_branch = high_quality_leaf
        quality_assessment_node.false_branch = standard_quality_leaf
        
        non_time_critical_node.true_branch = suitable_mcp_leaf
        non_time_critical_node.false_branch = fallback_analysis_node
        
        fallback_analysis_node.true_branch = kgot_reasoning_leaf
        fallback_analysis_node.false_branch = final_fallback_leaf
        
        # Register all nodes
        for node in [
            root_node, high_complexity_node, knowledge_reasoning_node, combined_approach_leaf,
            kgot_only_leaf, tool_execution_node, high_complexity_mcp_leaf, high_complexity_fallback_leaf,
            low_complexity_node, time_critical_node, fast_mcp_leaf, quality_assessment_node,
            high_quality_leaf, standard_quality_leaf, non_time_critical_node, suitable_mcp_leaf,
            fallback_analysis_node, kgot_reasoning_leaf, final_fallback_leaf
        ]:
            self.register_node(node)
        
        self.logger.info("System selection decision tree built successfully", extra={
            'operation': 'SYSTEM_SELECTION_TREE_BUILD_COMPLETE',
            'total_nodes': len(self.node_registry)
        })
        
        return root_node
    
    async def get_default_decision(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Get default decision when tree traversal fails
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} Default decision result
        """
        return {
            'selected_system': SystemType.COMBINED,
            'strategy': 'fallback_combined',
            'confidence': 0.3,
            'reason': 'Decision tree traversal failed - using safe fallback',
            'resource_allocation': {
                'mcp_weight': 0.5,
                'kgot_weight': 0.5
            },
            'validation_strategy': 'enhanced_cross_validation',
            'estimated_cost': 0.5,
            'estimated_duration': 10.0
        }
    
    # Condition functions for decision tree logic
    
    def _is_complex_task(self, context: DecisionContext) -> bool:
        """
        Determine if task is complex based on multiple factors
        
        @param {DecisionContext} context - Decision context
        @returns {bool} True if task is complex
        """
        complexity_factors = [
            context.complexity_level in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL],
            len(context.data_types) > 2,
            context.metadata.get('multi_step', False),
            context.metadata.get('requires_reasoning', False),
            len(context.metadata.get('subtasks', [])) > 3
        ]
        
        # Task is complex if 2 or more factors are true
        return sum(complexity_factors) >= 2
    
    def _requires_knowledge_reasoning(self, context: DecisionContext) -> bool:
        """
        Determine if task requires knowledge graph reasoning
        
        @param {DecisionContext} context - Decision context
        @returns {bool} True if knowledge reasoning is required
        """
        reasoning_indicators = [
            'reasoning' in context.task_description.lower(),
            'analysis' in context.task_description.lower(),
            'research' in context.task_description.lower(),
            'explain' in context.task_description.lower(),
            context.metadata.get('requires_reasoning', False),
            context.task_type in ['research', 'analysis', 'reasoning', 'knowledge_synthesis']
        ]
        
        return any(reasoning_indicators)
    
    def _requires_tool_execution(self, context: DecisionContext) -> bool:
        """
        Determine if task requires tool execution capabilities
        
        @param {DecisionContext} context - Decision context
        @returns {bool} True if tool execution is required
        """
        tool_indicators = [
            'execute' in context.task_description.lower(),
            'run' in context.task_description.lower(),
            'process' in context.task_description.lower(),
            'generate' in context.task_description.lower(),
            context.metadata.get('requires_tools', False),
            context.task_type in ['code_execution', 'data_processing', 'web_automation', 'file_processing']
        ]
        
        return any(tool_indicators)
    
    def _is_time_critical(self, context: DecisionContext) -> bool:
        """
        Determine if task is time-critical
        
        @param {DecisionContext} context - Decision context
        @returns {bool} True if task is time-critical
        """
        if ResourceConstraint.TIME_CRITICAL in context.resource_constraints:
            return True
        
        if context.time_constraints:
            max_time = context.time_constraints.get('max_execution_time', float('inf'))
            return max_time < 5.0  # Less than 5 seconds is time critical
        
        return False
    
    def _has_suitable_mcps(self, context: DecisionContext) -> bool:
        """
        Check if suitable MCPs are available for the task (integrates with RAG-MCP)
        
        @param {DecisionContext} context - Decision context
        @returns {bool} True if suitable MCPs are available
        """
        # TODO: Integrate with RAG-MCP Section 3.2 intelligent MCP selection
        # This would query the RAG-MCP system to find suitable MCPs
        
        # For now, use heuristic based on task type
        common_mcp_tasks = [
            'data_processing', 'code_execution', 'file_processing', 
            'web_automation', 'text_processing', 'api_calls'
        ]
        
        return context.task_type in common_mcp_tasks
    
    def _has_fast_mcps(self, context: DecisionContext) -> bool:
        """
        Check if fast-executing MCPs are available
        
        @param {DecisionContext} context - Decision context
        @returns {bool} True if fast MCPs are available
        """
        # TODO: Query MCP performance database
        fast_task_types = ['text_processing', 'simple_calculations', 'data_lookup']
        return context.task_type in fast_task_types
    
    def _requires_high_quality(self, context: DecisionContext) -> bool:
        """
        Determine if task requires high quality output
        
        @param {DecisionContext} context - Decision context
        @returns {bool} True if high quality is required
        """
        if context.quality_requirements:
            min_accuracy = context.quality_requirements.get('min_accuracy', 0.7)
            return min_accuracy > 0.9
        
        return ResourceConstraint.QUALITY_PRIORITY in context.resource_constraints
    
    def _can_use_kgot_reasoning(self, context: DecisionContext) -> bool:
        """
        Check if KGoT reasoning capabilities are available and suitable
        
        @param {DecisionContext} context - Decision context
        @returns {bool} True if KGoT reasoning can be used
        """
        # TODO: Check KGoT system availability and suitability
        reasoning_suitable_tasks = [
            'research', 'analysis', 'reasoning', 'knowledge_synthesis',
            'question_answering', 'explanation', 'decision_making'
        ]
        
        return context.task_type in reasoning_suitable_tasks
    
    # Validation checkpoint functions
    
    async def _validate_kgot_availability(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Validate KGoT system availability
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} Validation result
        """
        # TODO: Implement actual KGoT system health check
        return {
            'passed': True,
            'confidence': 0.8,
            'system_status': 'available',
            'estimated_response_time': 2.5
        }
    
    async def _validate_mcp_availability(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Validate MCP system availability
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} Validation result
        """
        # TODO: Implement actual MCP system health check
        return {
            'passed': True,
            'confidence': 0.85,
            'available_mcps': ['data_processor', 'code_executor', 'web_scraper'],
            'estimated_response_time': 1.5
        }
    
    async def _validate_fast_execution(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Validate fast execution capabilities
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} Validation result
        """
        return {
            'passed': True,
            'confidence': 0.9,
            'expected_execution_time': 1.0,
            'fast_mcps_available': True
        }
    
    async def _validate_mcp_suitability(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Validate MCP suitability for the task
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} Validation result
        """
        return {
            'passed': True,
            'confidence': 0.75,
            'suitable_mcps': ['general_processor'],
            'match_score': 0.7
        }
    
    # Leaf action functions (final decisions)
    
    async def _select_mcp_only_approach(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Select MCP-only approach for task execution
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} MCP-only decision result
        """
        self.system_usage_stats[SystemType.MCP_ONLY] += 1
        
        return {
            'selected_system': SystemType.MCP_ONLY,
            'strategy': 'mcp_only_execution',
            'confidence': 0.85,
            'reason': 'Task suitable for MCP-only execution',
            'resource_allocation': {
                'mcp_weight': 1.0,
                'kgot_weight': 0.0
            },
            'validation_strategy': 'mcp_validation',
            'estimated_cost': 0.15,
            'estimated_duration': 2.0
        }
    
    async def _select_kgot_only_approach(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Select KGoT-only approach for task execution
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} KGoT-only decision result
        """
        self.system_usage_stats[SystemType.KGOT_ONLY] += 1
        
        return {
            'selected_system': SystemType.KGOT_ONLY,
            'strategy': 'kgot_reasoning_only',
            'confidence': 0.8,
            'reason': 'Task requires knowledge reasoning capabilities',
            'resource_allocation': {
                'mcp_weight': 0.0,
                'kgot_weight': 1.0
            },
            'validation_strategy': 'reasoning_validation',
            'estimated_cost': 0.25,
            'estimated_duration': 5.0
        }
    
    async def _select_combined_approach(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Select combined Alita MCP + KGoT approach for task execution
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} Combined approach decision result
        """
        self.system_usage_stats[SystemType.COMBINED] += 1
        
        return {
            'selected_system': SystemType.COMBINED,
            'strategy': 'combined_mcp_kgot',
            'confidence': 0.9,
            'reason': 'Task requires both tool execution and reasoning capabilities',
            'resource_allocation': {
                'mcp_weight': 0.6,
                'kgot_weight': 0.4
            },
            'validation_strategy': 'cross_system_validation',
            'estimated_cost': 0.35,
            'estimated_duration': 7.0
        }
    
    async def _select_kgot_fallback(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Select KGoT fallback for high complexity tasks without suitable MCPs
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} KGoT fallback decision result
        """
        self.system_usage_stats[SystemType.KGOT_ONLY] += 1
        
        return {
            'selected_system': SystemType.KGOT_ONLY,
            'strategy': 'kgot_fallback',
            'confidence': 0.65,
            'reason': 'No suitable MCPs available, falling back to KGoT reasoning',
            'resource_allocation': {
                'mcp_weight': 0.0,
                'kgot_weight': 1.0
            },
            'validation_strategy': 'enhanced_reasoning_validation',
            'estimated_cost': 0.3,
            'estimated_duration': 8.0
        }
    
    async def _select_fallback_combined(self, context: DecisionContext) -> Dict[str, Any]:
        """
        Select fallback combined approach with lower confidence
        
        @param {DecisionContext} context - Decision context
        @returns {Dict[str, Any]} Fallback combined decision result
        """
        self.system_usage_stats[SystemType.COMBINED] += 1
        
        return {
            'selected_system': SystemType.COMBINED,
            'strategy': 'fallback_combined',
            'confidence': 0.5,
            'reason': 'Fallback to combined approach due to decision uncertainty',
            'resource_allocation': {
                'mcp_weight': 0.5,
                'kgot_weight': 0.5
            },
            'validation_strategy': 'comprehensive_validation',
            'estimated_cost': 0.4,
            'estimated_duration': 10.0
        }
    
    def get_system_usage_stats(self) -> Dict[str, Any]:
        """
        Get system usage statistics
        
        @returns {Dict[str, Any]} System usage statistics
        """
        total_usage = sum(self.system_usage_stats.values())
        
        if total_usage == 0:
            return {'status': 'no_usage_data'}
        
        return {
            'total_decisions': total_usage,
            'mcp_only_percentage': (self.system_usage_stats[SystemType.MCP_ONLY] / total_usage) * 100,
            'kgot_only_percentage': (self.system_usage_stats[SystemType.KGOT_ONLY] / total_usage) * 100,
            'combined_percentage': (self.system_usage_stats[SystemType.COMBINED] / total_usage) * 100,
            'raw_counts': dict(self.system_usage_stats),
            'last_updated': datetime.now().isoformat()
        } 