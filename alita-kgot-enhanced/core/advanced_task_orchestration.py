"""Advanced Task Orchestration System (Task 47)

This module implements sophisticated hierarchical task decomposition with DAG-based dependency
management and parallel execution coordination. It extends the Unified System Controller
with advanced orchestration capabilities inspired by Alita's Manager Agent, KGoT's scalability
principles, and RAG-MCP's tool selection optimization.

Key Features:
- DAG-based task decomposition using Sequential Thinking MCP
- Parallel execution coordination with dependency management
- Dynamic task prioritization and re-planning
- Adaptive task scheduling with resource optimization
- Integration with existing Unified System Controller architecture

Architecture:
- AdvancedTaskDecomposer: Creates dependency graphs from high-level tasks
- ParallelCoordinator: Manages concurrent execution with AsyncIO
- DynamicTaskPrioritizer: Handles priority calculation and re-planning
- AdaptiveTaskScheduler: Monitors and schedules tasks dynamically
- AdvancedTaskOrchestrator: Main orchestrator integrating all components

@module AdvancedTaskOrchestration
@author AI Assistant
@date 2025-01-22
"""

import asyncio
import json
import logging
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from uuid import uuid4

import networkx as nx
import redis
import httpx
from pydantic import BaseModel, Field

# Import configuration and logging
from ..config.logging.winston_config import get_logger

# Import existing system components
from .shared_state_utilities import (
    EnhancedSharedStateManager, 
    StateScope, 
    StateEventType
)
from .sequential_thinking_mcp_integration import SequentialThinkingMCPIntegration
from .advanced_monitoring_system import AdvancedMonitoringSystem
from .load_balancing_system import AdaptiveLoadBalancer, SystemInstance
from .unified_system_controller import UnifiedSystemController

# Create logger instance
logger = get_logger('advanced_task_orchestration')


class OrchestrationMode(Enum):
    """Orchestration execution modes"""
    SIMPLE = "simple"              # Direct execution (existing behavior)
    HIERARCHICAL = "hierarchical"  # DAG-based decomposition with parallel execution
    ADAPTIVE = "adaptive"          # Dynamic re-planning based on execution feedback


class TaskStatus(Enum):
    """Task execution status enumeration"""
    PENDING = "pending"        # Task created but not started
    READY = "ready"            # Dependencies satisfied, ready to execute
    RUNNING = "running"        # Currently executing
    COMPLETED = "completed"    # Successfully completed
    FAILED = "failed"          # Execution failed
    CANCELLED = "cancelled"    # Cancelled due to dependency failure
    RETRYING = "retrying"      # Retrying after failure


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


@dataclass
class TaskNode:
    """Represents a single task in the dependency graph"""
    task_id: str
    description: str
    system_preference: str  # "alita", "kgot", "hybrid", "auto"
    estimated_time_seconds: float = 30.0
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    requires_mcp_creation: bool = False
    requires_knowledge_graph: bool = False
    requires_web_interaction: bool = False
    requires_code_generation: bool = False
    complexity_score: int = 5
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key in ['created_at', 'started_at', 'completed_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        # Convert enums to values
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data


@dataclass
class TaskEdge:
    """Represents a dependency relationship between tasks"""
    source_task_id: str
    target_task_id: str
    data_requirements: List[str] = field(default_factory=list)  # What data flows between tasks
    is_optional: bool = False  # Whether this dependency is optional
    condition: Optional[str] = None  # Conditional dependency logic
    weight: float = 1.0  # Edge weight for priority calculations
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class TaskDAG:
    """Directed Acyclic Graph for task dependencies"""
    
    def __init__(self, dag_id: str = None):
        self.dag_id = dag_id or str(uuid4())
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, TaskNode] = {}
        self.edges: Dict[Tuple[str, str], TaskEdge] = {}
        self.created_at = datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    def add_task(self, task: TaskNode) -> None:
        """Add a task node to the DAG"""
        self.nodes[task.task_id] = task
        self.graph.add_node(task.task_id, task=task)
        
        logger.debug(f"Added task {task.task_id} to DAG {self.dag_id}", extra={
            'operation': 'DAG_ADD_TASK',
            'dag_id': self.dag_id,
            'task_id': task.task_id,
            'description': task.description[:100]
        })
    
    def add_dependency(self, edge: TaskEdge) -> None:
        """Add a dependency edge between tasks"""
        if edge.source_task_id not in self.nodes:
            raise ValueError(f"Source task {edge.source_task_id} not found in DAG")
        if edge.target_task_id not in self.nodes:
            raise ValueError(f"Target task {edge.target_task_id} not found in DAG")
        
        edge_key = (edge.source_task_id, edge.target_task_id)
        self.edges[edge_key] = edge
        self.graph.add_edge(edge.source_task_id, edge.target_task_id, edge=edge)
        
        logger.debug(f"Added dependency {edge.source_task_id} -> {edge.target_task_id}", extra={
            'operation': 'DAG_ADD_DEPENDENCY',
            'dag_id': self.dag_id,
            'source': edge.source_task_id,
            'target': edge.target_task_id,
            'optional': edge.is_optional
        })
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate DAG structure and return any issues"""
        issues = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            try:
                cycle = nx.find_cycle(self.graph)
                issues.append(f"Cycle detected: {' -> '.join([str(node) for node, _ in cycle])}")
            except nx.NetworkXNoCycle:
                issues.append("Graph contains cycles but specific cycle not found")
        
        # Check for isolated nodes (except root nodes)
        isolated = list(nx.isolates(self.graph))
        if isolated:
            issues.append(f"Isolated nodes found: {isolated}")
        
        # Check for missing dependencies
        for edge_key, edge in self.edges.items():
            if edge.source_task_id not in self.nodes:
                issues.append(f"Missing source task: {edge.source_task_id}")
            if edge.target_task_id not in self.nodes:
                issues.append(f"Missing target task: {edge.target_task_id}")
        
        is_valid = len(issues) == 0
        
        logger.info(f"DAG validation result: {'VALID' if is_valid else 'INVALID'}", extra={
            'operation': 'DAG_VALIDATION',
            'dag_id': self.dag_id,
            'is_valid': is_valid,
            'issues_count': len(issues),
            'node_count': len(self.nodes),
            'edge_count': len(self.edges)
        })
        
        return is_valid, issues
    
    def get_ready_tasks(self) -> List[TaskNode]:
        """Get tasks that are ready to execute (no pending dependencies)"""
        ready_tasks = []
        
        for task_id, task in self.nodes.items():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are satisfied
            dependencies_satisfied = True
            for pred_id in self.graph.predecessors(task_id):
                pred_task = self.nodes[pred_id]
                edge_key = (pred_id, task_id)
                edge = self.edges.get(edge_key)
                
                # Required dependency must be completed
                if not edge or not edge.is_optional:
                    if pred_task.status != TaskStatus.COMPLETED:
                        dependencies_satisfied = False
                        break
                # Optional dependency can be completed or failed
                elif edge.is_optional:
                    if pred_task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        dependencies_satisfied = False
                        break
            
            if dependencies_satisfied:
                task.status = TaskStatus.READY
                ready_tasks.append(task)
        
        logger.debug(f"Found {len(ready_tasks)} ready tasks", extra={
            'operation': 'DAG_GET_READY_TASKS',
            'dag_id': self.dag_id,
            'ready_count': len(ready_tasks),
            'ready_task_ids': [t.task_id for t in ready_tasks]
        })
        
        return ready_tasks
    
    def get_critical_path(self) -> List[str]:
        """Calculate critical path through the DAG"""
        try:
            # Use longest path algorithm with task duration as weights
            for node_id, task in self.nodes.items():
                self.graph.nodes[node_id]['weight'] = task.estimated_time_seconds
            
            # Find longest path (critical path)
            critical_path = nx.dag_longest_path(self.graph, weight='weight')
            
            logger.debug(f"Critical path calculated: {len(critical_path)} tasks", extra={
                'operation': 'DAG_CRITICAL_PATH',
                'dag_id': self.dag_id,
                'path_length': len(critical_path),
                'critical_tasks': critical_path
            })
            
            return critical_path
            
        except Exception as e:
            logger.error(f"Error calculating critical path: {str(e)}", extra={
                'operation': 'DAG_CRITICAL_PATH_ERROR',
                'dag_id': self.dag_id,
                'error': str(e)
            })
            return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DAG to dictionary for serialization"""
        return {
            'dag_id': self.dag_id,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'nodes': {task_id: task.to_dict() for task_id, task in self.nodes.items()},
            'edges': [edge.to_dict() for edge in self.edges.values()],
            'graph_info': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges),
                'is_dag': nx.is_directed_acyclic_graph(self.graph)
            }
        }


class AdvancedTaskDecomposer:
    """Decomposes high-level tasks into DAG structures using Sequential Thinking MCP"""
    
    def __init__(self, sequential_thinking: SequentialThinkingMCPIntegration):
        self.sequential_thinking = sequential_thinking
        self.decomposition_cache: Dict[str, TaskDAG] = {}
    
    async def decompose_task(self, 
                           task_description: str, 
                           context: Dict[str, Any] = None,
                           orchestration_mode: OrchestrationMode = OrchestrationMode.HIERARCHICAL) -> TaskDAG:
        """Decompose a high-level task into a DAG of subtasks"""
        
        # Create cache key
        cache_key = hashlib.md5(f"{task_description}_{json.dumps(context or {}, sort_keys=True)}".encode()).hexdigest()
        
        if cache_key in self.decomposition_cache:
            logger.info("Using cached task decomposition", extra={
                'operation': 'TASK_DECOMPOSITION_CACHE_HIT',
                'cache_key': cache_key
            })
            return self.decomposition_cache[cache_key]
        
        logger.info("Starting task decomposition", extra={
            'operation': 'TASK_DECOMPOSITION_START',
            'description': task_description[:100],
            'mode': orchestration_mode.value
        })
        
        # Prepare specialized prompt for DAG generation
        decomposition_prompt = self._create_decomposition_prompt(
            task_description, context, orchestration_mode
        )
        
        try:
            # Use Sequential Thinking MCP for intelligent decomposition
            thinking_result = await self.sequential_thinking.process_complex_task(
                task_description=decomposition_prompt,
                context=context or {},
                thinking_mode="task_decomposition"
            )
            
            # Parse the result to create DAG
            dag = await self._parse_thinking_result_to_dag(thinking_result, task_description)
            
            # Validate the generated DAG
            is_valid, issues = dag.validate()
            if not is_valid:
                logger.warning(f"Generated DAG has issues: {issues}", extra={
                    'operation': 'TASK_DECOMPOSITION_VALIDATION_FAILED',
                    'issues': issues
                })
                # Attempt to fix common issues
                dag = await self._fix_dag_issues(dag, issues)
            
            # Cache the result
            self.decomposition_cache[cache_key] = dag
            
            logger.info("Task decomposition completed", extra={
                'operation': 'TASK_DECOMPOSITION_COMPLETE',
                'dag_id': dag.dag_id,
                'task_count': len(dag.nodes),
                'dependency_count': len(dag.edges)
            })
            
            return dag
            
        except Exception as e:
            logger.error(f"Task decomposition failed: {str(e)}", extra={
                'operation': 'TASK_DECOMPOSITION_ERROR',
                'error': str(e),
                'description': task_description[:100]
            })
            
            # Return simple single-task DAG as fallback
            return self._create_fallback_dag(task_description, context)
    
    def _create_decomposition_prompt(self, 
                                   task_description: str, 
                                   context: Dict[str, Any],
                                   mode: OrchestrationMode) -> str:
        """Create specialized prompt for DAG generation"""
        
        prompt = f"""You are an expert task decomposition system. Your job is to break down a high-level task into a structured dependency graph (DAG) of subtasks.

Task to decompose: {task_description}

Context: {json.dumps(context or {}, indent=2)}

Orchestration Mode: {mode.value}

Please analyze this task and create a detailed decomposition that includes:

1. **Subtasks**: Break the main task into 3-8 specific, actionable subtasks
2. **Dependencies**: Identify which subtasks depend on others (create a dependency graph)
3. **System Preferences**: For each subtask, specify whether it's better suited for:
   - "alita": Tool creation, web interaction, code generation
   - "kgot": Knowledge graph operations, complex reasoning
   - "hybrid": Requires both systems
   - "auto": Let the system decide

4. **Complexity Scores**: Rate each subtask complexity from 1-10
5. **Time Estimates**: Estimate execution time in seconds for each subtask
6. **Resource Requirements**: Specify any special requirements (memory, API calls, etc.)

Format your response as a structured analysis that clearly identifies:
- The list of subtasks with their properties
- The dependency relationships between them
- Any parallel execution opportunities
- Critical path considerations

Focus on creating an efficient execution plan that maximizes parallelism while respecting dependencies."""
        
        return prompt
    
    async def _parse_thinking_result_to_dag(self, 
                                          thinking_result: Dict[str, Any], 
                                          original_description: str) -> TaskDAG:
        """Parse Sequential Thinking MCP result into TaskDAG structure"""
        
        dag = TaskDAG()
        dag.metadata['original_description'] = original_description
        dag.metadata['thinking_session_id'] = thinking_result.get('sessionId')
        
        try:
            # Extract conclusions from thinking result
            conclusions = thinking_result.get('thinkingResult', {}).get('conclusions', {})
            
            # Parse subtasks from the thinking result
            subtasks = self._extract_subtasks_from_conclusions(conclusions)
            
            # Create task nodes
            for i, subtask in enumerate(subtasks):
                task_node = TaskNode(
                    task_id=f"task_{i+1}",
                    description=subtask.get('description', f"Subtask {i+1}"),
                    system_preference=subtask.get('system_preference', 'auto'),
                    estimated_time_seconds=subtask.get('estimated_time', 30.0),
                    priority=TaskPriority(subtask.get('priority', 2)),
                    complexity_score=subtask.get('complexity_score', 5),
                    requires_mcp_creation=subtask.get('requires_mcp_creation', False),
                    requires_knowledge_graph=subtask.get('requires_knowledge_graph', False),
                    requires_web_interaction=subtask.get('requires_web_interaction', False),
                    requires_code_generation=subtask.get('requires_code_generation', False),
                    resource_requirements=subtask.get('resource_requirements', {}),
                    metadata=subtask.get('metadata', {})
                )
                dag.add_task(task_node)
            
            # Parse dependencies
            dependencies = self._extract_dependencies_from_conclusions(conclusions, subtasks)
            
            # Create dependency edges
            for dep in dependencies:
                edge = TaskEdge(
                    source_task_id=dep['source'],
                    target_task_id=dep['target'],
                    data_requirements=dep.get('data_requirements', []),
                    is_optional=dep.get('is_optional', False),
                    condition=dep.get('condition'),
                    weight=dep.get('weight', 1.0)
                )
                dag.add_dependency(edge)
            
            return dag
            
        except Exception as e:
            logger.error(f"Error parsing thinking result to DAG: {str(e)}", extra={
                'operation': 'PARSE_THINKING_RESULT_ERROR',
                'error': str(e)
            })
            
            # Return fallback DAG
            return self._create_fallback_dag(original_description, {})
    
    def _extract_subtasks_from_conclusions(self, conclusions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract subtask information from Sequential Thinking conclusions"""
        
        # This is a simplified parser - in production, you'd want more sophisticated NLP
        subtasks = []
        
        # Look for structured subtask information in conclusions
        key_insights = conclusions.get('keyInsights', {})
        recommended_actions = key_insights.get('recommendedActions', [])
        
        if isinstance(recommended_actions, list):
            for i, action in enumerate(recommended_actions):
                if isinstance(action, str):
                    subtasks.append({
                        'description': action,
                        'system_preference': 'auto',
                        'estimated_time': 30.0,
                        'priority': 2,
                        'complexity_score': 5
                    })
        
        # If no structured subtasks found, create default decomposition
        if not subtasks:
            subtasks = [
                {
                    'description': 'Analyze task requirements',
                    'system_preference': 'kgot',
                    'estimated_time': 15.0,
                    'priority': 3,
                    'complexity_score': 4
                },
                {
                    'description': 'Execute main task logic',
                    'system_preference': 'auto',
                    'estimated_time': 60.0,
                    'priority': 4,
                    'complexity_score': 7
                },
                {
                    'description': 'Validate and format results',
                    'system_preference': 'alita',
                    'estimated_time': 20.0,
                    'priority': 2,
                    'complexity_score': 3
                }
            ]
        
        return subtasks
    
    def _extract_dependencies_from_conclusions(self, 
                                             conclusions: Dict[str, Any], 
                                             subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract dependency information from conclusions"""
        
        dependencies = []
        
        # Create simple sequential dependencies as default
        for i in range(len(subtasks) - 1):
            dependencies.append({
                'source': f'task_{i+1}',
                'target': f'task_{i+2}',
                'data_requirements': ['result'],
                'is_optional': False,
                'weight': 1.0
            })
        
        return dependencies
    
    def _create_fallback_dag(self, description: str, context: Dict[str, Any]) -> TaskDAG:
        """Create a simple fallback DAG when decomposition fails"""
        
        dag = TaskDAG()
        dag.metadata['is_fallback'] = True
        dag.metadata['original_description'] = description
        
        # Create single task
        task = TaskNode(
            task_id="fallback_task_1",
            description=description,
            system_preference="auto",
            estimated_time_seconds=60.0,
            priority=TaskPriority.MEDIUM,
            complexity_score=5
        )
        
        dag.add_task(task)
        
        logger.warning("Created fallback DAG with single task", extra={
            'operation': 'FALLBACK_DAG_CREATED',
            'dag_id': dag.dag_id
        })
        
        return dag
    
    async def _fix_dag_issues(self, dag: TaskDAG, issues: List[str]) -> TaskDAG:
        """Attempt to fix common DAG issues"""
        
        logger.info(f"Attempting to fix DAG issues: {issues}", extra={
            'operation': 'DAG_FIX_ATTEMPT',
            'dag_id': dag.dag_id,
            'issues': issues
        })
        
        # For now, return the DAG as-is
        # In production, implement specific fixes for common issues
        return dag


class DynamicTaskPrioritizer:
    """Handles dynamic task prioritization and re-planning"""
    
    def __init__(self, 
                 sequential_thinking: SequentialThinkingMCPIntegration,
                 monitoring_system: AdvancedMonitoringSystem):
        self.sequential_thinking = sequential_thinking
        self.monitoring_system = monitoring_system
        self.priority_weights = {
            'critical_path_impact': 0.3,
            'estimated_time': 0.2,
            'dependency_count': 0.2,
            'mcp_accuracy': 0.15,
            'resource_availability': 0.15
        }
    
    async def calculate_task_priority(self, 
                                    task: TaskNode, 
                                    dag: TaskDAG,
                                    system_metrics: Dict[str, Any]) -> float:
        """Calculate dynamic priority score for a task"""
        
        priority_score = 0.0
        
        # Critical path impact
        critical_path = dag.get_critical_path()
        if task.task_id in critical_path:
            critical_path_score = 1.0
        else:
            critical_path_score = 0.5
        priority_score += critical_path_score * self.priority_weights['critical_path_impact']
        
        # Estimated time (shorter tasks get higher priority for quick wins)
        time_score = max(0, 1.0 - (task.estimated_time_seconds / 300.0))  # Normalize to 5 minutes
        priority_score += time_score * self.priority_weights['estimated_time']
        
        # Dependency count (tasks that unblock more tasks get higher priority)
        dependent_count = len(list(dag.graph.successors(task.task_id)))
        dependency_score = min(1.0, dependent_count / 5.0)  # Normalize to 5 dependents
        priority_score += dependency_score * self.priority_weights['dependency_count']
        
        # MCP accuracy prediction (placeholder - integrate with RAG-MCP)
        mcp_accuracy_score = 0.8  # Default accuracy
        priority_score += mcp_accuracy_score * self.priority_weights['mcp_accuracy']
        
        # Resource availability
        resource_score = self._calculate_resource_availability_score(task, system_metrics)
        priority_score += resource_score * self.priority_weights['resource_availability']
        
        logger.debug(f"Calculated priority {priority_score:.3f} for task {task.task_id}", extra={
            'operation': 'TASK_PRIORITY_CALCULATION',
            'task_id': task.task_id,
            'priority_score': priority_score,
            'critical_path': critical_path_score,
            'time_score': time_score,
            'dependency_score': dependency_score
        })
        
        return priority_score
    
    def _calculate_resource_availability_score(self, 
                                             task: TaskNode, 
                                             system_metrics: Dict[str, Any]) -> float:
        """Calculate resource availability score for task"""
        
        # Simplified resource scoring
        # In production, integrate with actual resource monitoring
        
        system_preference = task.system_preference
        if system_preference == "alita":
            alita_load = system_metrics.get('alita_load', 0.5)
            return max(0, 1.0 - alita_load)
        elif system_preference == "kgot":
            kgot_load = system_metrics.get('kgot_load', 0.5)
            return max(0, 1.0 - kgot_load)
        else:
            # For auto/hybrid, use average load
            avg_load = (system_metrics.get('alita_load', 0.5) + system_metrics.get('kgot_load', 0.5)) / 2
            return max(0, 1.0 - avg_load)
    
    async def replan_on_failure(self, 
                              failed_task: TaskNode, 
                              dag: TaskDAG,
                              failure_context: Dict[str, Any]) -> Tuple[TaskDAG, List[str]]:
        """Generate alternative plan when a critical task fails"""
        
        logger.info(f"Replanning due to task failure: {failed_task.task_id}", extra={
            'operation': 'REPLAN_ON_FAILURE',
            'failed_task_id': failed_task.task_id,
            'error': failed_task.error
        })
        
        # Use Sequential Thinking MCP to analyze failure and generate alternatives
        replan_prompt = f"""A critical task has failed in our execution plan. Please analyze the failure and suggest alternative approaches.

Failed Task: {failed_task.description}
Error: {failed_task.error}
Failure Context: {json.dumps(failure_context, indent=2)}

Current DAG has {len(dag.nodes)} tasks with {len(dag.edges)} dependencies.

Please suggest:
1. Alternative approaches for the failed task
2. Whether to retry with different parameters
3. Whether to skip this task and continue
4. Whether to modify dependent tasks
5. Any new tasks that might be needed

Focus on maintaining the overall goal while working around this failure."""
        
        try:
            thinking_result = await self.sequential_thinking.process_complex_task(
                task_description=replan_prompt,
                context=failure_context,
                thinking_mode="error_resolution"
            )
            
            # Parse recommendations
            recommendations = self._parse_replan_recommendations(thinking_result)
            
            # Apply recommendations to create new DAG
            new_dag = await self._apply_replan_recommendations(dag, failed_task, recommendations)
            
            return new_dag, recommendations
            
        except Exception as e:
            logger.error(f"Replanning failed: {str(e)}", extra={
                'operation': 'REPLAN_ERROR',
                'error': str(e)
            })
            
            # Return original DAG with failed task marked as cancelled
            failed_task.status = TaskStatus.CANCELLED
            return dag, ["Replanning failed - continuing with original plan"]
    
    def _parse_replan_recommendations(self, thinking_result: Dict[str, Any]) -> List[str]:
        """Parse replanning recommendations from Sequential Thinking result"""
        
        recommendations = []
        
        try:
            conclusions = thinking_result.get('thinkingResult', {}).get('conclusions', {})
            key_insights = conclusions.get('keyInsights', {})
            recommended_actions = key_insights.get('recommendedActions', [])
            
            if isinstance(recommended_actions, list):
                recommendations.extend(recommended_actions)
            
        except Exception as e:
            logger.error(f"Error parsing replan recommendations: {str(e)}", extra={
                'operation': 'PARSE_REPLAN_ERROR',
                'error': str(e)
            })
        
        if not recommendations:
            recommendations = ["Retry failed task with modified parameters"]
        
        return recommendations
    
    async def _apply_replan_recommendations(self, 
                                          dag: TaskDAG, 
                                          failed_task: TaskNode,
                                          recommendations: List[str]) -> TaskDAG:
        """Apply replanning recommendations to create new DAG"""
        
        # For now, implement simple retry logic
        # In production, implement more sophisticated replanning
        
        if failed_task.retry_count < failed_task.max_retries:
            failed_task.retry_count += 1
            failed_task.status = TaskStatus.PENDING
            failed_task.error = None
            
            logger.info(f"Retrying failed task {failed_task.task_id} (attempt {failed_task.retry_count})", extra={
                'operation': 'TASK_RETRY',
                'task_id': failed_task.task_id,
                'retry_count': failed_task.retry_count
            })
        else:
            failed_task.status = TaskStatus.CANCELLED
            
            logger.warning(f"Task {failed_task.task_id} exceeded max retries - cancelling", extra={
                'operation': 'TASK_CANCELLED',
                'task_id': failed_task.task_id,
                'max_retries': failed_task.max_retries
            })
        
        return dag


class ParallelCoordinator:
    """Manages parallel execution of tasks with dependency coordination"""
    
    def __init__(self, 
                 max_concurrent_tasks: int = 5,
                 resource_semaphore: Optional[asyncio.Semaphore] = None):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.resource_semaphore = resource_semaphore or asyncio.Semaphore(max_concurrent_tasks)
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        self.execution_stats = {
            'tasks_started': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0
        }
    
    async def execute_dag(self, 
                         dag: TaskDAG,
                         task_executor: 'TaskExecutor',
                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Execute DAG with parallel coordination"""
        
        logger.info(f"Starting DAG execution: {dag.dag_id}", extra={
            'operation': 'DAG_EXECUTION_START',
            'dag_id': dag.dag_id,
            'task_count': len(dag.nodes),
            'max_concurrent': self.max_concurrent_tasks
        })
        
        start_time = time.time()
        
        try:
            while True:
                # Get ready tasks
                ready_tasks = dag.get_ready_tasks()
                
                if not ready_tasks:
                    # Check if we're done or stuck
                    pending_tasks = [t for t in dag.nodes.values() if t.status == TaskStatus.PENDING]
                    running_tasks = [t for t in dag.nodes.values() if t.status == TaskStatus.RUNNING]
                    
                    if not pending_tasks and not running_tasks:
                        # All tasks completed
                        break
                    elif not running_tasks:
                        # Stuck - no ready tasks and no running tasks
                        logger.error("DAG execution stuck - no ready or running tasks", extra={
                            'operation': 'DAG_EXECUTION_STUCK',
                            'dag_id': dag.dag_id,
                            'pending_count': len(pending_tasks)
                        })
                        break
                    else:
                        # Wait for running tasks to complete
                        await asyncio.sleep(0.1)
                        continue
                
                # Start ready tasks (up to concurrency limit)
                tasks_to_start = ready_tasks[:self.max_concurrent_tasks - len(self.running_tasks)]
                
                for task in tasks_to_start:
                    if len(self.running_tasks) >= self.max_concurrent_tasks:
                        break
                    
                    # Start task execution
                    task_coroutine = self._execute_single_task(task, dag, task_executor)
                    async_task = asyncio.create_task(task_coroutine)
                    self.running_tasks[task.task_id] = async_task
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    self.execution_stats['tasks_started'] += 1
                    
                    logger.debug(f"Started task execution: {task.task_id}", extra={
                        'operation': 'TASK_EXECUTION_START',
                        'task_id': task.task_id,
                        'running_count': len(self.running_tasks)
                    })
                
                # Wait for at least one task to complete
                if self.running_tasks:
                    done, pending = await asyncio.wait(
                        self.running_tasks.values(),
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Process completed tasks
                    for completed_task in done:
                        task_id = None
                        for tid, t in self.running_tasks.items():
                            if t == completed_task:
                                task_id = tid
                                break
                        
                        if task_id:
                            del self.running_tasks[task_id]
                            
                            try:
                                result = await completed_task
                                self.task_results[task_id] = result
                                self.execution_stats['tasks_completed'] += 1
                                
                                # Update task status
                                task_node = dag.nodes[task_id]
                                if result.get('success', False):
                                    task_node.status = TaskStatus.COMPLETED
                                    task_node.result = result
                                else:
                                    task_node.status = TaskStatus.FAILED
                                    task_node.error = result.get('error', 'Unknown error')
                                    self.execution_stats['tasks_failed'] += 1
                                
                                task_node.completed_at = datetime.now()
                                
                                logger.debug(f"Task completed: {task_id}", extra={
                                    'operation': 'TASK_EXECUTION_COMPLETE',
                                    'task_id': task_id,
                                    'success': result.get('success', False),
                                    'duration': (task_node.completed_at - task_node.started_at).total_seconds()
                                })
                                
                            except Exception as e:
                                logger.error(f"Task execution failed: {task_id} - {str(e)}", extra={
                                    'operation': 'TASK_EXECUTION_ERROR',
                                    'task_id': task_id,
                                    'error': str(e)
                                })
                                
                                task_node = dag.nodes[task_id]
                                task_node.status = TaskStatus.FAILED
                                task_node.error = str(e)
                                task_node.completed_at = datetime.now()
                                self.execution_stats['tasks_failed'] += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_info = {
                        'completed_tasks': self.execution_stats['tasks_completed'],
                        'total_tasks': len(dag.nodes),
                        'running_tasks': len(self.running_tasks),
                        'failed_tasks': self.execution_stats['tasks_failed']
                    }
                    await progress_callback(progress_info)
            
            # Calculate final statistics
            end_time = time.time()
            self.execution_stats['total_execution_time'] = end_time - start_time
            
            # Compile final results
            execution_result = {
                'dag_id': dag.dag_id,
                'success': self.execution_stats['tasks_failed'] == 0,
                'statistics': self.execution_stats.copy(),
                'task_results': self.task_results.copy(),
                'completed_tasks': [t.task_id for t in dag.nodes.values() if t.status == TaskStatus.COMPLETED],
                'failed_tasks': [t.task_id for t in dag.nodes.values() if t.status == TaskStatus.FAILED],
                'execution_time_seconds': self.execution_stats['total_execution_time']
            }
            
            logger.info(f"DAG execution completed: {dag.dag_id}", extra={
                'operation': 'DAG_EXECUTION_COMPLETE',
                'dag_id': dag.dag_id,
                'success': execution_result['success'],
                'duration': execution_result['execution_time_seconds'],
                'completed_count': len(execution_result['completed_tasks']),
                'failed_count': len(execution_result['failed_tasks'])
            })
            
            return execution_result
            
        except Exception as e:
            logger.error(f"DAG execution failed: {str(e)}", extra={
                'operation': 'DAG_EXECUTION_ERROR',
                'dag_id': dag.dag_id,
                'error': str(e)
            })
            
            # Cancel all running tasks
            for task in self.running_tasks.values():
                task.cancel()
            
            raise
    
    async def _execute_single_task(self, 
                                 task: TaskNode, 
                                 dag: TaskDAG,
                                 task_executor: 'TaskExecutor') -> Dict[str, Any]:
        """Execute a single task with resource management"""
        
        async with self.resource_semaphore:
            try:
                # Collect input data from dependencies
                input_data = await self._collect_task_inputs(task, dag)
                
                # Execute the task
                result = await task_executor.execute_task(task, input_data)
                
                return result
                
            except Exception as e:
                logger.error(f"Single task execution failed: {task.task_id} - {str(e)}", extra={
                    'operation': 'SINGLE_TASK_ERROR',
                    'task_id': task.task_id,
                    'error': str(e)
                })
                
                return {
                    'success': False,
                    'error': str(e),
                    'task_id': task.task_id
                }
    
    async def _collect_task_inputs(self, task: TaskNode, dag: TaskDAG) -> Dict[str, Any]:
        """Collect input data from completed dependency tasks"""
        
        inputs = {}
        
        # Get all predecessor tasks
        for pred_id in dag.graph.predecessors(task.task_id):
            pred_task = dag.nodes[pred_id]
            
            if pred_task.status == TaskStatus.COMPLETED and pred_task.result:
                # Get edge information to understand data requirements
                edge_key = (pred_id, task.task_id)
                edge = dag.edges.get(edge_key)
                
                if edge and edge.data_requirements:
                    # Extract specific data based on requirements
                    for req in edge.data_requirements:
                        if req in pred_task.result:
                            inputs[f"{pred_id}_{req}"] = pred_task.result[req]
                else:
                    # Include all result data
                    inputs[pred_id] = pred_task.result
        
        return inputs


class TaskExecutor:
    """Executes individual tasks using appropriate systems (Alita/KGoT)"""
    
    def __init__(self, 
                 unified_controller: 'UnifiedSystemController',
                 state_manager: EnhancedSharedStateManager):
        self.unified_controller = unified_controller
        self.state_manager = state_manager
    
    async def execute_task(self, task: TaskNode, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task using the appropriate system"""
        
        logger.info(f"Executing task: {task.task_id}", extra={
            'operation': 'TASK_EXECUTION',
            'task_id': task.task_id,
            'system_preference': task.system_preference,
            'complexity': task.complexity_score
        })
        
        try:
            # Prepare task context for unified controller
            from .unified_system_controller import TaskContext, RoutingStrategy
            
            # Map system preference to routing strategy
            routing_strategy_map = {
                'alita': RoutingStrategy.ALITA_FIRST,
                'kgot': RoutingStrategy.KGOT_FIRST,
                'hybrid': RoutingStrategy.HYBRID,
                'auto': RoutingStrategy.HYBRID  # Default to hybrid for auto
            }
            
            task_context = TaskContext(
                task_id=task.task_id,
                description=task.description,
                complexity_score=task.complexity_score,
                routing_strategy=routing_strategy_map.get(task.system_preference, RoutingStrategy.HYBRID),
                requires_mcp_creation=task.requires_mcp_creation,
                requires_knowledge_graph=task.requires_knowledge_graph,
                requires_web_interaction=task.requires_web_interaction,
                requires_code_generation=task.requires_code_generation,
                timeout_seconds=int(task.estimated_time_seconds * 2)  # Allow 2x estimated time
            )
            
            # Add input data to context
            if input_data:
                task_context.budget_constraints = {'input_data': input_data}
            
            # Execute through unified controller
            execution_result = await self.unified_controller.execute_task(task_context)
            
            if execution_result.success:
                return {
                    'success': True,
                    'result': execution_result.result,
                    'systems_used': execution_result.systems_used,
                    'execution_time_ms': execution_result.execution_time_ms,
                    'task_id': task.task_id
                }
            else:
                return {
                    'success': False,
                    'error': execution_result.error,
                    'task_id': task.task_id
                }
                
        except Exception as e:
            logger.error(f"Task execution failed: {task.task_id} - {str(e)}", extra={
                'operation': 'TASK_EXECUTION_ERROR',
                'task_id': task.task_id,
                'error': str(e)
            })
            
            return {
                'success': False,
                'error': str(e),
                'task_id': task.task_id
            }


class AdaptiveTaskScheduler:
    """Monitors and schedules tasks dynamically based on system performance"""
    
    def __init__(self, 
                 monitoring_system: AdvancedMonitoringSystem,
                 load_balancer: AdaptiveLoadBalancer):
        self.monitoring_system = monitoring_system
        self.load_balancer = load_balancer
        self.scheduling_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
    
    async def schedule_tasks(self, 
                           ready_tasks: List[TaskNode],
                           system_metrics: Dict[str, Any]) -> List[TaskNode]:
        """Schedule tasks based on current system performance and load"""
        
        if not ready_tasks:
            return []
        
        logger.debug(f"Scheduling {len(ready_tasks)} ready tasks", extra={
            'operation': 'TASK_SCHEDULING',
            'ready_count': len(ready_tasks)
        })
        
        # Sort tasks by priority and system load
        scheduled_tasks = await self._prioritize_and_schedule(ready_tasks, system_metrics)
        
        # Record scheduling decision
        self.scheduling_history.append({
            'timestamp': datetime.now().isoformat(),
            'ready_tasks': len(ready_tasks),
            'scheduled_tasks': len(scheduled_tasks),
            'system_metrics': system_metrics.copy()
        })
        
        return scheduled_tasks
    
    async def _prioritize_and_schedule(self, 
                                     tasks: List[TaskNode],
                                     system_metrics: Dict[str, Any]) -> List[TaskNode]:
        """Prioritize and schedule tasks based on system state"""
        
        # Simple scheduling logic - in production, implement more sophisticated algorithms
        
        # Sort by priority and estimated time
        sorted_tasks = sorted(tasks, key=lambda t: (t.priority.value, -t.estimated_time_seconds), reverse=True)
        
        # Apply load balancing constraints
        scheduled_tasks = []
        alita_load = system_metrics.get('alita_load', 0.5)
        kgot_load = system_metrics.get('kgot_load', 0.5)
        
        for task in sorted_tasks:
            # Check if system can handle this task
            if task.system_preference == 'alita' and alita_load > 0.8:
                continue  # Skip if Alita is overloaded
            elif task.system_preference == 'kgot' and kgot_load > 0.8:
                continue  # Skip if KGoT is overloaded
            
            scheduled_tasks.append(task)
        
        return scheduled_tasks
    
    async def update_performance_metrics(self, 
                                       task_results: Dict[str, Any]) -> None:
        """Update performance metrics based on task execution results"""
        
        # Update scheduling performance metrics
        for task_id, result in task_results.items():
            if 'execution_time_ms' in result:
                self.performance_metrics[f'{task_id}_execution_time'] = result['execution_time_ms']
            
            if 'success' in result:
                self.performance_metrics[f'{task_id}_success'] = 1.0 if result['success'] else 0.0
        
        logger.debug("Updated performance metrics", extra={
            'operation': 'PERFORMANCE_METRICS_UPDATE',
            'metrics_count': len(self.performance_metrics)
        })


class AdvancedTaskOrchestrator:
    """Main orchestrator class integrating all advanced task orchestration components"""
    
    def __init__(self, 
                 unified_controller: 'UnifiedSystemController',
                 sequential_thinking: SequentialThinkingMCPIntegration,
                 state_manager: EnhancedSharedStateManager,
                 monitoring_system: AdvancedMonitoringSystem,
                 load_balancer: AdaptiveLoadBalancer,
                 max_concurrent_tasks: int = 5):
        
        self.unified_controller = unified_controller
        self.state_manager = state_manager
        self.monitoring_system = monitoring_system
        self.load_balancer = load_balancer
        
        # Initialize components
        self.task_decomposer = AdvancedTaskDecomposer(sequential_thinking)
        self.task_prioritizer = DynamicTaskPrioritizer(sequential_thinking, monitoring_system)
        self.parallel_coordinator = ParallelCoordinator(max_concurrent_tasks)
        self.task_scheduler = AdaptiveTaskScheduler(monitoring_system, load_balancer)
        self.task_executor = TaskExecutor(unified_controller, state_manager)
        
        # Orchestration state
        self.active_dags: Dict[str, TaskDAG] = {}
        self.orchestration_history: List[Dict[str, Any]] = []
    
    async def orchestrate_task(self, 
                             task_description: str,
                             context: Dict[str, Any] = None,
                             orchestration_mode: OrchestrationMode = OrchestrationMode.HIERARCHICAL,
                             progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Main orchestration method - decomposes and executes complex tasks"""
        
        orchestration_id = str(uuid4())
        start_time = time.time()
        
        logger.info(f"Starting advanced task orchestration: {orchestration_id}", extra={
            'operation': 'ORCHESTRATION_START',
            'orchestration_id': orchestration_id,
            'description': task_description[:100],
            'mode': orchestration_mode.value
        })
        
        try:
            # Step 1: Task Decomposition
            logger.info("Step 1: Task decomposition", extra={
                'operation': 'ORCHESTRATION_STEP_1',
                'orchestration_id': orchestration_id
            })
            
            dag = await self.task_decomposer.decompose_task(
                task_description, context, orchestration_mode
            )
            
            self.active_dags[orchestration_id] = dag
            
            # Step 2: DAG Validation and Optimization
            logger.info("Step 2: DAG validation and optimization", extra={
                'operation': 'ORCHESTRATION_STEP_2',
                'orchestration_id': orchestration_id,
                'task_count': len(dag.nodes)
            })
            
            is_valid, issues = dag.validate()
            if not is_valid:
                logger.warning(f"DAG validation issues: {issues}", extra={
                    'operation': 'DAG_VALIDATION_ISSUES',
                    'orchestration_id': orchestration_id,
                    'issues': issues
                })
            
            # Step 3: Parallel Execution with Dynamic Scheduling
            logger.info("Step 3: Parallel execution", extra={
                'operation': 'ORCHESTRATION_STEP_3',
                'orchestration_id': orchestration_id
            })
            
            execution_result = await self.parallel_coordinator.execute_dag(
                dag, self.task_executor, progress_callback
            )
            
            # Step 4: Result Compilation and Analysis
            logger.info("Step 4: Result compilation", extra={
                'operation': 'ORCHESTRATION_STEP_4',
                'orchestration_id': orchestration_id,
                'success': execution_result['success']
            })
            
            # Compile final orchestration result
            end_time = time.time()
            orchestration_result = {
                'orchestration_id': orchestration_id,
                'success': execution_result['success'],
                'mode': orchestration_mode.value,
                'original_description': task_description,
                'dag_info': {
                    'dag_id': dag.dag_id,
                    'task_count': len(dag.nodes),
                    'dependency_count': len(dag.edges),
                    'critical_path': dag.get_critical_path()
                },
                'execution_results': execution_result,
                'performance_metrics': {
                    'total_orchestration_time': end_time - start_time,
                    'decomposition_time': 0,  # TODO: Track decomposition time
                    'execution_time': execution_result.get('execution_time_seconds', 0),
                    'tasks_completed': len(execution_result.get('completed_tasks', [])),
                    'tasks_failed': len(execution_result.get('failed_tasks', [])),
                    'parallel_efficiency': self._calculate_parallel_efficiency(dag, execution_result)
                },
                'task_results': execution_result.get('task_results', {}),
                'completed_at': datetime.now().isoformat()
            }
            
            # Store orchestration history
            self.orchestration_history.append(orchestration_result)
            
            # Clean up
            if orchestration_id in self.active_dags:
                del self.active_dags[orchestration_id]
            
            logger.info(f"Advanced task orchestration completed: {orchestration_id}", extra={
                'operation': 'ORCHESTRATION_COMPLETE',
                'orchestration_id': orchestration_id,
                'success': orchestration_result['success'],
                'duration': orchestration_result['performance_metrics']['total_orchestration_time'],
                'tasks_completed': orchestration_result['performance_metrics']['tasks_completed']
            })
            
            return orchestration_result
            
        except Exception as e:
            logger.error(f"Advanced task orchestration failed: {str(e)}", extra={
                'operation': 'ORCHESTRATION_ERROR',
                'orchestration_id': orchestration_id,
                'error': str(e)
            })
            
            # Clean up on failure
            if orchestration_id in self.active_dags:
                del self.active_dags[orchestration_id]
            
            return {
                'orchestration_id': orchestration_id,
                'success': False,
                'error': str(e),
                'mode': orchestration_mode.value,
                'original_description': task_description,
                'completed_at': datetime.now().isoformat()
            }
    
    def _calculate_parallel_efficiency(self, dag: TaskDAG, execution_result: Dict[str, Any]) -> float:
        """Calculate parallel execution efficiency"""
        
        try:
            # Calculate theoretical sequential time
            sequential_time = sum(task.estimated_time_seconds for task in dag.nodes.values())
            
            # Get actual parallel execution time
            parallel_time = execution_result.get('execution_time_seconds', sequential_time)
            
            # Calculate efficiency (higher is better)
            if parallel_time > 0:
                efficiency = min(1.0, sequential_time / parallel_time)
            else:
                efficiency = 0.0
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Error calculating parallel efficiency: {str(e)}", extra={
                'operation': 'PARALLEL_EFFICIENCY_ERROR',
                'error': str(e)
            })
            return 0.0
    
    async def get_orchestration_status(self, orchestration_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an active orchestration"""
        
        if orchestration_id not in self.active_dags:
            return None
        
        dag = self.active_dags[orchestration_id]
        
        # Calculate status summary
        status_summary = {
            'orchestration_id': orchestration_id,
            'dag_id': dag.dag_id,
            'total_tasks': len(dag.nodes),
            'task_status_counts': {
                'pending': len([t for t in dag.nodes.values() if t.status == TaskStatus.PENDING]),
                'ready': len([t for t in dag.nodes.values() if t.status == TaskStatus.READY]),
                'running': len([t for t in dag.nodes.values() if t.status == TaskStatus.RUNNING]),
                'completed': len([t for t in dag.nodes.values() if t.status == TaskStatus.COMPLETED]),
                'failed': len([t for t in dag.nodes.values() if t.status == TaskStatus.FAILED]),
                'cancelled': len([t for t in dag.nodes.values() if t.status == TaskStatus.CANCELLED])
            },
            'progress_percentage': self._calculate_progress_percentage(dag),
            'estimated_completion_time': self._estimate_completion_time(dag),
            'current_running_tasks': [t.task_id for t in dag.nodes.values() if t.status == TaskStatus.RUNNING]
        }
        
        return status_summary
    
    def _calculate_progress_percentage(self, dag: TaskDAG) -> float:
        """Calculate overall progress percentage"""
        
        if not dag.nodes:
            return 0.0
        
        completed_count = len([t for t in dag.nodes.values() if t.status == TaskStatus.COMPLETED])
        total_count = len(dag.nodes)
        
        return (completed_count / total_count) * 100.0
    
    def _estimate_completion_time(self, dag: TaskDAG) -> Optional[float]:
        """Estimate remaining completion time in seconds"""
        
        try:
            # Calculate remaining time based on pending and running tasks
            remaining_tasks = [t for t in dag.nodes.values() 
                             if t.status in [TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING]]
            
            if not remaining_tasks:
                return 0.0
            
            # Simple estimation: sum of remaining task times
            # In production, use more sophisticated estimation considering dependencies
            total_remaining_time = sum(t.estimated_time_seconds for t in remaining_tasks)
            
            return total_remaining_time
            
        except Exception as e:
            logger.error(f"Error estimating completion time: {str(e)}", extra={
                'operation': 'COMPLETION_TIME_ESTIMATION_ERROR',
                'error': str(e)
            })
            return None
    
    async def cancel_orchestration(self, orchestration_id: str) -> bool:
        """Cancel an active orchestration"""
        
        if orchestration_id not in self.active_dags:
            return False
        
        logger.info(f"Cancelling orchestration: {orchestration_id}", extra={
            'operation': 'ORCHESTRATION_CANCEL',
            'orchestration_id': orchestration_id
        })
        
        try:
            dag = self.active_dags[orchestration_id]
            
            # Cancel all running and pending tasks
            for task in dag.nodes.values():
                if task.status in [TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING]:
                    task.status = TaskStatus.CANCELLED
            
            # Cancel any running async tasks in parallel coordinator
            for task_id, async_task in self.parallel_coordinator.running_tasks.items():
                async_task.cancel()
            
            # Clean up
            del self.active_dags[orchestration_id]
            
            logger.info(f"Orchestration cancelled successfully: {orchestration_id}", extra={
                'operation': 'ORCHESTRATION_CANCELLED',
                'orchestration_id': orchestration_id
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling orchestration: {str(e)}", extra={
                'operation': 'ORCHESTRATION_CANCEL_ERROR',
                'orchestration_id': orchestration_id,
                'error': str(e)
            })
            return False
    
    def get_orchestration_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent orchestration history"""
        
        return self.orchestration_history[-limit:] if self.orchestration_history else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall orchestration performance metrics"""
        
        if not self.orchestration_history:
            return {}
        
        # Calculate aggregate metrics
        total_orchestrations = len(self.orchestration_history)
        successful_orchestrations = len([o for o in self.orchestration_history if o.get('success', False)])
        
        avg_execution_time = sum(
            o.get('performance_metrics', {}).get('total_orchestration_time', 0) 
            for o in self.orchestration_history
        ) / total_orchestrations if total_orchestrations > 0 else 0
        
        avg_parallel_efficiency = sum(
            o.get('performance_metrics', {}).get('parallel_efficiency', 0) 
            for o in self.orchestration_history
        ) / total_orchestrations if total_orchestrations > 0 else 0
        
        return {
            'total_orchestrations': total_orchestrations,
            'successful_orchestrations': successful_orchestrations,
            'success_rate': (successful_orchestrations / total_orchestrations) * 100 if total_orchestrations > 0 else 0,
            'average_execution_time_seconds': avg_execution_time,
            'average_parallel_efficiency': avg_parallel_efficiency,
            'active_orchestrations': len(self.active_dags)
        }


# Factory function for creating orchestrator instances
def create_advanced_task_orchestrator(
    unified_controller: 'UnifiedSystemController',
    sequential_thinking: SequentialThinkingMCPIntegration,
    state_manager: EnhancedSharedStateManager,
    monitoring_system: AdvancedMonitoringSystem,
    load_balancer: AdaptiveLoadBalancer,
    max_concurrent_tasks: int = 5
) -> AdvancedTaskOrchestrator:
    """Factory function to create and configure an AdvancedTaskOrchestrator instance"""
    
    logger.info("Creating Advanced Task Orchestrator", extra={
        'operation': 'ORCHESTRATOR_CREATION',
        'max_concurrent_tasks': max_concurrent_tasks
    })
    
    orchestrator = AdvancedTaskOrchestrator(
        unified_controller=unified_controller,
        sequential_thinking=sequential_thinking,
        state_manager=state_manager,
        monitoring_system=monitoring_system,
        load_balancer=load_balancer,
        max_concurrent_tasks=max_concurrent_tasks
    )
    
    return orchestrator


# Utility functions for DAG operations
def export_dag_to_json(dag: TaskDAG) -> str:
    """Export DAG to JSON string for persistence or visualization"""
    return json.dumps(dag.to_dict(), indent=2)


def import_dag_from_json(json_str: str) -> TaskDAG:
    """Import DAG from JSON string"""
    
    data = json.loads(json_str)
    dag = TaskDAG(dag_id=data['dag_id'])
    dag.created_at = datetime.fromisoformat(data['created_at'])
    dag.metadata = data['metadata']
    
    # Reconstruct nodes
    for task_id, task_data in data['nodes'].items():
        task = TaskNode(
            task_id=task_data['task_id'],
            description=task_data['description'],
            system_preference=task_data['system_preference'],
            estimated_time_seconds=task_data['estimated_time_seconds'],
            priority=TaskPriority(task_data['priority']),
            status=TaskStatus(task_data['status']),
            requires_mcp_creation=task_data['requires_mcp_creation'],
            requires_knowledge_graph=task_data['requires_knowledge_graph'],
            requires_web_interaction=task_data['requires_web_interaction'],
            requires_code_generation=task_data['requires_code_generation'],
            complexity_score=task_data['complexity_score'],
            resource_requirements=task_data['resource_requirements'],
            metadata=task_data['metadata'],
            retry_count=task_data['retry_count'],
            max_retries=task_data['max_retries']
        )
        
        # Restore datetime fields
        if task_data['created_at']:
            task.created_at = datetime.fromisoformat(task_data['created_at'])
        if task_data['started_at']:
            task.started_at = datetime.fromisoformat(task_data['started_at'])
        if task_data['completed_at']:
            task.completed_at = datetime.fromisoformat(task_data['completed_at'])
        
        task.result = task_data['result']
        task.error = task_data['error']
        
        dag.add_task(task)
    
    # Reconstruct edges
    for edge_data in data['edges']:
        edge = TaskEdge(
            source_task_id=edge_data['source_task_id'],
            target_task_id=edge_data['target_task_id'],
            data_requirements=edge_data['data_requirements'],
            is_optional=edge_data['is_optional'],
            condition=edge_data['condition'],
            weight=edge_data['weight'],
            metadata=edge_data['metadata']
        )
        dag.add_dependency(edge)
    
    return dag


# Export main classes and functions
__all__ = [
    'OrchestrationMode',
    'TaskStatus', 
    'TaskPriority',
    'TaskNode',
    'TaskEdge', 
    'TaskDAG',
    'AdvancedTaskDecomposer',
    'DynamicTaskPrioritizer',
    'ParallelCoordinator',
    'TaskExecutor',
    'AdaptiveTaskScheduler',
    'AdvancedTaskOrchestrator',
    'create_advanced_task_orchestrator',
    'export_dag_to_json',
    'import_dag_from_json'
]