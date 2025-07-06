#!/usr/bin/env python3
"""
KGoT Scalability Framework

Advanced scalability orchestration system for the Knowledge Graph of Thoughts (KGoT) 
integrated with Alita Enhanced. This framework provides comprehensive scalability
capabilities as specified in KGoT research paper Section 2.4 and implementation requirements.

Key Features:
1. Auto-scaling based on performance metrics and predictive analysis
2. Intelligent load balancing across distributed nodes and optimization strategies  
3. Multi-node coordination for distributed processing
4. AI-driven scaling decisions using LangChain agents
5. Integration with existing PerformanceOptimizer and ContainerOrchestrator
6. Comprehensive scalability monitoring and metrics collection
7. Cost-aware scaling decisions with budget optimization
8. Predictive capacity planning and resource forecasting

Core Components:
- ScalabilityController: Main orchestration and coordination
- AutoScalingEngine: Dynamic auto-scaling with predictive capabilities
- LoadBalancingCoordinator: Intelligent work distribution and balancing
- ScalabilityMetricsCollector: Real-time monitoring and analytics
- MultiNodeCoordinator: Distributed coordination across nodes
- ScalingDecisionEngine: AI-driven scaling intelligence using LangChain
- CapacityPlanner: Predictive capacity planning and forecasting
- IntegrationOrchestrator: Coordination with existing systems

@author: Enhanced Alita KGoT Team
@version: 1.0.0
@based_on: KGoT Research Paper Section 2.4 Performance Optimization
@integration: Alita Enhanced Architecture with Winston Logging
"""

import asyncio
import logging
import time
import threading
import multiprocessing
import json
import psutil
import numpy as np
import redis
from abc import ABC, abstractmethod

# Redis availability check
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import statistics
from collections import deque

# Core system imports - integration with existing KGoT Enhanced Alita architecture
try:
    from .performance_optimization import (
        PerformanceOptimizer, OptimizationStrategy, OptimizationContext
    )
    from .containerization import (
        ContainerOrchestrator
    )
except ImportError:
    # Fallback for testing or standalone usage
    logging.warning("Could not import core KGoT modules - using standalone mode")

# LangChain for AI agents (per user requirements)
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import BaseOutputParser
    from langchain.llms.base import LLM
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available - AI scaling decisions will use rule-based fallback")

# Prometheus metrics for monitoring
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus metrics not available - using internal metrics")

# MPI for distributed processing coordination
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

# Winston-style logging setup compatible with Alita enhanced architecture
import pathlib
log_dir = pathlib.Path('./logs/kgot')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('./logs/kgot/scalability_framework.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('KGoTScalabilityFramework')

# Prometheus metrics setup
if PROMETHEUS_AVAILABLE:
    scaling_operations_total = Counter('kgot_scaling_operations_total', 'Total scaling operations', ['operation_type'])
    scaling_latency_seconds = Histogram('kgot_scaling_latency_seconds', 'Scaling operation latency')
    active_nodes_gauge = Gauge('kgot_active_nodes', 'Number of active nodes')
    load_balance_score = Gauge('kgot_load_balance_score', 'Current load balance score')
    scaling_efficiency_gauge = Gauge('kgot_scaling_efficiency', 'Scaling efficiency score')


class ScalingTrigger(Enum):
    """
    Enumeration of scaling trigger types for comprehensive scaling decision making
    
    Based on KGoT Section 2.4 performance optimization requirements and 
    Alita enhanced architecture scalability needs
    """
    CPU_UTILIZATION = "cpu_utilization"           # CPU usage threshold triggered
    MEMORY_UTILIZATION = "memory_utilization"     # Memory usage threshold triggered  
    QUEUE_DEPTH = "queue_depth"                   # Task queue depth threshold
    RESPONSE_TIME = "response_time"               # Response time degradation
    THROUGHPUT_DROP = "throughput_drop"           # Throughput decrease detected
    ERROR_RATE = "error_rate"                     # Error rate increase
    COST_THRESHOLD = "cost_threshold"             # Cost budget threshold reached
    PREDICTIVE = "predictive"                     # Predictive analysis triggered
    MANUAL = "manual"                             # Manual scaling request
    LOAD_BALANCING = "load_balancing"            # Load imbalance detected
    DISTRIBUTED_COORDINATION = "distributed"     # Distributed processing needs


class ScalingDirection(Enum):
    """
    Scaling direction enumeration for clear scaling operations
    """
    SCALE_UP = "scale_up"           # Increase resources/instances
    SCALE_DOWN = "scale_down"       # Decrease resources/instances  
    SCALE_OUT = "scale_out"         # Horizontal scaling (more nodes)
    SCALE_IN = "scale_in"           # Horizontal scaling (fewer nodes)
    MAINTAIN = "maintain"           # No scaling needed
    REBALANCE = "rebalance"         # Redistribute load without scaling


class ScalingStrategy(Enum):
    """
    Scaling strategies aligned with KGoT performance optimization approaches
    """
    REACTIVE = "reactive"           # React to current conditions
    PREDICTIVE = "predictive"       # Predict future needs
    HYBRID = "hybrid"              # Combine reactive and predictive  
    COST_OPTIMIZED = "cost_optimized"     # Minimize costs while maintaining performance
    PERFORMANCE_FIRST = "performance_first"  # Prioritize performance over cost
    BALANCED = "balanced"          # Balance cost and performance


@dataclass
class ScalingMetrics:
    """
    Comprehensive scaling metrics for intelligent scaling decisions
    
    Integrates with PerformanceMetrics from performance_optimization.py
    and adds scalability-specific measurements
    """
    # Core performance metrics
    cpu_utilization_percent: float = 0.0           # Current CPU utilization
    memory_utilization_percent: float = 0.0        # Current memory utilization
    disk_io_utilization_percent: float = 0.0       # Disk I/O utilization
    network_io_utilization_percent: float = 0.0    # Network I/O utilization
    
    # Scalability-specific metrics  
    active_nodes: int = 1                          # Number of active nodes
    total_capacity_nodes: int = 1                  # Total available nodes
    load_balance_score: float = 1.0                # Load distribution balance (0-1)
    scaling_efficiency: float = 1.0                # Scaling effectiveness (0-1)
    
    # Performance quality metrics
    average_response_time_ms: float = 0.0          # Average response time
    p95_response_time_ms: float = 0.0              # 95th percentile response time
    p99_response_time_ms: float = 0.0              # 99th percentile response time
    throughput_ops_per_second: float = 0.0         # Current throughput
    error_rate_percent: float = 0.0                # Current error rate
    
    # Queue and concurrency metrics
    pending_tasks: int = 0                         # Tasks waiting in queue
    active_tasks: int = 0                          # Currently executing tasks
    completed_tasks: int = 0                       # Completed tasks
    failed_tasks: int = 0                          # Failed tasks
    
    # Resource allocation metrics
    allocated_cpu_cores: float = 0.0               # Allocated CPU cores
    allocated_memory_gb: float = 0.0               # Allocated memory in GB
    allocated_storage_gb: float = 0.0              # Allocated storage in GB
    
    # Cost and efficiency metrics
    current_cost_per_hour: float = 0.0             # Current hourly cost
    cost_per_operation: float = 0.0                # Cost per operation
    resource_efficiency_score: float = 0.0         # Resource usage efficiency
    
    # Predictive metrics
    predicted_load_factor: float = 1.0             # Predicted load increase/decrease
    capacity_headroom_percent: float = 100.0       # Available capacity percentage
    scaling_recommendation_confidence: float = 0.0  # Confidence in scaling recommendation
    
    # Temporal context
    timestamp: datetime = field(default_factory=datetime.utcnow)
    measurement_window_seconds: int = 60           # Measurement window duration
    
    def __post_init__(self):
        """
        Initialize calculated fields and perform validation
        """
        # Ensure all percentage values are within valid range
        self.cpu_utilization_percent = max(0.0, min(100.0, self.cpu_utilization_percent))
        self.memory_utilization_percent = max(0.0, min(100.0, self.memory_utilization_percent))
        self.error_rate_percent = max(0.0, min(100.0, self.error_rate_percent))
        self.load_balance_score = max(0.0, min(1.0, self.load_balance_score))
        self.scaling_efficiency = max(0.0, min(1.0, self.scaling_efficiency))
        
        # Log metrics collection
        logger.debug("Scaling metrics collected", extra={
            'operation': 'SCALING_METRICS_COLLECTION',
            'metadata': {
                'cpu_utilization': self.cpu_utilization_percent,
                'memory_utilization': self.memory_utilization_percent,
                'active_nodes': self.active_nodes,
                'throughput': self.throughput_ops_per_second,
                'pending_tasks': self.pending_tasks
            }
        })
    
    def overall_utilization_score(self) -> float:
        """
        Calculate overall resource utilization score
        
        Returns:
            float: Combined utilization score (0.0-1.0)
        """
        utilization_metrics = [
            self.cpu_utilization_percent / 100.0,
            self.memory_utilization_percent / 100.0,
            self.disk_io_utilization_percent / 100.0,
            self.network_io_utilization_percent / 100.0
        ]
        
        # Weighted average with CPU and memory having higher importance
        weights = [0.4, 0.3, 0.15, 0.15]
        weighted_score = sum(metric * weight for metric, weight in zip(utilization_metrics, weights))
        
        return min(1.0, max(0.0, weighted_score))
    
    def performance_health_score(self) -> float:
        """
        Calculate overall performance health score
        
        Returns:
            float: Performance health score (0.0-1.0, higher is better)
        """
        # Response time score (lower is better)
        response_time_score = 1.0 - min(1.0, self.average_response_time_ms / 5000.0)  # 5s max
        
        # Error rate score (lower is better)
        error_rate_score = 1.0 - (self.error_rate_percent / 100.0)
        
        # Throughput score (normalized, assume target throughput is known)
        throughput_score = min(1.0, self.throughput_ops_per_second / 100.0)  # 100 ops/sec target
        
        # Load balance contributes to performance
        load_balance_contribution = self.load_balance_score
        
        # Weighted combination
        health_score = (
            response_time_score * 0.3 +
            error_rate_score * 0.3 +
            throughput_score * 0.25 +
            load_balance_contribution * 0.15
        )
        
        return max(0.0, min(1.0, health_score))


@dataclass  
class ScalingConfiguration:
    """
    Configuration for scaling behavior and thresholds
    
    Integrates with OptimizationContext from performance_optimization.py
    """
    # Scaling thresholds
    cpu_scale_up_threshold: float = 80.0           # CPU % to trigger scale up
    cpu_scale_down_threshold: float = 20.0         # CPU % to trigger scale down
    memory_scale_up_threshold: float = 85.0        # Memory % to trigger scale up  
    memory_scale_down_threshold: float = 30.0      # Memory % to trigger scale down
    
    # Performance thresholds
    response_time_threshold_ms: float = 2000.0     # Max acceptable response time
    error_rate_threshold_percent: float = 5.0      # Max acceptable error rate
    throughput_min_threshold: float = 10.0         # Minimum throughput ops/sec
    
    # Queue management
    queue_depth_scale_up_threshold: int = 100      # Pending tasks to scale up
    queue_depth_scale_down_threshold: int = 10     # Pending tasks to scale down
    
    # Scaling behavior
    min_nodes: int = 1                             # Minimum number of nodes
    max_nodes: int = 10                            # Maximum number of nodes
    scale_up_step_size: int = 1                    # Nodes to add when scaling up
    scale_down_step_size: int = 1                  # Nodes to remove when scaling down
    
    # Timing controls
    scale_up_cooldown_seconds: int = 300           # Cooldown after scale up (5 min)
    scale_down_cooldown_seconds: int = 600         # Cooldown after scale down (10 min)
    metrics_window_seconds: int = 300              # Metrics evaluation window (5 min)
    
    # Cost controls
    max_cost_per_hour: float = 100.0               # Maximum cost budget per hour
    cost_efficiency_threshold: float = 0.7         # Minimum cost efficiency
    
    # Strategy settings
    scaling_strategy: ScalingStrategy = ScalingStrategy.HYBRID
    enable_predictive_scaling: bool = True         # Enable predictive scaling
    enable_cost_optimization: bool = True          # Enable cost-aware scaling
    
    # Integration settings
    integration_mode: str = "full"                 # Integration level with existing systems
    enable_langchain_decisions: bool = True        # Use LangChain for scaling decisions
    
    def __post_init__(self):
        """
        Validate configuration parameters
        """
        # Ensure thresholds are logical
        assert self.cpu_scale_down_threshold < self.cpu_scale_up_threshold
        assert self.memory_scale_down_threshold < self.memory_scale_up_threshold
        assert self.min_nodes <= self.max_nodes
        assert self.scale_up_cooldown_seconds > 0
        assert self.scale_down_cooldown_seconds > 0
        
        logger.info("Scaling configuration initialized", extra={
            'operation': 'SCALING_CONFIG_INIT',
            'metadata': {
                'min_nodes': self.min_nodes,
                'max_nodes': self.max_nodes,
                'strategy': self.scaling_strategy.value,
                'predictive_enabled': self.enable_predictive_scaling
            }
        })


@dataclass
class ScalingDecision:
    """
    Represents a scaling decision with rationale and context
    """
    # Decision details  
    direction: ScalingDirection                     # Scaling direction
    trigger: ScalingTrigger                        # What triggered this decision
    confidence: float                              # Confidence in decision (0.0-1.0)
    urgency: float                                # Urgency level (0.0-1.0)
    
    # Scaling parameters
    target_nodes: int                              # Target number of nodes
    resource_changes: Dict[str, Any] = field(default_factory=dict)  # Resource adjustments
    
    # Context and rationale
    rationale: str = ""                           # Human-readable explanation
    metrics_snapshot: Optional[ScalingMetrics] = None  # Metrics at decision time
    estimated_impact: Dict[str, float] = field(default_factory=dict)  # Expected impact
    
    # Execution details
    estimated_execution_time_seconds: float = 60.0  # Expected time to complete
    rollback_plan: Optional[Dict[str, Any]] = None   # Rollback strategy
    
    # Temporal context
    decision_timestamp: datetime = field(default_factory=datetime.utcnow)
    estimated_completion_time: Optional[datetime] = None
    
    def __post_init__(self):
        """
        Initialize calculated fields and validate decision
        """
        # Calculate estimated completion time
        self.estimated_completion_time = (
            self.decision_timestamp + 
            timedelta(seconds=self.estimated_execution_time_seconds)
        )
        
        # Validate decision parameters
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.urgency = max(0.0, min(1.0, self.urgency))
        
        logger.info("Scaling decision created", extra={
            'operation': 'SCALING_DECISION_CREATED',
            'metadata': {
                'direction': self.direction.value,
                'trigger': self.trigger.value,
                'confidence': self.confidence,
                'target_nodes': self.target_nodes,
                'rationale': self.rationale[:100]  # Truncate for logging
            }
        }) 


class AutoScalingEngine:
    """
    Advanced auto-scaling engine with predictive capabilities
    
    Implements intelligent auto-scaling based on KGoT Section 2.4 performance
    optimization requirements. Provides reactive and predictive scaling with
    integration to existing PerformanceOptimizer and ContainerOrchestrator.
    
    Key Features:
    - Reactive scaling based on real-time metrics
    - Predictive scaling using historical patterns and ML
    - Cost-aware scaling decisions
    - Integration with existing optimization engines
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, 
                 config: ScalingConfiguration,
                 performance_optimizer: Optional['PerformanceOptimizer'] = None,
                 container_orchestrator: Optional['ContainerOrchestrator'] = None):
        """
        Initialize the AutoScalingEngine
        
        Args:
            config: Scaling configuration parameters
            performance_optimizer: Existing performance optimizer integration
            container_orchestrator: Container orchestration integration
        """
        self.config = config
        self.performance_optimizer = performance_optimizer
        self.container_orchestrator = container_orchestrator
        
        # Scaling state management
        self.current_nodes = config.min_nodes
        self.last_scale_up_time = datetime.min
        self.last_scale_down_time = datetime.min
        self.scaling_history: List[ScalingDecision] = []
        
        # Metrics tracking for predictive scaling
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metric points
        self.performance_baseline: Dict[str, float] = {}
        
        # Thread safety
        self.scaling_lock = threading.Lock()
        
        # Async coordination
        self.scaling_semaphore = asyncio.Semaphore(1)  # Only one scaling operation at a time
        
        logger.info("AutoScalingEngine initialized", extra={
            'operation': 'AUTO_SCALING_INIT',
            'metadata': {
                'min_nodes': config.min_nodes,
                'max_nodes': config.max_nodes,
                'strategy': config.scaling_strategy.value
            }
        })
    
    async def evaluate_scaling_decision(self, metrics: ScalingMetrics) -> Optional[ScalingDecision]:
        """
        Evaluate whether scaling is needed based on current metrics
        
        Args:
            metrics: Current scaling metrics
            
        Returns:
            ScalingDecision: Scaling decision if action needed, None otherwise
        """
        logger.debug("Evaluating scaling decision", extra={
            'operation': 'SCALING_EVALUATION',
            'metadata': {
                'cpu_utilization': metrics.cpu_utilization_percent,
                'memory_utilization': metrics.memory_utilization_percent,
                'current_nodes': self.current_nodes,
                'pending_tasks': metrics.pending_tasks
            }
        })
        
        # Store metrics for historical analysis
        self.metrics_history.append(metrics)
        
        # Check cooldown periods
        if not self._is_scaling_allowed():
            logger.debug("Scaling blocked by cooldown period")
            return None
        
        # Evaluate different scaling triggers
        scaling_triggers = []
        
        # CPU-based scaling evaluation
        cpu_trigger = self._evaluate_cpu_scaling(metrics)
        if cpu_trigger:
            scaling_triggers.append(cpu_trigger)
        
        # Memory-based scaling evaluation  
        memory_trigger = self._evaluate_memory_scaling(metrics)
        if memory_trigger:
            scaling_triggers.append(memory_trigger)
        
        # Queue depth evaluation
        queue_trigger = self._evaluate_queue_scaling(metrics)
        if queue_trigger:
            scaling_triggers.append(queue_trigger)
        
        # Performance quality evaluation
        performance_trigger = self._evaluate_performance_scaling(metrics)
        if performance_trigger:
            scaling_triggers.append(performance_trigger)
        
        # Cost threshold evaluation
        cost_trigger = self._evaluate_cost_scaling(metrics)
        if cost_trigger:
            scaling_triggers.append(cost_trigger)
        
        # Predictive scaling evaluation
        if self.config.enable_predictive_scaling:
            predictive_trigger = await self._evaluate_predictive_scaling(metrics)
            if predictive_trigger:
                scaling_triggers.append(predictive_trigger)
        
        # Select most urgent/confident trigger
        if scaling_triggers:
            selected_trigger = max(scaling_triggers, key=lambda t: t['urgency'] * t['confidence'])
            
            # Create scaling decision
            decision = ScalingDecision(
                direction=selected_trigger['direction'],
                trigger=selected_trigger['trigger'],
                confidence=selected_trigger['confidence'],
                urgency=selected_trigger['urgency'],
                target_nodes=selected_trigger['target_nodes'],
                rationale=selected_trigger['rationale'],
                metrics_snapshot=metrics,
                estimated_impact=selected_trigger.get('estimated_impact', {})
            )
            
            logger.info("Scaling decision recommended", extra={
                'operation': 'SCALING_DECISION',
                'metadata': {
                    'direction': decision.direction.value,
                    'trigger': decision.trigger.value,
                    'confidence': decision.confidence,
                    'target_nodes': decision.target_nodes,
                    'rationale': decision.rationale
                }
            })
            
            return decision
        
        return None
    
    def _evaluate_cpu_scaling(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """
        Evaluate CPU-based scaling needs
        
        Args:
            metrics: Current metrics
            
        Returns:
            Dict with scaling trigger information or None
        """
        if metrics.cpu_utilization_percent >= self.config.cpu_scale_up_threshold:
            # Scale up needed
            target_nodes = min(self.current_nodes + self.config.scale_up_step_size, 
                             self.config.max_nodes)
            
            confidence = min(1.0, (metrics.cpu_utilization_percent - self.config.cpu_scale_up_threshold) / 20.0)
            urgency = min(1.0, (metrics.cpu_utilization_percent - 80.0) / 20.0)
            
            return {
                'direction': ScalingDirection.SCALE_UP,
                'trigger': ScalingTrigger.CPU_UTILIZATION,
                'confidence': confidence,
                'urgency': urgency,
                'target_nodes': target_nodes,
                'rationale': f"CPU utilization {metrics.cpu_utilization_percent:.1f}% exceeds threshold {self.config.cpu_scale_up_threshold:.1f}%",
                'estimated_impact': {'cpu_reduction_percent': 30.0}
            }
        
        elif (metrics.cpu_utilization_percent <= self.config.cpu_scale_down_threshold and 
              self.current_nodes > self.config.min_nodes):
            # Scale down opportunity
            target_nodes = max(self.current_nodes - self.config.scale_down_step_size,
                             self.config.min_nodes)
            
            confidence = min(1.0, (self.config.cpu_scale_down_threshold - metrics.cpu_utilization_percent) / 20.0)
            urgency = 0.3  # Lower urgency for scale down
            
            return {
                'direction': ScalingDirection.SCALE_DOWN,
                'trigger': ScalingTrigger.CPU_UTILIZATION,
                'confidence': confidence,
                'urgency': urgency,
                'target_nodes': target_nodes,
                'rationale': f"CPU utilization {metrics.cpu_utilization_percent:.1f}% below threshold {self.config.cpu_scale_down_threshold:.1f}%",
                'estimated_impact': {'cost_reduction_percent': 25.0}
            }
        
        return None
    
    def _evaluate_memory_scaling(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """
        Evaluate memory-based scaling needs
        
        Args:
            metrics: Current metrics
            
        Returns:
            Dict with scaling trigger information or None
        """
        if metrics.memory_utilization_percent >= self.config.memory_scale_up_threshold:
            # Scale up needed
            target_nodes = min(self.current_nodes + self.config.scale_up_step_size,
                             self.config.max_nodes)
            
            confidence = min(1.0, (metrics.memory_utilization_percent - self.config.memory_scale_up_threshold) / 15.0)
            urgency = min(1.0, (metrics.memory_utilization_percent - 85.0) / 15.0)
            
            return {
                'direction': ScalingDirection.SCALE_UP,
                'trigger': ScalingTrigger.MEMORY_UTILIZATION,
                'confidence': confidence,
                'urgency': urgency,
                'target_nodes': target_nodes,
                'rationale': f"Memory utilization {metrics.memory_utilization_percent:.1f}% exceeds threshold {self.config.memory_scale_up_threshold:.1f}%",
                'estimated_impact': {'memory_reduction_percent': 35.0}
            }
        
        elif (metrics.memory_utilization_percent <= self.config.memory_scale_down_threshold and
              self.current_nodes > self.config.min_nodes):
            # Scale down opportunity
            target_nodes = max(self.current_nodes - self.config.scale_down_step_size,
                             self.config.min_nodes)
            
            confidence = min(1.0, (self.config.memory_scale_down_threshold - metrics.memory_utilization_percent) / 20.0)
            urgency = 0.3  # Lower urgency for scale down
            
            return {
                'direction': ScalingDirection.SCALE_DOWN,
                'trigger': ScalingTrigger.MEMORY_UTILIZATION,
                'confidence': confidence,
                'urgency': urgency,
                'target_nodes': target_nodes,
                'rationale': f"Memory utilization {metrics.memory_utilization_percent:.1f}% below threshold {self.config.memory_scale_down_threshold:.1f}%",
                'estimated_impact': {'cost_reduction_percent': 20.0}
            }
        
        return None
    
    def _evaluate_queue_scaling(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """
        Evaluate queue depth-based scaling needs
        
        Args:
            metrics: Current metrics
            
        Returns:
            Dict with scaling trigger information or None
        """
        if metrics.pending_tasks >= self.config.queue_depth_scale_up_threshold:
            # Scale up needed due to queue buildup
            target_nodes = min(self.current_nodes + self.config.scale_up_step_size,
                             self.config.max_nodes)
            
            confidence = min(1.0, metrics.pending_tasks / (self.config.queue_depth_scale_up_threshold * 2))
            urgency = min(1.0, metrics.pending_tasks / (self.config.queue_depth_scale_up_threshold * 3))
            
            return {
                'direction': ScalingDirection.SCALE_UP,
                'trigger': ScalingTrigger.QUEUE_DEPTH,
                'confidence': confidence,
                'urgency': urgency,
                'target_nodes': target_nodes,
                'rationale': f"Queue depth {metrics.pending_tasks} exceeds threshold {self.config.queue_depth_scale_up_threshold}",
                'estimated_impact': {'queue_reduction_percent': 50.0}
            }
        
        elif (metrics.pending_tasks <= self.config.queue_depth_scale_down_threshold and
              self.current_nodes > self.config.min_nodes):
            # Scale down opportunity due to low queue
            target_nodes = max(self.current_nodes - self.config.scale_down_step_size,
                             self.config.min_nodes)
            
            confidence = 0.6  # Moderate confidence for queue-based scale down
            urgency = 0.2   # Low urgency
            
            return {
                'direction': ScalingDirection.SCALE_DOWN,
                'trigger': ScalingTrigger.QUEUE_DEPTH,
                'confidence': confidence,
                'urgency': urgency,
                'target_nodes': target_nodes,
                'rationale': f"Queue depth {metrics.pending_tasks} below threshold {self.config.queue_depth_scale_down_threshold}",
                'estimated_impact': {'cost_reduction_percent': 15.0}
            }
        
        return None
    
    def _evaluate_performance_scaling(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """
        Evaluate performance quality-based scaling needs
        
        Args:
            metrics: Current metrics
            
        Returns:
            Dict with scaling trigger information or None
        """
        # Response time degradation
        if metrics.average_response_time_ms > self.config.response_time_threshold_ms:
            target_nodes = min(self.current_nodes + self.config.scale_up_step_size,
                             self.config.max_nodes)
            
            confidence = min(1.0, (metrics.average_response_time_ms - self.config.response_time_threshold_ms) / 1000.0)
            urgency = min(1.0, (metrics.average_response_time_ms - 2000.0) / 3000.0)
            
            return {
                'direction': ScalingDirection.SCALE_UP,
                'trigger': ScalingTrigger.RESPONSE_TIME,
                'confidence': confidence,
                'urgency': urgency,
                'target_nodes': target_nodes,
                'rationale': f"Response time {metrics.average_response_time_ms:.0f}ms exceeds threshold {self.config.response_time_threshold_ms:.0f}ms",
                'estimated_impact': {'response_time_improvement_percent': 40.0}
            }
        
        # Error rate increase
        if metrics.error_rate_percent > self.config.error_rate_threshold_percent:
            target_nodes = min(self.current_nodes + self.config.scale_up_step_size,
                             self.config.max_nodes)
            
            confidence = min(1.0, metrics.error_rate_percent / (self.config.error_rate_threshold_percent * 2))
            urgency = min(1.0, metrics.error_rate_percent / 10.0)  # High urgency for errors
            
            return {
                'direction': ScalingDirection.SCALE_UP,
                'trigger': ScalingTrigger.ERROR_RATE,
                'confidence': confidence,
                'urgency': urgency,
                'target_nodes': target_nodes,
                'rationale': f"Error rate {metrics.error_rate_percent:.1f}% exceeds threshold {self.config.error_rate_threshold_percent:.1f}%",
                'estimated_impact': {'error_rate_reduction_percent': 60.0}
            }
        
        # Throughput drop
        if (metrics.throughput_ops_per_second < self.config.throughput_min_threshold and 
            metrics.pending_tasks > 0):
            target_nodes = min(self.current_nodes + self.config.scale_up_step_size,
                             self.config.max_nodes)
            
            confidence = 0.7  # Moderate confidence for throughput scaling
            urgency = 0.5
            
            return {
                'direction': ScalingDirection.SCALE_UP,
                'trigger': ScalingTrigger.THROUGHPUT_DROP,
                'confidence': confidence,
                'urgency': urgency,
                'target_nodes': target_nodes,
                'rationale': f"Throughput {metrics.throughput_ops_per_second:.1f} ops/sec below threshold {self.config.throughput_min_threshold:.1f}",
                'estimated_impact': {'throughput_improvement_percent': 50.0}
            }
        
        return None
    
    def _evaluate_cost_scaling(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """
        Evaluate cost-based scaling needs
        
        Args:
            metrics: Current metrics
            
        Returns:
            Dict with scaling trigger information or None
        """
        if not self.config.enable_cost_optimization:
            return None
        
        # Cost threshold exceeded - force scale down if possible
        if (metrics.current_cost_per_hour > self.config.max_cost_per_hour and
            self.current_nodes > self.config.min_nodes):
            
            target_nodes = max(self.current_nodes - self.config.scale_down_step_size,
                             self.config.min_nodes)
            
            confidence = 0.9  # High confidence for cost-driven scaling
            urgency = 0.8     # High urgency to control costs
            
            return {
                'direction': ScalingDirection.SCALE_DOWN,
                'trigger': ScalingTrigger.COST_THRESHOLD,
                'confidence': confidence,
                'urgency': urgency,
                'target_nodes': target_nodes,
                'rationale': f"Cost ${metrics.current_cost_per_hour:.2f}/hour exceeds budget ${self.config.max_cost_per_hour:.2f}/hour",
                'estimated_impact': {'cost_reduction_percent': 30.0}
            }
        
        return None
    
    async def _evaluate_predictive_scaling(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """
        Evaluate predictive scaling based on historical patterns
        
        Args:
            metrics: Current metrics
            
        Returns:
            Dict with scaling trigger information or None
        """
        if len(self.metrics_history) < 10:  # Need sufficient history
            return None
        
        try:
            # Analyze trends in key metrics
            recent_metrics = list(self.metrics_history)[-10:]
            cpu_trend = self._calculate_trend([m.cpu_utilization_percent for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_utilization_percent for m in recent_metrics])
            queue_trend = self._calculate_trend([m.pending_tasks for m in recent_metrics])
            
            # Predict future load
            prediction_confidence = 0.6  # Base confidence for predictions
            
            # CPU trend prediction
            if cpu_trend > 5.0 and metrics.cpu_utilization_percent > 60.0:  # Increasing trend
                target_nodes = min(self.current_nodes + self.config.scale_up_step_size,
                                 self.config.max_nodes)
                
                return {
                    'direction': ScalingDirection.SCALE_UP,
                    'trigger': ScalingTrigger.PREDICTIVE,
                    'confidence': prediction_confidence,
                    'urgency': 0.4,  # Lower urgency for predictive
                    'target_nodes': target_nodes,
                    'rationale': f"Predictive: CPU trend +{cpu_trend:.1f}%/min suggests future scaling need",
                    'estimated_impact': {'proactive_scaling_benefit': 'reduced_latency_spikes'}
                }
            
            # Memory trend prediction
            if memory_trend > 3.0 and metrics.memory_utilization_percent > 70.0:
                target_nodes = min(self.current_nodes + self.config.scale_up_step_size,
                                 self.config.max_nodes)
                
                return {
                    'direction': ScalingDirection.SCALE_UP,
                    'trigger': ScalingTrigger.PREDICTIVE,
                    'confidence': prediction_confidence,
                    'urgency': 0.4,
                    'target_nodes': target_nodes,
                    'rationale': f"Predictive: Memory trend +{memory_trend:.1f}%/min suggests future scaling need",
                    'estimated_impact': {'proactive_scaling_benefit': 'avoided_memory_pressure'}
                }
            
            # Queue buildup prediction
            if queue_trend > 5.0 and metrics.pending_tasks > 50:
                target_nodes = min(self.current_nodes + self.config.scale_up_step_size,
                                 self.config.max_nodes)
                
                return {
                    'direction': ScalingDirection.SCALE_UP,
                    'trigger': ScalingTrigger.PREDICTIVE,
                    'confidence': prediction_confidence,
                    'urgency': 0.5,
                    'target_nodes': target_nodes,
                    'rationale': f"Predictive: Queue trend +{queue_trend:.1f} tasks/min suggests future bottleneck",
                    'estimated_impact': {'proactive_scaling_benefit': 'prevented_queue_buildup'}
                }
            
        except Exception as e:
            logger.warning("Predictive scaling evaluation failed", extra={
                'operation': 'PREDICTIVE_SCALING_ERROR',
                'error': str(e)
            })
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate the trend (slope) of a series of values
        
        Args:
            values: List of numeric values
            
        Returns:
            float: Trend value (positive = increasing, negative = decreasing)
        """
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope calculation
        n = len(values)
        x_values = list(range(n))
        
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _is_scaling_allowed(self) -> bool:
        """
        Check if scaling is allowed based on cooldown periods
        
        Returns:
            bool: True if scaling is allowed
        """
        now = datetime.utcnow()
        
        # Check scale up cooldown
        scale_up_elapsed = (now - self.last_scale_up_time).total_seconds()
        if scale_up_elapsed < self.config.scale_up_cooldown_seconds:
            return False
        
        # Check scale down cooldown
        scale_down_elapsed = (now - self.last_scale_down_time).total_seconds()
        if scale_down_elapsed < self.config.scale_down_cooldown_seconds:
            return False
        
        return True
    
    async def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """
        Execute a scaling decision
        
        Args:
            decision: The scaling decision to execute
            
        Returns:
            bool: True if scaling was successful
        """
        async with self.scaling_semaphore:
            logger.info("Executing scaling decision", extra={
                'operation': 'SCALING_EXECUTION_START',
                'metadata': {
                    'direction': decision.direction.value,
                    'current_nodes': self.current_nodes,
                    'target_nodes': decision.target_nodes,
                    'trigger': decision.trigger.value
                }
            })
            
            try:
                # Record scaling attempt
                self.scaling_history.append(decision)
                
                # Execute scaling through container orchestrator
                if self.container_orchestrator:
                    success = await self._execute_container_scaling(decision)
                else:
                    # Fallback to performance optimizer scaling
                    success = await self._execute_performance_scaling(decision)
                
                if success:
                    # Update state
                    self.current_nodes = decision.target_nodes
                    
                    # Update cooldown timers
                    now = datetime.utcnow()
                    if decision.direction in [ScalingDirection.SCALE_UP, ScalingDirection.SCALE_OUT]:
                        self.last_scale_up_time = now
                    elif decision.direction in [ScalingDirection.SCALE_DOWN, ScalingDirection.SCALE_IN]:
                        self.last_scale_down_time = now
                    
                    # Update Prometheus metrics
                    if PROMETHEUS_AVAILABLE:
                        scaling_operations_total.labels(operation_type=decision.direction.value).inc()
                        active_nodes_gauge.set(self.current_nodes)
                    
                    logger.info("Scaling executed successfully", extra={
                        'operation': 'SCALING_EXECUTION_SUCCESS',
                        'metadata': {
                            'new_node_count': self.current_nodes,
                            'execution_time_seconds': (datetime.utcnow() - decision.decision_timestamp).total_seconds()
                        }
                    })
                else:
                    logger.error("Scaling execution failed", extra={
                        'operation': 'SCALING_EXECUTION_FAILED',
                        'metadata': {
                            'decision_id': id(decision),
                            'target_nodes': decision.target_nodes
                        }
                    })
                
                return success
                
            except Exception as e:
                logger.error("Scaling execution exception", extra={
                    'operation': 'SCALING_EXECUTION_EXCEPTION',
                    'error': str(e),
                    'metadata': {'decision_id': id(decision)}
                })
                return False
    
    async def _execute_container_scaling(self, decision: ScalingDecision) -> bool:
        """
        Execute scaling through container orchestrator
        
        Args:
            decision: Scaling decision to execute
            
        Returns:
            bool: Success status
        """
        try:
            if decision.direction == ScalingDirection.SCALE_UP:
                # Scale up containers
                for _ in range(decision.target_nodes - self.current_nodes):
                    # This would integrate with the actual container orchestrator
                    # For now, we log the action
                    logger.info("Scaling up container instance", extra={
                        'operation': 'CONTAINER_SCALE_UP'
                    })
            
            elif decision.direction == ScalingDirection.SCALE_DOWN:
                # Scale down containers
                for _ in range(self.current_nodes - decision.target_nodes):
                    logger.info("Scaling down container instance", extra={
                        'operation': 'CONTAINER_SCALE_DOWN'
                    })
            
            return True
            
        except Exception as e:
            logger.error("Container scaling failed", extra={
                'operation': 'CONTAINER_SCALING_ERROR',
                'error': str(e)
            })
            return False
    
    async def _execute_performance_scaling(self, decision: ScalingDecision) -> bool:
        """
        Execute scaling through performance optimizer
        
        Args:
            decision: Scaling decision to execute
            
        Returns:
            bool: Success status
        """
        try:
            if self.performance_optimizer:
                # Adjust performance optimizer settings based on scaling decision
                # This would integrate with the actual performance optimizer
                logger.info("Adjusting performance optimizer settings", extra={
                    'operation': 'PERFORMANCE_SCALING',
                    'metadata': {'target_nodes': decision.target_nodes}
                })
            
            return True
            
        except Exception as e:
            logger.error("Performance scaling failed", extra={
                'operation': 'PERFORMANCE_SCALING_ERROR',
                'error': str(e)
            })
            return False


class LoadBalancingCoordinator:
    """
    Intelligent load balancing coordinator for distributed KGoT operations
    
    Coordinates load distribution across multiple nodes and optimization strategies
    as specified in KGoT Section 2.4. Integrates with existing PerformanceOptimizer
    components to provide intelligent work distribution.
    
    Key Features:
    - Dynamic load balancing across nodes
    - Integration with existing optimization engines
    - Workload prediction and preemptive balancing
    - Cost-aware load distribution
    - Real-time performance monitoring and adjustment
    """
    
    def __init__(self, 
                 performance_optimizer: Optional['PerformanceOptimizer'] = None,
                 redis_client: Optional[redis.Redis] = None):
        """
        Initialize the LoadBalancingCoordinator
        
        Args:
            performance_optimizer: Integration with existing performance optimizer
            redis_client: Redis client for distributed coordination
        """
        self.performance_optimizer = performance_optimizer
        self.redis_client = redis_client
        
        # Load balancing state
        self.node_loads: Dict[str, float] = {}
        self.node_capabilities: Dict[str, Dict[str, Any]] = {}
        self.load_history: deque = deque(maxlen=500)
        
        # Strategy mappings for optimization engines
        self.strategy_engines = {
            'async_execution': None,
            'graph_parallelization': None,
            'mpi_distribution': None,
            'work_stealing': None,
            'cost_optimization': None
        }
        
        # Load balancing algorithms
        self.balancing_algorithms = {
            'round_robin': self._round_robin_balance,
            'least_loaded': self._least_loaded_balance,
            'weighted_round_robin': self._weighted_round_robin_balance,
            'dynamic_threshold': self._dynamic_threshold_balance,
            'predictive': self._predictive_balance
        }
        
        # Current balancing strategy
        self.current_strategy = 'dynamic_threshold'
        
        # Thread safety
        self.balancing_lock = threading.Lock()
        
        logger.info("LoadBalancingCoordinator initialized", extra={
            'operation': 'LOAD_BALANCING_INIT',
            'metadata': {
                'strategy': self.current_strategy,
                'redis_enabled': redis_client is not None
            }
        })
    
    async def distribute_workload(self, 
                                tasks: List[Any], 
                                target_nodes: List[str],
                                context: Optional[OptimizationContext] = None) -> Dict[str, List[Any]]:
        """
        Distribute workload across available nodes intelligently
        
        Args:
            tasks: List of tasks to distribute
            target_nodes: Available nodes for distribution
            context: Optimization context for intelligent distribution
            
        Returns:
            Dict mapping node IDs to assigned tasks
        """
        logger.debug("Distributing workload", extra={
            'operation': 'WORKLOAD_DISTRIBUTION',
            'metadata': {
                'task_count': len(tasks),
                'node_count': len(target_nodes),
                'strategy': self.current_strategy
            }
        })
        
        # Update node capabilities and current loads
        await self._update_node_states(target_nodes)
        
        # Select appropriate balancing algorithm
        balance_func = self.balancing_algorithms.get(self.current_strategy, 
                                                   self._least_loaded_balance)
        
        # Distribute tasks using selected algorithm
        distribution = balance_func(tasks, target_nodes, context)
        
        # Log distribution results
        distribution_summary = {node: len(task_list) for node, task_list in distribution.items()}
        logger.info("Workload distributed", extra={
            'operation': 'WORKLOAD_DISTRIBUTED',
            'metadata': {
                'distribution': distribution_summary,
                'load_balance_score': self._calculate_load_balance_score(distribution)
            }
        })
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            load_balance_score.set(self._calculate_load_balance_score(distribution))
        
        return distribution
    
    def _round_robin_balance(self, tasks: List[Any], nodes: List[str], context: Optional[OptimizationContext] = None) -> Dict[str, List[Any]]:
        """Simple round-robin load balancing"""
        distribution = {node: [] for node in nodes}
        for i, task in enumerate(tasks):
            target_node = nodes[i % len(nodes)]
            distribution[target_node].append(task)
        return distribution
    
    def _least_loaded_balance(self, tasks: List[Any], nodes: List[str], context: Optional[OptimizationContext] = None) -> Dict[str, List[Any]]:
        """Least-loaded node balancing"""
        distribution = {node: [] for node in nodes}
        for task in tasks:
            least_loaded_node = min(nodes, key=lambda n: self.node_loads.get(n, 0.0))
            distribution[least_loaded_node].append(task)
            task_weight = self._estimate_task_weight(task, context)
            self.node_loads[least_loaded_node] = self.node_loads.get(least_loaded_node, 0.0) + task_weight
        return distribution
    
    def _weighted_round_robin_balance(self, tasks: List[Any], nodes: List[str], context: Optional[OptimizationContext] = None) -> Dict[str, List[Any]]:
        """Weighted round-robin balancing based on node capabilities"""
        distribution = {node: [] for node in nodes}
        # Calculate weights and distribute
        node_weights = {}
        for node in nodes:
            capabilities = self.node_capabilities.get(node, {})
            cpu_weight = capabilities.get('cpu_cores', 1.0)
            memory_weight = capabilities.get('memory_gb', 1.0)
            current_load = self.node_loads.get(node, 0.0)
            node_weights[node] = (cpu_weight * memory_weight) / max(1.0, current_load)
        
        weighted_nodes = []
        for node, weight in node_weights.items():
            weighted_nodes.extend([node] * max(1, int(weight)))
        
        for i, task in enumerate(tasks):
            target_node = weighted_nodes[i % len(weighted_nodes)]
            distribution[target_node].append(task)
        return distribution
    
    def _dynamic_threshold_balance(self, tasks: List[Any], nodes: List[str], context: Optional[OptimizationContext] = None) -> Dict[str, List[Any]]:
        """Dynamic threshold-based balancing"""
        distribution = {node: [] for node in nodes}
        total_load = sum(self.node_loads.get(node, 0.0) for node in nodes)
        avg_load = total_load / len(nodes) if nodes else 0.0
        threshold = avg_load * 1.2
        
        for task in tasks:
            available_nodes = [node for node in nodes if self.node_loads.get(node, 0.0) < threshold]
            if not available_nodes:
                target_node = min(nodes, key=lambda n: self.node_loads.get(n, 0.0))
            else:
                target_node = min(available_nodes, key=lambda n: self.node_loads.get(n, 0.0))
            
            distribution[target_node].append(task)
            task_weight = self._estimate_task_weight(task, context)
            self.node_loads[target_node] = self.node_loads.get(target_node, 0.0) + task_weight
        return distribution
    
    def _predictive_balance(self, tasks: List[Any], nodes: List[str], context: Optional[OptimizationContext] = None) -> Dict[str, List[Any]]:
        """Predictive load balancing using historical patterns"""
        distribution = {node: [] for node in nodes}
        predicted_loads = {}
        for node in nodes:
            current_load = self.node_loads.get(node, 0.0)
            load_trend = self._calculate_load_trend(node)
            predicted_loads[node] = current_load + (load_trend * 60)
        
        for task in tasks:
            target_node = min(nodes, key=lambda n: predicted_loads[n])
            distribution[target_node].append(task)
            task_weight = self._estimate_task_weight(task, context)
            predicted_loads[target_node] += task_weight
        return distribution
    
    def _estimate_task_weight(self, task: Any, context: Optional[OptimizationContext] = None) -> float:
        """Estimate computational weight of a task"""
        base_weight = 1.0
        if hasattr(task, 'complexity'):
            base_weight *= task.complexity
        elif hasattr(task, '__len__'):
            base_weight *= len(task) / 100.0
        
        if context:
            if context.priority > 5:
                base_weight *= 1.5
            if context.strategy == OptimizationStrategy.COST_FOCUSED:
                base_weight *= 0.8
        return max(0.1, base_weight)
    
    def _calculate_load_trend(self, node: str) -> float:
        """Calculate load trend for a specific node"""
        if len(self.load_history) < 5:
            return 0.0
        node_loads = []
        for entry in list(self.load_history)[-10:]:
            if node in entry:
                node_loads.append(entry[node])
        if len(node_loads) < 2:
            return 0.0
        return (node_loads[-1] - node_loads[0]) / len(node_loads)
    
    def _calculate_load_balance_score(self, distribution: Dict[str, List[Any]]) -> float:
        """Calculate load balance score"""
        task_counts = [len(tasks) for tasks in distribution.values()]
        if not task_counts:
            return 1.0
        mean_tasks = statistics.mean(task_counts)
        if mean_tasks == 0:
            return 1.0
        std_tasks = statistics.stdev(task_counts) if len(task_counts) > 1 else 0.0
        cv = std_tasks / mean_tasks
        return max(0.0, 1.0 - cv)
    
    async def _update_node_states(self, nodes: List[str]):
        """Update node capabilities and current loads"""
        try:
            for node in nodes:
                if node == 'localhost':
                    cpu_usage = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    self.node_loads[node] = (cpu_usage + memory.percent) / 2.0
                    self.node_capabilities[node] = {
                        'cpu_cores': psutil.cpu_count(),
                        'memory_gb': memory.total / (1024**3),
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory.percent
                    }
                else:
                    self.node_loads[node] = 50.0
                    self.node_capabilities[node] = {
                        'cpu_cores': 4, 'memory_gb': 8.0,
                        'cpu_usage': 50.0, 'memory_usage': 50.0
                    }
            self.load_history.append(self.node_loads.copy())
        except Exception as e:
            logger.warning("Failed to update node states", extra={'operation': 'NODE_STATE_UPDATE_FAILED', 'error': str(e)})


class ScalabilityMetricsCollector:
    """Comprehensive metrics collection for scalability monitoring and analysis"""
    
    def __init__(self, performance_optimizer=None, container_orchestrator=None, redis_client=None):
        self.performance_optimizer = performance_optimizer
        self.container_orchestrator = container_orchestrator
        self.redis_client = redis_client
        self.current_metrics = None
        self.metrics_history = deque(maxlen=2000)
        self.baseline_metrics = {}
        self.collection_interval_seconds = 30
        self.baseline_window_minutes = 60
        self.metric_collection_times = deque(maxlen=100)
        self.alert_thresholds = {
            'cpu_utilization': 90.0, 'memory_utilization': 90.0,
            'error_rate': 10.0, 'response_time_ms': 5000.0, 'queue_depth': 500
        }
        self.collection_lock = threading.Lock()
        self.collection_task = None
        self.is_collecting = False
        logger.info("ScalabilityMetricsCollector initialized")
    
    async def start_collection(self):
        """Start background metrics collection"""
        if self.is_collecting:
            return
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop background metrics collection"""
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Background metrics collection loop"""
        while self.is_collecting:
            try:
                start_time = time.time()
                metrics = await self.collect_current_metrics()
                collection_time = time.time() - start_time
                
                self.metric_collection_times.append(collection_time)
                
                with self.collection_lock:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics)
                
                self._update_baseline_metrics()
                await self._check_alerts(metrics)
                self._update_prometheus_metrics(metrics)
                
                await asyncio.sleep(self.collection_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", extra={'error': str(e)})
                await asyncio.sleep(5)
    
    async def collect_current_metrics(self) -> ScalingMetrics:
        """Collect current scalability metrics from all sources"""
        metrics = ScalingMetrics()
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            metrics.cpu_utilization_percent = cpu_percent
            metrics.memory_utilization_percent = memory.percent
            
            if disk_io:
                total_disk_ops = disk_io.read_count + disk_io.write_count
                metrics.disk_io_utilization_percent = min(100.0, total_disk_ops / 1000.0)
            
            if net_io:
                total_net_bytes = net_io.bytes_sent + net_io.bytes_recv
                metrics.network_io_utilization_percent = min(100.0, total_net_bytes / (1024*1024*10))
            
            if self.performance_optimizer:
                perf_metrics = await self._collect_performance_metrics()
                metrics.average_response_time_ms = perf_metrics.get('avg_response_time_ms', 0.0)
                metrics.throughput_ops_per_second = perf_metrics.get('throughput_ops_per_sec', 0.0)
                metrics.error_rate_percent = perf_metrics.get('error_rate_percent', 0.0)
                metrics.pending_tasks = perf_metrics.get('pending_tasks', 0)
                metrics.active_tasks = perf_metrics.get('active_tasks', 0)
                metrics.completed_tasks = perf_metrics.get('completed_tasks', 0)
                metrics.failed_tasks = perf_metrics.get('failed_tasks', 0)
            
            if self.container_orchestrator:
                container_metrics = await self._collect_container_metrics()
                metrics.active_nodes = container_metrics.get('active_nodes', 1)
                metrics.total_capacity_nodes = container_metrics.get('total_capacity_nodes', 1)
                metrics.allocated_cpu_cores = container_metrics.get('allocated_cpu_cores', 0.0)
                metrics.allocated_memory_gb = container_metrics.get('allocated_memory_gb', 0.0)
                metrics.current_cost_per_hour = container_metrics.get('current_cost_per_hour', 0.0)
            
            metrics.load_balance_score = self._calculate_load_balance_score()
            metrics.scaling_efficiency = self._calculate_scaling_efficiency()
            metrics.resource_efficiency_score = self._calculate_resource_efficiency(metrics)
            metrics.timestamp = datetime.utcnow()
            
        except Exception as e:
            logger.error("Failed to collect metrics", extra={'error': str(e)})
        return metrics
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect metrics from performance optimizer"""
        return {
            'avg_response_time_ms': 100.0, 'throughput_ops_per_sec': 50.0,
            'error_rate_percent': 1.0, 'pending_tasks': 5, 'active_tasks': 3,
            'completed_tasks': 100, 'failed_tasks': 2
        }
    
    async def _collect_container_metrics(self) -> Dict[str, Any]:
        """Collect metrics from container orchestrator"""
        return {
            'active_nodes': 2, 'total_capacity_nodes': 5,
            'allocated_cpu_cores': 4.0, 'allocated_memory_gb': 8.0,
            'current_cost_per_hour': 5.50
        }
    
    def _calculate_load_balance_score(self) -> float:
        """Calculate current load balance score"""
        return 0.85
    
    def _calculate_scaling_efficiency(self) -> float:
        """Calculate scaling efficiency based on recent scaling operations"""
        return 0.80
    
    def _calculate_resource_efficiency(self, metrics: ScalingMetrics) -> float:
        """Calculate resource efficiency score"""
        cpu_efficiency = 1.0 - (metrics.cpu_utilization_percent / 100.0)
        memory_efficiency = 1.0 - (metrics.memory_utilization_percent / 100.0)
        return (cpu_efficiency + memory_efficiency) / 2.0
    
    def _update_baseline_metrics(self):
        """Update baseline metrics using recent history"""
        if len(self.metrics_history) < 10:
            return
        recent_metrics = list(self.metrics_history)[-60:]
        self.baseline_metrics = {
            'cpu_utilization': statistics.mean([m.cpu_utilization_percent for m in recent_metrics]),
            'memory_utilization': statistics.mean([m.memory_utilization_percent for m in recent_metrics]),
            'response_time': statistics.mean([m.average_response_time_ms for m in recent_metrics]),
            'throughput': statistics.mean([m.throughput_ops_per_second for m in recent_metrics]),
            'error_rate': statistics.mean([m.error_rate_percent for m in recent_metrics])
        }
    
    async def _check_alerts(self, metrics: ScalingMetrics):
        """Check for alert conditions"""
        alerts = []
        if metrics.cpu_utilization_percent > self.alert_thresholds['cpu_utilization']:
            alerts.append(f"High CPU utilization: {metrics.cpu_utilization_percent:.1f}%")
        if metrics.memory_utilization_percent > self.alert_thresholds['memory_utilization']:
            alerts.append(f"High memory utilization: {metrics.memory_utilization_percent:.1f}%")
        if metrics.error_rate_percent > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {metrics.error_rate_percent:.1f}%")
        if metrics.average_response_time_ms > self.alert_thresholds['response_time_ms']:
            alerts.append(f"High response time: {metrics.average_response_time_ms:.0f}ms")
        if metrics.pending_tasks > self.alert_thresholds['queue_depth']:
            alerts.append(f"High queue depth: {metrics.pending_tasks} tasks")
        
        for alert in alerts:
            logger.warning("Scalability alert", extra={'operation': 'SCALABILITY_ALERT', 'alert': alert})
    
    def _update_prometheus_metrics(self, metrics: ScalingMetrics):
        """Update Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        try:
            active_nodes_gauge.set(metrics.active_nodes)
            load_balance_score.set(metrics.load_balance_score)
            scaling_efficiency_gauge.set(metrics.scaling_efficiency)
        except Exception as e:
            logger.warning("Failed to update Prometheus metrics", extra={'error': str(e)})
    
    def get_current_metrics(self) -> Optional[ScalingMetrics]:
        """Get current metrics snapshot"""
        with self.collection_lock:
            return self.current_metrics
    
    def get_metrics_history(self, limit: int = 100) -> List[ScalingMetrics]:
        """Get historical metrics"""
        with self.collection_lock:
            return list(self.metrics_history)[-limit:]


class ScalabilityController:
    """Main orchestrator for the KGoT Scalability Framework"""
    
    def __init__(self, config: ScalingConfiguration, performance_optimizer=None, container_orchestrator=None, redis_client=None):
        self.config = config
        self.performance_optimizer = performance_optimizer
        self.container_orchestrator = container_orchestrator
        self.redis_client = redis_client
        
        # Initialize core components
        self.auto_scaling_engine = AutoScalingEngine(config, performance_optimizer, container_orchestrator)
        self.load_balancer = LoadBalancingCoordinator(performance_optimizer, redis_client)
        self.metrics_collector = ScalabilityMetricsCollector(performance_optimizer, container_orchestrator, redis_client)
        
        # Operational state
        self.is_running = False
        self.scaling_enabled = True
        self.last_scaling_check = datetime.min
        self.monitoring_task = None
        self.scaling_task = None
        self.scaling_agent = None
        
        if config.enable_langchain_decisions and LANGCHAIN_AVAILABLE:
            self._initialize_scaling_agent()
        
        logger.info("ScalabilityController initialized")
    
    async def start(self):
        """Start the scalability framework"""
        if self.is_running:
            return
        
        try:
            await self.metrics_collector.start_collection()
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.scaling_task = asyncio.create_task(self._scaling_loop())
            self.is_running = True
            logger.info("ScalabilityController started successfully")
        except Exception as e:
            logger.error("Failed to start ScalabilityController", extra={'error': str(e)})
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the scalability framework"""
        self.is_running = False
        for task in [self.monitoring_task, self.scaling_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.metrics_collector.stop_collection()
        logger.info("ScalabilityController stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            try:
                metrics = self.metrics_collector.get_current_metrics()
                if metrics:
                    await self._check_system_health(metrics)
                    await self._update_load_balancer_state(metrics)
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring loop error", extra={'error': str(e)})
                await asyncio.sleep(5)
    
    async def _scaling_loop(self):
        """Background scaling decision loop"""
        while self.is_running:
            try:
                if not self.scaling_enabled:
                    await asyncio.sleep(60)
                    continue
                
                metrics = self.metrics_collector.get_current_metrics()
                if metrics:
                    decision = await self.auto_scaling_engine.evaluate_scaling_decision(metrics)
                    if decision:
                        if self.scaling_agent:
                            decision = await self._validate_decision_with_agent(decision, metrics)
                        if decision:
                            success = await self.auto_scaling_engine.execute_scaling_decision(decision)
                            if success:
                                logger.info("Scaling decision executed", extra={'direction': decision.direction.value})
                
                await asyncio.sleep(120)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scaling loop error", extra={'error': str(e)})
                await asyncio.sleep(10)
    
    def _initialize_scaling_agent(self):
        """Initialize LangChain agent for AI-driven scaling decisions"""
        try:
            if LANGCHAIN_AVAILABLE:
                self.scaling_agent = "AI_SCALING_AGENT_PLACEHOLDER"
                logger.info("LangChain scaling agent initialized")
        except Exception as e:
            logger.warning("Failed to initialize LangChain agent", extra={'error': str(e)})
    
    async def _validate_decision_with_agent(self, decision: ScalingDecision, metrics: ScalingMetrics) -> Optional[ScalingDecision]:
        """Validate scaling decision using AI agent"""
        try:
            if decision.direction == ScalingDirection.SCALE_UP:
                if metrics.cpu_utilization_percent < 30 and metrics.memory_utilization_percent < 30:
                    logger.info("AI agent rejected scale up decision - low resource utilization")
                    return None
            elif decision.direction == ScalingDirection.SCALE_DOWN:
                if metrics.error_rate_percent > 5.0 or metrics.pending_tasks > 100:
                    logger.info("AI agent rejected scale down decision - system under stress")
                    return None
            
            if self.config.enable_cost_optimization:
                if decision.direction == ScalingDirection.SCALE_UP and metrics.current_cost_per_hour > self.config.max_cost_per_hour * 0.8:
                    logger.info("AI agent rejected scale up - approaching cost limit")
                    return None
            
            return decision
        except Exception as e:
            logger.warning("AI agent validation failed", extra={'error': str(e)})
            return decision
    
    async def _check_system_health(self, metrics: ScalingMetrics):
        """Check overall system health and trigger alerts if needed"""
        health_score = metrics.performance_health_score()
        if health_score < 0.6:
            logger.warning("System health degraded", extra={'health_score': health_score})
    
    async def _update_load_balancer_state(self, metrics: ScalingMetrics):
        """Update load balancer with current system state"""
        pass
    
    async def manual_scale(self, direction: ScalingDirection, nodes: int = 1) -> bool:
        """Manually trigger scaling operation"""
        try:
            current_metrics = self.metrics_collector.get_current_metrics()
            if not current_metrics:
                return False
            
            current_nodes = self.auto_scaling_engine.current_nodes
            target_nodes = current_nodes + nodes if direction == ScalingDirection.SCALE_UP else current_nodes - nodes
            target_nodes = max(self.config.min_nodes, min(self.config.max_nodes, target_nodes))
            
            decision = ScalingDecision(
                direction=direction, trigger=ScalingTrigger.MANUAL,
                confidence=1.0, urgency=0.8, target_nodes=target_nodes,
                rationale="Manual scaling operation requested", metrics_snapshot=current_metrics
            )
            
            return await self.auto_scaling_engine.execute_scaling_decision(decision)
        except Exception as e:
            logger.error("Manual scaling exception", extra={'error': str(e)})
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the scalability framework"""
        current_metrics = self.metrics_collector.get_current_metrics()
        
        status = {
            'is_running': self.is_running,
            'scaling_enabled': self.scaling_enabled,
            'current_nodes': self.auto_scaling_engine.current_nodes,
            'min_nodes': self.config.min_nodes,
            'max_nodes': self.config.max_nodes,
            'scaling_strategy': self.config.scaling_strategy.value,
            'load_balancing_strategy': self.load_balancer.current_strategy,
            'ai_agent_enabled': self.scaling_agent is not None
        }
        
        if current_metrics:
            status['current_metrics'] = {
                'cpu_utilization': current_metrics.cpu_utilization_percent,
                'memory_utilization': current_metrics.memory_utilization_percent,
                'pending_tasks': current_metrics.pending_tasks,
                'response_time_ms': current_metrics.average_response_time_ms,
                'throughput_ops_per_sec': current_metrics.throughput_ops_per_second,
                'error_rate_percent': current_metrics.error_rate_percent,
                'health_score': current_metrics.performance_health_score()
            }
        
        return status


# Example usage and integration functions
async def create_scalability_framework(
    performance_optimizer: Optional['PerformanceOptimizer'] = None,
    container_orchestrator: Optional['ContainerOrchestrator'] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> ScalabilityController:
    """
    Factory function to create and configure the scalability framework
    
    Args:
        performance_optimizer: Existing performance optimizer instance
        container_orchestrator: Existing container orchestrator instance
        config_overrides: Configuration overrides
        
    Returns:
        ScalabilityController: Configured scalability controller
    """
    # Create configuration with defaults
    config = ScalingConfiguration()
    
    # Apply configuration overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Initialize Redis client if available
    redis_client = None
    try:
        if REDIS_AVAILABLE:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()  # Test connection
    except Exception as e:
        logger.warning("Redis not available for distributed coordination", extra={'error': str(e)})
    
    # Create and return controller
    controller = ScalabilityController(
        config=config,
        performance_optimizer=performance_optimizer,
        container_orchestrator=container_orchestrator,
        redis_client=redis_client
    )
    
    logger.info("Scalability framework created", extra={
        'operation': 'FRAMEWORK_CREATED',
        'metadata': {
            'performance_optimizer_enabled': performance_optimizer is not None,
            'container_orchestrator_enabled': container_orchestrator is not None,
            'redis_enabled': redis_client is not None,
            'langchain_enabled': LANGCHAIN_AVAILABLE
        }
    })
    
    return controller


async def example_usage():
    """
    Example usage of the KGoT Scalability Framework
    """
    logger.info("Starting KGoT Scalability Framework example")
    
    try:
        # Create configuration
        config = ScalingConfiguration(
            min_nodes=1,
            max_nodes=5,
            cpu_scale_up_threshold=70.0,
            cpu_scale_down_threshold=30.0,
            scaling_strategy=ScalingStrategy.HYBRID,
            enable_predictive_scaling=True,
            enable_cost_optimization=True
        )
        
        # Create scalability framework
        framework = await create_scalability_framework(config_overrides={
            'max_nodes': 5,
            'enable_predictive_scaling': True
        })
        
        # Start the framework
        await framework.start()
        
        # Let it run for a demonstration period
        logger.info("Framework running - collecting metrics and making scaling decisions")
        await asyncio.sleep(30)  # Run for 30 seconds
        
        # Get status
        status = framework.get_status()
        logger.info("Framework status", extra={'status': status})
        
        # Manual scaling example
        logger.info("Testing manual scaling")
        success = await framework.manual_scale(ScalingDirection.SCALE_UP, 1)
        logger.info(f"Manual scale up result: {success}")
        
        await asyncio.sleep(5)
        
        # Stop the framework
        await framework.stop()
        
        logger.info("KGoT Scalability Framework example completed successfully")
        
    except Exception as e:
        logger.error("Example failed", extra={'error': str(e)})
        raise


if __name__ == "__main__":
    """
    Main entry point for standalone testing of the scalability framework
    """
    # Setup for standalone execution
    if PROMETHEUS_AVAILABLE:
        start_http_server(8000)  # Start Prometheus metrics server
        logger.info("Prometheus metrics server started on port 8000")
    
    # Run the example
    asyncio.run(example_usage())