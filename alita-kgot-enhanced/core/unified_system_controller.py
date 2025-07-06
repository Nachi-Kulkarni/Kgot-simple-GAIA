"""
Unified System Controller (Task 46)

This module implements the meta-orchestrator that coordinates between the Alita Manager Agent 
and KGoT Controller systems, providing intelligent task routing, shared state management, 
and dynamic load balancing.

Key Features:
- Sequential Thinking MCP integration for task routing
- Shared state management using Redis
- Performance monitoring and load balancing
- Circuit breaker patterns for resilience
- Comprehensive Winston logging

Architecture:
- Meta-orchestrator pattern: doesn't replace existing systems but coordinates them
- Dual-LLM architecture integration: respects both Alita and KGoT design patterns
- Event-driven coordination with fallback mechanisms

@module UnifiedSystemController
@author AI Assistant
@date 2025-01-22
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from uuid import uuid4

import redis
import httpx
from pydantic import BaseModel, Field

# Import configuration and logging [[memory:1383804]]
from ..config.logging.winston_config import get_logger

# Import enhanced shared state utilities
from .shared_state_utilities import (
    EnhancedSharedStateManager, 
    StateScope, 
    StateEventType,
    RealTimeStateStreamer,
    DistributedLockManager,
    StateAnalytics
)

# Import Sequential Thinking MCP integration
from .sequential_thinking_mcp_integration import SequentialThinkingMCPIntegration

# Import advanced monitoring system
from .advanced_monitoring_system import AdvancedMonitoringSystem

# Import load balancing system
from .load_balancing_system import (
    AdaptiveLoadBalancer, 
    SystemInstance, 
    CircuitBreakerConfig,
    LoadBalancingStrategy
)

# Create logger instance
logger = get_logger('unified_system_controller')


class TaskComplexity(Enum):
    """Task complexity enumeration for routing decisions"""
    SIMPLE = "simple"          # Score 1-3: Direct execution
    MODERATE = "moderate"      # Score 4-6: Single system
    COMPLEX = "complex"        # Score 7-8: Sequential thinking + single system
    HIGHLY_COMPLEX = "highly_complex"  # Score 9-10: Sequential thinking + hybrid approach


class RoutingStrategy(Enum):
    """System routing strategy enumeration"""
    ALITA_FIRST = "alita_first"       # Tool creation and web interaction focused
    KGOT_FIRST = "kgot_first"         # Knowledge graph and reasoning focused  
    HYBRID = "hybrid"                 # Multi-step workflow alternating systems
    PARALLEL = "parallel"             # Concurrent execution on both systems


class SystemStatus(Enum):
    """System status enumeration for load balancing"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILED = "failed"


@dataclass
class TaskContext:
    """Context information for task processing"""
    task_id: str
    description: str
    complexity_score: int
    routing_strategy: RoutingStrategy
    requires_mcp_creation: bool = False
    requires_knowledge_graph: bool = False
    requires_web_interaction: bool = False
    requires_code_generation: bool = False
    requires_complex_reasoning: bool = False
    data_complexity: str = "low"
    interaction_type: str = "single_domain"
    budget_constraints: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 300
    sequential_thinking_trace: Optional[List[Dict[str, Any]]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass 
class SystemMetrics:
    """Performance metrics for system monitoring"""
    system_name: str
    response_time_ms: float
    success_rate: float
    error_count: int
    load_percentage: float
    last_updated: datetime
    status: SystemStatus = SystemStatus.HEALTHY
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class ExecutionResult:
    """Result container for task execution"""
    task_id: str
    success: bool
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    execution_time_ms: float
    systems_used: List[str]
    routing_decisions: List[Dict[str, Any]]
    metrics: Dict[str, SystemMetrics]
    sequential_thinking_trace: Optional[List[Dict[str, Any]]] = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()


class LegacySharedStateManager:
    """
    Legacy Redis-based shared state management (replaced by EnhancedSharedStateManager)
    Kept for compatibility during transition
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 db: int = 0, max_connections: int = 20):
        """Initialize legacy shared state manager - use EnhancedSharedStateManager instead"""
        logger.warning("Using legacy SharedStateManager - consider upgrading to EnhancedSharedStateManager", extra={
            'operation': 'LEGACY_STATE_MANAGER_INIT'
        })
        self.redis_url = redis_url
        self.db = db
        self.max_connections = max_connections
        self._redis_pool = None
        self._connected = False
    
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve state from Redis
        
        Args:
            key: State key to retrieve
            
        Returns:
            State data or None if not found
        """
        if not self._connected:
            raise RuntimeError("SharedStateManager not connected")
        
        try:
            async with redis.Redis(connection_pool=self._redis_pool) as client:
                data = await client.get(key)
                if data:
                    return json.loads(data)
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving state for key {key}: {str(e)}", extra={
                'operation': 'SHARED_STATE_GET_ERROR',
                'key': key,
                'error': str(e)
            })
            raise
    
    async def set_state(self, key: str, value: Dict[str, Any], 
                       expire_seconds: Optional[int] = None) -> None:
        """
        Store state in Redis
        
        Args:
            key: State key to store
            value: State data to store
            expire_seconds: Optional expiration time
        """
        if not self._connected:
            raise RuntimeError("SharedStateManager not connected")
        
        try:
            async with redis.Redis(connection_pool=self._redis_pool) as client:
                serialized = json.dumps(value, default=str)
                await client.set(key, serialized, ex=expire_seconds)
                
                logger.debug(f"State stored for key: {key}", extra={
                    'operation': 'SHARED_STATE_SET',
                    'key': key,
                    'expire_seconds': expire_seconds
                })
                
        except Exception as e:
            logger.error(f"Error storing state for key {key}: {str(e)}", extra={
                'operation': 'SHARED_STATE_SET_ERROR',
                'key': key,
                'error': str(e)
            })
            raise
    
    async def get_available_mcps(self) -> List[Dict[str, Any]]:
        """Get list of available MCPs from registry"""
        mcps = await self.get_state("mcp_registry")
        return mcps.get("available_mcps", []) if mcps else []
    
    async def update_mcp_registry(self, mcp_data: Dict[str, Any]) -> None:
        """Update MCP registry with new or modified MCP"""
        registry = await self.get_state("mcp_registry") or {"available_mcps": []}
        
        # Update or add MCP
        mcp_name = mcp_data.get("name")
        mcps = registry["available_mcps"]
        
        # Find existing MCP or add new one
        updated = False
        for i, existing_mcp in enumerate(mcps):
            if existing_mcp.get("name") == mcp_name:
                mcps[i] = mcp_data
                updated = True
                break
        
        if not updated:
            mcps.append(mcp_data)
        
        registry["last_updated"] = datetime.now().isoformat()
        await self.set_state("mcp_registry", registry)
        
        logger.info(f"MCP registry updated for: {mcp_name}", extra={
            'operation': 'MCP_REGISTRY_UPDATE',
            'mcp_name': mcp_name,
            'total_mcps': len(mcps)
        })
    
    async def get_kgot_graph_state(self) -> Optional[Dict[str, Any]]:
        """Get current KGoT knowledge graph state"""
        return await self.get_state("kgot_graph_state")
    
    async def update_kgot_graph_state(self, graph_state: Dict[str, Any]) -> None:
        """Update KGoT knowledge graph state"""
        graph_state["last_updated"] = datetime.now().isoformat()
        await self.set_state("kgot_graph_state", graph_state, expire_seconds=3600)
        
        logger.debug("KGoT graph state updated", extra={
            'operation': 'KGOT_GRAPH_STATE_UPDATE',
            'graph_size': len(graph_state.get("vertices", {}))
        })
    
    async def get_task_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent task execution history"""
        history = await self.get_state("task_history")
        if history and "tasks" in history:
            return history["tasks"][-limit:]
        return []
    
    async def add_task_to_history(self, task_result: ExecutionResult) -> None:
        """Add completed task to execution history"""
        history = await self.get_state("task_history") or {"tasks": []}
        
        # Convert result to dict for storage
        task_data = asdict(task_result)
        history["tasks"].append(task_data)
        
        # Keep only last 1000 tasks
        if len(history["tasks"]) > 1000:
            history["tasks"] = history["tasks"][-1000:]
        
        history["last_updated"] = datetime.now().isoformat()
        await self.set_state("task_history", history, expire_seconds=86400)  # 24 hours
        
        logger.info(f"Task added to history: {task_result.task_id}", extra={
            'operation': 'TASK_HISTORY_ADD',
            'task_id': task_result.task_id,
            'success': task_result.success
        })
    
    async def get_system_budget(self) -> Dict[str, Any]:
        """Get current system budget and usage"""
        budget = await self.get_state("system_budget")
        return budget or {
            "total_budget": 100.0,
            "used_budget": 0.0,
            "remaining_budget": 100.0,
            "cost_tracking": {},
            "last_updated": datetime.now().isoformat()
        }
    
    async def update_budget_usage(self, cost_data: Dict[str, float]) -> None:
        """Update budget usage with task costs"""
        budget = await self.get_system_budget()
        
        total_cost = sum(cost_data.values())
        budget["used_budget"] += total_cost
        budget["remaining_budget"] = budget["total_budget"] - budget["used_budget"]
        
        # Update cost tracking
        for system, cost in cost_data.items():
            if system not in budget["cost_tracking"]:
                budget["cost_tracking"][system] = 0.0
            budget["cost_tracking"][system] += cost
        
        budget["last_updated"] = datetime.now().isoformat()
        await self.set_state("system_budget", budget)
        
        logger.info(f"Budget updated with cost: ${total_cost:.4f}", extra={
            'operation': 'BUDGET_UPDATE',
            'total_cost': total_cost,
            'remaining_budget': budget["remaining_budget"]
        })


class PerformanceMonitor:
    """
    Performance monitoring and metrics collection for system load balancing
    
    Tracks:
    - Response times and success rates
    - Error counts and patterns
    - Resource utilization
    - Circuit breaker states
    - System health indicators
    """
    
    def __init__(self, shared_state: EnhancedSharedStateManager, 
                 health_check_interval: int = 30):
        """
        Initialize performance monitor
        
        Args:
            shared_state: Shared state manager instance
            health_check_interval: Health check interval in seconds
        """
        self.shared_state = shared_state
        self.health_check_interval = health_check_interval
        self._monitoring_task = None
        self._running = False
        
        # System endpoints for health checks
        self.system_endpoints = {
            "alita_manager": "http://localhost:3000/health",
            "kgot_controller": "http://localhost:3003/health",
            "web_agent": "http://localhost:3001/health",
            "mcp_creation": "http://localhost:3002/health",
            "validation": "http://localhost:3004/health",
            "multimodal": "http://localhost:3005/health"
        }
        
        logger.info("PerformanceMonitor initialized", extra={
            'operation': 'PERFORMANCE_MONITOR_INIT',
            'health_check_interval': health_check_interval
        })
    
    async def start_monitoring(self) -> None:
        """Start background performance monitoring"""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitoring started", extra={
            'operation': 'PERFORMANCE_MONITOR_START'
        })
    
    async def stop_monitoring(self) -> None:
        """Stop background performance monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped", extra={
            'operation': 'PERFORMANCE_MONITOR_STOP'
        })
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}", extra={
                    'operation': 'PERFORMANCE_MONITOR_ERROR',
                    'error': str(e)
                })
                await asyncio.sleep(self.health_check_interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect metrics from all systems"""
        metrics = {}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for system_name, endpoint in self.system_endpoints.items():
                try:
                    start_time = time.time()
                    response = await client.get(endpoint)
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Determine system status based on response
                    if response.status_code == 200:
                        status = SystemStatus.HEALTHY
                        success_rate = 1.0
                        error_count = 0
                    else:
                        status = SystemStatus.DEGRADED
                        success_rate = 0.5
                        error_count = 1
                    
                    metrics[system_name] = SystemMetrics(
                        system_name=system_name,
                        response_time_ms=response_time,
                        success_rate=success_rate,
                        error_count=error_count,
                        load_percentage=response_time / 10.0,  # Simple load calculation
                        status=status,
                        last_updated=datetime.now()
                    )
                    
                except Exception as e:
                    # System is down or unreachable
                    metrics[system_name] = SystemMetrics(
                        system_name=system_name,
                        response_time_ms=float('inf'),
                        success_rate=0.0,
                        error_count=1,
                        load_percentage=100.0,
                        status=SystemStatus.FAILED,
                        last_updated=datetime.now()
                    )
                    
                    logger.warning(f"Health check failed for {system_name}: {str(e)}", extra={
                        'operation': 'HEALTH_CHECK_FAILED',
                        'system': system_name,
                        'error': str(e)
                    })
        
        # Store metrics in shared state
        metrics_data = {system: asdict(metric) for system, metric in metrics.items()}
        await self.shared_state.set_state("system_metrics", {
            "metrics": metrics_data,
            "collected_at": datetime.now().isoformat()
        }, expire_seconds=300)  # 5 minutes
        
        logger.debug("System metrics collected", extra={
            'operation': 'METRICS_COLLECTED',
            'systems_count': len(metrics),
            'healthy_systems': sum(1 for m in metrics.values() if m.status == SystemStatus.HEALTHY)
        })
    
    async def get_system_metrics(self, system_name: Optional[str] = None) -> Union[SystemMetrics, Dict[str, SystemMetrics]]:
        """
        Get current system metrics
        
        Args:
            system_name: Specific system name or None for all systems
            
        Returns:
            SystemMetrics for specific system or dict of all metrics
        """
        metrics_data = await self.shared_state.get_state("system_metrics")
        if not metrics_data or "metrics" not in metrics_data:
            return {} if system_name is None else None
        
        metrics = {}
        for name, data in metrics_data["metrics"].items():
            # Convert back to SystemMetrics object
            data["status"] = SystemStatus(data["status"])
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
            metrics[name] = SystemMetrics(**data)
        
        if system_name:
            return metrics.get(system_name)
        return metrics
    
    async def is_system_healthy(self, system_name: str) -> bool:
        """Check if a specific system is healthy"""
        metric = await self.get_system_metrics(system_name)
        return metric and metric.status in [SystemStatus.HEALTHY, SystemStatus.DEGRADED]
    
    async def get_least_loaded_system(self, systems: List[str]) -> Optional[str]:
        """Get the least loaded system from a list of options"""
        metrics = await self.get_system_metrics()
        if not metrics:
            return systems[0] if systems else None
        
        # Filter to available systems and sort by load
        available_systems = [(name, metrics[name]) for name in systems 
                           if name in metrics and metrics[name].status != SystemStatus.FAILED]
        
        if not available_systems:
            return None
        
        # Sort by load percentage (ascending)
        available_systems.sort(key=lambda x: x[1].load_percentage)
        return available_systems[0][0]


class UnifiedSystemController:
    """
    Meta-orchestrator that coordinates between Alita Manager Agent and KGoT Controller
    
    This controller acts as the single entry point for all tasks, providing:
    - Intelligent task routing using Sequential Thinking MCP
    - Shared state management across systems
    - Dynamic load balancing and circuit breaker patterns
    - Comprehensive error handling and fallback strategies
    - Performance monitoring and optimization
    
    Architecture Pattern: Meta-orchestrator
    - Doesn't replace existing systems but coordinates their interactions
    - Maintains instances of both Alita Manager Agent and KGoT Controller
    - Uses Sequential Thinking MCP for high-level strategic planning
    - Implements circuit breaker patterns for resilience
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Unified System Controller
        
        Args:
            config: Configuration dictionary for the controller
        """
        # Default configuration
        self.config = {
            "redis_url": "redis://localhost:6379",
            "sequential_thinking_endpoint": "http://localhost:3000/sequential-thinking",
            "alita_manager_endpoint": "http://localhost:3000",
            "kgot_controller_endpoint": "http://localhost:3003",
            "complexity_threshold_sequential_thinking": 7,
            "max_execution_time": 300,  # 5 minutes
            "circuit_breaker_failure_threshold": 5,
            "circuit_breaker_timeout": 60,  # 1 minute
            "enable_performance_monitoring": True,
            "health_check_interval": 30
        }
        
        if config:
            self.config.update(config)
        
        # Initialize enhanced shared state manager
        self.shared_state = EnhancedSharedStateManager(
            redis_url=self.config["redis_url"],
            db=0,
            max_connections=20,
            enable_streaming=True,
            enable_analytics=True
        )
        
        # Initialize Sequential Thinking MCP integration
        self.sequential_thinking = SequentialThinkingMCPIntegration(
            base_url=self.config["sequential_thinking_endpoint"],
            timeout=60
        )
        
        # Initialize legacy performance monitor for backward compatibility
        self.performance_monitor = PerformanceMonitor(
            self.shared_state, 
            self.config["health_check_interval"]
        )
        
        # Initialize advanced monitoring system
        self.advanced_monitoring = AdvancedMonitoringSystem(
            shared_state=self.shared_state,
            monitoring_interval=self.config["health_check_interval"]
        )
        
        # Initialize adaptive load balancer
        self.load_balancer = AdaptiveLoadBalancer(
            shared_state=self.shared_state,
            strategy=LoadBalancingStrategy.ADAPTIVE
        )
        
        # Register system instances for load balancing
        self._register_system_instances()
        
        # Circuit breaker states
        self.circuit_breakers = {
            "alita": {"failures": 0, "last_failure": None, "state": "closed"},
            "kgot": {"failures": 0, "last_failure": None, "state": "closed"}
        }
        
        # HTTP client for API calls
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Controller state
        self._initialized = False
        self._running = False
        
        logger.info("UnifiedSystemController initialized", extra={
            'operation': 'UNIFIED_CONTROLLER_INIT',
            'config': self.config
        })
    
    def _register_system_instances(self) -> None:
        """Register system instances for load balancing"""
        try:
            # Register Alita Manager Agent instance
            alita_instance = SystemInstance(
                instance_id="alita_manager_primary",
                system_name="alita",
                endpoint=f"{self.config['alita_manager_endpoint']}/chat",
                weight=1.0,
                max_concurrent_requests=50
            )
            
            # Register KGoT Controller instance  
            kgot_instance = SystemInstance(
                instance_id="kgot_controller_primary",
                system_name="kgot", 
                endpoint=f"{self.config['kgot_controller_endpoint']}/run",
                weight=1.0,
                max_concurrent_requests=30
            )
            
            # Create circuit breaker configurations
            circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=self.config["circuit_breaker_failure_threshold"],
                recovery_timeout=self.config["circuit_breaker_timeout"],
                success_threshold=3,
                degraded_threshold=0.8,
                progressive_timeout=True
            )
            
            # Register instances with load balancer (async operations will be done in initialize)
            self._pending_instance_registrations = [
                (alita_instance, circuit_breaker_config),
                (kgot_instance, circuit_breaker_config)
            ]
            
            logger.info("System instances prepared for registration", extra={
                'operation': 'SYSTEM_INSTANCES_PREPARED',
                'instances_count': len(self._pending_instance_registrations)
            })
            
        except Exception as e:
            logger.error(f"Error preparing system instances: {str(e)}", extra={
                'operation': 'SYSTEM_INSTANCES_PREPARE_ERROR',
                'error': str(e)
            })
    
    async def initialize(self) -> None:
        """Initialize the controller and all subsystems"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Unified System Controller", extra={
                'operation': 'UNIFIED_CONTROLLER_INITIALIZE'
            })
            
            # Connect to shared state
            await self.shared_state.connect()
            
            # Register system instances with load balancer
            if hasattr(self, '_pending_instance_registrations'):
                for instance, config in self._pending_instance_registrations:
                    await self.load_balancer.register_instance(instance, config)
                delattr(self, '_pending_instance_registrations')
            
            # Start performance monitoring if enabled
            if self.config["enable_performance_monitoring"]:
                await self.performance_monitor.start_monitoring()
                await self.advanced_monitoring.start_monitoring()
                await self.load_balancer.start_health_monitoring()
            
            # Verify system connectivity
            await self._verify_system_connectivity()
            
            self._initialized = True
            self._running = True
            
            logger.info("Unified System Controller initialized successfully", extra={
                'operation': 'UNIFIED_CONTROLLER_INITIALIZE_SUCCESS'
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize Unified System Controller: {str(e)}", extra={
                'operation': 'UNIFIED_CONTROLLER_INITIALIZE_ERROR',
                'error': str(e)
            })
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the controller and all subsystems"""
        if not self._running:
            return
        
        try:
            logger.info("Shutting down Unified System Controller", extra={
                'operation': 'UNIFIED_CONTROLLER_SHUTDOWN'
            })
            
            self._running = False
            
            # Stop performance monitoring
            await self.performance_monitor.stop_monitoring()
            await self.advanced_monitoring.stop_monitoring()
            await self.load_balancer.stop_health_monitoring()
            
            # Close HTTP client
            await self.http_client.aclose()
            
            # Disconnect from shared state
            await self.shared_state.disconnect()
            
            logger.info("Unified System Controller shutdown completed", extra={
                'operation': 'UNIFIED_CONTROLLER_SHUTDOWN_SUCCESS'
            })
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}", extra={
                'operation': 'UNIFIED_CONTROLLER_SHUTDOWN_ERROR',
                'error': str(e)
            })
    
    async def _verify_system_connectivity(self) -> None:
        """Verify connectivity to required systems"""
        systems_to_check = [
            ("Sequential Thinking MCP", self.config["sequential_thinking_endpoint"] + "/health"),
            ("Alita Manager Agent", self.config["alita_manager_endpoint"] + "/health"),
        ]
        
        for system_name, endpoint in systems_to_check:
            try:
                response = await self.http_client.get(endpoint)
                if response.status_code == 200:
                    logger.info(f"{system_name} connectivity verified", extra={
                        'operation': 'SYSTEM_CONNECTIVITY_CHECK',
                        'system': system_name,
                        'status': 'healthy'
                    })
                else:
                    logger.warning(f"{system_name} returned status {response.status_code}", extra={
                        'operation': 'SYSTEM_CONNECTIVITY_CHECK',
                        'system': system_name,
                        'status': 'degraded',
                        'status_code': response.status_code
                    })
            except Exception as e:
                logger.warning(f"Cannot connect to {system_name}: {str(e)}", extra={
                    'operation': 'SYSTEM_CONNECTIVITY_CHECK',
                    'system': system_name,
                    'status': 'failed',
                    'error': str(e)
                })
                # Continue anyway - systems might not be running yet 

    async def process_task(self, task_description: str, 
                          context: Optional[Dict[str, Any]] = None,
                          timeout_seconds: Optional[int] = None) -> ExecutionResult:
        """
        Main entry point for task processing with intelligent routing
        
        Args:
            task_description: The task to be processed
            context: Optional context and constraints
            timeout_seconds: Optional timeout override
            
        Returns:
            ExecutionResult with task outcome and metrics
        """
        if not self._running:
            raise RuntimeError("UnifiedSystemController not initialized")
        
        # Create task context
        task_id = str(uuid4())
        start_time = time.time()
        
        logger.info(f"Processing task: {task_id}", extra={
            'operation': 'TASK_PROCESS_START',
            'task_id': task_id,
            'description_length': len(task_description)
        })
        
        try:
            # Step 1: Analyze task complexity and requirements
            task_context = await self._analyze_task_complexity(
                task_id, task_description, context, timeout_seconds
            )
            
            # Step 2: Determine routing strategy using Sequential Thinking MCP
            routing_strategy = await self._determine_routing_strategy(task_context)
            task_context.routing_strategy = routing_strategy
            
            # Step 3: Execute task based on routing strategy
            result = await self._execute_task_with_routing(task_context)
            
            # Step 4: Record execution metrics and update shared state
            execution_time = (time.time() - start_time) * 1000
            execution_result = ExecutionResult(
                task_id=task_id,
                success=True,
                result=result,
                error=None,
                execution_time_ms=execution_time,
                systems_used=result.get("systems_used", []),
                routing_decisions=result.get("routing_decisions", []),
                metrics=result.get("metrics", {}),
                sequential_thinking_trace=result.get("sequential_thinking_trace")
            )
            
            # Add to task history
            await self.shared_state.add_task_to_history(execution_result)
            
            logger.info(f"Task completed successfully: {task_id}", extra={
                'operation': 'TASK_PROCESS_SUCCESS',
                'task_id': task_id,
                'execution_time_ms': execution_time,
                'systems_used': result.get("systems_used", [])
            })
            
            return execution_result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            execution_result = ExecutionResult(
                task_id=task_id,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time,
                systems_used=[],
                routing_decisions=[],
                metrics={}
            )
            
            # Add failed task to history
            await self.shared_state.add_task_to_history(execution_result)
            
            logger.error(f"Task failed: {task_id} - {str(e)}", extra={
                'operation': 'TASK_PROCESS_ERROR',
                'task_id': task_id,
                'error': str(e),
                'execution_time_ms': execution_time
            })
            
            return execution_result
    
    async def _analyze_task_complexity(self, task_id: str, task_description: str,
                                     context: Optional[Dict[str, Any]],
                                     timeout_seconds: Optional[int]) -> TaskContext:
        """
        Analyze task to determine complexity and requirements
        
        Args:
            task_id: Unique task identifier
            task_description: Task description to analyze
            context: Optional context information
            timeout_seconds: Optional timeout
            
        Returns:
            TaskContext with complexity analysis
        """
        logger.debug(f"Analyzing task complexity: {task_id}", extra={
            'operation': 'TASK_COMPLEXITY_ANALYSIS',
            'task_id': task_id
        })
        
        # Initialize context with defaults
        if context is None:
            context = {}
        
        # Analyze task requirements using heuristics
        requires_mcp_creation = any(keyword in task_description.lower() for keyword in [
            "create tool", "build mcp", "generate script", "new functionality",
            "custom tool", "automation", "integration"
        ])
        
        requires_knowledge_graph = any(keyword in task_description.lower() for keyword in [
            "analyze", "reasoning", "complex", "relationships", "knowledge",
            "graph", "reasoning", "inference", "patterns"
        ])
        
        requires_web_interaction = any(keyword in task_description.lower() for keyword in [
            "web", "browse", "scrape", "website", "online", "search",
            "download", "fetch", "crawl"
        ])
        
        requires_code_generation = any(keyword in task_description.lower() for keyword in [
            "code", "program", "script", "function", "algorithm",
            "implementation", "develop", "build"
        ])
        
        requires_complex_reasoning = any(keyword in task_description.lower() for keyword in [
            "complex", "analyze", "evaluate", "reasoning", "logic",
            "sophisticated", "advanced", "multi-step"
        ])
        
        # Calculate complexity score (1-10)
        complexity_score = 1
        
        if requires_mcp_creation:
            complexity_score += 2
        if requires_knowledge_graph:
            complexity_score += 3
        if requires_web_interaction:
            complexity_score += 1
        if requires_code_generation:
            complexity_score += 2
        if requires_complex_reasoning:
            complexity_score += 3
        
        # Additional complexity based on task length and context
        if len(task_description) > 500:
            complexity_score += 1
        if context and len(context) > 5:
            complexity_score += 1
        
        # Cap at 10
        complexity_score = min(complexity_score, 10)
        
        # Determine data complexity and interaction type
        data_complexity = "high" if complexity_score >= 7 else "medium" if complexity_score >= 4 else "low"
        interaction_type = "multi_domain" if complexity_score >= 8 else "single_domain"
        
        task_context = TaskContext(
            task_id=task_id,
            description=task_description,
            complexity_score=complexity_score,
            routing_strategy=RoutingStrategy.ALITA_FIRST,  # Will be determined later
            requires_mcp_creation=requires_mcp_creation,
            requires_knowledge_graph=requires_knowledge_graph,
            requires_web_interaction=requires_web_interaction,
            requires_code_generation=requires_code_generation,
            requires_complex_reasoning=requires_complex_reasoning,
            data_complexity=data_complexity,
            interaction_type=interaction_type,
            budget_constraints=context.get("budget_constraints"),
            timeout_seconds=timeout_seconds or self.config["max_execution_time"]
        )
        
        logger.info(f"Task complexity analyzed: {task_id}", extra={
            'operation': 'TASK_COMPLEXITY_ANALYSIS_COMPLETE',
            'task_id': task_id,
            'complexity_score': complexity_score,
            'data_complexity': data_complexity,
            'interaction_type': interaction_type,
            'requires_mcp_creation': requires_mcp_creation,
            'requires_knowledge_graph': requires_knowledge_graph,
            'requires_web_interaction': requires_web_interaction,
            'requires_code_generation': requires_code_generation,
            'requires_complex_reasoning': requires_complex_reasoning
        })
        
        return task_context
    
    async def _determine_routing_strategy(self, task_context: TaskContext) -> RoutingStrategy:
        """
        Use Sequential Thinking MCP to determine optimal routing strategy
        
        Args:
            task_context: Analyzed task context
            
        Returns:
            Optimal routing strategy for the task
        """
        logger.info(f"Determining routing strategy: {task_context.task_id}", extra={
            'operation': 'ROUTING_STRATEGY_DETERMINATION',
            'task_id': task_context.task_id,
            'complexity_score': task_context.complexity_score
        })
        
        # For simple tasks, use direct routing heuristics
        if task_context.complexity_score < self.config["complexity_threshold_sequential_thinking"]:
            return self._simple_routing_heuristics(task_context)
        
        # For complex tasks, invoke Sequential Thinking MCP via integration class
        try:
            # Get current system health for context
            system_health = await self._get_system_health_summary()
            
            # Use the enhanced Sequential Thinking MCP integration for routing decisions
            thinking_result = await self.sequential_thinking.process_task_routing(
                task_description=task_context.description,
                task_complexity=task_context.complexity_score,
                system_requirements={
                    'requires_mcp_creation': task_context.requires_mcp_creation,
                    'requires_knowledge_graph': task_context.requires_knowledge_graph,
                    'requires_web_interaction': task_context.requires_web_interaction,
                    'requires_code_generation': task_context.requires_code_generation,
                    'requires_complex_reasoning': task_context.requires_complex_reasoning,
                    'data_complexity': task_context.data_complexity,
                    'interaction_type': task_context.interaction_type
                },
                available_systems=["alita", "kgot"],
                system_health=system_health,
                budget_constraints=task_context.budget_constraints
            )
            
            if thinking_result and thinking_result.get('success', False):
                # Extract routing strategy from the structured result
                suggested_strategy = thinking_result.get('routing_strategy')
                if suggested_strategy:
                    try:
                        routing_decision = RoutingStrategy(suggested_strategy)
                    except ValueError:
                        # If the strategy isn't valid, parse from reasoning
                        routing_decision = self._parse_sequential_thinking_routing(thinking_result)
                else:
                    routing_decision = self._parse_sequential_thinking_routing(thinking_result)
                
                # Store the thinking trace for analysis
                task_context.sequential_thinking_trace = thinking_result.get('thinking_steps', [])
                
                logger.info(f"Sequential thinking routing determined: {routing_decision}", extra={
                    'operation': 'SEQUENTIAL_THINKING_ROUTING_SUCCESS',
                    'task_id': task_context.task_id,
                    'routing_strategy': routing_decision.value,
                    'thought_count': len(thinking_result.get('thinking_steps', [])),
                    'confidence_score': thinking_result.get('confidence_score', 0)
                })
                
                return routing_decision
            else:
                logger.warning("Sequential thinking returned no valid result", extra={
                    'operation': 'SEQUENTIAL_THINKING_ROUTING_ERROR',
                    'task_id': task_context.task_id,
                    'result': thinking_result
                })
                # Fall back to heuristics
                return self._simple_routing_heuristics(task_context)
                
        except Exception as e:
            logger.error(f"Error in sequential thinking routing: {str(e)}", extra={
                'operation': 'SEQUENTIAL_THINKING_ROUTING_EXCEPTION',
                'task_id': task_context.task_id,
                'error': str(e)
            })
            # Fall back to heuristics
            return self._simple_routing_heuristics(task_context)
    
    def _simple_routing_heuristics(self, task_context: TaskContext) -> RoutingStrategy:
        """
        Simple heuristic-based routing for less complex tasks
        
        Args:
            task_context: Task context to analyze
            
        Returns:
            Routing strategy based on heuristics
        """
        # Decision logic based on task characteristics
        if task_context.requires_knowledge_graph and task_context.requires_complex_reasoning:
            if task_context.requires_mcp_creation:
                return RoutingStrategy.HYBRID  # KGoT first, then Alita for tools
            else:
                return RoutingStrategy.KGOT_FIRST
        
        if task_context.requires_mcp_creation and (task_context.requires_web_interaction or task_context.requires_code_generation):
            if task_context.requires_knowledge_graph:
                return RoutingStrategy.HYBRID  # Alita first, then KGoT for reasoning
            else:
                return RoutingStrategy.ALITA_FIRST
        
        if task_context.data_complexity == "high" or task_context.interaction_type == "multi_domain":
            return RoutingStrategy.PARALLEL  # Both systems in parallel
        
        # Default to Alita for most standard tasks
        return RoutingStrategy.ALITA_FIRST
    
    def _parse_sequential_thinking_routing(self, thinking_result: Dict[str, Any]) -> RoutingStrategy:
        """
        Parse Sequential Thinking MCP result to extract routing decision
        
        Args:
            thinking_result: Result from Sequential Thinking MCP
            
        Returns:
            Parsed routing strategy
        """
        # Extract the final decision from the thinking result
        final_answer = thinking_result.get("final_answer", "")
        thoughts = thinking_result.get("thoughts", [])
        
        # Look for routing keywords in the final answer and thoughts
        text_to_analyze = final_answer + " " + " ".join([t.get("content", "") for t in thoughts])
        text_lower = text_to_analyze.lower()
        
        if "hybrid" in text_lower or "both systems" in text_lower or "sequential" in text_lower:
            return RoutingStrategy.HYBRID
        elif "parallel" in text_lower or "concurrent" in text_lower:
            return RoutingStrategy.PARALLEL
        elif "kgot" in text_lower or "knowledge graph" in text_lower or "reasoning first" in text_lower:
            return RoutingStrategy.KGOT_FIRST
        elif "alita" in text_lower or "tool creation" in text_lower or "mcp first" in text_lower:
            return RoutingStrategy.ALITA_FIRST
        else:
            # Default based on task characteristics if no clear indication
            return RoutingStrategy.ALITA_FIRST
    
    async def _get_system_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health for routing decisions"""
        metrics = await self.performance_monitor.get_system_metrics()
        
        health_summary = {}
        for system_name, metric in metrics.items():
            health_summary[system_name] = {
                "status": metric.status.value,
                "response_time_ms": metric.response_time_ms,
                "success_rate": metric.success_rate,
                "load_percentage": metric.load_percentage
            }
        
        return health_summary
    
    async def _execute_task_with_routing(self, task_context: TaskContext) -> Dict[str, Any]:
        """
        Execute task based on determined routing strategy
        
        Args:
            task_context: Task context with routing strategy
            
        Returns:
            Execution result dictionary
        """
        logger.info(f"Executing task with routing: {task_context.routing_strategy.value}", extra={
            'operation': 'TASK_EXECUTION_START',
            'task_id': task_context.task_id,
            'routing_strategy': task_context.routing_strategy.value
        })
        
        routing_decisions = []
        systems_used = []
        execution_metrics = {}
        
        try:
            if task_context.routing_strategy == RoutingStrategy.ALITA_FIRST:
                result = await self._execute_alita_first(task_context, routing_decisions, systems_used)
            elif task_context.routing_strategy == RoutingStrategy.KGOT_FIRST:
                result = await self._execute_kgot_first(task_context, routing_decisions, systems_used)
            elif task_context.routing_strategy == RoutingStrategy.HYBRID:
                result = await self._execute_hybrid(task_context, routing_decisions, systems_used)
            elif task_context.routing_strategy == RoutingStrategy.PARALLEL:
                result = await self._execute_parallel(task_context, routing_decisions, systems_used)
            else:
                raise ValueError(f"Unknown routing strategy: {task_context.routing_strategy}")
            
            # Collect final metrics
            execution_metrics = await self.performance_monitor.get_system_metrics()
            
            return {
                "result": result,
                "systems_used": systems_used,
                "routing_decisions": routing_decisions,
                "metrics": {name: asdict(metric) for name, metric in execution_metrics.items() if metric}
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}", extra={
                'operation': 'TASK_EXECUTION_ERROR',
                'task_id': task_context.task_id,
                'error': str(e)
            })
            raise
    
    async def _execute_alita_first(self, task_context: TaskContext, 
                                 routing_decisions: List[Dict[str, Any]],
                                 systems_used: List[str]) -> Dict[str, Any]:
        """Execute task using Alita-first routing strategy"""
        routing_decisions.append({
            "decision": "alita_first",
            "reason": "Task requires tool creation or web interaction",
            "timestamp": datetime.now().isoformat()
        })
        
        # Check Alita system health with circuit breaker
        if not await self._check_circuit_breaker("alita"):
            # Fall back to KGoT if Alita is down
            logger.warning("Alita circuit breaker open, falling back to KGoT", extra={
                'operation': 'CIRCUIT_BREAKER_FALLBACK',
                'task_id': task_context.task_id,
                'system': 'alita'
            })
            return await self._execute_kgot_first(task_context, routing_decisions, systems_used)
        
        try:
            # Execute through Alita Manager Agent
            alita_request = {
                "message": task_context.description,
                "context": [],
                "sessionId": task_context.task_id,
                "timeout": task_context.timeout_seconds
            }
            
            response = await self.http_client.post(
                f"{self.config['alita_manager_endpoint']}/chat",
                json=alita_request,
                timeout=task_context.timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                systems_used.append("alita")
                
                # Reset circuit breaker on success
                await self._reset_circuit_breaker("alita")
                
                # Update shared state with any new MCPs or knowledge
                await self._update_shared_state_from_alita_result(result)
                
                return result
            else:
                raise Exception(f"Alita Manager Agent returned status {response.status_code}")
                
        except Exception as e:
            await self._record_circuit_breaker_failure("alita")
            raise Exception(f"Alita execution failed: {str(e)}")
    
    async def _execute_kgot_first(self, task_context: TaskContext,
                                routing_decisions: List[Dict[str, Any]],
                                systems_used: List[str]) -> Dict[str, Any]:
        """Execute task using KGoT-first routing strategy"""
        routing_decisions.append({
            "decision": "kgot_first",
            "reason": "Task requires knowledge graph reasoning",
            "timestamp": datetime.now().isoformat()
        })
        
        # Check KGoT system health with circuit breaker
        if not await self._check_circuit_breaker("kgot"):
            # Fall back to Alita if KGoT is down
            logger.warning("KGoT circuit breaker open, falling back to Alita", extra={
                'operation': 'CIRCUIT_BREAKER_FALLBACK',
                'task_id': task_context.task_id,
                'system': 'kgot'
            })
            return await self._execute_alita_first(task_context, routing_decisions, systems_used)
        
        try:
            # Execute through KGoT Controller
            # Note: This would need to be adapted based on actual KGoT Controller API
            kgot_request = {
                "problem": task_context.description,
                "attachments_file_path": "",
                "attachments_file_names": [],
                "timeout": task_context.timeout_seconds
            }
            
            response = await self.http_client.post(
                f"{self.config['kgot_controller_endpoint']}/process",
                json=kgot_request,
                timeout=task_context.timeout_seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                systems_used.append("kgot")
                
                # Reset circuit breaker on success
                await self._reset_circuit_breaker("kgot")
                
                # Update shared state with knowledge graph updates
                await self._update_shared_state_from_kgot_result(result)
                
                return result
            else:
                raise Exception(f"KGoT Controller returned status {response.status_code}")
                
        except Exception as e:
            await self._record_circuit_breaker_failure("kgot")
            raise Exception(f"KGoT execution failed: {str(e)}")
    
    async def _execute_hybrid(self, task_context: TaskContext,
                            routing_decisions: List[Dict[str, Any]],
                            systems_used: List[str]) -> Dict[str, Any]:
        """Execute task using hybrid routing strategy (sequential)"""
        routing_decisions.append({
            "decision": "hybrid",
            "reason": "Task requires both tool creation and knowledge reasoning",
            "timestamp": datetime.now().isoformat()
        })
        
        # Determine which system to start with based on task characteristics
        if task_context.requires_knowledge_graph and task_context.requires_complex_reasoning:
            # Start with KGoT for reasoning, then Alita for tools
            kgot_result = await self._execute_kgot_first(task_context, routing_decisions, systems_used)
            
            # Use KGoT result to inform Alita execution
            enhanced_context = TaskContext(
                task_id=task_context.task_id + "_alita_phase",
                description=f"Based on the reasoning: {kgot_result.get('response', '')}\n\nNow create tools or perform actions for: {task_context.description}",
                complexity_score=task_context.complexity_score,
                routing_strategy=RoutingStrategy.ALITA_FIRST,
                requires_mcp_creation=task_context.requires_mcp_creation,
                requires_web_interaction=task_context.requires_web_interaction,
                requires_code_generation=task_context.requires_code_generation,
                timeout_seconds=task_context.timeout_seconds // 2  # Split time
            )
            
            alita_result = await self._execute_alita_first(enhanced_context, routing_decisions, systems_used)
            
            # Combine results
            return {
                "hybrid_result": {
                    "kgot_reasoning": kgot_result,
                    "alita_execution": alita_result
                },
                "final_response": alita_result.get("response", "")
            }
        else:
            # Start with Alita for tool creation, then KGoT for reasoning
            alita_result = await self._execute_alita_first(task_context, routing_decisions, systems_used)
            
            # Use Alita result to inform KGoT reasoning
            enhanced_context = TaskContext(
                task_id=task_context.task_id + "_kgot_phase",
                description=f"Analyze and reason about the following results: {alita_result.get('response', '')}\n\nOriginal task: {task_context.description}",
                complexity_score=task_context.complexity_score,
                routing_strategy=RoutingStrategy.KGOT_FIRST,
                requires_complex_reasoning=True,
                requires_knowledge_graph=True,
                timeout_seconds=task_context.timeout_seconds // 2  # Split time
            )
            
            kgot_result = await self._execute_kgot_first(enhanced_context, routing_decisions, systems_used)
            
            # Combine results
            return {
                "hybrid_result": {
                    "alita_execution": alita_result,
                    "kgot_reasoning": kgot_result
                },
                "final_response": kgot_result.get("response", "")
            }
    
    async def _execute_parallel(self, task_context: TaskContext,
                              routing_decisions: List[Dict[str, Any]],
                              systems_used: List[str]) -> Dict[str, Any]:
        """Execute task using parallel routing strategy"""
        routing_decisions.append({
            "decision": "parallel",
            "reason": "Task complexity requires concurrent system execution",
            "timestamp": datetime.now().isoformat()
        })
        
        # Execute both systems concurrently
        alita_task = asyncio.create_task(
            self._execute_alita_first(task_context, [], [])
        )
        kgot_task = asyncio.create_task(
            self._execute_kgot_first(task_context, [], [])
        )
        
        try:
            # Wait for both with timeout
            alita_result, kgot_result = await asyncio.wait_for(
                asyncio.gather(alita_task, kgot_task, return_exceptions=True),
                timeout=task_context.timeout_seconds
            )
            
            systems_used.extend(["alita", "kgot"])
            
            # Handle any exceptions
            if isinstance(alita_result, Exception):
                logger.warning(f"Alita parallel execution failed: {str(alita_result)}", extra={
                    'operation': 'PARALLEL_EXECUTION_ALITA_FAILED',
                    'task_id': task_context.task_id,
                    'error': str(alita_result)
                })
                alita_result = {"error": str(alita_result)}
            
            if isinstance(kgot_result, Exception):
                logger.warning(f"KGoT parallel execution failed: {str(kgot_result)}", extra={
                    'operation': 'PARALLEL_EXECUTION_KGOT_FAILED',
                    'task_id': task_context.task_id,
                    'error': str(kgot_result)
                })
                kgot_result = {"error": str(kgot_result)}
            
            # Combine results intelligently
            return {
                "parallel_result": {
                    "alita_result": alita_result,
                    "kgot_result": kgot_result
                },
                "final_response": self._synthesize_parallel_results(alita_result, kgot_result)
            }
            
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            alita_task.cancel()
            kgot_task.cancel()
            raise Exception(f"Parallel execution timed out after {task_context.timeout_seconds} seconds")
    
    def _synthesize_parallel_results(self, alita_result: Dict[str, Any], 
                                   kgot_result: Dict[str, Any]) -> str:
        """Synthesize results from parallel execution"""
        alita_response = alita_result.get("response", "")
        kgot_response = kgot_result.get("response", "")
        
        if alita_response and kgot_response:
            return f"Combined Analysis:\n\nAlita Execution:\n{alita_response}\n\nKGoT Reasoning:\n{kgot_response}"
        elif alita_response:
            return alita_response
        elif kgot_response:
            return kgot_response
        else:
            return "Both systems encountered errors during parallel execution."
    
    async def _check_circuit_breaker(self, system: str) -> bool:
        """Check if circuit breaker allows execution for a system"""
        breaker = self.circuit_breakers.get(system, {"state": "closed"})
        
        if breaker["state"] == "closed":
            return True
        elif breaker["state"] == "open":
            # Check if timeout has passed
            if breaker["last_failure"]:
                time_since_failure = (datetime.now() - breaker["last_failure"]).total_seconds()
                if time_since_failure > self.config["circuit_breaker_timeout"]:
                    # Move to half-open state
                    breaker["state"] = "half-open"
                    logger.info(f"Circuit breaker for {system} moved to half-open", extra={
                        'operation': 'CIRCUIT_BREAKER_HALF_OPEN',
                        'system': system
                    })
                    return True
            return False
        elif breaker["state"] == "half-open":
            return True
        
        return False
    
    async def _reset_circuit_breaker(self, system: str) -> None:
        """Reset circuit breaker on successful execution"""
        self.circuit_breakers[system] = {
            "failures": 0,
            "last_failure": None,
            "state": "closed"
        }
        
        logger.debug(f"Circuit breaker reset for {system}", extra={
            'operation': 'CIRCUIT_BREAKER_RESET',
            'system': system
        })
    
    async def _record_circuit_breaker_failure(self, system: str) -> None:
        """Record failure and potentially open circuit breaker"""
        breaker = self.circuit_breakers.get(system, {"failures": 0, "state": "closed"})
        breaker["failures"] += 1
        breaker["last_failure"] = datetime.now()
        
        if breaker["failures"] >= self.config["circuit_breaker_failure_threshold"]:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for {system}", extra={
                'operation': 'CIRCUIT_BREAKER_OPEN',
                'system': system,
                'failure_count': breaker["failures"]
            })
        
        self.circuit_breakers[system] = breaker
    
    async def _update_shared_state_from_alita_result(self, result: Dict[str, Any]) -> None:
        """Update shared state with Alita execution results"""
        # Extract any new MCPs created
        if "mcps_created" in result:
            for mcp_data in result["mcps_created"]:
                await self.shared_state.update_mcp_registry(mcp_data)
        
        # Update budget if cost information is available
        if "cost_breakdown" in result:
            await self.shared_state.update_budget_usage(result["cost_breakdown"])
    
    async def _update_shared_state_from_kgot_result(self, result: Dict[str, Any]) -> None:
        """Update shared state with KGoT execution results"""
        # Extract knowledge graph updates
        if "graph_updates" in result:
            await self.shared_state.update_kgot_graph_state(result["graph_updates"])
        
        # Update budget if cost information is available
        if "cost_breakdown" in result:
            await self.shared_state.update_budget_usage(result["cost_breakdown"])
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and health information"""
        if not self._running:
            return {"status": "not_running"}
        
        try:
            # Get system metrics
            metrics = await self.performance_monitor.get_system_metrics()
            
            # Get circuit breaker states
            circuit_breaker_states = {
                name: breaker["state"] for name, breaker in self.circuit_breakers.items()
            }
            
            # Get shared state summary
            budget = await self.shared_state.get_system_budget()
            mcps = await self.shared_state.get_available_mcps()
            recent_tasks = await self.shared_state.get_task_history(limit=10)
            
            return {
                "status": "running",
                "initialized": self._initialized,
                "system_metrics": {name: asdict(metric) for name, metric in metrics.items() if metric},
                "circuit_breakers": circuit_breaker_states,
                "budget": budget,
                "available_mcps_count": len(mcps),
                "recent_tasks_count": len(recent_tasks),
                "performance_monitoring": self.config["enable_performance_monitoring"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}", extra={
                'operation': 'SYSTEM_STATUS_ERROR',
                'error': str(e)
            })
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring dashboard data
        
        Returns:
            Dashboard data including alerts, health scores, trends, and analytics
        """
        try:
            dashboard_data = await self.advanced_monitoring.get_monitoring_dashboard_data()
            
            # Add unified controller specific metrics
            dashboard_data['unified_controller'] = {
                'status': 'running' if self._running else 'stopped',
                'initialized': self._initialized,
                'circuit_breakers': self.circuit_breakers,
                'config': self.config
            }
            
            # Add shared state analytics if available
            if hasattr(self.shared_state, 'get_state_analytics'):
                try:
                    state_analytics = await self.shared_state.get_state_analytics()
                    dashboard_data['state_analytics'] = state_analytics
                except Exception as e:
                    logger.warning(f"Could not get state analytics: {str(e)}")
            
            # Add load balancing status
            try:
                load_balancing_status = await self.load_balancer.get_load_balancing_status()
                dashboard_data['load_balancing'] = load_balancing_status
            except Exception as e:
                logger.warning(f"Could not get load balancing status: {str(e)}")
            
            logger.info("Monitoring dashboard data retrieved", extra={
                'operation': 'MONITORING_DASHBOARD_GET',
                'active_alerts_count': len(dashboard_data.get('active_alerts', [])),
                'systems_monitored': len(dashboard_data.get('system_health', {}))
            })
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting monitoring dashboard: {str(e)}", extra={
                'operation': 'MONITORING_DASHBOARD_ERROR',
                'error': str(e)
            })
            raise

    async def force_routing_strategy(self, task_description: str, 
                                   strategy: RoutingStrategy,
                                   context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Force a specific routing strategy for testing or manual override
        
        Args:
            task_description: Task to process
            strategy: Forced routing strategy
            context: Optional context
            
        Returns:
            ExecutionResult with forced routing
        """
        logger.info(f"Forcing routing strategy: {strategy.value}", extra={
            'operation': 'FORCE_ROUTING_STRATEGY',
            'strategy': strategy.value
        })
        
        # Create task context with forced strategy
        task_context = await self._analyze_task_complexity(
            str(uuid4()), task_description, context, None
        )
        task_context.routing_strategy = strategy
        
        # Execute with forced strategy
        return await self.process_task(task_description, context) 