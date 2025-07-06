"""
Advanced Load Balancing and Circuit Breaker System

This module provides sophisticated load balancing and circuit breaker patterns
for the unified system controller, including:

- Adaptive load balancing based on real-time metrics
- Multi-level circuit breaker patterns with progressive degradation
- Intelligent failover and recovery strategies
- Dynamic threshold adjustment based on historical performance
- Health-aware routing decisions
- Resource-based load distribution

@module LoadBalancingSystem
@author AI Assistant  
@date 2025-01-22
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from uuid import uuid4
import statistics

# Import logging configuration [[memory:1383804]]
from ..config.logging.winston_config import get_logger

# Import shared state utilities
from .shared_state_utilities import (
    EnhancedSharedStateManager, 
    StateScope, 
    StateEventType
)

# Create logger instance
logger = get_logger('load_balancing_system')


class CircuitBreakerState(Enum):
    """Circuit breaker states with progressive degradation"""
    CLOSED = "closed"           # Normal operation
    HALF_OPEN = "half_open"     # Testing recovery
    OPEN = "open"               # Failing, routing disabled
    DEGRADED = "degraded"       # Partial capacity operation
    MAINTENANCE = "maintenance" # Planned downtime


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME_BASED = "response_time_based"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"


class HealthStatus(Enum):
    """System health status levels"""
    EXCELLENT = "excellent"     # > 95% performance
    GOOD = "good"              # 85-95% performance
    DEGRADED = "degraded"       # 70-85% performance
    POOR = "poor"              # 50-70% performance
    CRITICAL = "critical"       # < 50% performance


@dataclass
class SystemInstance:
    """Represents a system instance for load balancing"""
    instance_id: str
    system_name: str
    endpoint: str
    weight: float = 1.0
    max_concurrent_requests: int = 100
    current_connections: int = 0
    last_response_time: float = 0.0
    success_rate: float = 1.0
    health_status: HealthStatus = HealthStatus.GOOD
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    last_health_check: datetime = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    def __post_init__(self):
        if self.last_health_check is None:
            self.last_health_check = datetime.now()
    
    @property
    def load_percentage(self) -> float:
        """Calculate current load percentage"""
        return (self.current_connections / self.max_concurrent_requests) * 100
    
    @property 
    def is_available(self) -> bool:
        """Check if instance is available for new requests"""
        return (self.circuit_breaker_state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN, CircuitBreakerState.DEGRADED] 
                and self.current_connections < self.max_concurrent_requests)
    
    def update_metrics(self, response_time: float, success: bool) -> None:
        """Update instance metrics after request completion"""
        self.last_response_time = response_time
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Calculate success rate
        if self.total_requests > 0:
            self.success_rate = self.successful_requests / self.total_requests


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5           # Failures to open circuit
    recovery_timeout: int = 60           # Seconds before attempting recovery
    success_threshold: int = 3           # Successes to close circuit from half-open
    degraded_threshold: float = 0.7      # Success rate threshold for degraded state
    health_check_interval: int = 30      # Seconds between health checks
    max_concurrent_half_open: int = 1    # Max requests in half-open state
    progressive_timeout: bool = True     # Enable progressive timeout increases
    
    
@dataclass
class LoadBalancingMetrics:
    """Metrics for load balancing performance"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    requests_per_minute: float = 0.0
    load_distribution: Dict[str, int] = None
    circuit_breaker_trips: int = 0
    failover_events: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.load_distribution is None:
            self.load_distribution = {}
        if self.last_updated is None:
            self.last_updated = datetime.now()


class CircuitBreaker:
    """
    Advanced circuit breaker with progressive degradation and adaptive thresholds
    """
    
    def __init__(self, 
                 instance: SystemInstance, 
                 config: CircuitBreakerConfig,
                 shared_state: EnhancedSharedStateManager):
        self.instance = instance
        self.config = config
        self.shared_state = shared_state
        
        # Circuit breaker state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.half_open_requests = 0
        
        # Adaptive thresholds
        self.dynamic_failure_threshold = config.failure_threshold
        self.dynamic_recovery_timeout = config.recovery_timeout
        
        # Performance tracking
        self.recent_response_times: deque = deque(maxlen=100)
        self.recent_successes: deque = deque(maxlen=100)
        
        logger.info(f"CircuitBreaker initialized for {instance.instance_id}", extra={
            'operation': 'CIRCUIT_BREAKER_INIT',
            'instance_id': instance.instance_id,
            'config': asdict(config)
        })
    
    async def can_execute(self) -> bool:
        """
        Check if requests can be executed through this circuit breaker
        
        Returns:
            True if requests can be executed, False otherwise
        """
        current_time = datetime.now()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        elif self.state == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            if (self.last_failure_time and 
                (current_time - self.last_failure_time).total_seconds() >= self.dynamic_recovery_timeout):
                await self._transition_to_half_open()
                return True
            return False
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Allow limited requests in half-open state
            return self.half_open_requests < self.config.max_concurrent_half_open
        
        elif self.state == CircuitBreakerState.DEGRADED:
            # Allow requests but with reduced capacity
            return self.instance.current_connections < (self.instance.max_concurrent_requests * 0.7)
        
        elif self.state == CircuitBreakerState.MAINTENANCE:
            return False
        
        return False
    
    async def record_success(self, response_time: float) -> None:
        """
        Record a successful request
        
        Args:
            response_time: Response time in milliseconds
        """
        self.recent_response_times.append(response_time)
        self.recent_successes.append(True)
        self.last_success_time = datetime.now()
        self.success_count += 1
        
        # Update instance metrics
        self.instance.update_metrics(response_time, True)
        
        # Handle state transitions
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                await self._transition_to_closed()
            else:
                self.half_open_requests -= 1
        
        elif self.state == CircuitBreakerState.DEGRADED:
            # Check if we can transition back to normal
            if self.instance.success_rate > self.config.degraded_threshold:
                await self._transition_to_closed()
        
        # Store metrics in shared state
        await self._store_metrics()
        
        logger.debug(f"Success recorded: {self.instance.instance_id}", extra={
            'operation': 'CIRCUIT_BREAKER_SUCCESS',
            'instance_id': self.instance.instance_id,
            'response_time': response_time,
            'state': self.state.value,
            'success_rate': self.instance.success_rate
        })
    
    async def record_failure(self, error: str) -> None:
        """
        Record a failed request
        
        Args:
            error: Error description
        """
        self.recent_successes.append(False)
        self.last_failure_time = datetime.now()
        self.failure_count += 1
        
        # Update instance metrics
        self.instance.update_metrics(0.0, False)
        
        # Handle state transitions based on current state
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.dynamic_failure_threshold:
                await self._transition_to_open()
            elif self.instance.success_rate < self.config.degraded_threshold:
                await self._transition_to_degraded()
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Immediate transition back to open on failure
            await self._transition_to_open()
            self.half_open_requests = 0
        
        elif self.state == CircuitBreakerState.DEGRADED:
            if self.failure_count >= self.dynamic_failure_threshold * 2:
                await self._transition_to_open()
        
        # Adapt thresholds based on failure patterns
        await self._adapt_thresholds()
        
        # Store metrics in shared state
        await self._store_metrics()
        
        logger.warning(f"Failure recorded: {self.instance.instance_id} - {error}", extra={
            'operation': 'CIRCUIT_BREAKER_FAILURE',
            'instance_id': self.instance.instance_id,
            'error': error,
            'state': self.state.value,
            'failure_count': self.failure_count
        })
    
    async def _transition_to_open(self) -> None:
        """Transition circuit breaker to open state"""
        self.state = CircuitBreakerState.OPEN
        self.instance.circuit_breaker_state = CircuitBreakerState.OPEN
        
        logger.error(f"Circuit breaker opened: {self.instance.instance_id}", extra={
            'operation': 'CIRCUIT_BREAKER_OPEN',
            'instance_id': self.instance.instance_id,
            'failure_count': self.failure_count
        })
        
        # Notify monitoring system
        await self._notify_state_change("OPEN")
    
    async def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to half-open state"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.instance.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.half_open_requests = 0
        
        logger.info(f"Circuit breaker half-open: {self.instance.instance_id}", extra={
            'operation': 'CIRCUIT_BREAKER_HALF_OPEN',
            'instance_id': self.instance.instance_id
        })
        
        await self._notify_state_change("HALF_OPEN")
    
    async def _transition_to_closed(self) -> None:
        """Transition circuit breaker to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.instance.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        
        logger.info(f"Circuit breaker closed: {self.instance.instance_id}", extra={
            'operation': 'CIRCUIT_BREAKER_CLOSED',
            'instance_id': self.instance.instance_id
        })
        
        await self._notify_state_change("CLOSED")
    
    async def _transition_to_degraded(self) -> None:
        """Transition circuit breaker to degraded state"""
        self.state = CircuitBreakerState.DEGRADED
        self.instance.circuit_breaker_state = CircuitBreakerState.DEGRADED
        
        logger.warning(f"Circuit breaker degraded: {self.instance.instance_id}", extra={
            'operation': 'CIRCUIT_BREAKER_DEGRADED',
            'instance_id': self.instance.instance_id,
            'success_rate': self.instance.success_rate
        })
        
        await self._notify_state_change("DEGRADED")
    
    async def _adapt_thresholds(self) -> None:
        """Adapt failure and recovery thresholds based on performance patterns"""
        if self.config.progressive_timeout:
            # Increase recovery timeout with repeated failures
            failure_streaks = self._count_recent_failure_streaks()
            if failure_streaks > 2:
                self.dynamic_recovery_timeout = min(
                    self.config.recovery_timeout * (2 ** (failure_streaks - 2)),
                    300  # Max 5 minutes
                )
        
        # Adjust failure threshold based on historical performance
        if len(self.recent_successes) >= 50:
            recent_success_rate = sum(self.recent_successes) / len(self.recent_successes)
            if recent_success_rate < 0.8:  # Poor historical performance
                self.dynamic_failure_threshold = max(self.config.failure_threshold - 1, 2)
            else:
                self.dynamic_failure_threshold = self.config.failure_threshold
    
    def _count_recent_failure_streaks(self) -> int:
        """Count consecutive failure streaks in recent history"""
        if not self.recent_successes:
            return 0
        
        streaks = 0
        current_streak = 0
        
        for success in reversed(self.recent_successes):
            if not success:
                current_streak += 1
            else:
                if current_streak >= self.config.failure_threshold:
                    streaks += 1
                current_streak = 0
        
        return streaks
    
    async def _store_metrics(self) -> None:
        """Store circuit breaker metrics in shared state"""
        metrics = {
            'instance_id': self.instance.instance_id,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'success_rate': self.instance.success_rate,
            'last_updated': datetime.now().isoformat()
        }
        
        await self.shared_state.set_state_with_versioning(
            scope=StateScope.METRICS,
            key=f"circuit_breaker_{self.instance.instance_id}",
            value=metrics,
            source_system="load_balancer",
            expire_seconds=3600
        )
    
    async def _notify_state_change(self, new_state: str) -> None:
        """Notify systems of circuit breaker state changes"""
        event_data = {
            'instance_id': self.instance.instance_id,
            'system_name': self.instance.system_name,
            'old_state': getattr(self, '_previous_state', 'UNKNOWN'),
            'new_state': new_state,
            'timestamp': datetime.now().isoformat(),
            'failure_count': self.failure_count
        }
        
        await self.shared_state.set_state_with_versioning(
            scope=StateScope.SYSTEM,
            key=f"circuit_breaker_event_{uuid4()}",
            value=event_data,
            source_system="load_balancer",
            expire_seconds=3600
        )


class AdaptiveLoadBalancer:
    """
    Intelligent load balancer with adaptive routing strategies
    """
    
    def __init__(self, 
                 shared_state: EnhancedSharedStateManager,
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.shared_state = shared_state
        self.strategy = strategy
        self.instances: Dict[str, SystemInstance] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.metrics = LoadBalancingMetrics()
        
        # Load balancing state
        self.round_robin_index = 0
        self.strategy_performance: Dict[LoadBalancingStrategy, float] = defaultdict(float)
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        logger.info("AdaptiveLoadBalancer initialized", extra={
            'operation': 'LOAD_BALANCER_INIT',
            'strategy': strategy.value
        })
    
    async def register_instance(self, 
                              instance: SystemInstance,
                              circuit_breaker_config: Optional[CircuitBreakerConfig] = None) -> None:
        """
        Register a new system instance for load balancing
        
        Args:
            instance: System instance to register
            circuit_breaker_config: Circuit breaker configuration
        """
        self.instances[instance.instance_id] = instance
        
        # Create circuit breaker
        config = circuit_breaker_config or CircuitBreakerConfig()
        self.circuit_breakers[instance.instance_id] = CircuitBreaker(
            instance, config, self.shared_state
        )
        
        logger.info(f"Instance registered: {instance.instance_id}", extra={
            'operation': 'INSTANCE_REGISTER',
            'instance_id': instance.instance_id,
            'system_name': instance.system_name,
            'endpoint': instance.endpoint
        })
    
    async def select_instance(self, 
                            system_name: Optional[str] = None,
                            request_context: Optional[Dict[str, Any]] = None) -> Optional[SystemInstance]:
        """
        Select the best available instance for a request
        
        Args:
            system_name: Filter to specific system type
            request_context: Additional context for routing decisions
            
        Returns:
            Selected instance or None if no instances available
        """
        # Filter available instances
        available_instances = []
        for instance in self.instances.values():
            if system_name and instance.system_name != system_name:
                continue
            
            circuit_breaker = self.circuit_breakers.get(instance.instance_id)
            if circuit_breaker and await circuit_breaker.can_execute():
                available_instances.append(instance)
        
        if not available_instances:
            logger.warning("No available instances for request", extra={
                'operation': 'NO_INSTANCES_AVAILABLE',
                'system_name': system_name,
                'total_instances': len(self.instances)
            })
            return None
        
        # Select instance based on strategy
        if self.strategy == LoadBalancingStrategy.ADAPTIVE:
            selected = await self._adaptive_selection(available_instances, request_context)
        elif self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected = self._round_robin_selection(available_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            selected = self._weighted_round_robin_selection(available_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected = self._least_connections_selection(available_instances)
        elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME_BASED:
            selected = self._response_time_based_selection(available_instances)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            selected = self._resource_based_selection(available_instances)
        else:
            selected = available_instances[0]  # Fallback
        
        if selected:
            selected.current_connections += 1
            self.metrics.load_distribution[selected.instance_id] = (
                self.metrics.load_distribution.get(selected.instance_id, 0) + 1
            )
            
            logger.debug(f"Instance selected: {selected.instance_id}", extra={
                'operation': 'INSTANCE_SELECTED',
                'instance_id': selected.instance_id,
                'strategy': self.strategy.value,
                'load_percentage': selected.load_percentage
            })
        
        return selected
    
    async def _adaptive_selection(self, 
                                instances: List[SystemInstance],
                                request_context: Optional[Dict[str, Any]]) -> SystemInstance:
        """
        Adaptive selection that chooses the best strategy based on current conditions
        """
        # Analyze current system conditions
        avg_response_time = statistics.mean([i.last_response_time for i in instances if i.last_response_time > 0]) or 100
        avg_load = statistics.mean([i.load_percentage for i in instances])
        
        # Choose strategy based on conditions
        if avg_response_time > 1000:  # High response times
            return self._response_time_based_selection(instances)
        elif avg_load > 80:  # High load
            return self._least_connections_selection(instances)
        elif any(i.health_status in [HealthStatus.DEGRADED, HealthStatus.POOR] for i in instances):
            return self._resource_based_selection(instances)
        else:
            return self._weighted_round_robin_selection(instances)
    
    def _round_robin_selection(self, instances: List[SystemInstance]) -> SystemInstance:
        """Simple round-robin selection"""
        selected = instances[self.round_robin_index % len(instances)]
        self.round_robin_index += 1
        return selected
    
    def _weighted_round_robin_selection(self, instances: List[SystemInstance]) -> SystemInstance:
        """Weighted round-robin based on instance weights and health"""
        weights = []
        for instance in instances:
            # Adjust weight based on health and performance
            health_multiplier = {
                HealthStatus.EXCELLENT: 1.2,
                HealthStatus.GOOD: 1.0,
                HealthStatus.DEGRADED: 0.7,
                HealthStatus.POOR: 0.4,
                HealthStatus.CRITICAL: 0.1
            }.get(instance.health_status, 1.0)
            
            effective_weight = instance.weight * health_multiplier * instance.success_rate
            weights.append(effective_weight)
        
        # Select based on weights
        total_weight = sum(weights)
        if total_weight == 0:
            return instances[0]
        
        # Weighted random selection
        import random
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return instances[i]
        
        return instances[-1]  # Fallback
    
    def _least_connections_selection(self, instances: List[SystemInstance]) -> SystemInstance:
        """Select instance with least active connections"""
        return min(instances, key=lambda x: x.current_connections)
    
    def _response_time_based_selection(self, instances: List[SystemInstance]) -> SystemInstance:
        """Select instance with best response time"""
        # Filter out instances with no response time data
        instances_with_times = [i for i in instances if i.last_response_time > 0]
        if not instances_with_times:
            return instances[0]
        
        return min(instances_with_times, key=lambda x: x.last_response_time)
    
    def _resource_based_selection(self, instances: List[SystemInstance]) -> SystemInstance:
        """Select instance based on overall resource utilization"""
        def resource_score(instance):
            # Combine multiple factors into a single score (lower is better)
            load_score = instance.load_percentage / 100
            response_score = min(instance.last_response_time / 1000, 1.0) if instance.last_response_time > 0 else 0
            health_score = {
                HealthStatus.EXCELLENT: 0.0,
                HealthStatus.GOOD: 0.1,
                HealthStatus.DEGRADED: 0.3,
                HealthStatus.POOR: 0.6,
                HealthStatus.CRITICAL: 1.0
            }.get(instance.health_status, 0.5)
            
            return load_score + response_score + health_score
        
        return min(instances, key=resource_score)
    
    async def record_request_completion(self, 
                                      instance_id: str,
                                      response_time: float,
                                      success: bool,
                                      error: Optional[str] = None) -> None:
        """
        Record completion of a request for load balancing metrics
        
        Args:
            instance_id: ID of the instance that handled the request
            response_time: Response time in milliseconds
            success: Whether the request was successful
            error: Error message if request failed
        """
        instance = self.instances.get(instance_id)
        circuit_breaker = self.circuit_breakers.get(instance_id)
        
        if instance:
            instance.current_connections = max(0, instance.current_connections - 1)
        
        if circuit_breaker:
            if success:
                await circuit_breaker.record_success(response_time)
            else:
                await circuit_breaker.record_failure(error or "Unknown error")
        
        # Update global metrics
        self.metrics.total_requests += 1
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update average response time
        if self.metrics.total_requests == 1:
            self.metrics.average_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * response_time + (1 - alpha) * self.metrics.average_response_time
            )
        
        self.metrics.last_updated = datetime.now()
        
        # Store metrics in shared state
        await self._store_metrics()
        
        logger.debug(f"Request completion recorded: {instance_id}", extra={
            'operation': 'REQUEST_COMPLETION',
            'instance_id': instance_id,
            'response_time': response_time,
            'success': success,
            'error': error
        })
    
    async def start_health_monitoring(self) -> None:
        """Start health monitoring for all instances"""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
        
        logger.info("Health monitoring started", extra={
            'operation': 'HEALTH_MONITORING_START'
        })
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring"""
        self._is_monitoring = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped", extra={
            'operation': 'HEALTH_MONITORING_STOP'
        })
    
    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring loop"""
        while self._is_monitoring:
            try:
                for instance in self.instances.values():
                    await self._check_instance_health(instance)
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {str(e)}", extra={
                    'operation': 'HEALTH_MONITORING_ERROR',
                    'error': str(e)
                })
                await asyncio.sleep(30)
    
    async def _check_instance_health(self, instance: SystemInstance) -> None:
        """Perform health check on a specific instance"""
        try:
            # This would make an actual health check request to the instance
            # For now, using success rate and response time as health indicators
            
            health_score = 100
            
            # Deduct based on success rate
            if instance.success_rate < 1.0:
                health_score -= (1.0 - instance.success_rate) * 50
            
            # Deduct based on response time
            if instance.last_response_time > 1000:  # > 1 second
                health_score -= min((instance.last_response_time - 1000) / 100, 30)
            
            # Deduct based on load
            if instance.load_percentage > 80:
                health_score -= (instance.load_percentage - 80) / 2
            
            # Update health status
            if health_score >= 95:
                instance.health_status = HealthStatus.EXCELLENT
            elif health_score >= 85:
                instance.health_status = HealthStatus.GOOD
            elif health_score >= 70:
                instance.health_status = HealthStatus.DEGRADED
            elif health_score >= 50:
                instance.health_status = HealthStatus.POOR
            else:
                instance.health_status = HealthStatus.CRITICAL
            
            instance.last_health_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Health check failed for {instance.instance_id}: {str(e)}")
    
    async def _store_metrics(self) -> None:
        """Store load balancing metrics in shared state"""
        await self.shared_state.set_state_with_versioning(
            scope=StateScope.METRICS,
            key="load_balancer_metrics",
            value=asdict(self.metrics),
            source_system="load_balancer",
            expire_seconds=3600
        )
    
    async def get_load_balancing_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancing status"""
        status = {
            'strategy': self.strategy.value,
            'metrics': asdict(self.metrics),
            'instances': {},
            'circuit_breakers': {}
        }
        
        for instance_id, instance in self.instances.items():
            status['instances'][instance_id] = asdict(instance)
            
            circuit_breaker = self.circuit_breakers.get(instance_id)
            if circuit_breaker:
                status['circuit_breakers'][instance_id] = {
                    'state': circuit_breaker.state.value,
                    'failure_count': circuit_breaker.failure_count,
                    'success_count': circuit_breaker.success_count
                }
        
        return status 