"""
Enhanced Shared State Utilities for Unified System Controller

This module provides advanced shared state management utilities that extend the basic
SharedStateManager with specialized features for cross-system coordination, real-time
synchronization, and intelligent state caching.

Key Features:
- Cross-system state synchronization
- Real-time state event streaming
- Intelligent state versioning and conflict resolution
- Performance-optimized state caching
- State analytics and monitoring
- Distributed locks for critical operations

@module SharedStateUtilities
@author AI Assistant  
@date 2025-01-22
"""

import asyncio
import json
import pickle
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from uuid import uuid4

import redis.asyncio as redis
from redis.asyncio.lock import Lock as RedisLock

# Import logging configuration [[memory:1383804]]
from ..config.logging.winston_config import get_logger

# Create logger instance  
logger = get_logger('shared_state_utilities')


class StateEventType(Enum):
    """Types of state events for real-time synchronization"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    LOCK = "lock"
    UNLOCK = "unlock"
    SYNC = "sync"
    CONFLICT = "conflict"


class StateScope(Enum):
    """Scope of state data for organization and access control"""
    GLOBAL = "global"           # System-wide state
    SESSION = "session"         # User/task session state
    SYSTEM = "system"           # Individual system state (alita, kgot)
    CACHE = "cache"            # Temporary cached data
    METRICS = "metrics"        # Performance and monitoring data
    CONFIG = "config"          # Configuration data


@dataclass
class StateEvent:
    """State change event for real-time synchronization"""
    event_id: str
    event_type: StateEventType
    scope: StateScope
    key: str
    data: Optional[Any]
    timestamp: datetime
    source_system: str
    version: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StateVersion:
    """State version tracking for conflict resolution"""
    version: int
    data: Any
    timestamp: datetime
    source_system: str
    checksum: str
    
    @classmethod
    def create(cls, data: Any, source_system: str, version: int = 1) -> 'StateVersion':
        """Create a new state version"""
        serialized = json.dumps(data, sort_keys=True, default=str)
        checksum = hashlib.sha256(serialized.encode()).hexdigest()
        
        return cls(
            version=version,
            data=data,
            timestamp=datetime.now(),
            source_system=source_system,
            checksum=checksum
        )


class StateConflictResolver:
    """Handles state conflicts during concurrent modifications"""
    
    def __init__(self):
        self.resolution_strategies = {
            "last_write_wins": self._last_write_wins,
            "merge_data": self._merge_data,
            "version_increment": self._version_increment,
            "manual_review": self._manual_review
        }
    
    async def resolve_conflict(self, 
                             current_version: StateVersion,
                             incoming_version: StateVersion,
                             strategy: str = "last_write_wins") -> StateVersion:
        """
        Resolve conflict between state versions
        
        Args:
            current_version: Current state version
            incoming_version: Incoming state version  
            strategy: Resolution strategy to use
            
        Returns:
            Resolved state version
        """
        resolver = self.resolution_strategies.get(strategy, self._last_write_wins)
        return await resolver(current_version, incoming_version)
    
    async def _last_write_wins(self, current: StateVersion, incoming: StateVersion) -> StateVersion:
        """Last write wins strategy"""
        if incoming.timestamp > current.timestamp:
            return incoming
        return current
    
    async def _merge_data(self, current: StateVersion, incoming: StateVersion) -> StateVersion:
        """Merge data strategy for dictionary data"""
        if isinstance(current.data, dict) and isinstance(incoming.data, dict):
            merged_data = {**current.data, **incoming.data}
            return StateVersion.create(
                merged_data, 
                f"{current.source_system}+{incoming.source_system}",
                max(current.version, incoming.version) + 1
            )
        return await self._last_write_wins(current, incoming)
    
    async def _version_increment(self, current: StateVersion, incoming: StateVersion) -> StateVersion:
        """Version increment strategy"""
        return StateVersion.create(
            incoming.data,
            incoming.source_system,
            max(current.version, incoming.version) + 1
        )
    
    async def _manual_review(self, current: StateVersion, incoming: StateVersion) -> StateVersion:
        """Mark for manual review - keeps current version"""
        logger.warning("State conflict requires manual review", extra={
            'operation': 'STATE_CONFLICT_MANUAL_REVIEW',
            'current_version': current.version,
            'incoming_version': incoming.version,
            'current_source': current.source_system,
            'incoming_source': incoming.source_system
        })
        return current


class RealTimeStateStreamer:
    """Real-time state event streaming for system synchronization"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.streaming_task: Optional[asyncio.Task] = None
        self.is_streaming = False
        
        logger.info("RealTimeStateStreamer initialized", extra={
            'operation': 'STATE_STREAMER_INIT'
        })
    
    async def start_streaming(self) -> None:
        """Start real-time state streaming"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.streaming_task = asyncio.create_task(self._stream_events())
        
        logger.info("Real-time state streaming started", extra={
            'operation': 'STATE_STREAMING_START'
        })
    
    async def stop_streaming(self) -> None:
        """Stop real-time state streaming"""
        self.is_streaming = False
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time state streaming stopped", extra={
            'operation': 'STATE_STREAMING_STOP'
        })
    
    async def publish_event(self, event: StateEvent) -> None:
        """Publish state event to stream"""
        event_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'scope': event.scope.value,
            'key': event.key,
            'data': event.data,
            'timestamp': event.timestamp.isoformat(),
            'source_system': event.source_system,
            'version': event.version,
            'metadata': event.metadata
        }
        
        channel = f"state_events:{event.scope.value}"
        await self.redis_client.publish(channel, json.dumps(event_data, default=str))
        
        logger.debug(f"State event published: {event.event_id}", extra={
            'operation': 'STATE_EVENT_PUBLISH',
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'scope': event.scope.value
        })
    
    def subscribe_to_events(self, scope: StateScope, callback: Callable[[StateEvent], None]) -> None:
        """Subscribe to state events for a specific scope"""
        channel = f"state_events:{scope.value}"
        self.subscribers[channel].add(callback)
        
        logger.debug(f"Subscribed to state events: {scope.value}", extra={
            'operation': 'STATE_EVENT_SUBSCRIBE',
            'scope': scope.value,
            'subscribers_count': len(self.subscribers[channel])
        })
    
    def unsubscribe_from_events(self, scope: StateScope, callback: Callable[[StateEvent], None]) -> None:
        """Unsubscribe from state events"""
        channel = f"state_events:{scope.value}"
        self.subscribers[channel].discard(callback)
        
        logger.debug(f"Unsubscribed from state events: {scope.value}", extra={
            'operation': 'STATE_EVENT_UNSUBSCRIBE',
            'scope': scope.value
        })
    
    async def _stream_events(self) -> None:
        """Background task for streaming events"""
        pubsub = self.redis_client.pubsub()
        
        # Subscribe to all scope channels
        for scope in StateScope:
            channel = f"state_events:{scope.value}"
            await pubsub.subscribe(channel)
        
        try:
            while self.is_streaming:
                message = await pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    await self._handle_event_message(message)
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.close()
    
    async def _handle_event_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming event message"""
        try:
            event_data = json.loads(message['data'])
            event = StateEvent(
                event_id=event_data['event_id'],
                event_type=StateEventType(event_data['event_type']),
                scope=StateScope(event_data['scope']),
                key=event_data['key'],
                data=event_data['data'],
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                source_system=event_data['source_system'],
                version=event_data['version'],
                metadata=event_data.get('metadata', {})
            )
            
            # Notify subscribers
            channel = message['channel'].decode()
            for callback in self.subscribers[channel]:
                try:
                    await callback(event) if asyncio.iscoroutinefunction(callback) else callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {str(e)}", extra={
                        'operation': 'STATE_EVENT_CALLBACK_ERROR',
                        'event_id': event.event_id,
                        'error': str(e)
                    })
        
        except Exception as e:
            logger.error(f"Error handling event message: {str(e)}", extra={
                'operation': 'STATE_EVENT_HANDLE_ERROR',
                'error': str(e)
            })


class DistributedLockManager:
    """Distributed locking for critical state operations"""
    
    def __init__(self, redis_client: redis.Redis, default_timeout: int = 30):
        self.redis_client = redis_client
        self.default_timeout = default_timeout
        self.active_locks: Dict[str, RedisLock] = {}
        
        logger.info("DistributedLockManager initialized", extra={
            'operation': 'LOCK_MANAGER_INIT',
            'default_timeout': default_timeout
        })
    
    async def acquire_lock(self, 
                          lock_key: str, 
                          timeout: Optional[int] = None,
                          blocking: bool = True) -> RedisLock:
        """
        Acquire distributed lock
        
        Args:
            lock_key: Unique lock identifier
            timeout: Lock timeout in seconds
            blocking: Whether to block until lock is available
            
        Returns:
            Redis lock object
        """
        timeout = timeout or self.default_timeout
        full_key = f"lock:{lock_key}"
        
        lock = RedisLock(
            self.redis_client,
            full_key,
            timeout=timeout,
            blocking_timeout=timeout if blocking else 0
        )
        
        acquired = await lock.acquire()
        if acquired:
            self.active_locks[lock_key] = lock
            logger.debug(f"Lock acquired: {lock_key}", extra={
                'operation': 'LOCK_ACQUIRE',
                'lock_key': lock_key,
                'timeout': timeout
            })
            return lock
        else:
            logger.warning(f"Failed to acquire lock: {lock_key}", extra={
                'operation': 'LOCK_ACQUIRE_FAILED',
                'lock_key': lock_key
            })
            raise Exception(f"Failed to acquire lock: {lock_key}")
    
    async def release_lock(self, lock_key: str) -> None:
        """Release distributed lock"""
        if lock_key in self.active_locks:
            lock = self.active_locks[lock_key]
            await lock.release()
            del self.active_locks[lock_key]
            
            logger.debug(f"Lock released: {lock_key}", extra={
                'operation': 'LOCK_RELEASE',
                'lock_key': lock_key
            })
    
    async def cleanup_locks(self) -> None:
        """Cleanup all active locks"""
        for lock_key in list(self.active_locks.keys()):
            await self.release_lock(lock_key)
        
        logger.info("All locks cleaned up", extra={
            'operation': 'LOCK_CLEANUP_ALL'
        })


class StateAnalytics:
    """Analytics and monitoring for state operations"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.operation_times: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
    async def record_operation(self, 
                             operation: str, 
                             scope: StateScope,
                             duration_ms: float,
                             success: bool = True) -> None:
        """Record state operation for analytics"""
        timestamp = datetime.now()
        
        # Update metrics
        self.metrics[f"{scope.value}_{operation}_count"] += 1
        if success:
            self.metrics[f"{scope.value}_{operation}_success"] += 1
        else:
            self.metrics[f"{scope.value}_{operation}_error"] += 1
            self.error_counts[f"{scope.value}_{operation}"] += 1
        
        # Record timing
        self.operation_times.append({
            'operation': operation,
            'scope': scope.value,
            'duration_ms': duration_ms,
            'timestamp': timestamp,
            'success': success
        })
        
        # Store in Redis for cross-system analytics
        analytics_key = f"analytics:state_operations:{timestamp.strftime('%Y%m%d%H')}"
        operation_data = {
            'operation': operation,
            'scope': scope.value,
            'duration_ms': duration_ms,
            'timestamp': timestamp.isoformat(),
            'success': success
        }
        
        await self.redis_client.lpush(analytics_key, json.dumps(operation_data))
        await self.redis_client.expire(analytics_key, 86400)  # 24 hours
    
    async def get_performance_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance metrics for the specified time period"""
        metrics = {
            'total_operations': sum(len(self.operation_times)),
            'success_rate': 0.0,
            'average_duration_ms': 0.0,
            'operations_by_scope': defaultdict(int),
            'error_summary': dict(self.error_counts)
        }
        
        if self.operation_times:
            successful_ops = sum(1 for op in self.operation_times if op['success'])
            metrics['success_rate'] = (successful_ops / len(self.operation_times)) * 100
            
            total_duration = sum(op['duration_ms'] for op in self.operation_times)
            metrics['average_duration_ms'] = total_duration / len(self.operation_times)
            
            for op in self.operation_times:
                metrics['operations_by_scope'][op['scope']] += 1
        
        return metrics
    
    async def get_system_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state system summary"""
        # Get Redis info
        redis_info = await self.redis_client.info()
        
        # Get key counts by scope
        key_counts = {}
        for scope in StateScope:
            pattern = f"state:{scope.value}:*"
            keys = await self.redis_client.keys(pattern)
            key_counts[scope.value] = len(keys)
        
        summary = {
            'redis_connected_clients': redis_info.get('connected_clients', 0),
            'redis_used_memory': redis_info.get('used_memory_human', '0'),
            'total_keys_by_scope': key_counts,
            'total_keys': sum(key_counts.values()),
            'performance_metrics': await self.get_performance_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary


class EnhancedSharedStateManager:
    """
    Enhanced shared state manager with advanced features
    
    Extends the basic SharedStateManager with:
    - Real-time event streaming
    - State versioning and conflict resolution  
    - Distributed locking
    - Performance analytics
    - Intelligent caching strategies
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 db: int = 0,
                 max_connections: int = 20,
                 enable_streaming: bool = True,
                 enable_analytics: bool = True):
        """
        Initialize enhanced shared state manager
        
        Args:
            redis_url: Redis connection URL
            db: Redis database number
            max_connections: Maximum connection pool size
            enable_streaming: Enable real-time event streaming
            enable_analytics: Enable state analytics
        """
        self.redis_url = redis_url
        self.db = db
        self.max_connections = max_connections
        self.enable_streaming = enable_streaming
        self.enable_analytics = enable_analytics
        
        # Redis client
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False
        
        # Enhanced components
        self.conflict_resolver = StateConflictResolver()
        self.lock_manager: Optional[DistributedLockManager] = None
        self.event_streamer: Optional[RealTimeStateStreamer] = None
        self.analytics: Optional[StateAnalytics] = None
        
        # State versioning
        self.state_versions: Dict[str, StateVersion] = {}
        
        logger.info("EnhancedSharedStateManager initialized", extra={
            'operation': 'ENHANCED_STATE_MANAGER_INIT',
            'redis_url': redis_url,
            'db': db,
            'streaming_enabled': enable_streaming,
            'analytics_enabled': enable_analytics
        })
    
    async def connect(self) -> None:
        """Establish enhanced Redis connection and initialize components"""
        try:
            # Create Redis connection
            self.redis_client = redis.from_url(
                self.redis_url,
                db=self.db,
                max_connections=self.max_connections,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Initialize enhanced components
            self.lock_manager = DistributedLockManager(self.redis_client)
            
            if self.enable_streaming:
                self.event_streamer = RealTimeStateStreamer(self.redis_client)
                await self.event_streamer.start_streaming()
            
            if self.enable_analytics:
                self.analytics = StateAnalytics(self.redis_client)
            
            self._connected = True
            
            logger.info("Enhanced SharedStateManager connected successfully", extra={
                'operation': 'ENHANCED_STATE_CONNECT_SUCCESS'
            })
            
        except Exception as e:
            logger.error(f"Failed to connect enhanced state manager: {str(e)}", extra={
                'operation': 'ENHANCED_STATE_CONNECT_ERROR',
                'error': str(e)
            })
            raise
    
    async def disconnect(self) -> None:
        """Disconnect and cleanup all components"""
        if self.event_streamer:
            await self.event_streamer.stop_streaming()
        
        if self.lock_manager:
            await self.lock_manager.cleanup_locks()
        
        if self.redis_client:
            await self.redis_client.close()
        
        self._connected = False
        
        logger.info("Enhanced SharedStateManager disconnected", extra={
            'operation': 'ENHANCED_STATE_DISCONNECT'
        })
    
    async def set_state_with_versioning(self, 
                                       scope: StateScope,
                                       key: str, 
                                       value: Any,
                                       source_system: str,
                                       expire_seconds: Optional[int] = None) -> StateVersion:
        """
        Set state with automatic versioning and conflict resolution
        
        Args:
            scope: State scope
            key: State key
            value: State value
            source_system: System making the change
            expire_seconds: Optional expiration time
            
        Returns:
            StateVersion object for the stored state
        """
        if not self._connected:
            raise RuntimeError("Enhanced state manager not connected")
        
        start_time = time.time()
        full_key = f"state:{scope.value}:{key}"
        version_key = f"version:{scope.value}:{key}"
        
        try:
            # Acquire lock for atomic operation
            async with await self.lock_manager.acquire_lock(f"state_write_{full_key}"):
                # Get current version if exists
                current_version_data = await self.redis_client.get(version_key)
                current_version = None
                new_version_num = 1
                
                if current_version_data:
                    current_version = StateVersion(**json.loads(current_version_data))
                    new_version_num = current_version.version + 1
                
                # Create new version
                new_version = StateVersion.create(value, source_system, new_version_num)
                
                # Handle conflicts if current version exists
                if current_version and current_version.checksum != new_version.checksum:
                    resolved_version = await self.conflict_resolver.resolve_conflict(
                        current_version, new_version
                    )
                    new_version = resolved_version
                
                # Store state and version
                state_data = json.dumps(value, default=str)
                version_data = json.dumps(asdict(new_version), default=str)
                
                pipeline = self.redis_client.pipeline()
                pipeline.set(full_key, state_data, ex=expire_seconds)
                pipeline.set(version_key, version_data, ex=expire_seconds)
                await pipeline.execute()
                
                # Cache version locally
                self.state_versions[full_key] = new_version
                
                # Publish event
                if self.event_streamer:
                    event = StateEvent(
                        event_id=str(uuid4()),
                        event_type=StateEventType.UPDATE,
                        scope=scope,
                        key=key,
                        data=value,
                        timestamp=datetime.now(),
                        source_system=source_system,
                        version=new_version.version
                    )
                    await self.event_streamer.publish_event(event)
                
                # Record analytics
                if self.analytics:
                    duration_ms = (time.time() - start_time) * 1000
                    await self.analytics.record_operation(
                        "set_state_versioned", scope, duration_ms, True
                    )
                
                logger.debug(f"State set with versioning: {full_key}", extra={
                    'operation': 'STATE_SET_VERSIONED',
                    'scope': scope.value,
                    'key': key,
                    'version': new_version.version,
                    'source_system': source_system
                })
                
                return new_version
        
        except Exception as e:
            if self.analytics:
                duration_ms = (time.time() - start_time) * 1000
                await self.analytics.record_operation(
                    "set_state_versioned", scope, duration_ms, False
                )
            
            logger.error(f"Error setting versioned state: {str(e)}", extra={
                'operation': 'STATE_SET_VERSIONED_ERROR',
                'scope': scope.value,
                'key': key,
                'error': str(e)
            })
            raise
    
    async def get_state_with_version(self, 
                                   scope: StateScope, 
                                   key: str) -> Optional[Tuple[Any, StateVersion]]:
        """
        Get state with version information
        
        Args:
            scope: State scope
            key: State key
            
        Returns:
            Tuple of (state_data, version) or None if not found
        """
        if not self._connected:
            raise RuntimeError("Enhanced state manager not connected")
        
        start_time = time.time()
        full_key = f"state:{scope.value}:{key}"
        version_key = f"version:{scope.value}:{key}"
        
        try:
            # Get both state and version data
            pipeline = self.redis_client.pipeline()
            pipeline.get(full_key)
            pipeline.get(version_key)
            results = await pipeline.execute()
            
            state_data, version_data = results
            
            if state_data is None:
                return None
            
            # Parse state data
            parsed_state = json.loads(state_data)
            
            # Parse version data
            version = None
            if version_data:
                version = StateVersion(**json.loads(version_data))
            
            # Record analytics
            if self.analytics:
                duration_ms = (time.time() - start_time) * 1000
                await self.analytics.record_operation(
                    "get_state_versioned", scope, duration_ms, True
                )
            
            return (parsed_state, version)
        
        except Exception as e:
            if self.analytics:
                duration_ms = (time.time() - start_time) * 1000
                await self.analytics.record_operation(
                    "get_state_versioned", scope, duration_ms, False
                )
            
            logger.error(f"Error getting versioned state: {str(e)}", extra={
                'operation': 'STATE_GET_VERSIONED_ERROR',
                'scope': scope.value,
                'key': key,
                'error': str(e)
            })
            raise
    
    async def get_state_analytics(self) -> Dict[str, Any]:
        """Get comprehensive state analytics"""
        if not self.analytics:
            return {"error": "Analytics not enabled"}
        
        return await self.analytics.get_system_state_summary()
    
    async def cleanup_expired_state(self, scope: StateScope, max_age_hours: int = 24) -> int:
        """
        Cleanup expired state data for a specific scope
        
        Args:
            scope: State scope to cleanup
            max_age_hours: Maximum age in hours for state data
            
        Returns:
            Number of keys cleaned up
        """
        if not self._connected:
            raise RuntimeError("Enhanced state manager not connected")
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        pattern = f"version:{scope.value}:*"
        
        keys_to_delete = []
        cursor = '0'
        
        while cursor != 0:
            cursor, keys = await self.redis_client.scan(cursor=cursor, match=pattern, count=100)
            
            for version_key in keys:
                version_data = await self.redis_client.get(version_key)
                if version_data:
                    version = StateVersion(**json.loads(version_data))
                    if version.timestamp < cutoff_time:
                        # Add both version and state keys for deletion
                        state_key = version_key.replace("version:", "state:")
                        keys_to_delete.extend([version_key, state_key])
        
        # Delete expired keys
        if keys_to_delete:
            await self.redis_client.delete(*keys_to_delete)
        
        cleaned_count = len(keys_to_delete) // 2  # Divide by 2 since we delete both state and version
        
        logger.info(f"Cleaned up {cleaned_count} expired state entries", extra={
            'operation': 'STATE_CLEANUP',
            'scope': scope.value,
            'cleaned_count': cleaned_count,
            'max_age_hours': max_age_hours
        })
        
        return cleaned_count 

    # ============================================================================
    # Legacy Compatibility Methods
    # ============================================================================
    
    async def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Legacy compatibility method for get_state
        Maps to enhanced state management with GLOBAL scope
        """
        result = await self.get_state_with_version(StateScope.GLOBAL, key)
        return result[0] if result else None
    
    async def set_state(self, key: str, value: Dict[str, Any], 
                       expire_seconds: Optional[int] = None) -> None:
        """
        Legacy compatibility method for set_state  
        Maps to enhanced state management with GLOBAL scope
        """
        await self.set_state_with_versioning(
            scope=StateScope.GLOBAL,
            key=key,
            value=value,
            source_system="unified_controller",
            expire_seconds=expire_seconds
        )
    
    async def add_task_to_history(self, task_result) -> None:
        """
        Add task execution result to history
        
        Args:
            task_result: ExecutionResult object or dict
        """
        try:
            # Convert to dict if needed
            if hasattr(task_result, '__dict__'):
                task_data = asdict(task_result)
            else:
                task_data = task_result
            
            # Store in SESSION scope for task history
            history_key = f"task_history_{task_data.get('task_id', str(uuid4()))}"
            await self.set_state_with_versioning(
                scope=StateScope.SESSION,
                key=history_key,
                value=task_data,
                source_system="unified_controller",
                expire_seconds=86400  # 24 hours
            )
            
            # Also maintain a list of recent tasks
            recent_tasks = await self.get_state("recent_task_history") or []
            recent_tasks.append({
                'task_id': task_data.get('task_id'),
                'description': task_data.get('result', {}).get('description', 'Unknown'),
                'success': task_data.get('success', False),
                'completed_at': task_data.get('completed_at', datetime.now().isoformat()),
                'execution_time_ms': task_data.get('execution_time_ms', 0)
            })
            
            # Keep only last 100 tasks
            recent_tasks = recent_tasks[-100:]
            await self.set_state("recent_task_history", recent_tasks, expire_seconds=86400)
            
        except Exception as e:
            logger.error(f"Error adding task to history: {str(e)}", extra={
                'operation': 'ADD_TASK_HISTORY_ERROR',
                'task_id': getattr(task_result, 'task_id', 'unknown'),
                'error': str(e)
            })
    
    async def get_task_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent task execution history
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of recent task summaries
        """
        recent_tasks = await self.get_state("recent_task_history") or []
        return recent_tasks[-limit:] if recent_tasks else []
    
    async def update_mcp_registry(self, mcp_data: Dict[str, Any]) -> None:
        """
        Update MCP registry with new MCP information
        
        Args:
            mcp_data: MCP information to store
        """
        await self.set_state_with_versioning(
            scope=StateScope.SYSTEM,
            key=f"mcp_registry_{mcp_data.get('name', str(uuid4()))}",
            value=mcp_data,
            source_system="alita_manager",
            expire_seconds=3600  # 1 hour
        )
    
    async def get_available_mcps(self) -> List[Dict[str, Any]]:
        """
        Get list of available MCPs from registry
        
        Returns:
            List of available MCP information
        """
        pattern = "state:system:mcp_registry_*"
        mcps = []
        
        cursor = '0'
        while cursor != 0:
            cursor, keys = await self.redis_client.scan(cursor=cursor, match=pattern, count=100)
            
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    mcps.append(json.loads(data))
        
        return mcps
    
    async def update_kgot_graph_state(self, graph_state: Dict[str, Any]) -> None:
        """
        Update KGoT knowledge graph state
        
        Args:
            graph_state: Current graph state information
        """
        await self.set_state_with_versioning(
            scope=StateScope.SYSTEM,
            key="kgot_graph_state",
            value=graph_state,
            source_system="kgot_controller",
            expire_seconds=3600  # 1 hour
        )
    
    async def get_kgot_graph_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current KGoT knowledge graph state
        
        Returns:
            Current graph state or None if not available
        """
        result = await self.get_state_with_version(StateScope.SYSTEM, "kgot_graph_state")
        return result[0] if result else None
    
    async def update_budget_usage(self, cost_data: Dict[str, float]) -> None:
        """
        Update system budget usage tracking
        
        Args:
            cost_data: Cost breakdown data
        """
        current_budget = await self.get_state("system_budget") or {
            'total_allocated': 100.0,
            'used': 0.0,
            'remaining': 100.0,
            'last_updated': datetime.now().isoformat()
        }
        
        # Update usage
        total_cost = sum(cost_data.values())
        current_budget['used'] += total_cost
        current_budget['remaining'] = current_budget['total_allocated'] - current_budget['used']
        current_budget['last_updated'] = datetime.now().isoformat()
        
        await self.set_state("system_budget", current_budget, expire_seconds=86400)
    
    async def get_system_budget(self) -> Dict[str, Any]:
        """
        Get current system budget status
        
        Returns:
            Budget status information
        """
        return await self.get_state("system_budget") or {
            'total_allocated': 100.0,
            'used': 0.0,
            'remaining': 100.0,
            'last_updated': datetime.now().isoformat()
        }