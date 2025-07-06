"""
Enhanced Logging System for Unified System Controller

This module provides comprehensive structured logging capabilities that integrate with
the existing Winston logging infrastructure, including:

- Winston-style structured logging in Python
- Operation and request correlation tracking
- Performance metrics logging
- Error tracking and alerting integration
- Multi-level log aggregation and filtering
- Contextual logging with metadata enrichment
- Integration with monitoring and alerting systems

@module EnhancedLoggingSystem
@author AI Assistant
@date 2025-01-22
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from contextvars import ContextVar
from functools import wraps
import traceback
import uuid
import redis
from contextlib import contextmanager

# Import shared state utilities
from .shared_state_utilities import (
    EnhancedSharedStateManager, 
    StateScope, 
    StateEventType
)

# Context variable for request correlation
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})


class LogLevel(Enum):
    """Enhanced log levels matching Winston configuration"""
    ERROR = "error"      # System errors, failures
    WARN = "warn"        # Warning conditions
    INFO = "info"        # General information about system operation
    HTTP = "http"        # HTTP requests/responses
    VERBOSE = "verbose"  # Verbose information
    DEBUG = "debug"      # Debug information
    SILLY = "silly"      # Everything


class LogCategory(Enum):
    """Log categories for different system components"""
    SYSTEM = "system"
    API_CALL = "api_call"
    MODEL_USAGE = "model_usage"
    GRAPH_OPERATION = "graph_operation"
    CIRCUIT_BREAKER = "circuit_breaker"
    LOAD_BALANCING = "load_balancing"
    STATE_MANAGEMENT = "state_management"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ERROR_HANDLING = "error_handling"
    API = "api"
    MODEL = "model"
    GRAPH = "graph"
    STATE = "state"
    ROUTING = "routing"
    MONITORING = "monitoring"


@dataclass
class LogEntry:
    """Structured log entry with Winston-compatible format"""
    timestamp: str
    level: str
    component: str
    operation: str
    message: str
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    duration_ms: Optional[float] = None
    category: str = LogCategory.SYSTEM.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Remove None values
        return {k: v for k, v in result.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


class PerformanceTracker:
    """Track performance metrics for operations"""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.slow_operations: List[LogEntry] = []
        self.error_counts: Dict[str, int] = {}
    
    def record_operation(self, operation: str, duration_ms: float, success: bool = True) -> None:
        """Record operation performance metrics"""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        
        self.operation_times[operation].append(duration_ms)
        
        # Track slow operations (> 5 seconds)
        if duration_ms > 5000:
            entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level=LogLevel.WARN.value,
                component="PERFORMANCE",
                operation=operation,
                message=f"Slow operation detected: {operation} took {duration_ms:.2f}ms",
                duration_ms=duration_ms,
                category=LogCategory.PERFORMANCE.value
            )
            self.slow_operations.append(entry)
        
        # Track errors
        if not success:
            self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        summary = {
            'operations': {},
            'slow_operations_count': len(self.slow_operations),
            'error_counts': dict(self.error_counts)
        }
        
        for operation, times in self.operation_times.items():
            if times:
                summary['operations'][operation] = {
                    'count': len(times),
                    'avg_duration_ms': sum(times) / len(times),
                    'min_duration_ms': min(times),
                    'max_duration_ms': max(times),
                    'total_duration_ms': sum(times)
                }
        
        return summary


class CorrelationTracker:
    """Track request and operation correlations"""
    
    def __init__(self):
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.operation_traces: Dict[str, List[str]] = {}
    
    def start_request(self, request_id: str, operation: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start tracking a new request"""
        self.active_requests[request_id] = {
            'operation': operation,
            'start_time': time.time(),
            'metadata': metadata or {},
            'sub_operations': []
        }
        
        # Set context
        context = {
            'request_id': request_id,
            'correlation_id': str(uuid.uuid4()),
            'operation': operation
        }
        request_context.set(context)
    
    def add_operation(self, request_id: str, operation: str) -> None:
        """Add a sub-operation to an existing request"""
        if request_id in self.active_requests:
            self.active_requests[request_id]['sub_operations'].append({
                'operation': operation,
                'timestamp': time.time()
            })
    
    def end_request(self, request_id: str, success: bool = True, error: Optional[str] = None) -> float:
        """End request tracking and return duration"""
        if request_id in self.active_requests:
            request_info = self.active_requests[request_id]
            duration = (time.time() - request_info['start_time']) * 1000  # Convert to ms
            
            # Store trace for analysis
            self.operation_traces[request_id] = {
                'operation': request_info['operation'],
                'duration_ms': duration,
                'success': success,
                'error': error,
                'sub_operations': request_info['sub_operations'],
                'completed_at': datetime.now().isoformat()
            }
            
            del self.active_requests[request_id]
            return duration
        
        return 0.0


class StructuredLogger:
    """
    Enhanced logging system providing Winston-style functionality in Python
    with structured logging, correlation tracking, and performance monitoring
    """
    
    def __init__(self, 
                 component_name: str,
                 log_level: LogLevel = LogLevel.INFO,
                 correlation_id: Optional[str] = None,
                 redis_client: Optional[redis.Redis] = None):
        """
        Initialize the structured logger with Winston-style configuration
        
        Args:
            component_name: Name of the component using this logger
            log_level: Minimum log level to output
            correlation_id: Optional correlation ID for request tracking
            redis_client: Optional Redis client for centralized logging
        """
        self.component_name = component_name
        self.log_level = log_level
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.redis_client = redis_client
        self.performance_context = {}
        
        # Configure Python logger with structured formatting
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure the underlying Python logger with structured output"""
        self.logger = logging.getLogger(f"unified_system.{self.component_name}")
        self.logger.setLevel(self._get_python_log_level())
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create console handler with structured formatter
        console_handler = logging.StreamHandler()
        formatter = StructuredFormatter(self.component_name, self.correlation_id)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler for persistent logging
        log_dir = Path("logs/unified_system")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / f"{self.component_name}.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _get_python_log_level(self) -> int:
        """Convert Winston log level to Python logging level"""
        level_mapping = {
            LogLevel.ERROR: logging.ERROR,
            LogLevel.WARN: logging.WARNING,
            LogLevel.INFO: logging.INFO,
            LogLevel.HTTP: logging.INFO,
            LogLevel.VERBOSE: logging.DEBUG,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.SILLY: logging.DEBUG
        }
        return level_mapping.get(self.log_level, logging.INFO)
    
    def _should_log(self, level: LogLevel) -> bool:
        """Determine if message should be logged based on current log level"""
        level_hierarchy = [
            LogLevel.ERROR, LogLevel.WARN, LogLevel.INFO, 
            LogLevel.HTTP, LogLevel.VERBOSE, LogLevel.DEBUG, LogLevel.SILLY
        ]
        return level_hierarchy.index(level) <= level_hierarchy.index(self.log_level)
    
    def _create_log_entry(self, 
                         level: LogLevel,
                         message: str,
                         category: LogCategory,
                         metadata: Optional[Dict[str, Any]] = None,
                         error: Optional[Exception] = None) -> Dict[str, Any]:
        """Create a structured log entry with all necessary metadata"""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.value,
            "category": category.value,
            "component": self.component_name,
            "correlation_id": self.correlation_id,
            "message": message,
            "metadata": metadata or {}
        }
        
        if error:
            entry["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            }
        
        return entry
    
    def _emit_log(self, entry: Dict[str, Any]):
        """Emit log entry through multiple channels"""
        # Log to Python logger
        python_level = {
            LogLevel.ERROR: logging.ERROR,
            LogLevel.WARN: logging.WARNING,
            LogLevel.INFO: logging.INFO,
            LogLevel.HTTP: logging.INFO,
            LogLevel.VERBOSE: logging.DEBUG,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.SILLY: logging.DEBUG
        }.get(LogLevel(entry["level"]), logging.INFO)
        
        self.logger.log(python_level, json.dumps(entry, indent=2))
        
        # Send to Redis for centralized logging if available
        if self.redis_client:
            try:
                self.redis_client.lpush("unified_system:logs", json.dumps(entry))
                self.redis_client.expire("unified_system:logs", 86400)  # 24 hours
            except Exception as e:
                # Avoid infinite recursion by using basic logging
                self.logger.error(f"Failed to send log to Redis: {e}")
    
    # Core logging methods
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
              metadata: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None):
        """Log error level message"""
        if self._should_log(LogLevel.ERROR):
            entry = self._create_log_entry(LogLevel.ERROR, message, category, metadata, error)
            self._emit_log(entry)
    
    def warn(self, message: str, category: LogCategory = LogCategory.SYSTEM,
             metadata: Optional[Dict[str, Any]] = None):
        """Log warning level message"""
        if self._should_log(LogLevel.WARN):
            entry = self._create_log_entry(LogLevel.WARN, message, category, metadata)
            self._emit_log(entry)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM,
             metadata: Optional[Dict[str, Any]] = None):
        """Log info level message"""
        if self._should_log(LogLevel.INFO):
            entry = self._create_log_entry(LogLevel.INFO, message, category, metadata)
            self._emit_log(entry)
    
    def http(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log HTTP request/response information"""
        if self._should_log(LogLevel.HTTP):
            entry = self._create_log_entry(LogLevel.HTTP, message, LogCategory.API, metadata)
            self._emit_log(entry)
    
    def verbose(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                metadata: Optional[Dict[str, Any]] = None):
        """Log verbose level message"""
        if self._should_log(LogLevel.VERBOSE):
            entry = self._create_log_entry(LogLevel.VERBOSE, message, category, metadata)
            self._emit_log(entry)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM,
              metadata: Optional[Dict[str, Any]] = None):
        """Log debug level message"""
        if self._should_log(LogLevel.DEBUG):
            entry = self._create_log_entry(LogLevel.DEBUG, message, category, metadata)
            self._emit_log(entry)
    
    # Specialized logging methods for unified system components
    def log_api_call(self, method: str, endpoint: str, status_code: int, 
                     response_time_ms: float, metadata: Optional[Dict[str, Any]] = None):
        """Log API call with performance metrics"""
        api_metadata = {
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            **(metadata or {})
        }
        
        level = LogLevel.ERROR if status_code >= 500 else LogLevel.WARN if status_code >= 400 else LogLevel.HTTP
        message = f"{method} {endpoint} - {status_code} ({response_time_ms:.2f}ms)"
        
        if self._should_log(level):
            entry = self._create_log_entry(level, message, LogCategory.API, api_metadata)
            self._emit_log(entry)
    
    def log_model_usage(self, model_name: str, operation: str, token_count: int, 
                       cost: float, success: bool, metadata: Optional[Dict[str, Any]] = None):
        """Log AI model usage for cost tracking and performance monitoring"""
        model_metadata = {
            "model_name": model_name,
            "operation": operation,
            "token_count": token_count,
            "cost": cost,
            "success": success,
            **(metadata or {})
        }
        
        level = LogLevel.ERROR if not success else LogLevel.INFO
        message = f"Model {model_name} - {operation} ({'success' if success else 'failed'}) - {token_count} tokens, ${cost:.4f}"
        
        if self._should_log(level):
            entry = self._create_log_entry(level, message, LogCategory.MODEL, model_metadata)
            self._emit_log(entry)
    
    def log_graph_operation(self, operation: str, graph_backend: str, node_count: int, 
                           execution_time_ms: float, success: bool, 
                           metadata: Optional[Dict[str, Any]] = None):
        """Log knowledge graph operations for performance monitoring"""
        graph_metadata = {
            "operation": operation,
            "graph_backend": graph_backend,
            "node_count": node_count,
            "execution_time_ms": execution_time_ms,
            "success": success,
            **(metadata or {})
        }
        
        level = LogLevel.ERROR if not success else LogLevel.INFO
        message = f"Graph {operation} on {graph_backend} - {node_count} nodes ({execution_time_ms:.2f}ms)"
        
        if self._should_log(level):
            entry = self._create_log_entry(level, message, LogCategory.GRAPH, graph_metadata)
            self._emit_log(entry)
    
    def log_circuit_breaker_event(self, circuit_name: str, event_type: str, 
                                 failure_count: int, metadata: Optional[Dict[str, Any]] = None):
        """Log circuit breaker state changes and events"""
        cb_metadata = {
            "circuit_name": circuit_name,
            "event_type": event_type,
            "failure_count": failure_count,
            **(metadata or {})
        }
        
        level = LogLevel.WARN if event_type in ["OPEN", "HALF_OPEN"] else LogLevel.INFO
        message = f"Circuit breaker {circuit_name} - {event_type} (failures: {failure_count})"
        
        if self._should_log(level):
            entry = self._create_log_entry(level, message, LogCategory.CIRCUIT_BREAKER, cb_metadata)
            self._emit_log(entry)
    
    def log_state_operation(self, operation: str, key: str, scope: str, 
                           success: bool, metadata: Optional[Dict[str, Any]] = None):
        """Log shared state operations for debugging distributed state issues"""
        state_metadata = {
            "operation": operation,
            "key": key,
            "scope": scope,
            "success": success,
            **(metadata or {})
        }
        
        level = LogLevel.ERROR if not success else LogLevel.VERBOSE
        message = f"State {operation} - {scope}:{key} ({'success' if success else 'failed'})"
        
        if self._should_log(level):
            entry = self._create_log_entry(level, message, LogCategory.STATE, state_metadata)
            self._emit_log(entry)
    
    def log_routing_decision(self, task_type: str, complexity: str, strategy: str, 
                            selected_system: str, confidence: float, 
                            metadata: Optional[Dict[str, Any]] = None):
        """Log task routing decisions for optimization analysis"""
        routing_metadata = {
            "task_type": task_type,
            "complexity": complexity,
            "strategy": strategy,
            "selected_system": selected_system,
            "confidence": confidence,
            **(metadata or {})
        }
        
        message = f"Routing {task_type} ({complexity}) via {strategy} -> {selected_system} (confidence: {confidence:.2f})"
        
        if self._should_log(LogLevel.INFO):
            entry = self._create_log_entry(LogLevel.INFO, message, LogCategory.ROUTING, routing_metadata)
            self._emit_log(entry)
    
    @contextmanager
    def performance_timer(self, operation_name: str, category: LogCategory = LogCategory.PERFORMANCE):
        """Context manager for timing operations and automatically logging performance"""
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        self.verbose(f"Starting {operation_name}", category, {"operation_id": operation_id})
        
        try:
            yield operation_id
            duration_ms = (time.time() - start_time) * 1000
            self.info(f"Completed {operation_name} in {duration_ms:.2f}ms", category, {
                "operation_id": operation_id,
                "duration_ms": duration_ms,
                "success": True
            })
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.error(f"Failed {operation_name} after {duration_ms:.2f}ms", category, {
                "operation_id": operation_id,
                "duration_ms": duration_ms,
                "success": False
            }, error=e)
            raise
    
    def create_child_logger(self, child_component: str, 
                           new_correlation_id: Optional[str] = None) -> 'StructuredLogger':
        """Create a child logger with inherited configuration but separate correlation ID"""
        return StructuredLogger(
            component_name=f"{self.component_name}.{child_component}",
            log_level=self.log_level,
            correlation_id=new_correlation_id or str(uuid.uuid4()),
            redis_client=self.redis_client
        )

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log output matching Winston style"""
    
    def __init__(self, component_name: str, correlation_id: str):
        super().__init__()
        self.component_name = component_name
        self.correlation_id = correlation_id
    
    def format(self, record):
        # Check if the record message is already JSON (from our structured logger)
        try:
            # If it's already JSON, just return it
            json.loads(record.getMessage())
            return record.getMessage()
        except (json.JSONDecodeError, ValueError):
            # If it's not JSON, create a structured format
            return json.dumps({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname.lower(),
                "component": self.component_name,
                "correlation_id": self.correlation_id,
                "message": record.getMessage(),
                "metadata": {}
            }, indent=2)

class LoggingManager:
    """
    Centralized logging manager for the unified system
    Provides consistent logger instances and configuration management
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, 
                 default_log_level: LogLevel = LogLevel.INFO):
        self.redis_client = redis_client
        self.default_log_level = default_log_level
        self._loggers: Dict[str, StructuredLogger] = {}
    
    def get_logger(self, component_name: str, 
                   correlation_id: Optional[str] = None,
                   log_level: Optional[LogLevel] = None) -> StructuredLogger:
        """Get or create a logger for a specific component"""
        logger_key = f"{component_name}:{correlation_id or 'default'}"
        
        if logger_key not in self._loggers:
            self._loggers[logger_key] = StructuredLogger(
                component_name=component_name,
                log_level=log_level or self.default_log_level,
                correlation_id=correlation_id,
                redis_client=self.redis_client
            )
        
        return self._loggers[logger_key]
    
    def set_global_log_level(self, log_level: LogLevel):
        """Update log level for all existing loggers"""
        self.default_log_level = log_level
        for logger in self._loggers.values():
            logger.log_level = log_level
            logger._setup_logger()
    
    async def get_recent_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent logs from Redis for monitoring dashboard"""
        if not self.redis_client:
            return []
        
        try:
            log_entries = self.redis_client.lrange("unified_system:logs", 0, count - 1)
            return [json.loads(entry) for entry in log_entries]
        except Exception as e:
            # Create a basic logger to avoid recursion
            basic_logger = logging.getLogger("logging_manager")
            basic_logger.error(f"Failed to retrieve logs from Redis: {e}")
            return []


# Decorator for automatic operation logging
def log_operation(logger: StructuredLogger, operation: str, log_args: bool = False, log_result: bool = False):
    """
    Decorator to automatically log function operations
    
    Args:
        logger: Structured logger instance
        operation: Operation name
        log_args: Whether to log function arguments
        log_result: Whether to log function result
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())
            
            # Prepare metadata
            metadata = {}
            if log_args:
                metadata['args'] = str(args)
                metadata['kwargs'] = str(kwargs)
            
            # Start operation
            start_time = time.time()
            logger.start_request(request_id, operation, metadata)
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Log success
                duration_ms = (time.time() - start_time) * 1000
                success_metadata = metadata.copy()
                
                if log_result:
                    success_metadata['result'] = str(result)[:1000]  # Truncate large results
                
                logger.end_request(request_id, True)
                return result
                
            except Exception as e:
                # Log failure
                logger.end_request(request_id, False, str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())
            
            # Prepare metadata
            metadata = {}
            if log_args:
                metadata['args'] = str(args)
                metadata['kwargs'] = str(kwargs)
            
            # Start operation
            start_time = time.time()
            logger.start_request(request_id, operation, metadata)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                duration_ms = (time.time() - start_time) * 1000
                success_metadata = metadata.copy()
                
                if log_result:
                    success_metadata['result'] = str(result)[:1000]  # Truncate large results
                
                logger.end_request(request_id, True)
                return result
                
            except Exception as e:
                # Log failure
                logger.end_request(request_id, False, str(e))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Factory function for creating loggers
def get_enhanced_logger(component: str, 
                       shared_state: Optional[EnhancedSharedStateManager] = None,
                       **kwargs) -> StructuredLogger:
    """
    Factory function to create enhanced loggers
    
    Args:
        component: Component name
        shared_state: Shared state manager
        **kwargs: Additional logger configuration
        
    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(component, **kwargs) 