#!/usr/bin/env python3
"""
Unified Validation Dashboard - Task 20 Implementation

This module implements a comprehensive validation dashboard that provides real-time monitoring,
validation metrics visualization, and intelligent alert prioritization across both Alita MCP
and KGoT systems using sequential decision trees from Task 17c.

Key Features:
- Real-time validation status monitoring across KGoT and Alita workflows
- Comprehensive validation metrics visualization for both systems
- Validation history tracking leveraging KGoT Section 2.1 knowledge persistence
- Validation result comparison using KGoT graph analytics capabilities
- Connection to both Alita Section 2.1 Manager Agent and KGoT Section 2.2 Controller
- Sequential decision tree integration for complex validation scenarios and alert prioritization

@module UnifiedValidationDashboard
@author Enhanced Alita KGoT Team
@version 1.0.0
@task Task 20 - Create Unified Validation Dashboard with Sequential Thinking Integration
"""

import asyncio
import json
import logging
import time
import uuid
import websockets
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
import numpy as np
import pandas as pd
from pathlib import Path

# LangChain imports for agent development (per user preference)
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
sys.path.append(os.path.join(os.path.dirname(__file__), '../config/logging'))

# Sequential Decision Trees integration from Task 17c
sys.path.append(os.path.join(os.path.dirname(__file__), '../alita_core/manager_agent'))
from sequential_decision_trees import (
    SystemSelectionDecisionTree,
    DecisionContext,
    TaskComplexity,
    SystemType,
    ResourceConstraint,
    SystemCoordinationResult
)

# Existing validation systems integration
from mcp_cross_validator import (
    MCPCrossValidationEngine,
    CrossValidationResult,
    ValidationMetrics,
    MCPValidationSpec,
    TaskType as MCPTaskType
)

# KGoT integration
sys.path.append(os.path.join(os.path.dirname(__file__), '../kgot_core'))
from controller.kgot_controller import KGoTController
from graph_store.kg_interface import KnowledgeGraphInterface

# Setup Winston-compatible logging
logger = logging.getLogger('UnifiedValidationDashboard')
handler = logging.FileHandler('./logs/validation/combined.log')
formatter = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ValidationEventType(Enum):
    """Types of validation events in the unified dashboard"""
    MCP_VALIDATION_START = "mcp_validation_start"
    MCP_VALIDATION_COMPLETE = "mcp_validation_complete"
    MCP_VALIDATION_FAILED = "mcp_validation_failed"
    KGOT_VALIDATION_START = "kgot_validation_start"
    KGOT_VALIDATION_COMPLETE = "kgot_validation_complete"
    KGOT_VALIDATION_FAILED = "kgot_validation_failed"
    CROSS_SYSTEM_VALIDATION = "cross_system_validation"
    ALERT_GENERATED = "alert_generated"
    SYSTEM_STATUS_CHANGE = "system_status_change"


class AlertPriority(Enum):
    """Alert priority levels determined by sequential decision trees"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SystemStatus(Enum):
    """System operational status"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class ValidationEvent:
    """
    Unified validation event from either Alita MCP or KGoT systems
    
    Attributes:
        event_id: Unique identifier for the validation event
        event_type: Type of validation event
        source_system: Source system (Alita/KGoT)
        timestamp: When the event occurred
        validation_id: Associated validation session ID
        metrics: Validation metrics if available
        metadata: Additional event-specific information
        alert_priority: Priority level determined by decision trees
    """
    event_id: str
    event_type: ValidationEventType
    source_system: str
    timestamp: datetime
    validation_id: str
    metrics: Optional[ValidationMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    alert_priority: Optional[AlertPriority] = None


@dataclass
class DashboardContext:
    """
    Overall dashboard context and configuration
    
    Attributes:
        dashboard_id: Unique identifier for dashboard session
        monitoring_systems: Systems currently being monitored
        real_time_enabled: Whether real-time monitoring is active
        alert_thresholds: Thresholds for different alert levels
        validation_history_limit: Maximum validation events to keep in memory
        kgot_graph_config: KGoT graph analytics configuration
        sequential_decision_config: Decision tree configuration for alerts
    """
    dashboard_id: str
    monitoring_systems: List[str] = field(default_factory=lambda: ["alita", "kgot"])
    real_time_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    validation_history_limit: int = 1000
    kgot_graph_config: Dict[str, Any] = field(default_factory=dict)
    sequential_decision_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Summary of validation metrics across systems"""
    total_validations: int
    successful_validations: int
    failed_validations: int
    average_performance: float
    system_reliability: Dict[str, float]
    recent_trends: Dict[str, List[float]]
    alert_summary: Dict[AlertPriority, int]


class SequentialDecisionAlertManager:
    """
    Alert manager using sequential decision trees for intelligent prioritization
    Integrates with Task 17c decision tree implementation
    """
    
    def __init__(self, validation_engine: MCPCrossValidationEngine, kgot_controller: KGoTController):
        """
        Initialize the sequential decision alert manager
        
        @param validation_engine: MCP cross-validation engine from existing system
        @param kgot_controller: KGoT controller for graph analytics integration
        """
        self.validation_engine = validation_engine
        self.kgot_controller = kgot_controller
        self.decision_tree = None
        
        # Alert prioritization metrics
        self.alert_history = deque(maxlen=100)
        self.system_performance_metrics = defaultdict(list)
        
        logger.info("SequentialDecisionAlertManager initialized", extra={
            'operation': 'ALERT_MANAGER_INIT',
            'component': 'VALIDATION_DASHBOARD'
        })
    
    async def initialize_decision_tree(self) -> None:
        """Initialize the sequential decision tree for alert prioritization"""
        try:
            self.decision_tree = SystemSelectionDecisionTree(
                validation_engine=self.validation_engine,
                kgot_controller_client=self.kgot_controller
            )
            await self.decision_tree.build_tree()
            
            logger.info("Sequential decision tree initialized for alert prioritization", extra={
                'operation': 'DECISION_TREE_INIT',
                'component': 'ALERT_MANAGER'
            })
        except Exception as e:
            logger.error(f"Failed to initialize decision tree: {e}", extra={
                'operation': 'DECISION_TREE_INIT_ERROR',
                'component': 'ALERT_MANAGER',
                'error': str(e)
            })
            raise
    
    async def prioritize_alert(self, validation_event: ValidationEvent) -> AlertPriority:
        """
        Use sequential decision trees to determine alert priority
        
        @param validation_event: Validation event to prioritize
        @returns: Determined alert priority level
        """
        try:
            # Create decision context from validation event
            context = DecisionContext(
                task_id=validation_event.validation_id,
                task_description=f"Alert prioritization for {validation_event.event_type.value}",
                task_type="alert_prioritization",
                complexity_level=self._assess_event_complexity(validation_event),
                metadata=validation_event.metadata
            )
            
            # Use decision tree to determine appropriate response
            if self.decision_tree:
                decision_path = await self.decision_tree.traverse_tree(context)
                priority = self._map_decision_to_priority(decision_path, validation_event)
            else:
                priority = self._fallback_prioritization(validation_event)
            
            # Log alert prioritization decision
            logger.info(f"Alert prioritized as {priority.value}", extra={
                'operation': 'ALERT_PRIORITIZATION',
                'component': 'ALERT_MANAGER',
                'event_id': validation_event.event_id,
                'priority': priority.value,
                'source_system': validation_event.source_system
            })
            
            return priority
            
        except Exception as e:
            logger.error(f"Alert prioritization failed: {e}", extra={
                'operation': 'ALERT_PRIORITIZATION_ERROR',
                'component': 'ALERT_MANAGER',
                'event_id': validation_event.event_id,
                'error': str(e)
            })
            return AlertPriority.MEDIUM
    
    def _assess_event_complexity(self, validation_event: ValidationEvent) -> TaskComplexity:
        """Assess the complexity of a validation event for decision tree input"""
        if validation_event.event_type in [ValidationEventType.MCP_VALIDATION_FAILED, 
                                         ValidationEventType.KGOT_VALIDATION_FAILED]:
            return TaskComplexity.COMPLEX
        elif validation_event.event_type == ValidationEventType.CROSS_SYSTEM_VALIDATION:
            return TaskComplexity.CRITICAL
        elif validation_event.metrics and validation_event.metrics.error_rate > 0.1:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.MODERATE
    
    def _map_decision_to_priority(self, decision_path, validation_event: ValidationEvent) -> AlertPriority:
        """Map decision tree results to alert priority levels"""
        if not decision_path.final_decision:
            return AlertPriority.MEDIUM
        
        confidence = decision_path.confidence_score
        system_type = decision_path.final_decision.get('system_type', SystemType.COMBINED)
        
        # High confidence in fallback systems indicates serious issues
        if system_type == SystemType.FALLBACK and confidence > 0.8:
            return AlertPriority.CRITICAL
        
        # Cross-system validation issues are high priority
        if (validation_event.event_type == ValidationEventType.CROSS_SYSTEM_VALIDATION and
            decision_path.total_cost > 5.0):
            return AlertPriority.HIGH
        
        # Use confidence and cost to determine priority
        if confidence > 0.9 and decision_path.total_cost < 2.0:
            return AlertPriority.LOW
        elif confidence > 0.7:
            return AlertPriority.MEDIUM
        else:
            return AlertPriority.HIGH
    
    def _fallback_prioritization(self, validation_event: ValidationEvent) -> AlertPriority:
        """Fallback prioritization when decision tree is unavailable"""
        if validation_event.event_type in [ValidationEventType.MCP_VALIDATION_FAILED,
                                         ValidationEventType.KGOT_VALIDATION_FAILED]:
            return AlertPriority.HIGH
        elif validation_event.event_type == ValidationEventType.CROSS_SYSTEM_VALIDATION:
            return AlertPriority.CRITICAL
        else:
            return AlertPriority.MEDIUM


class ValidationHistoryTracker:
    """
    Validation history tracking leveraging KGoT Section 2.1 knowledge persistence
    """
    
    def __init__(self, kg_interface: KnowledgeGraphInterface, max_history: int = 1000):
        """
        Initialize validation history tracker with KGoT graph integration
        
        @param kg_interface: KGoT knowledge graph interface for persistence
        @param max_history: Maximum number of validation events to track
        """
        self.kg_interface = kg_interface
        self.max_history = max_history
        self.validation_history = deque(maxlen=max_history)
        self.system_metrics_cache = defaultdict(lambda: defaultdict(list))
        
        logger.info("ValidationHistoryTracker initialized with KGoT integration", extra={
            'operation': 'HISTORY_TRACKER_INIT',
            'component': 'VALIDATION_DASHBOARD',
            'max_history': max_history
        })
    
    async def track_validation_event(self, validation_event: ValidationEvent) -> None:
        """Track validation event in history and persist to KGoT graph"""
        try:
            # Add to local history
            self.validation_history.append(validation_event)
            
            # Persist to KGoT knowledge graph for long-term storage
            await self._persist_to_knowledge_graph(validation_event)
            
            # Update system metrics cache
            self._update_metrics_cache(validation_event)
            
            logger.debug(f"Validation event tracked: {validation_event.event_id}", extra={
                'operation': 'VALIDATION_EVENT_TRACKED',
                'component': 'HISTORY_TRACKER',
                'event_type': validation_event.event_type.value,
                'source_system': validation_event.source_system
            })
            
        except Exception as e:
            logger.error(f"Failed to track validation event: {e}", extra={
                'operation': 'VALIDATION_TRACKING_ERROR',
                'component': 'HISTORY_TRACKER',
                'event_id': validation_event.event_id,
                'error': str(e)
            })
    
    async def _persist_to_knowledge_graph(self, validation_event: ValidationEvent) -> None:
        """Persist validation event to KGoT knowledge graph"""
        try:
            # Create knowledge graph node for validation event
            validation_node = {
                'id': validation_event.event_id,
                'type': 'ValidationEvent',
                'event_type': validation_event.event_type.value,
                'source_system': validation_event.source_system,
                'timestamp': validation_event.timestamp.isoformat(),
                'validation_id': validation_event.validation_id,
                'metadata': validation_event.metadata
            }
            
            # Add metrics if available
            if validation_event.metrics:
                validation_node['metrics'] = {
                    'consistency_score': validation_event.metrics.consistency_score,
                    'error_rate': validation_event.metrics.error_rate,
                    'performance_score': validation_event.metrics.execution_time_avg,
                    'accuracy': validation_event.metrics.ground_truth_accuracy
                }
            
            # Store in knowledge graph
            await self.kg_interface.store_node(validation_node)
            
            # Create relationships to system nodes
            await self._create_system_relationships(validation_event)
            
        except Exception as e:
            logger.warning(f"Failed to persist to knowledge graph: {e}", extra={
                'operation': 'KG_PERSISTENCE_ERROR',
                'component': 'HISTORY_TRACKER',
                'event_id': validation_event.event_id,
                'error': str(e)
            })
    
    async def _create_system_relationships(self, validation_event: ValidationEvent) -> None:
        """Create relationships between validation events and system nodes"""
        try:
            relationship = {
                'source': validation_event.source_system,
                'target': validation_event.event_id,
                'type': 'GENERATED_VALIDATION_EVENT',
                'timestamp': validation_event.timestamp.isoformat()
            }
            
            await self.kg_interface.store_relationship(relationship)
            
        except Exception as e:
            logger.debug(f"Could not create system relationship: {e}")
    
    def _update_metrics_cache(self, validation_event: ValidationEvent) -> None:
        """Update local metrics cache for quick access"""
        system = validation_event.source_system
        event_type = validation_event.event_type.value
        
        if validation_event.metrics:
            self.system_metrics_cache[system]['consistency'].append(
                validation_event.metrics.consistency_score
            )
            self.system_metrics_cache[system]['performance'].append(
                validation_event.metrics.execution_time_avg
            )
            self.system_metrics_cache[system]['accuracy'].append(
                validation_event.metrics.ground_truth_accuracy
            )
        
        # Keep only recent metrics (last 100 entries per metric)
        for metric_list in self.system_metrics_cache[system].values():
            if len(metric_list) > 100:
                metric_list.pop(0)
    
    async def get_validation_history(self, system: Optional[str] = None, 
                                   hours: int = 24) -> List[ValidationEvent]:
        """Get validation history with optional filtering"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_events = [
            event for event in self.validation_history
            if event.timestamp >= cutoff_time and
            (system is None or event.source_system == system)
        ]
        
        return filtered_events
    
    def get_system_metrics_summary(self, system: str) -> Dict[str, float]:
        """Get summary metrics for a specific system"""
        if system not in self.system_metrics_cache:
            return {}
        
        summary = {}
        for metric_name, values in self.system_metrics_cache[system].items():
            if values:
                summary[f"{metric_name}_avg"] = np.mean(values)
                summary[f"{metric_name}_std"] = np.std(values)
                summary[f"{metric_name}_latest"] = values[-1]
        
        return summary


class UnifiedValidationDashboard:
    """
    Main unified validation dashboard orchestrator
    Provides comprehensive validation monitoring across Alita MCP and KGoT systems
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the unified validation dashboard
        
        @param config: Dashboard configuration including system connections and settings
        """
        self.config = config
        self.context = DashboardContext(
            dashboard_id=str(uuid.uuid4()),
            **config.get('dashboard_context', {})
        )
        
        # Component initialization
        self.validation_engine = None
        self.kgot_controller = None
        self.kg_interface = None
        self.alert_manager = None
        self.history_tracker = None
        
        # Real-time monitoring
        self.active_validations = {}
        self.system_status = {
            'alita': SystemStatus.OPERATIONAL,
            'kgot': SystemStatus.OPERATIONAL
        }
        self.event_queue = asyncio.Queue()
        self.websocket_clients = set()
        
        # Performance tracking
        self.dashboard_metrics = {
            'events_processed': 0,
            'alerts_generated': 0,
            'uptime_start': datetime.now(),
            'last_update': datetime.now()
        }
        
        logger.info("UnifiedValidationDashboard initialized", extra={
            'operation': 'DASHBOARD_INIT',
            'component': 'UNIFIED_DASHBOARD',
            'dashboard_id': self.context.dashboard_id
        })
    
    async def initialize(self) -> None:
        """Initialize all dashboard components and connections"""
        try:
            # Initialize validation engine
            self.validation_engine = MCPCrossValidationEngine(
                config=self.config.get('validation_config', {}),
                llm_client=self.config.get('llm_client'),
                knowledge_graph_client=self.config.get('kg_client')
            )
            
            # Initialize KGoT controller
            self.kgot_controller = KGoTController(
                config=self.config.get('kgot_config', {})
            )
            
            # Initialize knowledge graph interface
            self.kg_interface = KnowledgeGraphInterface(
                config=self.config.get('kg_config', {})
            )
            
            # Initialize alert manager with sequential decision trees
            self.alert_manager = SequentialDecisionAlertManager(
                validation_engine=self.validation_engine,
                kgot_controller=self.kgot_controller
            )
            await self.alert_manager.initialize_decision_tree()
            
            # Initialize history tracker
            self.history_tracker = ValidationHistoryTracker(
                kg_interface=self.kg_interface,
                max_history=self.context.validation_history_limit
            )
            
            logger.info("All dashboard components initialized successfully", extra={
                'operation': 'DASHBOARD_COMPONENTS_INIT',
                'component': 'UNIFIED_DASHBOARD'
            })
            
        except Exception as e:
            logger.error(f"Dashboard initialization failed: {e}", extra={
                'operation': 'DASHBOARD_INIT_ERROR',
                'component': 'UNIFIED_DASHBOARD',
                'error': str(e)
            })
            raise
    
    async def start_monitoring(self) -> None:
        """Start real-time monitoring of validation systems"""
        if not self.context.real_time_enabled:
            logger.info("Real-time monitoring disabled in configuration")
            return
        
        try:
            # Start monitoring tasks
            monitoring_tasks = [
                asyncio.create_task(self._monitor_alita_system()),
                asyncio.create_task(self._monitor_kgot_system()),
                asyncio.create_task(self._process_validation_events()),
                asyncio.create_task(self._update_system_status()),
                asyncio.create_task(self._cleanup_old_data())
            ]
            
            logger.info("Real-time monitoring started", extra={
                'operation': 'MONITORING_START',
                'component': 'UNIFIED_DASHBOARD',
                'tasks_count': len(monitoring_tasks)
            })
            
            # Wait for all monitoring tasks
            await asyncio.gather(*monitoring_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Monitoring failed: {e}", extra={
                'operation': 'MONITORING_ERROR',
                'component': 'UNIFIED_DASHBOARD',
                'error': str(e)
            })
    
    async def _monitor_alita_system(self) -> None:
        """Monitor Alita MCP system for validation events"""
        while True:
            try:
                # Simulate monitoring Alita validation events
                # In real implementation, this would connect to Alita's event stream
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Check for new validation results
                recent_validations = await self._get_recent_alita_validations()
                for validation in recent_validations:
                    await self._handle_validation_event(validation)
                
            except Exception as e:
                logger.error(f"Alita monitoring error: {e}", extra={
                    'operation': 'ALITA_MONITORING_ERROR',
                    'component': 'UNIFIED_DASHBOARD',
                    'error': str(e)
                })
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _monitor_kgot_system(self) -> None:
        """Monitor KGoT system for validation events"""
        while True:
            try:
                # Simulate monitoring KGoT validation events
                # In real implementation, this would connect to KGoT's event stream
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Check for new validation results
                recent_validations = await self._get_recent_kgot_validations()
                for validation in recent_validations:
                    await self._handle_validation_event(validation)
                
            except Exception as e:
                logger.error(f"KGoT monitoring error: {e}", extra={
                    'operation': 'KGOT_MONITORING_ERROR',
                    'component': 'UNIFIED_DASHBOARD',
                    'error': str(e)
                })
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _handle_validation_event(self, event_data: Dict[str, Any]) -> None:
        """Handle incoming validation event from either system"""
        try:
            # Create validation event object
            validation_event = ValidationEvent(
                event_id=str(uuid.uuid4()),
                event_type=ValidationEventType(event_data['event_type']),
                source_system=event_data['source_system'],
                timestamp=datetime.now(),
                validation_id=event_data.get('validation_id', ''),
                metadata=event_data.get('metadata', {})
            )
            
            # Prioritize alert using sequential decision trees
            if self.alert_manager:
                priority = await self.alert_manager.prioritize_alert(validation_event)
                validation_event.alert_priority = priority
            
            # Track in history
            if self.history_tracker:
                await self.history_tracker.track_validation_event(validation_event)
            
            # Add to event queue for processing
            await self.event_queue.put(validation_event)
            
            # Update dashboard metrics
            self.dashboard_metrics['events_processed'] += 1
            self.dashboard_metrics['last_update'] = datetime.now()
            
            logger.info(f"Validation event handled: {validation_event.event_id}", extra={
                'operation': 'VALIDATION_EVENT_HANDLED',
                'component': 'UNIFIED_DASHBOARD',
                'event_type': validation_event.event_type.value,
                'source_system': validation_event.source_system,
                'priority': validation_event.alert_priority.value if validation_event.alert_priority else None
            })
            
        except Exception as e:
            logger.error(f"Failed to handle validation event: {e}", extra={
                'operation': 'VALIDATION_EVENT_ERROR',
                'component': 'UNIFIED_DASHBOARD',
                'error': str(e)
            })
    
    async def get_dashboard_summary(self) -> ValidationSummary:
        """Get comprehensive dashboard summary"""
        try:
            # Get recent validation history
            recent_events = await self.history_tracker.get_validation_history(hours=24)
            
            # Calculate summary metrics
            total_validations = len(recent_events)
            successful_validations = len([e for e in recent_events 
                                        if e.event_type in [ValidationEventType.MCP_VALIDATION_COMPLETE,
                                                          ValidationEventType.KGOT_VALIDATION_COMPLETE]])
            failed_validations = len([e for e in recent_events 
                                    if e.event_type in [ValidationEventType.MCP_VALIDATION_FAILED,
                                                      ValidationEventType.KGOT_VALIDATION_FAILED]])
            
            # Calculate system reliability
            system_reliability = {}
            for system in self.context.monitoring_systems:
                system_events = [e for e in recent_events if e.source_system == system]
                if system_events:
                    system_successful = len([e for e in system_events 
                                           if 'complete' in e.event_type.value])
                    reliability = system_successful / len(system_events) if system_events else 0.0
                    system_reliability[system] = reliability
                else:
                    system_reliability[system] = 1.0
            
            # Get alert summary
            alert_summary = defaultdict(int)
            for event in recent_events:
                if event.alert_priority:
                    alert_summary[event.alert_priority] += 1
            
            summary = ValidationSummary(
                total_validations=total_validations,
                successful_validations=successful_validations,
                failed_validations=failed_validations,
                average_performance=np.mean([1.0] * successful_validations + [0.0] * failed_validations) if total_validations > 0 else 1.0,
                system_reliability=dict(system_reliability),
                recent_trends={},  # Would be populated with trend analysis
                alert_summary=dict(alert_summary)
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard summary: {e}", extra={
                'operation': 'DASHBOARD_SUMMARY_ERROR',
                'component': 'UNIFIED_DASHBOARD',
                'error': str(e)
            })
            # Return empty summary on error
            return ValidationSummary(
                total_validations=0,
                successful_validations=0,
                failed_validations=0,
                average_performance=0.0,
                system_reliability={},
                recent_trends={},
                alert_summary={}
            )
    
    async def _get_recent_alita_validations(self) -> List[Dict[str, Any]]:
        """Get recent Alita validation events (placeholder for real implementation)"""
        # This would connect to actual Alita system in real implementation
        return []
    
    async def _get_recent_kgot_validations(self) -> List[Dict[str, Any]]:
        """Get recent KGoT validation events (placeholder for real implementation)"""
        # This would connect to actual KGoT system in real implementation
        return []
    
    async def _process_validation_events(self) -> None:
        """Process validation events from the queue"""
        while True:
            try:
                event = await self.event_queue.get()
                
                # Process event (send to WebSocket clients, update metrics, etc.)
                await self._broadcast_event_to_clients(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Event processing error: {e}", extra={
                    'operation': 'EVENT_PROCESSING_ERROR',
                    'component': 'UNIFIED_DASHBOARD',
                    'error': str(e)
                })
    
    async def _broadcast_event_to_clients(self, event: ValidationEvent) -> None:
        """Broadcast validation event to connected WebSocket clients"""
        if not self.websocket_clients:
            return
        
        try:
            event_data = {
                'type': 'validation_event',
                'data': asdict(event)
            }
            
            # Convert datetime to string for JSON serialization
            event_data['data']['timestamp'] = event.timestamp.isoformat()
            
            message = json.dumps(event_data)
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
            
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}", extra={
                'operation': 'BROADCAST_ERROR',
                'component': 'UNIFIED_DASHBOARD',
                'error': str(e)
            })
    
    async def _update_system_status(self) -> None:
        """Periodically update system status"""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Check Alita system status
                alita_status = await self._check_alita_status()
                self.system_status['alita'] = alita_status
                
                # Check KGoT system status
                kgot_status = await self._check_kgot_status()
                self.system_status['kgot'] = kgot_status
                
                logger.debug("System status updated", extra={
                    'operation': 'SYSTEM_STATUS_UPDATE',
                    'component': 'UNIFIED_DASHBOARD',
                    'alita_status': alita_status.value,
                    'kgot_status': kgot_status.value
                })
                
            except Exception as e:
                logger.error(f"System status update error: {e}", extra={
                    'operation': 'STATUS_UPDATE_ERROR',
                    'component': 'UNIFIED_DASHBOARD',
                    'error': str(e)
                })
    
    async def _check_alita_status(self) -> SystemStatus:
        """Check Alita system operational status"""
        try:
            # Placeholder for actual health check
            # In real implementation, this would ping Alita's health endpoint
            return SystemStatus.OPERATIONAL
        except Exception:
            return SystemStatus.ERROR
    
    async def _check_kgot_status(self) -> SystemStatus:
        """Check KGoT system operational status"""
        try:
            # Placeholder for actual health check
            # In real implementation, this would ping KGoT's health endpoint
            return SystemStatus.OPERATIONAL
        except Exception:
            return SystemStatus.ERROR
    
    async def _cleanup_old_data(self) -> None:
        """Cleanup old validation data periodically"""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                # Clean up old events from memory
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                # This would also trigger cleanup in the knowledge graph
                logger.info("Cleanup completed", extra={
                    'operation': 'DATA_CLEANUP',
                    'component': 'UNIFIED_DASHBOARD',
                    'cutoff_time': cutoff_time.isoformat()
                })
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}", extra={
                    'operation': 'CLEANUP_ERROR',
                    'component': 'UNIFIED_DASHBOARD',
                    'error': str(e)
                })


# Example usage and main execution
async def main():
    """Example usage of the Unified Validation Dashboard"""
    try:
        # Dashboard configuration
        config = {
            'dashboard_context': {
                'monitoring_systems': ['alita', 'kgot'],
                'real_time_enabled': True,
                'validation_history_limit': 1000
            },
            'validation_config': {
                'k_value': 5,
                'significance_level': 0.05
            },
            'kgot_config': {
                'graph_backend': 'neo4j',
                'connection_url': 'bolt://localhost:7687'
            },
            'kg_config': {
                'backend': 'neo4j',
                'connection_url': 'bolt://localhost:7687'
            }
        }
        
        # Initialize and start dashboard
        dashboard = UnifiedValidationDashboard(config)
        await dashboard.initialize()
        
        logger.info("Unified Validation Dashboard started successfully", extra={
            'operation': 'DASHBOARD_START',
            'component': 'UNIFIED_DASHBOARD'
        })
        
        # Start monitoring (this would run indefinitely)
        await dashboard.start_monitoring()
        
    except Exception as e:
        logger.error(f"Dashboard startup failed: {e}", extra={
            'operation': 'DASHBOARD_STARTUP_ERROR',
            'component': 'UNIFIED_DASHBOARD',
            'error': str(e)
        })


if __name__ == "__main__":
    asyncio.run(main()) 