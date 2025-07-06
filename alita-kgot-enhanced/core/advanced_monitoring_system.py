"""
Advanced Monitoring and Alerting System for Unified System Controller

This module provides comprehensive monitoring, alerting, and analytics capabilities
for the unified system controller, including:

- Real-time performance metrics collection
- Intelligent alerting with configurable thresholds  
- Performance trend analysis and predictions
- System health scoring and recommendations
- Integration with shared state management and analytics
- Cost and resource usage monitoring
- SLA compliance tracking

@module AdvancedMonitoringSystem
@author AI Assistant
@date 2025-01-22
"""

import asyncio
import json
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
logger = get_logger('advanced_monitoring_system')


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"  
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics tracked by the monitoring system"""
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    COST = "cost"
    AVAILABILITY = "availability"
    LATENCY = "latency"


@dataclass
class AlertRule:
    """Configuration for alert rules"""
    rule_id: str
    name: str
    description: str
    metric_type: MetricType
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==
    severity: AlertSeverity
    system_filter: Optional[str] = None  # Filter to specific systems
    time_window_minutes: int = 5
    min_occurrences: int = 1
    enabled: bool = True
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ["log"]


@dataclass  
class Alert:
    """Active alert instance"""
    alert_id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    system_name: str
    metric_value: float
    threshold_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    ack_by: Optional[str] = None
    ack_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    @property
    def is_active(self) -> bool:
        return self.resolved_at is None
    
    @property
    def duration_minutes(self) -> float:
        end_time = self.resolved_at or datetime.now()
        return (end_time - self.triggered_at).total_seconds() / 60


@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    metric_type: MetricType
    system_name: str
    trend_direction: str  # "improving", "degrading", "stable"
    change_percentage: float
    confidence_score: float
    prediction_next_hour: float
    recommendation: str
    analysis_period_hours: int = 24


@dataclass
class SystemHealthScore:
    """Overall system health assessment"""
    system_name: str
    overall_score: float  # 0-100
    availability_score: float
    performance_score: float
    reliability_score: float
    cost_efficiency_score: float
    last_calculated: datetime
    health_status: str  # "excellent", "good", "fair", "poor", "critical"
    
    def __post_init__(self):
        # Determine health status based on overall score
        if self.overall_score >= 90:
            self.health_status = "excellent"
        elif self.overall_score >= 80:
            self.health_status = "good"
        elif self.overall_score >= 70:
            self.health_status = "fair"
        elif self.overall_score >= 50:
            self.health_status = "poor"
        else:
            self.health_status = "critical"


class AlertManager:
    """
    Manages alerting rules, active alerts, and notifications
    """
    
    def __init__(self, shared_state: EnhancedSharedStateManager):
        self.shared_state = shared_state
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_handlers: Dict[str, Callable] = {}
        
        # Default alert rules
        self._initialize_default_rules()
        
        logger.info("AlertManager initialized", extra={
            'operation': 'ALERT_MANAGER_INIT',
            'default_rules_count': len(self.alert_rules)
        })
    
    def _initialize_default_rules(self) -> None:
        """Initialize default alerting rules"""
        default_rules = [
            AlertRule(
                rule_id="high_response_time",
                name="High Response Time",
                description="System response time exceeds threshold",
                metric_type=MetricType.RESPONSE_TIME,
                threshold_value=5000.0,  # 5 seconds
                comparison_operator=">",
                severity=AlertSeverity.WARNING,
                time_window_minutes=3
            ),
            AlertRule(
                rule_id="low_success_rate",
                name="Low Success Rate",
                description="System success rate below threshold",
                metric_type=MetricType.SUCCESS_RATE,
                threshold_value=0.95,  # 95%
                comparison_operator="<",
                severity=AlertSeverity.ERROR,
                min_occurrences=2
            ),
            AlertRule(
                rule_id="system_unavailable",
                name="System Unavailable",
                description="System availability below critical threshold",
                metric_type=MetricType.AVAILABILITY,
                threshold_value=0.90,  # 90%
                comparison_operator="<",
                severity=AlertSeverity.CRITICAL,
                min_occurrences=1
            ),
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="Error rate exceeds acceptable threshold",
                metric_type=MetricType.ERROR_RATE,
                threshold_value=0.05,  # 5%
                comparison_operator=">",
                severity=AlertSeverity.WARNING,
                time_window_minutes=5,
                min_occurrences=3
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    async def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule"""
        self.alert_rules[rule.rule_id] = rule
        
        # Store in shared state for persistence
        await self.shared_state.set_state_with_versioning(
            scope=StateScope.CONFIG,
            key=f"alert_rule_{rule.rule_id}",
            value=asdict(rule),
            source_system="monitoring_system"
        )
        
        logger.info(f"Added alert rule: {rule.name}", extra={
            'operation': 'ALERT_RULE_ADD',
            'rule_id': rule.rule_id,
            'severity': rule.severity.value
        })
    
    async def check_metric_against_rules(self, 
                                       system_name: str,
                                       metric_type: MetricType,
                                       metric_value: float) -> List[Alert]:
        """
        Check a metric value against all applicable alert rules
        
        Args:
            system_name: Name of the system
            metric_type: Type of metric
            metric_value: Current metric value
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            if rule.metric_type != metric_type:
                continue
            
            if rule.system_filter and rule.system_filter != system_name:
                continue
            
            # Check if threshold is violated
            threshold_violated = self._evaluate_threshold(
                metric_value, rule.threshold_value, rule.comparison_operator
            )
            
            if threshold_violated:
                # Check if alert already exists for this rule and system
                existing_alert_key = f"{rule.rule_id}_{system_name}"
                
                if existing_alert_key not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        alert_id=str(uuid4()),
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"{rule.description}: {metric_value:.2f} {rule.comparison_operator} {rule.threshold_value}",
                        system_name=system_name,
                        metric_value=metric_value,
                        threshold_value=rule.threshold_value,
                        triggered_at=datetime.now()
                    )
                    
                    self.active_alerts[existing_alert_key] = alert
                    self.alert_history.append(alert)
                    triggered_alerts.append(alert)
                    
                    # Send notifications
                    await self._send_alert_notifications(alert)
                    
                    logger.warning(f"Alert triggered: {alert.message}", extra={
                        'operation': 'ALERT_TRIGGERED',
                        'alert_id': alert.alert_id,
                        'rule_id': rule.rule_id,
                        'system_name': system_name,
                        'severity': rule.severity.value,
                        'metric_value': metric_value
                    })
        
        return triggered_alerts
    
    def _evaluate_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate if a value violates a threshold"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.001  # Float comparison
        else:
            logger.error(f"Unknown comparison operator: {operator}")
            return False
    
    async def _send_alert_notifications(self, alert: Alert) -> None:
        """Send alert notifications through configured channels"""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            return
        
        for channel in rule.notification_channels:
            handler = self.notification_handlers.get(channel)
            if handler:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel}: {str(e)}")
    
    async def resolve_alert(self, alert_id: str, resolved_by: str, notes: Optional[str] = None) -> bool:
        """Resolve an active alert"""
        for key, alert in self.active_alerts.items():
            if alert.alert_id == alert_id:
                alert.resolved_at = datetime.now()
                alert.resolution_notes = notes
                
                # Remove from active alerts
                del self.active_alerts[key]
                
                logger.info(f"Alert resolved: {alert.rule_name}", extra={
                    'operation': 'ALERT_RESOLVED',
                    'alert_id': alert_id,
                    'resolved_by': resolved_by,
                    'duration_minutes': alert.duration_minutes
                })
                
                return True
        
        return False
    
    async def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get list of active alerts with optional severity filter"""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)


class PerformanceAnalyzer:
    """
    Analyzes performance trends and provides predictions and recommendations
    """
    
    def __init__(self, shared_state: EnhancedSharedStateManager):
        self.shared_state = shared_state
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        logger.info("PerformanceAnalyzer initialized", extra={
            'operation': 'PERFORMANCE_ANALYZER_INIT'
        })
    
    async def record_metric(self, 
                          system_name: str,
                          metric_type: MetricType,
                          value: float,
                          timestamp: Optional[datetime] = None) -> None:
        """Record a performance metric"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_key = f"{system_name}_{metric_type.value}"
        metric_point = {
            'timestamp': timestamp,
            'value': value,
            'system': system_name,
            'metric_type': metric_type.value
        }
        
        self.metric_history[metric_key].append(metric_point)
        
        # Store in shared state for persistence
        await self.shared_state.set_state_with_versioning(
            scope=StateScope.METRICS,
            key=f"metric_point_{metric_key}_{int(timestamp.timestamp())}",
            value=metric_point,
            source_system="monitoring_system",
            expire_seconds=86400  # 24 hours
        )
    
    async def analyze_trend(self, 
                          system_name: str,
                          metric_type: MetricType,
                          hours: int = 24) -> Optional[PerformanceTrend]:
        """
        Analyze performance trend for a specific metric
        
        Args:
            system_name: Name of the system
            metric_type: Type of metric to analyze
            hours: Number of hours to analyze
            
        Returns:
            Performance trend analysis or None if insufficient data
        """
        metric_key = f"{system_name}_{metric_type.value}"
        history = self.metric_history.get(metric_key, deque())
        
        if len(history) < 10:  # Need minimum data points
            return None
        
        # Filter to specified time window
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [point for point in history if point['timestamp'] > cutoff_time]
        
        if len(recent_data) < 5:
            return None
        
        # Calculate trend
        values = [point['value'] for point in recent_data]
        timestamps = [point['timestamp'] for point in recent_data]
        
        # Simple linear trend calculation
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Threshold for "stable"
            trend_direction = "stable"
            change_percentage = 0.0
        elif slope > 0:
            trend_direction = "improving" if metric_type in [MetricType.SUCCESS_RATE, MetricType.AVAILABILITY] else "degrading"
            change_percentage = (slope * 100) / statistics.mean(values)
        else:
            trend_direction = "degrading" if metric_type in [MetricType.SUCCESS_RATE, MetricType.AVAILABILITY] else "improving"
            change_percentage = (slope * 100) / statistics.mean(values)
        
        # Calculate confidence score based on data consistency
        variance = statistics.variance(values) if len(values) > 1 else 0
        mean_value = statistics.mean(values)
        coefficient_of_variation = (variance ** 0.5) / mean_value if mean_value > 0 else 1
        confidence_score = max(0, min(100, 100 - (coefficient_of_variation * 100)))
        
        # Simple prediction for next hour (linear extrapolation)
        prediction_next_hour = values[-1] + slope * (len(values) / hours)  # Normalize to hourly rate
        
        # Generate recommendation
        recommendation = self._generate_recommendation(metric_type, trend_direction, change_percentage, values[-1])
        
        return PerformanceTrend(
            metric_type=metric_type,
            system_name=system_name,
            trend_direction=trend_direction,
            change_percentage=abs(change_percentage),
            confidence_score=confidence_score,
            prediction_next_hour=prediction_next_hour,
            recommendation=recommendation,
            analysis_period_hours=hours
        )
    
    def _generate_recommendation(self, 
                               metric_type: MetricType,
                               trend_direction: str,
                               change_percentage: float,
                               current_value: float) -> str:
        """Generate actionable recommendations based on trend analysis"""
        
        if trend_direction == "stable":
            return f"System {metric_type.value} is stable. Continue monitoring."
        
        if metric_type == MetricType.RESPONSE_TIME:
            if trend_direction == "degrading":
                if change_percentage > 20:
                    return "URGENT: Response time degrading rapidly. Check system load and consider scaling."
                else:
                    return "Response time trending higher. Monitor for potential bottlenecks."
            else:
                return "Response time improving. Current optimizations are effective."
        
        elif metric_type == MetricType.SUCCESS_RATE:
            if trend_direction == "degrading":
                if current_value < 0.95:
                    return "CRITICAL: Success rate below SLA. Immediate investigation required."
                else:
                    return "Success rate declining. Review recent changes and error patterns."
            else:
                return "Success rate improving. System reliability is enhancing."
        
        elif metric_type == MetricType.ERROR_RATE:
            if trend_direction == "degrading":  # Increasing error rate
                return "Error rate increasing. Review logs and investigate root causes."
            else:
                return "Error rate decreasing. Error handling improvements are working."
        
        elif metric_type == MetricType.COST:
            if trend_direction == "degrading":  # Increasing cost
                if change_percentage > 15:
                    return "ALERT: Costs rising significantly. Review resource usage and optimization opportunities."
                else:
                    return "Costs trending higher. Monitor usage patterns and consider optimization."
            else:
                return "Costs decreasing. Cost optimization measures are effective."
        
        return f"Monitor {metric_type.value} trend: {trend_direction} by {change_percentage:.1f}%"


class SystemHealthCalculator:
    """
    Calculates comprehensive system health scores and provides recommendations
    """
    
    def __init__(self, shared_state: EnhancedSharedStateManager):
        self.shared_state = shared_state
        
        # Health score weights (should sum to 1.0)
        self.weights = {
            'availability': 0.35,
            'performance': 0.25,
            'reliability': 0.25,
            'cost_efficiency': 0.15
        }
        
        logger.info("SystemHealthCalculator initialized", extra={
            'operation': 'HEALTH_CALCULATOR_INIT',
            'weights': self.weights
        })
    
    async def calculate_system_health(self, system_name: str) -> SystemHealthScore:
        """
        Calculate comprehensive health score for a system
        
        Args:
            system_name: Name of the system to evaluate
            
        Returns:
            Comprehensive system health score
        """
        # Get recent metrics for the system
        availability_score = await self._calculate_availability_score(system_name)
        performance_score = await self._calculate_performance_score(system_name)
        reliability_score = await self._calculate_reliability_score(system_name)
        cost_efficiency_score = await self._calculate_cost_efficiency_score(system_name)
        
        # Calculate weighted overall score
        overall_score = (
            availability_score * self.weights['availability'] +
            performance_score * self.weights['performance'] +
            reliability_score * self.weights['reliability'] +
            cost_efficiency_score * self.weights['cost_efficiency']
        )
        
        health_score = SystemHealthScore(
            system_name=system_name,
            overall_score=overall_score,
            availability_score=availability_score,
            performance_score=performance_score,
            reliability_score=reliability_score,
            cost_efficiency_score=cost_efficiency_score,
            last_calculated=datetime.now()
        )
        
        # Store in shared state
        await self.shared_state.set_state_with_versioning(
            scope=StateScope.METRICS,
            key=f"system_health_{system_name}",
            value=asdict(health_score),
            source_system="monitoring_system",
            expire_seconds=3600  # 1 hour
        )
        
        logger.info(f"System health calculated: {system_name} = {overall_score:.1f}", extra={
            'operation': 'HEALTH_SCORE_CALCULATED',
            'system_name': system_name,
            'overall_score': overall_score,
            'health_status': health_score.health_status
        })
        
        return health_score
    
    async def _calculate_availability_score(self, system_name: str) -> float:
        """Calculate availability score (0-100) based on uptime metrics"""
        # This would integrate with actual uptime monitoring
        # For now, using placeholder logic
        return 95.0  # Placeholder
    
    async def _calculate_performance_score(self, system_name: str) -> float:
        """Calculate performance score (0-100) based on response time and throughput"""
        # This would analyze response time trends and throughput metrics
        # For now, using placeholder logic
        return 88.0  # Placeholder
    
    async def _calculate_reliability_score(self, system_name: str) -> float:
        """Calculate reliability score (0-100) based on error rates and success rates"""
        # This would analyze error patterns and success rate trends
        # For now, using placeholder logic
        return 92.0  # Placeholder
    
    async def _calculate_cost_efficiency_score(self, system_name: str) -> float:
        """Calculate cost efficiency score (0-100) based on resource utilization and costs"""
        # This would analyze cost per transaction, resource utilization, etc.
        # For now, using placeholder logic
        return 85.0  # Placeholder


class AdvancedMonitoringSystem:
    """
    Comprehensive monitoring system that orchestrates all monitoring components
    """
    
    def __init__(self, 
                 shared_state: EnhancedSharedStateManager,
                 monitoring_interval: int = 30):
        """
        Initialize the advanced monitoring system
        
        Args:
            shared_state: Enhanced shared state manager
            monitoring_interval: Monitoring interval in seconds
        """
        self.shared_state = shared_state
        self.monitoring_interval = monitoring_interval
        
        # Initialize components
        self.alert_manager = AlertManager(shared_state)
        self.performance_analyzer = PerformanceAnalyzer(shared_state)
        self.health_calculator = SystemHealthCalculator(shared_state)
        
        # Monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        logger.info("AdvancedMonitoringSystem initialized", extra={
            'operation': 'ADVANCED_MONITORING_INIT',
            'monitoring_interval': monitoring_interval
        })
    
    async def start_monitoring(self) -> None:
        """Start the monitoring system"""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Advanced monitoring started", extra={
            'operation': 'MONITORING_START'
        })
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        self._is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced monitoring stopped", extra={
            'operation': 'MONITORING_STOP'
        })
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._is_monitoring:
            try:
                await self._collect_and_analyze_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}", extra={
                    'operation': 'MONITORING_LOOP_ERROR',
                    'error': str(e)
                })
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_and_analyze_metrics(self) -> None:
        """Collect metrics from all systems and perform analysis"""
        systems = ["alita", "kgot", "unified_controller"]
        
        for system in systems:
            try:
                # Collect various metrics (placeholder implementation)
                metrics = await self._collect_system_metrics(system)
                
                # Record metrics
                for metric_type, value in metrics.items():
                    await self.performance_analyzer.record_metric(system, metric_type, value)
                    
                    # Check against alert rules
                    await self.alert_manager.check_metric_against_rules(system, metric_type, value)
                
                # Calculate health score periodically (every 5 cycles)
                if hasattr(self, '_cycle_count'):
                    self._cycle_count += 1
                else:
                    self._cycle_count = 1
                
                if self._cycle_count % 5 == 0:
                    await self.health_calculator.calculate_system_health(system)
                
            except Exception as e:
                logger.error(f"Error collecting metrics for {system}: {str(e)}", extra={
                    'operation': 'METRICS_COLLECTION_ERROR',
                    'system': system,
                    'error': str(e)
                })
    
    async def _collect_system_metrics(self, system_name: str) -> Dict[MetricType, float]:
        """
        Collect current metrics for a specific system
        
        Args:
            system_name: Name of the system
            
        Returns:
            Dictionary of metric types and their current values
        """
        # Placeholder implementation - would integrate with actual system monitoring
        # This would make HTTP calls to system health endpoints, query databases, etc.
        
        return {
            MetricType.RESPONSE_TIME: 150.0,  # milliseconds
            MetricType.SUCCESS_RATE: 0.98,   # 98%
            MetricType.ERROR_RATE: 0.02,     # 2%
            MetricType.THROUGHPUT: 450.0,    # requests per minute
            MetricType.AVAILABILITY: 0.99,   # 99%
        }
    
    async def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring data for dashboard display
        
        Returns:
            Dashboard data including alerts, trends, and health scores
        """
        dashboard_data = {
            'active_alerts': [],
            'system_health': {},
            'performance_trends': {},
            'summary_statistics': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Get active alerts
        active_alerts = await self.alert_manager.get_active_alerts()
        dashboard_data['active_alerts'] = [asdict(alert) for alert in active_alerts]
        
        # Get system health scores
        systems = ["alita", "kgot", "unified_controller"]
        for system in systems:
            try:
                health_score = await self.health_calculator.calculate_system_health(system)
                dashboard_data['system_health'][system] = asdict(health_score)
            except Exception as e:
                logger.error(f"Error getting health score for {system}: {str(e)}")
        
        # Get performance trends
        for system in systems:
            dashboard_data['performance_trends'][system] = {}
            for metric_type in [MetricType.RESPONSE_TIME, MetricType.SUCCESS_RATE]:
                try:
                    trend = await self.performance_analyzer.analyze_trend(system, metric_type)
                    if trend:
                        dashboard_data['performance_trends'][system][metric_type.value] = asdict(trend)
                except Exception as e:
                    logger.error(f"Error getting trend for {system}.{metric_type.value}: {str(e)}")
        
        # Calculate summary statistics
        dashboard_data['summary_statistics'] = {
            'total_active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'systems_monitored': len(systems),
            'monitoring_uptime_hours': 24,  # Placeholder
        }
        
        return dashboard_data 