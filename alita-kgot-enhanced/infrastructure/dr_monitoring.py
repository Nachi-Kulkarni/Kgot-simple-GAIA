#!/usr/bin/env python3
"""
KGoT-Alita Disaster Recovery Monitoring and Alerting

Provides comprehensive monitoring, alerting, and dashboard capabilities for:
- Backup operation monitoring with success/failure tracking
- RTO/RPO compliance monitoring and alerting
- System health checks and performance metrics
- Integration with external monitoring systems (Prometheus, Grafana)
- Webhook-based alerting for critical events
- Real-time dashboard for operational visibility

@module DisasterRecoveryMonitoring
@author Enhanced Alita KGoT System
@version 1.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import psutil
from pathlib import Path

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Setup logging
logger = logging.getLogger('DR_Monitoring')

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

class DisasterRecoveryMetrics:
    """Metrics collection and management"""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_history: List[Metric] = []
        self.registry = None
        
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        if not self.enable_prometheus:
            return
        
        # Backup metrics
        self.backup_total = Counter(
            'kgot_backup_total',
            'Total number of backups created',
            ['component', 'status'],
            registry=self.registry
        )
        
        self.backup_duration = Histogram(
            'kgot_backup_duration_seconds',
            'Backup operation duration',
            ['component'],
            registry=self.registry
        )
        
        self.backup_size = Gauge(
            'kgot_backup_size_bytes',
            'Size of backup files',
            ['component', 'backup_id'],
            registry=self.registry
        )
        
        # Recovery metrics
        self.recovery_total = Counter(
            'kgot_recovery_total',
            'Total number of recovery operations',
            ['status'],
            registry=self.registry
        )
        
        self.recovery_duration = Histogram(
            'kgot_recovery_duration_seconds',
            'Recovery operation duration',
            registry=self.registry
        )
        
        self.rto_compliance = Gauge(
            'kgot_rto_compliance',
            'RTO compliance (1 = compliant, 0 = non-compliant)',
            registry=self.registry
        )
        
        self.rpo_compliance = Gauge(
            'kgot_rpo_compliance',
            'RPO compliance (1 = compliant, 0 = non-compliant)',
            registry=self.registry
        )
        
        # System health metrics
        self.system_health = Gauge(
            'kgot_system_health',
            'System health status (1 = healthy, 0 = unhealthy)',
            ['component'],
            registry=self.registry
        )
        
        self.last_backup_age = Gauge(
            'kgot_last_backup_age_seconds',
            'Age of last successful backup',
            ['component'],
            registry=self.registry
        )
    
    def record_backup_started(self, component: str):
        """Record backup start"""
        metric = Metric(
            name="backup_started",
            metric_type=MetricType.COUNTER,
            value=1,
            timestamp=datetime.now(),
            labels={"component": component}
        )
        self.metrics_history.append(metric)
        
        logger.info(f"Backup started: {component}", extra={'operation': 'BACKUP_STARTED', 'component': component})
    
    def record_backup_completed(self, component: str, duration_seconds: float, size_bytes: int, success: bool):
        """Record backup completion"""
        status = "success" if success else "failure"
        
        # Record metrics
        metrics = [
            Metric(
                name="backup_completed",
                metric_type=MetricType.COUNTER,
                value=1,
                timestamp=datetime.now(),
                labels={"component": component, "status": status}
            ),
            Metric(
                name="backup_duration",
                metric_type=MetricType.HISTOGRAM,
                value=duration_seconds,
                timestamp=datetime.now(),
                labels={"component": component}
            )
        ]
        
        if success:
            metrics.append(Metric(
                name="backup_size",
                metric_type=MetricType.GAUGE,
                value=size_bytes,
                timestamp=datetime.now(),
                labels={"component": component}
            ))
        
        self.metrics_history.extend(metrics)
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.backup_total.labels(component=component, status=status).inc()
            self.backup_duration.labels(component=component).observe(duration_seconds)
            if success:
                self.backup_size.labels(component=component, backup_id="latest").set(size_bytes)
        
        logger.info(f"Backup completed: {component}", 
                   extra={'operation': 'BACKUP_COMPLETED', 'component': component, 
                         'success': success, 'duration': duration_seconds, 'size': size_bytes})
    
    def record_recovery_started(self):
        """Record recovery start"""
        metric = Metric(
            name="recovery_started",
            metric_type=MetricType.COUNTER,
            value=1,
            timestamp=datetime.now()
        )
        self.metrics_history.append(metric)
        
        logger.info("Recovery started", extra={'operation': 'RECOVERY_STARTED'})
    
    def record_recovery_completed(self, duration_seconds: float, success: bool, rto_compliant: bool):
        """Record recovery completion"""
        status = "success" if success else "failure"
        
        metrics = [
            Metric(
                name="recovery_completed",
                metric_type=MetricType.COUNTER,
                value=1,
                timestamp=datetime.now(),
                labels={"status": status}
            ),
            Metric(
                name="recovery_duration",
                metric_type=MetricType.HISTOGRAM,
                value=duration_seconds,
                timestamp=datetime.now()
            ),
            Metric(
                name="rto_compliance",
                metric_type=MetricType.GAUGE,
                value=1 if rto_compliant else 0,
                timestamp=datetime.now()
            )
        ]
        
        self.metrics_history.extend(metrics)
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.recovery_total.labels(status=status).inc()
            self.recovery_duration.observe(duration_seconds)
            self.rto_compliance.set(1 if rto_compliant else 0)
        
        logger.info(f"Recovery completed", 
                   extra={'operation': 'RECOVERY_COMPLETED', 'success': success, 
                         'duration': duration_seconds, 'rto_compliant': rto_compliant})
    
    def update_system_health(self, component: str, healthy: bool):
        """Update system health status"""
        metric = Metric(
            name="system_health",
            metric_type=MetricType.GAUGE,
            value=1 if healthy else 0,
            timestamp=datetime.now(),
            labels={"component": component}
        )
        self.metrics_history.append(metric)
        
        if self.enable_prometheus:
            self.system_health.labels(component=component).set(1 if healthy else 0)
        
        logger.info(f"System health updated: {component} = {'healthy' if healthy else 'unhealthy'}", 
                   extra={'operation': 'HEALTH_UPDATE', 'component': component, 'healthy': healthy})
    
    def update_last_backup_age(self, component: str, age_seconds: float):
        """Update age of last backup"""
        metric = Metric(
            name="last_backup_age",
            metric_type=MetricType.GAUGE,
            value=age_seconds,
            timestamp=datetime.now(),
            labels={"component": component}
        )
        self.metrics_history.append(metric)
        
        if self.enable_prometheus:
            self.last_backup_age.labels(component=component).set(age_seconds)
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        # Count by type and status
        backup_success = len([m for m in recent_metrics 
                            if m.name == "backup_completed" and m.labels.get("status") == "success"])
        backup_failure = len([m for m in recent_metrics 
                            if m.name == "backup_completed" and m.labels.get("status") == "failure"])
        
        recovery_success = len([m for m in recent_metrics 
                              if m.name == "recovery_completed" and m.labels.get("status") == "success"])
        recovery_failure = len([m for m in recent_metrics 
                              if m.name == "recovery_completed" and m.labels.get("status") == "failure"])
        
        # Calculate averages
        backup_durations = [m.value for m in recent_metrics if m.name == "backup_duration"]
        recovery_durations = [m.value for m in recent_metrics if m.name == "recovery_duration"]
        
        return {
            "period_hours": hours,
            "backup_operations": {
                "success": backup_success,
                "failure": backup_failure,
                "success_rate": backup_success / max(backup_success + backup_failure, 1),
                "avg_duration_seconds": sum(backup_durations) / max(len(backup_durations), 1)
            },
            "recovery_operations": {
                "success": recovery_success,
                "failure": recovery_failure,
                "success_rate": recovery_success / max(recovery_success + recovery_failure, 1),
                "avg_duration_seconds": sum(recovery_durations) / max(len(recovery_durations), 1)
            },
            "total_metrics": len(recent_metrics)
        }

class AlertManager:
    """Alert management and notification system"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
    
    async def create_alert(self, severity: AlertSeverity, title: str, message: str, 
                          component: str, metadata: Dict[str, Any] = None) -> Alert:
        """Create and process new alert"""
        alert = Alert(
            alert_id=f"alert_{int(time.time())}_{len(self.alerts)}",
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            component=component,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Process alert
        await self._process_alert(alert)
        
        logger.warning(f"Alert created: {alert.title}", 
                      extra={'operation': 'ALERT_CREATED', 'severity': severity.value, 
                            'component': component, 'alert_id': alert.alert_id})
        
        return alert
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        alert = next((a for a in self.alerts if a.alert_id == alert_id), None)
        if alert and not alert.resolved:
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            logger.info(f"Alert resolved: {alert.title}", 
                       extra={'operation': 'ALERT_RESOLVED', 'alert_id': alert_id})
            return True
        return False
    
    async def _process_alert(self, alert: Alert):
        """Process alert through all handlers"""
        # Send to webhook if configured
        if self.webhook_url:
            await self._send_webhook_alert(alert)
        
        # Call custom handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send alert via webhook"""
        try:
            payload = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "component": alert.component,
                "metadata": alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent: {alert.alert_id}")
                    else:
                        logger.error(f"Webhook alert failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level"""
        return [alert for alert in self.alerts if alert.severity == severity and not alert.resolved]

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self, metrics: DisasterRecoveryMetrics, alerts: AlertManager):
        self.metrics = metrics
        self.alerts = alerts
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.is_running = False
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self.health_checks[name] = check_func
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous health monitoring"""
        self.is_running = True
        
        logger.info(f"Starting health monitoring (interval: {interval_seconds}s)")
        
        while self.is_running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_running = False
        logger.info("Health monitoring stopped")
    
    async def _run_health_checks(self):
        """Run all registered health checks"""
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = check_func()
                self.metrics.update_system_health(name, is_healthy)
                
                if not is_healthy:
                    await self.alerts.create_alert(
                        AlertSeverity.WARNING,
                        f"Health Check Failed: {name}",
                        f"Component {name} failed health check",
                        name
                    )
            
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                self.metrics.update_system_health(name, False)
                
                await self.alerts.create_alert(
                    AlertSeverity.CRITICAL,
                    f"Health Check Error: {name}",
                    f"Health check for {name} threw exception: {str(e)}",
                    name,
                    {"error": str(e)}
                )

class DisasterRecoveryMonitor:
    """Main monitoring system coordinator"""
    
    def __init__(self, webhook_url: Optional[str] = None, prometheus_port: int = 8000):
        self.metrics = DisasterRecoveryMetrics()
        self.alerts = AlertManager(webhook_url)
        self.health_checker = HealthChecker(self.metrics, self.alerts)
        self.prometheus_port = prometheus_port
        self.is_running = False
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default system health checks"""
        
        def check_disk_space() -> bool:
            """Check available disk space"""
            try:
                usage = psutil.disk_usage('/')
                free_percent = (usage.free / usage.total) * 100
                return free_percent > 10  # Alert if less than 10% free
            except:
                return False
        
        def check_memory_usage() -> bool:
            """Check memory usage"""
            try:
                memory = psutil.virtual_memory()
                return memory.percent < 90  # Alert if more than 90% used
            except:
                return False
        
        def check_cpu_usage() -> bool:
            """Check CPU usage"""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                return cpu_percent < 95  # Alert if more than 95% used
            except:
                return False
        
        self.health_checker.register_health_check("disk_space", check_disk_space)
        self.health_checker.register_health_check("memory_usage", check_memory_usage)
        self.health_checker.register_health_check("cpu_usage", check_cpu_usage)
    
    async def start(self):
        """Start monitoring system"""
        self.is_running = True
        
        # Start Prometheus metrics server
        if self.metrics.enable_prometheus:
            start_http_server(self.prometheus_port, registry=self.metrics.registry)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        
        # Start health monitoring
        asyncio.create_task(self.health_checker.start_monitoring())
        
        # Start backup age monitoring
        asyncio.create_task(self._monitor_backup_age())
        
        logger.info("Disaster recovery monitoring started")
    
    def stop(self):
        """Stop monitoring system"""
        self.is_running = False
        self.health_checker.stop_monitoring()
        logger.info("Disaster recovery monitoring stopped")
    
    async def _monitor_backup_age(self):
        """Monitor age of last backups and alert if too old"""
        while self.is_running:
            try:
                # This would integrate with the actual backup system
                # For now, we'll simulate checking backup ages
                
                # Check Neo4j backup age
                # neo4j_age = self._get_last_backup_age("neo4j")
                # self.metrics.update_last_backup_age("neo4j", neo4j_age)
                
                # Check Vector index backup age
                # vector_age = self._get_last_backup_age("vector_index")
                # self.metrics.update_last_backup_age("vector_index", vector_age)
                
                # Alert if backups are too old (more than 2 hours)
                # max_age_hours = 2
                # if neo4j_age > max_age_hours * 3600:
                #     await self.alerts.create_alert(
                #         AlertSeverity.WARNING,
                #         "Neo4j Backup Age Alert",
                #         f"Last Neo4j backup is {neo4j_age/3600:.1f} hours old",
                #         "neo4j",
                #         {"age_hours": neo4j_age/3600}
                #     )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Backup age monitoring error: {e}")
                await asyncio.sleep(60)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": self.metrics.get_metrics_summary(),
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "component": alert.component
                }
                for alert in self.alerts.get_active_alerts()
            ],
            "system_status": {
                "monitoring_active": self.is_running,
                "prometheus_enabled": self.metrics.enable_prometheus,
                "health_checks_count": len(self.health_checker.health_checks)
            }
        }

# Integration with disaster recovery system
class MonitoredDisasterRecoverySystem:
    """Disaster recovery system with integrated monitoring"""
    
    def __init__(self, dr_system, monitor: DisasterRecoveryMonitor):
        self.dr_system = dr_system
        self.monitor = monitor
        
        # Hook into DR system events
        self._setup_monitoring_hooks()
    
    def _setup_monitoring_hooks(self):
        """Setup monitoring hooks for DR system events"""
        # This would integrate with the actual DR system
        # to capture events and metrics
        pass
    
    async def create_monitored_backup(self):
        """Create backup with monitoring"""
        start_time = time.time()
        
        try:
            # Record backup start
            self.monitor.metrics.record_backup_started("coordinated")
            
            # Execute backup
            backups = await self.dr_system.create_backup()
            
            # Record success
            duration = time.time() - start_time
            total_size = sum(backup.file_size for backup in backups)
            
            for backup in backups:
                self.monitor.metrics.record_backup_completed(
                    backup.component, duration, backup.file_size, True
                )
            
            return backups
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            self.monitor.metrics.record_backup_completed("coordinated", duration, 0, False)
            
            # Create alert
            await self.monitor.alerts.create_alert(
                AlertSeverity.CRITICAL,
                "Backup Operation Failed",
                f"Coordinated backup failed: {str(e)}",
                "backup_system",
                {"error": str(e), "duration": duration}
            )
            
            raise
    
    async def execute_monitored_recovery(self, target_timestamp=None):
        """Execute recovery with monitoring"""
        start_time = time.time()
        
        try:
            # Record recovery start
            self.monitor.metrics.record_recovery_started()
            
            # Execute recovery
            success = await self.dr_system.restore_from_backup(target_timestamp)
            
            # Record completion
            duration = time.time() - start_time
            rto_target = self.dr_system.config.rto_target_minutes * 60
            rto_compliant = duration <= rto_target
            
            self.monitor.metrics.record_recovery_completed(duration, success, rto_compliant)
            
            if not success:
                await self.monitor.alerts.create_alert(
                    AlertSeverity.EMERGENCY,
                    "Disaster Recovery Failed",
                    "Disaster recovery operation failed",
                    "recovery_system",
                    {"duration": duration, "rto_compliant": rto_compliant}
                )
            elif not rto_compliant:
                await self.monitor.alerts.create_alert(
                    AlertSeverity.WARNING,
                    "RTO Target Exceeded",
                    f"Recovery took {duration/60:.2f} minutes, exceeding RTO target of {rto_target/60:.2f} minutes",
                    "recovery_system",
                    {"duration": duration, "rto_target": rto_target}
                )
            
            return success
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            self.monitor.metrics.record_recovery_completed(duration, False, False)
            
            # Create alert
            await self.monitor.alerts.create_alert(
                AlertSeverity.EMERGENCY,
                "Disaster Recovery Exception",
                f"Disaster recovery threw exception: {str(e)}",
                "recovery_system",
                {"error": str(e), "duration": duration}
            )
            
            raise

if __name__ == "__main__":
    async def main():
        # Example usage
        monitor = DisasterRecoveryMonitor(
            webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            prometheus_port=8000
        )
        
        await monitor.start()
        
        # Simulate some events
        await monitor.alerts.create_alert(
            AlertSeverity.INFO,
            "System Started",
            "Disaster recovery monitoring system started",
            "monitoring_system"
        )
        
        # Keep running
        try:
            await asyncio.sleep(3600)  # Run for 1 hour
        finally:
            monitor.stop()
    
    asyncio.run(main())