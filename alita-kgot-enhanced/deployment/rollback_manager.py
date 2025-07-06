#!/usr/bin/env python3
"""
Automated Rollback Manager
==========================

This module implements automated rollback mechanisms for the KGoT-Alita
production deployment pipeline with advanced monitoring integration and
intelligent decision making.

Features:
- Real-time monitoring integration
- SLA violation detection
- Automated rollback triggers
- Health-based decision making
- Rollback strategy optimization
- Comprehensive logging and alerting

Author: Alita KGoT Enhanced Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import aiohttp
import prometheus_client
from prometheus_client.parser import text_string_to_metric_families

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.logging.winston_config import setup_winston_logging
from monitoring.advanced_monitoring import AdvancedMonitoring
from deployment.blue_green_manager import BlueGreenDeploymentManager

@dataclass
class RollbackTrigger:
    """Configuration for rollback trigger conditions"""
    name: str
    metric: str
    threshold: float
    window_seconds: int
    evaluation_frequency: int = 30  # seconds
    consecutive_violations: int = 1
    severity: str = "critical"  # low, medium, high, critical
    
@dataclass
class RollbackEvent:
    """Information about a rollback event"""
    timestamp: datetime
    environment: str
    trigger: str
    trigger_value: float
    threshold: float
    previous_version: str
    rolled_back_version: str
    duration_seconds: float
    success: bool
    reason: str

@dataclass
class MonitoringMetrics:
    """Current monitoring metrics snapshot"""
    timestamp: datetime
    error_rate: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    request_rate: float
    service_health: Dict[str, bool]
    custom_metrics: Dict[str, float]

class AutomatedRollbackManager:
    """
    Manages automated rollback decisions based on real-time monitoring
    data and configurable SLA thresholds.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the automated rollback manager
        
        Args:
            config_path: Path to rollback configuration file
        """
        self.logger = setup_winston_logging(
            'automated_rollback',
            log_file='logs/deployment/rollback.log'
        )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.monitoring = AdvancedMonitoring()
        self.blue_green_manager = BlueGreenDeploymentManager()
        
        # Rollback state
        self.active_monitors: Dict[str, bool] = {}
        self.rollback_history: List[RollbackEvent] = []
        self.baseline_metrics: Dict[str, MonitoringMetrics] = {}
        self.violation_counters: Dict[str, Dict[str, int]] = {}
        
        # Rollback triggers configuration
        self.triggers = self._initialize_triggers()
        
        # Prometheus metrics for rollback tracking
        self.rollback_counter = prometheus_client.Counter(
            'kgot_rollbacks_total',
            'Total number of rollbacks performed',
            ['environment', 'trigger', 'success']
        )
        
        self.rollback_duration = prometheus_client.Histogram(
            'kgot_rollback_duration_seconds',
            'Duration of rollback operations in seconds',
            ['environment']
        )
        
        self.sla_violations = prometheus_client.Counter(
            'kgot_sla_violations_total',
            'Total SLA violations detected',
            ['environment', 'metric', 'severity']
        )
        
        # Callbacks for custom triggers
        self.custom_triggers: Dict[str, Callable] = {}
        
        self.logger.info("Automated rollback manager initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load rollback configuration"""
        default_config = {
            'monitoring': {
                'check_interval': 30,  # seconds
                'baseline_window': 600,  # 10 minutes
                'cooldown_period': 300,  # 5 minutes between rollbacks
            },
            'triggers': {
                'error_rate': {
                    'threshold': 0.05,  # 5%
                    'window': 120,  # 2 minutes
                    'consecutive_violations': 2
                },
                'response_time': {
                    'threshold_multiplier': 2.0,  # 2x baseline
                    'window': 180,  # 3 minutes
                    'consecutive_violations': 3
                },
                'cpu_usage': {
                    'threshold': 0.85,  # 85%
                    'window': 300,  # 5 minutes
                    'consecutive_violations': 3
                },
                'memory_usage': {
                    'threshold': 0.90,  # 90%
                    'window': 180,  # 3 minutes
                    'consecutive_violations': 2
                },
                'service_health': {
                    'threshold': 0.8,  # 80% services healthy
                    'window': 60,  # 1 minute
                    'consecutive_violations': 2
                }
            },
            'rollback': {
                'enabled': True,
                'require_manual_approval': False,
                'max_rollbacks_per_hour': 3,
                'emergency_mode_threshold': 0.20,  # 20% error rate
            },
            'notifications': {
                'slack_webhook': None,
                'email_recipients': [],
                'pagerduty_key': None
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Deep merge configurations
            self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Deep merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _initialize_triggers(self) -> List[RollbackTrigger]:
        """Initialize rollback triggers from configuration"""
        triggers = []
        
        for trigger_name, trigger_config in self.config.get('triggers', {}).items():
            trigger = RollbackTrigger(
                name=trigger_name,
                metric=trigger_name,
                threshold=trigger_config.get('threshold', 0.0),
                window_seconds=trigger_config.get('window', 60),
                consecutive_violations=trigger_config.get('consecutive_violations', 1),
                severity=trigger_config.get('severity', 'critical')
            )
            triggers.append(trigger)
        
        return triggers
    
    async def start_monitoring(self, environment: str) -> None:
        """
        Start monitoring environment for rollback conditions
        
        Args:
            environment: Environment to monitor (staging/production)
        """
        if environment in self.active_monitors:
            self.logger.warning(f"Monitoring already active for {environment}")
            return
        
        self.active_monitors[environment] = True
        self.violation_counters[environment] = {}
        
        self.logger.info(f"Starting rollback monitoring for {environment}")
        
        # Establish baseline metrics
        await self._establish_baseline(environment)
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop(environment))
    
    async def stop_monitoring(self, environment: str) -> None:
        """
        Stop monitoring environment
        
        Args:
            environment: Environment to stop monitoring
        """
        if environment in self.active_monitors:
            self.active_monitors[environment] = False
            self.logger.info(f"Stopped rollback monitoring for {environment}")
    
    async def _monitoring_loop(self, environment: str) -> None:
        """
        Main monitoring loop for detecting rollback conditions
        
        Args:
            environment: Environment to monitor
        """
        check_interval = self.config.get('monitoring', {}).get('check_interval', 30)
        last_rollback_time = 0
        cooldown_period = self.config.get('monitoring', {}).get('cooldown_period', 300)
        
        while self.active_monitors.get(environment, False):
            try:
                # Get current metrics
                metrics = await self._collect_metrics(environment)
                
                # Check rollback triggers
                violations = await self._check_triggers(environment, metrics)
                
                # Determine if rollback is needed
                if violations and self._should_trigger_rollback(environment, violations):
                    current_time = time.time()
                    
                    # Check cooldown period
                    if current_time - last_rollback_time < cooldown_period:
                        self.logger.warning(
                            f"Rollback needed but in cooldown period for {environment}"
                        )
                    else:
                        # Execute rollback
                        rollback_success = await self._execute_rollback(
                            environment, violations
                        )
                        
                        if rollback_success:
                            last_rollback_time = current_time
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop for {environment}: {str(e)}")
                await asyncio.sleep(check_interval)
    
    async def _establish_baseline(self, environment: str) -> None:
        """
        Establish baseline metrics for the environment
        
        Args:
            environment: Environment to establish baseline for
        """
        self.logger.info(f"Establishing baseline metrics for {environment}")
        
        baseline_window = self.config.get('monitoring', {}).get('baseline_window', 600)
        samples = []
        
        # Collect samples over baseline window
        for _ in range(baseline_window // 30):  # Sample every 30 seconds
            try:
                metrics = await self._collect_metrics(environment)
                samples.append(metrics)
                await asyncio.sleep(30)
            except Exception as e:
                self.logger.warning(f"Failed to collect baseline sample: {str(e)}")
        
        if samples:
            # Calculate baseline from samples
            baseline = self._calculate_baseline(samples)
            self.baseline_metrics[environment] = baseline
            self.logger.info(f"Baseline established for {environment}")
        else:
            self.logger.error(f"Failed to establish baseline for {environment}")
    
    def _calculate_baseline(self, samples: List[MonitoringMetrics]) -> MonitoringMetrics:
        """Calculate baseline metrics from samples"""
        if not samples:
            return MonitoringMetrics(
                timestamp=datetime.now(),
                error_rate=0.0,
                avg_response_time=100.0,
                p95_response_time=200.0,
                p99_response_time=500.0,
                cpu_usage=0.1,
                memory_usage=0.2,
                disk_usage=0.1,
                active_connections=10,
                request_rate=1.0,
                service_health={},
                custom_metrics={}
            )
        
        # Calculate averages
        n = len(samples)
        return MonitoringMetrics(
            timestamp=datetime.now(),
            error_rate=sum(s.error_rate for s in samples) / n,
            avg_response_time=sum(s.avg_response_time for s in samples) / n,
            p95_response_time=sum(s.p95_response_time for s in samples) / n,
            p99_response_time=sum(s.p99_response_time for s in samples) / n,
            cpu_usage=sum(s.cpu_usage for s in samples) / n,
            memory_usage=sum(s.memory_usage for s in samples) / n,
            disk_usage=sum(s.disk_usage for s in samples) / n,
            active_connections=int(sum(s.active_connections for s in samples) / n),
            request_rate=sum(s.request_rate for s in samples) / n,
            service_health=samples[-1].service_health,  # Use latest
            custom_metrics=samples[-1].custom_metrics
        )
    
    async def _collect_metrics(self, environment: str) -> MonitoringMetrics:
        """
        Collect current metrics from monitoring system
        
        Args:
            environment: Environment to collect metrics for
            
        Returns:
            Current metrics snapshot
        """
        # Get metrics from advanced monitoring system
        monitoring_data = await self.monitoring.get_real_time_metrics(environment)
        
        return MonitoringMetrics(
            timestamp=datetime.now(),
            error_rate=monitoring_data.get('error_rate', 0.0),
            avg_response_time=monitoring_data.get('avg_response_time', 0.0),
            p95_response_time=monitoring_data.get('p95_response_time', 0.0),
            p99_response_time=monitoring_data.get('p99_response_time', 0.0),
            cpu_usage=monitoring_data.get('cpu_usage', 0.0),
            memory_usage=monitoring_data.get('memory_usage', 0.0),
            disk_usage=monitoring_data.get('disk_usage', 0.0),
            active_connections=monitoring_data.get('active_connections', 0),
            request_rate=monitoring_data.get('request_rate', 0.0),
            service_health=monitoring_data.get('service_health', {}),
            custom_metrics=monitoring_data.get('custom_metrics', {})
        )
    
    async def _check_triggers(
        self,
        environment: str,
        metrics: MonitoringMetrics
    ) -> List[Tuple[RollbackTrigger, float]]:
        """
        Check all triggers against current metrics
        
        Args:
            environment: Environment being monitored
            metrics: Current metrics
            
        Returns:
            List of violated triggers with their values
        """
        violations = []
        baseline = self.baseline_metrics.get(environment)
        
        for trigger in self.triggers:
            violation_value = await self._evaluate_trigger(
                trigger, metrics, baseline
            )
            
            if violation_value is not None:
                # Track consecutive violations
                counter_key = f"{trigger.name}"
                if counter_key not in self.violation_counters[environment]:
                    self.violation_counters[environment][counter_key] = 0
                
                self.violation_counters[environment][counter_key] += 1
                
                # Check if consecutive violation threshold is met
                if (self.violation_counters[environment][counter_key] >= 
                    trigger.consecutive_violations):
                    violations.append((trigger, violation_value))
                    
                    # Record SLA violation
                    self.sla_violations.labels(
                        environment=environment,
                        metric=trigger.metric,
                        severity=trigger.severity
                    ).inc()
            else:
                # Reset counter if no violation
                counter_key = f"{trigger.name}"
                if counter_key in self.violation_counters[environment]:
                    self.violation_counters[environment][counter_key] = 0
        
        return violations
    
    async def _evaluate_trigger(
        self,
        trigger: RollbackTrigger,
        metrics: MonitoringMetrics,
        baseline: Optional[MonitoringMetrics]
    ) -> Optional[float]:
        """
        Evaluate a single trigger against metrics
        
        Args:
            trigger: Trigger to evaluate
            metrics: Current metrics
            baseline: Baseline metrics for comparison
            
        Returns:
            Violation value if triggered, None otherwise
        """
        if trigger.name == 'error_rate':
            if metrics.error_rate > trigger.threshold:
                return metrics.error_rate
        
        elif trigger.name == 'response_time':
            if baseline:
                threshold_multiplier = self.config.get('triggers', {}).get(
                    'response_time', {}
                ).get('threshold_multiplier', 2.0)
                threshold = baseline.avg_response_time * threshold_multiplier
                if metrics.avg_response_time > threshold:
                    return metrics.avg_response_time
        
        elif trigger.name == 'cpu_usage':
            if metrics.cpu_usage > trigger.threshold:
                return metrics.cpu_usage
        
        elif trigger.name == 'memory_usage':
            if metrics.memory_usage > trigger.threshold:
                return metrics.memory_usage
        
        elif trigger.name == 'service_health':
            healthy_services = sum(1 for h in metrics.service_health.values() if h)
            total_services = len(metrics.service_health)
            if total_services > 0:
                health_ratio = healthy_services / total_services
                if health_ratio < trigger.threshold:
                    return health_ratio
        
        # Custom trigger evaluation
        elif trigger.name in self.custom_triggers:
            return await self.custom_triggers[trigger.name](trigger, metrics, baseline)
        
        return None
    
    def _should_trigger_rollback(
        self,
        environment: str,
        violations: List[Tuple[RollbackTrigger, float]]
    ) -> bool:
        """
        Determine if rollback should be triggered based on violations
        
        Args:
            environment: Environment being evaluated
            violations: List of trigger violations
            
        Returns:
            True if rollback should be triggered
        """
        if not violations:
            return False
        
        # Check if rollbacks are enabled
        if not self.config.get('rollback', {}).get('enabled', True):
            self.logger.info("Rollbacks are disabled in configuration")
            return False
        
        # Check emergency mode
        emergency_threshold = self.config.get('rollback', {}).get(
            'emergency_mode_threshold', 0.20
        )
        
        for trigger, value in violations:
            if trigger.name == 'error_rate' and value > emergency_threshold:
                self.logger.critical(
                    f"Emergency mode triggered! Error rate: {value:.3f}"
                )
                return True
        
        # Check if manual approval is required
        if self.config.get('rollback', {}).get('require_manual_approval', False):
            self.logger.warning(
                "Rollback conditions met but manual approval required"
            )
            return False
        
        # Check rollback rate limits
        max_rollbacks = self.config.get('rollback', {}).get('max_rollbacks_per_hour', 3)
        recent_rollbacks = [
            r for r in self.rollback_history 
            if (r.environment == environment and 
                r.timestamp > datetime.now() - timedelta(hours=1))
        ]
        
        if len(recent_rollbacks) >= max_rollbacks:
            self.logger.warning(
                f"Rollback rate limit exceeded for {environment}. "
                f"Recent rollbacks: {len(recent_rollbacks)}"
            )
            return False
        
        # Check severity levels
        critical_violations = [
            v for v in violations if v[0].severity == 'critical'
        ]
        
        if critical_violations:
            return True
        
        high_violations = [
            v for v in violations if v[0].severity == 'high'
        ]
        
        if len(high_violations) >= 2:  # Multiple high-severity violations
            return True
        
        return False
    
    async def _execute_rollback(
        self,
        environment: str,
        violations: List[Tuple[RollbackTrigger, float]]
    ) -> bool:
        """
        Execute automated rollback for environment
        
        Args:
            environment: Environment to rollback
            violations: Violations that triggered rollback
            
        Returns:
            True if rollback successful
        """
        rollback_start = time.time()
        
        # Get primary violation for logging
        primary_violation = violations[0] if violations else None
        trigger_name = primary_violation[0].name if primary_violation else "unknown"
        trigger_value = primary_violation[1] if primary_violation else 0.0
        
        self.logger.critical(
            f"Executing automated rollback for {environment}. "
            f"Trigger: {trigger_name} = {trigger_value}"
        )
        
        try:
            # Create deployment target
            from deployment.blue_green_manager import DeploymentTarget
            target = DeploymentTarget(
                environment=environment,
                namespace=f"kgot-{environment}",
                domain=f"{environment}.kgot.local",
                services=['kgot-controller', 'graph-store', 'manager-agent', 'web-agent'],
                resources={}
            )
            
            # Execute rollback using blue-green manager
            success = await self.blue_green_manager.rollback_deployment(
                target, f"automated_{trigger_name}"
            )
            
            rollback_duration = time.time() - rollback_start
            
            # Record rollback event
            rollback_event = RollbackEvent(
                timestamp=datetime.now(),
                environment=environment,
                trigger=trigger_name,
                trigger_value=trigger_value,
                threshold=primary_violation[0].threshold if primary_violation else 0.0,
                previous_version="unknown",  # Would need to track this
                rolled_back_version="previous",  # Would need to track this
                duration_seconds=rollback_duration,
                success=success,
                reason=f"Automated rollback due to {trigger_name} violation"
            )
            
            self.rollback_history.append(rollback_event)
            
            # Update metrics
            self.rollback_counter.labels(
                environment=environment,
                trigger=trigger_name,
                success=str(success).lower()
            ).inc()
            
            self.rollback_duration.labels(environment=environment).observe(rollback_duration)
            
            # Send notifications
            await self._send_rollback_notification(rollback_event)
            
            if success:
                self.logger.info(
                    f"Automated rollback completed successfully for {environment} "
                    f"in {rollback_duration:.2f} seconds"
                )
            else:
                self.logger.error(f"Automated rollback failed for {environment}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing rollback for {environment}: {str(e)}")
            return False
    
    async def _send_rollback_notification(self, event: RollbackEvent) -> None:
        """Send notification about rollback event"""
        try:
            # Slack notification
            slack_webhook = self.config.get('notifications', {}).get('slack_webhook')
            if slack_webhook:
                message = {
                    "text": f"🚨 Automated Rollback Executed",
                    "attachments": [
                        {
                            "color": "danger" if not event.success else "warning",
                            "fields": [
                                {"title": "Environment", "value": event.environment, "short": True},
                                {"title": "Trigger", "value": event.trigger, "short": True},
                                {"title": "Value", "value": f"{event.trigger_value:.3f}", "short": True},
                                {"title": "Threshold", "value": f"{event.threshold:.3f}", "short": True},
                                {"title": "Duration", "value": f"{event.duration_seconds:.1f}s", "short": True},
                                {"title": "Success", "value": str(event.success), "short": True},
                                {"title": "Reason", "value": event.reason, "short": False}
                            ]
                        }
                    ]
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(slack_webhook, json=message) as response:
                        if response.status == 200:
                            self.logger.info("Rollback notification sent to Slack")
                        else:
                            self.logger.warning(f"Failed to send Slack notification: {response.status}")
            
        except Exception as e:
            self.logger.error(f"Failed to send rollback notification: {str(e)}")
    
    def add_custom_trigger(
        self,
        name: str,
        evaluator: Callable[[RollbackTrigger, MonitoringMetrics, Optional[MonitoringMetrics]], float]
    ) -> None:
        """
        Add custom trigger evaluator
        
        Args:
            name: Trigger name
            evaluator: Function to evaluate trigger condition
        """
        self.custom_triggers[name] = evaluator
        self.logger.info(f"Added custom trigger: {name}")
    
    def get_rollback_history(self, environment: str = None) -> List[RollbackEvent]:
        """
        Get rollback history
        
        Args:
            environment: Optional environment filter
            
        Returns:
            List of rollback events
        """
        if environment:
            return [r for r in self.rollback_history if r.environment == environment]
        return self.rollback_history.copy()
    
    def get_violation_status(self, environment: str) -> Dict[str, int]:
        """
        Get current violation counters for environment
        
        Args:
            environment: Environment to check
            
        Returns:
            Current violation counters
        """
        return self.violation_counters.get(environment, {}).copy()

def main():
    """Main entry point for automated rollback manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Rollback Manager')
    parser.add_argument('--environment', required=True, choices=['staging', 'production'],
                       help='Environment to monitor')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = AutomatedRollbackManager(args.config)
    
    async def run_manager():
        try:
            # Start monitoring
            await manager.start_monitoring(args.environment)
            
            if args.test_mode:
                # Run for 5 minutes in test mode
                await asyncio.sleep(300)
            else:
                # Run indefinitely
                while True:
                    await asyncio.sleep(60)
                    
        except KeyboardInterrupt:
            print("\nShutting down rollback manager...")
        finally:
            await manager.stop_monitoring(args.environment)
    
    asyncio.run(run_manager())

if __name__ == "__main__":
    main() 