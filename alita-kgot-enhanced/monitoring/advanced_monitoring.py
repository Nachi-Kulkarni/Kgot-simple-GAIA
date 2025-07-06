#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 43: Advanced Monitoring and Alerting System

This module implements a comprehensive, AI-powered monitoring and alerting system
for the Alita-KGoT project. It aggregates logs, monitors MCP health, predicts
failures, and provides context-aware alerts.
"""

import asyncio
import json
import logging
import os
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from prometheus_client import Gauge, Histogram, Counter, start_http_server
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Data Structures ---

@dataclass
class LogEntry:
    """Represents a single parsed log entry."""
    timestamp: datetime
    level: str
    message: str
    service: str
    operation: Optional[str] = None
    mcp_name: Optional[str] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    http_status: Optional[int] = None
    error_message: Optional[str] = None
    trace_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthMetrics:
    """Stores health metrics for a single MCP or system component."""
    total_executions: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 1.0
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    error_codes: Dict[int, int] = field(default_factory=dict)
    last_seen: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    last_error: Optional[str] = None

@dataclass
class TrendAnalysis:
    """Stores trend analysis results."""
    latency_trend: float = 0.0  # Slope of latency regression
    error_rate_trend: float = 0.0  # Slope of error rate regression
    is_latency_increasing: bool = False
    is_error_rate_increasing: bool = False

@dataclass
class AlertRule:
    """Defines a rule for triggering an alert."""
    name: str
    mcp_pattern: str  # Regex to match MCP names
    metric: str  # e.g., 'failure_rate', 'latency_p95'
    threshold: float
    window_seconds: int
    min_failures: int = 1
    description: str = ""

@dataclass
class ActiveAlert:
    """Represents an active alert."""
    rule_name: str
    mcp_name: str
    triggered_at: datetime
    last_value: float
    message: str
    kgot_snapshot: Optional[Dict] = None # Placeholder for KGoT snapshot

# --- Components ---

class LogAggregator(FileSystemEventHandler):
    """
    Parses and aggregates logs from the filesystem in real-time.
    """
    def __init__(self, log_directory: str, callback):
        self.log_directory = log_directory
        self.callback = callback
        self.file_positions = {}
        logger.info(f"Initialized LogAggregator for directory: {self.log_directory}")

    def process_existing_logs(self):
        """Processes all logs in the directory that may have been missed."""
        logger.info("Starting processing of existing log files.")
        for root, _, files in os.walk(self.log_directory):
            for filename in files:
                if filename.endswith('.log'):
                    file_path = os.path.join(root, filename)
                    self._process_file(file_path)
        logger.info("Finished processing existing log files.")

    def on_modified(self, event):
        """Called when a file is modified."""
        if not event.is_directory and event.src_path.endswith('.log'):
            self._process_file(event.src_path)
    
    def _process_file(self, file_path: str):
        """Reads new lines from a log file since the last read."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                last_position = self.file_positions.get(file_path, 0)
                f.seek(last_position)
                for line in f:
                    if line.strip():
                        log_entry = self._parse_log_line(line)
                        if log_entry:
                            asyncio.run_coroutine_threadsafe(self.callback(log_entry), asyncio.get_event_loop())
                self.file_positions[file_path] = f.tell()
        except FileNotFoundError:
            logger.warning(f"Log file not found: {file_path}, it may have been rotated.")
            self.file_positions.pop(file_path, None)
        except Exception as e:
            logger.error(f"Error processing log file {file_path}: {e}", exc_info=True)

    def _parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parses a JSON log line into a LogEntry object."""
        try:
            data = json.loads(line)
            
            # Extract service from extra data, fallback to a default
            service = data.get('service', 'unknown_service')
            mcp_name = data.get('mcp_name') or data.get('operation')
            
            return LogEntry(
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                level=data.get('level', 'info'),
                message=data.get('message', ''),
                service=service,
                operation=data.get('operation'),
                mcp_name=mcp_name,
                duration_ms=data.get('duration_ms'),
                success=data.get('success'),
                http_status=data.get('http_status'),
                error_message=data.get('error', {}).get('message') if isinstance(data.get('error'), dict) else data.get('error'),
                trace_id=data.get('trace_id'),
                extra_data={k: v for k, v in data.items() if k not in ['timestamp', 'level', 'message', 'service', 'operation', 'mcp_name', 'duration_ms', 'success', 'http_status', 'error', 'trace_id']}
            )
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            # Handle non-JSON lines or lines with unexpected structure gracefully
            logger.debug(f"Could not parse log line: '{line.strip()}'. Error: {e}")
            return None

class HealthMonitor:
    """
    Tracks and calculates health metrics for all MCPs and components.
    """
    def __init__(self):
        self.metrics: Dict[str, HealthMetrics] = {}
        logger.info("HealthMonitor initialized.")

    async def update_metrics(self, log: LogEntry):
        """Updates health metrics based on a new log entry."""
        if not log.mcp_name:
            return

        mcp_name = log.mcp_name
        if mcp_name not in self.metrics:
            self.metrics[mcp_name] = HealthMetrics()
        
        stats = self.metrics[mcp_name]
        stats.total_executions += 1
        stats.last_seen = log.timestamp

        if log.success is not None:
            if log.success:
                stats.success_count += 1
            else:
                stats.failure_count += 1
                stats.last_error_time = log.timestamp
                stats.last_error = log.error_message or log.message

        if stats.total_executions > 0:
            stats.success_rate = stats.success_count / stats.total_executions
        
        if log.duration_ms is not None:
            stats.latencies.append(log.duration_ms)
            latencies_arr = np.array(stats.latencies)
            stats.avg_latency_ms = np.mean(latencies_arr)
            stats.p95_latency_ms = np.percentile(latencies_arr, 95)

        if log.http_status and 400 <= log.http_status < 600:
            stats.error_codes[log.http_status] = stats.error_codes.get(log.http_status, 0) + 1
        
        logger.debug(f"Updated metrics for {mcp_name}: Success Rate={stats.success_rate:.2f}, Avg Latency={stats.avg_latency_ms:.2f}ms")

    def get_metrics(self, mcp_name: str) -> Optional[HealthMetrics]:
        return self.metrics.get(mcp_name)

class PredictiveEngine:
    """
    Analyzes historical data to identify trends and predict failures.
    """
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.history: Dict[str, pd.DataFrame] = {} # Stores historical data for analysis
        self.analysis_results: Dict[str, TrendAnalysis] = {}
        self.history_window_size = 100 # Number of data points to keep for trend analysis
        logger.info("PredictiveEngine initialized.")

    async def analyze_trends(self):
        """Periodically analyzes trends for all monitored MCPs."""
        logger.info("Running trend analysis cycle...")
        for mcp_name, metrics in self.health_monitor.metrics.items():
            self._update_history(mcp_name, metrics)
            self._calculate_trends_for_mcp(mcp_name)

    def _update_history(self, mcp_name: str, metrics: HealthMetrics):
        """Updates the historical data for a given MCP."""
        if mcp_name not in self.history:
            self.history[mcp_name] = pd.DataFrame(columns=['timestamp', 'avg_latency', 'error_rate'])

        new_record = {
            'timestamp': datetime.now(),
            'avg_latency': metrics.avg_latency_ms,
            'error_rate': 1.0 - metrics.success_rate
        }
        
        # Using pandas.concat instead of append
        self.history[mcp_name] = pd.concat([self.history[mcp_name], pd.DataFrame([new_record])], ignore_index=True)

        # Trim history to window size
        if len(self.history[mcp_name]) > self.history_window_size:
            self.history[mcp_name] = self.history[mcp_name].tail(self.history_window_size)

    def _calculate_trends_for_mcp(self, mcp_name: str):
        """Calculates latency and error rate trends for a single MCP."""
        mcp_history = self.history.get(mcp_name)
        if mcp_history is None or len(mcp_history) < 10: # Need enough data to find a trend
            return

        analysis = TrendAnalysis()
        
        # Convert timestamp to seconds from start for regression
        X = (mcp_history['timestamp'] - mcp_history['timestamp'].min()).dt.total_seconds().values.reshape(-1, 1)

        # Latency trend
        y_latency = mcp_history['avg_latency'].values
        model_latency = LinearRegression()
        model_latency.fit(X, y_latency)
        analysis.latency_trend = model_latency.coef_[0]
        analysis.is_latency_increasing = analysis.latency_trend > 0.1 # Threshold for significant increase

        # Error rate trend
        y_error = mcp_history['error_rate'].values
        model_error = LinearRegression()
        model_error.fit(X, y_error)
        analysis.error_rate_trend = model_error.coef_[0]
        analysis.is_error_rate_increasing = analysis.error_rate_trend > 0.01 # Threshold for significant increase
        
        self.analysis_results[mcp_name] = analysis
        logger.debug(f"Trend analysis for {mcp_name}: Latency Trend={analysis.latency_trend:.4f}, Error Trend={analysis.error_rate_trend:.4f}")

    def get_analysis(self, mcp_name: str) -> Optional[TrendAnalysis]:
        return self.analysis_results.get(mcp_name)

class AlertingEngine:
    """
    A context-aware alerting system with a configurable rules engine.
    """
    def __init__(self, health_monitor: HealthMonitor, predictive_engine: PredictiveEngine):
        self.health_monitor = health_monitor
        self.predictive_engine = predictive_engine
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, ActiveAlert] = {}
        logger.info("AlertingEngine initialized.")

    def load_rules_from_config(self, config_path: str):
        """Loads alert rules from a YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                rules_data = yaml.safe_load(f)
                self.rules = [AlertRule(**rule) for rule in rules_data]
                logger.info(f"Successfully loaded {len(self.rules)} alert rules from {config_path}")
        except FileNotFoundError:
            logger.error(f"Alert rules file not found at {config_path}. No rules loaded.")
        except Exception as e:
            logger.error(f"Error loading alert rules from {config_path}: {e}", exc_info=True)

    async def check_alerts(self):
        """Periodically checks all rules against current metrics."""
        logger.debug("Running alert checks...")
        for mcp_name, metrics in self.health_monitor.metrics.items():
            for rule in self.rules:
                if re.match(rule.mcp_pattern, mcp_name):
                    self._evaluate_rule(rule, mcp_name, metrics)

    def _evaluate_rule(self, rule: AlertRule, mcp_name: str, metrics: HealthMetrics):
        """Evaluates a single rule for a given MCP."""
        value_to_check = None
        if rule.metric == "failure_rate":
            if metrics.total_executions > 0 and metrics.failure_count >= rule.min_failures:
                value_to_check = metrics.failure_count / metrics.total_executions
        elif rule.metric == "latency_p95":
            value_to_check = metrics.p95_latency_ms
        elif rule.metric.startswith("http_"):
             try:
                status_code = int(rule.metric.split('_')[1])
                value_to_check = metrics.error_codes.get(status_code, 0)
             except (ValueError, IndexError):
                 logger.warning(f"Could not parse status code from rule metric: {rule.metric}")
        elif rule.metric == "latency_trend":
            analysis = self.predictive_engine.get_analysis(mcp_name)
            if analysis:
                value_to_check = analysis.latency_trend
        
        if value_to_check is not None and value_to_check > rule.threshold:
            alert_key = f"{rule.name}-{mcp_name}"
            if alert_key not in self.active_alerts:
                 self._trigger_alert(rule, mcp_name, value_to_check)
                 self.active_alerts[alert_key] = ActiveAlert(
                     rule_name=rule.name,
                     mcp_name=mcp_name,
                     triggered_at=datetime.now(),
                     last_value=value_to_check,
                     message=f"Alert '{rule.name}' for {mcp_name}."
                 )
        else:
            # Simple resolution: clear alert if condition is no longer met
            alert_key = f"{rule.name}-{mcp_name}"
            if alert_key in self.active_alerts:
                logger.info(f"Alert '{rule.name}' for {mcp_name} has been resolved.")
                del self.active_alerts[alert_key]


    def _trigger_alert(self, rule: AlertRule, mcp_name: str, current_value: float):
        """Triggers and logs an alert."""
        message = (
            f"ALERT TRIGGERED: {rule.name} for MCP '{mcp_name}'.\n"
            f"  Description: {rule.description}\n"
            f"  Threshold: > {rule.threshold}\n"
            f"  Current Value: {current_value:.4f}\n"
            f"  MCP: {mcp_name}\n"
            f"  Timestamp: {datetime.now().isoformat()}\n"
            f"  KGoT Snapshot: (Placeholder for KGoT snapshot data)"
        )
        logger.warning(message)
        # In a real system, this would integrate with Slack, PagerDuty, etc.
        # e.g., self.slack_client.post_message(channel="#oncall-team", text=message)

class MetricsExporter:
    """
    Exposes metrics to Prometheus.
    """
    def __init__(self, health_monitor: HealthMonitor, predictive_engine: PredictiveEngine):
        self.health_monitor = health_monitor
        self.predictive_engine = predictive_engine
        # Define Prometheus metrics
        self.mcp_executions = Counter('mcp_executions_total', 'Total MCP executions', ['mcp_name'])
        self.mcp_failures = Counter('mcp_failures_total', 'Total MCP failures', ['mcp_name', 'error_code'])
        self.mcp_latency = Histogram('mcp_latency_seconds', 'MCP execution latency', ['mcp_name'])
        self.mcp_success_rate = Gauge('mcp_success_rate', 'Success rate of an MCP', ['mcp_name'])
        self.mcp_latency_avg = Gauge('mcp_latency_avg_seconds', 'Average latency of an MCP', ['mcp_name'])
        self.mcp_latency_p95 = Gauge('mcp_latency_p95_seconds', '95th percentile latency of an MCP', ['mcp_name'])
        self.mcp_latency_trend = Gauge('mcp_latency_trend_slope', 'Trend of MCP latency', ['mcp_name'])
        self.mcp_error_rate_trend = Gauge('mcp_error_rate_trend_slope', 'Trend of MCP error rate', ['mcp_name'])
        logger.info("MetricsExporter initialized.")

    def export_metrics(self):
        """Exports all current health and trend metrics to Prometheus."""
        logger.debug("Exporting metrics to Prometheus.")
        for mcp_name, metrics in self.health_monitor.metrics.items():
            labels = {'mcp_name': mcp_name}
            
            # Note: Prometheus counters should only be incremented, not set.
            # This should be handled where the event occurs.
            # For this example, we will assume the HealthMonitor can provide deltas,
            # but a better implementation would increment counters directly.
            # Here, we'll just set Gauges and observe Histograms.
            
            self.mcp_success_rate.labels(**labels).set(metrics.success_rate)
            self.mcp_latency_avg.labels(**labels).set(metrics.avg_latency_ms / 1000.0) # Convert to seconds
            self.mcp_latency_p95.labels(**labels).set(metrics.p95_latency_ms / 1000.0)

            # A more direct way to handle latency would be to .observe() it when it occurs.
            # Since we have a deque, we can't re-observe. This is a limitation of post-processing.
            # We will use the average as an example here.
            # A proper implementation would call self.mcp_latency.labels(...).observe(...) in HealthMonitor
            
            trend_analysis = self.predictive_engine.get_analysis(mcp_name)
            if trend_analysis:
                self.mcp_latency_trend.labels(**labels).set(trend_analysis.latency_trend)
                self.mcp_error_rate_trend.labels(**labels).set(trend_analysis.error_rate_trend)

    def observe_execution(self, log: LogEntry):
        """Observes a single execution to update counters and histograms."""
        if not log.mcp_name:
            return
        
        labels = {'mcp_name': log.mcp_name}
        self.mcp_executions.labels(**labels).inc()
        
        if log.duration_ms is not None:
            self.mcp_latency.labels(**labels).observe(log.duration_ms / 1000.0) # Observe in seconds
            
        if log.success == False:
            error_code = str(log.http_status or 'unknown')
            self.mcp_failures.labels(mcp_name=log.mcp_name, error_code=error_code).inc()

class AdvancedMonitoringService:
    """
    Main service to orchestrate the monitoring and alerting system.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.log_directory = config.get('log_directory', './logs')
        
        self.health_monitor = HealthMonitor()
        self.predictive_engine = PredictiveEngine(self.health_monitor)
        self.alerting_engine = AlertingEngine(self.health_monitor, self.predictive_engine)
        self.metrics_exporter = MetricsExporter(self.health_monitor, self.predictive_engine)
        
        # Chain the callbacks
        async def processing_callback(log_entry):
            await self.health_monitor.update_metrics(log_entry)
            self.metrics_exporter.observe_execution(log_entry)
        
        self.log_aggregator = LogAggregator(self.log_directory, processing_callback)
        
        self.observer = Observer()
        logger.info("AdvancedMonitoringService initialized.")

    async def start(self):
        """Starts the monitoring service and all its components."""
        logger.info("Starting Advanced Monitoring Service...")
        
        # Load alerting rules from config
        alert_rules_path = self.config.get('alert_rules_path', './config/monitoring/alert_rules.yml')
        self.alerting_engine.load_rules_from_config(alert_rules_path)

        # Start Prometheus metrics server
        prometheus_port = self.config.get('prometheus_port', 8008)
        start_http_server(prometheus_port)
        logger.info(f"Prometheus metrics server started on port {prometheus_port}")

        # Process logs that already exist
        self.log_aggregator.process_existing_logs()

        # Start watching for new log entries
        self.observer.schedule(self.log_aggregator, self.log_directory, recursive=True)
        self.observer.start()
        logger.info(f"Started monitoring log directory: {self.log_directory}")

        # Start periodic tasks for trend analysis and alerting
        asyncio.create_task(self._run_periodic_tasks())
        
        logger.info("Advanced Monitoring Service started successfully.")

    async def stop(self):
        """Stops the monitoring service."""
        logger.info("Stopping Advanced Monitoring Service...")
        self.observer.stop()
        self.observer.join()
        logger.info("Advanced Monitoring Service stopped.")

    async def _run_periodic_tasks(self):
        """Runs periodic analysis and alerting tasks."""
        while True:
            await asyncio.gather(
                self.predictive_engine.analyze_trends(),
                self.alerting_engine.check_alerts()
            )
            self.metrics_exporter.export_metrics()
            await asyncio.sleep(self.config.get('analysis_interval_seconds', 60))

# --- Main Execution ---

async def main():
    """Main function to run the monitoring service."""
    # Load configuration
    config = {
        'log_directory': os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs')),
        'prometheus_port': 8008,
        'analysis_interval_seconds': 60,
        'alert_rules_path': os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/monitoring/alert_rules.yml'))
    }
    
    service = AdvancedMonitoringService(config)
    
    # Create a dedicated thread for the watchdog observer
    # because the watchdog observer is blocking
    loop = asyncio.get_event_loop()
    
    try:
        await service.start()
        # Keep the service running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        await service.stop()
        logger.info("Service has been shut down.")

if __name__ == "__main__":
    # To run this:
    # 1. Make sure you have dependencies installed:
    #    pip install pandas numpy scikit-learn pyyaml watchdog prometheus-client
    # 2. Ensure the log directory structure exists.
    # 3. Run `python alita-kgot-enhanced/monitoring/advanced_monitoring.py`
    asyncio.run(main()) 