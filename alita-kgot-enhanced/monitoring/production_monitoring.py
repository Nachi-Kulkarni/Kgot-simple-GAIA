#!/usr/bin/env python3
"""
Task 48: Production Monitoring and Analytics System

Comprehensive production monitoring system that implements:
- Centralized structured logging service for Alita, KGoT, and MCPs
- Real-time MCP performance analytics with RAG-MCP metrics
- Predictive analytics for system optimization
- Production dashboard integration with Grafana

Based on:
- KGoT: Layered error containment & management with knowledge graph snapshots
- RAG-MCP: Performance metrics (Accuracy, Avg Prompt Tokens, Avg Completion Tokens)
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import pickle
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

# ML imports for predictive analytics
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit-learn not available. Predictive analytics will be limited.")

# Time series database simulation (can be replaced with InfluxDB/Prometheus)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available. Using in-memory storage.")


class LogSeverity(Enum):
    """Log severity levels for production monitoring"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ServiceType(Enum):
    """Types of services being monitored"""
    ALITA = "alita"
    KGOT = "kgot"
    MCP = "mcp"
    UNIFIED_CONTROLLER = "unified_controller"
    RAG_ENGINE = "rag_engine"
    ORCHESTRATOR = "orchestrator"


class MetricType(Enum):
    """Types of metrics tracked in production"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    COST = "cost"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class KGoTSnapshot:
    """Knowledge graph snapshot for KGoT error management"""
    graph_id: str
    node_count: int
    edge_count: int
    active_thoughts: List[str]
    reasoning_path: List[str]
    confidence_scores: Dict[str, float]
    timestamp: datetime
    serialized_graph: Optional[str] = None  # JSON serialized graph structure


@dataclass
class RAGMCPMetrics:
    """RAG-MCP specific performance metrics"""
    mcp_id: str
    accuracy_percentage: float
    avg_prompt_tokens: int
    avg_completion_tokens: int
    latency_ms: float
    success_rate: float
    cost_per_request: float
    timestamp: datetime
    task_type: Optional[str] = None


@dataclass
class ProductionLogEntry:
    """Structured log entry for production monitoring"""
    service_name: str
    service_type: ServiceType
    timestamp: datetime
    task_id: str
    severity: LogSeverity
    message: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    kgot_snapshot: Optional[KGoTSnapshot] = None
    rag_mcp_metrics: Optional[RAGMCPMetrics] = None
    execution_context: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convert log entry to JSON string"""
        data = asdict(self)
        # Convert datetime objects to ISO format
        data['timestamp'] = self.timestamp.isoformat()
        if self.kgot_snapshot:
            data['kgot_snapshot']['timestamp'] = self.kgot_snapshot.timestamp.isoformat()
        if self.rag_mcp_metrics:
            data['rag_mcp_metrics']['timestamp'] = self.rag_mcp_metrics.timestamp.isoformat()
        return json.dumps(data, default=str)


class CentralizedLoggingService:
    """Centralized logging service that ingests structured logs from all components"""
    
    def __init__(self, 
                 log_directory: str = "logs/production",
                 max_log_size_mb: int = 100,
                 retention_days: int = 30,
                 enable_real_time_processing: bool = True):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.retention_days = retention_days
        self.enable_real_time_processing = enable_real_time_processing
        
        # Initialize log files for different services
        self.log_files = {
            ServiceType.ALITA: self.log_directory / "alita_production.jsonl",
            ServiceType.KGOT: self.log_directory / "kgot_production.jsonl",
            ServiceType.MCP: self.log_directory / "mcp_production.jsonl",
            ServiceType.UNIFIED_CONTROLLER: self.log_directory / "controller_production.jsonl",
            ServiceType.RAG_ENGINE: self.log_directory / "rag_production.jsonl",
            ServiceType.ORCHESTRATOR: self.log_directory / "orchestrator_production.jsonl"
        }
        
        # Real-time processing queue
        self.log_queue = asyncio.Queue(maxsize=10000)
        self.processing_task = None
        
        # Metrics for monitoring the logging system itself
        self.logging_metrics = {
            'logs_ingested': 0,
            'logs_processed': 0,
            'processing_errors': 0,
            'queue_size': 0
        }
        
        self.logger = logging.getLogger(f"{__name__}.CentralizedLoggingService")
        
    async def start(self):
        """Start the centralized logging service"""
        if self.enable_real_time_processing and not self.processing_task:
            self.processing_task = asyncio.create_task(self._process_logs())
            self.logger.info("Centralized logging service started")
    
    async def stop(self):
        """Stop the centralized logging service"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Centralized logging service stopped")
    
    async def ingest_log(self, log_entry: ProductionLogEntry):
        """Ingest a structured log entry"""
        try:
            if self.enable_real_time_processing:
                await self.log_queue.put(log_entry)
            else:
                await self._write_log_entry(log_entry)
            
            self.logging_metrics['logs_ingested'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to ingest log entry: {e}")
            self.logging_metrics['processing_errors'] += 1
    
    async def _process_logs(self):
        """Process logs from the queue in real-time"""
        while True:
            try:
                # Get log entry from queue with timeout
                log_entry = await asyncio.wait_for(self.log_queue.get(), timeout=1.0)
                
                # Write to appropriate log file
                await self._write_log_entry(log_entry)
                
                # Update metrics
                self.logging_metrics['logs_processed'] += 1
                self.logging_metrics['queue_size'] = self.log_queue.qsize()
                
                # Mark task as done
                self.log_queue.task_done()
                
            except asyncio.TimeoutError:
                # No logs to process, continue
                continue
            except Exception as e:
                self.logger.error(f"Error processing log: {e}")
                self.logging_metrics['processing_errors'] += 1
    
    async def _write_log_entry(self, log_entry: ProductionLogEntry):
        """Write log entry to appropriate file"""
        log_file = self.log_files[log_entry.service_type]
        
        # Check if log rotation is needed
        if log_file.exists() and log_file.stat().st_size > self.max_log_size_bytes:
            await self._rotate_log_file(log_file)
        
        # Write log entry as JSON line
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry.to_json() + '\n')
    
    async def _rotate_log_file(self, log_file: Path):
        """Rotate log file when it gets too large"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_file = log_file.with_suffix(f".{timestamp}.jsonl")
        log_file.rename(rotated_file)
        self.logger.info(f"Rotated log file: {log_file} -> {rotated_file}")
    
    def get_logging_metrics(self) -> Dict[str, Any]:
        """Get metrics about the logging system itself"""
        return self.logging_metrics.copy()


class RAGMCPPerformanceAnalytics:
    """Real-time MCP performance analytics with RAG-MCP metrics"""
    
    def __init__(self, 
                 time_window_minutes: int = 60,
                 enable_time_series_storage: bool = True):
        self.time_window = timedelta(minutes=time_window_minutes)
        self.enable_time_series_storage = enable_time_series_storage
        
        # In-memory storage for recent metrics
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.aggregated_metrics = defaultdict(dict)
        
        # Time series storage (Redis or in-memory)
        if REDIS_AVAILABLE and enable_time_series_storage:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.use_redis = True
        else:
            self.time_series_data = defaultdict(list)
            self.use_redis = False
        
        self.logger = logging.getLogger(f"{__name__}.RAGMCPPerformanceAnalytics")
    
    async def record_mcp_metrics(self, metrics: RAGMCPMetrics):
        """Record MCP performance metrics"""
        try:
            # Add to buffer
            self.metrics_buffer[metrics.mcp_id].append(metrics)
            
            # Store in time series database
            if self.use_redis:
                await self._store_metrics_redis(metrics)
            else:
                self._store_metrics_memory(metrics)
            
            # Update aggregated metrics
            await self._update_aggregated_metrics(metrics.mcp_id)
            
        except Exception as e:
            self.logger.error(f"Failed to record MCP metrics: {e}")
    
    async def _store_metrics_redis(self, metrics: RAGMCPMetrics):
        """Store metrics in Redis time series"""
        timestamp = int(metrics.timestamp.timestamp())
        
        # Store each metric type separately
        metric_keys = {
            f"mcp:{metrics.mcp_id}:accuracy": metrics.accuracy_percentage,
            f"mcp:{metrics.mcp_id}:prompt_tokens": metrics.avg_prompt_tokens,
            f"mcp:{metrics.mcp_id}:completion_tokens": metrics.avg_completion_tokens,
            f"mcp:{metrics.mcp_id}:latency": metrics.latency_ms,
            f"mcp:{metrics.mcp_id}:success_rate": metrics.success_rate,
            f"mcp:{metrics.mcp_id}:cost": metrics.cost_per_request
        }
        
        for key, value in metric_keys.items():
            self.redis_client.zadd(key, {timestamp: value})
            # Keep only last 24 hours of data
            cutoff = timestamp - (24 * 60 * 60)
            self.redis_client.zremrangebyscore(key, 0, cutoff)
    
    def _store_metrics_memory(self, metrics: RAGMCPMetrics):
        """Store metrics in memory"""
        self.time_series_data[metrics.mcp_id].append({
            'timestamp': metrics.timestamp,
            'accuracy': metrics.accuracy_percentage,
            'prompt_tokens': metrics.avg_prompt_tokens,
            'completion_tokens': metrics.avg_completion_tokens,
            'latency': metrics.latency_ms,
            'success_rate': metrics.success_rate,
            'cost': metrics.cost_per_request
        })
        
        # Keep only recent data
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.time_series_data[metrics.mcp_id] = [
            m for m in self.time_series_data[metrics.mcp_id] 
            if m['timestamp'] > cutoff_time
        ]
    
    async def _update_aggregated_metrics(self, mcp_id: str):
        """Update aggregated metrics for an MCP"""
        recent_metrics = list(self.metrics_buffer[mcp_id])
        if not recent_metrics:
            return
        
        # Filter to time window
        cutoff_time = datetime.now() - self.time_window
        recent_metrics = [m for m in recent_metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return
        
        # Calculate aggregated metrics
        self.aggregated_metrics[mcp_id] = {
            'avg_accuracy': np.mean([m.accuracy_percentage for m in recent_metrics]),
            'avg_prompt_tokens': np.mean([m.avg_prompt_tokens for m in recent_metrics]),
            'avg_completion_tokens': np.mean([m.avg_completion_tokens for m in recent_metrics]),
            'avg_latency': np.mean([m.latency_ms for m in recent_metrics]),
            'avg_success_rate': np.mean([m.success_rate for m in recent_metrics]),
            'avg_cost': np.mean([m.cost_per_request for m in recent_metrics]),
            'total_requests': len(recent_metrics),
            'last_updated': datetime.now()
        }
    
    def get_mcp_performance(self, mcp_id: str) -> Optional[Dict[str, Any]]:
        """Get current performance metrics for an MCP"""
        return self.aggregated_metrics.get(mcp_id)
    
    def get_all_mcp_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all MCPs"""
        return dict(self.aggregated_metrics)
    
    def get_top_performing_mcps(self, metric: str = 'avg_accuracy', limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing MCPs by specified metric"""
        mcps = []
        for mcp_id, metrics in self.aggregated_metrics.items():
            if metric in metrics:
                mcps.append({
                    'mcp_id': mcp_id,
                    'value': metrics[metric],
                    'metrics': metrics
                })
        
        return sorted(mcps, key=lambda x: x['value'], reverse=True)[:limit]


class PredictiveAnalyticsEngine:
    """Machine learning engine for predictive system optimization"""
    
    def __init__(self, 
                 model_storage_path: str = "models/production",
                 retrain_interval_hours: int = 24):
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        self.retrain_interval = timedelta(hours=retrain_interval_hours)
        
        # ML models
        self.models = {}
        self.scalers = {}
        self.last_training = {}
        
        # Prediction types
        self.prediction_types = [
            'cost_spike_prediction',
            'failure_prediction', 
            'performance_degradation',
            'resource_exhaustion'
        ]
        
        self.logger = logging.getLogger(f"{__name__}.PredictiveAnalyticsEngine")
        
        if not ML_AVAILABLE:
            self.logger.warning("ML libraries not available. Predictive analytics disabled.")
    
    async def train_models(self, training_data: Dict[str, pd.DataFrame]):
        """Train predictive models using historical data"""
        if not ML_AVAILABLE:
            self.logger.warning("Cannot train models: ML libraries not available")
            return
        
        for prediction_type in self.prediction_types:
            if prediction_type in training_data:
                await self._train_model(prediction_type, training_data[prediction_type])
    
    async def _train_model(self, prediction_type: str, data: pd.DataFrame):
        """Train a specific prediction model"""
        try:
            # Prepare features and target
            features, target = self._prepare_training_data(prediction_type, data)
            
            if len(features) < 10:  # Need minimum data for training
                self.logger.warning(f"Insufficient data for {prediction_type} model training")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if prediction_type == 'failure_prediction':
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X_train_scaled)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.logger.info(f"{prediction_type} model trained - MSE: {mse:.4f}, R2: {r2:.4f}")
            
            # Store model and scaler
            self.models[prediction_type] = model
            self.scalers[prediction_type] = scaler
            self.last_training[prediction_type] = datetime.now()
            
            # Save to disk
            await self._save_model(prediction_type, model, scaler)
            
        except Exception as e:
            self.logger.error(f"Failed to train {prediction_type} model: {e}")
    
    def _prepare_training_data(self, prediction_type: str, data: pd.DataFrame) -> tuple:
        """Prepare training data for specific prediction type"""
        # This is a simplified example - in practice, feature engineering would be more sophisticated
        
        if prediction_type == 'cost_spike_prediction':
            features = data[['avg_prompt_tokens', 'avg_completion_tokens', 'request_rate', 'hour_of_day']]
            target = data['cost_per_hour']
        
        elif prediction_type == 'failure_prediction':
            features = data[['error_rate', 'latency', 'cpu_usage', 'memory_usage']]
            target = data['failure_occurred']
        
        elif prediction_type == 'performance_degradation':
            features = data[['success_rate', 'latency', 'throughput', 'resource_usage']]
            target = data['performance_score']
        
        elif prediction_type == 'resource_exhaustion':
            features = data[['cpu_usage', 'memory_usage', 'disk_usage', 'request_rate']]
            target = data['resource_exhaustion_risk']
        
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        
        return features.fillna(0), target.fillna(0)
    
    async def predict(self, prediction_type: str, current_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make prediction using trained model"""
        if not ML_AVAILABLE or prediction_type not in self.models:
            return None
        
        try:
            model = self.models[prediction_type]
            scaler = self.scalers[prediction_type]
            
            # Prepare input data
            features = self._prepare_prediction_input(prediction_type, current_data)
            features_scaled = scaler.transform([features])
            
            # Make prediction
            if prediction_type == 'failure_prediction':
                prediction = model.decision_function(features_scaled)[0]
                confidence = abs(prediction)
                risk_level = 'high' if prediction < -0.5 else 'medium' if prediction < 0 else 'low'
            else:
                prediction = model.predict(features_scaled)[0]
                confidence = 0.8  # Simplified confidence score
                risk_level = self._categorize_risk(prediction_type, prediction)
            
            return {
                'prediction_type': prediction_type,
                'predicted_value': float(prediction),
                'confidence': float(confidence),
                'risk_level': risk_level,
                'timestamp': datetime.now(),
                'input_features': current_data
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {prediction_type}: {e}")
            return None
    
    def _prepare_prediction_input(self, prediction_type: str, data: Dict[str, Any]) -> List[float]:
        """Prepare input data for prediction"""
        if prediction_type == 'cost_spike_prediction':
            return [
                data.get('avg_prompt_tokens', 0),
                data.get('avg_completion_tokens', 0),
                data.get('request_rate', 0),
                datetime.now().hour
            ]
        elif prediction_type == 'failure_prediction':
            return [
                data.get('error_rate', 0),
                data.get('latency', 0),
                data.get('cpu_usage', 0),
                data.get('memory_usage', 0)
            ]
        elif prediction_type == 'performance_degradation':
            return [
                data.get('success_rate', 100),
                data.get('latency', 0),
                data.get('throughput', 0),
                data.get('resource_usage', 0)
            ]
        elif prediction_type == 'resource_exhaustion':
            return [
                data.get('cpu_usage', 0),
                data.get('memory_usage', 0),
                data.get('disk_usage', 0),
                data.get('request_rate', 0)
            ]
        else:
            return []
    
    def _categorize_risk(self, prediction_type: str, value: float) -> str:
        """Categorize prediction value into risk levels"""
        # Simplified risk categorization - would be more sophisticated in practice
        if prediction_type == 'cost_spike_prediction':
            return 'high' if value > 100 else 'medium' if value > 50 else 'low'
        elif prediction_type == 'performance_degradation':
            return 'high' if value < 0.7 else 'medium' if value < 0.9 else 'low'
        elif prediction_type == 'resource_exhaustion':
            return 'high' if value > 0.8 else 'medium' if value > 0.6 else 'low'
        else:
            return 'medium'
    
    async def _save_model(self, prediction_type: str, model, scaler):
        """Save trained model to disk"""
        model_file = self.model_storage_path / f"{prediction_type}_model.pkl"
        scaler_file = self.model_storage_path / f"{prediction_type}_scaler.pkl"
        
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
    
    async def load_models(self):
        """Load trained models from disk"""
        for prediction_type in self.prediction_types:
            model_file = self.model_storage_path / f"{prediction_type}_model.pkl"
            scaler_file = self.model_storage_path / f"{prediction_type}_scaler.pkl"
            
            if model_file.exists() and scaler_file.exists():
                try:
                    with open(model_file, 'rb') as f:
                        self.models[prediction_type] = pickle.load(f)
                    
                    with open(scaler_file, 'rb') as f:
                        self.scalers[prediction_type] = pickle.load(f)
                    
                    self.logger.info(f"Loaded {prediction_type} model")
                except Exception as e:
                    self.logger.error(f"Failed to load {prediction_type} model: {e}")


class ProductionMonitoringDashboard:
    """Dashboard integration for Grafana and other monitoring tools"""
    
    def __init__(self, 
                 analytics_engine: RAGMCPPerformanceAnalytics,
                 predictive_engine: PredictiveAnalyticsEngine,
                 export_port: int = 8090):
        self.analytics_engine = analytics_engine
        self.predictive_engine = predictive_engine
        self.export_port = export_port
        
        # Dashboard configuration
        self.dashboard_config = {
            'refresh_interval': 30,  # seconds
            'data_retention': 7,     # days
            'alert_thresholds': {
                'error_rate': 0.05,
                'latency': 1000,  # ms
                'cost_spike': 100   # percentage increase
            }
        }
        
        self.logger = logging.getLogger(f"{__name__}.ProductionMonitoringDashboard")
    
    async def start_metrics_server(self):
        """Start HTTP server for metrics export (Prometheus format)"""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_get('/metrics', self._export_prometheus_metrics)
        app.router.add_get('/health', self._health_check)
        app.router.add_get('/dashboard/data', self._export_dashboard_data)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, 'localhost', self.export_port)
        await site.start()
        
        self.logger.info(f"Metrics server started on port {self.export_port}")
    
    async def _export_prometheus_metrics(self, request):
        """Export metrics in Prometheus format"""
        from aiohttp import web
        
        metrics = []
        
        # Get all MCP performance metrics
        all_mcp_metrics = self.analytics_engine.get_all_mcp_performance()
        
        for mcp_id, mcp_metrics in all_mcp_metrics.items():
            metrics.extend([
                f'mcp_accuracy{{mcp_id="{mcp_id}"}} {mcp_metrics.get("avg_accuracy", 0)}',
                f'mcp_latency_ms{{mcp_id="{mcp_id}"}} {mcp_metrics.get("avg_latency", 0)}',
                f'mcp_prompt_tokens{{mcp_id="{mcp_id}"}} {mcp_metrics.get("avg_prompt_tokens", 0)}',
                f'mcp_completion_tokens{{mcp_id="{mcp_id}"}} {mcp_metrics.get("avg_completion_tokens", 0)}',
                f'mcp_success_rate{{mcp_id="{mcp_id}"}} {mcp_metrics.get("avg_success_rate", 0)}',
                f'mcp_cost{{mcp_id="{mcp_id}"}} {mcp_metrics.get("avg_cost", 0)}',
                f'mcp_requests_total{{mcp_id="{mcp_id}"}} {mcp_metrics.get("total_requests", 0)}'
            ])
        
        metrics_text = '\n'.join(metrics)
        return web.Response(text=metrics_text, content_type='text/plain')
    
    async def _health_check(self, request):
        """Health check endpoint"""
        from aiohttp import web
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'analytics_engine': 'healthy',
                'predictive_engine': 'healthy' if ML_AVAILABLE else 'limited',
                'time_series_storage': 'healthy' if REDIS_AVAILABLE else 'memory_only'
            }
        }
        
        return web.json_response(health_status)
    
    async def _export_dashboard_data(self, request):
        """Export data for custom dashboard"""
        from aiohttp import web
        
        # Get query parameters
        time_range = request.query.get('time_range', '1h')
        mcp_id = request.query.get('mcp_id')
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'time_range': time_range,
            'summary': await self._get_system_summary(),
            'mcp_performance': self.analytics_engine.get_all_mcp_performance(),
            'top_performers': self.analytics_engine.get_top_performing_mcps(),
            'predictions': await self._get_current_predictions()
        }
        
        if mcp_id:
            dashboard_data['mcp_details'] = self.analytics_engine.get_mcp_performance(mcp_id)
        
        return web.json_response(dashboard_data)
    
    async def _get_system_summary(self) -> Dict[str, Any]:
        """Get overall system summary"""
        all_metrics = self.analytics_engine.get_all_mcp_performance()
        
        if not all_metrics:
            return {'total_mcps': 0, 'avg_accuracy': 0, 'avg_latency': 0}
        
        total_mcps = len(all_metrics)
        avg_accuracy = np.mean([m.get('avg_accuracy', 0) for m in all_metrics.values()])
        avg_latency = np.mean([m.get('avg_latency', 0) for m in all_metrics.values()])
        total_requests = sum([m.get('total_requests', 0) for m in all_metrics.values()])
        
        return {
            'total_mcps': total_mcps,
            'avg_accuracy': float(avg_accuracy),
            'avg_latency': float(avg_latency),
            'total_requests': total_requests,
            'last_updated': datetime.now().isoformat()
        }
    
    async def _get_current_predictions(self) -> List[Dict[str, Any]]:
        """Get current predictions from predictive engine"""
        predictions = []
        
        # Get current system state for predictions
        current_state = await self._get_current_system_state()
        
        for prediction_type in self.predictive_engine.prediction_types:
            prediction = await self.predictive_engine.predict(prediction_type, current_state)
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    async def _get_current_system_state(self) -> Dict[str, Any]:
        """Get current system state for predictions"""
        all_metrics = self.analytics_engine.get_all_mcp_performance()
        
        if not all_metrics:
            return {}
        
        # Aggregate current system state
        return {
            'error_rate': 1.0 - np.mean([m.get('avg_success_rate', 1.0) for m in all_metrics.values()]),
            'latency': np.mean([m.get('avg_latency', 0) for m in all_metrics.values()]),
            'avg_prompt_tokens': np.mean([m.get('avg_prompt_tokens', 0) for m in all_metrics.values()]),
            'avg_completion_tokens': np.mean([m.get('avg_completion_tokens', 0) for m in all_metrics.values()]),
            'request_rate': sum([m.get('total_requests', 0) for m in all_metrics.values()]) / 60,  # per minute
            'cpu_usage': 0.5,  # Placeholder - would get from system monitoring
            'memory_usage': 0.6,  # Placeholder
            'disk_usage': 0.3   # Placeholder
        }
    
    def generate_grafana_dashboard_config(self) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration"""
        return {
            "dashboard": {
                "id": None,
                "title": "Production Monitoring - KGoT Alita Enhanced",
                "tags": ["production", "kgot", "alita", "mcp"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "MCP Accuracy Over Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "mcp_accuracy",
                                "legendFormat": "{{mcp_id}}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Accuracy (%)",
                                "min": 0,
                                "max": 100
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Token Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "mcp_prompt_tokens",
                                "legendFormat": "Prompt Tokens - {{mcp_id}}"
                            },
                            {
                                "expr": "mcp_completion_tokens",
                                "legendFormat": "Completion Tokens - {{mcp_id}}"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "System Health",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "avg(mcp_success_rate)",
                                "legendFormat": "Overall Success Rate"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Cost Analysis",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "mcp_cost",
                                "legendFormat": "Cost - {{mcp_id}}"
                            }
                        ]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }


class ProductionMonitoringSystem:
    """Main production monitoring system that orchestrates all components"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.logging_service = CentralizedLoggingService(
            log_directory=self.config.get('log_directory', 'logs/production'),
            max_log_size_mb=self.config.get('max_log_size_mb', 100),
            retention_days=self.config.get('retention_days', 30)
        )
        
        self.analytics_engine = RAGMCPPerformanceAnalytics(
            time_window_minutes=self.config.get('analytics_time_window', 60)
        )
        
        self.predictive_engine = PredictiveAnalyticsEngine(
            model_storage_path=self.config.get('model_storage_path', 'models/production'),
            retrain_interval_hours=self.config.get('retrain_interval_hours', 24)
        )
        
        self.dashboard = ProductionMonitoringDashboard(
            analytics_engine=self.analytics_engine,
            predictive_engine=self.predictive_engine,
            export_port=self.config.get('dashboard_port', 8090)
        )
        
        self.logger = logging.getLogger(f"{__name__}.ProductionMonitoringSystem")
        self.running = False
    
    async def start(self):
        """Start the production monitoring system"""
        if self.running:
            self.logger.warning("Production monitoring system already running")
            return
        
        try:
            # Start components
            await self.logging_service.start()
            await self.predictive_engine.load_models()
            await self.dashboard.start_metrics_server()
            
            self.running = True
            self.logger.info("Production monitoring system started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start production monitoring system: {e}")
            raise
    
    async def stop(self):
        """Stop the production monitoring system"""
        if not self.running:
            return
        
        try:
            await self.logging_service.stop()
            self.running = False
            self.logger.info("Production monitoring system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping production monitoring system: {e}")
    
    async def log_kgot_event(self, 
                           task_id: str,
                           message: str,
                           severity: LogSeverity,
                           kgot_snapshot: KGoTSnapshot,
                           correlation_id: Optional[str] = None):
        """Log KGoT event with knowledge graph snapshot"""
        log_entry = ProductionLogEntry(
            service_name="kgot_controller",
            service_type=ServiceType.KGOT,
            timestamp=datetime.now(),
            task_id=task_id,
            severity=severity,
            message=message,
            payload={"event_type": "kgot_execution"},
            correlation_id=correlation_id,
            kgot_snapshot=kgot_snapshot
        )
        
        await self.logging_service.ingest_log(log_entry)
    
    async def log_mcp_performance(self, 
                                task_id: str,
                                mcp_metrics: RAGMCPMetrics,
                                correlation_id: Optional[str] = None):
        """Log MCP performance metrics"""
        # Log the event
        log_entry = ProductionLogEntry(
            service_name=f"mcp_{mcp_metrics.mcp_id}",
            service_type=ServiceType.MCP,
            timestamp=datetime.now(),
            task_id=task_id,
            severity=LogSeverity.INFO,
            message=f"MCP performance metrics recorded",
            payload={"event_type": "mcp_performance"},
            correlation_id=correlation_id,
            rag_mcp_metrics=mcp_metrics
        )
        
        await self.logging_service.ingest_log(log_entry)
        
        # Record in analytics engine
        await self.analytics_engine.record_mcp_metrics(mcp_metrics)
    
    async def log_system_event(self,
                             service_name: str,
                             service_type: ServiceType,
                             task_id: str,
                             message: str,
                             severity: LogSeverity,
                             payload: Optional[Dict[str, Any]] = None,
                             correlation_id: Optional[str] = None):
        """Log general system event"""
        log_entry = ProductionLogEntry(
            service_name=service_name,
            service_type=service_type,
            timestamp=datetime.now(),
            task_id=task_id,
            severity=severity,
            message=message,
            payload=payload or {},
            correlation_id=correlation_id
        )
        
        await self.logging_service.ingest_log(log_entry)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'running': self.running,
            'components': {
                'logging_service': {
                    'status': 'running' if self.running else 'stopped',
                    'metrics': self.logging_service.get_logging_metrics()
                },
                'analytics_engine': {
                    'status': 'running',
                    'mcp_count': len(self.analytics_engine.get_all_mcp_performance())
                },
                'predictive_engine': {
                    'status': 'available' if ML_AVAILABLE else 'limited',
                    'models_loaded': len(self.predictive_engine.models)
                },
                'dashboard': {
                    'status': 'running',
                    'port': self.dashboard.export_port
                }
            },
            'timestamp': datetime.now().isoformat()
        }


# Factory function for easy initialization
def create_production_monitoring_system(config: Optional[Dict[str, Any]] = None) -> ProductionMonitoringSystem:
    """Create and configure production monitoring system"""
    return ProductionMonitoringSystem(config)


# Example usage and testing
if __name__ == "__main__":
    async def demo_production_monitoring():
        """Demonstrate production monitoring system"""
        print("=== Production Monitoring System Demo ===")
        
        # Create monitoring system
        monitoring = create_production_monitoring_system({
            'log_directory': 'logs/production_demo',
            'analytics_time_window': 30,  # 30 minutes
            'dashboard_port': 8091
        })
        
        try:
            # Start system
            await monitoring.start()
            print("✅ Production monitoring system started")
            
            # Simulate KGoT event with knowledge graph snapshot
            kgot_snapshot = KGoTSnapshot(
                graph_id="demo_graph_001",
                node_count=15,
                edge_count=28,
                active_thoughts=["analyze_user_query", "generate_response", "validate_output"],
                reasoning_path=["input_processing", "knowledge_retrieval", "reasoning", "output_generation"],
                confidence_scores={"relevance": 0.92, "accuracy": 0.88, "completeness": 0.85},
                timestamp=datetime.now(),
                serialized_graph='{"nodes": [], "edges": []}'
            )
            
            await monitoring.log_kgot_event(
                task_id="demo_task_001",
                message="KGoT reasoning completed successfully",
                severity=LogSeverity.INFO,
                kgot_snapshot=kgot_snapshot,
                correlation_id="demo_correlation_001"
            )
            print("✅ KGoT event logged")
            
            # Simulate MCP performance metrics
            mcp_metrics = RAGMCPMetrics(
                mcp_id="demo_mcp_search",
                accuracy_percentage=89.5,
                avg_prompt_tokens=150,
                avg_completion_tokens=75,
                latency_ms=245.0,
                success_rate=0.96,
                cost_per_request=0.0023,
                timestamp=datetime.now(),
                task_type="information_retrieval"
            )
            
            await monitoring.log_mcp_performance(
                task_id="demo_task_001",
                mcp_metrics=mcp_metrics,
                correlation_id="demo_correlation_001"
            )
            print("✅ MCP performance logged")
            
            # Wait a bit for processing
            await asyncio.sleep(2)
            
            # Get system status
            status = monitoring.get_system_status()
            print(f"\n📊 System Status:")
            print(f"Running: {status['running']}")
            print(f"Logs ingested: {status['components']['logging_service']['metrics']['logs_ingested']}")
            print(f"MCPs tracked: {status['components']['analytics_engine']['mcp_count']}")
            
            # Get MCP performance
            mcp_performance = monitoring.analytics_engine.get_mcp_performance("demo_mcp_search")
            if mcp_performance:
                print(f"\n🎯 MCP Performance:")
                print(f"Average Accuracy: {mcp_performance['avg_accuracy']:.2f}%")
                print(f"Average Latency: {mcp_performance['avg_latency']:.2f}ms")
                print(f"Average Cost: ${mcp_performance['avg_cost']:.4f}")
            
            print(f"\n🌐 Dashboard available at: http://localhost:{monitoring.dashboard.export_port}/dashboard/data")
            print(f"📈 Metrics endpoint: http://localhost:{monitoring.dashboard.export_port}/metrics")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
        finally:
            await monitoring.stop()
            print("✅ Production monitoring system stopped")
    
    # Run demo
    asyncio.run(demo_production_monitoring())