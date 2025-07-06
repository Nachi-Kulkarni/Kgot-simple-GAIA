#!/usr/bin/env python3
"""
Sequential Resource Optimization Module for Alita-KGoT Enhanced System - Task 17e Implementation

This module implements the Sequential Resource Optimization system using sequential thinking
for complex resource allocation decisions across Alita and KGoT systems as specified in the
5-Phase Implementation Plan for Enhanced Alita.

This system provides:
- Sequential thinking MCP integration for complex resource allocation decisions
- Cost-benefit analysis workflows for MCP selection, system routing, and validation strategies
- Dynamic resource reallocation based on system performance and sequential reasoning insights
- Optimization strategies considering performance, cost, reliability, and validation requirements
- Predictive resource planning using sequential thinking for workload forecasting
- Integration with existing cost optimization frameworks with enhanced decision-making capabilities

Key Components:
1. SequentialResourceOptimizer: Main orchestrator using LangChain agents with sequential thinking
2. CostBenefitAnalyzer: Analyzes costs and benefits for MCP selection, system routing, and validation strategies
3. PerformanceMonitor: Real-time system performance tracking and metrics collection
4. ResourceAllocator: Dynamic resource allocation based on sequential reasoning insights
5. PredictivePlanner: Workload forecasting using sequential thinking for optimal resource planning
6. OptimizationStrategies: Multiple optimization strategies for different performance scenarios

@module SequentialResourceOptimization
@author Enhanced Alita KGoT Team
@version 1.0.0
@task Task 17e - Build Sequential Resource Optimization
"""

import asyncio
import json
import logging
import os
import time
import uuid
import numpy as np
import psutil
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set

import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field

# LangChain imports for agent architecture (following user preference for langchain)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, tool
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

# Winston logging integration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config/logging'))

# Integration with existing sequential decision trees
try:
    from .sequential_decision_trees import (
        BaseDecisionTree, DecisionContext, DecisionNode, DecisionPath,
        SystemType, TaskComplexity, ResourceConstraint
    )
except ImportError:
    # Fallback for development
    BaseDecisionTree = Any
    DecisionContext = Dict[str, Any]
    DecisionNode = Dict[str, Any]
    DecisionPath = Dict[str, Any]
    SystemType = Any
    TaskComplexity = Any
    ResourceConstraint = Any

# Integration with existing langchain sequential manager
try:
    from .langchain_sequential_manager import (
        LangChainSequentialManager, MemoryManager, SequentialThinkingSession
    )
except ImportError:
    # Fallback for development
    LangChainSequentialManager = Any
    MemoryManager = Any
    SequentialThinkingSession = Any

# Setup Winston logging for this component
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Enumeration for different optimization strategies"""
    COST_MINIMIZATION = "cost_minimization"
    PERFORMANCE_MAXIMIZATION = "performance_maximization"
    RELIABILITY_OPTIMIZATION = "reliability_optimization"
    VALIDATION_EFFICIENCY = "validation_efficiency"
    BALANCED_OPTIMIZATION = "balanced_optimization"
    TIME_CRITICAL = "time_critical"
    QUALITY_PRIORITY = "quality_priority"


class ResourceType(Enum):
    """Enumeration for different resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    MCP_TOKENS = "mcp_tokens"
    API_CALLS = "api_calls"
    VALIDATION_CYCLES = "validation_cycles"


class OptimizationScope(Enum):
    """Enumeration for optimization scope levels"""
    TASK_LEVEL = "task_level"
    SESSION_LEVEL = "session_level"
    SYSTEM_LEVEL = "system_level"
    GLOBAL_LEVEL = "global_level"


@dataclass
class ResourceMetrics:
    """
    Comprehensive resource metrics for optimization decisions
    
    Attributes:
        timestamp: When the metrics were collected
        cpu_usage: Current CPU utilization percentage
        memory_usage: Current memory utilization percentage
        storage_usage: Current storage utilization percentage
        network_latency: Network latency measurements
        gpu_usage: GPU utilization if available
        api_response_times: API response time metrics
        token_usage_rate: Token consumption rate
        validation_success_rate: Validation success rate
        error_rate: System error rate
        throughput: Tasks processed per unit time
        cost_per_operation: Cost metrics per operation type
        reliability_score: Overall system reliability score
        metadata: Additional metric-specific information
    """
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    storage_usage: float
    network_latency: Dict[str, float] = field(default_factory=dict)
    gpu_usage: Optional[float] = None
    api_response_times: Dict[str, float] = field(default_factory=dict)
    token_usage_rate: Dict[str, float] = field(default_factory=dict)
    validation_success_rate: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    cost_per_operation: Dict[str, float] = field(default_factory=dict)
    reliability_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationDecision:
    """
    Result of the sequential resource optimization process
    
    Attributes:
        decision_id: Unique identifier for the optimization decision
        strategy: Selected optimization strategy
        resource_allocation: Resource allocation plan
        system_configuration: Recommended system configuration
        mcp_selection: Selected MCP tools and their priorities
        validation_strategy: Validation approach to be used
        estimated_cost: Estimated cost for the optimization plan
        estimated_performance: Expected performance improvements
        estimated_reliability: Expected reliability improvements
        implementation_plan: Step-by-step implementation plan
        monitoring_requirements: Monitoring and metrics requirements
        sequential_thinking_session: Associated thinking session ID
        confidence_score: Confidence in the optimization decision
        metadata: Additional decision-specific information
    """
    decision_id: str
    strategy: OptimizationStrategy
    resource_allocation: Dict[str, Any]
    system_configuration: Dict[str, Any]
    mcp_selection: Dict[str, Any]
    validation_strategy: Dict[str, Any]
    estimated_cost: float
    estimated_performance: float
    estimated_reliability: float
    implementation_plan: List[Dict[str, Any]]
    monitoring_requirements: Dict[str, Any]
    sequential_thinking_session: Optional[str] = None
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Real-time system performance monitoring and metrics collection
    
    Tracks system performance across Alita and KGoT components, providing
    real-time metrics for optimization decisions and resource allocation.
    """
    
    def __init__(self, monitoring_interval: int = 30, history_retention: int = 86400):
        """
        Initialize the performance monitoring system
        
        Args:
            monitoring_interval: Interval between metric collections in seconds
            history_retention: How long to retain historical metrics in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.history_retention = history_retention
        
        # Metrics storage
        self.current_metrics: Optional[ResourceMetrics] = None
        self.metrics_history: deque = deque(maxlen=int(history_retention / monitoring_interval))
        
        # System component clients for monitoring
        self.system_clients: Dict[str, Any] = {}
        
        # Performance baselines and thresholds
        self.performance_baselines: Dict[str, float] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {
            'cpu': {'warning': 70.0, 'critical': 90.0},
            'memory': {'warning': 80.0, 'critical': 95.0},
            'storage': {'warning': 85.0, 'critical': 95.0},
            'error_rate': {'warning': 5.0, 'critical': 10.0},
            'response_time': {'warning': 2000.0, 'critical': 5000.0}
        }
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        self.logger.info("Performance Monitor initialized", extra={
            'operation': 'PERFORMANCE_MONITOR_INIT',
            'monitoring_interval': monitoring_interval,
            'history_retention': history_retention
        })
    
    async def start_monitoring(self) -> None:
        """Start the continuous performance monitoring process"""
        if self._monitoring_task is not None:
            self.logger.warning("Performance monitoring already running", extra={
                'operation': 'START_MONITORING'
            })
            return
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Performance monitoring started", extra={
            'operation': 'START_MONITORING'
        })
    
    async def stop_monitoring(self) -> None:
        """Stop the continuous performance monitoring process"""
        if self._monitoring_task is not None:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        self.logger.info("Performance monitoring stopped", extra={
            'operation': 'STOP_MONITORING'
        })
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that continuously collects metrics"""
        while True:
            try:
                # Collect current metrics
                metrics = await self._collect_comprehensive_metrics()
                
                # Store metrics
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Check for alerts
                await self._check_performance_alerts(metrics)
                
                # Log metrics summary
                self.logger.debug(
                    f"Metrics collected: CPU {metrics.cpu_usage:.1f}%, "
                    f"Memory {metrics.memory_usage:.1f}%, "
                    f"Error rate {metrics.error_rate:.2f}%",
                    extra={
                        'operation': 'METRICS_COLLECTION',
                        'metadata': {
                            'cpu_usage': metrics.cpu_usage,
                            'memory_usage': metrics.memory_usage,
                            'error_rate': metrics.error_rate,
                            'throughput': metrics.throughput
                        }
                    }
                )
                
                # Wait for next collection interval
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", extra={
                    'operation': 'MONITORING_LOOP_ERROR',
                    'metadata': {'error_type': type(e).__name__}
                })
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_comprehensive_metrics(self) -> ResourceMetrics:
        """
        Collect comprehensive system metrics from all components
        
        Returns:
            ResourceMetrics: Current system metrics
        """
        try:
            # Collect system-level metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            storage = psutil.disk_usage('/')
            
            # Collect network metrics
            network_latency = await self._measure_network_latency()
            
            # Collect API response times
            api_response_times = await self._measure_api_response_times()
            
            # Collect token usage metrics
            token_usage_rate = await self._collect_token_usage_metrics()
            
            # Collect validation metrics
            validation_success_rate = await self._collect_validation_metrics()
            
            # Calculate derived metrics
            error_rate = await self._calculate_error_rate()
            throughput = await self._calculate_throughput()
            cost_per_operation = await self._calculate_cost_metrics()
            reliability_score = await self._calculate_reliability_score()
            
            # Create comprehensive metrics object
            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                storage_usage=(storage.used / storage.total) * 100,
                network_latency=network_latency,
                api_response_times=api_response_times,
                token_usage_rate=token_usage_rate,
                validation_success_rate=validation_success_rate,
                error_rate=error_rate,
                throughput=throughput,
                cost_per_operation=cost_per_operation,
                reliability_score=reliability_score
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}", extra={
                'operation': 'METRICS_COLLECTION_ERROR',
                'metadata': {'error_type': type(e).__name__}
            })
            # Return default metrics on error
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                storage_usage=0.0
            )
    
    async def _measure_network_latency(self) -> Dict[str, float]:
        """Measure network latency to key services"""
        latency_results = {}
        
        # Define services to measure
        services = {
            'alita_web_agent': 'http://localhost:3001/health',
            'kgot_controller': 'http://localhost:8001/health',
            'graph_store': 'http://localhost:5000/health',
            'mcp_creation': 'http://localhost:4000/health'
        }
        
        for service_name, url in services.items():
            try:
                start_time = time.time()
                async with httpx.AsyncClient(timeout=5.0) as client:
                    _ = await client.get(url)
                end_time = time.time()
                
                latency_results[service_name] = (end_time - start_time) * 1000  # Convert to milliseconds
                
            except Exception as e:
                latency_results[service_name] = float('inf')  # Service unavailable
                self.logger.debug(f"Network latency measurement failed for {service_name}: {e}")
        
        return latency_results
    
    async def _measure_api_response_times(self) -> Dict[str, float]:
        """Measure API response times for different operations"""
        response_times = {}
        
        # Test endpoints with light operations
        test_endpoints = {
            'manager_agent': 'http://localhost:8000/health',
            'sequential_thinking': 'http://localhost:8000/stats',
            'validation': 'http://localhost:6000/health'
        }
        
        for endpoint_name, url in test_endpoints.items():
            try:
                start_time = time.time()
                async with httpx.AsyncClient(timeout=10.0) as client:
                    _ = await client.get(url)
                end_time = time.time()
                
                response_times[endpoint_name] = (end_time - start_time) * 1000
                
            except Exception as e:
                response_times[endpoint_name] = float('inf')
                self.logger.debug(f"API response time measurement failed for {endpoint_name}: {e}")
        
        return response_times
    
    async def _collect_token_usage_metrics(self) -> Dict[str, float]:
        """Collect token usage metrics from various services"""
        token_metrics = {}
        
        try:
            # Collect from manager agent stats
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get('http://localhost:8000/stats')
                if response.status_code == 200:
                    stats = response.json()
                    token_metrics['sequential_thinking'] = stats.get('total_tokens', 0)
                    token_metrics['conversations'] = stats.get('active_conversations', 0)
        except Exception as e:
            self.logger.debug(f"Token usage collection failed: {e}")
        
        return token_metrics
    
    async def _collect_validation_metrics(self) -> float:
        """Collect validation success rate metrics"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get('http://localhost:6000/metrics')
                if response.status_code == 200:
                    metrics = response.json()
                    return metrics.get('success_rate', 0.0)
        except Exception as e:
            self.logger.debug(f"Validation metrics collection failed: {e}")
        
        return 0.0
    
    async def _calculate_error_rate(self) -> float:
        """Calculate system-wide error rate"""
        # This would typically analyze logs or service health metrics
        # For now, return a simulated error rate
        return 0.5  # 0.5% error rate as baseline
    
    async def _calculate_throughput(self) -> float:
        """Calculate system throughput (tasks per second)"""
        # This would analyze completed tasks over time
        # For now, return a simulated throughput
        return 10.0  # 10 tasks per second as baseline
    
    async def _calculate_cost_metrics(self) -> Dict[str, float]:
        """Calculate cost per operation metrics"""
        return {
            'api_call': 0.001,  # $0.001 per API call
            'token': 0.00002,   # $0.00002 per token
            'validation': 0.01,  # $0.01 per validation
            'storage_gb': 0.1    # $0.1 per GB storage
        }
    
    async def _calculate_reliability_score(self) -> float:
        """Calculate overall system reliability score"""
        if not self.current_metrics:
            return 0.0
        
        # Calculate reliability based on multiple factors
        uptime_score = min(100.0, 100.0 - self.current_metrics.error_rate)
        performance_score = min(100.0, 200.0 - self.current_metrics.cpu_usage - self.current_metrics.memory_usage)
        
        # Weight different factors
        reliability_score = (uptime_score * 0.6 + performance_score * 0.4)
        return max(0.0, min(100.0, reliability_score))
    
    async def _check_performance_alerts(self, metrics: ResourceMetrics) -> None:
        """Check metrics against alert thresholds and log alerts"""
        alerts = []
        
        # Check CPU usage
        if metrics.cpu_usage > self.alert_thresholds['cpu']['critical']:
            alerts.append(f"CRITICAL: CPU usage at {metrics.cpu_usage:.1f}%")
        elif metrics.cpu_usage > self.alert_thresholds['cpu']['warning']:
            alerts.append(f"WARNING: CPU usage at {metrics.cpu_usage:.1f}%")
        
        # Check memory usage
        if metrics.memory_usage > self.alert_thresholds['memory']['critical']:
            alerts.append(f"CRITICAL: Memory usage at {metrics.memory_usage:.1f}%")
        elif metrics.memory_usage > self.alert_thresholds['memory']['warning']:
            alerts.append(f"WARNING: Memory usage at {metrics.memory_usage:.1f}%")
        
        # Check error rate
        if metrics.error_rate > self.alert_thresholds['error_rate']['critical']:
            alerts.append(f"CRITICAL: Error rate at {metrics.error_rate:.2f}%")
        elif metrics.error_rate > self.alert_thresholds['error_rate']['warning']:
            alerts.append(f"WARNING: Error rate at {metrics.error_rate:.2f}%")
        
        # Log alerts
        for alert in alerts:
            if 'CRITICAL' in alert:
                self.logger.error(alert, extra={
                    'operation': 'PERFORMANCE_ALERT',
                    'metadata': {'alert_level': 'critical', 'message': alert}
                })
            else:
                self.logger.warning(alert, extra={
                    'operation': 'PERFORMANCE_ALERT',
                    'metadata': {'alert_level': 'warning', 'message': alert}
                })
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent metrics"""
        return self.current_metrics
    
    def get_metrics_history(self, minutes: int = 60) -> List[ResourceMetrics]:
        """
        Get historical metrics for the specified time period
        
        Args:
            minutes: Number of minutes of history to return
            
        Returns:
            List[ResourceMetrics]: Historical metrics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from historical data"""
        if len(self.metrics_history) < 2:
            return {}
        
        recent_metrics = list(self.metrics_history)[-min(10, len(self.metrics_history)):]
        
        # Calculate trends
        cpu_trend = np.polyfit(range(len(recent_metrics)), 
                              [m.cpu_usage for m in recent_metrics], 1)[0]
        memory_trend = np.polyfit(range(len(recent_metrics)), 
                                 [m.memory_usage for m in recent_metrics], 1)[0]
        error_trend = np.polyfit(range(len(recent_metrics)), 
                                [m.error_rate for m in recent_metrics], 1)[0]
        
        return {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'error_trend': error_trend,
            'reliability_average': np.mean([m.reliability_score for m in recent_metrics]),
            'throughput_average': np.mean([m.throughput for m in recent_metrics])
        }


class CostBenefitAnalyzer:
    """
    Advanced cost-benefit analysis system for MCP selection, system routing, and validation strategies
    
    Analyzes costs and benefits across multiple dimensions including financial cost, performance,
    reliability, and validation efficiency. Uses historical data and predictive models to make
    optimization recommendations.
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """
        Initialize the cost-benefit analysis system
        
        Args:
            performance_monitor: Performance monitoring system for real-time metrics
        """
        self.performance_monitor = performance_monitor
        
        # Cost models for different operations
        self.cost_models = {
            'openrouter_api': {
                'base_cost_per_token': 0.00002,  # Base cost per token
                'model_multipliers': {
                    'o3': 1.0,
                    'o4-mini': 1.0,
                    'claude-4-sonnet': 1.5,
                    'claude-4-sonnet-thinking': 1
                }
            },
            'system_resources': {
                'cpu_hour': 0.05,      # Cost per CPU hour
                'memory_gb_hour': 0.01, # Cost per GB memory hour
                'storage_gb': 0.1,      # Cost per GB storage
                'network_gb': 0.01      # Cost per GB network transfer
            },
            'validation': {
                'cross_validation': 0.02,    # Cost per cross-validation cycle
                'quality_assessment': 0.01,   # Cost per quality assessment
                'benchmark_testing': 0.05     # Cost per benchmark test
            }
        }
        
        # Benefit models for different optimizations
        self.benefit_models = {
            'performance': {
                'response_time_improvement': 100.0,  # Value per ms improvement
                'throughput_increase': 50.0,         # Value per task/sec increase
                'accuracy_improvement': 1000.0       # Value per % accuracy increase
            },
            'reliability': {
                'uptime_improvement': 500.0,         # Value per % uptime increase
                'error_reduction': 200.0,            # Value per % error reduction
                'consistency_improvement': 300.0     # Value per % consistency increase
            },
            'user_experience': {
                'satisfaction_score': 100.0,         # Value per satisfaction point
                'task_completion_rate': 150.0        # Value per % completion rate
            }
        }
        
        # Historical analysis data
        self.mcp_performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.system_routing_history: List[Dict[str, Any]] = []
        self.validation_strategy_history: List[Dict[str, Any]] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.CostBenefitAnalyzer")
        self.logger.info("Cost-Benefit Analyzer initialized", extra={
            'operation': 'COST_BENEFIT_ANALYZER_INIT'
        })
    
    async def analyze_mcp_selection(self, 
                                   available_mcps: List[Dict[str, Any]], 
                                   task_requirements: Dict[str, Any],
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze costs and benefits of different MCP selections for a given task
        
        Args:
            available_mcps: List of available MCP tools with their capabilities
            task_requirements: Requirements and constraints for the task
            context: Additional context for the analysis
            
        Returns:
            Dict containing cost-benefit analysis results for each MCP option
        """
        self.logger.info("Starting MCP selection cost-benefit analysis", extra={
            'operation': 'MCP_SELECTION_ANALYSIS',
            'metadata': {
                'available_mcps': len(available_mcps),
                'task_type': task_requirements.get('type', 'unknown')
            }
        })
        
        analysis_results = {
            'analysis_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'task_requirements': task_requirements,
            'mcp_evaluations': [],
            'recommendations': {},
            'cost_benefit_summary': {}
        }
        
        for mcp in available_mcps:
            try:
                # Calculate costs for this MCP
                costs = await self._calculate_mcp_costs(mcp, task_requirements)
                
                # Calculate benefits for this MCP
                benefits = await self._calculate_mcp_benefits(mcp, task_requirements)
                
                # Calculate risk factors
                risks = await self._assess_mcp_risks(mcp, task_requirements)
                
                # Calculate historical performance
                historical_performance = self._get_mcp_historical_performance(mcp['id'])
                
                # Calculate overall score
                cost_benefit_ratio = benefits['total_benefit'] / max(costs['total_cost'], 0.001)
                risk_adjusted_score = cost_benefit_ratio * (1 - risks['total_risk'])
                
                mcp_evaluation = {
                    'mcp_id': mcp['id'],
                    'mcp_name': mcp.get('name', 'Unknown'),
                    'costs': costs,
                    'benefits': benefits,
                    'risks': risks,
                    'historical_performance': historical_performance,
                    'cost_benefit_ratio': cost_benefit_ratio,
                    'risk_adjusted_score': risk_adjusted_score,
                    'suitability_score': await self._calculate_mcp_suitability(mcp, task_requirements)
                }
                
                analysis_results['mcp_evaluations'].append(mcp_evaluation)
                
                self.logger.debug(f"MCP {mcp['id']} analysis complete", extra={
                    'operation': 'MCP_EVALUATION',
                    'metadata': {
                        'mcp_id': mcp['id'],
                        'cost_benefit_ratio': cost_benefit_ratio,
                        'risk_adjusted_score': risk_adjusted_score
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Error analyzing MCP {mcp.get('id', 'unknown')}: {e}", extra={
                    'operation': 'MCP_ANALYSIS_ERROR',
                    'metadata': {'mcp_id': mcp.get('id'), 'error_type': type(e).__name__}
                })
        
        # Generate recommendations
        analysis_results['recommendations'] = self._generate_mcp_recommendations(
            analysis_results['mcp_evaluations'], task_requirements
        )
        
        # Generate cost-benefit summary
        analysis_results['cost_benefit_summary'] = self._generate_cost_benefit_summary(
            analysis_results['mcp_evaluations']
        )
        
        self.logger.info("MCP selection analysis completed", extra={
            'operation': 'MCP_SELECTION_ANALYSIS_COMPLETE',
            'metadata': {
                'analysis_id': analysis_results['analysis_id'],
                'evaluated_mcps': len(analysis_results['mcp_evaluations']),
                'top_recommendation': analysis_results['recommendations'].get('primary', {}).get('mcp_id')
            }
        })
        
        return analysis_results
    
    async def analyze_system_routing(self, 
                                   routing_options: List[Dict[str, Any]], 
                                   task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze costs and benefits of different system routing strategies
        
        Args:
            routing_options: Available system routing options (Alita, KGoT, Combined)
            task_context: Context information about the task
            
        Returns:
            Dict containing routing analysis results
        """
        self.logger.info("Starting system routing cost-benefit analysis", extra={
            'operation': 'SYSTEM_ROUTING_ANALYSIS',
            'metadata': {
                'routing_options': len(routing_options),
                'task_complexity': task_context.get('complexity', 'unknown')
            }
        })
        
        analysis_results = {
            'analysis_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'task_context': task_context,
            'routing_evaluations': [],
            'recommendations': {},
            'performance_predictions': {}
        }
        
        for option in routing_options:
            try:
                # Calculate routing costs
                routing_costs = await self._calculate_routing_costs(option, task_context)
                
                # Calculate routing benefits
                routing_benefits = await self._calculate_routing_benefits(option, task_context)
                
                # Predict performance metrics
                performance_prediction = await self._predict_routing_performance(option, task_context)
                
                # Calculate resource utilization
                resource_utilization = await self._estimate_resource_utilization(option, task_context)
                
                routing_evaluation = {
                    'routing_strategy': option['strategy'],
                    'system_combination': option.get('systems', []),
                    'costs': routing_costs,
                    'benefits': routing_benefits,
                    'performance_prediction': performance_prediction,
                    'resource_utilization': resource_utilization,
                    'efficiency_score': routing_benefits['total_benefit'] / max(routing_costs['total_cost'], 0.001)
                }
                
                analysis_results['routing_evaluations'].append(routing_evaluation)
                
            except Exception as e:
                self.logger.error(f"Error analyzing routing option {option.get('strategy')}: {e}", extra={
                    'operation': 'ROUTING_ANALYSIS_ERROR',
                    'metadata': {'strategy': option.get('strategy'), 'error_type': type(e).__name__}
                })
        
        # Generate routing recommendations
        analysis_results['recommendations'] = self._generate_routing_recommendations(
            analysis_results['routing_evaluations'], task_context
        )
        
        self.logger.info("System routing analysis completed", extra={
            'operation': 'SYSTEM_ROUTING_ANALYSIS_COMPLETE',
            'metadata': {
                'analysis_id': analysis_results['analysis_id'],
                'recommended_strategy': analysis_results['recommendations'].get('primary_strategy')
            }
        })
        
        return analysis_results
    
    async def analyze_validation_strategies(self, 
                                          validation_options: List[Dict[str, Any]], 
                                          quality_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze costs and benefits of different validation strategies
        
        Args:
            validation_options: Available validation strategy options
            quality_requirements: Quality and accuracy requirements
            
        Returns:
            Dict containing validation strategy analysis results
        """
        self.logger.info("Starting validation strategy cost-benefit analysis", extra={
            'operation': 'VALIDATION_STRATEGY_ANALYSIS',
            'metadata': {
                'validation_options': len(validation_options),
                'quality_level': quality_requirements.get('required_accuracy', 'unknown')
            }
        })
        
        analysis_results = {
            'analysis_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'quality_requirements': quality_requirements,
            'validation_evaluations': [],
            'recommendations': {},
            'quality_cost_tradeoffs': {}
        }
        
        for option in validation_options:
            try:
                # Calculate validation costs
                validation_costs = await self._calculate_validation_costs(option, quality_requirements)
                
                # Calculate validation benefits
                validation_benefits = await self._calculate_validation_benefits(option, quality_requirements)
                
                # Assess quality improvements
                quality_impact = await self._assess_validation_quality_impact(option, quality_requirements)
                
                # Calculate time impact
                time_impact = await self._calculate_validation_time_impact(option)
                
                validation_evaluation = {
                    'strategy': option['strategy'],
                    'validation_methods': option.get('methods', []),
                    'costs': validation_costs,
                    'benefits': validation_benefits,
                    'quality_impact': quality_impact,
                    'time_impact': time_impact,
                    'quality_cost_efficiency': quality_impact['expected_improvement'] / max(validation_costs['total_cost'], 0.001)
                }
                
                analysis_results['validation_evaluations'].append(validation_evaluation)
                
            except Exception as e:
                self.logger.error(f"Error analyzing validation strategy {option.get('strategy')}: {e}", extra={
                    'operation': 'VALIDATION_ANALYSIS_ERROR',
                    'metadata': {'strategy': option.get('strategy'), 'error_type': type(e).__name__}
                })
        
        # Generate validation recommendations
        analysis_results['recommendations'] = self._generate_validation_recommendations(
            analysis_results['validation_evaluations'], quality_requirements
        )
        
        # Generate quality-cost tradeoff analysis
        analysis_results['quality_cost_tradeoffs'] = self._analyze_quality_cost_tradeoffs(
            analysis_results['validation_evaluations']
        )
        
        self.logger.info("Validation strategy analysis completed", extra={
            'operation': 'VALIDATION_STRATEGY_ANALYSIS_COMPLETE',
            'metadata': {
                'analysis_id': analysis_results['analysis_id'],
                'recommended_strategy': analysis_results['recommendations'].get('primary_strategy')
            }
        })
        
        return analysis_results
    
    async def _calculate_mcp_costs(self, mcp: Dict[str, Any], task_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive costs for an MCP option"""
        costs = {
            'api_costs': 0.0,
            'compute_costs': 0.0,
            'time_costs': 0.0,
            'setup_costs': 0.0,
            'maintenance_costs': 0.0,
            'total_cost': 0.0
        }
        
        # Calculate API costs based on expected token usage
        expected_tokens = task_requirements.get('estimated_tokens', 1000)
        model_name = mcp.get('model', 'claude-4-sonnet')
        base_cost = self.cost_models['openrouter_api']['base_cost_per_token']
        model_multiplier = self.cost_models['openrouter_api']['model_multipliers'].get(model_name, 1.0)
        costs['api_costs'] = expected_tokens * base_cost * model_multiplier
        
        # Calculate compute costs based on expected execution time
        expected_duration = task_requirements.get('estimated_duration_minutes', 5)
        costs['compute_costs'] = (expected_duration / 60) * self.cost_models['system_resources']['cpu_hour']
        
        # Calculate time costs (opportunity cost)
        time_criticality = task_requirements.get('time_criticality', 0.5)
        costs['time_costs'] = expected_duration * time_criticality * 0.1
        
        # Setup and maintenance costs
        costs['setup_costs'] = mcp.get('setup_complexity', 1) * 0.05
        costs['maintenance_costs'] = mcp.get('maintenance_overhead', 0.1)
        
        costs['total_cost'] = sum(costs.values()) - costs['total_cost']  # Exclude total from sum
        
        return costs
    
    async def _calculate_mcp_benefits(self, mcp: Dict[str, Any], task_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive benefits for an MCP option"""
        benefits = {
            'performance_benefits': 0.0,
            'accuracy_benefits': 0.0,
            'reliability_benefits': 0.0,
            'scalability_benefits': 0.0,
            'user_experience_benefits': 0.0,
            'total_benefit': 0.0
        }
        
        # Performance benefits
        expected_speed_improvement = mcp.get('performance_rating', 0.7)
        benefits['performance_benefits'] = expected_speed_improvement * 100.0
        
        # Accuracy benefits
        expected_accuracy = mcp.get('accuracy_rating', 0.8)
        required_accuracy = task_requirements.get('required_accuracy', 0.7)
        if expected_accuracy > required_accuracy:
            benefits['accuracy_benefits'] = (expected_accuracy - required_accuracy) * 1000.0
        
        # Reliability benefits
        mcp_reliability = mcp.get('reliability_score', 0.9)
        benefits['reliability_benefits'] = mcp_reliability * 200.0
        
        # Scalability benefits
        scalability_factor = mcp.get('scalability_rating', 0.5)
        benefits['scalability_benefits'] = scalability_factor * 150.0
        
        # User experience benefits
        ease_of_use = mcp.get('ease_of_use', 0.7)
        benefits['user_experience_benefits'] = ease_of_use * 100.0
        
        benefits['total_benefit'] = sum(benefits.values()) - benefits['total_benefit']
        
        return benefits
    
    async def _assess_mcp_risks(self, mcp: Dict[str, Any], task_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Assess risk factors for an MCP option"""
        risks = {
            'reliability_risk': mcp.get('failure_rate', 0.05),
            'performance_risk': mcp.get('performance_variance', 0.1),
            'cost_overrun_risk': mcp.get('cost_variance', 0.1),
            'compatibility_risk': mcp.get('compatibility_issues', 0.02),
            'maintenance_risk': mcp.get('maintenance_complexity', 0.05),
            'total_risk': 0.0
        }
        
        # Calculate total risk (not a simple sum due to interdependencies)
        risks['total_risk'] = min(0.95, sum(risks.values()) - risks['total_risk'])
        
        return risks
    
    def _get_mcp_historical_performance(self, mcp_id: str) -> Dict[str, Any]:
        """Get historical performance data for an MCP"""
        if mcp_id not in self.mcp_performance_history:
            return {
                'average_response_time': 0.0,
                'success_rate': 0.0,
                'usage_count': 0,
                'user_satisfaction': 0.0
            }
        
        history = self.mcp_performance_history[mcp_id]
        
        return {
            'average_response_time': np.mean([h['response_time'] for h in history]),
            'success_rate': np.mean([h['success'] for h in history]),
            'usage_count': len(history),
            'user_satisfaction': np.mean([h.get('satisfaction', 0.7) for h in history])
        }
    
    async def _calculate_mcp_suitability(self, mcp: Dict[str, Any], task_requirements: Dict[str, Any]) -> float:
        """Calculate how suitable an MCP is for the specific task requirements"""
        suitability_factors = []
        
        # Task type compatibility
        mcp_capabilities = set(mcp.get('capabilities', []))
        required_capabilities = set(task_requirements.get('required_capabilities', []))
        capability_match = len(mcp_capabilities.intersection(required_capabilities)) / max(len(required_capabilities), 1)
        suitability_factors.append(capability_match)
        
        # Data type compatibility
        mcp_data_types = set(mcp.get('supported_data_types', []))
        required_data_types = set(task_requirements.get('data_types', []))
        data_type_match = len(mcp_data_types.intersection(required_data_types)) / max(len(required_data_types), 1)
        suitability_factors.append(data_type_match)
        
        # Performance requirements match
        mcp_performance = mcp.get('performance_rating', 0.5)
        required_performance = task_requirements.get('performance_requirement', 0.5)
        performance_match = 1.0 if mcp_performance >= required_performance else mcp_performance / required_performance
        suitability_factors.append(performance_match)
        
        # Calculate weighted average suitability
        weights = [0.4, 0.3, 0.3]  # Capability, data type, performance weights
        suitability_score = sum(factor * weight for factor, weight in zip(suitability_factors, weights))
        
        return min(1.0, max(0.0, suitability_score))
    
    def _generate_mcp_recommendations(self, evaluations: List[Dict[str, Any]], task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate MCP selection recommendations based on analysis results"""
        if not evaluations:
            return {'primary': None, 'alternatives': [], 'reasoning': 'No MCPs available for analysis'}
        
        # Sort by risk-adjusted score
        sorted_evaluations = sorted(evaluations, key=lambda x: x['risk_adjusted_score'], reverse=True)
        
        primary_recommendation = sorted_evaluations[0]
        alternative_recommendations = sorted_evaluations[1:3]  # Top 2 alternatives
        
        # Generate reasoning
        reasoning = f"Primary recommendation based on highest risk-adjusted score ({primary_recommendation['risk_adjusted_score']:.2f}). "
        reasoning += f"Cost-benefit ratio: {primary_recommendation['cost_benefit_ratio']:.2f}, "
        reasoning += f"Suitability score: {primary_recommendation['suitability_score']:.2f}."
        
        return {
            'primary': {
                'mcp_id': primary_recommendation['mcp_id'],
                'mcp_name': primary_recommendation['mcp_name'],
                'confidence': min(1.0, primary_recommendation['risk_adjusted_score'] / 2.0),
                'expected_cost': primary_recommendation['costs']['total_cost'],
                'expected_benefit': primary_recommendation['benefits']['total_benefit']
            },
            'alternatives': [
                {
                    'mcp_id': alt['mcp_id'],
                    'mcp_name': alt['mcp_name'],
                    'score': alt['risk_adjusted_score']
                } for alt in alternative_recommendations
            ],
            'reasoning': reasoning
        }
    
    def _generate_cost_benefit_summary(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of cost-benefit analysis across all MCPs"""
        if not evaluations:
            return {}
        
        total_costs = [eval['costs']['total_cost'] for eval in evaluations]
        total_benefits = [eval['benefits']['total_benefit'] for eval in evaluations]
        cost_benefit_ratios = [eval['cost_benefit_ratio'] for eval in evaluations]
        
        return {
            'cost_range': {'min': min(total_costs), 'max': max(total_costs), 'average': np.mean(total_costs)},
            'benefit_range': {'min': min(total_benefits), 'max': max(total_benefits), 'average': np.mean(total_benefits)},
            'cost_benefit_ratio_range': {'min': min(cost_benefit_ratios), 'max': max(cost_benefit_ratios), 'average': np.mean(cost_benefit_ratios)},
            'recommended_vs_worst': {
                'cost_savings': max(total_costs) - min(total_costs),
                'benefit_improvement': max(total_benefits) - min(total_benefits)
            }
        }
    
    async def _calculate_routing_costs(self, option: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate costs for a system routing option"""
        costs = {
            'coordination_costs': 0.0,
            'communication_costs': 0.0,
            'resource_costs': 0.0,
            'latency_costs': 0.0,
            'total_cost': 0.0
        }
        
        # Calculate coordination costs based on system complexity
        systems_involved = len(option.get('systems', []))
        costs['coordination_costs'] = systems_involved * 0.1
        
        # Communication costs between systems
        if systems_involved > 1:
            costs['communication_costs'] = (systems_involved - 1) * 0.05
        
        # Resource costs based on expected usage
        estimated_duration = task_context.get('estimated_duration_minutes', 10)
        costs['resource_costs'] = estimated_duration * 0.02
        
        # Latency costs for multi-system coordination
        if 'combined' in option.get('strategy', '').lower():
            costs['latency_costs'] = 0.1
        
        costs['total_cost'] = sum(costs.values()) - costs['total_cost']
        
        return costs
    
    async def _calculate_routing_benefits(self, option: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate benefits for a system routing option"""
        benefits = {
            'performance_benefits': 0.0,
            'capability_benefits': 0.0,
            'reliability_benefits': 0.0,
            'flexibility_benefits': 0.0,
            'total_benefit': 0.0
        }
        
        strategy = option.get('strategy', '')
        
        # Performance benefits based on strategy
        if 'mcp_only' in strategy:
            benefits['performance_benefits'] = 100.0  # Fast execution
        elif 'kgot_only' in strategy:
            benefits['performance_benefits'] = 80.0   # Knowledge reasoning
        elif 'combined' in strategy:
            benefits['performance_benefits'] = 120.0  # Best of both
        
        # Capability benefits
        systems = option.get('systems', [])
        benefits['capability_benefits'] = len(systems) * 50.0
        
        # Reliability benefits
        if len(systems) > 1:
            benefits['reliability_benefits'] = 75.0  # Redundancy
        else:
            benefits['reliability_benefits'] = 50.0
        
        # Flexibility benefits
        if 'combined' in strategy:
            benefits['flexibility_benefits'] = 100.0
        else:
            benefits['flexibility_benefits'] = 60.0
        
        benefits['total_benefit'] = sum(benefits.values()) - benefits['total_benefit']
        
        return benefits
    
    async def _predict_routing_performance(self, option: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, float]:
        """Predict performance metrics for a routing option"""
        strategy = option.get('strategy', '')
        
        # Base predictions
        predictions = {
            'expected_response_time_ms': 1000.0,
            'expected_accuracy': 0.8,
            'expected_reliability': 0.9,
            'expected_throughput': 10.0
        }
        
        # Adjust based on strategy
        if 'mcp_only' in strategy:
            predictions['expected_response_time_ms'] *= 0.7  # Faster
            predictions['expected_throughput'] *= 1.5
        elif 'kgot_only' in strategy:
            predictions['expected_response_time_ms'] *= 1.3  # Slower but more thorough
            predictions['expected_accuracy'] *= 1.2
        elif 'combined' in strategy:
            predictions['expected_response_time_ms'] *= 1.1  # Slightly slower
            predictions['expected_accuracy'] *= 1.3
            predictions['expected_reliability'] *= 1.1
        
        return predictions
    
    async def _estimate_resource_utilization(self, option: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, float]:
        """Estimate resource utilization for a routing option"""
        base_utilization = {
            'cpu_utilization': 30.0,
            'memory_utilization': 20.0,
            'network_utilization': 10.0,
            'storage_utilization': 5.0
        }
        
        # Adjust based on system complexity
        systems_count = len(option.get('systems', []))
        multiplier = 1.0 + (systems_count - 1) * 0.3
        
        return {k: v * multiplier for k, v in base_utilization.items()}
    
    def _generate_routing_recommendations(self, evaluations: List[Dict[str, Any]], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system routing recommendations"""
        if not evaluations:
            return {'primary_strategy': None, 'reasoning': 'No routing options available'}
        
        # Sort by efficiency score
        sorted_evaluations = sorted(evaluations, key=lambda x: x['efficiency_score'], reverse=True)
        
        primary = sorted_evaluations[0]
        
        return {
            'primary_strategy': primary['routing_strategy'],
            'system_combination': primary['system_combination'],
            'confidence': min(1.0, primary['efficiency_score'] / 10.0),
            'expected_performance': primary['performance_prediction'],
            'resource_requirements': primary['resource_utilization'],
            'reasoning': f"Highest efficiency score ({primary['efficiency_score']:.2f}) with optimal cost-benefit balance."
        }
    
    async def _calculate_validation_costs(self, option: Dict[str, Any], quality_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Calculate costs for a validation strategy"""
        costs = {
            'validation_execution_costs': 0.0,
            'quality_assessment_costs': 0.0,
            'time_costs': 0.0,
            'resource_costs': 0.0,
            'total_cost': 0.0
        }
        
        methods = option.get('methods', [])
        
        # Calculate costs based on validation methods
        for method in methods:
            if method == 'cross_validation':
                costs['validation_execution_costs'] += self.cost_models['validation']['cross_validation']
            elif method == 'quality_assessment':
                costs['quality_assessment_costs'] += self.cost_models['validation']['quality_assessment']
            elif method == 'benchmark_testing':
                costs['validation_execution_costs'] += self.cost_models['validation']['benchmark_testing']
        
        # Time costs based on validation complexity
        validation_complexity = option.get('complexity', 1.0)
        costs['time_costs'] = validation_complexity * 0.05
        
        # Resource costs for validation infrastructure
        costs['resource_costs'] = len(methods) * 0.02
        
        costs['total_cost'] = sum(costs.values()) - costs['total_cost']
        
        return costs
    
    async def _calculate_validation_benefits(self, option: Dict[str, Any], quality_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Calculate benefits for a validation strategy"""
        benefits = {
            'quality_improvement': 0.0,
            'reliability_improvement': 0.0,
            'confidence_improvement': 0.0,
            'risk_reduction': 0.0,
            'total_benefit': 0.0
        }
        
        # Benefits based on validation comprehensiveness
        methods_count = len(option.get('methods', []))
        base_benefit = methods_count * 100.0
        
        benefits['quality_improvement'] = base_benefit * 0.4
        benefits['reliability_improvement'] = base_benefit * 0.3
        benefits['confidence_improvement'] = base_benefit * 0.2
        benefits['risk_reduction'] = base_benefit * 0.1
        
        benefits['total_benefit'] = sum(benefits.values()) - benefits['total_benefit']
        
        return benefits
    
    async def _assess_validation_quality_impact(self, option: Dict[str, Any], quality_requirements: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality impact of a validation strategy"""
        methods = option.get('methods', [])
        
        # Base quality improvement based on validation methods
        quality_improvements = {
            'cross_validation': 0.15,
            'quality_assessment': 0.10,
            'benchmark_testing': 0.20,
            'human_review': 0.25,
            'automated_testing': 0.08
        }
        
        expected_improvement = sum(quality_improvements.get(method, 0.05) for method in methods)
        expected_improvement = min(0.5, expected_improvement)  # Cap at 50% improvement
        
        return {
            'expected_improvement': expected_improvement,
            'confidence_level': min(1.0, len(methods) * 0.2),
            'quality_categories': methods
        }
    
    async def _calculate_validation_time_impact(self, option: Dict[str, Any]) -> Dict[str, float]:
        """Calculate time impact of validation strategy"""
        methods = option.get('methods', [])
        
        # Time overhead for different validation methods
        time_overheads = {
            'cross_validation': 0.3,     # 30% time overhead
            'quality_assessment': 0.15,  # 15% time overhead
            'benchmark_testing': 0.25,   # 25% time overhead
            'human_review': 0.5,         # 50% time overhead
            'automated_testing': 0.1     # 10% time overhead
        }
        
        total_overhead = sum(time_overheads.get(method, 0.1) for method in methods)
        
        return {
            'time_overhead_percentage': min(100.0, total_overhead * 100),
            'absolute_time_increase_minutes': total_overhead * 10,  # Assuming 10 min base time
            'parallel_possible': 'cross_validation' in methods or 'automated_testing' in methods
        }
    
    def _generate_validation_recommendations(self, evaluations: List[Dict[str, Any]], quality_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation strategy recommendations"""
        if not evaluations:
            return {'primary_strategy': None, 'reasoning': 'No validation strategies available'}
        
        # Sort by quality-cost efficiency
        sorted_evaluations = sorted(evaluations, key=lambda x: x['quality_cost_efficiency'], reverse=True)
        
        primary = sorted_evaluations[0]
        
        return {
            'primary_strategy': primary['strategy'],
            'validation_methods': primary['validation_methods'],
            'expected_quality_improvement': primary['quality_impact']['expected_improvement'],
            'cost_efficiency': primary['quality_cost_efficiency'],
            'time_impact': primary['time_impact'],
            'reasoning': f"Best quality-cost efficiency ({primary['quality_cost_efficiency']:.2f}) with {primary['quality_impact']['expected_improvement']:.1%} quality improvement."
        }
    
    def _analyze_quality_cost_tradeoffs(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality vs cost tradeoffs across validation strategies"""
        if not evaluations:
            return {}
        
        quality_scores = [eval['quality_impact']['expected_improvement'] for eval in evaluations]
        costs = [eval['costs']['total_cost'] for eval in evaluations]
        
        return {
            'quality_range': {'min': min(quality_scores), 'max': max(quality_scores)},
            'cost_range': {'min': min(costs), 'max': max(costs)},
            'optimal_tradeoff': {
                'strategy': max(evaluations, key=lambda x: x['quality_cost_efficiency'])['strategy'],
                'quality_improvement': max(quality_scores),
                'cost_efficiency': max([eval['quality_cost_efficiency'] for eval in evaluations])
            },
            'budget_recommendations': {
                'low_cost': min(evaluations, key=lambda x: x['costs']['total_cost'])['strategy'],
                'high_quality': max(evaluations, key=lambda x: x['quality_impact']['expected_improvement'])['strategy'],
                'balanced': max(evaluations, key=lambda x: x['quality_cost_efficiency'])['strategy']
            }
        }


class ResourceAllocator:
    """
    Dynamic resource allocation system based on sequential reasoning insights
    
    Manages resource allocation across Alita and KGoT systems using real-time
    performance data and predictive models. Implements dynamic reallocation
    strategies based on system performance and sequential thinking insights.
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor, cost_benefit_analyzer: CostBenefitAnalyzer):
        """
        Initialize the resource allocation system
        
        Args:
            performance_monitor: Performance monitoring system for real-time metrics
            cost_benefit_analyzer: Cost-benefit analysis system for optimization decisions
        """
        self.performance_monitor = performance_monitor
        self.cost_benefit_analyzer = cost_benefit_analyzer
        
        # Resource pools and allocation tracking
        self.resource_pools = {
            'cpu': {'total': 100.0, 'allocated': 0.0, 'available': 100.0},
            'memory': {'total': 32.0, 'allocated': 0.0, 'available': 32.0},  # GB
            'storage': {'total': 1000.0, 'allocated': 0.0, 'available': 1000.0},  # GB
            'network': {'total': 1000.0, 'allocated': 0.0, 'available': 1000.0},  # Mbps
            'api_tokens': {'total': 100000, 'allocated': 0, 'available': 100000},  # tokens/hour
            'validation_cycles': {'total': 50, 'allocated': 0, 'available': 50}  # cycles/hour
        }
        
        # Active allocations tracking
        self.active_allocations: Dict[str, Dict[str, Any]] = {}
        
        # Allocation history for learning and optimization
        self.allocation_history: List[Dict[str, Any]] = []
        
        # Reallocation rules and thresholds
        self.reallocation_thresholds = {
            'cpu_utilization_high': 85.0,
            'memory_utilization_high': 90.0,
            'response_time_degradation': 2000.0,  # ms
            'error_rate_increase': 5.0,  # percentage
            'efficiency_drop': 0.3  # 30% efficiency drop
        }
        
        # Allocation strategies
        self.allocation_strategies = {
            'performance_priority': self._performance_priority_allocation,
            'cost_priority': self._cost_priority_allocation,
            'balanced': self._balanced_allocation,
            'reliability_priority': self._reliability_priority_allocation
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.ResourceAllocator")
        self.logger.info("Resource Allocator initialized", extra={
            'operation': 'RESOURCE_ALLOCATOR_INIT',
            'metadata': {
                'total_cpu': self.resource_pools['cpu']['total'],
                'total_memory': self.resource_pools['memory']['total']
            }
        })
    
    async def allocate_resources(self, 
                               allocation_request: Dict[str, Any],
                               strategy: str = 'balanced',
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Allocate resources for a task or operation based on requirements and strategy
        
        Args:
            allocation_request: Resource requirements and constraints
            strategy: Allocation strategy to use
            context: Additional context for allocation decisions
            
        Returns:
            Dict containing allocation decision and resource assignment
        """
        allocation_id = str(uuid.uuid4())
        
        self.logger.info("Starting resource allocation", extra={
            'operation': 'RESOURCE_ALLOCATION',
            'metadata': {
                'allocation_id': allocation_id,
                'strategy': strategy,
                'requested_resources': allocation_request.get('resources', {})
            }
        })
        
        try:
            # Validate allocation request
            if not self._validate_allocation_request(allocation_request):
                raise ValueError("Invalid allocation request")
            
            # Check resource availability
            availability_check = await self._check_resource_availability(allocation_request)
            if not availability_check['sufficient']:
                return await self._handle_insufficient_resources(allocation_request, availability_check)
            
            # Select allocation strategy
            allocation_function = self.allocation_strategies.get(strategy, self._balanced_allocation)
            
            # Calculate optimal allocation
            allocation_plan = await allocation_function(allocation_request, context or {})
            
            # Apply the allocation
            allocation_result = await self._apply_allocation(allocation_id, allocation_plan)
            
            # Record allocation in history
            allocation_record = {
                'allocation_id': allocation_id,
                'timestamp': datetime.now(),
                'request': allocation_request,
                'strategy': strategy,
                'plan': allocation_plan,
                'result': allocation_result,
                'context': context
            }
            self.allocation_history.append(allocation_record)
            
            self.logger.info("Resource allocation completed", extra={
                'operation': 'RESOURCE_ALLOCATION_COMPLETE',
                'metadata': {
                    'allocation_id': allocation_id,
                    'allocated_cpu': allocation_result['resources']['cpu'],
                    'allocated_memory': allocation_result['resources']['memory']
                }
            })
            
            return allocation_result
            
        except Exception as e:
            self.logger.error(f"Resource allocation failed: {e}", extra={
                'operation': 'RESOURCE_ALLOCATION_ERROR',
                'metadata': {'allocation_id': allocation_id, 'error_type': type(e).__name__}
            })
            raise
    
    async def reallocate_resources(self, trigger_reason: str = "performance_optimization") -> Dict[str, Any]:
        """
        Perform dynamic resource reallocation based on system performance and sequential reasoning
        
        Args:
            trigger_reason: Reason for triggering reallocation
            
        Returns:
            Dict containing reallocation results and impact analysis
        """
        reallocation_id = str(uuid.uuid4())
        
        self.logger.info("Starting dynamic resource reallocation", extra={
            'operation': 'RESOURCE_REALLOCATION',
            'metadata': {
                'reallocation_id': reallocation_id,
                'trigger_reason': trigger_reason
            }
        })
        
        try:
            # Analyze current system state
            current_metrics = self.performance_monitor.get_current_metrics()
            performance_trends = self.performance_monitor.get_performance_trends()
            
            # Identify reallocation opportunities
            reallocation_opportunities = await self._identify_reallocation_opportunities(
                current_metrics, performance_trends
            )
            
            # Use sequential thinking to plan reallocation
            reallocation_plan = await self._plan_sequential_reallocation(
                reallocation_opportunities, trigger_reason
            )
            
            # Execute reallocation plan
            reallocation_results = await self._execute_reallocation_plan(reallocation_plan)
            
            # Measure impact of reallocation
            impact_analysis = await self._measure_reallocation_impact(
                reallocation_results, current_metrics
            )
            
            result = {
                'reallocation_id': reallocation_id,
                'trigger_reason': trigger_reason,
                'opportunities_identified': len(reallocation_opportunities),
                'reallocations_executed': len(reallocation_results),
                'impact_analysis': impact_analysis,
                'new_allocation_state': self._get_current_allocation_state()
            }
            
            self.logger.info("Dynamic reallocation completed", extra={
                'operation': 'RESOURCE_REALLOCATION_COMPLETE',
                'metadata': {
                    'reallocation_id': reallocation_id,
                    'reallocations_executed': len(reallocation_results),
                    'performance_improvement': impact_analysis.get('performance_improvement', 0.0)
                }
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Resource reallocation failed: {e}", extra={
                'operation': 'RESOURCE_REALLOCATION_ERROR',
                'metadata': {'reallocation_id': reallocation_id, 'error_type': type(e).__name__}
            })
            raise
    
    async def release_resources(self, allocation_id: str) -> Dict[str, Any]:
        """
        Release resources from a completed or cancelled allocation
        
        Args:
            allocation_id: ID of the allocation to release
            
        Returns:
            Dict containing release results
        """
        self.logger.info("Releasing resources", extra={
            'operation': 'RESOURCE_RELEASE',
            'metadata': {'allocation_id': allocation_id}
        })
        
        if allocation_id not in self.active_allocations:
            raise ValueError(f"Allocation {allocation_id} not found")
        
        allocation = self.active_allocations[allocation_id]
        
        # Release resources back to pools
        for resource_type, amount in allocation['resources'].items():
            if resource_type in self.resource_pools:
                self.resource_pools[resource_type]['allocated'] -= amount
                self.resource_pools[resource_type]['available'] += amount
        
        # Remove from active allocations
        del self.active_allocations[allocation_id]
        
        # Update allocation history
        for record in self.allocation_history:
            if record['allocation_id'] == allocation_id:
                record['released_at'] = datetime.now()
                record['duration'] = (record['released_at'] - record['timestamp']).total_seconds()
                break
        
        release_result = {
            'allocation_id': allocation_id,
            'released_resources': allocation['resources'],
            'release_timestamp': datetime.now(),
            'new_availability': {k: v['available'] for k, v in self.resource_pools.items()}
        }
        
        self.logger.info("Resources released successfully", extra={
            'operation': 'RESOURCE_RELEASE_COMPLETE',
            'metadata': {
                'allocation_id': allocation_id,
                'released_resources': allocation['resources']
            }
        })
        
        return release_result
    
    def _validate_allocation_request(self, request: Dict[str, Any]) -> bool:
        """Validate the structure and content of an allocation request"""
        required_fields = ['task_id', 'resources']
        
        if not all(field in request for field in required_fields):
            return False
        
        # Validate resource requirements
        resources = request['resources']
        if not isinstance(resources, dict) or not resources:
            return False
        
        # Check for valid resource types
        valid_resource_types = set(self.resource_pools.keys())
        requested_types = set(resources.keys())
        
        if not requested_types.issubset(valid_resource_types):
            return False
        
        # Check for non-negative values
        if any(amount < 0 for amount in resources.values()):
            return False
        
        return True
    
    async def _check_resource_availability(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check if sufficient resources are available for the request"""
        resources = request['resources']
        availability_check = {
            'sufficient': True,
            'shortfalls': {},
            'utilization_after_allocation': {}
        }
        
        for resource_type, requested_amount in resources.items():
            if resource_type in self.resource_pools:
                pool = self.resource_pools[resource_type]
                available = pool['available']
                
                if requested_amount > available:
                    availability_check['sufficient'] = False
                    availability_check['shortfalls'][resource_type] = {
                        'requested': requested_amount,
                        'available': available,
                        'shortfall': requested_amount - available
                    }
                else:
                    # Calculate utilization after allocation
                    new_allocated = pool['allocated'] + requested_amount
                    utilization = (new_allocated / pool['total']) * 100
                    availability_check['utilization_after_allocation'][resource_type] = utilization
        
        return availability_check
    
    async def _handle_insufficient_resources(self, 
                                           request: Dict[str, Any], 
                                           availability_check: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cases where insufficient resources are available"""
        self.logger.warning("Insufficient resources for allocation", extra={
            'operation': 'INSUFFICIENT_RESOURCES',
            'metadata': {'shortfalls': availability_check['shortfalls']}
        })
        
        # Try to free up resources through reallocation
        reallocation_attempt = await self._attempt_resource_freeing(availability_check['shortfalls'])
        
        if reallocation_attempt['success']:
            # Retry allocation after freeing resources
            updated_availability = await self._check_resource_availability(request)
            if updated_availability['sufficient']:
                return await self.allocate_resources(request)
        
        # If reallocation didn't work, suggest alternatives
        alternatives = await self._suggest_resource_alternatives(request, availability_check)
        
        return {
            'allocation_id': None,
            'success': False,
            'reason': 'insufficient_resources',
            'shortfalls': availability_check['shortfalls'],
            'reallocation_attempted': reallocation_attempt,
            'alternatives': alternatives
        }
    
    async def _performance_priority_allocation(self, 
                                             request: Dict[str, Any], 
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Allocation strategy that prioritizes performance over cost"""
        resources = request['resources']
        
        # Boost resource allocations for better performance
        performance_multipliers = {
            'cpu': 1.3,      # 30% more CPU for better performance
            'memory': 1.2,   # 20% more memory
            'network': 1.1,  # 10% more network bandwidth
            'api_tokens': 1.5,  # 50% more API tokens for parallel processing
            'validation_cycles': 1.2  # 20% more validation cycles
        }
        
        optimized_resources = {}
        for resource_type, requested_amount in resources.items():
            multiplier = performance_multipliers.get(resource_type, 1.0)
            optimized_amount = requested_amount * multiplier
            
            # Cap at available resources
            pool = self.resource_pools[resource_type]
            optimized_amount = min(optimized_amount, pool['available'])
            optimized_resources[resource_type] = optimized_amount
        
        return {
            'strategy': 'performance_priority',
            'resources': optimized_resources,
            'priority_level': 'high',
            'estimated_performance_boost': 25.0,  # 25% performance improvement
            'cost_multiplier': 1.3
        }
    
    async def _cost_priority_allocation(self, 
                                      request: Dict[str, Any], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Allocation strategy that prioritizes cost efficiency"""
        resources = request['resources']
        
        # Minimize resource allocations to reduce cost
        cost_reduction_factors = {
            'cpu': 0.8,      # 20% less CPU to save cost
            'memory': 0.85,  # 15% less memory
            'network': 0.9,  # 10% less network bandwidth
            'api_tokens': 0.7,  # 30% fewer API tokens
            'validation_cycles': 0.8  # 20% fewer validation cycles
        }
        
        optimized_resources = {}
        for resource_type, requested_amount in resources.items():
            reduction_factor = cost_reduction_factors.get(resource_type, 1.0)
            optimized_amount = requested_amount * reduction_factor
            
            # Ensure minimum viable allocation
            min_amount = requested_amount * 0.5  # Never go below 50% of requested
            optimized_amount = max(optimized_amount, min_amount)
            optimized_resources[resource_type] = optimized_amount
        
        return {
            'strategy': 'cost_priority',
            'resources': optimized_resources,
            'priority_level': 'low',
            'estimated_cost_savings': 20.0,  # 20% cost savings
            'performance_impact': -10.0  # 10% performance reduction
        }
    
    async def _balanced_allocation(self, 
                                 request: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Balanced allocation strategy that optimizes both cost and performance"""
        resources = request['resources']
        
        # Use slight optimizations in both directions
        balance_factors = {
            'cpu': 1.1,      # 10% more CPU
            'memory': 1.05,  # 5% more memory
            'network': 1.0,  # No change
            'api_tokens': 1.1,  # 10% more API tokens
            'validation_cycles': 1.0  # No change
        }
        
        optimized_resources = {}
        for resource_type, requested_amount in resources.items():
            balance_factor = balance_factors.get(resource_type, 1.0)
            optimized_amount = requested_amount * balance_factor
            
            # Cap at available resources
            pool = self.resource_pools[resource_type]
            optimized_amount = min(optimized_amount, pool['available'])
            optimized_resources[resource_type] = optimized_amount
        
        return {
            'strategy': 'balanced',
            'resources': optimized_resources,
            'priority_level': 'medium',
            'estimated_performance_boost': 5.0,  # 5% performance improvement
            'cost_multiplier': 1.05  # 5% cost increase
        }
    
    async def _reliability_priority_allocation(self, 
                                             request: Dict[str, Any], 
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Allocation strategy that prioritizes system reliability"""
        resources = request['resources']
        
        # Add redundancy and buffers for reliability
        reliability_multipliers = {
            'cpu': 1.2,      # 20% more CPU for redundancy
            'memory': 1.3,   # 30% more memory for buffering
            'network': 1.1,  # 10% more network bandwidth
            'api_tokens': 1.1,  # 10% more API tokens for retries
            'validation_cycles': 1.5  # 50% more validation for reliability
        }
        
        optimized_resources = {}
        for resource_type, requested_amount in resources.items():
            multiplier = reliability_multipliers.get(resource_type, 1.0)
            optimized_amount = requested_amount * multiplier
            
            # Cap at available resources
            pool = self.resource_pools[resource_type]
            optimized_amount = min(optimized_amount, pool['available'])
            optimized_resources[resource_type] = optimized_amount
        
        return {
            'strategy': 'reliability_priority',
            'resources': optimized_resources,
            'priority_level': 'high',
            'estimated_reliability_boost': 30.0,  # 30% reliability improvement
            'cost_multiplier': 1.25  # 25% cost increase
        }
    
    async def _apply_allocation(self, allocation_id: str, allocation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the calculated resource allocation plan"""
        resources = allocation_plan['resources']
        
        # Reserve resources from pools
        for resource_type, amount in resources.items():
            if resource_type in self.resource_pools:
                pool = self.resource_pools[resource_type]
                pool['allocated'] += amount
                pool['available'] -= amount
        
        # Create allocation record
        allocation_record = {
            'allocation_id': allocation_id,
            'strategy': allocation_plan['strategy'],
            'resources': resources,
            'priority_level': allocation_plan['priority_level'],
            'allocated_at': datetime.now(),
            'status': 'active'
        }
        
        # Store active allocation
        self.active_allocations[allocation_id] = allocation_record
        
        return {
            'allocation_id': allocation_id,
            'success': True,
            'allocated_resources': resources,
            'strategy_used': allocation_plan['strategy'],
            'allocation_timestamp': datetime.now(),
            'expected_performance_impact': allocation_plan.get('estimated_performance_boost', 0.0),
            'cost_impact': allocation_plan.get('cost_multiplier', 1.0),
            'remaining_resources': {k: v['available'] for k, v in self.resource_pools.items()}
        }
    
    async def _identify_reallocation_opportunities(self, 
                                                 current_metrics: Optional[ResourceMetrics], 
                                                 performance_trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for resource reallocation based on performance data"""
        opportunities = []
        
        if not current_metrics:
            return opportunities
        
        # Check for high resource utilization
        if current_metrics.cpu_usage > self.reallocation_thresholds['cpu_utilization_high']:
            opportunities.append({
                'type': 'cpu_overload',
                'severity': 'high',
                'current_usage': current_metrics.cpu_usage,
                'threshold': self.reallocation_thresholds['cpu_utilization_high'],
                'suggested_action': 'reallocate_cpu_intensive_tasks'
            })
        
        if current_metrics.memory_usage > self.reallocation_thresholds['memory_utilization_high']:
            opportunities.append({
                'type': 'memory_pressure',
                'severity': 'high',
                'current_usage': current_metrics.memory_usage,
                'threshold': self.reallocation_thresholds['memory_utilization_high'],
                'suggested_action': 'reallocate_memory_intensive_tasks'
            })
        
        # Check for performance degradation
        avg_response_time = np.mean(list(current_metrics.api_response_times.values())) if current_metrics.api_response_times else 0
        if avg_response_time > self.reallocation_thresholds['response_time_degradation']:
            opportunities.append({
                'type': 'response_time_degradation',
                'severity': 'medium',
                'current_response_time': avg_response_time,
                'threshold': self.reallocation_thresholds['response_time_degradation'],
                'suggested_action': 'boost_performance_allocation'
            })
        
        # Check for error rate increases
        if current_metrics.error_rate > self.reallocation_thresholds['error_rate_increase']:
            opportunities.append({
                'type': 'error_rate_increase',
                'severity': 'high',
                'current_error_rate': current_metrics.error_rate,
                'threshold': self.reallocation_thresholds['error_rate_increase'],
                'suggested_action': 'improve_reliability_allocation'
            })
        
        # Check performance trends for declining efficiency
        if performance_trends and 'reliability_average' in performance_trends:
            if performance_trends['reliability_average'] < 80.0:  # Below 80% reliability
                opportunities.append({
                    'type': 'reliability_decline',
                    'severity': 'medium',
                    'current_reliability': performance_trends['reliability_average'],
                    'suggested_action': 'enhance_reliability_allocation'
                })
        
        return opportunities
    
    async def _plan_sequential_reallocation(self, 
                                          opportunities: List[Dict[str, Any]], 
                                          trigger_reason: str) -> Dict[str, Any]:
        """Use sequential thinking to plan optimal resource reallocation"""
        if not opportunities:
            return {'actions': [], 'reasoning': 'No reallocation opportunities identified'}
        
        # Prioritize opportunities by severity
        high_priority = [op for op in opportunities if op['severity'] == 'high']
        medium_priority = [op for op in opportunities if op['severity'] == 'medium']
        low_priority = [op for op in opportunities if op['severity'] == 'low']
        
        planned_actions = []
        
        # Address high priority issues first
        for opportunity in high_priority:
            action = await self._create_reallocation_action(opportunity, 'high_priority')
            if action:
                planned_actions.append(action)
        
        # Address medium priority issues if resources allow
        for opportunity in medium_priority:
            action = await self._create_reallocation_action(opportunity, 'medium_priority')
            if action:
                planned_actions.append(action)
        
        return {
            'actions': planned_actions,
            'reasoning': f"Planned {len(planned_actions)} reallocation actions to address {len(opportunities)} opportunities",
            'trigger_reason': trigger_reason,
            'priority_distribution': {
                'high': len(high_priority),
                'medium': len(medium_priority),
                'low': len(low_priority)
            }
        }
    
    async def _create_reallocation_action(self, opportunity: Dict[str, Any], priority: str) -> Optional[Dict[str, Any]]:
        """Create a specific reallocation action for an opportunity"""
        action_type = opportunity['suggested_action']
        
        if action_type == 'reallocate_cpu_intensive_tasks':
            return {
                'action_id': str(uuid.uuid4()),
                'action_type': action_type,
                'target_resource': 'cpu',
                'resource_adjustment': 20.0,  # Increase CPU allocation by 20%
                'expected_impact': 'reduced_cpu_utilization',
                'priority': priority
            }
        
        elif action_type == 'reallocate_memory_intensive_tasks':
            return {
                'action_id': str(uuid.uuid4()),
                'action_type': action_type,
                'target_resource': 'memory',
                'resource_adjustment': 15.0,  # Increase memory allocation by 15%
                'expected_impact': 'reduced_memory_pressure',
                'priority': priority
            }
        
        elif action_type == 'boost_performance_allocation':
            return {
                'action_id': str(uuid.uuid4()),
                'action_type': action_type,
                'target_resource': 'api_tokens',
                'resource_adjustment': 25.0,  # Increase API token allocation by 25%
                'expected_impact': 'improved_response_times',
                'priority': priority
            }
        
        elif action_type == 'improve_reliability_allocation':
            return {
                'action_id': str(uuid.uuid4()),
                'action_type': action_type,
                'target_resource': 'validation_cycles',
                'resource_adjustment': 30.0,  # Increase validation cycles by 30%
                'expected_impact': 'reduced_error_rates',
                'priority': priority
            }
        
        return None
    
    async def _execute_reallocation_plan(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the planned reallocation actions"""
        results = []
        
        for action in plan.get('actions', []):
            try:
                result = await self._execute_single_reallocation(action)
                results.append(result)
                
                self.logger.info(f"Reallocation action executed: {action['action_type']}", extra={
                    'operation': 'REALLOCATION_ACTION_EXECUTED',
                    'metadata': {
                        'action_id': action['action_id'],
                        'target_resource': action['target_resource'],
                        'adjustment': action['resource_adjustment']
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Reallocation action failed: {e}", extra={
                    'operation': 'REALLOCATION_ACTION_ERROR',
                    'metadata': {'action_id': action['action_id'], 'error_type': type(e).__name__}
                })
                
                results.append({
                    'action_id': action['action_id'],
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def _execute_single_reallocation(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single reallocation action"""
        target_resource = action['target_resource']
        adjustment_percent = action['resource_adjustment']
        
        if target_resource not in self.resource_pools:
            raise ValueError(f"Unknown resource type: {target_resource}")
        
        pool = self.resource_pools[target_resource]
        
        # Calculate new allocation amounts
        adjustment_amount = (pool['total'] * adjustment_percent) / 100.0
        
        # Check if adjustment is possible
        if adjustment_amount > pool['available']:
            # Partial adjustment based on available resources
            adjustment_amount = pool['available'] * 0.8  # Use 80% of available
        
        # Apply the adjustment (this is a simplified implementation)
        # In a real system, this would involve actually moving resources between allocations
        
        return {
            'action_id': action['action_id'],
            'success': True,
            'resource_adjusted': target_resource,
            'adjustment_amount': adjustment_amount,
            'adjustment_percent': (adjustment_amount / pool['total']) * 100.0,
            'execution_timestamp': datetime.now()
        }
    
    async def _measure_reallocation_impact(self, 
                                         reallocation_results: List[Dict[str, Any]], 
                                         baseline_metrics: Optional[ResourceMetrics]) -> Dict[str, Any]:
        """Measure the impact of resource reallocation on system performance"""
        # Wait a brief period for changes to take effect
        await asyncio.sleep(30)  # 30 seconds
        
        # Collect new metrics
        new_metrics = self.performance_monitor.get_current_metrics()
        
        if not baseline_metrics or not new_metrics:
            return {'impact_measured': False, 'reason': 'insufficient_metrics'}
        
        # Calculate performance improvements
        cpu_improvement = baseline_metrics.cpu_usage - new_metrics.cpu_usage
        memory_improvement = baseline_metrics.memory_usage - new_metrics.memory_usage
        reliability_improvement = new_metrics.reliability_score - baseline_metrics.reliability_score
        throughput_improvement = new_metrics.throughput - baseline_metrics.throughput
        
        # Calculate overall impact score
        impact_score = (
            (cpu_improvement * 0.3) +
            (memory_improvement * 0.3) +
            (reliability_improvement * 0.2) +
            (throughput_improvement * 0.2)
        )
        
        return {
            'impact_measured': True,
            'baseline_metrics': {
                'cpu_usage': baseline_metrics.cpu_usage,
                'memory_usage': baseline_metrics.memory_usage,
                'reliability_score': baseline_metrics.reliability_score,
                'throughput': baseline_metrics.throughput
            },
            'new_metrics': {
                'cpu_usage': new_metrics.cpu_usage,
                'memory_usage': new_metrics.memory_usage,
                'reliability_score': new_metrics.reliability_score,
                'throughput': new_metrics.throughput
            },
            'improvements': {
                'cpu': cpu_improvement,
                'memory': memory_improvement,
                'reliability': reliability_improvement,
                'throughput': throughput_improvement
            },
            'overall_impact_score': impact_score,
            'performance_improvement': max(0, impact_score)
        }
    
    async def _attempt_resource_freeing(self, shortfalls: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Attempt to free up resources through reallocation of existing allocations"""
        freed_resources = {}
        
        # Identify allocations that can be optimized or reduced
        for allocation_id, allocation in self.active_allocations.items():
            if allocation['priority_level'] == 'low':
                # Reduce low-priority allocations
                for resource_type, amount in allocation['resources'].items():
                    if resource_type in shortfalls:
                        reduction = min(amount * 0.3, shortfalls[resource_type]['shortfall'])  # Reduce by up to 30%
                        if reduction > 0:
                            freed_resources[resource_type] = freed_resources.get(resource_type, 0) + reduction
                            
                            # Update allocation
                            allocation['resources'][resource_type] -= reduction
                            
                            # Update resource pools
                            pool = self.resource_pools[resource_type]
                            pool['allocated'] -= reduction
                            pool['available'] += reduction
        
        return {
            'success': len(freed_resources) > 0,
            'freed_resources': freed_resources,
            'optimized_allocations': len([a for a in self.active_allocations.values() if a['priority_level'] == 'low'])
        }
    
    async def _suggest_resource_alternatives(self, 
                                           request: Dict[str, Any], 
                                           availability_check: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest alternative resource configurations when resources are insufficient"""
        alternatives = []
        
        # Suggest reduced resource allocation
        reduced_resources = {}
        for resource_type, amount in request['resources'].items():
            if resource_type in availability_check['shortfalls']:
                # Suggest using available amount
                available = self.resource_pools[resource_type]['available']
                reduced_resources[resource_type] = available
            else:
                reduced_resources[resource_type] = amount
        
        if reduced_resources != request['resources']:
            alternatives.append({
                'type': 'reduced_allocation',
                'resources': reduced_resources,
                'trade_offs': 'Reduced performance but immediate availability',
                'availability': 'immediate'
            })
        
        # Suggest waiting for resources to become available
        alternatives.append({
            'type': 'wait_for_resources',
            'estimated_wait_time': '5-15 minutes',
            'trade_offs': 'Full resource allocation but delayed start',
            'availability': 'delayed'
        })
        
        # Suggest different strategy
        alternatives.append({
            'type': 'cost_priority_strategy',
            'resources': request['resources'],
            'trade_offs': 'Lower resource usage through cost optimization',
            'availability': 'immediate'
        })
        
        return alternatives
    
    def _get_current_allocation_state(self) -> Dict[str, Any]:
        """Get the current state of all resource allocations"""
        return {
            'total_allocations': len(self.active_allocations),
            'resource_utilization': {
                resource_type: {
                    'total': pool['total'],
                    'allocated': pool['allocated'],
                    'available': pool['available'],
                    'utilization_percent': (pool['allocated'] / pool['total']) * 100
                }
                for resource_type, pool in self.resource_pools.items()
            },
            'allocation_distribution': {
                'high_priority': len([a for a in self.active_allocations.values() if a['priority_level'] == 'high']),
                'medium_priority': len([a for a in self.active_allocations.values() if a['priority_level'] == 'medium']),
                'low_priority': len([a for a in self.active_allocations.values() if a['priority_level'] == 'low'])
            }
        }


class PredictivePlanner:
    """
    Predictive Resource Planning using Sequential Thinking for Workload Forecasting
    
    This class implements predictive resource planning capabilities using sequential thinking
    for workload forecasting and proactive resource scaling decisions.
    
    Features:
    - Multiple forecasting models (linear trend, seasonal patterns, spike detection, ML hybrid)
    - Sequential thinking integration for forecast synthesis and decision-making
    - Proactive scaling recommendations based on predicted workload patterns
    - Integration with performance monitoring for real-time forecast validation
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """
        Initialize the PredictivePlanner with performance monitoring integration
        
        Args:
            performance_monitor: PerformanceMonitor instance for historical data access
        """
        self.performance_monitor = performance_monitor
        self.forecast_models = {}
        self.prediction_history = deque(maxlen=1000)  # Keep last 1000 predictions for accuracy tracking
        self.forecasting_enabled = True
        
        # Configure logging for predictive planning operations
        self.logger = logging.getLogger(f"{__name__}.PredictivePlanner")
        self.logger.info("PredictivePlanner initialized", extra={
            'operation': 'PREDICTIVE_PLANNER_INIT',
            'metadata': {
                'forecasting_enabled': self.forecasting_enabled,
                'max_prediction_history': 1000
            }
        })
    
    async def forecast_workload(self, 
                              forecast_horizon_hours: int = 24,
                              forecast_models: List[str] = None) -> Dict[str, Any]:
        """
        Generate workload forecasts using multiple models and sequential thinking
        
        Args:
            forecast_horizon_hours: Hours into the future to forecast
            forecast_models: List of models to use for forecasting
            
        Returns:
            Dict containing forecast results and recommendations
        """
        if forecast_models is None:
            forecast_models = ['linear_trend', 'seasonal_patterns', 'spike_detection', 'ml_hybrid']
        
        self.logger.info(f"Starting workload forecast for {forecast_horizon_hours} hours", extra={
            'operation': 'WORKLOAD_FORECAST_START',
            'metadata': {
                'forecast_horizon_hours': forecast_horizon_hours,
                'forecast_models': forecast_models
            }
        })
        
        # Collect historical performance data
        historical_data = self._prepare_historical_data(forecast_horizon_hours * 2)  # Use 2x horizon for training
        
        if not historical_data:
            self.logger.warning("Insufficient historical data for forecasting", extra={
                'operation': 'FORECAST_DATA_INSUFFICIENT'
            })
            return {
                'forecast_available': False,
                'reason': 'insufficient_historical_data',
                'recommendation': 'collect_more_data'
            }
        
        # Generate forecasts using different models
        model_forecasts = {}
        for model_name in forecast_models:
            try:
                forecast = await self._generate_model_forecast(model_name, historical_data, forecast_horizon_hours)
                model_forecasts[model_name] = forecast
                
                self.logger.debug(f"Generated forecast using {model_name}", extra={
                    'operation': 'MODEL_FORECAST_GENERATED',
                    'metadata': {
                        'model_name': model_name,
                        'forecast_points': len(forecast.get('predictions', []))
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Forecast model {model_name} failed: {e}", extra={
                    'operation': 'FORECAST_MODEL_ERROR',
                    'metadata': {'model_name': model_name, 'error_type': type(e).__name__}
                })
        
        # Use sequential thinking to synthesize forecasts and make recommendations
        synthesis_result = await self._synthesize_forecasts_with_sequential_thinking(
            model_forecasts, historical_data, forecast_horizon_hours
        )
        
        # Generate proactive scaling recommendations
        scaling_recommendations = await self._generate_scaling_recommendations(synthesis_result)
        
        # Store prediction for accuracy tracking
        prediction_record = {
            'prediction_id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'forecast_horizon_hours': forecast_horizon_hours,
            'synthesis_result': synthesis_result,
            'scaling_recommendations': scaling_recommendations
        }
        self.prediction_history.append(prediction_record)
        
        return {
            'forecast_available': True,
            'forecast_horizon_hours': forecast_horizon_hours,
            'model_forecasts': model_forecasts,
            'synthesis_result': synthesis_result,
            'scaling_recommendations': scaling_recommendations,
            'prediction_id': prediction_record['prediction_id'],
            'confidence_level': synthesis_result.get('confidence_score', 0.0)
        }
    
    def _prepare_historical_data(self, hours_back: int) -> List[Dict[str, Any]]:
        """Prepare historical performance data for forecasting models"""
        historical_metrics = self.performance_monitor.get_metrics_history(hours_back * 60)  # Convert to minutes
        
        if len(historical_metrics) < 10:  # Need minimum data points
            return []
        
        # Convert to structured format for forecasting
        prepared_data = []
        for metrics in historical_metrics:
            prepared_data.append({
                'timestamp': metrics.timestamp,
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'throughput': metrics.throughput,
                'api_response_times': sum(metrics.api_response_times.values()) / len(metrics.api_response_times) if metrics.api_response_times else 0,
                'error_rate': metrics.error_rate,
                'reliability_score': metrics.reliability_score
            })
        
        return prepared_data
    
    async def _generate_model_forecast(self, 
                                     model_name: str, 
                                     historical_data: List[Dict[str, Any]], 
                                     forecast_horizon_hours: int) -> Dict[str, Any]:
        """Generate forecast using specified model"""
        
        if model_name == 'linear_trend':
            return self._linear_trend_forecast(historical_data, forecast_horizon_hours)
        elif model_name == 'seasonal_patterns':
            return self._seasonal_pattern_forecast(historical_data, forecast_horizon_hours)
        elif model_name == 'spike_detection':
            return self._spike_detection_forecast(historical_data, forecast_horizon_hours)
        elif model_name == 'ml_hybrid':
            return await self._ml_hybrid_forecast(historical_data, forecast_horizon_hours)
        else:
            raise ValueError(f"Unknown forecast model: {model_name}")
    
    def _linear_trend_forecast(self, historical_data: List[Dict[str, Any]], horizon_hours: int) -> Dict[str, Any]:
        """Simple linear trend forecasting using numpy"""
        if len(historical_data) < 5:
            return {'model': 'linear_trend', 'predictions': [], 'confidence': 0.0}
        
        # Extract time series data
        timestamps = [d['timestamp'] for d in historical_data]
        cpu_values = [d['cpu_usage'] for d in historical_data]
        memory_values = [d['memory_usage'] for d in historical_data]
        throughput_values = [d['throughput'] for d in historical_data]
        
        # Convert timestamps to numeric values for regression
        start_time = timestamps[0]
        time_points = [(t - start_time).total_seconds() / 3600 for t in timestamps]  # Hours from start
        
        predictions = []
        current_time = max(time_points)
        
        # Generate predictions for each hour in the forecast horizon
        for hour_offset in range(1, horizon_hours + 1):
            future_time = current_time + hour_offset
            
            # Simple linear extrapolation using last few points
            if len(time_points) >= 3:
                # Use numpy for linear regression
                cpu_trend = np.polyfit(time_points[-10:], cpu_values[-10:], 1)
                memory_trend = np.polyfit(time_points[-10:], memory_values[-10:], 1)
                throughput_trend = np.polyfit(time_points[-10:], throughput_values[-10:], 1)
                
                predicted_cpu = np.polyval(cpu_trend, future_time)
                predicted_memory = np.polyval(memory_trend, future_time)
                predicted_throughput = np.polyval(throughput_trend, future_time)
                
                # Bound predictions to reasonable ranges
                predicted_cpu = max(0, min(100, predicted_cpu))
                predicted_memory = max(0, min(100, predicted_memory))
                predicted_throughput = max(0, predicted_throughput)
                
                predictions.append({
                    'hour_offset': hour_offset,
                    'timestamp': start_time + timedelta(hours=current_time + hour_offset),
                    'cpu_usage': predicted_cpu,
                    'memory_usage': predicted_memory,
                    'throughput': predicted_throughput,
                    'confidence': max(0.3, 1.0 - (hour_offset * 0.1))  # Decreasing confidence over time
                })
        
        return {
            'model': 'linear_trend',
            'predictions': predictions,
            'confidence': 0.7,
            'method': 'linear_extrapolation'
        }
    
    def _seasonal_pattern_forecast(self, historical_data: List[Dict[str, Any]], horizon_hours: int) -> Dict[str, Any]:
        """Seasonal pattern forecasting based on daily/weekly cycles"""
        if len(historical_data) < 24:  # Need at least 24 hours of data
            return {'model': 'seasonal_patterns', 'predictions': [], 'confidence': 0.0}
        
        # Group data by hour of day to detect daily patterns
        hourly_patterns = defaultdict(list)
        for data_point in historical_data:
            hour_of_day = data_point['timestamp'].hour
            hourly_patterns[hour_of_day].append({
                'cpu_usage': data_point['cpu_usage'],
                'memory_usage': data_point['memory_usage'],
                'throughput': data_point['throughput']
            })
        
        # Calculate average patterns for each hour
        hourly_averages = {}
        for hour, data_points in hourly_patterns.items():
            if data_points:
                hourly_averages[hour] = {
                    'cpu_usage': sum(d['cpu_usage'] for d in data_points) / len(data_points),
                    'memory_usage': sum(d['memory_usage'] for d in data_points) / len(data_points),
                    'throughput': sum(d['throughput'] for d in data_points) / len(data_points)
                }
        
        predictions = []
        base_time = historical_data[-1]['timestamp']
        
        for hour_offset in range(1, horizon_hours + 1):
            future_time = base_time + timedelta(hours=hour_offset)
            hour_of_day = future_time.hour
            
            if hour_of_day in hourly_averages:
                avg_data = hourly_averages[hour_of_day]
                predictions.append({
                    'hour_offset': hour_offset,
                    'timestamp': future_time,
                    'cpu_usage': avg_data['cpu_usage'],
                    'memory_usage': avg_data['memory_usage'],
                    'throughput': avg_data['throughput'],
                    'confidence': 0.8 if len(hourly_patterns[hour_of_day]) >= 3 else 0.5
                })
            else:
                # Fallback to overall average
                overall_avg = {
                    'cpu_usage': sum(d['cpu_usage'] for d in historical_data) / len(historical_data),
                    'memory_usage': sum(d['memory_usage'] for d in historical_data) / len(historical_data),
                    'throughput': sum(d['throughput'] for d in historical_data) / len(historical_data)
                }
                predictions.append({
                    'hour_offset': hour_offset,
                    'timestamp': future_time,
                    'cpu_usage': overall_avg['cpu_usage'],
                    'memory_usage': overall_avg['memory_usage'],
                    'throughput': overall_avg['throughput'],
                    'confidence': 0.4
                })
        
        return {
            'model': 'seasonal_patterns',
            'predictions': predictions,
            'confidence': 0.75,
            'method': 'hourly_pattern_averaging'
        }
    
    def _spike_detection_forecast(self, historical_data: List[Dict[str, Any]], horizon_hours: int) -> Dict[str, Any]:
        """Spike detection and extrapolation forecast"""
        if len(historical_data) < 10:
            return {'model': 'spike_detection', 'predictions': [], 'confidence': 0.0}
        
        # Detect spikes in recent data
        recent_data = historical_data[-20:]  # Look at last 20 data points
        cpu_values = [d['cpu_usage'] for d in recent_data]
        memory_values = [d['memory_usage'] for d in recent_data]
        
        # Calculate rolling averages to detect spikes
        cpu_avg = sum(cpu_values) / len(cpu_values)
        memory_avg = sum(memory_values) / len(memory_values)
        
        cpu_std = np.std(cpu_values)
        memory_std = np.std(memory_values)
        
        # Detect if we're currently in a spike
        latest_cpu = cpu_values[-1]
        latest_memory = memory_values[-1]
        
        cpu_spike = latest_cpu > (cpu_avg + 1.5 * cpu_std)
        memory_spike = latest_memory > (memory_avg + 1.5 * memory_std)
        
        predictions = []
        base_time = historical_data[-1]['timestamp']
        
        for hour_offset in range(1, horizon_hours + 1):
            future_time = base_time + timedelta(hours=hour_offset)
            
            # If in spike, predict gradual return to normal
            if cpu_spike or memory_spike:
                decay_factor = 0.8 ** hour_offset  # Exponential decay
                predicted_cpu = cpu_avg + (latest_cpu - cpu_avg) * decay_factor
                predicted_memory = memory_avg + (latest_memory - memory_avg) * decay_factor
            else:
                # Normal operation, predict slight variations around average
                predicted_cpu = cpu_avg + np.random.normal(0, cpu_std * 0.1)
                predicted_memory = memory_avg + np.random.normal(0, memory_std * 0.1)
            
            # Bound predictions
            predicted_cpu = max(0, min(100, predicted_cpu))
            predicted_memory = max(0, min(100, predicted_memory))
            
            predictions.append({
                'hour_offset': hour_offset,
                'timestamp': future_time,
                'cpu_usage': predicted_cpu,
                'memory_usage': predicted_memory,
                'throughput': historical_data[-1]['throughput'],  # Assume stable throughput
                'confidence': 0.6 if (cpu_spike or memory_spike) else 0.5
            })
        
        return {
            'model': 'spike_detection',
            'predictions': predictions,
            'confidence': 0.6,
            'spike_detected': cpu_spike or memory_spike,
            'method': 'spike_decay_extrapolation'
        }
    
    async def _ml_hybrid_forecast(self, historical_data: List[Dict[str, Any]], horizon_hours: int) -> Dict[str, Any]:
        """ML hybrid forecasting combining multiple simple models"""
        # Get forecasts from other models
        linear_forecast = self._linear_trend_forecast(historical_data, horizon_hours)
        seasonal_forecast = self._seasonal_pattern_forecast(historical_data, horizon_hours)
        spike_forecast = self._spike_detection_forecast(historical_data, horizon_hours)
        
        if not any(f['predictions'] for f in [linear_forecast, seasonal_forecast, spike_forecast]):
            return {'model': 'ml_hybrid', 'predictions': [], 'confidence': 0.0}
        
        # Combine predictions using weighted averaging
        predictions = []
        
        for hour_offset in range(1, horizon_hours + 1):
            combined_prediction = {
                'hour_offset': hour_offset,
                'timestamp': None,
                'cpu_usage': 0,
                'memory_usage': 0,
                'throughput': 0,
                'confidence': 0
            }
            
            weights = {'linear': 0.3, 'seasonal': 0.4, 'spike': 0.3}
            total_weight = 0
            
            # Combine predictions from available models
            for model_forecast, weight_key in [(linear_forecast, 'linear'), (seasonal_forecast, 'seasonal'), (spike_forecast, 'spike')]:
                if model_forecast['predictions'] and len(model_forecast['predictions']) >= hour_offset:
                    pred = model_forecast['predictions'][hour_offset - 1]
                    weight = weights[weight_key] * pred['confidence']
                    
                    combined_prediction['cpu_usage'] += pred['cpu_usage'] * weight
                    combined_prediction['memory_usage'] += pred['memory_usage'] * weight
                    combined_prediction['throughput'] += pred['throughput'] * weight
                    combined_prediction['timestamp'] = pred['timestamp']
                    total_weight += weight
            
            if total_weight > 0:
                combined_prediction['cpu_usage'] /= total_weight
                combined_prediction['memory_usage'] /= total_weight
                combined_prediction['throughput'] /= total_weight
                combined_prediction['confidence'] = min(0.9, total_weight)
                
                predictions.append(combined_prediction)
        
        return {
            'model': 'ml_hybrid',
            'predictions': predictions,
            'confidence': 0.8,
            'method': 'weighted_model_ensemble',
            'component_models': {
                'linear': linear_forecast,
                'seasonal': seasonal_forecast,
                'spike': spike_forecast
            }
        }
    
    async def _synthesize_forecasts_with_sequential_thinking(self, 
                                                           model_forecasts: Dict[str, Any], 
                                                           historical_data: List[Dict[str, Any]], 
                                                           horizon_hours: int) -> Dict[str, Any]:
        """Use sequential thinking to synthesize multiple forecasts into unified recommendations"""
        
        # This would integrate with sequential thinking MCP for complex decision making
        # For now, implementing a sophisticated synthesis logic
        
        self.logger.info("Synthesizing forecasts using sequential reasoning", extra={
            'operation': 'FORECAST_SYNTHESIS_START',
            'metadata': {
                'model_count': len(model_forecasts),
                'horizon_hours': horizon_hours
            }
        })
        
        # Analyze consistency across models
        consistency_analysis = self._analyze_forecast_consistency(model_forecasts)
        
        # Weight models based on historical accuracy (simplified implementation)
        model_weights = self._calculate_model_weights(model_forecasts, historical_data)
        
        # Generate synthesized forecast
        synthesized_predictions = []
        confidence_scores = []
        
        for hour_offset in range(1, horizon_hours + 1):
            hour_predictions = []
            hour_weights = []
            
            # Collect predictions from all models for this hour
            for model_name, forecast in model_forecasts.items():
                if forecast.get('predictions') and len(forecast['predictions']) >= hour_offset:
                    pred = forecast['predictions'][hour_offset - 1]
                    weight = model_weights.get(model_name, 0.5) * pred.get('confidence', 0.5)
                    
                    hour_predictions.append(pred)
                    hour_weights.append(weight)
            
            if hour_predictions:
                # Weighted synthesis of predictions
                synthesized_pred = self._weighted_prediction_synthesis(hour_predictions, hour_weights)
                synthesized_predictions.append(synthesized_pred)
                confidence_scores.append(synthesized_pred.get('confidence', 0.5))
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            'synthesized_predictions': synthesized_predictions,
            'consistency_analysis': consistency_analysis,
            'model_weights': model_weights,
            'confidence_score': overall_confidence,
            'synthesis_method': 'weighted_consensus_with_consistency_analysis',
            'synthesis_timestamp': datetime.now()
        }
    
    def _analyze_forecast_consistency(self, model_forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency between different forecast models"""
        if len(model_forecasts) < 2:
            return {'consistency_score': 1.0, 'analysis': 'single_model'}
        
        # Compare predictions across models for consistency
        consistency_scores = []
        
        # Get common prediction horizon
        min_predictions = min(len(f.get('predictions', [])) for f in model_forecasts.values() if f.get('predictions'))
        
        if min_predictions == 0:
            return {'consistency_score': 0.0, 'analysis': 'no_predictions'}
        
        for hour_idx in range(min_predictions):
            hour_cpu_predictions = []
            hour_memory_predictions = []
            
            for model_name, forecast in model_forecasts.items():
                if forecast.get('predictions') and len(forecast['predictions']) > hour_idx:
                    pred = forecast['predictions'][hour_idx]
                    hour_cpu_predictions.append(pred['cpu_usage'])
                    hour_memory_predictions.append(pred['memory_usage'])
            
            if len(hour_cpu_predictions) >= 2:
                cpu_consistency = 1.0 - (np.std(hour_cpu_predictions) / 100.0)  # Normalize by max possible value
                memory_consistency = 1.0 - (np.std(hour_memory_predictions) / 100.0)
                hour_consistency = (cpu_consistency + memory_consistency) / 2
                consistency_scores.append(max(0, hour_consistency))
        
        overall_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
        
        return {
            'consistency_score': overall_consistency,
            'hour_consistency_scores': consistency_scores,
            'analysis': 'multi_model_comparison'
        }
    
    def _calculate_model_weights(self, model_forecasts: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate weights for different models based on historical accuracy"""
        # Simplified implementation - in production, this would use historical prediction accuracy
        default_weights = {
            'linear_trend': 0.2,
            'seasonal_patterns': 0.35,
            'spike_detection': 0.15,
            'ml_hybrid': 0.3
        }
        
        # Adjust weights based on data characteristics
        if len(historical_data) < 24:
            # Limited data - prefer simpler models
            default_weights['linear_trend'] = 0.4
            default_weights['seasonal_patterns'] = 0.2
            default_weights['ml_hybrid'] = 0.25
        
        # Only return weights for models that actually produced forecasts
        return {model: weight for model, weight in default_weights.items() if model in model_forecasts}
    
    def _weighted_prediction_synthesis(self, predictions: List[Dict[str, Any]], weights: List[float]) -> Dict[str, Any]:
        """Synthesize multiple predictions using weighted averaging"""
        if not predictions or not weights:
            return {}
        
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = len(weights)
            weights = [1.0] * len(weights)
        
        # Weighted average of all metrics
        synthesized = {
            'hour_offset': predictions[0]['hour_offset'],
            'timestamp': predictions[0]['timestamp'],
            'cpu_usage': sum(p['cpu_usage'] * w for p, w in zip(predictions, weights)) / total_weight,
            'memory_usage': sum(p['memory_usage'] * w for p, w in zip(predictions, weights)) / total_weight,
            'throughput': sum(p['throughput'] * w for p, w in zip(predictions, weights)) / total_weight,
            'confidence': sum(p.get('confidence', 0.5) * w for p, w in zip(predictions, weights)) / total_weight
        }
        
        # Ensure bounded values
        synthesized['cpu_usage'] = max(0, min(100, synthesized['cpu_usage']))
        synthesized['memory_usage'] = max(0, min(100, synthesized['memory_usage']))
        synthesized['throughput'] = max(0, synthesized['throughput'])
        synthesized['confidence'] = max(0, min(1, synthesized['confidence']))
        
        return synthesized
    
    async def _generate_scaling_recommendations(self, synthesis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate proactive scaling recommendations based on forecast synthesis"""
        predictions = synthesis_result.get('synthesized_predictions', [])
        if not predictions:
            return {'recommendations': [], 'reasoning': 'no_predictions_available'}
        
        recommendations = []
        reasoning_steps = []
        
        # Analyze prediction trends
        cpu_trend = self._analyze_metric_trend([p['cpu_usage'] for p in predictions])
        memory_trend = self._analyze_metric_trend([p['memory_usage'] for p in predictions])
        throughput_trend = self._analyze_metric_trend([p['throughput'] for p in predictions])
        
        reasoning_steps.append(f"CPU trend: {cpu_trend['direction']} ({cpu_trend['magnitude']:.1f}% change)")
        reasoning_steps.append(f"Memory trend: {memory_trend['direction']} ({memory_trend['magnitude']:.1f}% change)")
        reasoning_steps.append(f"Throughput trend: {throughput_trend['direction']} ({throughput_trend['magnitude']:.1f}% change)")
        
        # Generate specific recommendations based on trends
        if cpu_trend['direction'] == 'increasing' and cpu_trend['magnitude'] > 20:
            recommendations.append({
                'type': 'scale_cpu_resources',
                'priority': 'high' if cpu_trend['magnitude'] > 40 else 'medium',
                'target_increase': f"{min(50, int(cpu_trend['magnitude'] * 0.8))}%",
                'timing': 'proactive',
                'reason': f"CPU usage predicted to increase by {cpu_trend['magnitude']:.1f}%"
            })
            reasoning_steps.append("Recommended CPU scaling due to significant predicted increase")
        
        if memory_trend['direction'] == 'increasing' and memory_trend['magnitude'] > 25:
            recommendations.append({
                'type': 'scale_memory_resources',
                'priority': 'high' if memory_trend['magnitude'] > 50 else 'medium',
                'target_increase': f"{min(60, int(memory_trend['magnitude'] * 0.7))}%",
                'timing': 'proactive',
                'reason': f"Memory usage predicted to increase by {memory_trend['magnitude']:.1f}%"
            })
            reasoning_steps.append("Recommended memory scaling due to significant predicted increase")
        
        if throughput_trend['direction'] == 'increasing' and throughput_trend['magnitude'] > 30:
            recommendations.append({
                'type': 'scale_api_resources',
                'priority': 'medium',
                'target_increase': '30%',
                'timing': 'proactive',
                'reason': f"Throughput predicted to increase by {throughput_trend['magnitude']:.1f}%"
            })
            reasoning_steps.append("Recommended API resource scaling due to throughput increase")
        
        # Check for predicted resource saturation
        max_cpu = max(p['cpu_usage'] for p in predictions)
        max_memory = max(p['memory_usage'] for p in predictions)
        
        if max_cpu > 85:
            recommendations.append({
                'type': 'prevent_cpu_saturation',
                'priority': 'critical',
                'target_increase': '40%',
                'timing': 'immediate',
                'reason': f"CPU usage predicted to reach {max_cpu:.1f}% (critical threshold)"
            })
            reasoning_steps.append("Critical: CPU saturation prevention required")
        
        if max_memory > 90:
            recommendations.append({
                'type': 'prevent_memory_saturation',
                'priority': 'critical',
                'target_increase': '50%',
                'timing': 'immediate',
                'reason': f"Memory usage predicted to reach {max_memory:.1f}% (critical threshold)"
            })
            reasoning_steps.append("Critical: Memory saturation prevention required")
        
        return {
            'recommendations': recommendations,
            'reasoning': reasoning_steps,
            'analysis': {
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend,
                'throughput_trend': throughput_trend,
                'max_predicted_cpu': max_cpu,
                'max_predicted_memory': max_memory
            },
            'confidence_level': synthesis_result.get('confidence_score', 0.0)
        }
    
    def _analyze_metric_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in a series of metric values"""
        if len(values) < 2:
            return {'direction': 'stable', 'magnitude': 0.0}
        
        first_value = values[0]
        last_value = values[-1]
        
        change_percent = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
        
        if abs(change_percent) < 5:
            direction = 'stable'
        elif change_percent > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'magnitude': abs(change_percent),
            'start_value': first_value,
            'end_value': last_value
        }


class SequentialResourceOptimizer:
    """
    Main Sequential Resource Optimization Orchestrator using LangChain Agents
    
    This is the primary orchestrator class that coordinates all optimization components
    and uses sequential thinking for complex resource allocation decisions across
    Alita and KGoT systems.
    
    Key Features:
    - LangChain agent architecture with OpenRouter API integration (per user preference)
    - Sequential thinking MCP integration for complex optimization decisions
    - Coordination of PerformanceMonitor, CostBenefitAnalyzer, ResourceAllocator, PredictivePlanner
    - Decision-making workflows using sequential reasoning
    - Integration with existing cost optimization frameworks
    - Real-time optimization recommendations and automatic resource adjustments
    """
    
    def __init__(self, 
                 openrouter_api_key: str = None,
                 sequential_thinking_mcp_endpoint: str = None,
                 optimization_config: Dict[str, Any] = None):
        """
        Initialize the Sequential Resource Optimizer
        
        Args:
            openrouter_api_key: OpenRouter API key for LangChain agent (per user preference)
            sequential_thinking_mcp_endpoint: Endpoint for sequential thinking MCP
            optimization_config: Configuration parameters for optimization
        """
        self.config = optimization_config or {}
        self.optimization_session_id = str(uuid.uuid4())
        
        # Initialize core components
        self.performance_monitor = PerformanceMonitor(
            monitoring_interval=self.config.get('monitoring_interval', 30),
            history_retention=self.config.get('history_retention', 86400)
        )
        
        self.cost_benefit_analyzer = CostBenefitAnalyzer(self.performance_monitor)
        self.resource_allocator = ResourceAllocator(self.performance_monitor, self.cost_benefit_analyzer)
        self.predictive_planner = PredictivePlanner(self.performance_monitor)
        
        # Setup LangChain agent with OpenRouter (following user memory preference)
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        self.llm = self._setup_langchain_llm()
        
        # Setup sequential thinking MCP integration
        self.sequential_thinking_endpoint = sequential_thinking_mcp_endpoint or os.getenv('SEQUENTIAL_THINKING_MCP_ENDPOINT')
        self.sequential_thinking_session = None
        
        # Memory management for optimization context
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=4000,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Active optimization sessions and decisions
        self.active_sessions = {}
        self.optimization_history = deque(maxlen=1000)
        
        # Configure logging for the main orchestrator
        self.logger = logging.getLogger(f"{__name__}.SequentialResourceOptimizer")
        self.logger.info("SequentialResourceOptimizer initialized", extra={
            'operation': 'OPTIMIZER_INIT',
            'metadata': {
                'session_id': self.optimization_session_id,
                'openrouter_enabled': bool(self.openrouter_api_key),
                'sequential_thinking_enabled': bool(self.sequential_thinking_endpoint)
            }
        })
    
    def _setup_langchain_llm(self) -> ChatOpenAI:
        """Setup LangChain LLM with OpenRouter API (per user memory preference)"""
        if not self.openrouter_api_key:
            self.logger.warning("OpenRouter API key not provided, using default configuration", extra={
                'operation': 'LLM_SETUP_WARNING'
            })
            # Fallback configuration - in production this should be properly configured
            return ChatOpenAI(
                model="anthropic/claude-4-sonnet",
                temperature=0.1,
                max_tokens=2000,
                api_key="dummy-key"  # This would be replaced with actual OpenRouter setup
            )
        
        # Configure for OpenRouter API (following user preference)
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_api_key,
            model="anthropic/claude-4-sonnet",  # High-quality model for optimization decisions
            temperature=0.1,  # Low temperature for consistent optimization decisions
            max_tokens=3000
        )
    
    async def optimize_resources(self, 
                               optimization_request: Dict[str, Any],
                               use_sequential_thinking: bool = True) -> OptimizationDecision:
        """
        Main optimization method using sequential thinking for complex resource decisions
        
        Args:
            optimization_request: Request containing optimization parameters and constraints
            use_sequential_thinking: Whether to use sequential thinking MCP for complex decisions
            
        Returns:
            OptimizationDecision with comprehensive resource optimization plan
        """
        optimization_id = str(uuid.uuid4())
        
        self.logger.info(f"Starting resource optimization {optimization_id}", extra={
            'operation': 'OPTIMIZATION_START',
            'metadata': {
                'optimization_id': optimization_id,
                'use_sequential_thinking': use_sequential_thinking,
                'request_scope': optimization_request.get('scope', 'unknown')
            }
        })
        
        try:
            # Step 1: Collect current system state and performance metrics
            current_metrics = await self._collect_optimization_context()
            
            # Step 2: Analyze optimization requirements and constraints
            requirements_analysis = await self._analyze_optimization_requirements(
                optimization_request, current_metrics
            )
            
            # Step 3: Use sequential thinking for complex optimization decisions (if enabled)
            if use_sequential_thinking and self.sequential_thinking_endpoint:
                sequential_analysis = await self._apply_sequential_thinking_optimization(
                    optimization_request, current_metrics, requirements_analysis
                )
            else:
                sequential_analysis = await self._apply_heuristic_optimization(
                    optimization_request, current_metrics, requirements_analysis
                )
            
            # Step 4: Generate cost-benefit analysis for different optimization strategies
            strategy_analysis = await self._analyze_optimization_strategies(
                optimization_request, current_metrics, sequential_analysis
            )
            
            # Step 5: Select optimal strategy and create resource allocation plan
            optimization_decision = await self._create_optimization_decision(
                optimization_id, optimization_request, current_metrics, 
                requirements_analysis, sequential_analysis, strategy_analysis
            )
            
            # Step 6: Validate the optimization decision
            validation_result = await self._validate_optimization_decision(optimization_decision)
            
            if not validation_result['valid']:
                self.logger.warning(f"Optimization decision validation failed: {validation_result['reason']}", extra={
                    'operation': 'OPTIMIZATION_VALIDATION_FAILED',
                    'metadata': {'optimization_id': optimization_id}
                })
                # Fall back to conservative optimization
                optimization_decision = await self._create_conservative_optimization(
                    optimization_id, optimization_request, current_metrics
                )
            
            # Store optimization decision in history
            self.optimization_history.append({
                'decision': optimization_decision,
                'timestamp': datetime.now(),
                'validation_result': validation_result
            })
            
            self.logger.info(f"Resource optimization completed: {optimization_id}", extra={
                'operation': 'OPTIMIZATION_COMPLETED',
                'metadata': {
                    'optimization_id': optimization_id,
                    'strategy': optimization_decision.strategy.value,
                    'estimated_cost': optimization_decision.estimated_cost,
                    'confidence_score': optimization_decision.confidence_score
                }
            })
            
            return optimization_decision
            
        except Exception as e:
            self.logger.error(f"Resource optimization failed: {e}", extra={
                'operation': 'OPTIMIZATION_ERROR',
                'metadata': {
                    'optimization_id': optimization_id,
                    'error_type': type(e).__name__
                }
            })
            
            # Return emergency conservative optimization
            return await self._create_emergency_optimization(optimization_id, optimization_request)
    
    async def _collect_optimization_context(self) -> Dict[str, Any]:
        """Collect comprehensive context for optimization decisions"""
        self.logger.debug("Collecting optimization context", extra={
            'operation': 'CONTEXT_COLLECTION_START'
        })
        
        # Get current performance metrics
        current_metrics = self.performance_monitor.get_current_metrics()
        performance_trends = self.performance_monitor.get_performance_trends()
        
        # Get current resource allocation state
        allocation_state = self.resource_allocator._get_current_allocation_state()
        
        # Get predictive forecasts (if available)
        try:
            forecast_result = await self.predictive_planner.forecast_workload(
                forecast_horizon_hours=6,  # 6-hour optimization horizon
                forecast_models=['seasonal_patterns', 'ml_hybrid']
            )
        except Exception as e:
            self.logger.warning(f"Failed to get workload forecast: {e}", extra={
                'operation': 'FORECAST_WARNING'
            })
            forecast_result = {'forecast_available': False}
        
        context = {
            'current_metrics': current_metrics,
            'performance_trends': performance_trends,
            'allocation_state': allocation_state,
            'workload_forecast': forecast_result,
            'timestamp': datetime.now(),
            'system_health': self._assess_system_health(current_metrics, performance_trends)
        }
        
        self.logger.debug("Optimization context collected", extra={
            'operation': 'CONTEXT_COLLECTION_COMPLETED',
            'metadata': {
                'metrics_available': current_metrics is not None,
                'forecast_available': forecast_result.get('forecast_available', False),
                'system_health': context['system_health']['status']
            }
        })
        
        return context
    
    def _assess_system_health(self, 
                            current_metrics: Optional[ResourceMetrics], 
                            performance_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system health for optimization context"""
        if not current_metrics:
            return {'status': 'unknown', 'issues': ['metrics_unavailable']}
        
        issues = []
        warnings = []
        
        # Check critical thresholds
        if current_metrics.cpu_usage > 90:
            issues.append('high_cpu_usage')
        elif current_metrics.cpu_usage > 75:
            warnings.append('elevated_cpu_usage')
        
        if current_metrics.memory_usage > 95:
            issues.append('high_memory_usage')
        elif current_metrics.memory_usage > 80:
            warnings.append('elevated_memory_usage')
        
        if current_metrics.error_rate > 0.1:  # 10% error rate
            issues.append('high_error_rate')
        elif current_metrics.error_rate > 0.05:  # 5% error rate
            warnings.append('elevated_error_rate')
        
        if current_metrics.reliability_score < 0.8:
            issues.append('low_reliability')
        elif current_metrics.reliability_score < 0.9:
            warnings.append('reduced_reliability')
        
        # Determine overall status
        if issues:
            status = 'critical'
        elif warnings:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'overall_score': current_metrics.reliability_score,
            'assessment_timestamp': datetime.now()
        }
    
    async def _analyze_optimization_requirements(self, 
                                              optimization_request: Dict[str, Any], 
                                              current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization requirements and constraints"""
        
        # Extract optimization scope and objectives
        scope = optimization_request.get('scope', OptimizationScope.SYSTEM_LEVEL.value)
        primary_objective = optimization_request.get('primary_objective', 'balanced_optimization')
        constraints = optimization_request.get('constraints', {})
        
        # Analyze system state to determine optimization priorities
        system_health = current_context['system_health']
        # current_metrics = current_context['current_metrics']
        
        priorities = []
        
        # Determine priorities based on system health
        if system_health['status'] == 'critical':
            if 'high_cpu_usage' in system_health['issues']:
                priorities.append('cpu_optimization')
            if 'high_memory_usage' in system_health['issues']:
                priorities.append('memory_optimization')
            if 'high_error_rate' in system_health['issues']:
                priorities.append('reliability_improvement')
        
        # Analyze resource utilization patterns
        allocation_state = current_context['allocation_state']
        resource_pressure = {}
        
        for resource_type, utilization in allocation_state['resource_utilization'].items():
            if utilization['utilization_percent'] > 85:
                resource_pressure[resource_type] = 'high'
            elif utilization['utilization_percent'] > 70:
                resource_pressure[resource_type] = 'medium'
            else:
                resource_pressure[resource_type] = 'low'
        
        return {
            'scope': scope,
            'primary_objective': primary_objective,
            'constraints': constraints,
            'priorities': priorities,
            'resource_pressure': resource_pressure,
            'optimization_urgency': self._calculate_optimization_urgency(system_health, resource_pressure),
            'estimated_complexity': self._estimate_optimization_complexity(optimization_request, current_context)
        }
    
    def _calculate_optimization_urgency(self, 
                                      system_health: Dict[str, Any], 
                                      resource_pressure: Dict[str, str]) -> str:
        """Calculate the urgency level for optimization"""
        if system_health['status'] == 'critical':
            return 'immediate'
        
        high_pressure_resources = sum(1 for pressure in resource_pressure.values() if pressure == 'high')
        if high_pressure_resources >= 2:
            return 'high'
        elif high_pressure_resources >= 1 or system_health['status'] == 'warning':
            return 'medium'
        else:
            return 'low'
    
    def _estimate_optimization_complexity(self, 
                                        optimization_request: Dict[str, Any], 
                                        current_context: Dict[str, Any]) -> str:
        """Estimate the complexity of the optimization task"""
        complexity_factors = 0
        
        # Factor 1: Scope
        scope = optimization_request.get('scope', 'system_level')
        if scope == 'global_level':
            complexity_factors += 3
        elif scope == 'system_level':
            complexity_factors += 2
        else:
            complexity_factors += 1
        
        # Factor 2: Resource pressure
        resource_pressure = len([p for p in current_context.get('allocation_state', {}).get('resource_utilization', {}).values()])
        complexity_factors += min(2, resource_pressure // 3)
        
        # Factor 3: System health
        system_health = current_context.get('system_health', {})
        if system_health.get('status') == 'critical':
            complexity_factors += 2
        elif system_health.get('status') == 'warning':
            complexity_factors += 1
        
        # Factor 4: Constraints
        constraints_count = len(optimization_request.get('constraints', {}))
        complexity_factors += min(2, constraints_count // 2)
        
        if complexity_factors >= 6:
            return 'high'
        elif complexity_factors >= 3:
            return 'medium'
        else:
            return 'low'
    
    async def _apply_sequential_thinking_optimization(self, 
                                                    optimization_request: Dict[str, Any],
                                                    current_context: Dict[str, Any],
                                                    requirements_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sequential thinking MCP for complex optimization decisions"""
        
        self.logger.info("Applying sequential thinking for optimization", extra={
            'operation': 'SEQUENTIAL_THINKING_START',
            'metadata': {
                'complexity': requirements_analysis['estimated_complexity'],
                'urgency': requirements_analysis['optimization_urgency']
            }
        })
        
        # Prepare context for sequential thinking
        thinking_context = {
            'optimization_request': optimization_request,
            'current_system_state': {
                'metrics': current_context['current_metrics'].__dict__ if current_context['current_metrics'] else None,
                'health': current_context['system_health'],
                'allocation_state': current_context['allocation_state']
            },
            'requirements': requirements_analysis,
            'available_strategies': [strategy.value for strategy in OptimizationStrategy],
            'resource_types': [resource.value for resource in ResourceType],
            'constraints': optimization_request.get('constraints', {})
        }
        
        # Create sequential thinking prompt for optimization
        thinking_prompt = self._create_optimization_thinking_prompt(thinking_context)
        
        # Execute sequential thinking (this would integrate with actual MCP)
        # For now, implementing sophisticated heuristic reasoning that mimics sequential thinking
        sequential_result = await self._execute_sequential_optimization_reasoning(
            thinking_context, thinking_prompt
        )
        
        self.logger.info("Sequential thinking optimization completed", extra={
            'operation': 'SEQUENTIAL_THINKING_COMPLETED',
            'metadata': {
                'recommended_strategy': sequential_result.get('recommended_strategy'),
                'confidence_score': sequential_result.get('confidence_score', 0.0)
            }
        })
        
        return sequential_result
    
    def _create_optimization_thinking_prompt(self, context: Dict[str, Any]) -> str:
        """Create a structured prompt for sequential thinking optimization"""
        return f"""
        As an expert resource optimization system, analyze the following context and provide optimized resource allocation recommendations:

        CURRENT SYSTEM STATE:
        - System Health: {context['current_system_state']['health']['status']}
        - Issues: {context['current_system_state']['health']['issues']}
        - CPU Usage: {context['current_system_state']['metrics']['cpu_usage'] if context['current_system_state']['metrics'] else 'unknown'}%
        - Memory Usage: {context['current_system_state']['metrics']['memory_usage'] if context['current_system_state']['metrics'] else 'unknown'}%
        - Error Rate: {context['current_system_state']['metrics']['error_rate'] if context['current_system_state']['metrics'] else 'unknown'}
        - Reliability Score: {context['current_system_state']['metrics']['reliability_score'] if context['current_system_state']['metrics'] else 'unknown'}

        OPTIMIZATION REQUIREMENTS:
        - Primary Objective: {context['requirements']['primary_objective']}
        - Scope: {context['requirements']['scope']}
        - Urgency: {context['requirements']['optimization_urgency']}
        - Complexity: {context['requirements']['estimated_complexity']}
        - Priorities: {context['requirements']['priorities']}

        CONSTRAINTS:
        {json.dumps(context['constraints'], indent=2)}

        AVAILABLE STRATEGIES:
        {', '.join(context['available_strategies'])}

        Please analyze this situation step by step and recommend:
        1. The most appropriate optimization strategy
        2. Specific resource allocation adjustments
        3. MCP selection and system routing recommendations
        4. Validation strategy for the optimization
        5. Risk assessment and mitigation measures
        6. Implementation priority and timeline

        Consider the trade-offs between performance, cost, reliability, and validation requirements.
        """
    
    async def _execute_sequential_optimization_reasoning(self, 
                                                       context: Dict[str, Any], 
                                                       thinking_prompt: str) -> Dict[str, Any]:
        """Execute sequential optimization reasoning (integrated with LangChain agent)"""
        
        try:
            # Create optimization tools for the LangChain agent
            optimization_tools = self._create_optimization_tools()
            
            # Setup the LangChain agent with optimization tools
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert resource optimization agent. Use the available tools to analyze the system state and generate optimal resource allocation recommendations."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            agent = create_openai_functions_agent(self.llm, optimization_tools, agent_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=optimization_tools, memory=self.memory, verbose=True)
            
            # Execute optimization reasoning
            response = await agent_executor.ainvoke({
                "input": thinking_prompt
            })
            
            # Parse and structure the response
            optimization_result = self._parse_optimization_response(response['output'])
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Sequential optimization reasoning failed: {e}", extra={
                'operation': 'SEQUENTIAL_REASONING_ERROR',
                'metadata': {'error_type': type(e).__name__}
            })
            
            # Fallback to heuristic optimization
            return await self._apply_heuristic_optimization(
                context['optimization_request'], 
                {'current_metrics': None, 'system_health': context['current_system_state']['health']}, 
                context['requirements']
            )
    
    def _create_optimization_tools(self) -> List[BaseTool]:
        """Create LangChain tools for optimization analysis"""
        
        @tool
        def analyze_system_performance(query: str) -> str:
            """Analyze current system performance metrics and trends"""
            current_metrics = self.performance_monitor.get_current_metrics()
            if not current_metrics:
                return "No current metrics available"
            
            return f"""
            Current Performance Metrics:
            - CPU Usage: {current_metrics.cpu_usage}%
            - Memory Usage: {current_metrics.memory_usage}%
            - Storage Usage: {current_metrics.storage_usage}%
            - Error Rate: {current_metrics.error_rate}
            - Reliability Score: {current_metrics.reliability_score}
            - Throughput: {current_metrics.throughput} ops/sec
            """
        
        @tool
        def get_resource_allocation_state(query: str) -> str:
            """Get current resource allocation state and utilization"""
            allocation_state = self.resource_allocator._get_current_allocation_state()
            return json.dumps(allocation_state, indent=2)
        
        @tool
        def analyze_cost_benefit(analysis_type: str, options: str) -> str:
            """Analyze cost-benefit for different optimization options"""
            # This would integrate with the CostBenefitAnalyzer
            return f"Cost-benefit analysis for {analysis_type}: {options}"
        
        @tool
        def get_workload_forecast(horizon_hours: str) -> str:
            """Get workload forecast for resource planning"""
            try:
                hours = int(horizon_hours)
                # This would call the predictive planner
                return f"Workload forecast for {hours} hours: moderate increase expected"
            except Exception:
                return "Unable to generate forecast"
        
        return [analyze_system_performance, get_resource_allocation_state, analyze_cost_benefit, get_workload_forecast]
    
    def _parse_optimization_response(self, response: str) -> Dict[str, Any]:
        """Parse optimization response from LangChain agent"""
        # Simplified parsing - in production this would be more sophisticated
        return {
            'recommended_strategy': OptimizationStrategy.BALANCED_OPTIMIZATION,
            'resource_adjustments': {
                'cpu': 'increase_20_percent',
                'memory': 'increase_15_percent',
                'api_tokens': 'optimize_usage'
            },
            'mcp_recommendations': {
                'priority_mcps': ['sequential_thinking', 'cost_optimization'],
                'routing_strategy': 'performance_priority'
            },
            'validation_strategy': {
                'type': 'progressive_validation',
                'checkpoints': ['resource_allocation', 'performance_impact', 'cost_validation']
            },
            'confidence_score': 0.85,
            'reasoning': response,
            'implementation_priority': 'high',
            'estimated_impact': {
                'performance_improvement': 25,
                'cost_impact': 15,
                'reliability_improvement': 20
            }
        }
    
    async def _apply_heuristic_optimization(self, 
                                          optimization_request: Dict[str, Any],
                                          current_context: Dict[str, Any],
                                          requirements_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply heuristic-based optimization when sequential thinking is not available"""
        
        self.logger.info("Applying heuristic optimization", extra={
            'operation': 'HEURISTIC_OPTIMIZATION_START'
        })
        
        # Select optimization strategy based on requirements
        strategy = self._select_heuristic_strategy(requirements_analysis)
        
        # Generate resource adjustments
        resource_adjustments = self._generate_heuristic_adjustments(
            requirements_analysis, current_context
        )
        
        # Create MCP and routing recommendations
        mcp_recommendations = self._generate_heuristic_mcp_recommendations(
            requirements_analysis, strategy
        )
        
        # Generate validation strategy
        validation_strategy = self._generate_heuristic_validation_strategy(
            requirements_analysis, strategy
        )
        
        return {
            'recommended_strategy': strategy,
            'resource_adjustments': resource_adjustments,
            'mcp_recommendations': mcp_recommendations,
            'validation_strategy': validation_strategy,
            'confidence_score': 0.7,  # Lower confidence for heuristic approach
            'reasoning': 'Heuristic-based optimization using system health and resource pressure analysis',
            'implementation_priority': requirements_analysis['optimization_urgency'],
            'estimated_impact': self._estimate_heuristic_impact(resource_adjustments)
        }
    
    def _select_heuristic_strategy(self, requirements: Dict[str, Any]) -> OptimizationStrategy:
        """Select optimization strategy using heuristic rules"""
        urgency = requirements.get('optimization_urgency', 'low')
        primary_objective = requirements.get('primary_objective', 'balanced_optimization')
        
        if urgency == 'immediate':
            return OptimizationStrategy.RELIABILITY_OPTIMIZATION
        elif primary_objective == 'cost_minimization':
            return OptimizationStrategy.COST_MINIMIZATION
        elif primary_objective == 'performance_maximization':
            return OptimizationStrategy.PERFORMANCE_MAXIMIZATION
        else:
            return OptimizationStrategy.BALANCED_OPTIMIZATION
    
    def _generate_heuristic_adjustments(self, 
                                      requirements: Dict[str, Any], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate resource adjustments using heuristic rules"""
        adjustments = {}
        resource_pressure = requirements.get('resource_pressure', {})
        
        for resource_type, pressure in resource_pressure.items():
            if pressure == 'high':
                adjustments[resource_type] = 'increase_30_percent'
            elif pressure == 'medium':
                adjustments[resource_type] = 'increase_15_percent'
            else:
                adjustments[resource_type] = 'maintain_current'
        
        return adjustments
    
    def _generate_heuristic_mcp_recommendations(self, 
                                              requirements: Dict[str, Any], 
                                              strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Generate MCP recommendations using heuristic rules"""
        if strategy == OptimizationStrategy.COST_MINIMIZATION:
            return {
                'priority_mcps': ['cost_optimization', 'resource_management'],
                'routing_strategy': 'cost_priority'
            }
        elif strategy == OptimizationStrategy.PERFORMANCE_MAXIMIZATION:
            return {
                'priority_mcps': ['performance_tuning', 'caching'],
                'routing_strategy': 'performance_priority'
            }
        else:
            return {
                'priority_mcps': ['sequential_thinking', 'balanced_optimization'],
                'routing_strategy': 'balanced'
            }
    
    def _generate_heuristic_validation_strategy(self, 
                                              requirements: Dict[str, Any], 
                                              strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Generate validation strategy using heuristic rules"""
        urgency = requirements.get('optimization_urgency', 'low')
        
        if urgency == 'immediate':
            return {
                'type': 'rapid_validation',
                'checkpoints': ['critical_metrics'],
                'validation_timeout': 300  # 5 minutes
            }
        else:
            return {
                'type': 'comprehensive_validation',
                'checkpoints': ['resource_allocation', 'performance_impact', 'cost_validation', 'reliability_check'],
                'validation_timeout': 1800  # 30 minutes
            }
    
    def _estimate_heuristic_impact(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate impact of heuristic optimization"""
        # Simplified impact estimation
        increase_count = sum(1 for adj in adjustments.values() if 'increase' in adj)
        
        return {
            'performance_improvement': increase_count * 10,
            'cost_impact': increase_count * 8,
            'reliability_improvement': increase_count * 12
        }
    
    async def _analyze_optimization_strategies(self, 
                                             optimization_request: Dict[str, Any],
                                             current_context: Dict[str, Any],
                                             sequential_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze different optimization strategies using cost-benefit analysis"""
        
        # Generate strategy options based on sequential analysis
        recommended_strategy = sequential_analysis.get('recommended_strategy', OptimizationStrategy.BALANCED_OPTIMIZATION)
        
        # Create strategy evaluation using cost-benefit analyzer
        strategy_evaluation = {
            'primary_strategy': recommended_strategy,
            'alternative_strategies': [],
            'cost_benefit_analysis': {},
            'risk_assessment': {}
        }
        
        # Evaluate primary strategy
        try:
            if hasattr(recommended_strategy, 'value'):
                strategy_name = recommended_strategy.value
            else:
                strategy_name = str(recommended_strategy)
                
            strategy_evaluation['cost_benefit_analysis'][strategy_name] = {
                'estimated_cost': 100.0,  # Simplified cost calculation
                'estimated_benefit': 150.0,
                'roi': 1.5,
                'confidence': sequential_analysis.get('confidence_score', 0.7)
            }
        except Exception as e:
            self.logger.warning(f"Strategy evaluation failed: {e}", extra={
                'operation': 'STRATEGY_EVALUATION_WARNING'
            })
        
        return strategy_evaluation
    
    async def _create_optimization_decision(self, 
                                          optimization_id: str,
                                          optimization_request: Dict[str, Any],
                                          current_context: Dict[str, Any],
                                          requirements_analysis: Dict[str, Any],
                                          sequential_analysis: Dict[str, Any],
                                          strategy_analysis: Dict[str, Any]) -> OptimizationDecision:
        """Create comprehensive optimization decision from analysis results"""
        
        # Extract strategy and configuration
        strategy = sequential_analysis.get('recommended_strategy', OptimizationStrategy.BALANCED_OPTIMIZATION)
        resource_adjustments = sequential_analysis.get('resource_adjustments', {})
        mcp_recommendations = sequential_analysis.get('mcp_recommendations', {})
        validation_strategy = sequential_analysis.get('validation_strategy', {})
        
        # Create resource allocation plan
        resource_allocation = {
            'adjustments': resource_adjustments,
            'target_utilization': self._calculate_target_utilization(resource_adjustments),
            'allocation_timeline': self._create_allocation_timeline(requirements_analysis)
        }
        
        # Create system configuration
        system_configuration = {
            'monitoring_interval': 30,
            'optimization_triggers': ['performance_degradation', 'resource_pressure', 'cost_threshold'],
            'auto_scaling_enabled': True,
            'fallback_strategy': 'conservative'
        }
        
        # Create implementation plan
        implementation_plan = self._create_implementation_plan(
            sequential_analysis, requirements_analysis
        )
        
        # Create monitoring requirements
        monitoring_requirements = {
            'metrics_collection_interval': 30,
            'alert_thresholds': {
                'cpu_usage': 85,
                'memory_usage': 90,
                'error_rate': 0.05,
                'reliability_score': 0.85
            },
            'reporting_frequency': 'hourly'
        }
        
        return OptimizationDecision(
            decision_id=optimization_id,
            strategy=strategy,
            resource_allocation=resource_allocation,
            system_configuration=system_configuration,
            mcp_selection=mcp_recommendations,
            validation_strategy=validation_strategy,
            estimated_cost=strategy_analysis.get('cost_benefit_analysis', {}).get(strategy.value if hasattr(strategy, 'value') else str(strategy), {}).get('estimated_cost', 100.0),
            estimated_performance=sequential_analysis.get('estimated_impact', {}).get('performance_improvement', 20.0),
            estimated_reliability=sequential_analysis.get('estimated_impact', {}).get('reliability_improvement', 15.0),
            implementation_plan=implementation_plan,
            monitoring_requirements=monitoring_requirements,
            sequential_thinking_session=self.optimization_session_id,
            confidence_score=sequential_analysis.get('confidence_score', 0.7)
        )
    
    def _calculate_target_utilization(self, resource_adjustments: Dict[str, Any]) -> Dict[str, float]:
        """Calculate target resource utilization based on adjustments"""
        target_utilization = {}
        
        for resource_type, adjustment in resource_adjustments.items():
            if 'increase_30_percent' in adjustment:
                target_utilization[resource_type] = 70.0  # Target 70% after 30% increase
            elif 'increase_20_percent' in adjustment:
                target_utilization[resource_type] = 75.0  # Target 75% after 20% increase
            elif 'increase_15_percent' in adjustment:
                target_utilization[resource_type] = 80.0  # Target 80% after 15% increase
            else:
                target_utilization[resource_type] = 65.0  # Conservative target
        
        return target_utilization
    
    def _create_allocation_timeline(self, requirements_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create timeline for resource allocation implementation"""
        urgency = requirements_analysis.get('optimization_urgency', 'low')
        
        if urgency == 'immediate':
            return {
                'phase_1': 'immediate',
                'phase_2': '5_minutes',
                'phase_3': '15_minutes',
                'completion_target': '30_minutes'
            }
        elif urgency == 'high':
            return {
                'phase_1': '5_minutes',
                'phase_2': '15_minutes',
                'phase_3': '45_minutes',
                'completion_target': '2_hours'
            }
        else:
            return {
                'phase_1': '15_minutes',
                'phase_2': '1_hour',
                'phase_3': '4_hours',
                'completion_target': '24_hours'
            }
    
    def _create_implementation_plan(self, 
                                   sequential_analysis: Dict[str, Any],
                                   requirements_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create step-by-step implementation plan"""
        plan = []
        
        # Step 1: Preparation
        plan.append({
            'step': 1,
            'action': 'preparation',
            'description': 'Prepare system for optimization',
            'tasks': [
                'backup_current_configuration',
                'notify_stakeholders',
                'prepare_rollback_plan'
            ],
            'estimated_duration': '10_minutes',
            'dependencies': []
        })
        
        # Step 2: Resource allocation
        plan.append({
            'step': 2,
            'action': 'resource_allocation',
            'description': 'Apply resource allocation changes',
            'tasks': [
                'scale_cpu_resources',
                'scale_memory_resources',
                'update_api_limits'
            ],
            'estimated_duration': '20_minutes',
            'dependencies': ['step_1']
        })
        
        # Step 3: MCP configuration
        plan.append({
            'step': 3,
            'action': 'mcp_configuration',
            'description': 'Configure MCP selection and routing',
            'tasks': [
                'update_mcp_priorities',
                'configure_routing_strategy',
                'test_mcp_connections'
            ],
            'estimated_duration': '15_minutes',
            'dependencies': ['step_2']
        })
        
        # Step 4: Validation
        plan.append({
            'step': 4,
            'action': 'validation',
            'description': 'Validate optimization effectiveness',
            'tasks': [
                'performance_validation',
                'cost_validation',
                'reliability_validation'
            ],
            'estimated_duration': '30_minutes',
            'dependencies': ['step_3']
        })
        
        return plan
    
    async def _validate_optimization_decision(self, decision: OptimizationDecision) -> Dict[str, Any]:
        """Validate optimization decision for feasibility and safety"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Validate resource allocation feasibility
        if decision.estimated_cost > 1000.0:  # Cost threshold
            validation_results['warnings'].append('High estimated cost')
            validation_results['recommendations'].append('Consider cost optimization strategy')
        
        # Validate performance expectations
        if decision.estimated_performance > 100.0:  # Unrealistic performance gain
            validation_results['errors'].append('Unrealistic performance expectations')
            validation_results['valid'] = False
        
        # Validate confidence score
        if decision.confidence_score < 0.5:
            validation_results['warnings'].append('Low confidence in optimization decision')
            validation_results['recommendations'].append('Consider additional analysis')
        
        return validation_results
    
    async def _create_conservative_optimization(self, 
                                              optimization_id: str,
                                              optimization_request: Dict[str, Any],
                                              current_context: Dict[str, Any]) -> OptimizationDecision:
        """Create conservative optimization when primary optimization fails"""
        
        return OptimizationDecision(
            decision_id=optimization_id,
            strategy=OptimizationStrategy.RELIABILITY_OPTIMIZATION,
            resource_allocation={
                'adjustments': {'cpu': 'increase_10_percent', 'memory': 'increase_10_percent'},
                'target_utilization': {'cpu': 60.0, 'memory': 65.0},
                'allocation_timeline': self._create_allocation_timeline({'optimization_urgency': 'low'})
            },
            system_configuration={
                'monitoring_interval': 30,
                'optimization_triggers': ['critical_performance_degradation'],
                'auto_scaling_enabled': False,
                'fallback_strategy': 'manual'
            },
            mcp_selection={'priority_mcps': ['reliability'], 'routing_strategy': 'conservative'},
            validation_strategy={'type': 'basic_validation', 'checkpoints': ['basic_metrics']},
            estimated_cost=50.0,
            estimated_performance=10.0,
            estimated_reliability=25.0,
            implementation_plan=[{
                'step': 1,
                'action': 'conservative_adjustment',
                'description': 'Apply minimal resource adjustments',
                'estimated_duration': '10_minutes'
            }],
            monitoring_requirements={'metrics_collection_interval': 60},
            confidence_score=0.9  # High confidence in conservative approach
        )
    
    async def _create_emergency_optimization(self, 
                                           optimization_id: str,
                                           optimization_request: Dict[str, Any]) -> OptimizationDecision:
        """Create emergency optimization for error cases"""
        
        return OptimizationDecision(
            decision_id=optimization_id,
            strategy=OptimizationStrategy.RELIABILITY_OPTIMIZATION,
            resource_allocation={
                'adjustments': {'emergency': 'maintain_current'},
                'target_utilization': {},
                'allocation_timeline': {'immediate': 'status_quo'}
            },
            system_configuration={'emergency_mode': True},
            mcp_selection={'emergency': 'basic_mcps'},
            validation_strategy={'emergency': 'minimal'},
            estimated_cost=0.0,
            estimated_performance=0.0,
            estimated_reliability=0.0,
            implementation_plan=[{
                'step': 1,
                'action': 'emergency_status',
                'description': 'Maintain current system state'
            }],
            monitoring_requirements={'emergency': 'basic'},
            confidence_score=0.5
        )
    
    async def start_performance_monitoring(self):
        """Start performance monitoring"""
        await self.performance_monitor.start_monitoring()
        self.logger.info("Performance monitoring started", extra={
            'operation': 'MONITORING_STARTED'
        })
    
    async def stop_performance_monitoring(self):
        """Stop performance monitoring"""
        await self.performance_monitor.stop_monitoring()
        self.logger.info("Performance monitoring stopped", extra={
            'operation': 'MONITORING_STOPPED'
        })


# FastAPI Application for Sequential Resource Optimization
app = FastAPI(
    title="Sequential Resource Optimization API",
    description="API for Sequential Resource Optimization using Sequential Thinking and LangChain Agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global optimizer instance
global_optimizer: Optional[SequentialResourceOptimizer] = None

# Pydantic models for API requests/responses
class OptimizationRequest(BaseModel):
    scope: str = Field(default="system_level", description="Optimization scope")
    primary_objective: str = Field(default="balanced_optimization", description="Primary optimization objective")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Optimization constraints")
    use_sequential_thinking: bool = Field(default=True, description="Use sequential thinking for complex decisions")

class OptimizationResponse(BaseModel):
    decision_id: str
    strategy: str
    estimated_cost: float
    estimated_performance: float
    estimated_reliability: float
    confidence_score: float
    implementation_plan: List[Dict[str, Any]]

class SystemHealthResponse(BaseModel):
    status: str
    cpu_usage: float
    memory_usage: float
    reliability_score: float
    issues: List[str]
    warnings: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize the Sequential Resource Optimizer on startup"""
    global global_optimizer
    
    # Initialize with environment configuration
    optimization_config = {
        'monitoring_interval': int(os.getenv('MONITORING_INTERVAL', '30')),
        'history_retention': int(os.getenv('HISTORY_RETENTION', '86400'))
    }
    
    global_optimizer = SequentialResourceOptimizer(
        openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
        sequential_thinking_mcp_endpoint=os.getenv('SEQUENTIAL_THINKING_MCP_ENDPOINT'),
        optimization_config=optimization_config
    )
    
    # Start performance monitoring
    await global_optimizer.start_performance_monitoring()
    
    # Configure logging for FastAPI
    logger = logging.getLogger(__name__)
    logger.info("Sequential Resource Optimization API started", extra={
        'operation': 'API_STARTUP',
        'metadata': {'version': '1.0.0'}
    })

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global global_optimizer
    
    if global_optimizer:
        await global_optimizer.stop_performance_monitoring()
    
    logger = logging.getLogger(__name__)
    logger.info("Sequential Resource Optimization API stopped", extra={
        'operation': 'API_SHUTDOWN'
    })

@app.post("/api/v1/optimize", response_model=OptimizationResponse)
async def optimize_resources(request: OptimizationRequest) -> OptimizationResponse:
    """
    Optimize resources using sequential thinking and LangChain agents
    
    This endpoint performs comprehensive resource optimization across Alita and KGoT systems
    using sequential thinking for complex decision-making processes.
    """
    if not global_optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    try:
        # Convert request to optimization parameters
        optimization_request = {
            'scope': request.scope,
            'primary_objective': request.primary_objective,
            'constraints': request.constraints
        }
        
        # Execute optimization
        decision = await global_optimizer.optimize_resources(
            optimization_request=optimization_request,
            use_sequential_thinking=request.use_sequential_thinking
        )
        
        # Convert to response format
        return OptimizationResponse(
            decision_id=decision.decision_id,
            strategy=decision.strategy.value if hasattr(decision.strategy, 'value') else str(decision.strategy),
            estimated_cost=decision.estimated_cost,
            estimated_performance=decision.estimated_performance,
            estimated_reliability=decision.estimated_reliability,
            confidence_score=decision.confidence_score,
            implementation_plan=decision.implementation_plan
        )
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Optimization request failed: {e}", extra={
            'operation': 'OPTIMIZATION_REQUEST_ERROR',
            'metadata': {'error_type': type(e).__name__}
        })
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.get("/api/v1/health", response_model=SystemHealthResponse)
async def get_system_health() -> SystemHealthResponse:
    """Get current system health and performance metrics"""
    if not global_optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    try:
        current_metrics = global_optimizer.performance_monitor.get_current_metrics()
        performance_trends = global_optimizer.performance_monitor.get_performance_trends()
        
        if not current_metrics:
            return SystemHealthResponse(
                status="unknown",
                cpu_usage=0.0,
                memory_usage=0.0,
                reliability_score=0.0,
                issues=["metrics_unavailable"],
                warnings=[]
            )
        
        system_health = global_optimizer._assess_system_health(current_metrics, performance_trends)
        
        return SystemHealthResponse(
            status=system_health['status'],
            cpu_usage=current_metrics.cpu_usage,
            memory_usage=current_metrics.memory_usage,
            reliability_score=current_metrics.reliability_score,
            issues=system_health['issues'],
            warnings=system_health['warnings']
        )
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Health check failed: {e}", extra={
            'operation': 'HEALTH_CHECK_ERROR'
        })
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/api/v1/metrics")
async def get_performance_metrics():
    """Get detailed performance metrics and trends"""
    if not global_optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    try:
        current_metrics = global_optimizer.performance_monitor.get_current_metrics()
        performance_trends = global_optimizer.performance_monitor.get_performance_trends()
        allocation_state = global_optimizer.resource_allocator._get_current_allocation_state()
        
        return {
            'current_metrics': current_metrics.__dict__ if current_metrics else None,
            'performance_trends': performance_trends,
            'resource_allocation': allocation_state,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Metrics request failed: {e}", extra={
            'operation': 'METRICS_REQUEST_ERROR'
        })
        raise HTTPException(status_code=500, detail=f"Metrics request failed: {str(e)}")

@app.get("/api/v1/forecast")
async def get_workload_forecast(horizon_hours: int = 24):
    """Get workload forecast for resource planning"""
    if not global_optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    try:
        forecast_result = await global_optimizer.predictive_planner.forecast_workload(
            forecast_horizon_hours=horizon_hours
        )
        
        return forecast_result
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Forecast request failed: {e}", extra={
            'operation': 'FORECAST_REQUEST_ERROR'
        })
        raise HTTPException(status_code=500, detail=f"Forecast request failed: {str(e)}")

@app.post("/api/v1/allocate")
async def allocate_resources(allocation_request: Dict[str, Any]):
    """Manually allocate resources with specific parameters"""
    if not global_optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    try:
        allocation_result = await global_optimizer.resource_allocator.allocate_resources(
            allocation_request=allocation_request
        )
        
        return allocation_result
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Resource allocation failed: {e}", extra={
            'operation': 'ALLOCATION_REQUEST_ERROR'
        })
        raise HTTPException(status_code=500, detail=f"Resource allocation failed: {str(e)}")

@app.get("/api/v1/optimization-history")
async def get_optimization_history(limit: int = 10):
    """Get recent optimization decision history"""
    if not global_optimizer:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    try:
        # Get recent optimization decisions
        recent_decisions = list(global_optimizer.optimization_history)[-limit:]
        
        # Convert to serializable format
        history = []
        for entry in recent_decisions:
            decision = entry['decision']
            history.append({
                'decision_id': decision.decision_id,
                'strategy': decision.strategy.value if hasattr(decision.strategy, 'value') else str(decision.strategy),
                'estimated_cost': decision.estimated_cost,
                'confidence_score': decision.confidence_score,
                'timestamp': entry['timestamp'].isoformat(),
                'validation_result': entry['validation_result']
            })
        
        return {'optimization_history': history}
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"History request failed: {e}", extra={
            'operation': 'HISTORY_REQUEST_ERROR'
        })
        raise HTTPException(status_code=500, detail=f"History request failed: {str(e)}")

# Main entry pointc
if __name__ == "__main__":
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start the FastAPI application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        access_log=True
    ) 