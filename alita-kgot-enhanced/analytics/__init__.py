"""
Analytics Package for Enhanced Alita KGoT System

This package provides comprehensive performance analytics and monitoring
for MCP (Model Context Protocol) components, implementing advanced
metrics collection, predictive analytics, and dynamic optimization.

Modules:
- mcp_performance_analytics: Main analytics engine with predictive capabilities
"""

from .mcp_performance_analytics import (
    MCPPerformanceAnalytics,
    UsagePatternTracker,
    PredictiveAnalyticsEngine,
    EffectivenessScorer,
    DynamicParetoOptimizer,
    ExperimentalMetricsCollector,
    PerformanceAnalyticsResult,
    UsagePattern,
    EffectivenessScore,
    ParetoOptimizationResult
)

__version__ = "1.0.0"
__author__ = "Enhanced Alita KGoT Team"

__all__ = [
    "MCPPerformanceAnalytics",
    "UsagePatternTracker", 
    "PredictiveAnalyticsEngine",
    "EffectivenessScorer",
    "DynamicParetoOptimizer",
    "ExperimentalMetricsCollector",
    "PerformanceAnalyticsResult",
    "UsagePattern",
    "EffectivenessScore",
    "ParetoOptimizationResult"
] 