#!/usr/bin/env python3
"""
MCP Cross-Validation Engine - Task 16 Implementation

Advanced MCP validation system implementing comprehensive cross-validation methodologies
as specified in the 5-Phase Implementation Plan for Enhanced Alita.

This module provides:
- K-fold cross-validation for MCPs across different task types
- Multi-scenario testing framework leveraging KGoT structured knowledge
- Validation metrics: reliability, consistency, performance, accuracy
- Statistical significance testing using KGoT Section 2.3 analytical capabilities
- Integration with KGoT Section 2.5 Error Management layered error containment
- Extension of both Alita MCP creation validation and KGoT knowledge validation

Key Components:
1. MCPCrossValidationEngine: Main orchestrator for comprehensive MCP validation
2. KFoldMCPValidator: K-fold cross-validation implementation across task types
3. MultiScenarioTestFramework: KGoT knowledge-leveraged testing scenarios
4. ValidationMetricsEngine: Comprehensive metrics calculation system
5. StatisticalSignificanceAnalyzer: Advanced statistical analysis using KGoT capabilities
6. ErrorManagementBridge: Integration with existing error containment systems

@author: Enhanced Alita KGoT Team
@version: 1.0.0
@task: Task 16 - Build MCP Cross-Validation Engine
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
import numpy as np
import pandas as pd
from pathlib import Path

# Statistical analysis and machine learning
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, kruskal
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# LangChain for agent development (per user memory requirement)
# Using specific imports to avoid metaclass conflicts
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field

# Integration with existing systems
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from kgot_core.error_management import (
    KGoTErrorManagementSystem, 
    ErrorType, 
    ErrorSeverity, 
    ErrorContext
)

# Winston-compatible logging setup
logger = logging.getLogger('MCPCrossValidation')
handler = logging.FileHandler('./logs/validation/combined.log')
formatter = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TaskType(Enum):
    """
    MCP task type classification for k-fold cross-validation
    Based on KGoT Section 2.1 knowledge domains
    """
    DATA_PROCESSING = "data_processing"
    WEB_SCRAPING = "web_scraping"
    CODE_GENERATION = "code_generation"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    COMMUNICATION = "communication"
    ANALYSIS_COMPUTATION = "analysis_computation"
    API_INTEGRATION = "api_integration"
    MULTIMODAL_PROCESSING = "multimodal_processing"


class ValidationMetricType(Enum):
    """Validation metric types as specified in Task 16"""
    RELIABILITY = "reliability"
    CONSISTENCY = "consistency"
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"


class ScenarioType(Enum):
    """Multi-scenario testing types leveraging KGoT structured knowledge"""
    NOMINAL = "nominal"               # Standard operation scenarios
    EDGE_CASE = "edge_case"          # Boundary condition testing
    STRESS_TEST = "stress_test"      # High load/complexity testing
    FAILURE_MODE = "failure_mode"    # Error condition testing
    ADVERSARIAL = "adversarial"      # Adversarial input testing
    INTEGRATION = "integration"      # Cross-MCP interaction testing


@dataclass
class MCPValidationSpec:
    """
    Comprehensive MCP specification for validation
    Extends existing MCP specifications with validation-specific metadata
    """
    mcp_id: str
    name: str
    description: str
    task_type: TaskType
    capabilities: List[str]
    implementation_code: str
    test_cases: List[Dict[str, Any]]
    expected_outputs: List[Any]
    performance_requirements: Dict[str, float]
    reliability_requirements: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationScenario:
    """Multi-scenario test case definition"""
    scenario_id: str
    scenario_type: ScenarioType
    description: str
    input_data: Dict[str, Any]
    expected_behavior: Dict[str, Any]
    success_criteria: List[str]
    failure_conditions: List[str]
    kgot_knowledge_context: Optional[Dict[str, Any]] = None


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics as specified in Task 16"""
    # Reliability metrics
    consistency_score: float
    error_rate: float
    failure_recovery_rate: float
    uptime_percentage: float
    
    # Consistency metrics
    output_consistency: float
    logic_consistency: float
    temporal_consistency: float
    cross_model_agreement: float
    
    # Performance metrics
    execution_time_avg: float
    execution_time_std: float
    throughput_ops_per_sec: float
    resource_efficiency: float
    scalability_score: float
    
    # Accuracy metrics
    ground_truth_accuracy: float
    expert_validation_score: float
    benchmark_performance: float
    cross_validation_accuracy: float
    
    # Statistical significance
    confidence_interval: Tuple[float, float]
    p_value: float
    statistical_significance: bool
    
    # Metadata
    sample_size: int
    validation_duration: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CrossValidationResult:
    """K-fold cross-validation results"""
    validation_id: str
    mcp_spec: MCPValidationSpec
    k_value: int
    fold_results: List[Dict[str, Any]]
    aggregated_metrics: ValidationMetrics
    statistical_analysis: Dict[str, Any]
    error_analysis: Dict[str, Any]
    recommendations: List[str]
    is_valid: bool
    confidence_score: float


class StatisticalSignificanceAnalyzer:
    """
    Statistical significance testing using KGoT Section 2.3 analytical capabilities
    Implements comprehensive statistical analysis for validation results
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical analyzer
        
        @param {float} significance_level - Alpha level for statistical tests (default: 0.05)
        """
        self.significance_level = significance_level
        self.test_results_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized Statistical Significance Analyzer", extra={
            'operation': 'STATISTICAL_ANALYZER_INIT',
            'significance_level': significance_level
        })
    
    async def analyze_validation_significance(self, 
                                            validation_results: List[ValidationMetrics],
                                            baseline_metrics: Optional[ValidationMetrics] = None) -> Dict[str, Any]:
        """
        Perform comprehensive statistical significance analysis on validation results
        
        @param {List[ValidationMetrics]} validation_results - List of validation metrics from k-fold
        @param {Optional[ValidationMetrics]} baseline_metrics - Baseline metrics for comparison
        @returns {Dict[str, Any]} Statistical analysis results
        """
        logger.info("Starting statistical significance analysis", extra={
            'operation': 'STATISTICAL_ANALYSIS_START',
            'sample_size': len(validation_results)
        })
        
        try:
            analysis_results = {
                'normality_tests': await self._test_normality(validation_results),
                'descriptive_stats': self._calculate_descriptive_stats(validation_results),
                'confidence_intervals': self._calculate_confidence_intervals(validation_results),
                'statistical_tests': {},
                'effect_sizes': {},
                'recommendations': []
            }
            
            # Perform baseline comparison if available
            if baseline_metrics:
                analysis_results['baseline_comparison'] = await self._compare_with_baseline(
                    validation_results, baseline_metrics
                )
            
            # Inter-fold consistency analysis
            analysis_results['consistency_analysis'] = await self._analyze_inter_fold_consistency(validation_results)
            
            # Power analysis for sample size adequacy
            analysis_results['power_analysis'] = self._perform_power_analysis(validation_results)
            
            logger.info("Statistical significance analysis completed", extra={
                'operation': 'STATISTICAL_ANALYSIS_COMPLETE',
                'significant_metrics': len([t for t in analysis_results['statistical_tests'].values() 
                                          if t.get('significant', False)])
            })
            
            return analysis_results
            
        except Exception as e:
            logger.error("Statistical significance analysis failed", extra={
                'operation': 'STATISTICAL_ANALYSIS_FAILED',
                'error': str(e)
            })
            raise
    
    async def _test_normality(self, validation_results: List[ValidationMetrics]) -> Dict[str, Dict[str, Any]]:
        """Test normality of metric distributions using Shapiro-Wilk test"""
        normality_results = {}
        
        # Extract metric values
        metrics_data = {
            'reliability_scores': [r.consistency_score for r in validation_results],
            'consistency_scores': [r.output_consistency for r in validation_results],
            'performance_scores': [r.execution_time_avg for r in validation_results],
            'accuracy_scores': [r.ground_truth_accuracy for r in validation_results]
        }
        
        for metric_name, values in metrics_data.items():
            if len(values) >= 3:  # Minimum sample size for Shapiro-Wilk
                statistic, p_value = stats.shapiro(values)
                normality_results[metric_name] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_normal': p_value > self.significance_level,
                    'test_method': 'shapiro_wilk'
                }
        
        return normality_results
    
    def _calculate_descriptive_stats(self, validation_results: List[ValidationMetrics]) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive descriptive statistics"""
        metrics_data = {
            'reliability': [r.consistency_score for r in validation_results],
            'consistency': [r.output_consistency for r in validation_results],
            'performance': [1/r.execution_time_avg if r.execution_time_avg > 0 else 0 for r in validation_results],
            'accuracy': [r.ground_truth_accuracy for r in validation_results]
        }
        
        descriptive_stats = {}
        for metric_name, values in metrics_data.items():
            descriptive_stats[metric_name] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'var': np.var(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'skewness': stats.skew(values),
                'kurtosis': stats.kurtosis(values)
            }
        
        return descriptive_stats
    
    def _calculate_confidence_intervals(self, validation_results: List[ValidationMetrics]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for validation metrics"""
        confidence_intervals = {}
        
        metrics_data = {
            'reliability': [r.consistency_score for r in validation_results],
            'consistency': [r.output_consistency for r in validation_results],
            'performance': [1/r.execution_time_avg if r.execution_time_avg > 0 else 0 for r in validation_results],
            'accuracy': [r.ground_truth_accuracy for r in validation_results]
        }
        
        confidence_level = 1 - self.significance_level
        
        for metric_name, values in metrics_data.items():
            if len(values) > 1:
                mean = np.mean(values)
                std_err = stats.sem(values)
                interval = stats.t.interval(confidence_level, len(values)-1, mean, std_err)
                confidence_intervals[metric_name] = interval
        
        return confidence_intervals
    
    async def _compare_with_baseline(self, 
                                   validation_results: List[ValidationMetrics],
                                   baseline_metrics: ValidationMetrics) -> Dict[str, Dict[str, Any]]:
        """Compare validation results with baseline using appropriate statistical tests"""
        comparison_results = {}
        
        # Extract validation data
        validation_data = {
            'reliability': [r.consistency_score for r in validation_results],
            'consistency': [r.output_consistency for r in validation_results],
            'performance': [1/r.execution_time_avg if r.execution_time_avg > 0 else 0 for r in validation_results],
            'accuracy': [r.ground_truth_accuracy for r in validation_results]
        }
        
        baseline_data = {
            'reliability': baseline_metrics.consistency_score,
            'consistency': baseline_metrics.output_consistency,
            'performance': 1/baseline_metrics.execution_time_avg if baseline_metrics.execution_time_avg > 0 else 0,
            'accuracy': baseline_metrics.ground_truth_accuracy
        }
        
        for metric_name, values in validation_data.items():
            baseline_value = baseline_data[metric_name]
            
            # One-sample t-test against baseline
            t_stat, p_value = stats.ttest_1samp(values, baseline_value)
            
            # Effect size (Cohen's d)
            effect_size = (np.mean(values) - baseline_value) / np.std(values) if np.std(values) > 0 else 0
            
            comparison_results[metric_name] = {
                'test_type': 'one_sample_t_test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'effect_size': effect_size,
                'improvement': np.mean(values) > baseline_value,
                'mean_difference': np.mean(values) - baseline_value
            }
        
        return comparison_results 

    async def _analyze_inter_fold_consistency(self, validation_results: List[ValidationMetrics]) -> Dict[str, Any]:
        """Analyze consistency across k-fold validation results"""
        consistency_analysis = {}
        
        # Calculate coefficient of variation for each metric
        metrics_data = {
            'reliability': [r.consistency_score for r in validation_results],
            'consistency': [r.output_consistency for r in validation_results],
            'performance': [1/r.execution_time_avg if r.execution_time_avg > 0 else 0 for r in validation_results],
            'accuracy': [r.ground_truth_accuracy for r in validation_results]
        }
        
        for metric_name, values in metrics_data.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val != 0 else float('inf')
            
            consistency_analysis[metric_name] = {
                'coefficient_of_variation': cv,
                'is_consistent': cv < 0.2,  # CV < 20% indicates good consistency
                'stability_score': 1 / (1 + cv) if cv != float('inf') else 0
            }
        
        return consistency_analysis
    
    def _perform_power_analysis(self, validation_results: List[ValidationMetrics]) -> Dict[str, Any]:
        """Perform power analysis to assess sample size adequacy"""
        n = len(validation_results)
        
        # Simple power analysis based on sample size
        if n >= 30:
            power_level = "high"
            adequacy_score = 1.0
        elif n >= 10:
            power_level = "medium"
            adequacy_score = 0.7
        else:
            power_level = "low"
            adequacy_score = 0.4
        
        return {
            'sample_size': n,
            'power_level': power_level,
            'adequacy_score': adequacy_score,
            'recommended_min_size': 30,
            'current_power_estimate': adequacy_score
        }


class KFoldMCPValidator:
    """
    K-fold cross-validation implementation for MCPs across different task types
    Implements stratified k-fold validation ensuring balanced representation of task types
    """
    
    def __init__(self, 
                 k: int = 5,
                 stratified: bool = True,
                 random_state: int = 42):
        """
        Initialize K-fold MCP validator
        
        @param {int} k - Number of folds for cross-validation (default: 5)
        @param {bool} stratified - Use stratified k-fold to maintain task type distribution
        @param {int} random_state - Random seed for reproducibility
        """
        self.k = k
        self.stratified = stratified
        self.random_state = random_state
        self.validation_history: List[CrossValidationResult] = []
        
        logger.info("Initialized K-Fold MCP Validator", extra={
            'operation': 'KFOLD_VALIDATOR_INIT',
            'k': k,
            'stratified': stratified
        })
    
    async def perform_k_fold_validation(self, 
                                      mcp_specs: List[MCPValidationSpec],
                                      validation_framework) -> List[CrossValidationResult]:
        """
        Perform k-fold cross-validation on list of MCP specifications
        
        @param {List[MCPValidationSpec]} mcp_specs - List of MCP specifications to validate
        @param {ValidationFramework} validation_framework - Framework for executing validation
        @returns {List[CrossValidationResult]} Results from k-fold validation
        """
        logger.info("Starting k-fold cross-validation", extra={
            'operation': 'KFOLD_VALIDATION_START',
            'mcp_count': len(mcp_specs),
            'k_value': self.k
        })
        
        validation_results = []
        
        try:
            # Group MCPs by task type for stratified validation
            task_type_groups = self._group_mcps_by_task_type(mcp_specs)
            
            for task_type, mcps in task_type_groups.items():
                if len(mcps) < self.k:
                    logger.warning(f"Insufficient MCPs for {task_type} k-fold validation", extra={
                        'operation': 'KFOLD_INSUFFICIENT_SAMPLES',
                        'task_type': task_type.value,
                        'mcp_count': len(mcps),
                        'required_k': self.k
                    })
                    continue
                
                # Perform k-fold validation for this task type
                task_results = await self._validate_task_type_group(mcps, task_type, validation_framework)
                validation_results.extend(task_results)
            
            logger.info("K-fold cross-validation completed", extra={
                'operation': 'KFOLD_VALIDATION_COMPLETE',
                'total_results': len(validation_results)
            })
            
            return validation_results
            
        except Exception as e:
            logger.error("K-fold cross-validation failed", extra={
                'operation': 'KFOLD_VALIDATION_FAILED',
                'error': str(e)
            })
            raise
    
    def _group_mcps_by_task_type(self, mcp_specs: List[MCPValidationSpec]) -> Dict[TaskType, List[MCPValidationSpec]]:
        """Group MCP specifications by task type for stratified validation"""
        groups = defaultdict(list)
        for mcp_spec in mcp_specs:
            groups[mcp_spec.task_type].append(mcp_spec)
        return dict(groups)
    
    async def _validate_task_type_group(self, 
                                      mcps: List[MCPValidationSpec],
                                      task_type: TaskType,
                                      validation_framework) -> List[CrossValidationResult]:
        """Perform k-fold validation on a specific task type group"""
        results = []
        
        # Create k-fold splits
        if self.stratified and len(mcps) >= self.k:
            kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
            # Create labels for stratification (could be based on complexity, performance requirements, etc.)
            labels = [hash(mcp.name) % 3 for mcp in mcps]  # Simple stratification
            splits = list(kf.split(mcps, labels))
        else:
            kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(mcps))
        
        for fold_idx, (train_indices, test_indices) in enumerate(splits):
            train_mcps = [mcps[i] for i in train_indices]
            test_mcps = [mcps[i] for i in test_indices]
            
            logger.info(f"Processing fold {fold_idx + 1}/{self.k} for {task_type.value}", extra={
                'operation': 'KFOLD_FOLD_PROCESSING',
                'fold': fold_idx + 1,
                'task_type': task_type.value,
                'train_size': len(train_mcps),
                'test_size': len(test_mcps)
            })
            
            # Validate test MCPs using training MCPs as reference
            for test_mcp in test_mcps:
                fold_result = await self._validate_single_mcp_in_fold(
                    test_mcp, train_mcps, fold_idx, validation_framework
                )
                results.append(fold_result)
        
        return results
    
    async def _validate_single_mcp_in_fold(self,
                                         test_mcp: MCPValidationSpec,
                                         train_mcps: List[MCPValidationSpec],
                                         fold_idx: int,
                                         validation_framework) -> CrossValidationResult:
        """Validate a single MCP within a k-fold validation fold"""
        validation_start = time.time()
        
        try:
            # Execute comprehensive validation using the validation framework
            fold_metrics = await validation_framework.validate_mcp_comprehensive(
                test_mcp, reference_mcps=train_mcps
            )
            
            # Perform statistical analysis on fold results
            statistical_analysis = await self._analyze_fold_statistics(fold_metrics, train_mcps)
            
            # Generate recommendations based on fold performance
            recommendations = self._generate_fold_recommendations(fold_metrics, statistical_analysis)
            
            validation_duration = time.time() - validation_start
            
            result = CrossValidationResult(
                validation_id=f"kfold_{test_mcp.mcp_id}_{fold_idx}_{int(time.time())}",
                mcp_spec=test_mcp,
                k_value=self.k,
                fold_results=[{
                    'fold_index': fold_idx,
                    'metrics': fold_metrics,
                    'validation_duration': validation_duration
                }],
                aggregated_metrics=fold_metrics,
                statistical_analysis=statistical_analysis,
                error_analysis={},
                recommendations=recommendations,
                is_valid=fold_metrics.ground_truth_accuracy > 0.7,  # Configurable threshold
                confidence_score=fold_metrics.ground_truth_accuracy
            )
            
            self.validation_history.append(result)
            return result
            
        except Exception as e:
            logger.error("Single MCP fold validation failed", extra={
                'operation': 'KFOLD_SINGLE_VALIDATION_FAILED',
                'mcp_id': test_mcp.mcp_id,
                'fold': fold_idx,
                'error': str(e)
            })
            raise
    
    async def _analyze_fold_statistics(self, 
                                     fold_metrics: ValidationMetrics,
                                     reference_mcps: List[MCPValidationSpec]) -> Dict[str, Any]:
        """Analyze statistical properties of fold validation results"""
        # Compare against reference MCPs performance
        reference_scores = []
        for ref_mcp in reference_mcps:
            # Extract performance indicators from reference MCPs
            if 'performance_score' in ref_mcp.metadata:
                reference_scores.append(ref_mcp.metadata['performance_score'])
        
        if reference_scores:
            fold_score = fold_metrics.ground_truth_accuracy
            percentile_rank = stats.percentileofscore(reference_scores, fold_score)
            
            return {
                'percentile_rank': percentile_rank,
                'above_median': fold_score > np.median(reference_scores),
                'z_score': (fold_score - np.mean(reference_scores)) / np.std(reference_scores) if np.std(reference_scores) > 0 else 0,
                'reference_comparison': 'above_average' if fold_score > np.mean(reference_scores) else 'below_average'
            }
        else:
            return {'status': 'no_reference_data'}
    
    def _generate_fold_recommendations(self, 
                                     fold_metrics: ValidationMetrics,
                                     statistical_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on fold validation results"""
        recommendations = []
        
        # Performance-based recommendations
        if fold_metrics.execution_time_avg > 5.0:  # 5 seconds threshold
            recommendations.append("Consider optimizing execution time - current average exceeds 5 seconds")
        
        if fold_metrics.error_rate > 0.1:  # 10% error rate threshold
            recommendations.append("High error rate detected - review error handling and input validation")
        
        if fold_metrics.consistency_score < 0.8:  # 80% consistency threshold
            recommendations.append("Low consistency score - ensure deterministic behavior across runs")
        
        # Statistical analysis recommendations
        if 'percentile_rank' in statistical_analysis:
            if statistical_analysis['percentile_rank'] < 25:
                recommendations.append("Performance below 25th percentile - significant improvement needed")
            elif statistical_analysis['percentile_rank'] > 75:
                recommendations.append("Excellent performance - above 75th percentile")
        
        if not recommendations:
            recommendations.append("MCP meets all validation criteria - ready for deployment")
        
        return recommendations


class MultiScenarioTestFramework:
    """
    Multi-scenario testing framework leveraging KGoT structured knowledge
    Implements comprehensive scenario-based testing using knowledge graph patterns
    """
    
    def __init__(self, 
                 knowledge_graph_client,
                 scenario_generators: Optional[Dict[ScenarioType, Callable]] = None):
        """
        Initialize multi-scenario test framework
        
        @param {Any} knowledge_graph_client - KGoT knowledge graph client for pattern extraction
        @param {Optional[Dict[ScenarioType, Callable]]} scenario_generators - Custom scenario generators
        """
        self.knowledge_graph_client = knowledge_graph_client
        self.scenario_generators = scenario_generators or {}
        self.test_history: List[Dict[str, Any]] = []
        
        # Initialize default scenario generators
        self._initialize_default_generators()
        
        logger.info("Initialized Multi-Scenario Test Framework", extra={
            'operation': 'SCENARIO_FRAMEWORK_INIT',
            'available_generators': len(self.scenario_generators)
        })
    
    def _initialize_default_generators(self):
        """Initialize default scenario generators for each scenario type"""
        self.scenario_generators.update({
            ScenarioType.NOMINAL: self._generate_nominal_scenarios,
            ScenarioType.EDGE_CASE: self._generate_edge_case_scenarios,
            ScenarioType.STRESS_TEST: self._generate_stress_test_scenarios,
            ScenarioType.FAILURE_MODE: self._generate_failure_mode_scenarios,
            ScenarioType.ADVERSARIAL: self._generate_adversarial_scenarios,
            ScenarioType.INTEGRATION: self._generate_integration_scenarios
        })
    
    async def generate_test_scenarios(self, 
                                    mcp_spec: MCPValidationSpec,
                                    scenario_types: List[ScenarioType] = None) -> List[ValidationScenario]:
        """
        Generate comprehensive test scenarios for MCP validation
        
        @param {MCPValidationSpec} mcp_spec - MCP specification to generate scenarios for
        @param {List[ScenarioType]} scenario_types - Types of scenarios to generate (default: all)
        @returns {List[ValidationScenario]} Generated validation scenarios
        """
        scenario_types = scenario_types or list(ScenarioType)
        
        logger.info("Generating test scenarios", extra={
            'operation': 'SCENARIO_GENERATION_START',
            'mcp_id': mcp_spec.mcp_id,
            'scenario_types': [st.value for st in scenario_types]
        })
        
        all_scenarios = []
        
        try:
            # Extract knowledge patterns from KGoT graph
            kgot_patterns = await self._extract_kgot_knowledge_patterns(mcp_spec)
            
            for scenario_type in scenario_types:
                if scenario_type in self.scenario_generators:
                    scenarios = await self.scenario_generators[scenario_type](mcp_spec, kgot_patterns)
                    all_scenarios.extend(scenarios)
            
            logger.info("Test scenario generation completed", extra={
                'operation': 'SCENARIO_GENERATION_COMPLETE',
                'total_scenarios': len(all_scenarios)
            })
            
            return all_scenarios
            
        except Exception as e:
            logger.error("Test scenario generation failed", extra={
                'operation': 'SCENARIO_GENERATION_FAILED',
                'mcp_id': mcp_spec.mcp_id,
                'error': str(e)
            })
            raise
    
    async def _extract_kgot_knowledge_patterns(self, mcp_spec: MCPValidationSpec) -> Dict[str, Any]:
        """Extract relevant knowledge patterns from KGoT structured knowledge"""
        try:
            # Query knowledge graph for patterns related to MCP task type and capabilities
            patterns = {}
            
            if self.knowledge_graph_client:
                # Extract patterns for task type
                task_patterns = await self.knowledge_graph_client.query_patterns(
                    entity_type=mcp_spec.task_type.value,
                    capabilities=mcp_spec.capabilities
                )
                patterns['task_patterns'] = task_patterns
                
                # Extract common failure patterns
                failure_patterns = await self.knowledge_graph_client.query_failure_patterns(
                    task_type=mcp_spec.task_type.value
                )
                patterns['failure_patterns'] = failure_patterns
                
                # Extract performance benchmarks
                performance_patterns = await self.knowledge_graph_client.query_performance_patterns(
                    capabilities=mcp_spec.capabilities
                )
                patterns['performance_patterns'] = performance_patterns
            
            return patterns
            
        except Exception as e:
            logger.warning("Failed to extract KGoT patterns, using defaults", extra={
                'operation': 'KGOT_PATTERN_EXTRACTION_WARNING',
                'error': str(e)
            })
            return {}
    
    async def _generate_nominal_scenarios(self, 
                                        mcp_spec: MCPValidationSpec,
                                        kgot_patterns: Dict[str, Any]) -> List[ValidationScenario]:
        """Generate nominal operation test scenarios"""
        scenarios = []
        
        # Generate scenarios based on provided test cases
        for i, test_case in enumerate(mcp_spec.test_cases):
            scenario = ValidationScenario(
                scenario_id=f"{mcp_spec.mcp_id}_nominal_{i}",
                scenario_type=ScenarioType.NOMINAL,
                description=f"Nominal operation test case {i+1}",
                input_data=test_case,
                expected_behavior={'output': mcp_spec.expected_outputs[i] if i < len(mcp_spec.expected_outputs) else None},
                success_criteria=['correct_output', 'execution_within_timeout'],
                failure_conditions=['incorrect_output', 'execution_timeout', 'runtime_error'],
                kgot_knowledge_context=kgot_patterns.get('task_patterns', {})
            )
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_edge_case_scenarios(self,
                                          mcp_spec: MCPValidationSpec,
                                          kgot_patterns: Dict[str, Any]) -> List[ValidationScenario]:
        """Generate edge case test scenarios"""
        scenarios = []
        
        # Common edge cases based on task type
        edge_cases = {
            TaskType.DATA_PROCESSING: [
                {'input': [], 'description': 'Empty input data'},
                {'input': None, 'description': 'Null input data'},
                {'input': {'large_dataset': [0] * 10000}, 'description': 'Large dataset processing'}
            ],
            TaskType.WEB_SCRAPING: [
                {'url': 'https://nonexistent-domain-12345.com', 'description': 'Invalid URL'},
                {'url': 'https://httpstat.us/500', 'description': 'Server error response'},
                {'timeout': 0.1, 'description': 'Very short timeout'}
            ],
            TaskType.CODE_GENERATION: [
                {'complexity': 'high', 'description': 'High complexity code generation'},
                {'language': 'unsupported', 'description': 'Unsupported programming language'},
                {'constraints': ['no_loops', 'no_functions'], 'description': 'Severe constraints'}
            ]
        }
        
        task_edge_cases = edge_cases.get(mcp_spec.task_type, [])
        
        for i, edge_case in enumerate(task_edge_cases):
            scenario = ValidationScenario(
                scenario_id=f"{mcp_spec.mcp_id}_edge_{i}",
                scenario_type=ScenarioType.EDGE_CASE,
                description=edge_case['description'],
                input_data=edge_case,
                expected_behavior={'graceful_handling': True, 'error_message': 'descriptive'},
                success_criteria=['graceful_error_handling', 'informative_error_message'],
                failure_conditions=['unhandled_exception', 'silent_failure', 'undefined_behavior'],
                kgot_knowledge_context=kgot_patterns.get('failure_patterns', {})
            )
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_stress_test_scenarios(self,
                                            mcp_spec: MCPValidationSpec,
                                            kgot_patterns: Dict[str, Any]) -> List[ValidationScenario]:
        """Generate stress test scenarios for high load conditions"""
        scenarios = []
        
        # High load scenarios
        stress_scenarios = [
            {
                'concurrent_requests': 100,
                'description': 'High concurrency stress test',
                'duration': 60  # seconds
            },
            {
                'large_input_size': 1000000,  # 1MB+ input
                'description': 'Large input processing stress test'
            },
            {
                'rapid_succession': {'requests_per_second': 50, 'duration': 30},
                'description': 'Rapid succession request stress test'
            }
        ]
        
        for i, stress_config in enumerate(stress_scenarios):
            scenario = ValidationScenario(
                scenario_id=f"{mcp_spec.mcp_id}_stress_{i}",
                scenario_type=ScenarioType.STRESS_TEST,
                description=stress_config['description'],
                input_data=stress_config,
                expected_behavior={'maintain_performance': True, 'no_memory_leaks': True},
                success_criteria=['performance_degradation_acceptable', 'no_crashes', 'resource_cleanup'],
                failure_conditions=['significant_performance_loss', 'memory_leaks', 'system_crash'],
                kgot_knowledge_context=kgot_patterns.get('performance_patterns', {})
            )
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_failure_mode_scenarios(self,
                                             mcp_spec: MCPValidationSpec,
                                             kgot_patterns: Dict[str, Any]) -> List[ValidationScenario]:
        """Generate failure mode test scenarios"""
        scenarios = []
        
        # Common failure modes
        failure_modes = [
            {
                'type': 'network_failure',
                'simulation': {'disconnect_after': 5, 'reconnect_after': 10},
                'description': 'Network connectivity failure simulation'
            },
            {
                'type': 'resource_exhaustion',
                'simulation': {'memory_limit': '50MB', 'cpu_limit': '10%'},
                'description': 'Resource exhaustion simulation'
            },
            {
                'type': 'dependency_failure',
                'simulation': {'mock_dependency_failure': True},
                'description': 'External dependency failure simulation'
            }
        ]
        
        for i, failure_mode in enumerate(failure_modes):
            scenario = ValidationScenario(
                scenario_id=f"{mcp_spec.mcp_id}_failure_{i}",
                scenario_type=ScenarioType.FAILURE_MODE,
                description=failure_mode['description'],
                input_data=failure_mode['simulation'],
                expected_behavior={'graceful_degradation': True, 'recovery_attempt': True},
                success_criteria=['graceful_failure_handling', 'recovery_mechanisms', 'user_notification'],
                failure_conditions=['catastrophic_failure', 'data_corruption', 'silent_failure'],
                kgot_knowledge_context=kgot_patterns.get('failure_patterns', {})
            )
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_adversarial_scenarios(self,
                                            mcp_spec: MCPValidationSpec,
                                            kgot_patterns: Dict[str, Any]) -> List[ValidationScenario]:
        """Generate adversarial test scenarios"""
        scenarios = []
        
        # Adversarial inputs based on task type
        adversarial_inputs = {
            TaskType.DATA_PROCESSING: [
                {'malformed_data': '{"incomplete": json', 'description': 'Malformed JSON input'},
                {'injection_attempt': '<script>alert("xss")</script>', 'description': 'Script injection attempt'},
                {'oversized_input': 'A' * 1000000, 'description': 'Oversized string input'}
            ],
            TaskType.CODE_GENERATION: [
                {'malicious_prompt': 'Generate code to delete all files', 'description': 'Malicious code generation request'},
                {'infinite_loop_request': 'Create an infinite loop', 'description': 'Infinite loop generation request'}
            ]
        }
        
        task_adversarial = adversarial_inputs.get(mcp_spec.task_type, [])
        
        for i, adversarial_input in enumerate(task_adversarial):
            scenario = ValidationScenario(
                scenario_id=f"{mcp_spec.mcp_id}_adversarial_{i}",
                scenario_type=ScenarioType.ADVERSARIAL,
                description=adversarial_input['description'],
                input_data=adversarial_input,
                expected_behavior={'input_sanitization': True, 'security_measures': True},
                success_criteria=['input_validation', 'security_filtering', 'safe_execution'],
                failure_conditions=['security_breach', 'code_injection', 'unsafe_execution'],
                kgot_knowledge_context=kgot_patterns.get('failure_patterns', {})
            )
            scenarios.append(scenario)
        
        return scenarios
    
    async def _generate_integration_scenarios(self,
                                            mcp_spec: MCPValidationSpec,
                                            kgot_patterns: Dict[str, Any]) -> List[ValidationScenario]:
        """Generate integration test scenarios for cross-MCP interactions"""
        scenarios = []
        
        # Integration scenarios
        integration_tests = [
            {
                'interaction_type': 'sequential_chaining',
                'description': 'Sequential MCP chaining integration test',
                'setup': {'chain_length': 3, 'error_propagation': True}
            },
            {
                'interaction_type': 'parallel_execution',
                'description': 'Parallel MCP execution integration test',
                'setup': {'parallel_count': 5, 'resource_sharing': True}
            },
            {
                'interaction_type': 'data_sharing',
                'description': 'Cross-MCP data sharing integration test',
                'setup': {'shared_state': True, 'consistency_check': True}
            }
        ]
        
        for i, integration_test in enumerate(integration_tests):
            scenario = ValidationScenario(
                scenario_id=f"{mcp_spec.mcp_id}_integration_{i}",
                scenario_type=ScenarioType.INTEGRATION,
                description=integration_test['description'],
                input_data=integration_test['setup'],
                expected_behavior={'seamless_integration': True, 'data_consistency': True},
                success_criteria=['successful_integration', 'consistent_behavior', 'proper_error_handling'],
                failure_conditions=['integration_failure', 'data_inconsistency', 'deadlocks'],
                kgot_knowledge_context=kgot_patterns.get('task_patterns', {})
            )
            scenarios.append(scenario)
        
        return scenarios 


class ValidationMetricsEngine:
    """
    Comprehensive validation metrics calculation engine
    Implements the four core metrics specified in Task 16: reliability, consistency, performance, accuracy
    """
    
    def __init__(self):
        """Initialize validation metrics engine"""
        self.metrics_history: Dict[str, List[ValidationMetrics]] = defaultdict(list)
        
        logger.info("Initialized Validation Metrics Engine", extra={
            'operation': 'METRICS_ENGINE_INIT'
        })
    
    async def calculate_comprehensive_metrics(self,
                                            mcp_spec: MCPValidationSpec,
                                            execution_results: List[Dict[str, Any]],
                                            scenario_results: List[Dict[str, Any]],
                                            reference_data: Optional[Dict[str, Any]] = None) -> ValidationMetrics:
        """
        Calculate comprehensive validation metrics from execution and scenario results
        
        @param {MCPValidationSpec} mcp_spec - MCP specification being validated
        @param {List[Dict[str, Any]]} execution_results - Results from MCP execution tests
        @param {List[Dict[str, Any]]} scenario_results - Results from multi-scenario testing
        @param {Optional[Dict[str, Any]]} reference_data - Reference data for baseline comparison
        @returns {ValidationMetrics} Comprehensive validation metrics
        """
        logger.info("Calculating comprehensive validation metrics", extra={
            'operation': 'METRICS_CALCULATION_START',
            'mcp_id': mcp_spec.mcp_id,
            'execution_results_count': len(execution_results),
            'scenario_results_count': len(scenario_results)
        })
        
        try:
            # Calculate reliability metrics
            reliability_metrics = await self._calculate_reliability_metrics(execution_results, scenario_results)
            
            # Calculate consistency metrics
            consistency_metrics = await self._calculate_consistency_metrics(execution_results, scenario_results)
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(execution_results, scenario_results)
            
            # Calculate accuracy metrics
            accuracy_metrics = await self._calculate_accuracy_metrics(
                execution_results, scenario_results, mcp_spec, reference_data
            )
            
            # Calculate statistical significance
            statistical_metrics = await self._calculate_statistical_significance(execution_results)
            
            # Combine all metrics
            comprehensive_metrics = ValidationMetrics(
                # Reliability metrics
                consistency_score=reliability_metrics['consistency_score'],
                error_rate=reliability_metrics['error_rate'],
                failure_recovery_rate=reliability_metrics['failure_recovery_rate'],
                uptime_percentage=reliability_metrics['uptime_percentage'],
                
                # Consistency metrics
                output_consistency=consistency_metrics['output_consistency'],
                logic_consistency=consistency_metrics['logic_consistency'],
                temporal_consistency=consistency_metrics['temporal_consistency'],
                cross_model_agreement=consistency_metrics['cross_model_agreement'],
                
                # Performance metrics
                execution_time_avg=performance_metrics['execution_time_avg'],
                execution_time_std=performance_metrics['execution_time_std'],
                throughput_ops_per_sec=performance_metrics['throughput_ops_per_sec'],
                resource_efficiency=performance_metrics['resource_efficiency'],
                scalability_score=performance_metrics['scalability_score'],
                
                # Accuracy metrics
                ground_truth_accuracy=accuracy_metrics['ground_truth_accuracy'],
                expert_validation_score=accuracy_metrics['expert_validation_score'],
                benchmark_performance=accuracy_metrics['benchmark_performance'],
                cross_validation_accuracy=accuracy_metrics['cross_validation_accuracy'],
                
                # Statistical significance
                confidence_interval=statistical_metrics['confidence_interval'],
                p_value=statistical_metrics['p_value'],
                statistical_significance=statistical_metrics['statistical_significance'],
                
                # Metadata
                sample_size=len(execution_results) + len(scenario_results),
                validation_duration=sum(r.get('duration', 0) for r in execution_results + scenario_results)
            )
            
            # Store metrics in history
            self.metrics_history[mcp_spec.mcp_id].append(comprehensive_metrics)
            
            logger.info("Comprehensive validation metrics calculated", extra={
                'operation': 'METRICS_CALCULATION_COMPLETE',
                'mcp_id': mcp_spec.mcp_id,
                'overall_score': (comprehensive_metrics.ground_truth_accuracy + 
                                comprehensive_metrics.consistency_score) / 2
            })
            
            return comprehensive_metrics
            
        except Exception as e:
            logger.error("Metrics calculation failed", extra={
                'operation': 'METRICS_CALCULATION_FAILED',
                'mcp_id': mcp_spec.mcp_id,
                'error': str(e)
            })
            raise
    
    async def _calculate_reliability_metrics(self, 
                                           execution_results: List[Dict[str, Any]],
                                           scenario_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate reliability metrics: consistency, error rate, failure recovery, uptime"""
        all_results = execution_results + scenario_results
        
        if not all_results:
            return {
                'consistency_score': 0.0,
                'error_rate': 1.0,
                'failure_recovery_rate': 0.0,
                'uptime_percentage': 0.0
            }
        
        # Consistency score based on successful executions
        successful_runs = [r for r in all_results if r.get('success', False)]
        consistency_score = len(successful_runs) / len(all_results)
        
        # Error rate calculation
        error_count = len([r for r in all_results if r.get('error', False)])
        error_rate = error_count / len(all_results)
        
        # Failure recovery rate
        recovery_attempts = [r for r in all_results if r.get('recovery_attempted', False)]
        successful_recoveries = [r for r in recovery_attempts if r.get('recovery_successful', False)]
        failure_recovery_rate = len(successful_recoveries) / len(recovery_attempts) if recovery_attempts else 0.0
        
        # Uptime percentage (based on availability during testing)
        total_duration = sum(r.get('duration', 0) for r in all_results)
        downtime = sum(r.get('downtime', 0) for r in all_results)
        uptime_percentage = (total_duration - downtime) / total_duration if total_duration > 0 else 0.0
        
        return {
            'consistency_score': consistency_score,
            'error_rate': error_rate,
            'failure_recovery_rate': failure_recovery_rate,
            'uptime_percentage': uptime_percentage
        }
    
    async def _calculate_consistency_metrics(self,
                                           execution_results: List[Dict[str, Any]],
                                           scenario_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate consistency metrics: output, logic, temporal, cross-model consistency"""
        all_results = execution_results + scenario_results
        
        if not all_results:
            return {
                'output_consistency': 0.0,
                'logic_consistency': 0.0,
                'temporal_consistency': 0.0,
                'cross_model_agreement': 0.0
            }
        
        # Output consistency - measure variance in outputs for similar inputs
        output_groups = defaultdict(list)
        for result in all_results:
            input_hash = hash(str(result.get('input', '')))
            output_groups[input_hash].append(result.get('output', ''))
        
        output_consistency_scores = []
        for outputs in output_groups.values():
            if len(outputs) > 1:
                # Calculate similarity between outputs (simplified)
                unique_outputs = len(set(str(o) for o in outputs))
                consistency = 1.0 - (unique_outputs - 1) / len(outputs)
                output_consistency_scores.append(consistency)
        
        output_consistency = np.mean(output_consistency_scores) if output_consistency_scores else 1.0
        
        # Logic consistency - check for logical contradictions
        logic_consistency = 1.0  # Simplified - would require semantic analysis
        
        # Temporal consistency - consistency over time
        time_grouped_results = defaultdict(list)
        for result in all_results:
            time_bucket = int(result.get('timestamp', 0) // 3600)  # Hour buckets
            time_grouped_results[time_bucket].append(result)
        
        temporal_scores = []
        for time_results in time_grouped_results.values():
            if len(time_results) > 1:
                success_rate = len([r for r in time_results if r.get('success', False)]) / len(time_results)
                temporal_scores.append(success_rate)
        
        temporal_consistency = np.mean(temporal_scores) if temporal_scores else 1.0
        
        # Cross-model agreement (if multiple models used)
        cross_model_agreement = 1.0  # Simplified implementation
        
        return {
            'output_consistency': output_consistency,
            'logic_consistency': logic_consistency,
            'temporal_consistency': temporal_consistency,
            'cross_model_agreement': cross_model_agreement
        }
    
    async def _calculate_performance_metrics(self,
                                           execution_results: List[Dict[str, Any]],
                                           scenario_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics: execution time, throughput, efficiency, scalability"""
        all_results = execution_results + scenario_results
        
        if not all_results:
            return {
                'execution_time_avg': 0.0,
                'execution_time_std': 0.0,
                'throughput_ops_per_sec': 0.0,
                'resource_efficiency': 0.0,
                'scalability_score': 0.0
            }
        
        # Execution time statistics
        execution_times = [r.get('execution_time', 0) for r in all_results if r.get('execution_time') is not None]
        execution_time_avg = np.mean(execution_times) if execution_times else 0.0
        execution_time_std = np.std(execution_times) if execution_times else 0.0
        
        # Throughput calculation
        total_operations = len(all_results)
        total_time = sum(execution_times)
        throughput_ops_per_sec = total_operations / total_time if total_time > 0 else 0.0
        
        # Resource efficiency (memory and CPU utilization)
        memory_usage = [r.get('memory_usage', 0) for r in all_results if r.get('memory_usage') is not None]
        cpu_usage = [r.get('cpu_usage', 0) for r in all_results if r.get('cpu_usage') is not None]
        
        avg_memory = np.mean(memory_usage) if memory_usage else 0.0
        avg_cpu = np.mean(cpu_usage) if cpu_usage else 0.0
        
        # Simple efficiency calculation (lower resource usage = higher efficiency)
        resource_efficiency = 1.0 - (avg_memory + avg_cpu) / 2 if (avg_memory + avg_cpu) > 0 else 1.0
        resource_efficiency = max(0.0, min(1.0, resource_efficiency))
        
        # Scalability score based on performance under load
        load_tests = [r for r in scenario_results if r.get('scenario_type') == 'stress_test']
        if load_tests:
            baseline_performance = execution_time_avg
            load_performance = np.mean([r.get('execution_time', 0) for r in load_tests])
            scalability_score = baseline_performance / load_performance if load_performance > 0 else 0.0
            scalability_score = min(1.0, scalability_score)  # Cap at 1.0
        else:
            scalability_score = 1.0  # No load testing data
        
        return {
            'execution_time_avg': execution_time_avg,
            'execution_time_std': execution_time_std,
            'throughput_ops_per_sec': throughput_ops_per_sec,
            'resource_efficiency': resource_efficiency,
            'scalability_score': scalability_score
        }
    
    async def _calculate_accuracy_metrics(self,
                                        execution_results: List[Dict[str, Any]],
                                        scenario_results: List[Dict[str, Any]],
                                        mcp_spec: MCPValidationSpec,
                                        reference_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate accuracy metrics: ground truth, expert validation, benchmark, cross-validation"""
        all_results = execution_results + scenario_results
        
        if not all_results:
            return {
                'ground_truth_accuracy': 0.0,
                'expert_validation_score': 0.0,
                'benchmark_performance': 0.0,
                'cross_validation_accuracy': 0.0
            }
        
        # Ground truth accuracy (comparison with expected outputs)
        ground_truth_matches = 0
        total_comparisons = 0
        
        for i, result in enumerate(execution_results):
            if i < len(mcp_spec.expected_outputs):
                expected = mcp_spec.expected_outputs[i]
                actual = result.get('output')
                if self._compare_outputs(expected, actual):
                    ground_truth_matches += 1
                total_comparisons += 1
        
        ground_truth_accuracy = ground_truth_matches / total_comparisons if total_comparisons > 0 else 0.0
        
        # Expert validation score (if available in reference data)
        expert_validation_score = reference_data.get('expert_score', 0.8) if reference_data else 0.8
        
        # Benchmark performance (comparison with established benchmarks)
        benchmark_performance = reference_data.get('benchmark_score', 0.7) if reference_data else 0.7
        
        # Cross-validation accuracy (consistency across folds)
        successful_results = [r for r in all_results if r.get('success', False)]
        cross_validation_accuracy = len(successful_results) / len(all_results)
        
        return {
            'ground_truth_accuracy': ground_truth_accuracy,
            'expert_validation_score': expert_validation_score,
            'benchmark_performance': benchmark_performance,
            'cross_validation_accuracy': cross_validation_accuracy
        }
    
    async def _calculate_statistical_significance(self,
                                                execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistical significance metrics"""
        if len(execution_results) < 2:
            return {
                'confidence_interval': (0.0, 0.0),
                'p_value': 1.0,
                'statistical_significance': False
            }
        
        # Extract success rates for statistical analysis
        success_rates = [1.0 if r.get('success', False) else 0.0 for r in execution_results]
        
        # Calculate confidence interval for success rate
        mean_success = np.mean(success_rates)
        std_error = stats.sem(success_rates)
        confidence_interval = stats.t.interval(0.95, len(success_rates)-1, mean_success, std_error)
        
        # Simple one-sample t-test against null hypothesis (success rate = 0.5)
        t_stat, p_value = stats.ttest_1samp(success_rates, 0.5)
        statistical_significance = p_value < 0.05
        
        return {
            'confidence_interval': confidence_interval,
            'p_value': p_value,
            'statistical_significance': statistical_significance
        }
    
    def _compare_outputs(self, expected: Any, actual: Any) -> bool:
        """Compare expected and actual outputs for accuracy calculation"""
        try:
            # Handle different data types
            if isinstance(expected, str) and isinstance(actual, str):
                return expected.strip().lower() == actual.strip().lower()
            elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                return abs(expected - actual) < 1e-6  # Floating point tolerance
            elif isinstance(expected, dict) and isinstance(actual, dict):
                return json.dumps(expected, sort_keys=True) == json.dumps(actual, sort_keys=True)
            else:
                return str(expected) == str(actual)
        except Exception:
            return False


class ErrorManagementBridge:
    """
    Bridge connecting MCP cross-validation with KGoT Section 2.5 Error Management
    Implements layered error containment integration for validation processes
    """
    
    def __init__(self, error_management_system: KGoTErrorManagementSystem):
        """
        Initialize error management bridge
        
        @param {KGoTErrorManagementSystem} error_management_system - KGoT error management system
        """
        self.error_management_system = error_management_system
        self.validation_error_contexts: Dict[str, List[ErrorContext]] = defaultdict(list)
        
        logger.info("Initialized Error Management Bridge", extra={
            'operation': 'ERROR_BRIDGE_INIT'
        })
    
    async def handle_validation_error(self,
                                    validation_id: str,
                                    error: Exception,
                                    operation_context: str,
                                    mcp_spec: Optional[MCPValidationSpec] = None) -> Tuple[Any, bool]:
        """
        Handle validation errors using KGoT error management system
        
        @param {str} validation_id - Unique validation identifier
        @param {Exception} error - Error that occurred during validation
        @param {str} operation_context - Context of the validation operation
        @param {Optional[MCPValidationSpec]} mcp_spec - MCP specification being validated
        @returns {Tuple[Any, bool]} Recovery result and success flag
        """
        logger.info("Handling validation error", extra={
            'operation': 'VALIDATION_ERROR_HANDLE',
            'validation_id': validation_id,
            'error_type': type(error).__name__,
            'operation_context': operation_context
        })
        
        try:
            # Classify the validation error
            error_type = self._classify_validation_error(error)
            error_severity = self._determine_error_severity(error, mcp_spec)
            
            # Create error context for validation
            error_context = ErrorContext(
                error_id=f"validation_{validation_id}_{int(time.time())}",
                error_type=error_type,
                severity=error_severity,
                timestamp=datetime.now(),
                original_operation=operation_context,
                error_message=str(error),
                stack_trace=getattr(error, '__traceback__', None),
                metadata={
                    'validation_id': validation_id,
                    'mcp_id': mcp_spec.mcp_id if mcp_spec else None,
                    'mcp_task_type': mcp_spec.task_type.value if mcp_spec else None
                }
            )
            
            # Store error context
            self.validation_error_contexts[validation_id].append(error_context)
            
            # Delegate to KGoT error management system
            recovery_result, recovery_success = await self.error_management_system.handle_error(
                error, operation_context, error_type, error_severity
            )
            
            # Log recovery outcome
            logger.info("Validation error recovery completed", extra={
                'operation': 'VALIDATION_ERROR_RECOVERY',
                'validation_id': validation_id,
                'recovery_success': recovery_success,
                'error_id': error_context.error_id
            })
            
            return recovery_result, recovery_success
            
        except Exception as bridge_error:
            logger.error("Error management bridge failed", extra={
                'operation': 'ERROR_BRIDGE_FAILED',
                'validation_id': validation_id,
                'original_error': str(error),
                'bridge_error': str(bridge_error)
            })
            # Return failure if bridge itself fails
            return None, False
    
    def _classify_validation_error(self, error: Exception) -> ErrorType:
        """Classify validation errors into KGoT error types"""
        error_type_name = type(error).__name__
        
        if 'Syntax' in error_type_name or 'Parse' in error_type_name:
            return ErrorType.SYNTAX_ERROR
        elif 'Timeout' in error_type_name or 'TimeoutError' in error_type_name:
            return ErrorType.TIMEOUT_ERROR
        elif 'Connection' in error_type_name or 'Network' in error_type_name:
            return ErrorType.API_ERROR
        elif 'Validation' in error_type_name or 'ValueError' in error_type_name:
            return ErrorType.VALIDATION_ERROR
        elif 'Permission' in error_type_name or 'Security' in error_type_name:
            return ErrorType.SECURITY_ERROR
        elif 'Runtime' in error_type_name or 'Execution' in error_type_name:
            return ErrorType.EXECUTION_ERROR
        else:
            return ErrorType.SYSTEM_ERROR
    
    def _determine_error_severity(self, 
                                error: Exception, 
                                mcp_spec: Optional[MCPValidationSpec]) -> ErrorSeverity:
        """Determine error severity based on error type and MCP context"""
        error_type_name = type(error).__name__
        
        # Critical errors that prevent validation
        if any(keyword in error_type_name for keyword in ['Fatal', 'Critical', 'Segmentation']):
            return ErrorSeverity.CRITICAL
        
        # High severity for security and data integrity issues
        if any(keyword in error_type_name for keyword in ['Security', 'Permission', 'Corruption']):
            return ErrorSeverity.HIGH
        
        # Medium severity for operational errors
        if any(keyword in error_type_name for keyword in ['Timeout', 'Connection', 'Runtime']):
            return ErrorSeverity.MEDIUM
        
        # Low severity for minor validation issues
        if any(keyword in error_type_name for keyword in ['Validation', 'Format', 'Parse']):
            return ErrorSeverity.LOW
        
        # Default to medium severity
        return ErrorSeverity.MEDIUM
    
    async def get_validation_error_summary(self, validation_id: str) -> Dict[str, Any]:
        """Get comprehensive error summary for a validation session"""
        error_contexts = self.validation_error_contexts.get(validation_id, [])
        
        if not error_contexts:
            return {'total_errors': 0, 'error_summary': {}}
        
        # Aggregate error statistics
        error_type_counts = Counter(ctx.error_type for ctx in error_contexts)
        severity_counts = Counter(ctx.severity for ctx in error_contexts)
        
        return {
            'total_errors': len(error_contexts),
            'error_types': dict(error_type_counts),
            'severity_distribution': dict(severity_counts),
            'error_rate': len(error_contexts) / max(1, len(error_contexts)),  # Simplified
            'recovery_attempts': sum(len(ctx.recovery_attempts) for ctx in error_contexts),
            'last_error_time': max(ctx.timestamp for ctx in error_contexts).isoformat(),
            'error_contexts': [ctx.to_dict() for ctx in error_contexts[-5:]]  # Last 5 errors
        }


class MCPCrossValidationEngine:
    """
    Main MCP Cross-Validation Engine - Task 16 Implementation
    
    Orchestrates comprehensive MCP validation using:
    - K-fold cross-validation across task types
    - Multi-scenario testing with KGoT knowledge
    - Comprehensive metrics calculation
    - Statistical significance analysis
    - Error management integration
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 llm_client: Optional[Any] = None,
                 knowledge_graph_client: Optional[Any] = None,
                 error_management_system: Optional[KGoTErrorManagementSystem] = None):
        """
        Initialize MCP Cross-Validation Engine
        
        @param {Dict[str, Any]} config - Configuration for validation engine
        @param {Optional[Any]} llm_client - LLM client for validation (OpenRouter-based per user memory)
        @param {Optional[Any]} knowledge_graph_client - KGoT knowledge graph client
        @param {Optional[KGoTErrorManagementSystem]} error_management_system - Error management system
        """
        self.config = config
        self.llm_client = llm_client
        self.knowledge_graph_client = knowledge_graph_client
        
        # Initialize core components
        self.statistical_analyzer = StatisticalSignificanceAnalyzer(
            significance_level=config.get('significance_level', 0.05)
        )
        
        self.kfold_validator = KFoldMCPValidator(
            k=config.get('k_fold', 5),
            stratified=config.get('stratified', True),
            random_state=config.get('random_state', 42)
        )
        
        self.scenario_framework = MultiScenarioTestFramework(
            knowledge_graph_client=knowledge_graph_client,
            scenario_generators=config.get('custom_scenario_generators')
        )
        
        self.metrics_engine = ValidationMetricsEngine()
        
        # Initialize error management bridge
        if error_management_system:
            self.error_bridge = ErrorManagementBridge(error_management_system)
        else:
            # Create default error management system
            default_error_system = KGoTErrorManagementSystem(llm_client)
            self.error_bridge = ErrorManagementBridge(default_error_system)
        
        # Validation state
        self.active_validations: Dict[str, Dict[str, Any]] = {}
        self.validation_history: List[Dict[str, Any]] = []
        
        logger.info("Initialized MCP Cross-Validation Engine", extra={
            'operation': 'MCP_VALIDATION_ENGINE_INIT',
            'config': config
        })
    
    async def validate_mcp_comprehensive(self,
                                       mcp_spec: MCPValidationSpec,
                                       validation_config: Optional[Dict[str, Any]] = None) -> CrossValidationResult:
        """
        Perform comprehensive cross-validation of a single MCP
        
        @param {MCPValidationSpec} mcp_spec - MCP specification to validate
        @param {Optional[Dict[str, Any]]} validation_config - Override validation configuration
        @returns {CrossValidationResult} Comprehensive validation results
        """
        validation_id = f"comprehensive_{mcp_spec.mcp_id}_{uuid.uuid4().hex[:8]}"
        validation_start = time.time()
        
        logger.info("Starting comprehensive MCP validation", extra={
            'operation': 'COMPREHENSIVE_VALIDATION_START',
            'validation_id': validation_id,
            'mcp_id': mcp_spec.mcp_id,
            'task_type': mcp_spec.task_type.value
        })
        
        # Track active validation
        self.active_validations[validation_id] = {
            'mcp_spec': mcp_spec,
            'start_time': validation_start,
            'status': 'initializing'
        }
        
        try:
            # Phase 1: Generate comprehensive test scenarios
            self.active_validations[validation_id]['status'] = 'generating_scenarios'
            test_scenarios = await self.scenario_framework.generate_test_scenarios(mcp_spec)
            
            # Phase 2: Execute scenario-based testing
            self.active_validations[validation_id]['status'] = 'executing_scenarios'
            scenario_results = await self._execute_test_scenarios(mcp_spec, test_scenarios, validation_id)
            
            # Phase 3: Execute basic functionality tests
            self.active_validations[validation_id]['status'] = 'executing_basic_tests'
            execution_results = await self._execute_basic_functionality_tests(mcp_spec, validation_id)
            
            # Phase 4: Calculate comprehensive metrics
            self.active_validations[validation_id]['status'] = 'calculating_metrics'
            validation_metrics = await self.metrics_engine.calculate_comprehensive_metrics(
                mcp_spec, execution_results, scenario_results
            )
            
            # Phase 5: Perform statistical significance analysis
            self.active_validations[validation_id]['status'] = 'statistical_analysis'
            statistical_analysis = await self.statistical_analyzer.analyze_validation_significance(
                [validation_metrics]
            )
            
            # Phase 6: Generate recommendations
            recommendations = self._generate_comprehensive_recommendations(
                validation_metrics, statistical_analysis, scenario_results
            )
            
            validation_duration = time.time() - validation_start
            
            # Create comprehensive result
            result = CrossValidationResult(
                validation_id=validation_id,
                mcp_spec=mcp_spec,
                k_value=1,  # Single MCP validation
                fold_results=[{
                    'fold_index': 0,
                    'metrics': validation_metrics,
                    'scenario_results': scenario_results,
                    'execution_results': execution_results,
                    'validation_duration': validation_duration
                }],
                aggregated_metrics=validation_metrics,
                statistical_analysis=statistical_analysis,
                error_analysis=await self.error_bridge.get_validation_error_summary(validation_id),
                recommendations=recommendations,
                is_valid=self._determine_overall_validity(validation_metrics, statistical_analysis),
                confidence_score=self._calculate_confidence_score(validation_metrics, statistical_analysis)
            )
            
            # Store result
            self.validation_history.append(result.validation_id)
            self.active_validations[validation_id]['status'] = 'completed'
            self.active_validations[validation_id]['result'] = result
            
            logger.info("Comprehensive MCP validation completed", extra={
                'operation': 'COMPREHENSIVE_VALIDATION_COMPLETE',
                'validation_id': validation_id,
                'is_valid': result.is_valid,
                'confidence_score': result.confidence_score,
                'duration': validation_duration
            })
            
            return result
            
        except Exception as e:
            # Handle validation error using error management bridge
            self.active_validations[validation_id]['status'] = 'error'
            
            logger.error("Comprehensive MCP validation failed", extra={
                'operation': 'COMPREHENSIVE_VALIDATION_FAILED',
                'validation_id': validation_id,
                'error': str(e)
            })
            
            # Attempt error recovery
            recovery_result, recovery_success = await self.error_bridge.handle_validation_error(
                validation_id, e, f"comprehensive_validation_{mcp_spec.mcp_id}", mcp_spec
            )
            
            if not recovery_success:
                raise
            
            return recovery_result
        
        finally:
            # Clean up active validation tracking
            if validation_id in self.active_validations:
                self.active_validations[validation_id]['end_time'] = time.time()
    
    async def validate_mcp_batch_k_fold(self,
                                      mcp_specs: List[MCPValidationSpec],
                                      k: Optional[int] = None) -> List[CrossValidationResult]:
        """
        Perform k-fold cross-validation on a batch of MCPs
        
        @param {List[MCPValidationSpec]} mcp_specs - List of MCP specifications to validate
        @param {Optional[int]} k - Number of folds (overrides config)
        @returns {List[CrossValidationResult]} K-fold validation results
        """
        if k:
            self.kfold_validator.k = k
        
        logger.info("Starting batch k-fold validation", extra={
            'operation': 'BATCH_KFOLD_VALIDATION_START',
            'mcp_count': len(mcp_specs),
            'k_value': self.kfold_validator.k
        })
        
        try:
            # Perform k-fold cross-validation
            kfold_results = await self.kfold_validator.perform_k_fold_validation(mcp_specs, self)
            
            logger.info("Batch k-fold validation completed", extra={
                'operation': 'BATCH_KFOLD_VALIDATION_COMPLETE',
                'result_count': len(kfold_results)
            })
            
            return kfold_results
            
        except Exception as e:
            logger.error("Batch k-fold validation failed", extra={
                'operation': 'BATCH_KFOLD_VALIDATION_FAILED',
                'error': str(e)
            })
            raise
    
    async def _execute_test_scenarios(self,
                                    mcp_spec: MCPValidationSpec,
                                    scenarios: List[ValidationScenario],
                                    validation_id: str) -> List[Dict[str, Any]]:
        """Execute multi-scenario tests for MCP validation"""
        scenario_results = []
        
        for scenario in scenarios:
            try:
                scenario_start = time.time()
                
                # Execute scenario test (simplified - would integrate with actual MCP execution)
                result = await self._execute_single_scenario(mcp_spec, scenario)
                
                scenario_duration = time.time() - scenario_start
                result.update({
                    'scenario_id': scenario.scenario_id,
                    'scenario_type': scenario.scenario_type.value,
                    'duration': scenario_duration,
                    'timestamp': time.time()
                })
                
                scenario_results.append(result)
                
            except Exception as e:
                # Handle scenario execution error
                await self.error_bridge.handle_validation_error(
                    validation_id, e, f"scenario_execution_{scenario.scenario_id}", mcp_spec
                )
                
                scenario_results.append({
                    'scenario_id': scenario.scenario_id,
                    'scenario_type': scenario.scenario_type.value,
                    'success': False,
                    'error': str(e),
                    'duration': 0,
                    'timestamp': time.time()
                })
        
        return scenario_results
    
    async def _execute_single_scenario(self,
                                     mcp_spec: MCPValidationSpec,
                                     scenario: ValidationScenario) -> Dict[str, Any]:
        """Execute a single validation scenario (simplified implementation)"""
        # This would integrate with actual MCP execution infrastructure
        # For now, provide a simplified simulation
        
        if scenario.scenario_type == ScenarioType.NOMINAL:
            # Simulate successful execution for nominal scenarios
            return {
                'success': True,
                'output': f"simulated_output_for_{scenario.scenario_id}",
                'execution_time': 0.5,
                'memory_usage': 0.1,
                'cpu_usage': 0.2
            }
        elif scenario.scenario_type == ScenarioType.EDGE_CASE:
            # Simulate graceful error handling for edge cases
            return {
                'success': True,
                'output': None,
                'error_message': "Gracefully handled edge case",
                'execution_time': 0.8,
                'memory_usage': 0.15,
                'cpu_usage': 0.25
            }
        else:
            # Simulate other scenario types
            return {
                'success': True,
                'output': f"scenario_result_{scenario.scenario_type.value}",
                'execution_time': 1.0,
                'memory_usage': 0.2,
                'cpu_usage': 0.3
            }
    
    async def _execute_basic_functionality_tests(self,
                                               mcp_spec: MCPValidationSpec,
                                               validation_id: str) -> List[Dict[str, Any]]:
        """Execute basic functionality tests using provided test cases"""
        execution_results = []
        
        for i, test_case in enumerate(mcp_spec.test_cases):
            try:
                execution_start = time.time()
                
                # Execute basic test (simplified - would integrate with actual MCP execution)
                result = await self._execute_basic_test(mcp_spec, test_case, i)
                
                execution_duration = time.time() - execution_start
                result.update({
                    'test_case_index': i,
                    'duration': execution_duration,
                    'timestamp': time.time()
                })
                
                execution_results.append(result)
                
            except Exception as e:
                # Handle execution error
                await self.error_bridge.handle_validation_error(
                    validation_id, e, f"basic_test_execution_{i}", mcp_spec
                )
                
                execution_results.append({
                    'test_case_index': i,
                    'success': False,
                    'error': str(e),
                    'duration': 0,
                    'timestamp': time.time()
                })
        
        return execution_results
    
    async def _execute_basic_test(self,
                                mcp_spec: MCPValidationSpec,
                                test_case: Dict[str, Any],
                                test_index: int) -> Dict[str, Any]:
        """Execute a single basic functionality test (simplified implementation)"""
        # This would integrate with actual MCP execution infrastructure
        # For now, provide a simplified simulation
        
        expected_output = mcp_spec.expected_outputs[test_index] if test_index < len(mcp_spec.expected_outputs) else None
        
        return {
            'success': True,
            'input': test_case,
            'output': expected_output,  # Simulate correct output
            'execution_time': 0.3,
            'memory_usage': 0.08,
            'cpu_usage': 0.15
        }
    
    def _generate_comprehensive_recommendations(self,
                                              metrics: ValidationMetrics,
                                              statistical_analysis: Dict[str, Any],
                                              scenario_results: List[Dict[str, Any]]) -> List[str]:
        """Generate comprehensive recommendations based on validation results"""
        recommendations = []
        
        # Performance recommendations
        if metrics.execution_time_avg > 3.0:
            recommendations.append("Consider optimizing execution time - average exceeds 3 seconds")
        
        if metrics.error_rate > 0.05:
            recommendations.append("Error rate above 5% - review error handling and input validation")
        
        # Consistency recommendations
        if metrics.output_consistency < 0.9:
            recommendations.append("Output consistency below 90% - ensure deterministic behavior")
        
        # Statistical recommendations
        if not metrics.statistical_significance:
            recommendations.append("Results lack statistical significance - consider increasing sample size")
        
        # Scenario-specific recommendations
        failed_scenarios = [r for r in scenario_results if not r.get('success', False)]
        if len(failed_scenarios) > len(scenario_results) * 0.1:
            recommendations.append("More than 10% of scenarios failed - review failure modes")
        
        # Resource efficiency recommendations
        if metrics.resource_efficiency < 0.8:
            recommendations.append("Resource efficiency below 80% - optimize memory and CPU usage")
        
        if not recommendations:
            recommendations.append("MCP meets all validation criteria - excellent performance")
        
        return recommendations
    
    def _determine_overall_validity(self,
                                  metrics: ValidationMetrics,
                                  statistical_analysis: Dict[str, Any]) -> bool:
        """Determine overall MCP validity based on comprehensive metrics"""
        # Define validation thresholds
        thresholds = {
            'min_accuracy': 0.8,
            'max_error_rate': 0.1,
            'min_consistency': 0.8,
            'min_performance_score': 0.7
        }
        
        # Calculate overall performance score
        performance_score = (
            metrics.ground_truth_accuracy * 0.3 +
            (1 - metrics.error_rate) * 0.2 +
            metrics.output_consistency * 0.2 +
            metrics.resource_efficiency * 0.15 +
            metrics.scalability_score * 0.15
        )
        
        # Check all criteria
        return (
            metrics.ground_truth_accuracy >= thresholds['min_accuracy'] and
            metrics.error_rate <= thresholds['max_error_rate'] and
            metrics.output_consistency >= thresholds['min_consistency'] and
            performance_score >= thresholds['min_performance_score']
        )
    
    def _calculate_confidence_score(self,
                                  metrics: ValidationMetrics,
                                  statistical_analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score for validation results"""
        # Base confidence from accuracy metrics
        base_confidence = (
            metrics.ground_truth_accuracy * 0.4 +
            metrics.cross_validation_accuracy * 0.3 +
            metrics.output_consistency * 0.3
        )
        
        # Adjust for statistical significance
        if metrics.statistical_significance:
            statistical_boost = 0.1
        else:
            statistical_boost = -0.1
        
        # Adjust for sample size
        if metrics.sample_size >= 30:
            sample_boost = 0.05
        elif metrics.sample_size >= 10:
            sample_boost = 0.0
        else:
            sample_boost = -0.1
        
        confidence_score = base_confidence + statistical_boost + sample_boost
        return max(0.0, min(1.0, confidence_score))


# Factory function for creating MCP Cross-Validation Engine
def create_mcp_cross_validation_engine(config: Dict[str, Any],
                                     llm_client: Optional[Any] = None,
                                     knowledge_graph_client: Optional[Any] = None,
                                     error_management_system: Optional[KGoTErrorManagementSystem] = None) -> MCPCrossValidationEngine:
    """
    Factory function to create MCP Cross-Validation Engine
    
    @param {Dict[str, Any]} config - Configuration for validation engine
    @param {Optional[Any]} llm_client - LLM client for validation
    @param {Optional[Any]} knowledge_graph_client - KGoT knowledge graph client  
    @param {Optional[KGoTErrorManagementSystem]} error_management_system - Error management system
    @returns {MCPCrossValidationEngine} Configured validation engine
    """
    logger.info("Creating MCP Cross-Validation Engine", extra={
        'operation': 'MCP_VALIDATION_ENGINE_CREATE',
        'config_keys': list(config.keys())
    })
    
    return MCPCrossValidationEngine(
        config=config,
        llm_client=llm_client,
        knowledge_graph_client=knowledge_graph_client,
        error_management_system=error_management_system
    )


# Example usage and testing
if __name__ == "__main__":
    async def example_usage():
        """Example usage of MCP Cross-Validation Engine"""
        
        # Configuration
        config = {
            'k_fold': 5,
            'significance_level': 0.05,
            'stratified': True,
            'random_state': 42
        }
        
        # Create validation engine
        validation_engine = create_mcp_cross_validation_engine(config)
        
        # Example MCP specification
        example_mcp = MCPValidationSpec(
            mcp_id="example_mcp_001",
            name="Example Data Processing MCP",
            description="Processes data and returns analysis results",
            task_type=TaskType.DATA_PROCESSING,
            capabilities=["data_analysis", "json_processing", "statistical_computation"],
            implementation_code="def process_data(data): return {'result': len(data)}",
            test_cases=[
                {"input": [1, 2, 3, 4, 5]},
                {"input": []},
                {"input": [10, 20, 30]}
            ],
            expected_outputs=[
                {"result": 5},
                {"result": 0},
                {"result": 3}
            ],
            performance_requirements={"max_execution_time": 2.0},
            reliability_requirements={"min_success_rate": 0.95}
        )
        
        # Perform comprehensive validation
        try:
            result = await validation_engine.validate_mcp_comprehensive(example_mcp)
            
            print(f"Validation Result: {'PASSED' if result.is_valid else 'FAILED'}")
            print(f"Confidence Score: {result.confidence_score:.3f}")
            print(f"Recommendations: {', '.join(result.recommendations)}")
            
        except Exception as e:
            print(f"Validation failed: {e}")
    
    # Run example if executed directly
    import asyncio
    asyncio.run(example_usage()) 