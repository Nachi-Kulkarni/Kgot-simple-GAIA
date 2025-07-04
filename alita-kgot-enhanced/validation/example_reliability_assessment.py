#!/usr/bin/env python3
"""
Example Usage: KGoT Knowledge-Driven Reliability Assessor

This example demonstrates how to use the comprehensive reliability assessment system
built for Task 19, implementing all KGoT research paper insights.

Usage:
    python example_reliability_assessment.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation.kgot_reliability_assessor import (
    KGoTReliabilityAssessor,
    ReliabilityAssessmentConfig
)
from config.logging.winston_config import logger


async def run_comprehensive_reliability_assessment():
    """
    Run a complete reliability assessment using the KGoT Knowledge-Driven Reliability Assessor
    
    This demonstrates all five reliability dimensions from the KGoT research paper:
    - Section 2.1: Graph Store Module reliability
    - Section 2.5: Error Management robustness
    - Section 1.3: Extraction method consistency
    - Section 2.4: Stress resilience
    - Section 2.3.3: Alita integration stability
    """
    
    logger.info("Starting KGoT Knowledge-Driven Reliability Assessment Example", extra={
        'operation': 'EXAMPLE_RELIABILITY_ASSESSMENT_START',
        'task': 'Task_19_Demo'
    })
    
    try:
        # Step 1: Configure the reliability assessment
        logger.info("Configuring reliability assessment parameters")
        
        config = ReliabilityAssessmentConfig(
            # Graph Store Module Configuration (Section 2.1)
            graph_backends_to_test=['networkx', 'neo4j'],
            graph_validation_iterations=5,  # Reduced for demo
            graph_stress_node_count=100,    # Reduced for demo
            graph_stress_edge_count=500,    # Reduced for demo
            
            # Error Management Configuration (Section 2.5)
            error_injection_scenarios=10,   # Reduced for demo
            error_recovery_timeout=10,      # Reduced for demo
            error_escalation_levels=3,
            
            # Extraction Methods Configuration (Section 1.3)
            extraction_consistency_samples=5,  # Reduced for demo
            extraction_backends_parallel=True,
            extraction_query_variations=3,     # Reduced for demo
            
            # Stress Testing Configuration (Section 2.4)
            stress_test_duration=30,        # Reduced for demo (30 seconds)
            stress_concurrent_operations=10, # Reduced for demo
            stress_resource_limits={
                'memory_limit_mb': 512,     # Reduced for demo
                'cpu_limit_percent': 70
            },
            
            # Alita Integration Configuration (Section 2.3.3)
            alita_refinement_cycles=3,      # Reduced for demo
            alita_error_scenarios=5,        # Reduced for demo
            alita_session_timeout=30,       # Reduced for demo
            
            # Statistical Analysis
            confidence_level=0.95,
            statistical_significance_threshold=0.05,
            reliability_threshold=0.80      # 80% reliability threshold
        )
        
        # Step 2: Initialize the KGoT Reliability Assessor
        logger.info("Initializing KGoT Reliability Assessor")
        
        # For demo purposes, we'll use mock LLM client
        mock_llm_client = {"client_type": "demo", "api_key": "demo_key"}
        
        reliability_assessor = KGoTReliabilityAssessor(
            config=config,
            llm_client=mock_llm_client
        )
        
        # Step 3: Initialize required dependencies
        logger.info("Initializing assessment dependencies")
        
        # Mock graph instances for different backends
        mock_graph_instances = {
            'networkx': MockKnowledgeGraphInterface('networkx'),
            'neo4j': MockKnowledgeGraphInterface('neo4j')
        }
        
        # Mock performance optimizer
        mock_performance_optimizer = MockPerformanceOptimizer()
        
        # Mock Alita refinement bridge
        mock_alita_bridge = MockAlitaRefinementBridge()
        
        # Initialize dependencies
        await reliability_assessor.initialize_dependencies(
            graph_instances=mock_graph_instances,
            performance_optimizer=mock_performance_optimizer,
            alita_bridge=mock_alita_bridge
        )
        
        # Step 4: Perform comprehensive reliability assessment
        logger.info("Performing comprehensive reliability assessment")
        
        reliability_metrics = await reliability_assessor.perform_comprehensive_assessment()
        
        # Step 5: Display results
        display_assessment_results(reliability_metrics)
        
        logger.info("KGoT Reliability Assessment completed successfully", extra={
            'operation': 'EXAMPLE_RELIABILITY_ASSESSMENT_COMPLETE',
            'overall_score': reliability_metrics.overall_reliability_score,
            'assessment_duration': (reliability_assessor.assessment_end_time - reliability_assessor.assessment_start_time).total_seconds()
        })
        
        return reliability_metrics
        
    except Exception as e:
        logger.error(f"Reliability assessment example failed: {e}")
        raise


def display_assessment_results(metrics):
    """Display comprehensive assessment results in a readable format"""
    
    print("\n" + "="*80)
    print("🔍 KGoT KNOWLEDGE-DRIVEN RELIABILITY ASSESSMENT RESULTS")
    print("="*80)
    
    print(f"\n📊 OVERALL RELIABILITY SCORE: {metrics.overall_reliability_score:.3f}")
    print(f"📅 Assessment Timestamp: {metrics.assessment_timestamp}")
    
    print("\n📋 DETAILED RELIABILITY METRICS:")
    print("-" * 50)
    
    # Section 2.1 - Graph Store Module Reliability
    print(f"📈 Graph Store Reliability:")
    print(f"   ├─ Validation Success Rate: {metrics.graph_validation_success_rate:.3f}")
    print(f"   ├─ Consistency Score: {metrics.graph_consistency_score:.3f}")
    print(f"   └─ Performance Stability: {metrics.graph_performance_stability:.3f}")
    
    # Section 2.5 - Error Management Robustness
    print(f"\n🛡️  Error Management Robustness:")
    print(f"   ├─ Overall Effectiveness: {metrics.error_management_effectiveness:.3f}")
    print(f"   ├─ Recovery Success Rate: {metrics.error_recovery_success_rate:.3f}")
    print(f"   ├─ Escalation Efficiency: {metrics.error_escalation_efficiency:.3f}")
    print(f"   └─ Handling Coverage: {metrics.error_handling_coverage:.3f}")
    
    # Section 1.3 - Extraction Consistency
    print(f"\n🔄 Extraction Method Consistency:")
    print(f"   ├─ Overall Consistency: {metrics.extraction_method_consistency:.3f}")
    print(f"   ├─ Cross-Backend Agreement: {metrics.cross_backend_agreement:.3f}")
    print(f"   ├─ Query Result Stability: {metrics.query_result_stability:.3f}")
    print(f"   └─ Performance Variance: {metrics.extraction_performance_variance:.3f}")
    
    # Section 2.4 - Stress Resilience
    print(f"\n💪 Stress Resilience:")
    print(f"   ├─ Survival Rate: {metrics.stress_test_survival_rate:.3f}")
    print(f"   ├─ Resource Efficiency: {metrics.resource_efficiency_under_load:.3f}")
    print(f"   ├─ Degradation Tolerance: {metrics.performance_degradation_tolerance:.3f}")
    print(f"   └─ Concurrent Stability: {metrics.concurrent_operation_stability:.3f}")
    
    # Section 2.3.3 - Alita Integration Stability
    print(f"\n🤖 Alita Integration Stability:")
    print(f"   ├─ Integration Success Rate: {metrics.alita_integration_success_rate:.3f}")
    print(f"   ├─ Refinement Effectiveness: {metrics.alita_refinement_effectiveness:.3f}")
    print(f"   ├─ Error Recovery Rate: {metrics.alita_error_recovery_rate:.3f}")
    print(f"   └─ Cross-System Coordination: {metrics.cross_system_coordination_stability:.3f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    for i, recommendation in enumerate(metrics.recommendations, 1):
        print(f"{i}. {recommendation}")
    
    print("\n" + "="*80)


# Mock classes for demonstration purposes
class MockKnowledgeGraphInterface:
    """Mock Knowledge Graph Interface for demonstration"""
    
    def __init__(self, backend_type: str):
        self.backend_type = backend_type
        self.mock_data = f"Mock graph data for {backend_type} backend"
    
    async def getCurrentGraphState(self) -> str:
        """Mock graph state retrieval"""
        await asyncio.sleep(0.01)  # Simulate processing time
        return self.mock_data
    
    async def addTriplet(self, triplet: dict) -> bool:
        """Mock triplet addition"""
        await asyncio.sleep(0.01)  # Simulate processing time
        return True


class MockPerformanceOptimizer:
    """Mock Performance Optimizer for demonstration"""
    
    async def optimize_operation(self, operation_context: dict) -> dict:
        """Mock optimization operation"""
        await asyncio.sleep(0.01)  # Simulate processing time
        return {'optimization_result': 'mock_optimized', 'success': True}


class MockAlitaRefinementBridge:
    """Mock Alita Refinement Bridge for demonstration"""
    
    async def refine_with_alita(self, **kwargs) -> dict:
        """Mock Alita refinement"""
        await asyncio.sleep(0.01)  # Simulate processing time
        return {'success': True, 'refined_code': 'mock_refined_code'}
    
    async def handle_error_with_alita(self, **kwargs) -> dict:
        """Mock Alita error handling"""
        await asyncio.sleep(0.01)  # Simulate processing time
        return {'recovery_success': True, 'error_analysis': 'mock_analysis'}
    
    async def start_refinement_session(self, **kwargs) -> str:
        """Mock session start"""
        return "mock_session_id"
    
    async def perform_session_operation(self, **kwargs) -> dict:
        """Mock session operation"""
        return {'success': True}
    
    async def end_refinement_session(self, session_id: str) -> bool:
        """Mock session end"""
        return True
    
    async def iterative_improvement(self, **kwargs) -> dict:
        """Mock iterative improvement"""
        return {
            'improved_code': 'mock_improved_code',
            'improvement_metrics': {'performance_improvement': 0.1, 'readability_improvement': 0.2}
        }
    
    async def send_kgot_data_to_alita(self, **kwargs) -> dict:
        """Mock KGoT to Alita communication"""
        return {'success': True}
    
    async def receive_alita_feedback(self, **kwargs) -> dict:
        """Mock Alita feedback"""
        return {'success': True}
    
    async def coordinate_systems(self, **kwargs) -> dict:
        """Mock system coordination"""
        return {'coordination_success': True}


if __name__ == "__main__":
    """Run the reliability assessment example"""
    
    print("🚀 Starting KGoT Knowledge-Driven Reliability Assessment Example")
    print("   Task 19: Build KGoT Knowledge-Driven Reliability Assessor")
    print("   Implementing all KGoT research paper insights\n")
    
    try:
        # Run the assessment
        metrics = asyncio.run(run_comprehensive_reliability_assessment())
        
        print(f"\n✅ Assessment completed successfully!")
        print(f"Overall Reliability Score: {metrics.overall_reliability_score:.3f}")
        
    except Exception as e:
        print(f"\n❌ Assessment failed: {e}")
        sys.exit(1) 