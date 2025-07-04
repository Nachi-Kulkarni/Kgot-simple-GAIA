#!/usr/bin/env python3
"""
Comprehensive Test Suite for Sequential Error Resolution System

This test suite validates all components of Task 17d implementation:
- SequentialErrorClassifier functionality
- ErrorResolutionDecisionTree path selection
- CascadingFailureAnalyzer system coordination
- SequentialErrorResolutionSystem integration
- Configuration file loading and validation
- Integration with KGoT error management system

@module TestSequentialErrorResolution
@author Enhanced Alita-KGoT Development Team
@version 1.0.0
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import the sequential error resolution system
from alita_core.manager_agent.sequential_error_resolution import (
    SequentialErrorResolutionSystem,
    SequentialErrorClassifier,
    ErrorResolutionDecisionTree,
    CascadingFailureAnalyzer,
    EnhancedErrorContext,
    ErrorResolutionSession,
    ErrorComplexity,
    ErrorPattern,
    ResolutionStrategy,
    ErrorPreventionEngine,
    ErrorLearningSystem,
    create_sequential_error_resolution_system
)

# Import existing error management components
from kgot_core.error_management import (
    ErrorType, ErrorSeverity, ErrorContext,
    KGoTErrorManagementSystem
)

# Import LangChain Sequential Manager
from alita_core.manager_agent.langchain_sequential_manager import (
    LangChainSequentialManager, SystemType
)


class MockSequentialManager:
    """Mock LangChain Sequential Manager for testing"""
    
    def __init__(self):
        self.logger = logging.getLogger("MockSequentialManager")
        
    async def invoke_sequential_thinking(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock sequential thinking invocation"""
        return {
            'session_id': f"mock_session_{int(asyncio.get_event_loop().time())}",
            'conclusions': {
                'complexity_assessment': 'COMPOUND',
                'error_classification': 'Systematic error requiring structured analysis',
                'resolution_strategy': 'Multi-step recovery with validation',
                'system_coordination': 'Multi-system approach needed',
                'risk_factors': ['cascading failure potential', 'system dependencies']
            },
            'system_recommendations': {
                'primary_system': 'kgot',
                'coordination_needed': True,
                'recovery_steps': [
                    {'step': 1, 'action': 'isolate_affected_systems', 'timeout': 30},
                    {'step': 2, 'action': 'analyze_root_cause', 'timeout': 60},
                    {'step': 3, 'action': 'apply_corrective_measures', 'timeout': 120}
                ]
            }
        }


class MockKGoTErrorSystem:
    """Mock KGoT Error Management System for testing"""
    
    def __init__(self):
        self.logger = logging.getLogger("MockKGoTErrorSystem")
        
    async def handle_error(self, error: Exception, operation_context: str, **kwargs) -> tuple:
        """Mock error handling"""
        return {"status": "handled", "recovery_applied": True}, True


class TestSequentialErrorResolution(unittest.TestCase):
    """
    Comprehensive test suite for Sequential Error Resolution System
    """
    
    def setUp(self):
        """Set up test environment with mock dependencies"""
        self.mock_sequential_manager = MockSequentialManager()
        self.mock_kgot_system = MockKGoTErrorSystem()
        self.test_logger = logging.getLogger("TestSequentialErrorResolution")
        
        # Create test configuration
        self.test_config = {
            'max_resolution_attempts': 3,
            'default_timeout': 60,
            'enable_cascading_analysis': True,
            'enable_prevention_engine': True,
            'enable_learning_system': True,
            'confidence_threshold': 0.7
        }
    
    async def async_setUp(self):
        """Async setup for components that require async initialization"""
        self.error_classifier = SequentialErrorClassifier(
            self.mock_sequential_manager, 
            self.test_logger
        )
        
        self.decision_tree = ErrorResolutionDecisionTree(
            self.mock_sequential_manager,
            self.test_logger
        )
        
        self.cascading_analyzer = CascadingFailureAnalyzer(
            self.mock_sequential_manager,
            self.test_logger
        )
        
        self.resolution_system = SequentialErrorResolutionSystem(
            self.mock_sequential_manager,
            self.mock_kgot_system,
            self.test_config
        )
    
    def test_configuration_files_exist(self):
        """Test that all required configuration files exist and are valid"""
        config_files = [
            'error_resolution_patterns.json',
            'decision_trees.json',
            'system_dependencies.json'
        ]
        
        base_path = Path(__file__).parent
        
        for config_file in config_files:
            file_path = base_path / config_file
            self.assertTrue(file_path.exists(), f"Configuration file {config_file} not found")
            
            # Validate JSON structure
            with open(file_path, 'r') as f:
                try:
                    config_data = json.load(f)
                    self.assertIsInstance(config_data, dict, f"Invalid JSON structure in {config_file}")
                    self.test_logger.info(f"✅ Configuration file {config_file} is valid")
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSON in {config_file}: {e}")
    
    async def test_error_classification_workflow(self):
        """Test the complete error classification workflow"""
        await self.async_setUp()
        
        # Create test error
        test_error = ValueError("Test error for classification")
        operation_context = "alita_mcp_creation_test"
        
        # Test classification
        enhanced_context = await self.error_classifier.classify_error_with_sequential_thinking(
            test_error, operation_context
        )
        
        # Validate enhanced context
        self.assertIsInstance(enhanced_context, EnhancedErrorContext)
        self.assertIsInstance(enhanced_context.error_complexity, ErrorComplexity)
        self.assertIsInstance(enhanced_context.error_pattern, ErrorPattern)
        self.assertIsInstance(enhanced_context.system_impact_map, dict)
        
        self.test_logger.info("✅ Error classification workflow completed successfully")
    
    async def test_decision_tree_resolution_path(self):
        """Test decision tree resolution path selection"""
        await self.async_setUp()
        
        # Create enhanced error context
        enhanced_context = EnhancedErrorContext(
            error_id="test_error_001",
            error_type=ErrorType.SYNTAX_ERROR,
            severity=ErrorSeverity.MEDIUM,
            timestamp=datetime.now(),
            original_operation="test_operation",
            error_message="Test syntax error",
            error_complexity=ErrorComplexity.COMPOUND,
            error_pattern=ErrorPattern.RECURRING
        )
        enhanced_context.system_impact_map = {SystemType.ALITA: 0.8, SystemType.KGOT: 0.6}
        
        # Test path selection
        resolution_path = await self.decision_tree.select_resolution_path(enhanced_context)
        
        # Validate resolution path
        self.assertIsInstance(resolution_path, dict)
        self.assertIn('strategy', resolution_path)
        self.assertIn('confidence', resolution_path)
        self.assertIn('steps', resolution_path)
        
        self.test_logger.info("✅ Decision tree resolution path selection completed")
    
    async def test_cascading_failure_analysis(self):
        """Test cascading failure analysis functionality"""
        await self.async_setUp()
        
        # Create test error context with system impact
        enhanced_context = EnhancedErrorContext(
            error_id="test_cascading_001",
            error_type=ErrorType.SYSTEM_ERROR,
            severity=ErrorSeverity.HIGH,
            timestamp=datetime.now(),
            original_operation="multi_system_operation",
            error_message="System coordination failure",
            error_complexity=ErrorComplexity.CASCADING
        )
        enhanced_context.system_impact_map = {
            SystemType.ALITA: 0.9,
            SystemType.KGOT: 0.8,
            SystemType.VALIDATION: 0.6
        }
        
        # Test cascading analysis
        cascading_analysis = await self.cascading_analyzer.analyze_cascading_potential(enhanced_context)
        
        # Validate analysis results
        self.assertIsInstance(cascading_analysis, dict)
        self.assertIn('risk_assessment', cascading_analysis)
        self.assertIn('propagation_paths', cascading_analysis)
        self.assertIn('mitigation_plan', cascading_analysis)
        
        self.test_logger.info("✅ Cascading failure analysis completed successfully")
    
    async def test_complete_resolution_workflow(self):
        """Test the complete end-to-end error resolution workflow"""
        await self.async_setUp()
        
        # Create test error
        test_error = ConnectionError("API connection failed")
        operation_context = "kgot_knowledge_extraction"
        
        # Test complete resolution workflow
        resolution_result = await self.resolution_system.resolve_error_with_sequential_thinking(
            test_error, operation_context
        )
        
        # Validate resolution result
        self.assertIsInstance(resolution_result, dict)
        self.assertIn('session_id', resolution_result)
        self.assertIn('resolution_status', resolution_result)
        self.assertIn('recovery_applied', resolution_result)
        self.assertIn('error_context', resolution_result)
        
        # Validate resolution status
        self.assertIn(resolution_result['resolution_status'], ['resolved', 'partially_resolved', 'escalated'])
        
        self.test_logger.info("✅ Complete resolution workflow completed successfully")
    
    async def test_prevention_engine(self):
        """Test error prevention engine functionality"""
        prevention_engine = ErrorPreventionEngine(self.mock_sequential_manager, self.test_logger)
        
        operation_plan = """
        Plan to execute multi-system operation involving:
        1. Alita MCP creation with complex requirements
        2. KGoT knowledge graph operations
        3. Cross-system validation
        4. High-frequency API calls to OpenRouter
        """
        
        risk_assessment = await prevention_engine.assess_operation_risk(operation_plan)
        
        # Validate risk assessment
        self.assertIsInstance(risk_assessment, dict)
        self.assertIn('risk_level', risk_assessment)
        self.assertIn('potential_issues', risk_assessment)
        self.assertIn('prevention_recommendations', risk_assessment)
        
        self.test_logger.info("✅ Error prevention engine test completed")
    
    async def test_learning_system(self):
        """Test error learning system functionality"""
        learning_system = ErrorLearningSystem(self.test_logger)
        
        # Create mock resolution session
        enhanced_context = EnhancedErrorContext(
            error_id="test_learning_001",
            error_type=ErrorType.API_ERROR,
            severity=ErrorSeverity.MEDIUM,
            timestamp=datetime.now(),
            original_operation="test_learning_operation",
            error_message="Rate limit exceeded"
        )
        
        session = ErrorResolutionSession(
            session_id="learning_session_001",
            error_context=enhanced_context,
            resolution_strategy=ResolutionStrategy.SEQUENTIAL_ANALYSIS,
            status="completed",
            confidence_score=0.85
        )
        
        session.learning_outcomes = {
            'pattern_recognized': True,
            'resolution_effectiveness': 0.9,
            'new_insights': ['Rate limiting patterns', 'Exponential backoff optimization']
        }
        
        learning_result = await learning_system.learn_from_resolution(session)
        
        # Validate learning result
        self.assertIsInstance(learning_result, dict)
        self.assertIn('patterns_updated', learning_result)
        self.assertIn('insights_gained', learning_result)
        
        self.test_logger.info("✅ Error learning system test completed")
    
    async def test_system_integration(self):
        """Test integration with existing KGoT error management system"""
        await self.async_setUp()
        
        # Test system creation factory function
        integrated_system = create_sequential_error_resolution_system(
            self.mock_sequential_manager,
            self.mock_kgot_system,
            self.test_config
        )
        
        # Validate system creation
        self.assertIsInstance(integrated_system, SequentialErrorResolutionSystem)
        
        # Test fallback to KGoT system
        test_error = RuntimeError("Unhandled system error")
        fallback_result, success = await integrated_system._fallback_to_kgot_system(
            test_error, "fallback_test_context"
        )
        
        # Validate fallback behavior
        self.assertIsInstance(fallback_result, dict)
        self.assertIsInstance(success, bool)
        
        self.test_logger.info("✅ System integration test completed")
    
    def test_data_structures_and_enums(self):
        """Test all data structures and enumerations"""
        # Test ErrorComplexity enum
        self.assertEqual(len(ErrorComplexity), 4)
        self.assertIn(ErrorComplexity.SIMPLE, ErrorComplexity)
        self.assertIn(ErrorComplexity.CASCADING, ErrorComplexity)
        
        # Test ErrorPattern enum
        self.assertEqual(len(ErrorPattern), 4)
        self.assertIn(ErrorPattern.RECURRING, ErrorPattern)
        self.assertIn(ErrorPattern.NOVEL, ErrorPattern)
        
        # Test ResolutionStrategy enum
        self.assertEqual(len(ResolutionStrategy), 5)
        self.assertIn(ResolutionStrategy.SEQUENTIAL_ANALYSIS, ResolutionStrategy)
        self.assertIn(ResolutionStrategy.CASCADING_RECOVERY, ResolutionStrategy)
        
        # Test EnhancedErrorContext
        enhanced_context = EnhancedErrorContext(
            error_id="test_structure_001",
            error_type=ErrorType.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            timestamp=datetime.now(),
            original_operation="structure_test",
            error_message="Test message"
        )
        
        # Test dictionary conversion
        context_dict = enhanced_context.to_enhanced_dict()
        self.assertIsInstance(context_dict, dict)
        self.assertIn('error_id', context_dict)
        self.assertIn('error_complexity', context_dict)
        
        self.test_logger.info("✅ Data structures and enums test completed")


async def run_comprehensive_test_suite():
    """
    Run the complete test suite with async support
    """
    print("🧪 Starting Comprehensive Sequential Error Resolution Test Suite")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_suite = TestSequentialErrorResolution()
    test_suite.setUp()
    
    # List of async tests to run
    async_tests = [
        ('Configuration Files Validation', test_suite.test_configuration_files_exist),
        ('Error Classification Workflow', test_suite.test_error_classification_workflow),
        ('Decision Tree Resolution Path', test_suite.test_decision_tree_resolution_path),
        ('Cascading Failure Analysis', test_suite.test_cascading_failure_analysis),
        ('Complete Resolution Workflow', test_suite.test_complete_resolution_workflow),
        ('Prevention Engine', test_suite.test_prevention_engine),
        ('Learning System', test_suite.test_learning_system),
        ('System Integration', test_suite.test_system_integration),
        ('Data Structures and Enums', test_suite.test_data_structures_and_enums)
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_name, test_method in async_tests:
        try:
            print(f"\n🔍 Running: {test_name}")
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            print(f"✅ PASSED: {test_name}")
            passed_tests += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {str(e)}")
            failed_tests += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {(passed_tests / (passed_tests + failed_tests)) * 100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Sequential Error Resolution System is fully functional.")
        return True
    else:
        print(f"\n⚠️  {failed_tests} tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    """Run the test suite when executed directly"""
    success = asyncio.run(run_comprehensive_test_suite())
    exit(0 if success else 1) 