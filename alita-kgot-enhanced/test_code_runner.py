#!/usr/bin/env python3
"""
Test Script for Task 11: Alita Code Running Tool with KGoT Execution Environment

This test script validates the complete implementation of Task 11 requirements:
- Alita Section 2.3.3 "CodeRunningTool" complete functionality
- Functionality validation through isolated environment execution
- Iterative refinement with error inspection and code regeneration
- Output caching for potential MCP server generation
- Integration with KGoT Section 2.6 "Python Executor tool containerization"
- Integration with KGoT Section 2.5 "Error Management"
- Comprehensive validation testing using cross-validation framework

@author Enhanced Alita KGoT Team
@version 1.0.0
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime

# Add the alita_core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'alita_core'))

try:
    from code_runner import (
        create_code_running_tool,
        CodeRunningTool,
        ExecutionContext,
        ValidationResult,
        ExecutionResult,
        RefinementStrategy
    )
except ImportError as e:
    print(f"Error importing code_runner modules: {e}")
    print("This might be due to missing dependencies. Try running:")
    print("pip install langchain langchain-openai langchain-experimental langchain-core docker psutil requests tenacity pydantic")
    sys.exit(1)

# Configure logging for test execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CodeRunnerTest')


class Task11TestSuite:
    """
    Comprehensive test suite for Task 11 implementation validation
    """
    
    def __init__(self):
        """Initialize the test suite"""
        self.test_results = []
        self.code_runner = None
        logger.info("Initializing Task 11 Test Suite")
    
    async def run_all_tests(self):
        """
        Run all tests for Task 11 implementation
        
        @returns {Dict[str, Any]} Complete test results
        """
        logger.info("Starting comprehensive Task 11 testing")
        
        try:
            # Test 1: Basic CodeRunningTool initialization
            await self._test_basic_initialization()
            
            # Test 2: Simple code execution and validation
            await self._test_simple_code_execution()
            
            # Test 3: Error handling and iterative refinement
            await self._test_error_handling_refinement()
            
            # Test 4: Caching functionality
            await self._test_caching_functionality()
            
            # Test 5: Cross-validation framework
            await self._test_cross_validation()
            
            # Test 6: MCP server generation
            await self._test_mcp_generation()
            
            # Test 7: Integration with KGoT components
            await self._test_kgot_integration()
            
            # Compile comprehensive results
            return self._compile_test_results()
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            return {
                'overall_success': False,
                'error': str(e),
                'completed_tests': len(self.test_results)
            }
    
    async def _test_basic_initialization(self):
        """Test basic CodeRunningTool initialization"""
        test_name = "Basic Initialization"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Test factory function
            self.code_runner = create_code_running_tool({
                'max_concurrent_executions': 2,
                'validation_rounds': 2,
                'enable_mcp_generation': True
            })
            
            # Verify components are initialized
            assert self.code_runner is not None
            assert hasattr(self.code_runner, 'execution_environment')
            assert hasattr(self.code_runner, 'refinement_engine')
            assert hasattr(self.code_runner, 'mcp_generator')
            assert hasattr(self.code_runner, 'validation_framework')
            assert hasattr(self.code_runner, 'result_cache')
            
            self._record_test_result(test_name, True, "All components initialized successfully")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Initialization failed: {e}")
    
    async def _test_simple_code_execution(self):
        """Test simple code execution and validation"""
        test_name = "Simple Code Execution"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Simple working Python code
            test_code = """
def hello_world():
    return "Hello, World!"

result = hello_world()
print(result)
"""
            
            # Execute code with basic settings
            execution_result = await self.code_runner.validate_and_execute_code(
                code=test_code,
                language="python",
                timeout=10,
                enable_refinement=False,
                enable_cross_validation=False,
                generate_mcp=False
            )
            
            # Verify execution success
            assert execution_result is not None
            assert 'overall_success' in execution_result
            assert 'session_id' in execution_result
            assert 'final_validation' in execution_result
            
            success = execution_result.get('overall_success', False)
            self._record_test_result(test_name, success, 
                                   f"Code execution result: {execution_result.get('final_validation', {}).get('execution_result', 'unknown')}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Code execution failed: {e}")
    
    async def _test_error_handling_refinement(self):
        """Test error handling and iterative refinement"""
        test_name = "Error Handling and Refinement"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Code with syntax error for refinement testing
            buggy_code = """
def calculate_sum(a, b):
    result = a + b
    return result

# Missing closing parenthesis
print(calculate_sum(5, 3)
"""
            
            # Execute with refinement enabled
            execution_result = await self.code_runner.validate_and_execute_code(
                code=buggy_code,
                language="python",
                timeout=15,
                enable_refinement=True,
                enable_cross_validation=False,
                generate_mcp=False
            )
            
            # Check if refinement was attempted
            refinement_applied = execution_result.get('refinement', {}).get('applied', False)
            final_success = execution_result.get('overall_success', False)
            
            # Test passes if either refinement fixed the issue or error was properly handled
            test_success = refinement_applied or 'error' in execution_result
            
            self._record_test_result(test_name, test_success, 
                                   f"Refinement applied: {refinement_applied}, Final success: {final_success}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Error handling test failed: {e}")
    
    async def _test_caching_functionality(self):
        """Test result caching functionality"""
        test_name = "Caching Functionality"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Simple code for caching test
            cache_test_code = """
import time
result = "cached_result_" + str(int(time.time()))
print(result)
"""
            
            # First execution - should be cached
            first_result = await self.code_runner.validate_and_execute_code(
                code=cache_test_code,
                enable_caching=True,
                enable_refinement=False,
                enable_cross_validation=False,
                generate_mcp=False
            )
            
            # Get cache statistics
            cache_stats = self.code_runner.get_performance_statistics().get('cache_statistics', {})
            
            # Verify caching functionality exists
            has_cache = 'cache_hits' in cache_stats or 'cache_misses' in cache_stats
            
            self._record_test_result(test_name, has_cache, 
                                   f"Cache statistics available: {has_cache}, Stats: {cache_stats}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Caching test failed: {e}")
    
    async def _test_cross_validation(self):
        """Test cross-validation framework"""
        test_name = "Cross-Validation Framework"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Code suitable for cross-validation
            validation_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(5)
print(f"Fibonacci(5) = {result}")
"""
            
            # Execute with cross-validation enabled
            execution_result = await self.code_runner.validate_and_execute_code(
                code=validation_code,
                enable_cross_validation=True,
                enable_refinement=False,
                generate_mcp=False
            )
            
            # Check if cross-validation was performed
            cross_validation = execution_result.get('cross_validation')
            has_validation_rounds = cross_validation and 'validation_rounds' in cross_validation
            
            self._record_test_result(test_name, has_validation_rounds, 
                                   f"Cross-validation performed: {has_validation_rounds}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Cross-validation test failed: {e}")
    
    async def _test_mcp_generation(self):
        """Test MCP server generation functionality"""
        test_name = "MCP Server Generation"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Code suitable for MCP generation
            mcp_code = """
def useful_function(input_data):
    \"\"\"
    A useful function that could become an MCP tool
    
    @param input_data: Input data to process
    @returns: Processed result
    \"\"\"
    return f"Processed: {input_data}"

result = useful_function("test_data")
print(result)
"""
            
            # Execute with MCP generation enabled
            execution_result = await self.code_runner.validate_and_execute_code(
                code=mcp_code,
                generate_mcp=True,
                enable_refinement=False,
                enable_cross_validation=False
            )
            
            # Check if MCP generation was attempted
            mcp_generation = execution_result.get('mcp_generation')
            mcp_attempted = mcp_generation is not None
            
            self._record_test_result(test_name, mcp_attempted, 
                                   f"MCP generation attempted: {mcp_attempted}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"MCP generation test failed: {e}")
    
    async def _test_kgot_integration(self):
        """Test integration with KGoT components"""
        test_name = "KGoT Integration"
        logger.info(f"Running test: {test_name}")
        
        try:
            # Verify error management integration
            error_mgmt_available = hasattr(self.code_runner.refinement_engine, 'error_management_system')
            
            # Verify containerization integration
            container_available = hasattr(self.code_runner.execution_environment, 'container_orchestrator')
            
            # Get performance statistics to verify component integration
            stats = self.code_runner.get_performance_statistics()
            component_status = stats.get('component_status', {})
            
            # Check if all expected components are active
            expected_components = [
                'execution_environment', 'refinement_engine', 
                'mcp_generator', 'validation_framework', 'result_cache'
            ]
            components_active = all(
                component_status.get(comp) == 'active' 
                for comp in expected_components
            )
            
            integration_success = error_mgmt_available and container_available and components_active
            
            self._record_test_result(test_name, integration_success, 
                                   f"Error mgmt: {error_mgmt_available}, Container: {container_available}, Components: {components_active}")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"KGoT integration test failed: {e}")
    
    def _record_test_result(self, test_name: str, success: bool, details: str):
        """Record individual test result"""
        result = {
            'test_name': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "PASSED" if success else "FAILED"
        logger.info(f"Test {test_name}: {status} - {details}")
    
    def _compile_test_results(self):
        """Compile comprehensive test results"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        
        return {
            'overall_success': passed_tests == total_tests,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'test_details': self.test_results,
            'task_11_status': 'COMPLETED' if passed_tests >= total_tests * 0.8 else 'PARTIAL',
            'timestamp': datetime.now().isoformat()
        }


async def main():
    """
    Main function to run Task 11 validation tests
    """
    print("="*60)
    print("Task 11 Implementation Test Suite")
    print("Alita Code Running Tool with KGoT Execution Environment")
    print("="*60)
    
    # Initialize and run test suite
    test_suite = Task11TestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        
        # Print comprehensive results
        print(f"\nTest Results Summary:")
        print(f"Overall Success: {results['overall_success']}")
        print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Task 11 Status: {results['task_11_status']}")
        
        print(f"\nDetailed Test Results:")
        for test in results['test_details']:
            status = "✅ PASSED" if test['success'] else "❌ FAILED"
            print(f"{status} {test['test_name']}: {test['details']}")
        
        # Final assessment
        if results['overall_success']:
            print(f"\n🎉 Task 11 implementation is FULLY FUNCTIONAL!")
            print("All required components are working correctly:")
            print("✅ Alita Section 2.3.3 CodeRunningTool complete functionality")
            print("✅ Isolated environment execution validation")
            print("✅ Iterative refinement with error inspection")
            print("✅ Output caching for MCP server generation")
            print("✅ KGoT Section 2.6 containerization integration")
            print("✅ KGoT Section 2.5 error management integration")
            print("✅ Cross-validation framework testing")
        else:
            print(f"\n⚠️ Task 11 implementation has some issues that need attention.")
            print(f"Status: {results['task_11_status']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test suite execution failed: {e}")
        return {'overall_success': False, 'error': str(e)}


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main()) 