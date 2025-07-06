"""
Comprehensive Testing Suite for Unified System Controller

This module provides extensive testing coverage for all components of the unified system,
including unit tests, integration tests, performance tests, and failure scenarios.

Author: Advanced AI Development Team
Version: 1.0.0
"""

import pytest
import asyncio
import json
import time
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import the modules being tested
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_system_controller import (
    UnifiedSystemController, TaskComplexity, RoutingStrategy, 
    SystemStatus, TaskContext, SystemMetrics, ExecutionResult
)
from sequential_thinking_mcp_integration import (
    SequentialThinkingMCPIntegration, ThinkingMode
)
from shared_state_utilities import (
    EnhancedSharedStateManager, StateScope, RealTimeStateStreamer
)
from advanced_monitoring_system import (
    AdvancedMonitoringSystem, AlertManager, PerformanceAnalyzer
)
from load_balancing_system import (
    AdaptiveLoadBalancer, CircuitBreaker, LoadBalancingStrategy
)
from error_handling_system import (
    ComprehensiveErrorHandler, ErrorCategory, ErrorSeverity
)
from enhanced_logging_system import (
    StructuredLogger, LogLevel, LogCategory
)

class TestUnifiedSystemController:
    """Test suite for the main UnifiedSystemController class"""
    
    @pytest.fixture
    async def controller(self):
        """Create a UnifiedSystemController instance for testing"""
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        
        # Create controller with mocked dependencies
        controller = UnifiedSystemController(
            alita_base_url="http://localhost:3001",
            kgot_base_url="http://localhost:8000",
            redis_host="localhost",
            redis_port=6379,
            redis_password="test_password"
        )
        
        # Replace Redis client with mock
        controller.state_manager.redis_client = mock_redis
        controller.monitoring_system.state_manager.redis_client = mock_redis
        
        yield controller
        
        # Cleanup
        if hasattr(controller, 'cleanup'):
            await controller.cleanup()
    
    @pytest.mark.asyncio
    async def test_controller_initialization(self, controller):
        """Test proper initialization of the controller"""
        assert controller.alita_base_url == "http://localhost:3001"
        assert controller.kgot_base_url == "http://localhost:8000"
        assert controller.status == SystemStatus.INITIALIZING
        assert isinstance(controller.logger, StructuredLogger)
        assert isinstance(controller.state_manager, EnhancedSharedStateManager)
        assert isinstance(controller.monitoring_system, AdvancedMonitoringSystem)
        assert isinstance(controller.load_balancer, AdaptiveLoadBalancer)
        assert isinstance(controller.error_handler, ComprehensiveErrorHandler)
    
    @pytest.mark.asyncio
    async def test_system_startup(self, controller):
        """Test system startup process"""
        with patch.object(controller, '_check_system_health', return_value=True):
            with patch.object(controller, '_initialize_performance_baseline', return_value=None):
                await controller.startup()
                assert controller.status == SystemStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_task_complexity_analysis(self, controller):
        """Test task complexity analysis"""
        # Simple task
        simple_context = TaskContext(
            task_id="test_simple",
            task_type="query",
            content="What is 2+2?",
            metadata={}
        )
        complexity = controller._analyze_task_complexity(simple_context)
        assert complexity in [TaskComplexity.LOW, TaskComplexity.MEDIUM]
        
        # Complex task
        complex_context = TaskContext(
            task_id="test_complex",
            task_type="reasoning",
            content="Analyze the economic implications of quantum computing on cryptocurrency markets",
            metadata={"requires_graph": True, "multi_step": True}
        )
        complexity = controller._analyze_task_complexity(complex_context)
        assert complexity in [TaskComplexity.HIGH, TaskComplexity.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_routing_strategy_determination(self, controller):
        """Test routing strategy determination logic"""
        # Test different task complexities
        low_complexity_context = TaskContext(
            task_id="test_low", task_type="query", content="Simple question"
        )
        strategy = controller._determine_routing_strategy(low_complexity_context, TaskComplexity.LOW)
        assert strategy in [RoutingStrategy.ALITA_FIRST, RoutingStrategy.KGOT_FIRST]
        
        # Test high complexity requiring hybrid approach
        high_complexity_context = TaskContext(
            task_id="test_high", task_type="reasoning", content="Complex reasoning task",
            metadata={"requires_graph": True, "multi_step": True}
        )
        strategy = controller._determine_routing_strategy(high_complexity_context, TaskComplexity.HIGH)
        assert strategy in [RoutingStrategy.HYBRID, RoutingStrategy.PARALLEL]
    
    @pytest.mark.asyncio
    async def test_task_processing_success(self, controller):
        """Test successful task processing"""
        task_context = TaskContext(
            task_id="test_success",
            task_type="query",
            content="Test question",
            metadata={}
        )
        
        # Mock successful system calls
        with patch.object(controller, '_call_alita_system', return_value={
            "success": True, "response": "Test response", "execution_time_ms": 100
        }):
            with patch.object(controller, '_update_performance_metrics'):
                result = await controller.process_task(task_context)
                
                assert result.success is True
                assert result.response == "Test response"
                assert result.system_used == "alita"
    
    @pytest.mark.asyncio
    async def test_task_processing_with_fallback(self, controller):
        """Test task processing with system fallback"""
        task_context = TaskContext(
            task_id="test_fallback",
            task_type="query", 
            content="Test question",
            metadata={}
        )
        
        # Mock Alita failure and KGoT success
        alita_error = Exception("Alita system unavailable")
        with patch.object(controller, '_call_alita_system', side_effect=alita_error):
            with patch.object(controller, '_call_kgot_system', return_value={
                "success": True, "response": "KGoT response", "execution_time_ms": 200
            }):
                with patch.object(controller.error_handler, 'handle_error', return_value={
                    "handled": True, "recovery_recommended": True
                }):
                    result = await controller.process_task(task_context)
                    
                    assert result.success is True
                    assert result.system_used == "kgot"
                    assert "fallback" in result.metadata

class TestSequentialThinkingIntegration:
    """Test suite for Sequential Thinking MCP Integration"""
    
    @pytest.fixture
    def st_integration(self):
        """Create SequentialThinkingMCPIntegration instance for testing"""
        mock_logger = Mock(spec=StructuredLogger)
        return SequentialThinkingMCPIntegration(
            mcp_endpoint="http://localhost:8080",
            logger=mock_logger
        )
    
    @pytest.mark.asyncio
    async def test_routing_analysis(self, st_integration):
        """Test routing analysis functionality"""
        task_context = TaskContext(
            task_id="test_routing",
            task_type="query",
            content="Test question",
            metadata={}
        )
        
        # Mock HTTP response
        mock_response = {
            "success": True,
            "analysis": {
                "recommended_system": "alita",
                "confidence": 0.85,
                "reasoning": "Task is suitable for Alita's capabilities"
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await st_integration.analyze_task_routing(task_context, {})
            
            assert result["recommended_system"] == "alita"
            assert result["confidence"] == 0.85
    
    @pytest.mark.asyncio
    async def test_system_coordination(self, st_integration):
        """Test system coordination analysis"""
        coordination_context = {
            "alita_status": "healthy",
            "kgot_status": "degraded",
            "current_load": {"alita": 0.7, "kgot": 0.9}
        }
        
        mock_response = {
            "success": True,
            "coordination": {
                "primary_system": "alita",
                "load_balancing": "redirect_to_alita",
                "reasoning": "KGoT is under high load"
            }
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aenter__.return_value.status = 200
            
            result = await st_integration.analyze_system_coordination(coordination_context)
            
            assert result["primary_system"] == "alita"
            assert result["load_balancing"] == "redirect_to_alita"

class TestSharedStateManager:
    """Test suite for Enhanced Shared State Manager"""
    
    @pytest.fixture
    def state_manager(self):
        """Create EnhancedSharedStateManager instance for testing"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        
        mock_logger = Mock(spec=StructuredLogger)
        
        manager = EnhancedSharedStateManager(
            redis_host="localhost",
            redis_port=6379,
            redis_password="test_password",
            logger=mock_logger
        )
        manager.redis_client = mock_redis
        
        return manager
    
    @pytest.mark.asyncio
    async def test_state_operations(self, state_manager):
        """Test basic state operations"""
        # Test set state
        await state_manager.set_state("test_key", {"value": "test"}, StateScope.SESSION)
        
        # Verify Redis was called correctly
        state_manager.redis_client.hset.assert_called()
        
        # Test get state
        state_manager.redis_client.hget.return_value = json.dumps({"value": "test"})
        result = await state_manager.get_state("test_key", StateScope.SESSION)
        
        assert result["value"] == "test"
    
    @pytest.mark.asyncio
    async def test_distributed_locking(self, state_manager):
        """Test distributed locking mechanism"""
        lock_key = "test_lock"
        
        # Mock successful lock acquisition
        state_manager.redis_client.set.return_value = True
        
        async with state_manager.lock_manager.acquire_lock(lock_key, timeout=5.0):
            # Verify lock was acquired
            state_manager.redis_client.set.assert_called()
            
            # Test lock release
            pass  # Lock should be released automatically
        
        # Verify lock was released
        state_manager.redis_client.delete.assert_called()

class TestAdvancedMonitoring:
    """Test suite for Advanced Monitoring System"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create AdvancedMonitoringSystem instance for testing"""
        mock_state_manager = Mock(spec=EnhancedSharedStateManager)
        mock_logger = Mock(spec=StructuredLogger)
        
        return AdvancedMonitoringSystem(
            state_manager=mock_state_manager,
            logger=mock_logger
        )
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, monitoring_system):
        """Test metrics collection functionality"""
        test_metrics = SystemMetrics(
            alita_response_time=100.0,
            kgot_response_time=150.0,
            alita_success_rate=0.95,
            kgot_success_rate=0.90,
            total_requests=1000,
            error_rate=0.05
        )
        
        await monitoring_system.collect_metrics(test_metrics)
        
        # Verify metrics were stored
        monitoring_system.state_manager.set_state.assert_called()
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, monitoring_system):
        """Test alert generation for critical conditions"""
        # Simulate high error rate
        test_metrics = SystemMetrics(
            alita_response_time=100.0,
            kgot_response_time=150.0,
            alita_success_rate=0.70,  # Below threshold
            kgot_success_rate=0.90,
            total_requests=1000,
            error_rate=0.30  # High error rate
        )
        
        await monitoring_system.collect_metrics(test_metrics)
        
        # Check if alerts were generated
        alerts = await monitoring_system.alert_manager.get_active_alerts()
        assert len(alerts) > 0

class TestLoadBalancer:
    """Test suite for Adaptive Load Balancer"""
    
    @pytest.fixture
    def load_balancer(self):
        """Create AdaptiveLoadBalancer instance for testing"""
        mock_logger = Mock(spec=StructuredLogger)
        return AdaptiveLoadBalancer(logger=mock_logger)
    
    def test_strategy_selection(self, load_balancer):
        """Test load balancing strategy selection"""
        # Test round-robin strategy
        load_balancer.current_strategy = LoadBalancingStrategy.ROUND_ROBIN
        
        system1 = load_balancer.select_system()
        system2 = load_balancer.select_system()
        
        # Should alternate between systems
        assert system1 != system2 or len(load_balancer.systems) == 1
    
    def test_circuit_breaker(self, load_balancer):
        """Test circuit breaker functionality"""
        circuit = CircuitBreaker("test_system", logger=Mock())
        
        # Simulate failures to trigger circuit breaker
        for _ in range(6):  # Exceed failure threshold
            circuit.record_failure()
        
        assert circuit.is_open()
        
        # Test circuit breaker prevents calls
        with pytest.raises(Exception):
            circuit.call(lambda: "test")

class TestErrorHandling:
    """Test suite for Comprehensive Error Handler"""
    
    @pytest.fixture
    def error_handler(self):
        """Create ComprehensiveErrorHandler instance for testing"""
        mock_logger = Mock(spec=StructuredLogger)
        mock_state_manager = Mock(spec=EnhancedSharedStateManager)
        
        return ComprehensiveErrorHandler(
            logger=mock_logger,
            state_manager=mock_state_manager
        )
    
    @pytest.mark.asyncio
    async def test_error_categorization(self, error_handler):
        """Test error categorization logic"""
        # Test network error
        network_error = ConnectionError("Network unavailable")
        category = error_handler.categorize_error(network_error)
        assert category == ErrorCategory.NETWORK_ERROR
        
        # Test authentication error
        auth_error = PermissionError("Unauthorized access")
        category = error_handler.categorize_error(auth_error)
        assert category == ErrorCategory.AUTHENTICATION_ERROR
        
        # Test validation error
        validation_error = ValueError("Invalid input")
        category = error_handler.categorize_error(validation_error)
        assert category == ErrorCategory.VALIDATION_ERROR
    
    @pytest.mark.asyncio
    async def test_error_handling_flow(self, error_handler):
        """Test complete error handling flow"""
        test_exception = ConnectionError("Network timeout")
        
        result = await error_handler.handle_error(
            exception=test_exception,
            component="test_component",
            operation="test_operation",
            correlation_id="test_correlation"
        )
        
        assert "error_id" in result
        assert result["category"] == ErrorCategory.NETWORK_ERROR.value
        assert "handler_result" in result
        assert "fallback_result" in result

class TestPerformanceScenarios:
    """Performance and stress testing scenarios"""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self):
        """Test processing multiple tasks concurrently"""
        # Create controller instance
        controller = UnifiedSystemController(
            alita_base_url="http://localhost:3001",
            kgot_base_url="http://localhost:8000"
        )
        
        # Mock system calls to avoid actual network requests
        with patch.object(controller, '_call_alita_system', return_value={
            "success": True, "response": "Mock response", "execution_time_ms": 50
        }):
            with patch.object(controller, '_call_kgot_system', return_value={
                "success": True, "response": "Mock response", "execution_time_ms": 75
            }):
                
                # Create multiple tasks
                tasks = []
                for i in range(10):
                    task_context = TaskContext(
                        task_id=f"concurrent_task_{i}",
                        task_type="query",
                        content=f"Test question {i}",
                        metadata={}
                    )
                    tasks.append(controller.process_task(task_context))
                
                # Process all tasks concurrently
                start_time = time.time()
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                # Verify all tasks completed successfully
                assert len(results) == 10
                assert all(result.success for result in results)
                
                # Verify reasonable performance (should be much faster than sequential)
                total_time = end_time - start_time
                assert total_time < 2.0  # Should complete in under 2 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage remains stable under load"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        controller = UnifiedSystemController(
            alita_base_url="http://localhost:3001",
            kgot_base_url="http://localhost:8000"
        )
        
        # Mock system calls
        with patch.object(controller, '_call_alita_system', return_value={
            "success": True, "response": "Mock response", "execution_time_ms": 50
        }):
            
            # Process many tasks to test memory stability
            for i in range(100):
                task_context = TaskContext(
                    task_id=f"memory_test_{i}",
                    task_type="query",
                    content="Memory test question",
                    metadata={}
                )
                await controller.process_task(task_context)
                
                # Force garbage collection periodically
                if i % 20 == 0:
                    gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

class TestIntegrationScenarios:
    """Integration testing scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self):
        """Test complete system integration with all components"""
        # This test would require actual system instances running
        # For now, we'll test the integration with mocked services
        
        controller = UnifiedSystemController(
            alita_base_url="http://localhost:3001",
            kgot_base_url="http://localhost:8000"
        )
        
        # Mock all external dependencies
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful responses from all systems
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "success": True,
                "response": "Integration test response"
            })
            mock_post.return_value.__aenter__.return_value = mock_response
            
            task_context = TaskContext(
                task_id="integration_test",
                task_type="complex_reasoning",
                content="Complex integration test question",
                metadata={"requires_graph": True}
            )
            
            result = await controller.process_task(task_context)
            
            assert result.success is True
            assert "integration test response" in result.response.lower()

# Test fixtures and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    mock = Mock()
    mock.ping.return_value = True
    mock.hset.return_value = True
    mock.hget.return_value = None
    mock.delete.return_value = True
    mock.set.return_value = True
    mock.expire.return_value = True
    return mock

@pytest.fixture
def sample_task_context():
    """Sample task context for testing"""
    return TaskContext(
        task_id="sample_test",
        task_type="query",
        content="Sample test question",
        metadata={"test": True}
    )

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_task_routing_performance(self, benchmark):
        """Benchmark task routing performance"""
        controller = UnifiedSystemController(
            alita_base_url="http://localhost:3001",
            kgot_base_url="http://localhost:8000"
        )
        
        task_context = TaskContext(
            task_id="benchmark_test",
            task_type="query",
            content="Benchmark test question",
            metadata={}
        )
        
        def routing_operation():
            complexity = controller._analyze_task_complexity(task_context)
            strategy = controller._determine_routing_strategy(task_context, complexity)
            return complexity, strategy
        
        result = benchmark(routing_operation)
        assert result is not None
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_state_management_performance(self, benchmark):
        """Benchmark state management performance"""
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.hset.return_value = True
        mock_redis.hget.return_value = json.dumps({"test": "value"})
        
        state_manager = EnhancedSharedStateManager(
            redis_host="localhost",
            redis_port=6379,
            logger=Mock()
        )
        state_manager.redis_client = mock_redis
        
        async def state_operation():
            await state_manager.set_state("benchmark_key", {"test": "value"}, StateScope.SESSION)
            return await state_manager.get_state("benchmark_key", StateScope.SESSION)
        
        result = await benchmark(state_operation)
        assert result is not None

# Test configuration
pytest_plugins = ["pytest_asyncio", "pytest_benchmark"]

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "benchmark: mark test as a performance benchmark")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"]) 