"""
Unified System Integration Script

This script demonstrates how to use the Unified System Controller and provides
examples of integrating it with existing Alita and KGoT systems.

Author: Advanced AI Development Team
Version: 1.0.0
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import argparse
import sys
import os

# Add the core directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_system_controller import (
    UnifiedSystemController, TaskContext, TaskComplexity, 
    RoutingStrategy, SystemStatus
)
from enhanced_logging_system import StructuredLogger, LogLevel, LogCategory
from shared_state_utilities import StateScope

class UnifiedSystemDemo:
    """
    Demonstration class showing how to integrate and use the Unified System Controller
    """
    
    def __init__(self, 
                 alita_url: str = "http://localhost:3001",
                 kgot_url: str = "http://localhost:8000",
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_password: Optional[str] = None):
        """
        Initialize the demo with system configurations
        
        Args:
            alita_url: Base URL for Alita Manager Agent
            kgot_url: Base URL for KGoT Controller
            redis_host: Redis server host
            redis_port: Redis server port
            redis_password: Redis server password
        """
        self.alita_url = alita_url
        self.kgot_url = kgot_url
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.controller: Optional[UnifiedSystemController] = None
        self.logger = StructuredLogger("unified_system_demo", LogLevel.INFO)
    
    async def initialize_system(self) -> bool:
        """
        Initialize the unified system controller
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Unified System Controller...", LogCategory.SYSTEM)
            
            self.controller = UnifiedSystemController(
                alita_base_url=self.alita_url,
                kgot_base_url=self.kgot_url,
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                redis_password=self.redis_password
            )
            
            # Start the system
            await self.controller.startup()
            
            self.logger.info(
                "Unified System Controller initialized successfully",
                LogCategory.SYSTEM,
                {"status": self.controller.status.value}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize Unified System Controller: {e}",
                LogCategory.SYSTEM,
                error=e
            )
            return False
    
    async def run_basic_demo(self):
        """Run a basic demonstration of the unified system"""
        if not self.controller:
            self.logger.error("Controller not initialized", LogCategory.SYSTEM)
            return
        
        self.logger.info("Starting basic demo...", LogCategory.SYSTEM)
        
        # Demo tasks with different complexities
        demo_tasks = [
            {
                "task_id": "demo_simple_query",
                "task_type": "query",
                "content": "What is the capital of France?",
                "metadata": {"demo": True, "complexity": "low"}
            },
            {
                "task_id": "demo_complex_reasoning",
                "task_type": "reasoning",
                "content": "Analyze the potential impact of quantum computing on cryptographic security and propose mitigation strategies",
                "metadata": {"demo": True, "complexity": "high", "requires_graph": True}
            },
            {
                "task_id": "demo_calculation",
                "task_type": "calculation",
                "content": "Calculate the compound interest on $10,000 invested at 5% annually for 10 years",
                "metadata": {"demo": True, "complexity": "medium"}
            }
        ]
        
        results = []
        
        for task_data in demo_tasks:
            try:
                self.logger.info(
                    f"Processing demo task: {task_data['task_id']}",
                    LogCategory.ROUTING,
                    {"task_type": task_data["task_type"]}
                )
                
                # Create task context
                task_context = TaskContext(
                    task_id=task_data["task_id"],
                    task_type=task_data["task_type"],
                    content=task_data["content"],
                    metadata=task_data["metadata"]
                )
                
                # Process the task
                start_time = time.time()
                result = await self.controller.process_task(task_context)
                end_time = time.time()
                
                # Log the result
                self.logger.info(
                    f"Task {task_data['task_id']} completed",
                    LogCategory.ROUTING,
                    {
                        "success": result.success,
                        "system_used": result.system_used,
                        "execution_time_ms": (end_time - start_time) * 1000,
                        "response_length": len(result.response) if result.response else 0
                    }
                )
                
                results.append({
                    "task_id": task_data["task_id"],
                    "success": result.success,
                    "system_used": result.system_used,
                    "execution_time": end_time - start_time,
                    "response": result.response[:200] + "..." if len(result.response) > 200 else result.response
                })
                
            except Exception as e:
                self.logger.error(
                    f"Failed to process task {task_data['task_id']}: {e}",
                    LogCategory.SYSTEM,
                    error=e
                )
                results.append({
                    "task_id": task_data["task_id"],
                    "success": False,
                    "error": str(e)
                })
        
        # Display results
        self.logger.info("Demo completed. Results summary:", LogCategory.SYSTEM)
        for result in results:
            print(f"\nTask: {result['task_id']}")
            print(f"Success: {result['success']}")
            if result['success']:
                print(f"System Used: {result['system_used']}")
                print(f"Execution Time: {result['execution_time']:.2f}s")
                print(f"Response: {result['response']}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
    
    async def run_load_balancing_demo(self):
        """Demonstrate load balancing capabilities"""
        if not self.controller:
            self.logger.error("Controller not initialized", LogCategory.SYSTEM)
            return
        
        self.logger.info("Starting load balancing demo...", LogCategory.PERFORMANCE)
        
        # Create multiple concurrent tasks
        concurrent_tasks = []
        task_count = 5
        
        for i in range(task_count):
            task_context = TaskContext(
                task_id=f"load_test_{i}",
                task_type="query",
                content=f"Load balancing test question {i}",
                metadata={"demo": True, "load_test": True}
            )
            concurrent_tasks.append(self.controller.process_task(task_context))
        
        # Execute tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_tasks = sum(1 for result in results if hasattr(result, 'success') and result.success)
        total_time = end_time - start_time
        
        self.logger.info(
            "Load balancing demo completed",
            LogCategory.PERFORMANCE,
            {
                "total_tasks": task_count,
                "successful_tasks": successful_tasks,
                "total_time": total_time,
                "tasks_per_second": task_count / total_time if total_time > 0 else 0
            }
        )
        
        print(f"\nLoad Balancing Demo Results:")
        print(f"Total Tasks: {task_count}")
        print(f"Successful Tasks: {successful_tasks}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {task_count / total_time:.2f} tasks/second")
        
        # Show system distribution
        system_usage = {}
        for result in results:
            if hasattr(result, 'system_used'):
                system_usage[result.system_used] = system_usage.get(result.system_used, 0) + 1
        
        print(f"System Usage Distribution:")
        for system, count in system_usage.items():
            print(f"  {system}: {count} tasks ({count/task_count*100:.1f}%)")
    
    async def run_error_handling_demo(self):
        """Demonstrate error handling and recovery capabilities"""
        if not self.controller:
            self.logger.error("Controller not initialized", LogCategory.SYSTEM)
            return
        
        self.logger.info("Starting error handling demo...", LogCategory.SYSTEM)
        
        # Create tasks that might fail
        error_test_tasks = [
            {
                "task_id": "error_test_invalid_input",
                "task_type": "invalid_type",
                "content": "",
                "metadata": {"demo": True, "expect_error": True}
            },
            {
                "task_id": "error_test_timeout",
                "task_type": "query",
                "content": "This is a test for timeout handling",
                "metadata": {"demo": True, "simulate_timeout": True}
            }
        ]
        
        for task_data in error_test_tasks:
            try:
                task_context = TaskContext(
                    task_id=task_data["task_id"],
                    task_type=task_data["task_type"],
                    content=task_data["content"],
                    metadata=task_data["metadata"]
                )
                
                self.logger.info(
                    f"Testing error handling for: {task_data['task_id']}",
                    LogCategory.SYSTEM
                )
                
                result = await self.controller.process_task(task_context)
                
                print(f"\nError Test: {task_data['task_id']}")
                print(f"Result: {'Success' if result.success else 'Failed (as expected)'}")
                
                if not result.success:
                    print(f"Error handled gracefully: {result.error_message}")
                
            except Exception as e:
                self.logger.info(
                    f"Error handling test completed for {task_data['task_id']}: {e}",
                    LogCategory.SYSTEM
                )
                print(f"\nError Test: {task_data['task_id']}")
                print(f"Exception caught and handled: {str(e)}")
    
    async def display_system_status(self):
        """Display current system status and metrics"""
        if not self.controller:
            self.logger.error("Controller not initialized", LogCategory.SYSTEM)
            return
        
        try:
            # Get system status
            status = await self.controller.get_system_status()
            
            print("\n" + "="*60)
            print("UNIFIED SYSTEM STATUS")
            print("="*60)
            print(f"Overall Status: {status['status']}")
            print(f"Uptime: {status.get('uptime', 'Unknown')}")
            print(f"Total Requests: {status.get('total_requests', 0)}")
            print(f"Success Rate: {status.get('success_rate', 0):.2%}")
            
            # Display system-specific status
            if 'systems' in status:
                print("\nSystem Health:")
                for system_name, system_status in status['systems'].items():
                    print(f"  {system_name.upper()}: {system_status.get('status', 'Unknown')}")
                    print(f"    Response Time: {system_status.get('avg_response_time', 0):.2f}ms")
                    print(f"    Success Rate: {system_status.get('success_rate', 0):.2%}")
            
            # Display recent metrics
            if 'recent_metrics' in status:
                metrics = status['recent_metrics']
                print(f"\nRecent Performance:")
                print(f"  Error Rate: {metrics.get('error_rate', 0):.2%}")
                print(f"  Avg Response Time: {metrics.get('avg_response_time', 0):.2f}ms")
                print(f"  Active Requests: {metrics.get('active_requests', 0)}")
            
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}", LogCategory.SYSTEM, error=e)
    
    async def cleanup(self):
        """Clean up resources"""
        if self.controller:
            try:
                await self.controller.shutdown()
                self.logger.info("System shutdown completed", LogCategory.SYSTEM)
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}", LogCategory.SYSTEM, error=e)

async def main():
    """Main function to run the unified system demo"""
    parser = argparse.ArgumentParser(description="Unified System Controller Demo")
    parser.add_argument("--alita-url", default="http://localhost:3001", 
                       help="Alita Manager Agent URL")
    parser.add_argument("--kgot-url", default="http://localhost:8000", 
                       help="KGoT Controller URL")
    parser.add_argument("--redis-host", default="localhost", 
                       help="Redis server host")
    parser.add_argument("--redis-port", type=int, default=6379, 
                       help="Redis server port")
    parser.add_argument("--redis-password", 
                       help="Redis server password")
    parser.add_argument("--demo", choices=["basic", "load", "error", "all"], 
                       default="all", help="Demo type to run")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = UnifiedSystemDemo(
        alita_url=args.alita_url,
        kgot_url=args.kgot_url,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_password=args.redis_password
    )
    
    try:
        # Initialize the system
        if not await demo.initialize_system():
            print("Failed to initialize system. Exiting.")
            return 1
        
        # Display initial status
        await demo.display_system_status()
        
        # Run selected demos
        if args.demo in ["basic", "all"]:
            print("\n" + "="*60)
            print("RUNNING BASIC DEMO")
            print("="*60)
            await demo.run_basic_demo()
        
        if args.demo in ["load", "all"]:
            print("\n" + "="*60)
            print("RUNNING LOAD BALANCING DEMO")
            print("="*60)
            await demo.run_load_balancing_demo()
        
        if args.demo in ["error", "all"]:
            print("\n" + "="*60)
            print("RUNNING ERROR HANDLING DEMO")
            print("="*60)
            await demo.run_error_handling_demo()
        
        # Display final status
        print("\n" + "="*60)
        print("FINAL SYSTEM STATUS")
        print("="*60)
        await demo.display_system_status()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return 0
    except Exception as e:
        print(f"Demo failed with error: {e}")
        return 1
    finally:
        # Cleanup
        await demo.cleanup()

def print_integration_guide():
    """Print integration guide for developers"""
    guide = """
    
    ===============================================
    UNIFIED SYSTEM CONTROLLER INTEGRATION GUIDE
    ===============================================
    
    1. BASIC USAGE:
    
    from unified_system_controller import UnifiedSystemController, TaskContext
    
    # Initialize controller
    controller = UnifiedSystemController(
        alita_base_url="http://localhost:3001",
        kgot_base_url="http://localhost:8000"
    )
    
    # Start the system
    await controller.startup()
    
    # Process a task
    task = TaskContext(
        task_id="my_task",
        task_type="query",
        content="What is machine learning?",
        metadata={}
    )
    
    result = await controller.process_task(task)
    print(f"Response: {result.response}")
    
    2. ADVANCED CONFIGURATION:
    
    # Custom configuration
    controller = UnifiedSystemController(
        alita_base_url="http://alita.example.com",
        kgot_base_url="http://kgot.example.com",
        redis_host="redis.example.com",
        redis_port=6379,
        redis_password="secure_password",
        sequential_thinking_endpoint="http://st.example.com"
    )
    
    3. ERROR HANDLING:
    
    try:
        result = await controller.process_task(task)
    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        # Error is automatically handled by the system
    
    4. MONITORING:
    
    # Get system status
    status = await controller.get_system_status()
    
    # Get performance metrics
    metrics = await controller.get_performance_metrics()
    
    5. CUSTOM ROUTING:
    
    # Force specific system
    task.metadata["force_system"] = "alita"  # or "kgot"
    
    # Set priority
    task.metadata["priority"] = "high"
    
    6. CIRCUIT BREAKER:
    
    # Circuit breaker automatically handles failures
    # No manual intervention required
    
    7. SHARED STATE:
    
    # Access shared state
    state_value = await controller.state_manager.get_state("key", StateScope.GLOBAL)
    await controller.state_manager.set_state("key", value, StateScope.GLOBAL)
    
    ===============================================
    
    For more examples, see the demo functions above.
    
    """
    print(guide)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--guide":
        print_integration_guide()
    else:
        exit_code = asyncio.run(main())
        sys.exit(exit_code) 