#!/usr/bin/env python3
"""
Test Suite for Simple MCP System
===============================
Comprehensive tests for the local, no-authentication MCP server system.

Usage:
    python test_simple_mcp_system.py              # Run all tests
    python test_simple_mcp_system.py --unit       # Unit tests only
    python test_simple_mcp_system.py --integration # Integration tests only
    python test_simple_mcp_system.py --verbose    # Verbose output
"""

import asyncio
import json
import requests
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add the project root and federation directory to the path
project_root = str(Path(__file__).parent.parent)
federation_dir = str(Path(__file__).parent)
sys.path.insert(0, project_root)
sys.path.insert(0, federation_dir)

try:
    from simple_local_mcp_server import app, get_local_mcp_registry
    from simple_federated_rag_mcp_engine import SimpleFederatedRAGMCPEngine, create_simple_federated_engine
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Available modules in federation directory:")
    import os
    federation_files = [f for f in os.listdir(federation_dir) if f.endswith('.py')]
    print(f"Python files: {federation_files}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)


class TestSimpleMCPServer(unittest.TestCase):
    """Unit tests for the Simple MCP Server."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.server_url = "http://127.0.0.1:8081"  # Use different port for testing
        cls.server_process = None
        cls.start_test_server()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()
    
    @classmethod
    def start_test_server(cls):
        """Start a test server instance."""
        cmd = [
            sys.executable,
            "simple_local_mcp_server.py",
            "--host", "127.0.0.1",
            "--port", "8081"
        ]
        
        cls.server_process = subprocess.Popen(cmd)
        
        # Wait for server to be ready
        max_retries = 20
        for i in range(max_retries):
            try:
                response = requests.get(f"{cls.server_url}/health", timeout=2)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Test server failed to start")
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = requests.get(f"{self.server_url}/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        self.assertIn("mcps_available", data)
    
    def test_discover_endpoint_empty(self):
        """Test discovery with empty registry."""
        response = requests.get(f"{self.server_url}/discover")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIsInstance(data, list)
    
    def test_register_mcp(self):
        """Test MCP registration."""
        test_mcp = {
            "name": "test_mcp",
            "description": "A test MCP for unit testing",
            "version": "1.0.0"
        }
        
        response = requests.post(f"{self.server_url}/register", json=test_mcp)
        self.assertEqual(response.status_code, 201)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "registered")
        self.assertIn("name", data)
        self.assertEqual(data["name"], "test_mcp")
    
    def test_discover_after_registration(self):
        """Test discovery after registering an MCP."""
        # First register an MCP
        test_mcp = {
            "name": "test_mcp_2",
            "description": "Another test MCP",
            "version": "1.1.0"
        }
        
        requests.post(f"{self.server_url}/register", json=test_mcp)
        
        # Then discover
        response = requests.get(f"{self.server_url}/discover")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIsInstance(data, list)
        
        # Check if our MCP is in the list
        mcp_names = [mcp["name"] for mcp in data]
        self.assertIn("test_mcp_2", mcp_names)
    
    def test_execute_nonexistent_mcp(self):
        """Test executing a non-existent MCP."""
        execute_request = {
            "mcp_name": "nonexistent_mcp",
            "args": ["test"]
        }
        
        response = requests.post(f"{self.server_url}/execute", json=execute_request)
        self.assertEqual(response.status_code, 404)
        
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())
    
    def test_invalid_registration_data(self):
        """Test registration with invalid data."""
        invalid_mcp = {
            "description": "Missing name field",
            "version": "1.0.0"
        }
        
        response = requests.post(f"{self.server_url}/register", json=invalid_mcp)
        self.assertEqual(response.status_code, 422)  # Validation error


class TestSimpleFederatedEngine(unittest.TestCase):
    """Unit tests for the Simple Federated RAG MCP Engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_servers = ["http://127.0.0.1:8081"]
        self.engine = SimpleFederatedRAGMCPEngine(federation_nodes=self.test_servers)
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertIsInstance(self.engine, SimpleFederatedRAGMCPEngine)
        self.assertEqual(self.engine.federation_nodes, self.test_servers)
    
    def test_create_simple_federated_engine(self):
        """Test the convenience function for creating engines."""
        engine = create_simple_federated_engine(self.test_servers)
        self.assertIsInstance(engine, SimpleFederatedRAGMCPEngine)
        self.assertEqual(engine.federation_nodes, self.test_servers)
    
    def test_list_local_server_mcps(self):
        """Test listing MCPs from local servers."""
        try:
            mcps = self.engine.list_local_server_mcps()
            self.assertIsInstance(mcps, list)
        except Exception as e:
            # If server is not running, this is expected
            self.assertIn("connection", str(e).lower())


class IntegrationTests(unittest.TestCase):
    """Integration tests for the complete system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up integration test environment."""
        cls.server_urls = ["http://127.0.0.1:8082", "http://127.0.0.1:8083"]
        cls.server_processes = []
        cls.start_test_servers()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up integration test environment."""
        for process in cls.server_processes:
            if process:
                process.terminate()
                process.wait()
    
    @classmethod
    def start_test_servers(cls):
        """Start multiple test server instances."""
        for i, url in enumerate(cls.server_urls):
            port = 8082 + i
            cmd = [
                sys.executable,
                "simple_local_mcp_server.py",
                "--host", "127.0.0.1",
                "--port", str(port)
            ]
            
            process = subprocess.Popen(cmd)
            cls.server_processes.append(process)
            time.sleep(2)  # Stagger startup
        
        # Wait for all servers to be ready
        for url in cls.server_urls:
            max_retries = 20
            for i in range(max_retries):
                try:
                    response = requests.get(f"{url}/health", timeout=2)
                    if response.status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(f"Test server {url} failed to start")
    
    def test_multi_server_federation(self):
        """Test federation across multiple servers."""
        # Register different MCPs on different servers
        test_mcps = [
            {
                "name": "server1_mcp",
                "description": "MCP on server 1",
                "version": "1.0.0"
            },
            {
                "name": "server2_mcp",
                "description": "MCP on server 2",
                "version": "1.0.0"
            }
        ]
        
        # Register MCPs on different servers
        for i, (url, mcp) in enumerate(zip(self.server_urls, test_mcps)):
            response = requests.post(f"{url}/register", json=mcp)
            self.assertEqual(response.status_code, 201)
        
        # Create federated engine
        engine = create_simple_federated_engine(self.server_urls)
        
        # Test federation discovery
        try:
            all_mcps = engine.list_local_server_mcps()
            self.assertIsInstance(all_mcps, list)
            
            # Should find MCPs from both servers
            mcp_names = [mcp.get("name", "") for mcp in all_mcps]
            self.assertIn("server1_mcp", mcp_names)
            self.assertIn("server2_mcp", mcp_names)
            
        except Exception as e:
            self.fail(f"Federation discovery failed: {e}")
    
    def test_server_failover(self):
        """Test behavior when one server is unavailable."""
        # Include one invalid server URL
        mixed_servers = self.server_urls + ["http://127.0.0.1:9999"]
        
        engine = create_simple_federated_engine(mixed_servers)
        
        # Should still work with available servers
        try:
            mcps = engine.list_local_server_mcps()
            self.assertIsInstance(mcps, list)
        except Exception as e:
            # Some connection errors are expected, but shouldn't crash
            self.assertIn("connection", str(e).lower())


def run_performance_tests():
    """Run basic performance tests."""
    print("\n🚀 Running Performance Tests")
    print("=" * 40)
    
    server_url = "http://127.0.0.1:8081"
    
    # Test discovery performance
    start_time = time.time()
    for _ in range(10):
        try:
            response = requests.get(f"{server_url}/discover", timeout=5)
            if response.status_code != 200:
                break
        except requests.exceptions.RequestException:
            break
    
    discovery_time = (time.time() - start_time) / 10
    print(f"📊 Average discovery time: {discovery_time:.3f}s")
    
    # Test registration performance
    start_time = time.time()
    for i in range(5):
        test_mcp = {
            "name": f"perf_test_mcp_{i}",
            "description": f"Performance test MCP {i}",
            "version": "1.0.0"
        }
        try:
            response = requests.post(f"{server_url}/register", json=test_mcp, timeout=5)
            if response.status_code != 201:
                break
        except requests.exceptions.RequestException:
            break
    
    registration_time = (time.time() - start_time) / 5
    print(f"📊 Average registration time: {registration_time:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Test Simple MCP System")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Change to the federation directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    print("🧪 Simple MCP System Test Suite")
    print("=" * 35)
    
    # Configure test verbosity
    verbosity = 2 if args.verbose else 1
    
    # Create test suite
    suite = unittest.TestSuite()
    
    if args.unit or (not args.integration and not args.performance):
        print("\n🔬 Adding Unit Tests")
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSimpleMCPServer))
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSimpleFederatedEngine))
    
    if args.integration or (not args.unit and not args.performance):
        print("\n🔗 Adding Integration Tests")
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(IntegrationTests))
    
    # Run tests
    if suite.countTestCases() > 0:
        print(f"\n🏃 Running {suite.countTestCases()} tests...\n")
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        # Print summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\n❌ FAILURES:")
            for test, traceback in result.failures:
                error_line = traceback.split('\n')[-2]
                print(f"  - {test}: {error_line}")
        
        if result.errors:
            print("\n💥 ERRORS:")
            for test, traceback in result.errors:
                error_line = traceback.split('\n')[-2]
                print(f"  - {test}: {error_line}")
        
        if result.wasSuccessful():
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed")
            return 1
    
    # Run performance tests if requested
    if args.performance or (not args.unit and not args.integration):
        run_performance_tests()
    
    print("\n🎉 Test suite completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())