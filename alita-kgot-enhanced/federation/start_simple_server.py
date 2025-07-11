#!/usr/bin/env python3
"""
Simple MCP Server Startup Script
===============================
Convenience script to start the simple local MCP server with common configurations.

Usage:
    python start_simple_server.py                    # Default: localhost:8080
    python start_simple_server.py --dev              # Development mode with reload
    python start_simple_server.py --multi            # Start multiple servers
    python start_simple_server.py --demo             # Demo mode with example data
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path


def start_single_server(host="127.0.0.1", port=8080, reload=False):
    """Start a single MCP server instance."""
    cmd = [
        sys.executable,
        "simple_local_mcp_server.py",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    print(f"🚀 Starting Simple MCP Server on {host}:{port}")
    if reload:
        print("🔄 Auto-reload enabled for development")
    
    return subprocess.Popen(cmd)


def start_multiple_servers(base_port=8080, count=3):
    """Start multiple MCP server instances on different ports."""
    processes = []
    
    print(f"🚀 Starting {count} MCP servers...")
    
    for i in range(count):
        port = base_port + i
        print(f"   Server {i+1}: http://127.0.0.1:{port}")
        
        cmd = [
            sys.executable,
            "simple_local_mcp_server.py",
            "--host", "127.0.0.1",
            "--port", str(port)
        ]
        
        process = subprocess.Popen(cmd)
        processes.append(process)
        time.sleep(1)  # Stagger startup
    
    print(f"\n✅ All {count} servers started!")
    print("\n📋 Server URLs:")
    for i in range(count):
        port = base_port + i
        print(f"   Server {i+1}: http://127.0.0.1:{port}")
        print(f"   Health:   http://127.0.0.1:{port}/health")
        print(f"   Docs:     http://127.0.0.1:{port}/docs")
        print()
    
    return processes


def setup_demo_data():
    """Register some demo MCPs for testing."""
    import requests
    import json
    
    demo_mcps = [
        {
            "name": "text_processor",
            "description": "Processes and analyzes text content",
            "version": "1.0.0"
        },
        {
            "name": "data_validator",
            "description": "Validates data formats and schemas",
            "version": "1.1.0"
        },
        {
            "name": "file_converter",
            "description": "Converts files between different formats",
            "version": "2.0.0"
        }
    ]
    
    print("📝 Setting up demo data...")
    
    # Wait for server to be ready
    server_url = "http://127.0.0.1:8080"
    max_retries = 10
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{server_url}/health", timeout=2)
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            print(f"   Waiting for server... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("❌ Server not responding, skipping demo data setup")
        return
    
    # Register demo MCPs
    for mcp in demo_mcps:
        try:
            response = requests.post(f"{server_url}/register", json=mcp, timeout=5)
            if response.status_code == 201:
                print(f"   ✅ Registered: {mcp['name']}")
            else:
                print(f"   ⚠️  Failed to register {mcp['name']}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error registering {mcp['name']}: {e}")
    
    print("\n🎉 Demo data setup complete!")
    print(f"\n🔍 View registered MCPs at: {server_url}/discover")


def show_usage_examples():
    """Show usage examples after startup."""
    print("\n" + "="*60)
    print("📚 USAGE EXAMPLES")
    print("="*60)
    
    print("\n🐍 Python Client:")
    print("```python")
    print("from federation.simple_local_mcp_server import simple_discover, simple_execute")
    print("")
    print("# Discover MCPs")
    print("mcps = simple_discover('http://127.0.0.1:8080')")
    print("print(f'Found {len(mcps)} MCPs')")
    print("")
    print("# Execute an MCP")
    print("result = simple_execute('http://127.0.0.1:8080', 'text_processor', args=['hello'])")
    print("```")
    
    print("\n🌐 cURL Examples:")
    print("```bash")
    print("# Health check")
    print("curl http://127.0.0.1:8080/health")
    print("")
    print("# Discover MCPs")
    print("curl http://127.0.0.1:8080/discover")
    print("")
    print("# Execute MCP")
    print("curl -X POST http://127.0.0.1:8080/execute \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"mcp_name\": \"text_processor\", \"args\": [\"hello\"]}' ")
    print("```")
    
    print("\n🔧 Federated Engine:")
    print("```python")
    print("from federation.simple_federated_rag_mcp_engine import create_simple_federated_engine")
    print("")
    print("engine = create_simple_federated_engine(['http://127.0.0.1:8080'])")
    print("local_mcps = engine.list_local_server_mcps()")
    print("```")


def main():
    parser = argparse.ArgumentParser(
        description="Start Simple MCP Server with various configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_simple_server.py                 # Basic server on port 8080
  python start_simple_server.py --dev           # Development mode with reload
  python start_simple_server.py --multi         # Start 3 servers on ports 8080-8082
  python start_simple_server.py --demo          # Start with demo data
  python start_simple_server.py --port 9000     # Custom port
        """
    )
    
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to (default: 8080)")
    parser.add_argument("--dev", action="store_true", help="Enable development mode with auto-reload")
    parser.add_argument("--multi", action="store_true", help="Start multiple servers (ports 8080-8082)")
    parser.add_argument("--demo", action="store_true", help="Setup demo data after startup")
    parser.add_argument("--count", type=int, default=3, help="Number of servers for --multi mode (default: 3)")
    parser.add_argument("--examples", action="store_true", help="Show usage examples after startup")
    
    args = parser.parse_args()
    
    # Change to the federation directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🎯 Simple MCP Server Startup")
    print("=" * 30)
    
    processes = []
    
    try:
        if args.multi:
            processes = start_multiple_servers(args.port, args.count)
            
            # Set environment variable for federated engine
            servers = [f"http://127.0.0.1:{args.port + i}" for i in range(args.count)]
            os.environ["SIMPLE_MCP_SERVERS"] = ",".join(servers)
            print(f"\n🔧 Environment variable set: SIMPLE_MCP_SERVERS={os.environ['SIMPLE_MCP_SERVERS']}")
            
        else:
            process = start_single_server(args.host, args.port, args.dev)
            processes = [process]
            
            # Set environment variable for single server
            os.environ["SIMPLE_MCP_SERVERS"] = f"http://{args.host}:{args.port}"
        
        if args.demo:
            setup_demo_data()
        
        if args.examples:
            show_usage_examples()
        
        print("\n⌨️  Press Ctrl+C to stop all servers")
        
        # Wait for processes
        try:
            for process in processes:
                process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            for process in processes:
                process.terminate()
            
            # Wait for graceful shutdown
            time.sleep(2)
            
            # Force kill if needed
            for process in processes:
                if process.poll() is None:
                    process.kill()
            
            print("✅ All servers stopped")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()