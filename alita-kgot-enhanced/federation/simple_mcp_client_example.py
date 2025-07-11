#!/usr/bin/env python3
"""
Simple MCP Client Example
========================
Demonstrates how to use the Simple Local MCP Server without authentication.

This example shows:
1. Starting a local MCP server
2. Discovering available MCPs
3. Executing MCPs
4. Registering new MCPs
"""

import time
import subprocess
import requests
from typing import List, Dict, Any
import json


def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Wait for the server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def discover_mcps(base_url: str) -> List[Dict[str, Any]]:
    """Discover available MCPs on the server."""
    response = requests.get(f"{base_url}/discover")
    response.raise_for_status()
    return response.json()


def execute_mcp(base_url: str, mcp_name: str, args: List[Any] = None, kwargs: Dict[str, Any] = None) -> Any:
    """Execute an MCP on the server."""
    payload = {
        "mcp_name": mcp_name,
        "args": args or [],
        "kwargs": kwargs or {}
    }
    response = requests.post(f"{base_url}/execute", json=payload)
    response.raise_for_status()
    return response.json()["result"]


def register_mcp(base_url: str, name: str, description: str = None, version: str = "1.0.0") -> Dict[str, Any]:
    """Register a new MCP on the server."""
    payload = {
        "name": name,
        "description": description,
        "version": version
    }
    response = requests.post(f"{base_url}/register", json=payload)
    response.raise_for_status()
    return response.json()


def main():
    """Main example function."""
    server_url = "http://127.0.0.1:8080"
    
    print("🚀 Simple MCP Client Example")
    print("=" * 40)
    
    # Check if server is running
    try:
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is already running!")
        else:
            print("❌ Server is not responding properly")
            return
    except requests.exceptions.RequestException:
        print("❌ Server is not running. Please start it first with:")
        print("   python simple_local_mcp_server.py --port 8080")
        return
    
    print(f"\n📍 Using server at: {server_url}")
    
    # 1. Discover available MCPs
    print("\n🔍 Discovering available MCPs...")
    try:
        mcps = discover_mcps(server_url)
        print(f"Found {len(mcps)} MCPs:")
        for mcp in mcps:
            print(f"  - {mcp['name']}: {mcp.get('description', 'No description')}")
    except Exception as e:
        print(f"❌ Error discovering MCPs: {e}")
        return
    
    # 2. Try to execute an MCP (if any are available)
    if mcps:
        print("\n⚡ Trying to execute the first available MCP...")
        first_mcp = mcps[0]
        try:
            result = execute_mcp(server_url, first_mcp['name'])
            print(f"✅ Execution result: {result}")
        except Exception as e:
            print(f"⚠️  Execution failed (expected for placeholder MCPs): {e}")
    
    # 3. Register a new MCP
    print("\n📝 Registering a new test MCP...")
    try:
        register_result = register_mcp(
            server_url,
            "test_mcp",
            "A test MCP for demonstration purposes",
            "1.0.0"
        )
        print(f"✅ Registration result: {register_result}")
    except Exception as e:
        print(f"⚠️  Registration failed: {e}")
    
    # 4. Discover again to see the new MCP
    print("\n🔍 Discovering MCPs again (should include the new one)...")
    try:
        mcps_updated = discover_mcps(server_url)
        print(f"Found {len(mcps_updated)} MCPs:")
        for mcp in mcps_updated:
            status = "🆕" if mcp['name'] == "test_mcp" else "  "
            print(f"{status} - {mcp['name']}: {mcp.get('description', 'No description')}")
    except Exception as e:
        print(f"❌ Error discovering MCPs: {e}")
    
    # 5. Show server health
    print("\n❤️  Server health check...")
    try:
        health = requests.get(f"{server_url}/health").json()
        print(f"Status: {health['status']}")
        print(f"MCPs available: {health['mcps_available']}")
        print(f"Timestamp: {health['timestamp']}")
    except Exception as e:
        print(f"❌ Error checking health: {e}")
    
    print("\n✨ Example completed!")
    print("\n💡 Tips:")
    print("   - Visit http://127.0.0.1:8080/docs for interactive API documentation")
    print("   - Use the client functions in this script for your own applications")
    print("   - The server runs without authentication for easy local development")


if __name__ == "__main__":
    main()