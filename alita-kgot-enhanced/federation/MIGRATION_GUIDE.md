# Simple Federation System Guide

## Overview

This guide covers the Simple Local MCP Server system, which provides a streamlined, authentication-free approach to MCP federation for development and testing.

## 🚀 Quick Start

### 1. Start a Simple Federation Server

```bash
# Start with demo data
python federation/start_simple_server.py --demo

# Start on custom port
python federation/start_simple_server.py --port 8081

# Start with auto-reload for development
python federation/start_simple_server.py --reload
```

### 2. Use the Simple Federated Engine

```python
from federation.simple_federated_rag_mcp_engine import create_simple_federated_engine

# Create engine with local servers
engine = create_simple_federated_engine([
    "http://localhost:8080",
    "http://localhost:8081"
])

# List available MCPs
mcps = engine.list_local_server_mcps()
print(f"Found {len(mcps)} MCPs across federation nodes")

# Execute a remote MCP
result = engine.execute_remote_mcp("text_processor@localhost:8080", text="Hello World")
```

### 3. Direct Client Usage

```python
from federation.simple_local_mcp_server import simple_discover, simple_execute

# Discover MCPs
mcps = simple_discover("http://localhost:8080")

# Execute an MCP
result = simple_execute(
    "http://localhost:8080",
    "text_processor",
    kwargs={"text": "Hello World"}
)
```

## 🔧 Configuration

### Environment Variables

```bash
# Default local servers for the federated engine
export SIMPLE_MCP_SERVERS="http://localhost:8080,http://localhost:8081"

# Server configuration (optional)
export SIMPLE_MCP_HOST="127.0.0.1"
export SIMPLE_MCP_PORT="8080"
```

### Server Configuration

```python
# Custom server setup
from federation.simple_local_mcp_server import app
import uvicorn

uvicorn.run(app, host="127.0.0.1", port=8080)
```

## 🗂️ System Components

| Component | File | Purpose |
|-----------|------|----------|
| **Server** | `simple_local_mcp_server.py` | Local MCP server with REST API |
| **Engine** | `simple_federated_rag_mcp_engine.py` | Federation-aware RAG engine |
| **Tests** | `test_simple_mcp_system.py` | Test suite for all components |
| **Startup** | `start_simple_server.py` | Server startup with demo data |

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
python federation/test_simple_mcp_system.py --unit

# Test specific components
python federation/test_simple_mcp_system.py --server
python federation/test_simple_mcp_system.py --engine
```

### Manual Testing

```bash
# Start server with demo data
python federation/start_simple_server.py --demo

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/discover

# Test MCP execution
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{"mcp_name": "text_processor", "kwargs": {"text": "Hello"}}'
```

## 🔄 Multi-Server Setup

### Development Workflow

```bash
# Terminal 1: Main server
python federation/start_simple_server.py --port 8080 --demo

# Terminal 2: Secondary server
python federation/start_simple_server.py --port 8081

# Terminal 3: Test federation
python -c "
from federation.simple_federated_rag_mcp_engine import create_simple_federated_engine
engine = create_simple_federated_engine(['http://localhost:8080', 'http://localhost:8081'])
print(f'Federation ready with {len(engine.list_local_server_mcps())} MCPs')
"
```

### Register MCPs on Different Servers

```python
import requests

# Register different MCPs on different servers
servers = [
    ("http://localhost:8080", "data_processor", "Processes data files"),
    ("http://localhost:8081", "email_sender", "Sends emails"),
    ("http://localhost:8082", "report_generator", "Generates reports")
]

for server_url, mcp_name, description in servers:
    response = requests.post(f"{server_url}/register", json={
        "name": mcp_name,
        "description": description,
        "version": "1.0.0"
    })
    print(f"Registered {mcp_name} on {server_url}: {response.json()}")
```

## 🔒 Security Considerations

### Development Use Only

⚠️ **Important**: The simple federation system is designed for local development only:

- **No authentication** - anyone with network access can use the server
- **No input validation** - malicious payloads could cause issues
- **No rate limiting** - vulnerable to abuse
- **No encryption** - all traffic is in plain text

### Best Practices

1. **Local Only**: Only bind to `127.0.0.1` or `localhost`
2. **Firewall**: Ensure ports are not exposed externally
3. **Development Environment**: Use only in development/testing
4. **Clean Shutdown**: Stop servers when not in use

## 📊 Monitoring and Debugging

### Server Status

```python
import requests

# Check server health
response = requests.get("http://localhost:8080/health")
print(f"Server status: {response.json()}")

# List registered MCPs
response = requests.get("http://localhost:8080/discover")
print(f"Available MCPs: {len(response.json())}")
```

### Engine Debugging

```python
from federation.simple_federated_rag_mcp_engine import create_simple_federated_engine

# Create engine with debug info
engine = create_simple_federated_engine([
    "http://localhost:8080",
    "http://localhost:8081"
])

# Check federation status
local_mcps = engine.list_local_server_mcps()
print(f"Federation nodes: {len(engine.local_servers)}")
print(f"Total MCPs: {len(local_mcps)}")

for mcp in local_mcps:
    print(f"  - {mcp['name']} @ {mcp.get('server_url', 'unknown')}")
```

## 🚀 Advanced Usage

### Custom MCP Registration

```python
from federation.simple_local_mcp_server import get_local_mcp_registry

# Register a custom MCP function
def my_custom_mcp(text: str) -> str:
    """Custom MCP that processes text."""
    return f"Processed: {text.upper()}"

# Add to registry
registry = get_local_mcp_registry()
registry["my_custom_mcp"] = {
    "name": "my_custom_mcp",
    "description": "Custom text processor",
    "function": my_custom_mcp,
    "version": "1.0.0"
}
```

### Engine Integration

```python
from federation.simple_federated_rag_mcp_engine import SimpleFederatedRAGMCPEngine
from alita_kgot_enhanced.rag_mcp_engine import RAGMCPEngine

# Create engine with custom configuration
base_engine = RAGMCPEngine(enable_llm_validation=False)
federated_engine = SimpleFederatedRAGMCPEngine(
    base_engine=base_engine,
    local_servers=["http://localhost:8080"]
)

# Use as a regular RAG engine
result = federated_engine.execute_mcp("text_processor@localhost:8080", text="Hello")
```

## 📚 API Reference

### Server Endpoints

| Endpoint | Method | Purpose | Example |
|----------|--------|---------|----------|
| `/` | GET | Server info | `curl http://localhost:8080/` |
| `/health` | GET | Health check | `curl http://localhost:8080/health` |
| `/discover` | GET | List MCPs | `curl http://localhost:8080/discover` |
| `/execute` | POST | Execute MCP | See examples above |
| `/register` | POST | Register MCP | See examples above |

### Client Functions

| Function | Purpose | Example |
|----------|---------|----------|
| `simple_discover(url)` | Discover MCPs | `simple_discover("http://localhost:8080")` |
| `simple_execute(url, name, ...)` | Execute MCP | `simple_execute(url, "mcp_name", args=[])` |
| `create_simple_federated_engine(servers)` | Create engine | `create_simple_federated_engine(["http://localhost:8080"])` |

## 🆘 Troubleshooting

### Common Issues

1. **Server Won't Start**
   ```bash
   # Check if port is in use
   lsof -i :8080
   
   # Use different port
   python federation/start_simple_server.py --port 8081
   ```

2. **Connection Refused**
   ```python
   # Ensure server is running
   import requests
   try:
       response = requests.get("http://localhost:8080/health", timeout=5)
       print("Server is running:", response.json())
   except requests.exceptions.ConnectionError:
       print("Server is not running")
   ```

3. **MCPs Not Found**
   ```python
   # Check MCP registration
   response = requests.get("http://localhost:8080/discover")
   mcps = response.json()
   print(f"Available MCPs: {[mcp['name'] for mcp in mcps]}")
   ```

4. **Import Errors**
   ```python
   # Ensure you're in the right directory
   import sys
   print("Python path:", sys.path)
   
   # Check module availability
   try:
       from federation.simple_federated_rag_mcp_engine import create_simple_federated_engine
       print("✅ Module imported successfully")
   except ImportError as e:
       print(f"❌ Import error: {e}")
   ```

## 📈 Performance Tips

1. **Use Multiple Servers**: Distribute MCPs across multiple servers for better performance
2. **Local Network**: Keep all servers on the same local network
3. **Connection Pooling**: Reuse engine instances instead of creating new ones
4. **Async Operations**: Consider async patterns for high-throughput scenarios

---

**Ready to start?** Run `python federation/start_simple_server.py --demo` and begin exploring the simple federation system! 🚀