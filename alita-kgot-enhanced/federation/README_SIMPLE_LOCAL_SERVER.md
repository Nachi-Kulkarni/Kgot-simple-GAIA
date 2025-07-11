# Simple Local MCP Server

> **Easy-to-use local MCP server without authentication for development and testing**

## Overview

The Simple Local MCP Server provides a streamlined alternative to the full MCP Federation System. It removes authentication requirements and simplifies the setup process, making it perfect for:

- **Local development** and testing
- **Rapid prototyping** of MCP integrations
- **Educational purposes** and demonstrations
- **Internal team collaboration** without complex auth setup

## Quick Start

### 1. Start the Server

```bash
# Basic usage (runs on http://127.0.0.1:8080)
python simple_local_mcp_server.py

# Custom host and port
python simple_local_mcp_server.py --host 0.0.0.0 --port 8000

# Enable auto-reload for development
python simple_local_mcp_server.py --reload
```

### 2. Verify Server is Running

```bash
# Check health
curl http://127.0.0.1:8080/health

# Discover available MCPs
curl http://127.0.0.1:8080/discover
```

### 3. Use the Client Example

```bash
python simple_mcp_client_example.py
```

## API Endpoints

### Health Check
```http
GET /health
```
Returns server status and basic metrics.

### Discover MCPs
```http
GET /discover
```
Lists all available MCPs with metadata.

### Execute MCP
```http
POST /execute
Content-Type: application/json

{
  "mcp_name": "example_mcp",
  "args": [],
  "kwargs": {}
}
```

### Register New MCP
```http
POST /register
Content-Type: application/json

{
  "name": "my_custom_mcp",
  "description": "A custom MCP for testing",
  "version": "1.0.0"
}
```

## Python Client Usage

### Basic Client Functions

```python
from federation.simple_local_mcp_server import simple_discover, simple_execute

# Discover MCPs
mcps = simple_discover("http://127.0.0.1:8080")
print(f"Found {len(mcps)} MCPs")

# Execute an MCP
result = simple_execute(
    "http://127.0.0.1:8080",
    "example_mcp",
    args=["arg1", "arg2"],
    kwargs={"param": "value"}
)
```

### Using with Federated RAG-MCP Engine

```python
from federation.simple_federated_rag_mcp_engine import create_simple_federated_engine

# Create engine that connects to local servers
engine = create_simple_federated_engine([
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8081"
])

# List available MCPs from local servers
local_mcps = engine.list_local_server_mcps()

# Execute a remote MCP
result = engine.execute_remote_mcp("mcp_name@127.0.0.1:8080", arg1="value")
```

## Environment Configuration

### Server Configuration

No environment variables are required, but you can optionally set:

```bash
# For logging level (optional)
export LOG_LEVEL=INFO
```

### Client Configuration

```bash
# Default local servers for the federated engine
export SIMPLE_MCP_SERVERS="http://127.0.0.1:8080,http://127.0.0.1:8081"
```

## Comparison with Full Federation System

| Feature | Simple Local Server | Full Federation System |
|---------|-------------------|------------------------|
| **Authentication** | None required | Bearer token required |
| **Setup Complexity** | Minimal | Complex |
| **Security Checks** | Disabled | Quality + Security validation |
| **Use Case** | Development/Testing | Production deployment |
| **Network** | Local only | Cross-organization |
| **Performance Tracking** | Basic | Advanced metrics |

## Development Workflow

### 1. Start Multiple Servers

```bash
# Terminal 1: Main server
python simple_local_mcp_server.py --port 8080

# Terminal 2: Secondary server
python simple_local_mcp_server.py --port 8081

# Terminal 3: Test server
python simple_local_mcp_server.py --port 8082
```

### 2. Configure Federated Engine

```python
# Connect to all local servers
engine = create_simple_federated_engine([
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8081",
    "http://127.0.0.1:8082"
])
```

### 3. Test MCP Distribution

```python
# Register different MCPs on different servers
import requests

# Server 1: Data processing MCPs
requests.post("http://127.0.0.1:8080/register", json={
    "name": "data_processor",
    "description": "Processes data files"
})

# Server 2: Communication MCPs
requests.post("http://127.0.0.1:8081/register", json={
    "name": "email_sender",
    "description": "Sends emails"
})

# Server 3: Analysis MCPs
requests.post("http://127.0.0.1:8082/register", json={
    "name": "data_analyzer",
    "description": "Analyzes datasets"
})
```

## Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://127.0.0.1:8080/docs
- **ReDoc**: http://127.0.0.1:8080/redoc

## Troubleshooting

### Server Won't Start

```bash
# Check if port is already in use
lsof -i :8080

# Use a different port
python simple_local_mcp_server.py --port 8090
```

### MCPs Not Loading

```bash
# Check server logs for import errors
python simple_local_mcp_server.py --reload

# Verify MCP toolbox modules are available
python -c "import mcp_toolbox.data_processing_mcps"
```

### Connection Issues

```python
# Test basic connectivity
import requests
response = requests.get("http://127.0.0.1:8080/health")
print(response.json())
```

## Integration with RAG Engine

The simple local server integrates seamlessly with the RAG-MCP engine:

### Basic Integration

```python
from federation.simple_federated_rag_mcp_engine import create_simple_federated_engine

# Create federated engine
engine = create_simple_federated_engine([
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8081"
])

# Use like a regular RAG engine
result = engine.execute_mcp("text_processor@127.0.0.1:8080", text="Hello World")
```

### Advanced Configuration

```python
from federation.simple_federated_rag_mcp_engine import SimpleFederatedRAGMCPEngine
from alita_kgot_enhanced.rag_mcp_engine import RAGMCPEngine

# Custom base engine
base_engine = RAGMCPEngine(enable_llm_validation=False)
engine = SimpleFederatedRAGMCPEngine(
    base_engine=base_engine,
    local_servers=["http://127.0.0.1:8080"]
)
```

## Security Considerations

⚠️ **Important**: This simple server is designed for local development only.

- **No authentication** - anyone with network access can use the server
- **No input validation** - malicious payloads could cause issues
- **No rate limiting** - vulnerable to abuse
- **No encryption** - all traffic is in plain text

**For production use**, stick with the full federation system that includes proper security measures.

## Contributing

To extend the simple server:

1. **Add new endpoints** in `simple_local_mcp_server.py`
2. **Update client functions** for new functionality
3. **Extend the federated engine** for advanced features
4. **Add tests** in the `tests/` directory

## Examples

See the `examples/` directory for:
- Basic client usage
- Multi-server setup
- Custom MCP registration
- Integration with existing systems

---

**Happy MCP development! 🚀**