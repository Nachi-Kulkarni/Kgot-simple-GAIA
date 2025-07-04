# KGoT Integrated Tools API Reference

## Overview

This document provides comprehensive API reference for the KGoT Integrated Tools system implemented in Task 4 of the Enhanced Alita project.

## Core Classes

### AlitaIntegratedToolsManager

The main orchestrator for all KGoT tools with AI model assignments.

#### Constructor

```python
class AlitaIntegratedToolsManager:
    def __init__(self, model_config: ModelConfiguration, usage_stats: UsageStatistics = None)
```

**Parameters:**
- `model_config` (ModelConfiguration): Configuration for AI model assignments
- `usage_stats` (UsageStatistics, optional): Usage tracking instance

#### Methods

##### get_tool(tool_name: str) → Tool
Gets a specific tool instance by name.

**Parameters:**
- `tool_name` (str): Name of the tool to retrieve

**Returns:**
- Tool instance or None if not found

**Example:**
```python
python_tool = manager.get_tool('python_code')
```

##### get_all_tools() → Dict[str, Tool]
Returns all registered tools.

**Returns:**
- Dictionary mapping tool names to tool instances

##### get_tools_by_category(category: str) → List[Tool]
Gets tools filtered by category.

**Parameters:**
- `category` (str): Category to filter by ('development', 'research', 'web', 'multimodal')

**Returns:**
- List of tool instances in the specified category

##### register_tool(name: str, tool: Tool, metadata: ToolMetadata) → bool
Registers a new tool with the manager.

**Parameters:**
- `name` (str): Unique tool identifier
- `tool` (Tool): Tool instance to register
- `metadata` (ToolMetadata): Tool metadata and configuration

**Returns:**
- bool: True if registration successful

##### export_configuration() → Dict
Exports complete system configuration including tools and model assignments.

**Returns:**
- Dictionary containing full system configuration

##### validate_configuration() → bool
Validates current configuration and model assignments.

**Returns:**
- bool: True if configuration is valid

### ModelConfiguration

Configuration dataclass for AI model assignments.

#### Constructor

```python
@dataclass
class ModelConfiguration:
    vision_model: str = "openai/o3"
    orchestration_model: str = "google/gemini-2.5-pro"
    web_agent_model: str = "anthropic/claude-4-sonnet"
    temperature: float = 0.3
    max_tokens: int = 32000
    timeout: int = 60
    max_retries: int = 3
```

**Attributes:**
- `vision_model` (str): Model for multimodal and vision tasks
- `orchestration_model` (str): Model for complex reasoning and orchestration
- `web_agent_model` (str): Model for web interaction and navigation
- `temperature` (float): Sampling temperature for model responses
- `max_tokens` (int): Maximum tokens for model responses
- `timeout` (int): Timeout in seconds for model requests
- `max_retries` (int): Maximum retry attempts for failed requests

### ToolMetadata

Metadata container for tool registration and tracking.

#### Constructor

```python
@dataclass
class ToolMetadata:
    tool_name: str
    tool_type: str
    model_assignment: str
    description: str
    category: str
    input_schema: Dict
    output_schema: Dict
    capabilities: List[str]
    prerequisites: List[str]
    status: str = "active"
```

**Attributes:**
- `tool_name` (str): Unique identifier for the tool
- `tool_type` (str): Type classification of the tool
- `model_assignment` (str): Assigned AI model for this tool
- `description` (str): Human-readable description
- `category` (str): Tool category ('development', 'research', 'web', 'multimodal')
- `input_schema` (Dict): JSON schema for tool inputs
- `output_schema` (Dict): JSON schema for tool outputs
- `capabilities` (List[str]): List of tool capabilities
- `prerequisites` (List[str]): Required dependencies or setup
- `status` (str): Tool status ('active', 'inactive', 'error')

## Tool Interfaces

### RunPythonCodeTool

Python code execution tool with optional LLM-based error fixing.

#### Constructor

```python
class RunPythonCodeTool:
    def __init__(
        self,
        try_to_fix: bool = False,
        python_executor_uri: str = "http://localhost:16000",
        usage_statistics: UsageStatistics = None,
        llm_instance: LLM = None
    )
```

**Parameters:**
- `try_to_fix` (bool): Enable LLM-based error fixing
- `python_executor_uri` (str): URI for Python execution service
- `usage_statistics` (UsageStatistics): Usage tracking instance
- `llm_instance` (LLM): Language model for error fixing

#### Methods

##### _run(code: str, required_modules: List[str] = None) → str
Executes Python code and returns output.

**Parameters:**
- `code` (str): Python code to execute
- `required_modules` (List[str], optional): Required Python packages

**Returns:**
- str: Execution output or error message

**Example:**
```python
result = tool._run(
    code="import numpy as np\nprint(np.array([1, 2, 3]))",
    required_modules=["numpy"]
)
```

### UsageStatistics

Usage tracking and analytics for tool usage patterns.

#### Constructor

```python
class UsageStatistics:
    def __init__(self)
```

#### Methods

##### record_tool_usage(tool_name: str, execution_time: float, success: bool) → None
Records tool usage statistics.

**Parameters:**
- `tool_name` (str): Name of the tool used
- `execution_time` (float): Execution time in seconds
- `success` (bool): Whether execution was successful

##### get_statistics() → Dict
Returns aggregated usage statistics.

**Returns:**
- Dict: Usage statistics including success rates, execution times, etc.

## JavaScript Bridge API

### KGoTToolBridge

JavaScript bridge for web integration of KGoT tools.

#### Constructor

```javascript
class KGoTToolBridge {
  constructor(options = {}) {
    this.apiEndpoint = options.apiEndpoint || 'http://localhost:8000';
    this.timeout = options.timeout || 30000;
    this.retries = options.retries || 3;
    
    // Model configurations
    this.modelConfigurations = {
      vision: 'openai/o3',
      orchestration: 'google/gemini-2.5-pro',
      webAgent: 'anthropic/claude-4-sonnet'
    };
  }
}
```

#### Methods

##### async executeCommand(command, model = 'orchestration')
Executes a command using specified AI model.

**Parameters:**
- `command` (string): Command to execute
- `model` (string): Model to use ('vision', 'orchestration', 'webAgent')

**Returns:**
- Promise\<Object\>: Command execution result

**Example:**
```javascript
const result = await bridge.executeCommand(
  'Analyze this image for objects',
  'vision'
);
```

##### async getAvailableTools()
Gets list of available tools from the Python backend.

**Returns:**
- Promise\<Array\>: List of available tools

##### async validateConnection()
Validates connection to the Python backend.

**Returns:**
- Promise\<boolean\>: Connection status

## Configuration Management

### LLM Configuration

Configuration for language models using OpenRouter endpoints.

**File**: `knowledge-graph-of-thoughts/kgot/config_llms.json`

**Schema:**
```json
{
  "<model_name>": {
    "model": "string",
    "temperature": "number",
    "api_key": "string",
    "base_url": "string",
    "model_family": "string"
  }
}
```

**Example:**
```json
{
  "openai/o3": {
    "model": "openai/o3",
    "temperature": 0.3,
    "api_key": "your-openrouter-key",
    "base_url": "https://openrouter.ai/api/v1",
    "model_family": "OpenAI"
  }
}
```

### Tools Configuration

Configuration for external tool APIs and services.

**File**: `knowledge-graph-of-thoughts/kgot/config_tools.json`

**Schema:**
```json
[
  {
    "name": "string",
    "env": {
      "<ENV_VAR>": "string"
    }
  }
]
```

## Factory Functions

### create_integrated_tools_manager(config: ModelConfiguration) → AlitaIntegratedToolsManager

Factory function for creating a fully configured tools manager.

**Parameters:**
- `config` (ModelConfiguration): Model configuration instance

**Returns:**
- AlitaIntegratedToolsManager: Configured tools manager

**Example:**
```python
from integrated_tools_manager import create_integrated_tools_manager, ModelConfiguration

config = ModelConfiguration(
    vision_model="openai/o3",
    orchestration_model="google/gemini-2.5-pro",
    web_agent_model="anthropic/claude-4-sonnet"
)

manager = create_integrated_tools_manager(config)
```

## Error Handling

### Exception Classes

#### ToolRegistrationError
Raised when tool registration fails.

```python
class ToolRegistrationError(Exception):
    """Raised when tool registration fails"""
    pass
```

#### ModelConfigurationError
Raised when model configuration is invalid.

```python
class ModelConfigurationError(Exception):
    """Raised when model configuration is invalid"""
    pass
```

#### ToolExecutionError
Raised when tool execution fails.

```python
class ToolExecutionError(Exception):
    """Raised when tool execution fails"""
    pass
```

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": true,
  "error_type": "string",
  "message": "string",
  "details": "object",
  "timestamp": "string"
}
```

## Logging

### Winston Logger Integration

The system uses Winston-compatible logging throughout.

**Log Levels:**
- `error`: Critical errors requiring attention
- `warn`: Warnings about non-critical issues
- `info`: General information about operations
- `debug`: Detailed debugging information

**Log Format:**
```json
{
  "timestamp": "2025-01-27T13:23:30.741Z",
  "level": "info",
  "message": "Tool registered successfully",
  "service": "integrated_tools",
  "tool_name": "python_code",
  "metadata": {}
}
```

## Performance Monitoring

### Metrics Collected

- **Tool Execution Time**: Time taken for tool execution
- **Success Rate**: Percentage of successful tool executions
- **Model Response Time**: AI model response latency
- **Error Rate**: Frequency of errors by tool and model
- **Resource Usage**: Memory and CPU usage patterns

### Monitoring API

```python
# Get performance metrics
metrics = manager.usage_statistics.get_statistics()

# Example response
{
  "total_executions": 150,
  "success_rate": 0.94,
  "average_execution_time": 2.3,
  "tools": {
    "python_code": {
      "executions": 100,
      "success_rate": 0.96,
      "avg_time": 1.8
    }
  },
  "models": {
    "google/gemini-2.5-pro": {
      "requests": 50,
      "avg_response_time": 3.2,
      "success_rate": 0.92
    }
  }
}
```

## Integration Examples

### Basic Integration

```python
from integrated_tools_manager import create_integrated_tools_manager, ModelConfiguration

# Setup
config = ModelConfiguration()
manager = create_integrated_tools_manager(config)

# Execute Python code
python_tool = manager.get_tool('python_code')
result = python_tool._run("print('Hello, KGoT!')")
print(result)

# Get configuration
config_export = manager.export_configuration()
```

### Web Integration

```javascript
// Initialize bridge
const bridge = new KGoTToolBridge({
  apiEndpoint: 'http://localhost:8000',
  timeout: 30000
});

// Validate connection
const isConnected = await bridge.validateConnection();

// Execute vision task
if (isConnected) {
  const result = await bridge.executeCommand(
    'Identify objects in this image',
    'vision'
  );
  console.log(result);
}
```

### Advanced Usage

```python
# Custom tool registration
from tools.custom_tool import MyCustomTool

custom_tool = MyCustomTool()
metadata = ToolMetadata(
    tool_name='my_tool',
    tool_type='custom',
    model_assignment='google/gemini-2.5-pro',
    description='My custom tool',
    category='development',
    input_schema={'type': 'object'},
    output_schema={'type': 'string'},
    capabilities=['custom_processing'],
    prerequisites=['custom_dependency']
)

success = manager.register_tool('my_tool', custom_tool, metadata)
```

## Security Considerations

### API Key Management
- Store API keys in secure configuration files
- Use environment variables for sensitive data
- Implement key rotation mechanisms

### Input Validation
- All tool inputs are validated against schemas
- Code execution is sandboxed
- SQL injection protection for database queries

### Access Control
- Model assignments enforce capability boundaries
- Tool access can be restricted by user roles
- Audit logging for all tool executions

---

*This API reference is part of the KGoT Integrated Tools documentation suite.* 