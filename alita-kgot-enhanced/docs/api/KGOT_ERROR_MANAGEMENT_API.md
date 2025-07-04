# KGoT Error Management System - API Reference

## 🚀 **Quick Start**

```python
# Initialize the error management system
from error_management import create_kgot_error_management_system

error_system = create_kgot_error_management_system(
    llm_client=your_openrouter_client,
    config={'syntax_max_retries': 3, 'api_max_retries': 6}
)

# Handle an error
result, success = await error_system.handle_error(
    error=your_exception,
    operation_context="Your operation description"
)
```

---

## 📋 **Core Classes**

### **KGoTErrorManagementSystem**

Main orchestrator for all error management operations.

#### Methods

```python
async def handle_error(
    error: Exception,
    operation_context: str,
    error_type: Optional[ErrorType] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> Tuple[Any, bool]
```
**Purpose**: Handle any type of error with appropriate recovery strategy  
**Returns**: `(recovery_result, success_flag)`

```python
def get_comprehensive_statistics() -> Dict[str, Any]
```
**Purpose**: Get detailed error management statistics  
**Returns**: Statistics dictionary with metrics and health data

```python
def cleanup()
```
**Purpose**: Clean up resources and containers

---

### **SyntaxErrorManager**

Handles LLM-generated syntax errors with intelligent correction.

#### Methods

```python
async def handle_syntax_error(
    problematic_content: str,
    operation_context: str,
    error_details: str
) -> Tuple[str, bool]
```
**Purpose**: Correct syntax errors using multiple strategies  
**Strategy**: Unicode escape → LangChain JSON parsing → LLM correction  
**Returns**: `(corrected_content, success_flag)`

---

### **APIErrorManager**

Manages API and system errors with exponential backoff.

#### Methods

```python
@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
async def execute_with_retry(
    operation: Callable,
    operation_name: str,
    *args, **kwargs
) -> Any
```
**Purpose**: Execute operations with automatic retry and exponential backoff  
**Configuration**: 6 attempts, 1-60 second intervals

---

### **PythonExecutorManager**

Secure containerized Python code execution.

#### Methods

```python
async def execute_code_safely(
    code: str,
    execution_context: str,
    timeout: Optional[int] = None
) -> Dict[str, Any]
```
**Purpose**: Execute Python code in secure Docker container  
**Security**: Memory limits, CPU limits, network disabled, read-only  
**Returns**: Execution result with output, errors, and metadata

---

### **ErrorRecoveryOrchestrator**

Implements majority voting and iterative refinement.

#### Methods

```python
async def execute_with_majority_voting(
    operation: Callable,
    operation_name: str,
    *args, **kwargs
) -> Tuple[Any, float]
```
**Purpose**: Execute operation multiple times and find consensus  
**Returns**: `(majority_result, confidence_score)`

```python
async def iterative_error_refinement(
    failed_operation: Callable,
    error_context: ErrorContext,
    refinement_strategy: Callable
) -> Tuple[Any, bool]
```
**Purpose**: Refine failed operations iteratively  
**Returns**: `(refined_result, success_flag)`

---

## 🔗 **Integration Classes**

### **KGoTAlitaErrorIntegrationOrchestrator**

Main integration coordinator for KGoT-Alita error management.

#### Methods

```python
async def handle_integrated_error(
    error: Exception,
    context: Dict[str, Any],
    error_source: str = 'unknown'
) -> Dict[str, Any]
```
**Purpose**: Handle errors with full KGoT-Alita integration  
**Features**: Cross-system recovery, Alita refinement integration

```python
def get_integration_health_report() -> Dict[str, Any]
```
**Purpose**: Generate comprehensive integration health report

---

### **AlitaRefinementBridge**

Connects KGoT error management with Alita's iterative refinement.

#### Methods

```python
async def execute_iterative_refinement_with_alita(
    failed_operation: Callable,
    error_context: ErrorContext,
    alita_context: Dict[str, Any]
) -> Tuple[Any, bool]
```
**Purpose**: Execute refinement using both KGoT and Alita capabilities

---

## 🛠️ **Tool Integration**

### **JavaScript Tool Bridge Integration**

```javascript
// Enhanced KGoT Tool Bridge with error management
const { KGoTToolBridge } = require('./kgot_core/integrated_tools/kgot_tool_bridge');

const toolBridge = new KGoTToolBridge({
  enableErrorManagement: true,
  enableAlitaIntegration: true,
  maxRetries: 3
});

// Execute tool with automatic error recovery
const result = await toolBridge.executeTool('tool_name', toolInput, context);

// Get error statistics
const errorStats = toolBridge.getErrorManagementStatistics();
```

#### Enhanced Methods

```javascript
async executeTool(toolName, toolInput, context = {})
```
**Enhanced Features**: Automatic retry, error recovery, statistics tracking

```javascript
getErrorManagementStatistics()
```
**Returns**: Comprehensive error statistics and health metrics

---

## 📊 **Data Types**

### **ErrorType Enum**

```python
class ErrorType(Enum):
    SYNTAX_ERROR = "syntax_error"
    API_ERROR = "api_error"
    SYSTEM_ERROR = "system_error"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT_ERROR = "timeout_error"
    SECURITY_ERROR = "security_error"
```

### **ErrorSeverity Enum**

```python
class ErrorSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
```

### **ErrorContext DataClass**

```python
@dataclass
class ErrorContext:
    error_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    timestamp: datetime
    original_operation: str
    error_message: str
    stack_trace: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## ⚙️ **Configuration**

### **Error Management Config**

```python
config = {
    'syntax_max_retries': 3,
    'api_max_retries': 6,
    'voting_rounds': 3,
    'consensus_threshold': 0.6,
    'container_timeout': 30,
    'memory_limit': '256m',
    'cpu_limit': '0.5'
}
```

### **Integration Config**

```python
@dataclass
class IntegrationConfig:
    enable_alita_refinement: bool = True
    enable_cross_system_recovery: bool = True
    enable_unified_logging: bool = True
    max_cross_system_retries: int = 2
    error_escalation_threshold: int = 3
```

---

## 🔍 **Usage Examples**

### **Basic Error Handling**

```python
# Handle syntax error
syntax_error = SyntaxError("Invalid JSON format")
result, success = await error_system.handle_error(
    error=syntax_error,
    operation_context="JSON parsing",
    severity=ErrorSeverity.MEDIUM
)
```

### **Secure Code Execution**

```python
# Execute code safely
code = "import math; print(math.sqrt(16))"
result = await error_system.python_executor.execute_code_safely(
    code=code,
    execution_context="Mathematical calculation",
    timeout=15
)
```

### **Tool Integration**

```javascript
// Tool execution with error management
const result = await toolBridge.executeTool('python_code_tool', {
  code: 'print("Hello, World!")',
  context: 'greeting'
});

if (result.success) {
  console.log('Tool executed successfully:', result.result);
} else {
  console.log('Tool execution failed:', result.error);
}
```

### **Health Monitoring**

```python
# Get system health
health_report = orchestrator.get_integration_health_report()
health_score = health_report['unified_metrics']['integration_health_score']

if health_score < 0.8:
    print(f"Warning: System health degraded ({health_score})")
```

---

## 📈 **Monitoring & Metrics**

### **Key Metrics**

```python
stats = error_system.get_comprehensive_statistics()

# Important metrics to monitor
error_recovery_rate = stats['kgot_error_management']['recovery_rate']
total_errors = stats['kgot_error_management']['total_errors_handled']
syntax_errors = stats['syntax_error_manager']['total_syntax_errors']
api_errors = stats['api_error_manager']['total_api_errors']
integration_health = stats['integration_health_score']
```

### **Health Thresholds**

- **Recovery Rate**: Should be > 95%
- **Integration Health**: Should be > 0.8
- **Average Recovery Time**: Should be < 30 seconds
- **Container Memory Usage**: Should be < 512MB

---

## 🚨 **Error Codes**

| Code | Type | Description | Recovery Strategy |
|------|------|-------------|------------------|
| `SYN001` | Syntax Error | JSON parsing failure | Unicode escape + LangChain correction |
| `API001` | API Error | Connection timeout | Exponential backoff retry |
| `EXE001` | Execution Error | Container failure | Resource adjustment + retry |
| `INT001` | Integration Error | Alita bridge failure | Fallback to basic recovery |
| `SEC001` | Security Error | Container security violation | Immediate termination |

---

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Container Creation Failed**
   ```bash
   # Check Docker status
   docker --version
   sudo systemctl status docker
   ```

2. **Import Errors**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Memory Issues**
   ```python
   # Increase container memory limit
   executor = PythonExecutorManager(memory_limit="512m")
   ```

4. **Integration Failures**
   ```python
   # Enable fallback mode
   config = IntegrationConfig(enable_cross_system_recovery=False)
   ```

---

## 📝 **Factory Functions**

```python
# Create error management system
create_kgot_error_management_system(llm_client, config=None)

# Create integration orchestrator
create_kgot_alita_error_integration(llm_client, config=None)

# Create Alita integrator
create_alita_integrator(config=None)
```

---

*For detailed implementation examples and advanced usage, refer to the main documentation file.* 