# Task 12: Alita Environment Management with KGoT Integration

## 📋 Overview

This document provides comprehensive documentation for **Task 12: Alita Environment Management with KGoT Integration**, a complete implementation of Alita Section 2.3.4 "Environment Management" integrated with KGoT containerization and error management capabilities.

The Environment Management system provides intelligent, automated creation and management of isolated execution environments for various programming projects, with advanced features including repository analysis, dependency resolution, automated recovery, and LangChain agent integration.

## 🏗️ Architecture

### Core Components

```
EnvironmentManager (Main Orchestrator)
├── EnvironmentMetadataParser (Repository Analysis)
├── CondaEnvironmentManager (Conda Environment Management)
├── EnvironmentProfileBuilder (Multi-Environment Support)
├── RecoveryManager (Automated Error Recovery)
└── LangChain Agent (Intelligent Management)
```

### Integration Points

- **KGoT Containerization** (Section 2.6): Enhanced environment isolation
- **KGoT Error Management** (Section 2.5): Automated recovery procedures
- **TextInspectorTool**: Repository metadata parsing
- **OpenRouter API**: LLM-powered intelligent analysis
- **Winston Logging**: Comprehensive operation logging

## 🔧 Component Details

### 1. EnvironmentMetadataParser

**Purpose**: Intelligent repository analysis using TextInspectorTool patterns

**Key Features**:
- Parses multiple file formats: `requirements.txt`, `pyproject.toml`, `setup.py`, `Pipfile`, `README.md`
- Extracts dependencies, environment variables, installation commands, system requirements
- LLM-enhanced analysis using OpenRouter API for intelligent framework detection
- Supports various configuration formats: `.env`, `config.yaml`, `config.json`

**File Format Support**:
```python
supported_files = {
    'requirements': ['requirements.txt', 'requirements-dev.txt', 'pyproject.toml', 'setup.py', 'Pipfile'],
    'setup': ['setup.py', 'setup.cfg', 'pyproject.toml', 'Makefile', 'install.sh'],
    'config': ['.env', '.env.example', 'config.yaml', 'config.yml', 'config.json'],
    'docs': ['README.md', 'README.rst', 'INSTALL.md', 'SETUP.md', 'docs/installation.md']
}
```

**Example Usage**:
```python
parser = EnvironmentMetadataParser(llm_client=openrouter_client)
metadata = await parser.parse_repository_metadata("/path/to/repository")
print(f"Python Version: {metadata.python_version}")
print(f"Framework: {metadata.framework}")
print(f"Dependencies: {metadata.dependencies}")
```

### 2. CondaEnvironmentManager

**Purpose**: Manages Conda environments with unique naming and parallel initialization

**Key Features**:
- Creates isolated Conda environments with unique names (task ID or repository hash-based)
- Parallel initialization avoiding administrative privileges
- Supports both conda install and pip install procedures
- Automatic environment validation and cleanup
- Activation/deactivation script generation

**Environment Naming Strategy**:
```python
# Task ID based: alita_task_001_django_1699123456
# Repository hash based: alita_a1b2c3d4_flask_1699123456
env_name = f"alita_{task_id or repo_hash}_{framework}_{timestamp}"
```

**Parallel Initialization Process**:
1. **Task 1**: Create base conda environment
2. **Task 2**: Prepare dependency installation commands
3. **Task 3**: Setup environment activation/deactivation scripts
4. **Task 4**: Install dependencies (sequential after base setup)
5. **Task 5**: Validate environment setup

### 3. EnvironmentProfileBuilder

**Purpose**: Constructs isolated execution profiles for multiple environment types

**Supported Environment Types**:
- **CONDA**: Anaconda/Miniconda environments
- **DOCKER**: Docker containers with KGoT integration
- **SARUS**: HPC-optimized containers
- **VENV**: Python virtual environments
- **SYSTEM**: System-level environments

**Environment Type Selection Logic**:
```python
def determine_optimal_environment_type(metadata, options):
    # Complex scientific dependencies → CONDA
    if has_scientific_packages(metadata.dependencies):
        return EnvironmentType.CONDA
    
    # Simple Python projects → VENV
    if is_simple_python_project(metadata):
        return EnvironmentType.VENV
    
    # Container preference → DOCKER
    if options.get('prefer_containers'):
        return EnvironmentType.DOCKER
    
    # Default for Python → CONDA
    return EnvironmentType.CONDA
```

**Profile Caching**: Implements intelligent caching with validation to reuse compatible environments.

### 4. RecoveryManager

**Purpose**: Automated recovery procedures following KGoT error management patterns

**Recovery Procedures**:

1. **Dependency Resolution Recovery**:
   - Relax version constraints
   - Find minimal dependency sets
   - Try alternative packages
   - Max attempts: 3, Backoff: exponential

2. **Environment Creation Recovery**:
   - Cleanup partial environments
   - Try alternative environment types
   - Use system environment fallback
   - Max attempts: 3, Backoff: linear

3. **Container Deployment Recovery**:
   - Reduce container resources
   - Try alternative base images
   - Fallback to local environment
   - Max attempts: 2, Backoff: exponential

**Recovery Strategy Example**:
```python
async def execute_recovery(error_context, failed_profile, options):
    applicable_procedures = find_applicable_procedures(error_context.error_type)
    
    for procedure in applicable_procedures:
        for attempt in range(procedure.max_attempts):
            for step in procedure.recovery_steps:
                if await step(failed_profile, error_context, options):
                    if await validate_recovery(failed_profile):
                        return True, failed_profile
            
            await apply_backoff(procedure.backoff_strategy, attempt)
    
    # Execute fallback action
    return await procedure.fallback_action(failed_profile, error_context)
```

### 5. EnvironmentManager (Main Orchestrator)

**Purpose**: Coordinates all components and provides the main interface

**Key Capabilities**:
- Environment lifecycle management (create, track, cleanup)
- OpenRouter LLM integration for intelligent decision-making
- LangChain agent development for enhanced user interaction
- Comprehensive statistics and monitoring
- Error recovery coordination

**LangChain Agent Tools**:
- `analyze_repository_metadata`: Repository analysis tool
- `recommend_environment_optimization`: Environment optimization recommendations
- `resolve_dependency_conflicts`: Dependency conflict resolution

## 🚀 Usage Examples

### Basic Environment Creation

```python
from alita_core.environment_manager import EnvironmentManager, EnvironmentType

# Initialize Environment Manager
env_manager = EnvironmentManager(
    openrouter_api_key="your_openrouter_key",
    container_orchestrator=kgot_container_system,
    error_management_system=kgot_error_system
)

# Create environment for a repository
profile = await env_manager.create_environment(
    repository_path="/path/to/your/project",
    task_id="my_task_001",
    environment_type=EnvironmentType.CONDA,
    options={
        'prefer_containers': False,
        'optimize_for_performance': True
    }
)

print(f"Environment created: {profile.profile_id}")
print(f"Python executable: {profile.python_executable}")
print(f"Status: {profile.status.value}")
```

### Advanced Configuration

```python
# Configuration with all options
config = {
    'conda_base_path': '/opt/miniconda3',
    'cache_enabled': True,
    'parallel_initialization': True,
    'recovery_enabled': True,
    'logging_level': 'INFO'
}

env_manager = EnvironmentManager(
    openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
    container_orchestrator=container_system,
    error_management_system=error_system,
    config=config
)
```

### Environment Management Operations

```python
# List all active environments
environments = await env_manager.list_environments()
for env in environments:
    print(f"- {env.profile_id} ({env.environment_type.value})")

# Get specific environment
profile = await env_manager.get_environment("profile_id_here")

# Get statistics
stats = env_manager.get_environment_statistics()
print(f"Total environments: {stats['total_active_environments']}")
print(f"Environment types: {stats['environment_types']}")

# Cleanup environment
success = await env_manager.cleanup_environment("profile_id_here")

# Cleanup all environments
results = await env_manager.cleanup_all_environments()
```

### Using with LangChain Agent

```python
# If LangChain agent is available
if env_manager.agent_executor:
    response = await env_manager.agent_executor.ainvoke({
        "input": "Analyze this Python project and recommend the best environment setup"
    })
    print(response["output"])
```

## 📊 Data Structures

### EnvironmentMetadata

```python
@dataclass
class EnvironmentMetadata:
    repository_path: str
    language: Optional[str] = None
    framework: Optional[str] = None
    python_version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    system_requirements: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    installation_commands: List[str] = field(default_factory=list)
    requirements_files: List[str] = field(default_factory=list)
    setup_files: List[str] = field(default_factory=list)
    readme_content: Optional[str] = None
    main_script: Optional[str] = None
```

### EnvironmentProfile

```python
@dataclass
class EnvironmentProfile:
    profile_id: str
    name: str
    environment_type: EnvironmentType
    base_path: str
    python_executable: Optional[str] = None
    conda_env_name: Optional[str] = None
    container_config: Optional[Dict[str, Any]] = None
    metadata: Optional[EnvironmentMetadata] = None
    status: InitializationStatus = InitializationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    activation_script: Optional[str] = None
    deactivation_script: Optional[str] = None
```

### ErrorContext

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
    recovery_attempts: int = 0
```

## 🔧 Configuration

### Environment Variables

```bash
# Required for LLM integration
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional conda configuration
CONDA_PREFIX_1=/path/to/conda/base
CONDA_ROOT=/path/to/conda

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE_PATH=/var/log/alita/environment_manager.log
```

### Configuration File (config.json)

```json
{
  "conda_base_path": "/opt/miniconda3",
  "cache_enabled": true,
  "cache_ttl": 3600,
  "parallel_initialization": true,
  "max_parallel_workers": 4,
  "recovery_enabled": true,
  "max_recovery_attempts": 3,
  "container_integration": {
    "docker_enabled": true,
    "sarus_enabled": false,
    "default_memory_limit": "2g",
    "default_cpu_limit": "1"
  },
  "logging": {
    "level": "INFO",
    "winston_compatible": true,
    "structured_logging": true
  }
}
```

## 🏃‍♂️ Running the Demo

```bash
# Navigate to the alita core directory
cd alita-kgot-enhanced/alita_core

# Run the demonstration
python environment_manager.py
```

**Expected Output**:
```
🚀 Alita Environment Manager - Task 12 Implementation Demo
============================================================
📋 Initializing Environment Manager...
📁 Repository: .
🔖 Task ID: demo_task_001

🔧 Creating environment...
✅ Environment created successfully!
   Profile ID: conda_demo_task_001_1699123456
   Environment Type: conda
   Status: ready
   Python Executable: /opt/miniconda3/envs/alita_demo_task_001_1699123456/bin/python

📊 Environment Statistics:
   total_active_environments: 1
   environment_types: {'conda': 1, 'docker': 0, 'venv': 0, 'system': 0, 'sarus': 0}
   environment_statuses: {'ready': 1, 'pending': 0, 'initializing': 0, 'error': 0}
   total_created: 1
   recovery_statistics: {}

📋 Active Environments:
   - conda_demo_task_001_1699123456 (conda) - ready

🧹 Cleanup demonstration...
   Cleanup successful: True

✨ Environment Manager demonstration complete!
🎯 Task 12 implementation successfully demonstrated!
```

## 🔍 API Reference

### EnvironmentManager Class

#### Constructor
```python
def __init__(self, 
             openrouter_api_key: Optional[str] = None,
             container_orchestrator: Optional[Any] = None,
             error_management_system: Optional[Any] = None,
             config: Optional[Dict[str, Any]] = None)
```

#### Main Methods

**create_environment()**
```python
async def create_environment(self, 
                           repository_path: str,
                           task_id: Optional[str] = None,
                           environment_type: Optional[EnvironmentType] = None,
                           options: Optional[Dict[str, Any]] = None) -> EnvironmentProfile
```

**get_environment()**
```python
async def get_environment(self, profile_id: str) -> Optional[EnvironmentProfile]
```

**list_environments()**
```python
async def list_environments(self) -> List[EnvironmentProfile]
```

**cleanup_environment()**
```python
async def cleanup_environment(self, profile_id: str) -> bool
```

**get_environment_statistics()**
```python
def get_environment_statistics(self) -> Dict[str, Any]
```

### EnvironmentMetadataParser Class

**parse_repository_metadata()**
```python
async def parse_repository_metadata(self, repository_path: str) -> EnvironmentMetadata
```

### CondaEnvironmentManager Class

**create_environment()**
```python
async def create_environment(self, 
                           metadata: EnvironmentMetadata, 
                           task_id: Optional[str] = None,
                           force_recreate: bool = False) -> EnvironmentProfile
```

### RecoveryManager Class

**execute_recovery()**
```python
async def execute_recovery(self, 
                         error_context: ErrorContext,
                         failed_profile: EnvironmentProfile,
                         recovery_options: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[EnvironmentProfile]]
```

## 🔗 Integration Guidelines

### KGoT Containerization Integration

The Environment Manager integrates with KGoT's containerization system through the `container_orchestrator` parameter:

```python
# Initialize with KGoT container orchestrator
from kgot_core.containerization import ContainerOrchestrator

container_system = ContainerOrchestrator()
env_manager = EnvironmentManager(
    container_orchestrator=container_system
)

# Container environments will be automatically managed
profile = await env_manager.create_environment(
    repository_path="/path/to/project",
    environment_type=EnvironmentType.DOCKER
)
```

### KGoT Error Management Integration

Integration with KGoT's error management system enables automatic recovery:

```python
from kgot_core.error_management import ErrorManagementSystem

error_system = ErrorManagementSystem()
env_manager = EnvironmentManager(
    error_management_system=error_system
)

# Automatic recovery will be triggered on failures
try:
    profile = await env_manager.create_environment(repository_path)
except Exception as e:
    # Recovery procedures will have been attempted automatically
    logger.error(f"Environment creation failed even after recovery: {e}")
```

### LangChain Agent Development

The system includes LangChain agents following the user's hard rule for agent development:

```python
# Agents are automatically initialized if OpenRouter API key is available
if env_manager.agent_executor:
    # Use the agent for intelligent environment management
    response = await env_manager.agent_executor.ainvoke({
        "input": "What's the best environment type for a Django project with PostgreSQL?"
    })
    
    print(response["output"])
    # Expected: Detailed recommendation with reasoning
```

## 📝 Logging

### Winston-Compatible Logging Structure

All operations use comprehensive logging with Winston-compatible structure:

```python
logger.info("Environment creation successful", extra={
    'operation': 'CREATE_ENV_SUCCESS',
    'profile_id': profile.profile_id,
    'environment_type': profile.environment_type.value,
    'duration': time.time() - start_time,
    'dependencies_count': len(profile.metadata.dependencies),
    'python_version': profile.metadata.python_version
})
```

### Log Levels Used

- **DEBUG**: Detailed operation steps, parsing details
- **INFO**: Major operation milestones, successful completions
- **WARNING**: Recoverable errors, fallback usage
- **ERROR**: Serious errors requiring attention

### Log Files

- **Combined logs**: `logs/alita/combined.log`
- **Error logs**: `logs/alita/error.log`
- **HTTP logs**: `logs/alita/http.log` (if web interface is used)

## 🐛 Troubleshooting

### Common Issues

**1. Conda Not Found**
```
Error: conda command not found
Solution: Install Anaconda/Miniconda or add conda to PATH
```

**2. OpenRouter API Key Missing**
```
Warning: OpenRouter API key not available, LLM features disabled
Solution: Set OPENROUTER_API_KEY environment variable
```

**3. Permission Denied During Environment Creation**
```
Error: Permission denied when creating conda environment
Solution: Ensure user has write permissions to conda envs directory
```

**4. Dependency Conflicts**
```
Error: Could not solve for environment
Solution: Automatic recovery will attempt to resolve conflicts by:
- Relaxing version constraints
- Using minimal dependency sets
- Trying alternative packages
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger('alita.environment_manager').setLevel(logging.DEBUG)

env_manager = EnvironmentManager(config={
    'logging': {
        'level': 'DEBUG',
        'structured_logging': True
    }
})
```

### Recovery Information

Check recovery statistics to understand common failure patterns:

```python
stats = env_manager.get_environment_statistics()
recovery_stats = stats['recovery_statistics']

for procedure_id, success_count in recovery_stats.items():
    print(f"Recovery procedure '{procedure_id}' succeeded {success_count} times")
```

## 📈 Performance Considerations

### Optimization Strategies

1. **Profile Caching**: Reuse compatible environments when possible
2. **Parallel Initialization**: Create environments concurrently when safe
3. **Minimal Dependencies**: Identify and install only essential packages
4. **Container Optimization**: Use lightweight base images and resource limits

### Resource Management

- **Memory Usage**: Monitor conda environment sizes
- **Disk Space**: Implement cleanup policies for old environments
- **CPU Usage**: Limit parallel operations based on system capacity
- **Network**: Cache package downloads when possible

### Scaling Considerations

- **Multiple Users**: Implement user-specific environment isolation
- **High Throughput**: Use environment pools for common configurations
- **Distributed Systems**: Consider container orchestration for multiple nodes

## 🔒 Security Considerations

### Environment Isolation

- Each environment is isolated with unique naming
- Container integration provides additional security boundaries
- System environment access is carefully controlled

### Dependency Security

- Package verification through conda and pip security features
- Automated scanning for known vulnerabilities (future enhancement)
- Dependency pinning for reproducible builds

### Access Control

- User-specific environment creation and management
- Proper cleanup to prevent resource leaks
- Logging of all environment operations for audit trails

## 🚀 Future Enhancements

### Planned Features

1. **Multi-User Support**: User-specific environment isolation and management
2. **Environment Templates**: Pre-configured templates for common frameworks
3. **Resource Quotas**: Per-user resource limits and monitoring
4. **Environment Sharing**: Secure sharing of environments between users
5. **Advanced Analytics**: Detailed usage analytics and optimization recommendations
6. **GUI Interface**: Web-based interface for environment management
7. **CI/CD Integration**: Integration with continuous integration pipelines
8. **Cloud Provider Support**: Support for cloud-based environment deployment

### Extension Points

The system is designed for extensibility:

- **Custom Environment Types**: Add support for new environment types
- **Additional Recovery Strategies**: Implement custom recovery procedures
- **Enhanced Metadata Parsing**: Support for additional file formats
- **Alternative Container Systems**: Support for Kubernetes, Podman, etc.
- **Plugin Architecture**: Modular plugins for specialized functionality

## 📚 References

### Related Documentation

- [Alita Section 2.3.4 Environment Management](../architecture/alita_environment_management.md)
- [KGoT Section 2.6 Containerization](../architecture/kgot_containerization.md)
- [KGoT Section 2.5 Error Management](../api/KGOT_ERROR_MANAGEMENT_API.md)
- [Winston Logging Configuration](../config/logging/winston_config.js)

### External Dependencies

- **LangChain**: Agent development framework
- **OpenRouter API**: LLM integration service
- **Anaconda/Miniconda**: Conda environment management
- **Docker**: Container runtime (optional)
- **Sarus**: HPC container runtime (optional)

### Standards and Patterns

- **JSDoc3**: Code documentation standard
- **Winston**: Logging pattern compatibility
- **Async/Await**: Asynchronous operation patterns
- **Error-First Callbacks**: Error handling patterns
- **Factory Pattern**: Component initialization
- **Observer Pattern**: Event-driven architecture

---

## 📄 License and Contributing

This implementation is part of the Alita-KGoT Enhanced system. For contributing guidelines, please refer to the main project documentation.

**Task 12 Status**: ✅ **COMPLETE**  
**Implementation Date**: November 2024  
**Last Updated**: November 2024  
**Version**: 1.0.0 