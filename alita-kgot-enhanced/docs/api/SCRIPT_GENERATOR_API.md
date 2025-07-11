# Script Generator API Reference

## Overview

Complete API reference for the Alita Script Generation Tool implementing Section 2.3.2 architecture.

## Classes

### ScriptGenerationConfig

Configuration dataclass for the script generation system.

```python
@dataclass
class ScriptGenerationConfig:
    # OpenRouter API configuration (per user rules)
    openrouter_api_key: str = os.getenv('OPENROUTER_API_KEY', '')
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Model specialization per user rules
    orchestration_model: str = "x-ai/grok-4"   # Main reasoning
    webagent_model: str = "anthropic/claude-4-sonnet"    # Web agent capabilities
    vision_model: str = "openai/o3"                      # Vision processing
    
    # Integration endpoints
    mcp_brainstorming_endpoint: str = "http://localhost:8001/api/mcp-brainstorming"
    web_agent_endpoint: str = "http://localhost:8000/api/web-agent"
    kgot_python_tool_endpoint: str = "http://localhost:16000/run"
    
    # Generation settings
    supported_languages: List[str] = field(default_factory=lambda: ['python', 'bash', 'javascript', 'dockerfile'])
    max_script_size: int = 1048576  # 1MB default
    execution_timeout: int = 300    # 5 minutes default
    enable_template_caching: bool = True
    template_directory: str = "./templates/rag_mcp"
    temp_directory: str = "/tmp"
    cleanup_on_completion: bool = True
    log_level: str = "INFO"
```

### ScriptGeneratingTool

Main orchestrator class for script generation.

#### Constructor

```python
def __init__(self, config: Optional[ScriptGenerationConfig] = None)
```

**Parameters:**
- `config`: Optional configuration object. Uses defaults if not provided.

**Example:**
```python
from alita_core.script_generator import ScriptGeneratingTool, ScriptGenerationConfig

# Default configuration
generator = ScriptGeneratingTool()

# Custom configuration
config = ScriptGenerationConfig(
    openrouter_api_key="your_key",
    orchestration_model="x-ai/grok-4"
)
generator = ScriptGeneratingTool(config)
```

#### Core Methods

##### generate_script()

Primary method for generating complete scripts with environment management.

```python
async def generate_script(
    self,
    task_description: str,
    requirements: Optional[Dict[str, Any]] = None,
    github_urls: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None
) -> GeneratedScript
```

**Parameters:**

- `task_description` (str): High-level description of the script's purpose
- `requirements` (Dict[str, Any], optional): Technical requirements and specifications
- `github_urls` (List[str], optional): Reference GitHub repositories for analysis
- `options` (Dict[str, Any], optional): Additional generation options

**Requirements Dictionary Structure:**
```python
requirements = {
    # Core settings
    'language': str,                    # Target programming language
    'dependencies': List[str],          # Required packages/libraries
    'framework': str,                   # Framework to use (optional)
    
    # Features
    'features': List[str],              # Specific features to include
    'auth_methods': List[str],          # Authentication methods
    'output_format': str,               # Output format (json, csv, etc.)
    'error_handling': str,              # Error handling level
    'logging_level': str,               # Logging configuration
    'async_support': bool,              # Async/await support
    'testing': str,                     # Testing framework
    
    # Performance & behavior
    'performance': str,                 # Performance level
    'memory_efficient': bool,           # Memory optimization
    'concurrent': bool,                 # Concurrency support
    'caching': bool,                    # Response caching
    
    # Environment
    'target_env': str,                  # Target environment
    'base_image': str,                  # Docker base image
    'expose_port': int,                 # Port to expose
    'stages': List[str],                # Multi-stage builds
    
    # Data handling
    'data_sources': List[str],          # Data source types
    'validation': str,                  # Data validation level
    'batch_processing': bool,           # Batch processing support
    
    # Security
    'security': str,                    # Security level
    'encryption': bool,                 # Encryption support
    'input_validation': bool,           # Input validation
}
```

**Options Dictionary Structure:**
```python
options = {
    # Generation options
    'include_tests': bool,              # Generate test cases
    'docker_containerize': bool,        # Create Dockerfile
    'documentation': str,               # Documentation format
    'examples': bool,                   # Include usage examples
    
    # Processing options
    'parallel_processing': bool,        # Parallel execution
    'progress_tracking': bool,          # Progress indicators
    'monitoring': bool,                 # Monitoring/metrics
    'error_recovery': bool,             # Error recovery mechanisms
    
    # Output options
    'code_comments': str,               # Comment level
    'type_hints': bool,                 # Type annotations
    'docstrings': str,                  # Docstring format
    'format_code': bool,                # Code formatting
}
```

**Returns:** `GeneratedScript` object

**Example:**
```python
script = await generator.generate_script(
    task_description="Create a REST API client with authentication",
    requirements={
        'language': 'python',
        'dependencies': ['requests', 'pydantic'],
        'auth_methods': ['oauth2', 'api_key'],
        'response_caching': True,
        'retry_strategy': 'exponential_backoff',
        'type_hints': True
    },
    github_urls=[
        'https://github.com/requests/requests',
        'https://github.com/pydantic/pydantic'
    ],
    options={
        'include_tests': True,
        'documentation': 'sphinx',
        'async_client': True,
        'examples': True
    }
)
```

##### get_session_stats()

Retrieve current session statistics.

```python
def get_session_stats(self) -> Dict[str, Any]
```

**Returns:**
```python
{
    'current_session_id': str,          # Current session identifier
    'total_generations': int,           # Total scripts generated
    'successful_generations': int,      # Successful generations
    'failed_generations': int,          # Failed generations
    'success_rate': float,              # Success percentage
    'avg_duration': float,              # Average generation time (seconds)
    'total_duration': float,            # Total time spent
    'languages_used': List[str],        # Languages generated
    'models_used': List[str],           # Models utilized
    'start_time': datetime,             # Session start time
    'last_generation': Optional[datetime] # Last generation time
}
```

**Example:**
```python
stats = generator.get_session_stats()
print(f"Generated {stats['total_generations']} scripts")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Average duration: {stats['avg_duration']:.2f}s")
```

##### get_generation_history()

Retrieve complete generation history for the session.

```python
def get_generation_history(self) -> List[Dict[str, Any]]
```

**Returns:** List of generation records with detailed information.

**Example:**
```python
history = generator.get_generation_history()
for entry in history:
    print(f"Script: {entry['script_name']} - Success: {entry['success']}")
    print(f"Duration: {entry['duration']:.2f}s")
    print(f"Language: {entry['language']}")
```

##### reset_session()

Reset the current session, clearing history and statistics.

```python
def reset_session(self) -> str
```

**Returns:** New session ID

**Example:**
```python
new_session_id = generator.reset_session()
print(f"New session started: {new_session_id}")
```

#### Specialized Methods

##### _create_specialized_llm()

Create specialized LLM instances for specific purposes.

```python
def _create_specialized_llm(self, purpose: str) -> Optional[ChatOpenAI]
```

**Parameters:**
- `purpose`: One of "orchestration", "webagent", "vision"

**Returns:** Configured LLM instance or None if API key unavailable

**Example:**
```python
# Create web-focused LLM (uses claude-4-sonnet)
web_llm = generator._create_specialized_llm('webagent')

# Create vision LLM (uses o3)
vision_llm = generator._create_specialized_llm('vision')

# Create orchestration LLM (uses grok-4)
orchestration_llm = generator._create_specialized_llm('orchestration')
```

## Data Models

### GeneratedScript

Complete script generation result with all components.

```python
@dataclass
class GeneratedScript:
    id: str                                 # Unique script identifier (UUID)
    name: str                               # Script filename
    description: str                        # Script description
    language: str                           # Programming language
    code: str                               # Complete script code
    environment_spec: EnvironmentSpec       # Environment requirements
    setup_script: str                       # Environment setup commands
    cleanup_script: str                     # Cleanup commands
    execution_instructions: str             # How to run the script
    test_cases: List[str]                   # Generated test cases
    documentation: str                      # Complete documentation
    github_sources: List[GitHubLinkInfo]    # Source repositories analyzed
    generation_metadata: Dict[str, Any]     # Generation metadata
    created_at: datetime                    # Creation timestamp
```

### EnvironmentSpec

Environment specification for script execution.

```python
@dataclass
class EnvironmentSpec:
    language: str                           # Programming language
    version: str                            # Language version
    dependencies: List[str]                 # Required packages
    environment_variables: Dict[str, str]   # Environment variables
    system_requirements: List[str]          # System-level requirements
    setup_commands: List[str]               # Setup commands
    cleanup_commands: List[str]             # Cleanup commands
    runtime_flags: List[str]                # Runtime flags/options
```

### SubtaskDescription

Data structure for MCP Brainstorming integration.

```python
@dataclass
class SubtaskDescription:
    id: str                                 # Subtask identifier
    title: str                              # Subtask title
    description: str                        # Detailed description
    requirements: Dict[str, Any]            # Technical requirements
    dependencies: List[str]                 # Dependencies on other subtasks
    priority: int                           # Priority level (1-10)
    estimated_complexity: int               # Complexity estimate (1-10)
    success_criteria: List[str]             # Success criteria
```

### GitHubLinkInfo

GitHub repository information and analysis.

```python
@dataclass
class GitHubLinkInfo:
    url: str                                # Repository URL
    owner: str                              # Repository owner
    repo: str                               # Repository name
    branch: str                             # Branch analyzed
    description: str                        # Repository description
    primary_language: str                   # Primary programming language
    topics: List[str]                       # Repository topics
    star_count: int                         # GitHub stars
    relevant_files: List[str]               # Relevant files found
    code_snippets: List[Dict[str, str]]     # Extracted code snippets
    analysis_summary: str                   # AI analysis summary
```

## Component Classes

### MCPBrainstormingBridge

Interface with MCP Brainstorming for task decomposition.

```python
class MCPBrainstormingBridge:
    def __init__(self, config: ScriptGenerationConfig)
    
    async def initialize_session(self, task_description: str) -> str
    async def receive_subtask_descriptions(self, requirements: Dict[str, Any]) -> List[SubtaskDescription]
    async def validate_decomposition(self, subtasks: List[SubtaskDescription]) -> bool
    def cleanup_session(self) -> None
```

### GitHubLinksProcessor

Process and analyze GitHub repositories.

```python
class GitHubLinksProcessor:
    def __init__(self, config: ScriptGenerationConfig)
    
    async def process_github_links(self, github_urls: List[str], context: Dict[str, Any]) -> List[GitHubLinkInfo]
    async def extract_code_snippets(self, links_info: List[GitHubLinkInfo], requirements: Dict[str, Any]) -> Dict[str, List[str]]
    async def analyze_repository_structure(self, link_info: GitHubLinkInfo) -> Dict[str, Any]
```

### RAGMCPTemplateEngine

Template-based script generation with RAG capabilities.

```python
class RAGMCPTemplateEngine:
    def __init__(self, config: ScriptGenerationConfig)
    
    async def generate_from_template(self, template_type: str, context: Dict[str, Any]) -> str
    def load_templates(self) -> None
    def get_available_templates(self) -> List[str]
    def cache_template(self, template_name: str, content: str) -> None
```

### EnvironmentSetupGenerator

Generate environment setup and cleanup scripts.

```python
class EnvironmentSetupGenerator:
    def __init__(self, config: ScriptGenerationConfig)
    
    async def generate_setup_script(self, environment_spec: EnvironmentSpec) -> str
    async def generate_cleanup_script(self, environment_spec: EnvironmentSpec) -> str
    def validate_environment_spec(self, spec: EnvironmentSpec) -> bool
    async def create_environment_spec(self, requirements: Dict[str, Any]) -> EnvironmentSpec
```

### KGoTPythonToolBridge

Integration with KGoT Python Code Tool for enhanced generation.

```python
class KGoTPythonToolBridge:
    def __init__(self, config: ScriptGenerationConfig)
    
    async def enhance_generated_code(self, code: str, context: Dict[str, Any]) -> str
    async def validate_code_quality(self, code: str) -> Dict[str, Any]
    async def suggest_improvements(self, code: str) -> List[str]
    def is_available(self) -> bool
```

## Error Handling

All methods implement comprehensive error handling with specific exception types:

```python
# Custom exceptions
class ScriptGenerationError(Exception): pass
class ConfigurationError(ScriptGenerationError): pass
class ModelInitializationError(ScriptGenerationError): pass
class TemplateLoadError(ScriptGenerationError): pass
class IntegrationError(ScriptGenerationError): pass
class ValidationError(ScriptGenerationError): pass
```

## Logging

Winston-compatible logging is implemented throughout with operation context:

```python
logger.info("Script generation started", extra={
    'operation': 'SCRIPT_GENERATION',
    'session_id': self.current_session_id,
    'task_description': task_description[:100],
    'language': requirements.get('language', 'unspecified')
})
```

## Usage Examples

### Basic Usage
```python
import asyncio
from alita_core.script_generator import ScriptGeneratingTool

async def basic_example():
    generator = ScriptGeneratingTool()
    
    script = await generator.generate_script(
        task_description="Create a CSV data processor",
        requirements={'language': 'python', 'dependencies': ['pandas']}
    )
    
    print(f"Generated: {script.name}")
    return script

script = asyncio.run(basic_example())
```

### Advanced Configuration
```python
from alita_core.script_generator import ScriptGenerationConfig, ScriptGeneratingTool

config = ScriptGenerationConfig(
    openrouter_api_key="your_key",
    orchestration_model="x-ai/grok-4",
    webagent_model="anthropic/claude-4-sonnet",
    vision_model="openai/o3",
    enable_template_caching=True,
    max_script_size=2097152  # 2MB
)

generator = ScriptGeneratingTool(config)
```

### Error Handling
```python
from alita_core.script_generator import ScriptGeneratingTool, ScriptGenerationError

async def robust_generation():
    generator = ScriptGeneratingTool()
    
    try:
        script = await generator.generate_script(
            task_description="Complex task",
            requirements={'language': 'python'}
        )
        return script
    except ScriptGenerationError as e:
        print(f"Generation failed: {e}")
        # Handle gracefully
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

result = asyncio.run(robust_generation())
```

---

**Generated by**: Alita Script Generation Tool API Documentation System  
**Last Updated**: 2025-06-28  
**Version**: 1.0.0 