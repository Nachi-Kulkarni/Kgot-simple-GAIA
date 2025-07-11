# Alita Script Generation Tool - Complete Guide

## 🎯 Overview

The **Alita Script Generation Tool** is an advanced AI-powered script generation system implementing Alita Section 2.3.2 architecture with full KGoT integration. It leverages multiple specialized AI models through OpenRouter to generate high-quality, executable scripts with comprehensive environment management.

## 🏗️ Architecture & Models

### Model Specialization (Per User Rules)
- **🤖 Orchestration**: `x-ai/grok-4` - Main reasoning and coordination
- **🌐 Web Agent**: `anthropic/claude-4-sonnet` - GitHub analysis and web tasks  
- **👁️ Vision**: `openai/o3` - Visual processing and multimodal tasks

### Core Components
```
ScriptGeneratingTool (Main Orchestrator)
├── LangChain Agent Executor (grok-4)
├── MCP Brainstorming Bridge (Task decomposition)
├── GitHub Links Processor (claude-4-sonnet)
├── RAG-MCP Template Engine (Template generation)
├── Environment Setup Generator (Setup/cleanup)
└── KGoT Python Tool Bridge (Code enhancement)
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Set OpenRouter API key
export OPENROUTER_API_KEY="your_openrouter_api_key"

# Verify installation
cd alita-kgot-enhanced
python3 -c "from alita_core.script_generator import ScriptGeneratingTool; print('✅ Ready!')"
```

### 2. Basic Usage
```python
import asyncio
from alita_core.script_generator import ScriptGeneratingTool

async def generate_simple_script():
    generator = ScriptGeneratingTool()
    
    script = await generator.generate_script(
        task_description="Create a CSV data processor with error handling",
        requirements={
            'language': 'python',
            'dependencies': ['pandas', 'logging'],
            'features': ['error_handling', 'progress_tracking']
        }
    )
    
    print(f"Generated: {script.name}")
    print(f"Setup command: {script.setup_script}")
    return script

# Run generation
script = asyncio.run(generate_simple_script())
```

## 📋 API Reference

### ScriptGeneratingTool Class

#### Main Generation Method
```python
async def generate_script(
    task_description: str,
    requirements: Optional[Dict[str, Any]] = None,
    github_urls: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None
) -> GeneratedScript
```

**Parameters:**
- `task_description`: High-level description of what the script should do
- `requirements`: Technical requirements (language, dependencies, features)
- `github_urls`: Reference repositories for code patterns
- `options`: Generation options (testing, containerization, etc.)

**Example Requirements:**
```python
requirements = {
    'language': 'python',           # Target language
    'dependencies': ['requests'],   # Required packages
    'output_format': 'json',       # Output format
    'error_handling': 'robust',    # Error handling level
    'logging_level': 'INFO',       # Logging configuration
    'async_support': True,         # Async/await support
    'testing': 'pytest'            # Testing framework
}
```

#### Configuration
```python
from alita_core.script_generator import ScriptGenerationConfig

config = ScriptGenerationConfig(
    openrouter_api_key="your_key",
    orchestration_model="x-ai/grok-4",
    webagent_model="anthropic/claude-4-sonnet", 
    vision_model="openai/o3",
    supported_languages=['python', 'bash', 'javascript', 'dockerfile'],
    enable_template_caching=True
)

generator = ScriptGeneratingTool(config)
```

### GeneratedScript Object

Complete script package with all components:

```python
class GeneratedScript:
    id: str                    # Unique identifier
    name: str                  # Script filename
    description: str           # What the script does
    language: str             # Programming language
    code: str                 # Complete script code
    setup_script: str         # Environment setup commands
    cleanup_script: str       # Cleanup commands
    execution_instructions: str # How to run
    test_cases: List[str]     # Generated tests
    documentation: str        # Complete docs
    environment_spec: EnvironmentSpec # Requirements
    github_sources: List[GitHubLinkInfo] # Sources used
    created_at: datetime      # Generation timestamp
```

## 💡 Usage Examples

### Web Scraping Script
```python
async def web_scraper_example():
    generator = ScriptGeneratingTool()
    
    script = await generator.generate_script(
        task_description="Create a web scraper for product prices with rate limiting",
        requirements={
            'language': 'python',
            'dependencies': ['requests', 'beautifulsoup4', 'time'],
            'features': ['rate_limiting', 'retry_logic', 'csv_output'],
            'target_sites': ['ecommerce'],
            'respect_robots_txt': True
        },
        github_urls=[
            'https://github.com/scrapy/scrapy',
            'https://github.com/requests/requests'
        ],
        options={
            'include_tests': True,
            'docker_containerize': True,
            'monitoring': True
        }
    )
    
    # Save complete package
    with open(f'{script.name}.py', 'w') as f:
        f.write(script.code)
    
    with open(f'setup_{script.name}.sh', 'w') as f:
        f.write(script.setup_script)
    
    print(f"✅ Generated web scraper: {script.name}")
    print(f"📦 Includes {len(script.test_cases)} test cases")

asyncio.run(web_scraper_example())
```

### Data Processing Pipeline
```python
async def data_pipeline_example():
    generator = ScriptGeneratingTool()
    
    script = await generator.generate_script(
        task_description="Build ETL pipeline for large CSV files with validation",
        requirements={
            'language': 'python',
            'dependencies': ['pandas', 'numpy', 'sqlalchemy'],
            'data_sources': ['csv', 'database'],
            'validation': 'comprehensive',
            'performance': 'optimized',
            'memory_efficient': True
        },
        github_urls=[
            'https://github.com/pandas-dev/pandas',
            'https://github.com/apache/airflow'
        ],
        options={
            'parallel_processing': True,
            'progress_tracking': True,
            'error_recovery': True
        }
    )
    
    print(f"Generated ETL pipeline: {script.name}")
    print(f"Environment: {script.environment_spec.language} {script.environment_spec.version}")
    print(f"Dependencies: {', '.join(script.environment_spec.dependencies)}")

asyncio.run(data_pipeline_example())
```

### API Client Generator
```python
async def api_client_example():
    generator = ScriptGeneratingTool()
    
    script = await generator.generate_script(
        task_description="Generate REST API client with authentication and caching",
        requirements={
            'language': 'python',
            'dependencies': ['requests', 'cachetools', 'pydantic'],
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
            'async_client': True,
            'documentation': 'sphinx',
            'examples': True
        }
    )
    
    print(f"Generated API client: {script.name}")
    print("📖 Documentation:")
    print(script.documentation[:500] + "...")

asyncio.run(api_client_example())
```

### Multi-Language Project
```python
async def multi_language_project():
    generator = ScriptGeneratingTool()
    
    # Python data processor
    python_script = await generator.generate_script(
        task_description="Data preprocessing and ML model training",
        requirements={'language': 'python', 'dependencies': ['scikit-learn', 'pandas']}
    )
    
    # Bash deployment script  
    bash_script = await generator.generate_script(
        task_description="Automated deployment with health checks",
        requirements={'language': 'bash', 'target_env': 'linux', 'monitoring': True}
    )
    
    # JavaScript API server
    js_script = await generator.generate_script(
        task_description="Express.js API server with middleware",
        requirements={'language': 'javascript', 'framework': 'express', 'database': 'mongodb'}
    )
    
    # Dockerfile for containerization
    docker_script = await generator.generate_script(
        task_description="Multi-stage Docker build for the complete application",
        requirements={
            'language': 'dockerfile',
            'base_image': 'node:18-alpine',
            'stages': ['build', 'production'],
            'optimization': True
        }
    )
    
    print("🚀 Complete project generated:")
    print(f"  🐍 Python: {python_script.name}")
    print(f"  📜 Bash: {bash_script.name}")
    print(f"  🟨 JavaScript: {js_script.name}")
    print(f"  🐳 Docker: {docker_script.name}")

asyncio.run(multi_language_project())
```

## 🔧 Advanced Configuration

### Custom Template Directory
```python
config = ScriptGenerationConfig(
    template_directory="./my_custom_templates",
    enable_template_caching=True
)

# Create custom template
custom_template = """
#!/usr/bin/env python3
'''
Custom {{task_type}} Script
Generated: {{generation_date}}
Purpose: {{task_description}}
'''

import logging
{{imports}}

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    {{main_logic}}

if __name__ == "__main__":
    main()
"""
```

### Specialized Model Usage
```python
# Create specialized LLMs for different tasks
generator = ScriptGeneratingTool()

# Web-focused generation (uses claude-4-sonnet)
web_llm = generator._create_specialized_llm('webagent')

# Vision processing (uses o3)
vision_llm = generator._create_specialized_llm('vision')

# Main orchestration (uses grok-4)
orchestration_llm = generator._create_specialized_llm('orchestration')
```

## 🔍 Monitoring & Debugging

### Session Statistics
```python
generator = ScriptGeneratingTool()

# Generate some scripts...
await generator.generate_script("Create a file processor", {'language': 'python'})
await generator.generate_script("Build a web crawler", {'language': 'python'})

# Check statistics
stats = generator.get_session_stats()
print(f"Total generations: {stats['total_generations']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Average duration: {stats['avg_duration']:.2f}s")

# Get detailed history
history = generator.get_generation_history()
for entry in history:
    print(f"Script: {entry['script_name']} - Success: {entry['success']}")
```

### Logging Configuration
```python
import logging

# Enable comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(operation)s] %(message)s'
)

# Component-specific logging
logger = logging.getLogger('ScriptGenerator')
logger.info("Starting script generation session")
```

### Health Check
```python
async def system_health_check():
    generator = ScriptGeneratingTool()
    
    print("🔍 System Health Check:")
    print(f"  ✅ Config: {generator.config is not None}")
    print(f"  ✅ LLM: {generator.llm is not None}")
    print(f"  ✅ Templates: {len(generator.template_engine.templates)} loaded")
    
    # Test model connectivity
    if generator.config.openrouter_api_key:
        try:
            test_llm = generator._create_specialized_llm('orchestration')
            print(f"  ✅ OpenRouter: Connected")
        except Exception as e:
            print(f"  ❌ OpenRouter: {str(e)}")
    else:
        print(f"  ⚠️  OpenRouter: API key not set")

asyncio.run(system_health_check())
```

## 🔒 Security & Best Practices

### Safe Execution
```python
# Generated scripts are NOT automatically executed
script = await generator.generate_script("File processor", {'language': 'python'})

# Review before execution
print("Generated code preview:")
print(script.code[:500] + "...")

# Manual execution after review
if input("Execute this script? (y/N): ").lower() == 'y':
    exec(script.code)  # Only after manual review
```

### Environment Isolation
```python
# Use virtual environments for setup
script = await generator.generate_script(
    "Data analysis script", 
    {'language': 'python', 'dependencies': ['pandas']}
)

# Setup creates isolated environment
print("Setup script (creates virtual env):")
print(script.setup_script)
```

### API Key Management
```python
import os
from pathlib import Path

# Load from secure file
def load_api_key():
    key_file = Path.home() / '.config' / 'openrouter' / 'api_key'
    if key_file.exists():
        return key_file.read_text().strip()
    return os.getenv('OPENROUTER_API_KEY')

config = ScriptGenerationConfig(openrouter_api_key=load_api_key())
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Missing API Key
```
❌ Error: OPENROUTER_API_KEY not found in environment
✅ Solution: export OPENROUTER_API_KEY="your_key_here"
```

#### 2. Model Access Issues
```
❌ Error: Failed to initialize OpenRouter LLM
✅ Check: API key validity, network connection, OpenRouter credits
```

#### 3. Agent Executor Disabled
```
⚠️ Warning: LLM not available, agent executor will be disabled
✅ Impact: Falls back to template generation (still functional)
✅ Solution: Set valid OpenRouter API key
```

#### 4. Template Loading Failed
```
❌ Error: Failed to load template
✅ Check: Template directory exists, file permissions, syntax validity
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger('ScriptGenerator').setLevel(logging.DEBUG)

# Test specific components
generator = ScriptGeneratingTool()
try:
    script = await generator.generate_script("Test script", {'language': 'python'})
    print("✅ Generation successful")
except Exception as e:
    print(f"❌ Generation failed: {e}")
    logging.exception("Full traceback:")
```

## 📊 Performance Tips

### Optimize Generation Speed
```python
# Enable caching
config = ScriptGenerationConfig(
    enable_template_caching=True,
    temp_directory="/tmp/fast_ssd"  # Use fast storage
)

# Batch similar requests
tasks = [
    ("Data processor 1", {'language': 'python'}),
    ("Data processor 2", {'language': 'python'}),
    ("Data processor 3", {'language': 'python'})
]

# Generate in parallel (be mindful of API rate limits)
scripts = await asyncio.gather(*[
    generator.generate_script(desc, req) for desc, req in tasks
])
```

### Model Selection Strategy
The system automatically routes to optimal models:
- **Complex reasoning** → grok-4 
- **Web/GitHub analysis** → Claude-4-Sonnet
- **Visual processing** → O3

## 🤝 Integration Examples

### With CI/CD Pipeline
```yaml
# .github/workflows/generate-scripts.yml
name: Generate Scripts
on: [push]
jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Generate deployment scripts
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          python generate_deployment_scripts.py
```

### With Web Framework
```python
from flask import Flask, request, jsonify
from alita_core.script_generator import ScriptGeneratingTool

app = Flask(__name__)
generator = ScriptGeneratingTool()

@app.route('/generate', methods=['POST'])
async def generate_script_api():
    data = request.json
    
    script = await generator.generate_script(
        task_description=data['description'],
        requirements=data.get('requirements', {}),
        github_urls=data.get('github_urls', [])
    )
    
    return jsonify({
        'id': script.id,
        'name': script.name,
        'code': script.code,
        'setup': script.setup_script,
        'instructions': script.execution_instructions
    })
```

## 📚 Additional Resources

- **Source Code**: `alita-kgot-enhanced/alita_core/script_generator.py`
- **Templates**: `alita-kgot-enhanced/templates/rag_mcp/`
- **Logs**: `alita-kgot-enhanced/logs/alita/script_generation.log`
- **Examples**: See usage examples above
- **API Reference**: Complete method documentation in source

---

**Generated by**: Alita Script Generation Tool Documentation System  
**Last Updated**: 2025-06-28  
**Version**: 1.0.0 