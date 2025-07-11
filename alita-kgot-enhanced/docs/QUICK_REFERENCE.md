# 🚀 Alita Enhanced System - Quick Reference

## 📝 Script Generator & 🔍 Intelligent Search

## Essential Setup
```bash
export OPENROUTER_API_KEY="your_key_here"
python3 -c "from alita_core.script_generator import ScriptGeneratingTool; print('✅ Ready!')"
```

## Basic Usage Pattern
```python
import asyncio
from alita_core.script_generator import ScriptGeneratingTool

async def generate():
    generator = ScriptGeneratingTool()
    script = await generator.generate_script(
        task_description="What you want the script to do",
        requirements={'language': 'python', 'dependencies': ['pandas']}
    )
    return script

script = asyncio.run(generate())
```

## 🔍 Hugging Face Search Quick Start
```javascript
const { HuggingFaceSearchTool } = require('./alita_core/web_agent/index.js');

const searchTool = new HuggingFaceSearchTool(null, {
  model_name: 'webagent',
  kgot_path: './knowledge-graph-of-thoughts'
});

const result = await searchTool.searchWithHuggingFace(JSON.stringify({
  query: "Latest AI developments 2024",
  searchType: "research",  // informational|navigational|research
  includeWikipedia: true,
  detailed: true
}));
```

## Models Used (Per User Rules)
- **🤖 Orchestration**: `x-ai/grok-4`
- **🌐 Web Agent**: `anthropic/claude-4-sonnet`  
- **👁️ Vision**: `openai/o3`
- **🔍 Search**: Hugging Face Agents + KGoT Surfer

## Common Requirements Examples

### Python Data Processing
```python
requirements = {
    'language': 'python',
    'dependencies': ['pandas', 'numpy'],
    'features': ['error_handling', 'logging'],
    'output_format': 'csv'
}
```

### Web Scraping
```python
requirements = {
    'language': 'python',
    'dependencies': ['requests', 'beautifulsoup4'],
    'features': ['rate_limiting', 'retry_logic'],
    'respect_robots_txt': True
}
```

### API Client
```python
requirements = {
    'language': 'python',
    'dependencies': ['requests', 'pydantic'],
    'auth_methods': ['oauth2', 'api_key'],
    'response_caching': True
}
```

### Bash Automation
```python
requirements = {
    'language': 'bash',
    'target_env': 'linux',
    'features': ['error_handling', 'logging'],
    'monitoring': True
}
```

## Generated Script Structure
```python
script = await generator.generate_script(...)

script.name              # Filename
script.code              # Complete script code
script.setup_script      # Environment setup
script.cleanup_script    # Cleanup commands
script.documentation     # Full documentation
script.test_cases        # Generated tests
script.execution_instructions  # How to run
```

## Save Generated Scripts
```python
# Save main script
with open(f"{script.name}.py", 'w') as f:
    f.write(script.code)

# Save setup script
with open(f"setup_{script.name}.sh", 'w') as f:
    f.write(script.setup_script)
```

## Common Options
```python
options = {
    'include_tests': True,        # Generate test cases
    'docker_containerize': True,  # Create Dockerfile
    'monitoring': True,           # Add monitoring
    'documentation': 'sphinx',    # Documentation format
    'async_client': True,         # Async support
    'error_recovery': True        # Error handling
}
```

## Health Check
```python
async def health_check():
    generator = ScriptGeneratingTool()
    stats = generator.get_session_stats()
    print(f"Session: {stats['current_session_id']}")
    print(f"Config OK: {generator.config is not None}")
    print(f"LLM OK: {generator.llm is not None}")

asyncio.run(health_check())
```

## Troubleshooting Quick Fixes

### Missing API Key
```bash
export OPENROUTER_API_KEY="your_key_here"
```

### Import Error
```bash
cd alita-kgot-enhanced
pip install -r kgot_core/requirements.txt
```

### Debug Mode
```python
import logging
logging.getLogger('ScriptGenerator').setLevel(logging.DEBUG)
```

## GitHub Integration
```python
script = await generator.generate_script(
    task_description="Build web scraper",
    requirements={'language': 'python'},
    github_urls=[
        'https://github.com/scrapy/scrapy',
        'https://github.com/requests/requests'
    ]
)
```

## Multi-Language Project
```python
# Generate Python, Bash, and Docker files
languages = ['python', 'bash', 'dockerfile']
scripts = []

for lang in languages:
    script = await generator.generate_script(
        f"Component for {lang}",
        {'language': lang}
    )
    scripts.append(script)
```

## 🔍 Search Types Quick Reference

### Informational Search
```javascript
// For comprehensive research
{
  "query": "What are transformer neural networks?",
  "searchType": "informational",
  "includeWikipedia": true,
  "detailed": true
}
```

### Navigational Search  
```javascript
// To find specific pages/resources
{
  "query": "OpenAI GPT-4 official documentation",
  "searchType": "navigational", 
  "includeWikipedia": false,
  "detailed": false
}
```

### Research Search
```javascript
// For deep analysis and comparison
{
  "query": "Compare RAG vs fine-tuning for AI applications",
  "searchType": "research",
  "includeWikipedia": true,
  "detailed": true
}
```

## 🔧 Web Agent Integration
```javascript
const { AlitaWebAgent } = require('./alita_core/web_agent/index.js');

const agent = new AlitaWebAgent({
  openrouterApiKey: process.env.OPENROUTER_API_KEY
});

// Search is now powered by Hugging Face Agents
const results = await agent.performWebSearch(
  "AI safety research papers", 
  { searchType: "research", detailed: true }
);
```

---
📖 **Full Documentation**: [Script Generator Guide](./script_generator_guide.md)  
🔍 **Search Guide**: [Hugging Face Search Integration](./HUGGINGFACE_SEARCH_INTEGRATION.md)  
🔧 **Source Code**: `alita_core/script_generator.py` | `alita_core/web_agent/index.js`  
📊 **Logs**: `logs/alita/script_generation.log` | `logs/web_agent/combined.log` 