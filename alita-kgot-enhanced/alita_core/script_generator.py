#!/usr/bin/env python3
"""
Alita Script Generation Tool with KGoT Knowledge Support

Implementation of Alita Section 2.3.2 "ScriptGeneratingTool" complete architecture
with integration to KGoT Section 2.3 "Python Code Tool" for enhanced code generation.

Key Features:
- Code-building utility for constructing external tools as specified in Alita paper
- Explicit subtask descriptions and suggestions reception from MCP Brainstorming
- GitHub links integration from Alita Section 2.2 Web Agent capabilities  
- Environment setup and cleanup script generation following Alita Section 2.3.4
- RAG-MCP template-based generation for efficient script creation
- LangChain-based agent orchestration for intelligent script generation

@module ScriptGenerator
@author AI Assistant
@date 2025
"""

import os
import json
import logging
import asyncio
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import requests
import aiohttp
import yaml
from pydantic import BaseModel, Field, validator

# LangChain imports for agent orchestration (per user's hard rule)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Setup Winston-compatible logging with proper path handling
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent
log_dir = project_root / 'logs' / 'alita'

# Ensure log directory exists
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging with relative paths
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(operation)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'script_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ScriptGenerator')

@dataclass
class ScriptGenerationConfig:
    """Configuration for Script Generation Tool"""
    # OpenRouter API configuration (per user rules)
    openrouter_api_key: str = os.getenv('OPENROUTER_API_KEY', '')
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Model configuration per user rules: o3(vision), claude-4-sonnet(webagent), gemini-2.5-pro(orchestration)
    orchestration_model: str = "google/gemini-2.5-pro"  # For main orchestration
    webagent_model: str = "anthropic/claude-4-sonnet"   # For web agent capabilities
    vision_model: str = "openai/o3"                     # For vision tasks
    model_name: str = "google/gemini-2.5-pro"          # Default to orchestration model
    
    # MCP Brainstorming integration
    mcp_brainstorming_endpoint: str = "http://localhost:8001/api/mcp-brainstorming"
    
    # Web Agent integration for GitHub links
    web_agent_endpoint: str = "http://localhost:8000/api/web-agent"
    
    # KGoT Python Tool integration
    kgot_python_tool_endpoint: str = "http://localhost:16000/run"
    
    # Script generation settings
    temp_directory: str = "/tmp/alita_script_generation"
    max_script_size: int = 1048576  # 1MB
    execution_timeout: int = 300    # 5 minutes
    cleanup_on_completion: bool = True
    
    # RAG-MCP template settings
    template_directory: str = "./templates/rag_mcp"
    enable_template_caching: bool = True
    
    # Environment settings
    supported_languages: List[str] = field(default_factory=lambda: ['python', 'bash', 'javascript', 'dockerfile'])
    default_python_version: str = "3.9"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.openrouter_api_key:
            logger.warning("OpenRouter API key not provided, falling back to local models",
                         extra={'operation': 'CONFIG_VALIDATION'})

@dataclass
class SubtaskDescription:
    """Structure for subtask descriptions from MCP Brainstorming"""
    id: str
    title: str
    description: str
    requirements: List[str]
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    estimated_complexity: int = 1
    tools_needed: List[str] = field(default_factory=list)
    
class GitHubLinkInfo(BaseModel):
    """GitHub link information from Web Agent"""
    url: str = Field(..., description="GitHub repository or file URL")
    repository_name: str = Field(..., description="Repository name")
    file_path: Optional[str] = Field(None, description="Specific file path")
    branch: str = Field(default="main", description="Git branch")
    description: Optional[str] = Field(None, description="Repository description")
    language: Optional[str] = Field(None, description="Primary programming language")
    stars: Optional[int] = Field(None, description="Repository stars count")
    
    @validator('url')
    def validate_github_url(cls, v):
        """Validate GitHub URL format"""
        if not v.startswith('https://github.com/'):
            raise ValueError('URL must be a valid GitHub URL')
        return v

class EnvironmentSpec(BaseModel):
    """Environment specification for script execution"""
    language: str = Field(..., description="Programming language")
    version: str = Field(..., description="Language version")
    dependencies: List[str] = Field(default_factory=list, description="Required packages/modules")
    system_requirements: List[str] = Field(default_factory=list, description="System-level requirements")
    environment_variables: Dict[str, str] = Field(default_factory=dict, description="Required environment variables")
    docker_base_image: Optional[str] = Field(None, description="Docker base image if containerized")

class GeneratedScript(BaseModel):
    """Structure for generated scripts"""
    id: str = Field(..., description="Unique script identifier")
    name: str = Field(..., description="Script name")
    description: str = Field(..., description="Script description")
    language: str = Field(..., description="Programming language")
    code: str = Field(..., description="Generated script code")
    environment_spec: EnvironmentSpec = Field(..., description="Environment specification")
    setup_script: str = Field(..., description="Environment setup script")
    cleanup_script: str = Field(..., description="Cleanup script")
    execution_instructions: str = Field(..., description="How to execute the script")
    test_cases: List[str] = Field(default_factory=list, description="Generated test cases")
    documentation: str = Field(..., description="Script documentation")
    github_sources: List[GitHubLinkInfo] = Field(default_factory=list, description="GitHub sources used")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

class MCPBrainstormingBridge:
    """
    Bridge to receive subtask descriptions and suggestions from MCP Brainstorming
    Implements Alita Section 2.3.1 integration with Section 2.3.2
    """
    
    def __init__(self, config: ScriptGenerationConfig):
        """Initialize MCP Brainstorming Bridge"""
        self.config = config
        self.session_id: Optional[str] = None
        
        logger.info("Initializing MCP Brainstorming Bridge", extra={
            'operation': 'MCP_BRIDGE_INIT',
            'endpoint': config.mcp_brainstorming_endpoint
        })
    
    async def initialize_session(self, task_context: str) -> str:
        """
        Initialize session with MCP Brainstorming component
        
        Args:
            task_context: Context for the task requiring script generation
            
        Returns:
            Session ID for the brainstorming session
        """
        try:
            logger.info("Initializing MCP Brainstorming session", extra={
                'operation': 'MCP_SESSION_INIT',
                'task_context': task_context[:100] + '...' if len(task_context) > 100 else task_context
            })
            
            session_data = {
                'task_context': task_context,
                'workflow_type': 'script_generation',
                'capabilities_needed': ['code_generation', 'environment_setup', 'tool_integration'],
                'timestamp': datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.mcp_brainstorming_endpoint}/session/create",
                    json=session_data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.session_id = result.get('session_id')
                        
                        logger.info("MCP Brainstorming session initialized", extra={
                            'operation': 'MCP_SESSION_SUCCESS',
                            'session_id': self.session_id
                        })
                        
                        return self.session_id
                    else:
                        error_text = await response.text()
                        raise Exception(f"Session initialization failed: {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to initialize MCP Brainstorming session: {str(e)}", extra={
                'operation': 'MCP_SESSION_FAILED',
                'error': str(e)
            })
            # Fallback to local session
            self.session_id = f"local_mcp_session_{datetime.now().timestamp()}"
            return self.session_id
    
    async def receive_subtask_descriptions(self, script_requirements: Dict[str, Any]) -> List[SubtaskDescription]:
        """
        Receive subtask descriptions and suggestions from MCP Brainstorming
        
        Args:
            script_requirements: Requirements for script generation
            
        Returns:
            List of subtask descriptions for script generation
        """
        if not self.session_id:
            await self.initialize_session(str(script_requirements))
        
        try:
            logger.info("Requesting subtask descriptions from MCP Brainstorming", extra={
                'operation': 'MCP_SUBTASK_REQUEST',
                'session_id': self.session_id
            })
            
            request_data = {
                'session_id': self.session_id,
                'script_requirements': script_requirements,
                'analysis_type': 'subtask_decomposition',
                'generation_context': 'script_generation_tool'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.mcp_brainstorming_endpoint}/subtasks/analyze",
                    json=request_data,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        subtasks_data = result.get('subtasks', [])
                        
                        # Convert to SubtaskDescription objects
                        subtasks = []
                        for subtask_data in subtasks_data:
                            subtask = SubtaskDescription(
                                id=subtask_data.get('id', f"subtask_{len(subtasks)}"),
                                title=subtask_data.get('title', ''),
                                description=subtask_data.get('description', ''),
                                requirements=subtask_data.get('requirements', []),
                                dependencies=subtask_data.get('dependencies', []),
                                priority=subtask_data.get('priority', 1),
                                estimated_complexity=subtask_data.get('complexity', 1),
                                tools_needed=subtask_data.get('tools_needed', [])
                            )
                            subtasks.append(subtask)
                        
                        logger.info(f"Received {len(subtasks)} subtask descriptions", extra={
                            'operation': 'MCP_SUBTASK_SUCCESS',
                            'subtasks_count': len(subtasks)
                        })
                        
                        return subtasks
                    else:
                        error_text = await response.text()
                        raise Exception(f"Subtask request failed: {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to receive subtask descriptions: {str(e)}", extra={
                'operation': 'MCP_SUBTASK_FAILED',
                'error': str(e)
            })
            
            # Fallback: Create basic subtasks from requirements
            return self._create_fallback_subtasks(script_requirements)
    
    def _create_fallback_subtasks(self, requirements: Dict[str, Any]) -> List[SubtaskDescription]:
        """Create fallback subtasks when MCP Brainstorming is unavailable"""
        logger.info("Creating fallback subtasks", extra={'operation': 'MCP_FALLBACK_SUBTASKS'})
        
        fallback_subtasks = [
            SubtaskDescription(
                id="subtask_environment_setup",
                title="Environment Setup",
                description="Set up the required environment and dependencies",
                requirements=["python", "pip", "virtual environment"],
                priority=1,
                estimated_complexity=2
            ),
            SubtaskDescription(
                id="subtask_core_implementation", 
                title="Core Implementation",
                description="Implement the main functionality",
                requirements=requirements.get('core_requirements', []),
                priority=2,
                estimated_complexity=3
            ),
            SubtaskDescription(
                id="subtask_testing",
                title="Testing and Validation",
                description="Create tests and validate the implementation",
                requirements=["pytest", "test data"],
                priority=3,
                estimated_complexity=2
            ),
            SubtaskDescription(
                id="subtask_cleanup",
                title="Cleanup and Finalization",
                description="Clean up resources and finalize the script",
                requirements=["cleanup procedures"],
                priority=4,
                estimated_complexity=1
            )
        ]
        
        return fallback_subtasks 

class GitHubLinksProcessor:
    """
    Processes GitHub links from Web Agent capabilities (Alita Section 2.2)
    Extracts relevant code snippets and documentation from GitHub repositories
    """
    
    def __init__(self, config: ScriptGenerationConfig):
        """Initialize GitHub Links Processor"""
        self.config = config
        self.web_agent_session: Optional[str] = None
        
        logger.info("Initializing GitHub Links Processor", extra={
            'operation': 'GITHUB_PROCESSOR_INIT'
        })
    
    async def process_github_links(self, github_urls: List[str], context: str = "") -> List[GitHubLinkInfo]:
        """
        Process GitHub links to extract relevant information
        
        Args:
            github_urls: List of GitHub URLs to process
            context: Context for processing (what we're looking for)
            
        Returns:
            List of processed GitHub link information
        """
        try:
            logger.info(f"Processing {len(github_urls)} GitHub links", extra={
                'operation': 'GITHUB_PROCESS_LINKS',
                'urls_count': len(github_urls),
                'context': context[:50] + '...' if len(context) > 50 else context
            })
            
            processed_links = []
            
            for url in github_urls:
                try:
                    # Extract repository information
                    repo_info = self._extract_repository_info(url)
                    
                    # Get additional details from Web Agent
                    enhanced_info = await self._enhance_with_web_agent(repo_info, context)
                    
                    processed_links.append(enhanced_info)
                    
                except Exception as e:
                    logger.warning(f"Failed to process GitHub URL {url}: {str(e)}", extra={
                        'operation': 'GITHUB_PROCESS_URL_FAILED',
                        'url': url,
                        'error': str(e)
                    })
                    continue
            
            logger.info(f"Successfully processed {len(processed_links)} GitHub links", extra={
                'operation': 'GITHUB_PROCESS_SUCCESS',
                'processed_count': len(processed_links)
            })
            
            return processed_links
            
        except Exception as e:
            logger.error(f"GitHub links processing failed: {str(e)}", extra={
                'operation': 'GITHUB_PROCESS_FAILED',
                'error': str(e)
            })
            return []
    
    def _extract_repository_info(self, url: str) -> GitHubLinkInfo:
        """Extract basic repository information from GitHub URL"""
        # Parse GitHub URL: https://github.com/owner/repo[/path/to/file]
        parts = url.replace('https://github.com/', '').split('/')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub URL format: {url}")
        
        owner, repo = parts[0], parts[1]
        file_path = '/'.join(parts[2:]) if len(parts) > 2 else None
        
        return GitHubLinkInfo(
            url=url,
            repository_name=f"{owner}/{repo}",
            file_path=file_path,
            branch="main"  # Default, will be updated by Web Agent
        )
    
    async def _enhance_with_web_agent(self, repo_info: GitHubLinkInfo, context: str) -> GitHubLinkInfo:
        """Enhance repository information using Web Agent capabilities"""
        try:
            request_data = {
                'url': repo_info.url,
                'analysis_type': 'repository_analysis',
                'context': context,
                'extract_info': ['description', 'language', 'stars', 'readme', 'code_samples']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.web_agent_endpoint}/analyze/github",
                    json=request_data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update repository information with Web Agent data
                        repo_info.description = result.get('description')
                        repo_info.language = result.get('language')
                        repo_info.stars = result.get('stars')
                        
                        logger.info(f"Enhanced GitHub info for {repo_info.repository_name}", extra={
                            'operation': 'GITHUB_ENHANCE_SUCCESS',
                            'repository': repo_info.repository_name
                        })
                    else:
                        logger.warning(f"Web Agent enhancement failed for {repo_info.url}", extra={
                            'operation': 'GITHUB_ENHANCE_FAILED',
                            'status': response.status
                        })
        
        except Exception as e:
            logger.warning(f"Failed to enhance GitHub info: {str(e)}", extra={
                'operation': 'GITHUB_ENHANCE_ERROR',
                'error': str(e)
            })
        
        return repo_info
    
    async def extract_code_snippets(self, github_links: List[GitHubLinkInfo], requirements: List[str]) -> Dict[str, str]:
        """
        Extract relevant code snippets from GitHub repositories
        
        Args:
            github_links: Processed GitHub link information
            requirements: Specific requirements for code extraction
            
        Returns:
            Dictionary mapping requirement to relevant code snippet
        """
        code_snippets = {}
        
        for link in github_links:
            try:
                logger.info(f"Extracting code snippets from {link.repository_name}", extra={
                    'operation': 'GITHUB_EXTRACT_CODE',
                    'repository': link.repository_name
                })
                
                # Request specific code extraction from Web Agent
                request_data = {
                    'github_url': link.url,
                    'file_path': link.file_path,
                    'requirements': requirements,
                    'extraction_type': 'code_snippets_targeted'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.config.web_agent_endpoint}/extract/code",
                        json=request_data,
                        timeout=45
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            extracted_snippets = result.get('code_snippets', {})
                            
                            # Merge extracted snippets
                            for requirement, snippet in extracted_snippets.items():
                                if requirement in code_snippets:
                                    code_snippets[requirement] += f"\n\n# From {link.repository_name}:\n{snippet}"
                                else:
                                    code_snippets[requirement] = f"# From {link.repository_name}:\n{snippet}"
                        
            except Exception as e:
                logger.warning(f"Failed to extract code from {link.repository_name}: {str(e)}", extra={
                    'operation': 'GITHUB_EXTRACT_FAILED',
                    'repository': link.repository_name,
                    'error': str(e)
                })
        
        logger.info(f"Extracted code snippets for {len(code_snippets)} requirements", extra={
            'operation': 'GITHUB_EXTRACT_SUCCESS',
            'snippets_count': len(code_snippets)
        })
        
        return code_snippets

class RAGMCPTemplateEngine:
    """
    RAG-MCP template-based generation engine for efficient script creation
    Implements template-based generation using RAG-MCP patterns
    """
    
    def __init__(self, config: ScriptGenerationConfig):
        """Initialize RAG-MCP Template Engine"""
        self.config = config
        self.template_cache: Dict[str, str] = {}
        self.template_directory = Path(config.template_directory)
        
        logger.info("Initializing RAG-MCP Template Engine", extra={
            'operation': 'TEMPLATE_ENGINE_INIT',
            'template_dir': str(self.template_directory)
        })
        
        # Initialize template directory if it doesn't exist
        self.template_directory.mkdir(parents=True, exist_ok=True)
        
        # Load default templates
        if config.enable_template_caching:
            asyncio.create_task(self._load_default_templates())
    
    async def _load_default_templates(self):
        """Load default script generation templates"""
        try:
            default_templates = {
                'python_script': self._get_python_script_template(),
                'bash_script': self._get_bash_script_template(),
                'javascript_script': self._get_javascript_script_template(),
                'dockerfile': self._get_dockerfile_template(),
                'environment_setup': self._get_environment_setup_template(),
                'cleanup_script': self._get_cleanup_script_template()
            }
            
            for template_name, template_content in default_templates.items():
                template_file = self.template_directory / f"{template_name}.template"
                if not template_file.exists():
                    template_file.write_text(template_content)
                
                if self.config.enable_template_caching:
                    self.template_cache[template_name] = template_content
            
            logger.info(f"Loaded {len(default_templates)} default templates", extra={
                'operation': 'TEMPLATE_LOAD_SUCCESS',
                'templates_count': len(default_templates)
            })
            
        except Exception as e:
            logger.error(f"Failed to load default templates: {str(e)}", extra={
                'operation': 'TEMPLATE_LOAD_FAILED',
                'error': str(e)
            })
    
    def _get_python_script_template(self) -> str:
        """Get Python script generation template"""
        return '''#!/usr/bin/env python3
"""
{script_title}

{script_description}

Generated by Alita Script Generation Tool
KGoT Integration: {kgot_integration}
Generated at: {timestamp}
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
{additional_imports}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

{environment_validation}

{main_functions}

def main():
    """Main execution function"""
    try:
        logger.info("Starting {script_name}")
        {main_logic}
        logger.info("Completed {script_name} successfully")
    except Exception as e:
        logger.error(f"Error in {script_name}: {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _get_bash_script_template(self) -> str:
        """Get Bash script generation template"""
        return '''#!/bin/bash
# {script_title}
# {script_description}
# Generated by Alita Script Generation Tool
# Generated at: {timestamp}

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
LOG_FILE="${{SCRIPT_DIR}}/{script_name}.log"

# Logging function
log() {{
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}}

# Error handling
error_exit() {{
    log "ERROR: $1"
    exit 1
}}

{environment_validation}

{main_functions}

# Main execution
main() {{
    log "Starting {script_name}"
    {main_logic}
    log "Completed {script_name} successfully"
}}

# Execute main function
main "$@"
'''
    
    def _get_javascript_script_template(self) -> str:
        """Get JavaScript script generation template"""
        return '''#!/usr/bin/env node
/**
 * {script_title}
 * 
 * {script_description}
 * 
 * Generated by Alita Script Generation Tool
 * Generated at: {timestamp}
 */

const fs = require('fs');
const path = require('path');
{additional_imports}

// Setup logging
const winston = require('winston');
const logger = winston.createLogger({{
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.printf(({{ timestamp, level, message }}) => {{
            return `${{timestamp}} - ${{level.toUpperCase()}}: ${{message}}`;
        }})
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({{ filename: '{script_name}.log' }})
    ]
}});

{environment_validation}

{main_functions}

// Main execution function
async function main() {{
    try {{
        logger.info('Starting {script_name}');
        {main_logic}
        logger.info('Completed {script_name} successfully');
    }} catch (error) {{
        logger.error(`Error in {script_name}: ${{error.message}}`);
        process.exit(1);
    }}
}}

// Execute if run directly
if (require.main === module) {{
    main();
}}

module.exports = {{ main }};
'''
    
    def _get_dockerfile_template(self) -> str:
        """Get Dockerfile generation template"""
        return '''# {script_title} Dockerfile
# {script_description}
# Generated by Alita Script Generation Tool
# Generated at: {timestamp}

FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
{system_dependencies}

# Copy requirements and install dependencies
{dependency_installation}

# Copy application code
COPY . .

# Set environment variables
{environment_variables}

# Expose ports if needed
{exposed_ports}

# Create non-root user for security
{user_setup}

# Health check
{health_check}

# Default command
{default_command}
'''
    
    def _get_environment_setup_template(self) -> str:
        """Get environment setup script template"""
        return '''#!/bin/bash
# Environment Setup Script for {script_name}
# Generated by Alita Script Generation Tool
# Generated at: {timestamp}

set -euo pipefail

log() {{
    echo "$(date '+%Y-%m-%d %H:%M:%S') - SETUP: $1"
}}

log "Setting up environment for {script_name}"

# Create virtual environment if needed
{virtual_env_setup}

# Install system dependencies
{system_deps_install}

# Install language-specific dependencies
{language_deps_install}

# Set up environment variables
{env_vars_setup}

# Create necessary directories
{directories_setup}

# Download required resources
{resources_download}

# Validate environment
{environment_validation}

log "Environment setup completed successfully"
'''
    
    def _get_cleanup_script_template(self) -> str:
        """Get cleanup script template"""
        return '''#!/bin/bash
# Cleanup Script for {script_name}
# Generated by Alita Script Generation Tool
# Generated at: {timestamp}

set -euo pipefail

log() {{
    echo "$(date '+%Y-%m-%d %H:%M:%S') - CLEANUP: $1"
}}

log "Starting cleanup for {script_name}"

# Remove temporary files
{temp_files_cleanup}

# Clean up virtual environments
{venv_cleanup}

# Remove downloaded resources
{resources_cleanup}

# Clean up processes
{processes_cleanup}

# Remove log files if specified
{logs_cleanup}

log "Cleanup completed successfully"
'''
    
    async def generate_from_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Generate script content from template
        
        Args:
            template_name: Name of the template to use
            context: Context variables for template substitution
            
        Returns:
            Generated script content
        """
        try:
            logger.info(f"Generating from template: {template_name}", extra={
                'operation': 'TEMPLATE_GENERATE',
                'template': template_name
            })
            
            # Get template content
            template_content = await self._get_template_content(template_name)
            
            # Add default context variables
            default_context = {
                'timestamp': datetime.now().isoformat(),
                'generator': 'Alita Script Generation Tool',
                'kgot_integration': 'Enabled'
            }
            
            # Merge contexts
            full_context = {**default_context, **context}
            
            # Substitute template variables
            generated_content = template_content.format(**full_context)
            
            logger.info(f"Successfully generated content from template {template_name}", extra={
                'operation': 'TEMPLATE_GENERATE_SUCCESS',
                'template': template_name,
                'content_length': len(generated_content)
            })
            
            return generated_content
            
        except Exception as e:
            logger.error(f"Template generation failed for {template_name}: {str(e)}", extra={
                'operation': 'TEMPLATE_GENERATE_FAILED',
                'template': template_name,
                'error': str(e)
            })
            raise
    
    async def _get_template_content(self, template_name: str) -> str:
        """Get template content from cache or file"""
        # Check cache first
        if self.config.enable_template_caching and template_name in self.template_cache:
            return self.template_cache[template_name]
        
        # Load from file
        template_file = self.template_directory / f"{template_name}.template"
        if template_file.exists():
            content = template_file.read_text()
            
            # Cache if enabled
            if self.config.enable_template_caching:
                self.template_cache[template_name] = content
            
            return content
        
        raise FileNotFoundError(f"Template {template_name} not found")

class EnvironmentSetupGenerator:
    """
    Environment setup and cleanup script generation following Alita Section 2.3.4
    Handles dependency management, environment validation, and resource cleanup
    """
    
    def __init__(self, config: ScriptGenerationConfig):
        """Initialize Environment Setup Generator"""
        self.config = config
        self.supported_languages = config.supported_languages
        
        logger.info("Initializing Environment Setup Generator", extra={
            'operation': 'ENV_GENERATOR_INIT',
            'supported_languages': self.supported_languages
        })
    
    def generate_environment_spec(self, requirements: Dict[str, Any], github_sources: List[GitHubLinkInfo]) -> EnvironmentSpec:
        """
        Generate environment specification based on requirements and GitHub sources
        
        Args:
            requirements: Script requirements and constraints
            github_sources: GitHub sources for additional context
            
        Returns:
            Complete environment specification
        """
        try:
            logger.info("Generating environment specification", extra={
                'operation': 'ENV_SPEC_GENERATE',
                'requirements_keys': list(requirements.keys()),
                'github_sources_count': len(github_sources)
            })
            
            # Determine primary language
            language = self._determine_language(requirements, github_sources)
            
            # Get language version
            version = self._determine_version(language, requirements)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(requirements, github_sources, language)
            
            # Determine system requirements
            system_requirements = self._extract_system_requirements(requirements, dependencies)
            
            # Extract environment variables
            env_vars = self._extract_environment_variables(requirements)
            
            # Determine Docker base image if containerization is needed
            docker_image = self._determine_docker_image(language, version, requirements)
            
            env_spec = EnvironmentSpec(
                language=language,
                version=version,
                dependencies=dependencies,
                system_requirements=system_requirements,
                environment_variables=env_vars,
                docker_base_image=docker_image
            )
            
            logger.info("Environment specification generated successfully", extra={
                'operation': 'ENV_SPEC_SUCCESS',
                'language': language,
                'dependencies_count': len(dependencies)
            })
            
            return env_spec
            
        except Exception as e:
            logger.error(f"Environment specification generation failed: {str(e)}", extra={
                'operation': 'ENV_SPEC_FAILED',
                'error': str(e)
            })
            raise 
    
    def _determine_language(self, requirements: Dict[str, Any], github_sources: List[GitHubLinkInfo]) -> str:
        """Determine the primary programming language for the script"""
        # Check explicit language requirement
        if 'language' in requirements:
            lang = requirements['language'].lower()
            if lang in self.supported_languages:
                return lang
        
        # Check GitHub sources for language hints
        github_languages = [source.language for source in github_sources if source.language]
        if github_languages:
            # Find most common language
            lang_counts = {}
            for lang in github_languages:
                lang_lower = lang.lower()
                if lang_lower in self.supported_languages:
                    lang_counts[lang_lower] = lang_counts.get(lang_lower, 0) + 1
            
            if lang_counts:
                return max(lang_counts, key=lang_counts.get)
        
        # Check requirements for language-specific keywords
        req_text = str(requirements).lower()
        for lang in self.supported_languages:
            if lang in req_text:
                return lang
        
        # Default to Python
        return 'python'
    
    def _determine_version(self, language: str, requirements: Dict[str, Any]) -> str:
        """Determine the version for the specified language"""
        version_key = f"{language}_version"
        if version_key in requirements:
            return requirements[version_key]
        
        # Language-specific defaults
        version_defaults = {
            'python': self.config.default_python_version,
            'javascript': '18',
            'bash': '5.0',
            'dockerfile': 'latest'
        }
        
        return version_defaults.get(language, 'latest')
    
    def _extract_dependencies(self, requirements: Dict[str, Any], github_sources: List[GitHubLinkInfo], language: str) -> List[str]:
        """Extract language-specific dependencies"""
        dependencies = []
        
        # Direct dependencies from requirements
        if 'dependencies' in requirements:
            dependencies.extend(requirements['dependencies'])
        
        # Language-specific dependency keys
        dep_keys = [f"{language}_dependencies", f"{language}_packages", f"{language}_modules"]
        for key in dep_keys:
            if key in requirements:
                dependencies.extend(requirements[key])
        
        # Extract from GitHub sources (would need actual repository analysis)
        for source in github_sources:
            if source.language and source.language.lower() == language:
                # In a real implementation, this would analyze package files
                # like requirements.txt, package.json, etc.
                pass
        
        # Add common dependencies based on language
        common_deps = self._get_common_dependencies(language)
        dependencies.extend(common_deps)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_deps = []
        for dep in dependencies:
            if dep not in seen:
                seen.add(dep)
                unique_deps.append(dep)
        
        return unique_deps
    
    def _get_common_dependencies(self, language: str) -> List[str]:
        """Get common dependencies for each language"""
        common_deps = {
            'python': ['requests', 'pyyaml', 'python-dotenv'],
            'javascript': ['axios', 'dotenv', 'winston'],
            'bash': [],
            'dockerfile': []
        }
        
        return common_deps.get(language, [])
    
    def _extract_system_requirements(self, requirements: Dict[str, Any], dependencies: List[str]) -> List[str]:
        """Extract system-level requirements"""
        system_reqs = []
        
        # Direct system requirements
        if 'system_requirements' in requirements:
            system_reqs.extend(requirements['system_requirements'])
        
        # Infer system requirements from dependencies
        for dep in dependencies:
            if 'opencv' in dep.lower():
                system_reqs.extend(['libopencv-dev', 'python3-opencv'])
            elif 'pillow' in dep.lower() or 'pil' in dep.lower():
                system_reqs.extend(['libjpeg-dev', 'libpng-dev'])
            elif 'psycopg2' in dep.lower():
                system_reqs.extend(['libpq-dev'])
            elif 'mysql' in dep.lower():
                system_reqs.extend(['libmysqlclient-dev'])
        
        # Remove duplicates
        return list(set(system_reqs))
    
    def _extract_environment_variables(self, requirements: Dict[str, Any]) -> Dict[str, str]:
        """Extract required environment variables"""
        env_vars = {}
        
        if 'environment_variables' in requirements:
            env_vars.update(requirements['environment_variables'])
        
        if 'env_vars' in requirements:
            env_vars.update(requirements['env_vars'])
        
        # Add default environment variables
        env_vars.update({
            'PYTHONPATH': '.',
            'PYTHONUNBUFFERED': '1'
        })
        
        return env_vars
    
    def _determine_docker_image(self, language: str, version: str, requirements: Dict[str, Any]) -> Optional[str]:
        """Determine appropriate Docker base image"""
        if not requirements.get('containerized', False):
            return None
        
        # Custom image specified
        if 'docker_image' in requirements:
            return requirements['docker_image']
        
        # Language-specific defaults
        image_map = {
            'python': f"python:{version}-slim",
            'javascript': f"node:{version}-alpine",
            'bash': "ubuntu:22.04",
            'dockerfile': None
        }
        
        return image_map.get(language)
    
    async def generate_setup_script(self, env_spec: EnvironmentSpec, script_name: str) -> str:
        """Generate environment setup script"""
        try:
            logger.info(f"Generating setup script for {script_name}", extra={
                'operation': 'SETUP_SCRIPT_GENERATE',
                'language': env_spec.language,
                'script_name': script_name
            })
            
            setup_components = {
                'virtual_env_setup': self._generate_virtual_env_setup(env_spec),
                'system_deps_install': self._generate_system_deps_install(env_spec),
                'language_deps_install': self._generate_language_deps_install(env_spec),
                'env_vars_setup': self._generate_env_vars_setup(env_spec),
                'directories_setup': self._generate_directories_setup(script_name),
                'resources_download': self._generate_resources_download(env_spec),
                'environment_validation': self._generate_env_validation(env_spec)
            }
            
            # Use template engine to generate setup script
            template_engine = RAGMCPTemplateEngine(self.config)
            context = {
                'script_name': script_name,
                **setup_components
            }
            
            setup_script = await template_engine.generate_from_template('environment_setup', context)
            
            logger.info(f"Setup script generated for {script_name}", extra={
                'operation': 'SETUP_SCRIPT_SUCCESS',
                'script_length': len(setup_script)
            })
            
            return setup_script
            
        except Exception as e:
            logger.error(f"Setup script generation failed: {str(e)}", extra={
                'operation': 'SETUP_SCRIPT_FAILED',
                'error': str(e)
            })
            raise
    
    async def generate_cleanup_script(self, env_spec: EnvironmentSpec, script_name: str) -> str:
        """Generate cleanup script"""
        try:
            logger.info(f"Generating cleanup script for {script_name}", extra={
                'operation': 'CLEANUP_SCRIPT_GENERATE',
                'script_name': script_name
            })
            
            cleanup_components = {
                'temp_files_cleanup': self._generate_temp_cleanup(script_name),
                'venv_cleanup': self._generate_venv_cleanup(env_spec),
                'resources_cleanup': self._generate_resources_cleanup(script_name),
                'processes_cleanup': self._generate_processes_cleanup(script_name),
                'logs_cleanup': self._generate_logs_cleanup(script_name)
            }
            
            # Use template engine to generate cleanup script
            template_engine = RAGMCPTemplateEngine(self.config)
            context = {
                'script_name': script_name,
                **cleanup_components
            }
            
            cleanup_script = await template_engine.generate_from_template('cleanup_script', context)
            
            logger.info(f"Cleanup script generated for {script_name}", extra={
                'operation': 'CLEANUP_SCRIPT_SUCCESS',
                'script_length': len(cleanup_script)
            })
            
            return cleanup_script
            
        except Exception as e:
            logger.error(f"Cleanup script generation failed: {str(e)}", extra={
                'operation': 'CLEANUP_SCRIPT_FAILED',
                'error': str(e)
            })
            raise
    
    def _generate_virtual_env_setup(self, env_spec: EnvironmentSpec) -> str:
        """Generate virtual environment setup commands"""
        if env_spec.language != 'python':
            return "# No virtual environment needed for this language"
        
        return f'''
if [ ! -d "venv" ]; then
    log "Creating Python virtual environment"
    python{env_spec.version} -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
else
    log "Virtual environment already exists, activating"
    source venv/bin/activate
fi
'''
    
    def _generate_system_deps_install(self, env_spec: EnvironmentSpec) -> str:
        """Generate system dependencies installation commands"""
        if not env_spec.system_requirements:
            return "# No system dependencies required"
        
        deps_list = ' '.join(env_spec.system_requirements)
        return f'''
log "Installing system dependencies"
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y {deps_list}
elif command -v yum &> /dev/null; then
    sudo yum install -y {deps_list}
elif command -v brew &> /dev/null; then
    brew install {deps_list}
else
    log "WARNING: Could not detect package manager for system dependencies"
fi
'''
    
    def _generate_language_deps_install(self, env_spec: EnvironmentSpec) -> str:
        """Generate language-specific dependencies installation"""
        if not env_spec.dependencies:
            return "# No language dependencies required"
        
        if env_spec.language == 'python':
            deps_file = '\n'.join(env_spec.dependencies)
            return f'''
log "Installing Python dependencies"
cat > requirements.txt << EOF
{deps_file}
EOF
pip install -r requirements.txt
'''
        elif env_spec.language == 'javascript':
            deps_list = ' '.join(env_spec.dependencies)
            return f'''
log "Installing Node.js dependencies"
npm install {deps_list}
'''
        else:
            return f"# Dependencies for {env_spec.language}: {', '.join(env_spec.dependencies)}"
    
    def _generate_env_vars_setup(self, env_spec: EnvironmentSpec) -> str:
        """Generate environment variables setup"""
        if not env_spec.environment_variables:
            return "# No environment variables to set"
        
        env_lines = []
        for key, value in env_spec.environment_variables.items():
            env_lines.append(f'export {key}="{value}"')
        
        return f'''
log "Setting up environment variables"
{chr(10).join(env_lines)}
'''
    
    def _generate_directories_setup(self, script_name: str) -> str:
        """Generate necessary directories setup"""
        return f'''
log "Creating necessary directories"
mkdir -p logs
mkdir -p temp
mkdir -p output
mkdir -p data
'''
    
    def _generate_resources_download(self, env_spec: EnvironmentSpec) -> str:
        """Generate resources download commands"""
        return '''
log "Downloading required resources (if any)"
# Add specific resource downloads here if needed
'''
    
    def _generate_env_validation(self, env_spec: EnvironmentSpec) -> str:
        """Generate environment validation commands"""
        if env_spec.language == 'python':
            return f'''
log "Validating Python environment"
python{env_spec.version} --version
pip --version
python{env_spec.version} -c "import sys; print(f'Python {{sys.version}}')"
'''
        elif env_spec.language == 'javascript':
            return f'''
log "Validating Node.js environment"
node --version
npm --version
'''
        else:
            return f'''
log "Validating {env_spec.language} environment"
which {env_spec.language} || echo "WARNING: {env_spec.language} not found in PATH"
'''
    
    def _generate_temp_cleanup(self, script_name: str) -> str:
        """Generate temporary files cleanup"""
        return '''
if [ -d "temp" ]; then
    log "Removing temporary files"
    rm -rf temp/*
fi

if [ -d "/tmp/alita_script_generation" ]; then
    log "Removing script generation temp files"
    rm -rf /tmp/alita_script_generation/*
fi
'''
    
    def _generate_venv_cleanup(self, env_spec: EnvironmentSpec) -> str:
        """Generate virtual environment cleanup"""
        if env_spec.language == 'python':
            return '''
if [ -d "venv" ] && [ "$KEEP_VENV" != "true" ]; then
    log "Removing Python virtual environment"
    rm -rf venv
fi
'''
        else:
            return "# No virtual environment to clean up"
    
    def _generate_resources_cleanup(self, script_name: str) -> str:
        """Generate downloaded resources cleanup"""
        return '''
log "Cleaning up downloaded resources"
# Remove any downloaded files or data that are no longer needed
'''
    
    def _generate_processes_cleanup(self, script_name: str) -> str:
        """Generate processes cleanup"""
        return f'''
log "Cleaning up any remaining processes"
# Kill any background processes started by {script_name}
pkill -f "{script_name}" || true
'''
    
    def _generate_logs_cleanup(self, script_name: str) -> str:
        """Generate log files cleanup"""
        return f'''
if [ "$KEEP_LOGS" != "true" ]; then
    log "Cleaning up log files"
    rm -f logs/{script_name}*.log
fi
'''

class KGoTPythonToolBridge:
    """
    Bridge to integrate with KGoT Section 2.3 Python Code Tool for enhanced code generation
    Provides seamless integration with KGoT's Python execution capabilities
    """
    
    def __init__(self, config: ScriptGenerationConfig):
        """Initialize KGoT Python Tool Bridge"""
        self.config = config
        self.tool_endpoint = config.kgot_python_tool_endpoint
        
        logger.info("Initializing KGoT Python Tool Bridge", extra={
            'operation': 'KGOT_BRIDGE_INIT',
            'endpoint': self.tool_endpoint
        })
    
    async def enhance_code_generation(self, code_requirements: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance code generation using KGoT Python Tool capabilities
        
        Args:
            code_requirements: Description of code to generate
            context: Additional context for code generation
            
        Returns:
            Enhanced code generation results
        """
        try:
            logger.info("Enhancing code generation with KGoT", extra={
                'operation': 'KGOT_ENHANCE_CODE',
                'requirements_length': len(code_requirements)
            })
            
            # Prepare code generation request
            generation_code = f'''
# Code generation request: {code_requirements}
# Context: {context}

# Generate the requested code
import inspect
import ast

def generate_code():
    """Generate code based on requirements"""
    requirements = """{code_requirements}"""
    context = {context}
    
    # Analysis and generation logic here
    # This would be enhanced with actual KGoT capabilities
    
    generated_code = """
# Generated code for: {code_requirements}

def main():
    # Implementation based on requirements
    pass

if __name__ == "__main__":
    main()
"""
    
    return {{
        "generated_code": generated_code,
        "requirements": requirements,
        "context": context,
        "success": True
    }}

# Execute generation
result = generate_code()
print(f"Generated code result: {{result}}")
'''
            
            # Execute code generation using KGoT Python Tool
            execution_result = await self._execute_with_kgot_tool(
                generation_code,
                ['ast', 'inspect']
            )
            
            if execution_result.get('error'):
                logger.warning(f"KGoT code generation had issues: {execution_result['error']}", extra={
                    'operation': 'KGOT_ENHANCE_WARNING'
                })
                # Fallback to basic generation
                return self._fallback_code_generation(code_requirements, context)
            
            # Parse and return enhanced results
            enhanced_result = self._parse_kgot_result(execution_result)
            
            logger.info("Code generation enhanced successfully with KGoT", extra={
                'operation': 'KGOT_ENHANCE_SUCCESS',
                'result_keys': list(enhanced_result.keys())
            })
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"KGoT code enhancement failed: {str(e)}", extra={
                'operation': 'KGOT_ENHANCE_FAILED',
                'error': str(e)
            })
            return self._fallback_code_generation(code_requirements, context)
    
    async def _execute_with_kgot_tool(self, code: str, required_modules: List[str]) -> Dict[str, Any]:
        """Execute code using KGoT Python Tool"""
        try:
            request_data = {
                'code': code,
                'required_modules': required_modules
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.tool_endpoint,
                    json=request_data,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        return {'error': f"KGoT tool execution failed: {error_text}"}
                        
        except Exception as e:
            return {'error': f"KGoT tool communication failed: {str(e)}"}
    
    def _parse_kgot_result(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse KGoT execution result"""
        # Extract useful information from KGoT tool result
        output = execution_result.get('output', '')
        
        # In a real implementation, this would parse the actual output
        # and extract generated code, analysis, etc.
        
        return {
            'enhanced_code': output,
            'analysis': 'KGoT-enhanced analysis',
            'suggestions': ['Use proper error handling', 'Add logging', 'Include documentation'],
            'kgot_integration': True
        }
    
    def _fallback_code_generation(self, requirements: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback code generation when KGoT is unavailable"""
        logger.info("Using fallback code generation", extra={
            'operation': 'KGOT_FALLBACK'
        })
        
        return {
            'enhanced_code': f'# Fallback generated code for: {requirements}\n\ndef main():\n    pass\n',
            'analysis': 'Basic analysis without KGoT enhancement',
            'suggestions': ['Add error handling', 'Include logging'],
            'kgot_integration': False
        } 

class ScriptGeneratingTool:
    """
    Main Script Generation Tool implementing Alita Section 2.3.2 "ScriptGeneratingTool" complete architecture
    
    This tool orchestrates the complete script generation workflow:
    1. Receives explicit subtask descriptions from MCP Brainstorming
    2. Processes GitHub links from Web Agent capabilities
    3. Uses KGoT Python Code Tool for enhanced code generation
    4. Generates environment setup and cleanup scripts
    5. Integrates RAG-MCP template-based generation
    
    Features:
    - Code-building utility for constructing external tools
    - Intelligent script generation with LangChain orchestration
    - Template-based generation with RAG-MCP integration
    - Environment validation and cleanup automation
    - GitHub repository analysis and code extraction
    """
    
    def __init__(self, config: Optional[ScriptGenerationConfig] = None):
        """
        Initialize the Script Generation Tool
        
        Args:
            config: Configuration for script generation (optional)
        """
        self.config = config or ScriptGenerationConfig()
        
        # Initialize component bridges
        self.mcp_bridge = MCPBrainstormingBridge(self.config)
        self.github_processor = GitHubLinksProcessor(self.config)
        self.template_engine = RAGMCPTemplateEngine(self.config)
        self.env_generator = EnvironmentSetupGenerator(self.config)
        self.kgot_bridge = KGoTPythonToolBridge(self.config)
        
        # Initialize LangChain agent for intelligent orchestration (per user's hard rule)
        self.llm = self._initialize_llm()
        self.agent_executor = None
        
        # Script generation session state
        self.current_session_id: Optional[str] = None
        self.generation_history: List[Dict[str, Any]] = []
        
        logger.info("Alita Script Generation Tool initialized", extra={
            'operation': 'SCRIPT_TOOL_INIT',
            'config': {
                'model_name': self.config.model_name,
                'supported_languages': self.config.supported_languages,
                'template_caching': self.config.enable_template_caching
            }
        })
    
    def _initialize_llm(self) -> Optional[ChatOpenAI]:
        """Initialize LangChain LLM with OpenRouter per user rules (gemini-2.5-pro for orchestration)"""
        try:
            # Check for OpenRouter API key
            if not self.config.openrouter_api_key:
                logger.warning("OPENROUTER_API_KEY not found in environment", extra={
                    'operation': 'LLM_INIT_WARNING',
                    'hint': 'Set OPENROUTER_API_KEY environment variable to enable LLM functionality'
                })
                return None
            
            # Configure for OpenRouter with user rules models
            llm = ChatOpenAI(
                model_name=self.config.orchestration_model,  # google/gemini-2.5-pro for orchestration
                openai_api_key=self.config.openrouter_api_key,
                openai_api_base=self.config.openrouter_base_url,
                temperature=0.1,  # Lower temperature for more consistent code generation
                max_tokens=4000,
                timeout=60
            )
            
            logger.info("LangChain LLM initialized with OpenRouter per user rules", extra={
                'operation': 'LLM_INIT_SUCCESS',
                'orchestration_model': self.config.orchestration_model,
                'webagent_model': self.config.webagent_model,
                'vision_model': self.config.vision_model,
                'base_url': self.config.openrouter_base_url
            })
            
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter LLM: {str(e)}", extra={
                'operation': 'LLM_INIT_FAILED',
                'error': str(e),
                'hint': 'Check OPENROUTER_API_KEY and network connectivity'
            })
            return None
    
    def _create_specialized_llm(self, purpose: str) -> Optional[ChatOpenAI]:
        """Create specialized LLM for specific purposes per user rules"""
        if not self.config.openrouter_api_key:
            return None
            
        try:
            # Select model based on purpose per user rules
            if purpose == "webagent":
                model = self.config.webagent_model  # claude-4-sonnet for web agent
            elif purpose == "vision":
                model = self.config.vision_model    # o3 for vision tasks
            elif purpose == "orchestration":
                model = self.config.orchestration_model  # gemini-2.5-pro for orchestration
            else:
                model = self.config.orchestration_model  # default to orchestration
            
            llm = ChatOpenAI(
                model_name=model,
                openai_api_key=self.config.openrouter_api_key,
                openai_api_base=self.config.openrouter_base_url,
                temperature=0.1,
                max_tokens=4000,
                timeout=60
            )
            
            logger.info(f"Specialized LLM created for {purpose}", extra={
                'operation': 'SPECIALIZED_LLM_CREATED',
                'purpose': purpose,
                'model': model
            })
            
            return llm
            
        except Exception as e:
            logger.error(f"Failed to create specialized LLM for {purpose}: {str(e)}", extra={
                'operation': 'SPECIALIZED_LLM_FAILED',
                'purpose': purpose,
                'error': str(e)
            })
            return None
    
    async def initialize_agent_executor(self):
        """Initialize LangChain agent executor for intelligent script generation"""
        try:
            # Check if LLM is available
            if self.llm is None:
                logger.warning("LLM not available, agent executor will be disabled", extra={
                    'operation': 'AGENT_INIT_SKIPPED',
                    'reason': 'No LLM available (likely missing OPENROUTER_API_KEY)'
                })
                self.agent_executor = None
                return
            
            # Define tools for the agent
            tools = [
                self._create_mcp_brainstorming_tool(),
                self._create_github_analysis_tool(),
                self._create_kgot_enhancement_tool(),
                self._create_template_generation_tool()
            ]
            
            # Create agent prompt optimized for user's model setup
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert script generation assistant implementing Alita's ScriptGeneratingTool.
Using advanced orchestration capabilities with gemini-2.5-pro for intelligent reasoning,
claude-4-sonnet for web agent tasks, and o3 for vision-related processing.

Your role is to:
1. Analyze task requirements and break them into subtasks using orchestration model
2. Process GitHub repositories for relevant code examples via web agent capabilities
3. Generate high-quality, executable scripts with proper environment setup
4. Ensure scripts are self-contained, well-documented, and follow best practices
5. Create comprehensive setup and cleanup scripts
6. Handle multimodal inputs when vision capabilities are required

Use the available tools to gather information and generate scripts methodically.
Always prioritize code quality, security, and maintainability."""),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create agent with orchestration LLM
            agent = create_openai_functions_agent(self.llm, tools, prompt)
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=10,
                return_intermediate_steps=True
            )
            
            logger.info("LangChain agent executor initialized with user rules models", extra={
                'operation': 'AGENT_INIT_SUCCESS',
                'tools_count': len(tools),
                'orchestration_model': self.config.orchestration_model
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize agent executor: {str(e)}", extra={
                'operation': 'AGENT_INIT_FAILED',
                'error': str(e)
            })
            self.agent_executor = None
    
    def _create_mcp_brainstorming_tool(self) -> BaseTool:
        """Create LangChain tool for MCP Brainstorming integration"""
        class MCPBrainstormingTool(BaseTool):
            name = "mcp_brainstorming"
            description = "Get subtask descriptions and suggestions from MCP Brainstorming for script generation"
            
            def _run(self, requirements: str) -> str:
                """Execute MCP Brainstorming analysis"""
                try:
                    # This would be called asynchronously in practice
                    import asyncio
                    loop = asyncio.get_event_loop()
                    
                    req_dict = {'task_description': requirements}
                    subtasks = loop.run_until_complete(
                        self.mcp_bridge.receive_subtask_descriptions(req_dict)
                    )
                    
                    result = {
                        'subtasks_count': len(subtasks),
                        'subtasks': [
                            {
                                'id': st.id,
                                'title': st.title,
                                'description': st.description,
                                'complexity': st.estimated_complexity
                            } for st in subtasks
                        ]
                    }
                    
                    return json.dumps(result, indent=2)
                    
                except Exception as e:
                    return f"Error in MCP Brainstorming: {str(e)}"
            
            async def _arun(self, requirements: str) -> str:
                """Async execution"""
                return self._run(requirements)
        
        return MCPBrainstormingTool()
    
    def _create_github_analysis_tool(self) -> BaseTool:
        """Create LangChain tool for GitHub analysis"""
        class GitHubAnalysisTool(BaseTool):
            name = "github_analysis"
            description = "Analyze GitHub repositories and extract relevant code snippets"
            
            def _run(self, github_urls: str) -> str:
                """Execute GitHub analysis"""
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    
                    urls_list = github_urls.split(',')
                    processed_links = loop.run_until_complete(
                        self.github_processor.process_github_links(urls_list)
                    )
                    
                    result = {
                        'processed_count': len(processed_links),
                        'repositories': [
                            {
                                'name': link.repository_name,
                                'language': link.language,
                                'description': link.description,
                                'stars': link.stars
                            } for link in processed_links
                        ]
                    }
                    
                    return json.dumps(result, indent=2)
                    
                except Exception as e:
                    return f"Error in GitHub analysis: {str(e)}"
            
            async def _arun(self, github_urls: str) -> str:
                """Async execution"""
                return self._run(github_urls)
        
        return GitHubAnalysisTool()
    
    def _create_kgot_enhancement_tool(self) -> BaseTool:
        """Create LangChain tool for KGoT code enhancement"""
        class KGoTEnhancementTool(BaseTool):
            name = "kgot_enhancement"
            description = "Enhance code generation using KGoT Python Tool capabilities"
            
            def _run(self, code_requirements: str) -> str:
                """Execute KGoT enhancement"""
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    
                    enhanced_result = loop.run_until_complete(
                        self.kgot_bridge.enhance_code_generation(code_requirements, {})
                    )
                    
                    return json.dumps(enhanced_result, indent=2)
                    
                except Exception as e:
                    return f"Error in KGoT enhancement: {str(e)}"
            
            async def _arun(self, code_requirements: str) -> str:
                """Async execution"""
                return self._run(code_requirements)
        
        return KGoTEnhancementTool()
    
    def _create_template_generation_tool(self) -> BaseTool:
        """Create LangChain tool for template-based generation"""
        class TemplateGenerationTool(BaseTool):
            name = "template_generation"
            description = "Generate scripts using RAG-MCP template engine"
            
            def _run(self, template_spec: str) -> str:
                """Execute template generation"""
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    
                    # Parse template specification
                    spec = json.loads(template_spec)
                    template_name = spec.get('template', 'python_script')
                    context = spec.get('context', {})
                    
                    generated_content = loop.run_until_complete(
                        self.template_engine.generate_from_template(template_name, context)
                    )
                    
                    return f"Generated script content:\n{generated_content}"
                    
                except Exception as e:
                    return f"Error in template generation: {str(e)}"
            
            async def _arun(self, template_spec: str) -> str:
                """Async execution"""
                return self._run(template_spec)
        
        return TemplateGenerationTool()
    
    async def generate_script(self, 
                            task_description: str,
                            requirements: Optional[Dict[str, Any]] = None,
                            github_urls: Optional[List[str]] = None,
                            options: Optional[Dict[str, Any]] = None) -> GeneratedScript:
        """
        Main script generation method implementing complete Alita Section 2.3.2 workflow
        
        Args:
            task_description: Description of the task requiring a script
            requirements: Additional requirements and constraints
            github_urls: List of GitHub URLs for reference and code extraction
            options: Additional generation options
            
        Returns:
            Complete generated script with environment setup and cleanup
        """
        try:
            start_time = datetime.now()
            
            logger.info("Starting script generation", extra={
                'operation': 'SCRIPT_GENERATE_START',
                'task_description': task_description[:100] + '...' if len(task_description) > 100 else task_description,
                'requirements_keys': list((requirements or {}).keys()),
                'github_urls_count': len(github_urls or [])
            })
            
            # Initialize session with MCP Brainstorming
            self.current_session_id = await self.mcp_bridge.initialize_session(task_description)
            
            # Step 1: Receive subtask descriptions from MCP Brainstorming
            logger.info("Step 1: Receiving subtask descriptions", extra={'operation': 'STEP_1_SUBTASKS'})
            subtasks = await self.mcp_bridge.receive_subtask_descriptions(requirements or {})
            
            # Step 2: Process GitHub links from Web Agent
            github_links = []
            if github_urls:
                logger.info("Step 2: Processing GitHub links", extra={'operation': 'STEP_2_GITHUB'})
                github_links = await self.github_processor.process_github_links(
                    github_urls, 
                    task_description
                )
                
                # Extract code snippets
                requirements_list = [st.title for st in subtasks]
                code_snippets = await self.github_processor.extract_code_snippets(
                    github_links, 
                    requirements_list
                )
            else:
                code_snippets = {}
            
            # Step 3: Generate environment specification
            logger.info("Step 3: Generating environment specification", extra={'operation': 'STEP_3_ENVIRONMENT'})
            env_spec = self.env_generator.generate_environment_spec(
                requirements or {}, 
                github_links
            )
            
            # Step 4: Use LangChain agent for intelligent script generation
            logger.info("Step 4: Intelligent script generation with LangChain", extra={'operation': 'STEP_4_LANGCHAIN'})
            
            if not self.agent_executor:
                await self.initialize_agent_executor()
            
            # Prepare agent input
            agent_input = {
                'input': f"""Generate a comprehensive script for the following task:

Task Description: {task_description}

Requirements: {json.dumps(requirements or {}, indent=2)}

Subtasks identified:
{json.dumps([{'title': st.title, 'description': st.description} for st in subtasks], indent=2)}

Environment: {env_spec.language} {env_spec.version}

Please generate a complete, executable script with proper error handling, logging, and documentation."""
            }
            
            # Execute with agent if available, otherwise use direct generation
            if self.agent_executor:
                try:
                    agent_result = await self.agent_executor.ainvoke(agent_input)
                    generated_code = agent_result.get('output', '')
                except Exception as e:
                    logger.warning(f"Agent execution failed, using direct generation: {str(e)}", extra={
                        'operation': 'AGENT_FALLBACK'
                    })
                    generated_code = await self._direct_script_generation(subtasks, env_spec, code_snippets)
            else:
                generated_code = await self._direct_script_generation(subtasks, env_spec, code_snippets)
            
            # Step 5: Enhance with KGoT Python Tool
            logger.info("Step 5: Enhancing with KGoT Python Tool", extra={'operation': 'STEP_5_KGOT'})
            kgot_enhancement = await self.kgot_bridge.enhance_code_generation(
                task_description,
                {
                    'subtasks': subtasks,
                    'environment': env_spec.dict(),
                    'code_snippets': code_snippets
                }
            )
            
            # Merge KGoT enhancements if available
            if kgot_enhancement.get('kgot_integration'):
                enhanced_code = kgot_enhancement.get('enhanced_code', generated_code)
                if enhanced_code and len(enhanced_code) > len(generated_code):
                    generated_code = enhanced_code
            
            # Step 6: Generate setup and cleanup scripts
            logger.info("Step 6: Generating setup and cleanup scripts", extra={'operation': 'STEP_6_SCRIPTS'})
            script_name = self._generate_script_name(task_description)
            
            setup_script = await self.env_generator.generate_setup_script(env_spec, script_name)
            cleanup_script = await self.env_generator.generate_cleanup_script(env_spec, script_name)
            
            # Step 7: Create final GeneratedScript object
            logger.info("Step 7: Finalizing generated script", extra={'operation': 'STEP_7_FINALIZE'})
            
            script_id = f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{script_name}"
            
            generated_script = GeneratedScript(
                id=script_id,
                name=script_name,
                description=task_description,
                language=env_spec.language,
                code=generated_code,
                environment_spec=env_spec,
                setup_script=setup_script,
                cleanup_script=cleanup_script,
                execution_instructions=self._generate_execution_instructions(env_spec, script_name),
                test_cases=self._generate_test_cases(subtasks, env_spec),
                documentation=self._generate_documentation(task_description, subtasks, env_spec),
                github_sources=github_links,
                created_at=datetime.now()
            )
            
            # Log generation completion
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info("Script generation completed successfully", extra={
                'operation': 'SCRIPT_GENERATE_SUCCESS',
                'script_id': script_id,
                'duration_seconds': duration,
                'code_length': len(generated_code),
                'subtasks_count': len(subtasks),
                'github_sources_count': len(github_links)
            })
            
            # Add to generation history
            self.generation_history.append({
                'script_id': script_id,
                'task_description': task_description,
                'duration': duration,
                'success': True,
                'timestamp': datetime.now().isoformat()
            })
            
            return generated_script
            
        except Exception as e:
            logger.error(f"Script generation failed: {str(e)}", extra={
                'operation': 'SCRIPT_GENERATE_FAILED',
                'error': str(e),
                'task_description': task_description[:100]
            })
            
            # Add failure to history
            self.generation_history.append({
                'task_description': task_description,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            raise
    
    async def _direct_script_generation(self, 
                                      subtasks: List[SubtaskDescription], 
                                      env_spec: EnvironmentSpec,
                                      code_snippets: Dict[str, str]) -> str:
        """Direct script generation without LangChain agent"""
        try:
            # Determine appropriate template
            template_name = f"{env_spec.language}_script"
            
            # Prepare template context
            context = {
                'script_name': 'generated_script',
                'script_title': 'Generated Script',
                'script_description': 'Auto-generated script based on requirements',
                'additional_imports': self._generate_imports(env_spec, code_snippets),
                'environment_validation': self._generate_env_validation_code(env_spec),
                'main_functions': self._generate_main_functions(subtasks, code_snippets),
                'main_logic': self._generate_main_logic(subtasks)
            }
            
            # Generate using template engine
            generated_code = await self.template_engine.generate_from_template(template_name, context)
            
            return generated_code
            
        except Exception as e:
            logger.error(f"Direct script generation failed: {str(e)}", extra={
                'operation': 'DIRECT_GENERATION_FAILED'
            })
            
            # Ultimate fallback
            return self._generate_minimal_script(env_spec)
    
    def _generate_script_name(self, task_description: str) -> str:
        """Generate appropriate script name from task description"""
        # Clean and truncate task description
        clean_desc = ''.join(c if c.isalnum() else '_' for c in task_description.lower())
        clean_desc = clean_desc[:30]  # Limit length
        
        # Remove consecutive underscores
        while '__' in clean_desc:
            clean_desc = clean_desc.replace('__', '_')
        
        return clean_desc.strip('_') or 'generated_script'
    
    def _generate_imports(self, env_spec: EnvironmentSpec, code_snippets: Dict[str, str]) -> str:
        """Generate additional imports based on dependencies and code snippets"""
        imports = []
        
        # Add imports based on dependencies
        for dep in env_spec.dependencies:
            if dep in ['requests', 'aiohttp']:
                imports.append(f"import {dep}")
            elif dep == 'pyyaml':
                imports.append("import yaml")
            elif dep == 'python-dotenv':
                imports.append("from dotenv import load_dotenv")
        
        # Extract imports from code snippets
        for snippet in code_snippets.values():
            # Basic import extraction (would be more sophisticated in practice)
            lines = snippet.split('\n')
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    imports.append(stripped)
        
        return '\n'.join(sorted(set(imports)))
    
    def _generate_env_validation_code(self, env_spec: EnvironmentSpec) -> str:
        """Generate environment validation code"""
        validation_lines = [
            "# Environment validation",
            "def validate_environment():",
            "    \"\"\"Validate required environment and dependencies\"\"\"",
            "    logger.info('Validating environment')"
        ]
        
        # Add dependency checks
        for dep in env_spec.dependencies:
            validation_lines.append(f"    try:")
            validation_lines.append(f"        import {dep.replace('-', '_')}")
            validation_lines.append(f"        logger.info('{dep} is available')")
            validation_lines.append(f"    except ImportError:")
            validation_lines.append(f"        logger.error('{dep} is not installed')")
            validation_lines.append(f"        raise")
        
        # Add environment variable checks
        for env_var in env_spec.environment_variables:
            validation_lines.append(f"    if not os.getenv('{env_var}'):")
            validation_lines.append(f"        logger.warning('Environment variable {env_var} not set')")
        
        validation_lines.append("")
        validation_lines.append("# Validate environment on import")
        validation_lines.append("validate_environment()")
        
        return '\n'.join(validation_lines)
    
    def _generate_main_functions(self, subtasks: List[SubtaskDescription], code_snippets: Dict[str, str]) -> str:
        """Generate main functions based on subtasks"""
        functions = []
        
        for subtask in subtasks:
            func_name = ''.join(c if c.isalnum() else '_' for c in subtask.title.lower())
            func_name = func_name.strip('_')
            
            function_code = f'''
def {func_name}():
    """
    {subtask.description}
    
    Requirements: {', '.join(subtask.requirements)}
    Priority: {subtask.priority}
    Complexity: {subtask.estimated_complexity}
    """
    logger.info("Executing: {subtask.title}")
    
    try:
        # Implementation for {subtask.title}
        # TODO: Add specific implementation based on requirements
        
        logger.info("Completed: {subtask.title}")
        return True
        
    except Exception as e:
        logger.error(f"Failed {subtask.title}: {{str(e)}}")
        raise
'''
            functions.append(function_code)
        
        return '\n'.join(functions)
    
    def _generate_main_logic(self, subtasks: List[SubtaskDescription]) -> str:
        """Generate main execution logic"""
        logic_lines = []
        
        # Sort subtasks by priority
        sorted_subtasks = sorted(subtasks, key=lambda x: x.priority)
        
        for subtask in sorted_subtasks:
            func_name = ''.join(c if c.isalnum() else '_' for c in subtask.title.lower())
            func_name = func_name.strip('_')
            
            logic_lines.append(f"        # Execute: {subtask.title}")
            logic_lines.append(f"        {func_name}()")
            logic_lines.append("")
        
        return '\n'.join(logic_lines)
    
    def _generate_minimal_script(self, env_spec: EnvironmentSpec) -> str:
        """Generate minimal fallback script"""
        if env_spec.language == 'python':
            return '''#!/usr/bin/env python3
"""
Minimal Generated Script
Generated by Alita Script Generation Tool (Fallback)
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    logger.info("Script execution started")
    # TODO: Add implementation
    logger.info("Script execution completed")

if __name__ == "__main__":
    main()
'''
        else:
            return f'''#!/bin/bash
# Minimal Generated Script
# Language: {env_spec.language}
# Generated by Alita Script Generation Tool (Fallback)

echo "Script execution started"
# TODO: Add implementation
echo "Script execution completed"
'''
    
    def _generate_execution_instructions(self, env_spec: EnvironmentSpec, script_name: str) -> str:
        """Generate execution instructions for the script"""
        if env_spec.language == 'python':
            return f'''
Execution Instructions for {script_name}:

1. Setup Environment:
   bash setup_script.sh

2. Execute Script:
   python{env_spec.version} {script_name}.py

3. Cleanup (optional):
   bash cleanup_script.sh

Environment Requirements:
- Python {env_spec.version}
- Dependencies: {', '.join(env_spec.dependencies)}
- System Requirements: {', '.join(env_spec.system_requirements)}

Environment Variables:
{chr(10).join([f"- {k}={v}" for k, v in env_spec.environment_variables.items()])}
'''
        else:
            return f'''
Execution Instructions for {script_name}:

1. Setup Environment:
   bash setup_script.sh

2. Execute Script:
   bash {script_name}.sh

3. Cleanup (optional):
   bash cleanup_script.sh

Requirements:
- {env_spec.language} {env_spec.version}
- Dependencies: {', '.join(env_spec.dependencies)}
'''
    
    def _generate_test_cases(self, subtasks: List[SubtaskDescription], env_spec: EnvironmentSpec) -> List[str]:
        """Generate basic test cases for the script"""
        test_cases = []
        
        if env_spec.language == 'python':
            test_cases.append(f"""
# Test case 1: Basic execution
python{env_spec.version} script.py --help
""")
            
            test_cases.append(f"""
# Test case 2: Environment validation
python{env_spec.version} -c "import script; script.validate_environment()"
""")
        
        for subtask in subtasks:
            test_cases.append(f"""
# Test case for: {subtask.title}
# {subtask.description}
# Expected: Successful execution with proper logging
""")
        
        return test_cases
    
    def _generate_documentation(self, task_description: str, subtasks: List[SubtaskDescription], env_spec: EnvironmentSpec) -> str:
        """Generate comprehensive documentation for the script"""
        doc_lines = [
            f"# Script Documentation",
            f"",
            f"## Overview",
            f"{task_description}",
            f"",
            f"## Environment",
            f"- Language: {env_spec.language} {env_spec.version}",
            f"- Dependencies: {', '.join(env_spec.dependencies)}",
            f"- System Requirements: {', '.join(env_spec.system_requirements)}",
            f"",
            f"## Subtasks",
            ""
        ]
        
        for i, subtask in enumerate(subtasks, 1):
            doc_lines.extend([
                f"### {i}. {subtask.title}",
                f"**Description:** {subtask.description}",
                f"**Priority:** {subtask.priority}",
                f"**Complexity:** {subtask.estimated_complexity}",
                f"**Requirements:** {', '.join(subtask.requirements)}",
                ""
            ])
        
        doc_lines.extend([
            "## Usage",
            "1. Run setup script to prepare environment",
            "2. Execute main script",
            "3. Run cleanup script when finished",
            "",
            "## Generated by",
            "Alita Script Generation Tool with KGoT Integration",
            f"Generated at: {datetime.now().isoformat()}"
        ])
        
        return '\n'.join(doc_lines)

    def get_generation_history(self) -> List[Dict[str, Any]]:
        """Get script generation history"""
        return self.generation_history.copy()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        successful = sum(1 for h in self.generation_history if h.get('success', False))
        failed = len(self.generation_history) - successful
        
        return {
            'current_session_id': self.current_session_id,
            'total_generations': len(self.generation_history),
            'successful_generations': successful,
            'failed_generations': failed,
            'success_rate': successful / len(self.generation_history) if self.generation_history else 0,
            'avg_duration': sum(h.get('duration', 0) for h in self.generation_history if h.get('duration')) / max(successful, 1)
        }

# Main entry point and example usage
async def main():
    """
    Example usage of the Alita Script Generation Tool
    """
    try:
        # Initialize configuration
        config = ScriptGenerationConfig()
        
        # Create script generator
        script_generator = ScriptGeneratingTool(config)
        
        # Example task
        task_description = """
        Create a data processing script that downloads CSV files from multiple URLs,
        performs data cleaning and validation, and generates summary reports.
        The script should handle errors gracefully and provide detailed logging.
        """
        
        requirements = {
            'language': 'python',
            'dependencies': ['pandas', 'requests', 'matplotlib'],
            'output_format': 'csv',
            'logging_level': 'INFO'
        }
        
        github_urls = [
            'https://github.com/pandas-dev/pandas',
            'https://github.com/psf/requests'
        ]
        
        # Generate script
        logger.info("Starting example script generation")
        
        generated_script = await script_generator.generate_script(
            task_description=task_description,
            requirements=requirements,
            github_urls=github_urls
        )
        
        logger.info(f"Generated script: {generated_script.id}")
        logger.info(f"Language: {generated_script.language}")
        logger.info(f"Code length: {len(generated_script.code)} characters")
        
        # Print session stats
        stats = script_generator.get_session_stats()
        logger.info(f"Session stats: {stats}")
        
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 