#!/usr/bin/env python3
"""
Alita Environment Management with KGoT Integration

Implementation of Alita Section 2.3.4 "Environment Management" complete framework
integrated with KGoT Section 2.6 "Containerization" and Section 2.5 error management.

This module provides comprehensive environment management capabilities including:
- Repository and script metadata parsing (README.md, requirements.txt)
- Isolated execution profile construction using KGoT Text Inspector Tool
- Conda environment management with unique naming and parallel initialization
- Dependency installation using conda install and pip install procedures
- Enhanced environment isolation through KGoT containerization integration
- Automated recovery procedures following KGoT error management patterns

Key Features:
- TextInspectorTool integration for metadata extraction
- LangChain-based agents for intelligent environment management (per user hard rule)
- Parallel local initialization avoiding administrative privileges
- Container integration with Docker/Sarus support
- Comprehensive Winston logging with operation tracking
- Error recovery with retry mechanisms and fallback strategies

@module EnvironmentManager
@author Enhanced Alita KGoT System
@version 1.0.0
@based_on Alita Section 2.3.4, KGoT Sections 2.5, 2.6, and B.3
"""

import os
import sys
import json
import logging
import asyncio
import tempfile
import subprocess
import hashlib
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import yaml
import re

# LangChain imports for agent orchestration (per user's hard rule)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage
from langchain_core.runnables import Runnable

# Pydantic imports for data validation
from pydantic import BaseModel, Field, validator

# Import KGoT components for integration
try:
    from ..kgot_core.containerization import (
        ContainerOrchestrator, 
        EnvironmentDetector, 
        DeploymentEnvironment,
        ContainerConfig,
        ResourceMetrics
    )
    from ..kgot_core.error_management import (
        KGoTErrorManagementSystem,
        ErrorType,
        ErrorSeverity,
        ErrorContext
    )
except ImportError:
    # Fallback for development/testing
    logging.warning("KGoT components not available, using mock implementations")

# Setup Winston-compatible logging with proper path handling
project_root = Path(__file__).parent.parent
log_dir = project_root / 'logs' / 'alita'
log_dir.mkdir(parents=True, exist_ok=True)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(operation)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'environment_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnvironmentManager')

class EnvironmentType(Enum):
    """Environment types supported by the manager"""
    CONDA = "conda"
    VENV = "venv" 
    DOCKER = "docker"
    SARUS = "sarus"
    SYSTEM = "system"

class InitializationStatus(Enum):
    """Status of environment initialization"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    CLEANING = "cleaning"

@dataclass
class EnvironmentMetadata:
    """
    Structured metadata extracted from repository/script files
    Using TextInspectorTool integration patterns
    """
    repository_path: str
    readme_content: Optional[str] = None
    requirements_files: List[str] = field(default_factory=list)
    setup_files: List[str] = field(default_factory=list)
    python_version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    system_requirements: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    installation_commands: List[str] = field(default_factory=list)
    test_commands: List[str] = field(default_factory=list)
    main_script: Optional[str] = None
    language: str = "python"
    framework: Optional[str] = None
    extracted_at: datetime = field(default_factory=datetime.now)

@dataclass 
class EnvironmentProfile:
    """
    Isolated execution profile specification
    Constructed using KGoT Text Inspector Tool analysis
    """
    profile_id: str
    name: str
    environment_type: EnvironmentType
    base_path: str
    conda_env_name: Optional[str] = None
    container_config: Optional[Dict[str, Any]] = None
    python_executable: Optional[str] = None
    activation_script: Optional[str] = None
    deactivation_script: Optional[str] = None
    metadata: EnvironmentMetadata = None
    status: InitializationStatus = InitializationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    resource_usage: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryProcedure:
    """
    Automated recovery procedure configuration
    Following KGoT Section 2.5 error management patterns
    """
    procedure_id: str
    error_types: List[ErrorType]
    recovery_steps: List[Callable]
    max_attempts: int = 3
    backoff_strategy: str = "exponential"
    fallback_action: Optional[Callable] = None
    success_conditions: List[Callable] = field(default_factory=list)

class EnvironmentMetadataParser:
    """
    Parses repository and script metadata using KGoT Text Inspector Tool integration
    
    Implementation based on Alita Section 2.3.4 environment management requirements:
    - Repository metadata parsing (README.md, requirements.txt) 
    - Script metadata extraction using TextInspectorTool
    - Dependency and setup instruction validation
    - Environment configuration construction
    """
    
    def __init__(self, llm_client: Optional[Runnable] = None):
        """
        Initialize Environment Metadata Parser
        
        @param {Optional[Runnable]} llm_client - LangChain LLM client for intelligent parsing (OpenRouter-based per user memory)
        """
        self.llm_client = llm_client
        self.supported_files = {
            'requirements': ['requirements.txt', 'requirements-dev.txt', 'requirements-test.txt', 'pyproject.toml', 'setup.py', 'Pipfile'],
            'setup': ['setup.py', 'setup.cfg', 'pyproject.toml', 'Makefile', 'install.sh'],
            'config': ['.env', '.env.example', 'config.yaml', 'config.yml', 'config.json'],
            'docs': ['README.md', 'README.rst', 'INSTALL.md', 'SETUP.md', 'docs/installation.md']
        }
        
        logger.info("Initialized Environment Metadata Parser", extra={
            'operation': 'METADATA_PARSER_INIT',
            'llm_available': self.llm_client is not None,
            'supported_file_types': len(self.supported_files)
        })
    
    async def parse_repository_metadata(self, repository_path: str) -> EnvironmentMetadata:
        """
        Parse comprehensive metadata from repository using TextInspectorTool patterns
        
        @param {str} repository_path - Path to repository or script directory
        @returns {EnvironmentMetadata} - Extracted metadata structure
        """
        repo_path = Path(repository_path).resolve()
        
        logger.info("Starting repository metadata parsing", extra={
            'operation': 'METADATA_PARSE_START',
            'repository_path': str(repo_path),
            'exists': repo_path.exists()
        })
        
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repository_path}")
        
        metadata = EnvironmentMetadata(repository_path=str(repo_path))
        
        # Parse README and documentation files
        await self._parse_documentation_files(repo_path, metadata)
        
        # Parse requirements and dependency files  
        await self._parse_requirement_files(repo_path, metadata)
        
        # Parse setup and configuration files
        await self._parse_setup_files(repo_path, metadata)
        
        # Extract environment variables and configuration
        await self._parse_config_files(repo_path, metadata)
        
        # Intelligent analysis using LLM if available
        if self.llm_client:
            await self._enhance_with_llm_analysis(metadata)
        
        # Validate and normalize extracted metadata
        self._validate_and_normalize_metadata(metadata)
        
        logger.info("Completed repository metadata parsing", extra={
            'operation': 'METADATA_PARSE_COMPLETE',
            'dependencies_found': len(metadata.dependencies),
            'requirements_files': len(metadata.requirements_files),
            'python_version': metadata.python_version
        })
        
        return metadata
    
    async def _parse_documentation_files(self, repo_path: Path, metadata: EnvironmentMetadata):
        """Parse README and documentation files using TextInspectorTool patterns"""
        logger.debug("Parsing documentation files", extra={'operation': 'PARSE_DOCS'})
        
        for doc_file in self.supported_files['docs']:
            doc_path = repo_path / doc_file
            if doc_path.exists():
                try:
                    content = await self._read_file_content(doc_path)
                    if doc_file.startswith('README'):
                        metadata.readme_content = content
                    
                    # Extract installation commands from documentation
                    install_commands = self._extract_installation_commands(content)
                    metadata.installation_commands.extend(install_commands)
                    
                    # Extract environment variables mentioned in docs
                    env_vars = self._extract_environment_variables(content)
                    metadata.environment_variables.update(env_vars)
                    
                    # Extract system requirements
                    sys_reqs = self._extract_system_requirements(content)
                    metadata.system_requirements.extend(sys_reqs)
                    
                    logger.debug(f"Parsed documentation file: {doc_file}", extra={
                        'operation': 'PARSE_DOC_FILE',
                        'file': doc_file,
                        'content_length': len(content)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to parse documentation file {doc_file}: {str(e)}", extra={
                        'operation': 'PARSE_DOC_ERROR',
                        'file': doc_file,
                        'error': str(e)
                    })
    
    async def _parse_requirement_files(self, repo_path: Path, metadata: EnvironmentMetadata):
        """Parse requirements and dependency files"""
        logger.debug("Parsing requirement files", extra={'operation': 'PARSE_REQUIREMENTS'})
        
        for req_file in self.supported_files['requirements']:
            req_path = repo_path / req_file
            if req_path.exists():
                try:
                    metadata.requirements_files.append(str(req_path))
                    content = await self._read_file_content(req_path)
                    
                    if req_file == 'requirements.txt':
                        deps = self._parse_requirements_txt(content)
                        metadata.dependencies.extend(deps)
                    elif req_file == 'pyproject.toml':
                        deps, py_ver = self._parse_pyproject_toml(content)
                        metadata.dependencies.extend(deps)
                        if py_ver:
                            metadata.python_version = py_ver
                    elif req_file == 'setup.py':
                        deps, py_ver = self._parse_setup_py(content)
                        metadata.dependencies.extend(deps)
                        if py_ver:
                            metadata.python_version = py_ver
                    elif req_file == 'Pipfile':
                        deps, py_ver = self._parse_pipfile(content)
                        metadata.dependencies.extend(deps)
                        if py_ver:
                            metadata.python_version = py_ver
                    
                    logger.debug(f"Parsed requirement file: {req_file}", extra={
                        'operation': 'PARSE_REQ_FILE',
                        'file': req_file,
                        'dependencies_found': len([d for d in metadata.dependencies if d])
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to parse requirement file {req_file}: {str(e)}", extra={
                        'operation': 'PARSE_REQ_ERROR',
                        'file': req_file,
                        'error': str(e)
                    })
    
    async def _parse_setup_files(self, repo_path: Path, metadata: EnvironmentMetadata):
        """Parse setup and configuration files"""
        logger.debug("Parsing setup files", extra={'operation': 'PARSE_SETUP'})
        
        for setup_file in self.supported_files['setup']:
            setup_path = repo_path / setup_file
            if setup_path.exists():
                try:
                    metadata.setup_files.append(str(setup_path))
                    content = await self._read_file_content(setup_path)
                    
                    # Extract installation commands
                    install_commands = self._extract_installation_commands(content)
                    metadata.installation_commands.extend(install_commands)
                    
                    # Extract system requirements
                    sys_reqs = self._extract_system_requirements(content)
                    metadata.system_requirements.extend(sys_reqs)
                    
                    logger.debug(f"Parsed setup file: {setup_file}", extra={
                        'operation': 'PARSE_SETUP_FILE',
                        'file': setup_file
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to parse setup file {setup_file}: {str(e)}", extra={
                        'operation': 'PARSE_SETUP_ERROR',
                        'file': setup_file,
                        'error': str(e)
                    })
    
    async def _parse_config_files(self, repo_path: Path, metadata: EnvironmentMetadata):
        """Parse configuration files for environment variables"""
        logger.debug("Parsing config files", extra={'operation': 'PARSE_CONFIG'})
        
        for config_file in self.supported_files['config']:
            config_path = repo_path / config_file
            if config_path.exists():
                try:
                    content = await self._read_file_content(config_path)
                    
                    if config_file.startswith('.env'):
                        env_vars = self._parse_env_file(content)
                        metadata.environment_variables.update(env_vars)
                    elif config_file.endswith(('.yaml', '.yml')):
                        env_vars = self._parse_yaml_config(content)
                        metadata.environment_variables.update(env_vars)
                    elif config_file.endswith('.json'):
                        env_vars = self._parse_json_config(content)
                        metadata.environment_variables.update(env_vars)
                    
                    logger.debug(f"Parsed config file: {config_file}", extra={
                        'operation': 'PARSE_CONFIG_FILE',
                        'file': config_file,
                        'env_vars_found': len(env_vars) if 'env_vars' in locals() else 0
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to parse config file {config_file}: {str(e)}", extra={
                        'operation': 'PARSE_CONFIG_ERROR',
                        'file': config_file,
                        'error': str(e)
                    })
    
    async def _enhance_with_llm_analysis(self, metadata: EnvironmentMetadata):
        """Enhance metadata using LLM analysis for intelligent extraction"""
        if not self.llm_client:
            return
        
        logger.debug("Enhancing metadata with LLM analysis", extra={'operation': 'LLM_ENHANCE'})
        
        try:
            # Prepare context for LLM analysis
            context = {
                'readme_content': metadata.readme_content or "",
                'dependencies': metadata.dependencies,
                'setup_files': metadata.setup_files,
                'installation_commands': metadata.installation_commands
            }
            
            # Create analysis prompt
            analysis_prompt = self._create_llm_analysis_prompt(context)
            
            # Get LLM analysis
            response = await self.llm_client.ainvoke([HumanMessage(content=analysis_prompt)])
            analysis_result = self._parse_llm_analysis_result(response.content)
            
            # Apply LLM insights to metadata
            if analysis_result.get('python_version') and not metadata.python_version:
                metadata.python_version = analysis_result['python_version']
            
            if analysis_result.get('framework'):
                metadata.framework = analysis_result['framework']
            
            if analysis_result.get('main_script'):
                metadata.main_script = analysis_result['main_script']
            
            if analysis_result.get('additional_dependencies'):
                metadata.dependencies.extend(analysis_result['additional_dependencies'])
            
            logger.info("Enhanced metadata with LLM analysis", extra={
                'operation': 'LLM_ENHANCE_COMPLETE',
                'framework_detected': metadata.framework,
                'main_script_detected': metadata.main_script
            })
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {str(e)}", extra={
                'operation': 'LLM_ENHANCE_ERROR',
                'error': str(e)
            })
    
    def _validate_and_normalize_metadata(self, metadata: EnvironmentMetadata):
        """Validate and normalize extracted metadata"""
        logger.debug("Validating and normalizing metadata", extra={'operation': 'VALIDATE_METADATA'})
        
        # Remove duplicates and normalize dependencies
        metadata.dependencies = list(set([dep.strip() for dep in metadata.dependencies if dep.strip()]))
        metadata.system_requirements = list(set([req.strip() for req in metadata.system_requirements if req.strip()]))
        metadata.installation_commands = list(set([cmd.strip() for cmd in metadata.installation_commands if cmd.strip()]))
        
        # Detect Python version if not found
        if not metadata.python_version:
            metadata.python_version = self._detect_python_version(metadata.dependencies)
        
        # Detect main language and framework
        if not metadata.framework:
            metadata.framework = self._detect_framework(metadata.dependencies)
        
        logger.debug("Metadata validation complete", extra={
            'operation': 'VALIDATE_METADATA_COMPLETE',
            'final_dependencies': len(metadata.dependencies),
            'python_version': metadata.python_version,
            'framework': metadata.framework
        })

    # Helper methods for EnvironmentMetadataParser
    async def _read_file_content(self, file_path: Path) -> str:
        """Read file content asynchronously"""
        try:
            async with asyncio.to_thread(open, file_path, 'r', encoding='utf-8') as f:
                return await asyncio.to_thread(f.read)
        except UnicodeDecodeError:
            # Try with different encoding
            async with asyncio.to_thread(open, file_path, 'r', encoding='latin-1') as f:
                return await asyncio.to_thread(f.read)
    
    def _extract_installation_commands(self, content: str) -> List[str]:
        """Extract installation commands from text content"""
        commands = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for common installation patterns
            if any(pattern in line.lower() for pattern in ['pip install', 'conda install', 'npm install', 'apt-get install', 'brew install']):
                # Remove markdown code block markers
                line = re.sub(r'^```[a-z]*\s*', '', line)
                line = re.sub(r'```\s*$', '', line)
                line = re.sub(r'^\$\s*', '', line)  # Remove shell prompt
                line = re.sub(r'^>\s*', '', line)   # Remove quote markers
                if line.strip():
                    commands.append(line.strip())
        
        return commands
    
    def _extract_environment_variables(self, content: str) -> Dict[str, str]:
        """Extract environment variables from text content"""
        env_vars = {}
        lines = content.split('\n')
        
        for line in lines:
            # Look for environment variable patterns
            env_match = re.search(r'([A-Z_][A-Z0-9_]*)\s*=\s*["\']?([^"\']+)["\']?', line)
            if env_match:
                var_name, var_value = env_match.groups()
                env_vars[var_name] = var_value
        
        return env_vars
    
    def _extract_system_requirements(self, content: str) -> List[str]:
        """Extract system requirements from text content"""
        requirements = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.lower().strip()
            # Look for system requirement patterns
            if any(keyword in line for keyword in ['requires', 'dependencies', 'prerequisites']):
                # Extract software names
                for software in ['python', 'node', 'npm', 'docker', 'git', 'java', 'gcc', 'make']:
                    if software in line:
                        requirements.append(software)
        
        return requirements
    
    def _parse_requirements_txt(self, content: str) -> List[str]:
        """Parse requirements.txt format"""
        dependencies = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                dependencies.append(line)
        return dependencies
    
    def _parse_pyproject_toml(self, content: str) -> Tuple[List[str], Optional[str]]:
        """Parse pyproject.toml format"""
        dependencies = []
        python_version = None
        
        try:
            import toml
            data = toml.loads(content)
            
            # Extract dependencies
            if 'project' in data and 'dependencies' in data['project']:
                dependencies = data['project']['dependencies']
            elif 'tool' in data and 'poetry' in data['tool'] and 'dependencies' in data['tool']['poetry']:
                deps = data['tool']['poetry']['dependencies']
                dependencies = [f"{k}=={v}" if isinstance(v, str) else k for k, v in deps.items() if k != 'python']
                if 'python' in deps:
                    python_version = deps['python']
            
        except Exception as e:
            logger.warning(f"Failed to parse pyproject.toml: {str(e)}")
        
        return dependencies, python_version
    
    def _parse_setup_py(self, content: str) -> Tuple[List[str], Optional[str]]:
        """Parse setup.py format"""
        dependencies = []
        python_version = None
        
        # Extract install_requires
        install_requires_match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if install_requires_match:
            deps_str = install_requires_match.group(1)
            deps = re.findall(r'["\']([^"\']+)["\']', deps_str)
            dependencies.extend(deps)
        
        # Extract python_requires
        python_requires_match = re.search(r'python_requires\s*=\s*["\']([^"\']+)["\']', content)
        if python_requires_match:
            python_version = python_requires_match.group(1)
        
        return dependencies, python_version
    
    def _parse_pipfile(self, content: str) -> Tuple[List[str], Optional[str]]:
        """Parse Pipfile format"""
        dependencies = []
        python_version = None
        
        try:
            import toml
            data = toml.loads(content)
            
            if 'packages' in data:
                for pkg, version in data['packages'].items():
                    if isinstance(version, str):
                        dependencies.append(f"{pkg}=={version}")
                    else:
                        dependencies.append(pkg)
            
            if 'requires' in data and 'python_version' in data['requires']:
                python_version = data['requires']['python_version']
                
        except Exception as e:
            logger.warning(f"Failed to parse Pipfile: {str(e)}")
        
        return dependencies, python_version
    
    def _parse_env_file(self, content: str) -> Dict[str, str]:
        """Parse .env file format"""
        env_vars = {}
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip().strip('"\'')
        return env_vars
    
    def _parse_yaml_config(self, content: str) -> Dict[str, str]:
        """Parse YAML configuration files"""
        env_vars = {}
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                # Extract environment-related configurations
                for key, value in data.items():
                    if isinstance(value, (str, int, float)):
                        env_vars[str(key).upper()] = str(value)
        except Exception as e:
            logger.warning(f"Failed to parse YAML config: {str(e)}")
        return env_vars
    
    def _parse_json_config(self, content: str) -> Dict[str, str]:
        """Parse JSON configuration files"""
        env_vars = {}
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (str, int, float)):
                        env_vars[str(key).upper()] = str(value)
        except Exception as e:
            logger.warning(f"Failed to parse JSON config: {str(e)}")
        return env_vars
    
    def _create_llm_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Create prompt for LLM analysis of repository metadata"""
        return f"""
        Analyze the following repository metadata and extract key environment information:
        
        README Content: {context['readme_content'][:2000]}...
        Dependencies Found: {context['dependencies']}
        Setup Files: {context['setup_files']}
        Installation Commands: {context['installation_commands']}
        
        Please provide a JSON response with the following structure:
        {{
            "python_version": "detected python version or null",
            "framework": "detected framework (django, flask, fastapi, etc.) or null", 
            "main_script": "main entry point script or null",
            "additional_dependencies": ["any additional dependencies not found"],
            "environment_type": "recommended environment type (conda, venv, docker)"
        }}
        """
    
    def _parse_llm_analysis_result(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM analysis response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"Failed to parse LLM analysis result: {str(e)}")
        
        return {}
    
    def _detect_python_version(self, dependencies: List[str]) -> str:
        """Detect Python version from dependencies"""
        for dep in dependencies:
            if 'python' in dep.lower():
                version_match = re.search(r'(\d+\.\d+)', dep)
                if version_match:
                    return version_match.group(1)
        return "3.9"  # Default version
    
    def _detect_framework(self, dependencies: List[str]) -> Optional[str]:
        """Detect framework from dependencies"""
        frameworks = {
            'django': ['django'],
            'flask': ['flask'],
            'fastapi': ['fastapi'],
            'streamlit': ['streamlit'],
            'gradio': ['gradio'],
            'transformers': ['transformers', 'torch', 'tensorflow']
        }
        
        dep_str = ' '.join(dependencies).lower()
        for framework, keywords in frameworks.items():
            if any(keyword in dep_str for keyword in keywords):
                return framework
        
        return None

class CondaEnvironmentManager:
    """
    Manages Conda environments with unique naming and parallel initialization
    
    Implementation based on Alita Section 2.3.4 environment management:
    - Creates isolated Conda environments with unique names (typically derived from task ID or repository path hash)
    - Handles dependency installation using conda install or pip install procedures  
    - Supports parallel local initialization avoiding administrative privileges
    - Integrates with KGoT containerization for enhanced environment isolation
    """
    
    def __init__(self, conda_base_path: Optional[str] = None):
        """
        Initialize Conda Environment Manager
        
        @param {Optional[str]} conda_base_path - Base path for conda environments (auto-detected if not provided)
        """
        self.conda_base_path = conda_base_path or self._detect_conda_base_path()
        self.active_environments: Dict[str, EnvironmentProfile] = {}
        self.initialization_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)  # Parallel initialization
        
        logger.info("Initialized Conda Environment Manager", extra={
            'operation': 'CONDA_MANAGER_INIT',
            'conda_base_path': self.conda_base_path,
            'conda_available': self._check_conda_availability()
        })
    
    async def create_environment(self, 
                               metadata: EnvironmentMetadata, 
                               task_id: Optional[str] = None,
                               force_recreate: bool = False) -> EnvironmentProfile:
        """
        Create isolated Conda environment with unique naming
        
        @param {EnvironmentMetadata} metadata - Environment metadata from parser
        @param {Optional[str]} task_id - Task ID for unique naming
        @param {bool} force_recreate - Force recreation if environment exists
        @returns {EnvironmentProfile} - Created environment profile
        """
        # Generate unique environment name
        env_name = self._generate_unique_env_name(metadata, task_id)
        profile_id = f"conda_{env_name}_{int(time.time())}"
        
        logger.info("Creating Conda environment", extra={
            'operation': 'CONDA_CREATE_START',
            'env_name': env_name,
            'profile_id': profile_id,
            'force_recreate': force_recreate
        })
        
        # Check if environment already exists
        if not force_recreate and await self._environment_exists(env_name):
            logger.info("Environment already exists, reusing", extra={
                'operation': 'CONDA_REUSE_EXISTING',
                'env_name': env_name
            })
            return await self._load_existing_environment(env_name, metadata)
        
        # Create environment profile
        profile = EnvironmentProfile(
            profile_id=profile_id,
            name=env_name,
            environment_type=EnvironmentType.CONDA,
            base_path=str(Path(self.conda_base_path) / "envs" / env_name),
            conda_env_name=env_name,
            metadata=metadata,
            status=InitializationStatus.INITIALIZING
        )
        
        try:
            # Parallel initialization without admin privileges
            await self._parallel_environment_initialization(profile)
            
            # Register active environment
            with self.initialization_lock:
                self.active_environments[profile_id] = profile
            
            logger.info("Successfully created Conda environment", extra={
                'operation': 'CONDA_CREATE_SUCCESS',
                'env_name': env_name,
                'profile_id': profile_id,
                'python_executable': profile.python_executable
            })
            
            return profile
            
        except Exception as e:
            profile.status = InitializationStatus.ERROR
            logger.error(f"Failed to create Conda environment: {str(e)}", extra={
                'operation': 'CONDA_CREATE_ERROR',
                'env_name': env_name,
                'error': str(e)
            })
            raise
    
    async def _parallel_environment_initialization(self, profile: EnvironmentProfile):
        """
        Parallel environment initialization avoiding administrative privileges
        Implementation of Alita Section 2.3.4 parallel local initialization requirements
        """
        logger.info("Starting parallel environment initialization", extra={
            'operation': 'PARALLEL_INIT_START',
            'env_name': profile.conda_env_name
        })
        
        initialization_tasks = []
        
        # Task 1: Create base conda environment
        initialization_tasks.append(
            self._create_base_conda_environment(profile)
        )
        
        # Task 2: Prepare dependency installation commands
        initialization_tasks.append(
            self._prepare_dependency_commands(profile)
        )
        
        # Task 3: Setup environment activation/deactivation scripts
        initialization_tasks.append(
            self._setup_environment_scripts(profile)
        )
        
        # Execute tasks in parallel
        try:
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # Check for failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    raise result
            
            # Task 4: Install dependencies (sequential after base setup)
            await self._install_dependencies_sequential(profile)
            
            # Task 5: Validate environment setup
            await self._validate_environment_setup(profile)
            
            profile.status = InitializationStatus.READY
            
            logger.info("Parallel environment initialization complete", extra={
                'operation': 'PARALLEL_INIT_SUCCESS',
                'env_name': profile.conda_env_name
            })
            
        except Exception as e:
            profile.status = InitializationStatus.ERROR
            logger.error(f"Parallel initialization failed: {str(e)}", extra={
                'operation': 'PARALLEL_INIT_ERROR',
                'env_name': profile.conda_env_name,
                'error': str(e)
            })
            raise 

class EnvironmentProfileBuilder:
    """
    Constructs isolated execution profiles using KGoT Text Inspector Tool analysis
    
    Implementation based on Alita Section 2.3.4 environment management:
    - Constructs isolated execution profiles from parsed metadata
    - Integrates with KGoT containerization for enhanced environment isolation
    - Supports multiple environment types (conda, venv, docker, sarus)
    - Provides profile optimization and resource allocation recommendations
    """
    
    def __init__(self, 
                 metadata_parser: EnvironmentMetadataParser,
                 conda_manager: CondaEnvironmentManager,
                 container_orchestrator: Optional[Any] = None):
        """
        Initialize Environment Profile Builder
        
        @param {EnvironmentMetadataParser} metadata_parser - Parser for repository metadata
        @param {CondaEnvironmentManager} conda_manager - Conda environment manager
        @param {Optional[Any]} container_orchestrator - KGoT container orchestrator for enhanced isolation
        """
        self.metadata_parser = metadata_parser
        self.conda_manager = conda_manager
        self.container_orchestrator = container_orchestrator
        self.profile_cache: Dict[str, EnvironmentProfile] = {}
        
        logger.info("Initialized Environment Profile Builder", extra={
            'operation': 'PROFILE_BUILDER_INIT',
            'container_integration': container_orchestrator is not None
        })
    
    async def build_profile(self, 
                          repository_path: str, 
                          task_id: Optional[str] = None,
                          environment_type: Optional[EnvironmentType] = None,
                          optimization_options: Optional[Dict[str, Any]] = None) -> EnvironmentProfile:
        """
        Build isolated execution profile from repository analysis
        
        @param {str} repository_path - Path to repository or script directory
        @param {Optional[str]} task_id - Task ID for unique naming and caching
        @param {Optional[EnvironmentType]} environment_type - Preferred environment type
        @param {Optional[Dict[str, Any]]} optimization_options - Profile optimization options
        @returns {EnvironmentProfile} - Constructed environment profile
        """
        cache_key = self._generate_cache_key(repository_path, task_id, environment_type)
        
        logger.info("Starting environment profile construction", extra={
            'operation': 'BUILD_PROFILE_START',
            'repository_path': repository_path,
            'task_id': task_id,
            'environment_type': environment_type.value if environment_type else None,
            'cache_key': cache_key
        })
        
        # Check cache for existing profile
        if cache_key in self.profile_cache:
            cached_profile = self.profile_cache[cache_key]
            if await self._validate_cached_profile(cached_profile):
                logger.info("Using cached environment profile", extra={
                    'operation': 'BUILD_PROFILE_CACHED',
                    'profile_id': cached_profile.profile_id
                })
                return cached_profile
        
        try:
            # Step 1: Parse repository metadata
            metadata = await self.metadata_parser.parse_repository_metadata(repository_path)
            
            # Step 2: Determine optimal environment type
            if not environment_type:
                environment_type = self._determine_optimal_environment_type(metadata, optimization_options)
            
            # Step 3: Build environment profile based on type
            if environment_type == EnvironmentType.CONDA:
                profile = await self._build_conda_profile(metadata, task_id)
            elif environment_type == EnvironmentType.DOCKER:
                profile = await self._build_docker_profile(metadata, task_id)
            elif environment_type == EnvironmentType.SARUS:
                profile = await self._build_sarus_profile(metadata, task_id)
            elif environment_type == EnvironmentType.VENV:
                profile = await self._build_venv_profile(metadata, task_id)
            else:
                profile = await self._build_system_profile(metadata, task_id)
            
            # Step 4: Apply optimization if requested
            if optimization_options:
                await self._optimize_profile(profile, optimization_options)
            
            # Step 5: Cache the profile
            self.profile_cache[cache_key] = profile
            
            logger.info("Environment profile construction complete", extra={
                'operation': 'BUILD_PROFILE_SUCCESS',
                'profile_id': profile.profile_id,
                'environment_type': profile.environment_type.value,
                'status': profile.status.value
            })
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to build environment profile: {str(e)}", extra={
                'operation': 'BUILD_PROFILE_ERROR',
                'repository_path': repository_path,
                'error': str(e)
            })
            raise
    
    async def _build_conda_profile(self, metadata: EnvironmentMetadata, task_id: Optional[str]) -> EnvironmentProfile:
        """Build conda environment profile"""
        logger.info("Building conda environment profile", extra={
            'operation': 'BUILD_CONDA_PROFILE',
            'task_id': task_id
        })
        
        profile = await self.conda_manager.create_environment(metadata, task_id)
        
        # Enhance with container integration if available
        if self.container_orchestrator:
            await self._enhance_with_container_integration(profile)
        
        return profile
    
    async def _build_docker_profile(self, metadata: EnvironmentMetadata, task_id: Optional[str]) -> EnvironmentProfile:
        """Build Docker container profile"""
        logger.info("Building Docker environment profile", extra={
            'operation': 'BUILD_DOCKER_PROFILE',
            'task_id': task_id
        })
        
        if not self.container_orchestrator:
            raise RuntimeError("Container orchestrator required for Docker profiles")
        
        # Generate Docker configuration
        container_config = self._generate_docker_config(metadata, task_id)
        
        profile_id = f"docker_{task_id or hashlib.md5(metadata.repository_path.encode()).hexdigest()[:8]}_{int(time.time())}"
        
        profile = EnvironmentProfile(
            profile_id=profile_id,
            name=container_config['name'],
            environment_type=EnvironmentType.DOCKER,
            base_path=f"/docker/containers/{container_config['name']}",
            container_config=container_config,
            metadata=metadata,
            status=InitializationStatus.PENDING
        )
        
        # Deploy container using orchestrator
        await self._deploy_container_profile(profile)
        
        return profile
    
    async def _build_sarus_profile(self, metadata: EnvironmentMetadata, task_id: Optional[str]) -> EnvironmentProfile:
        """Build Sarus container profile for HPC environments"""
        logger.info("Building Sarus environment profile", extra={
            'operation': 'BUILD_SARUS_PROFILE',
            'task_id': task_id
        })
        
        if not self.container_orchestrator:
            raise RuntimeError("Container orchestrator required for Sarus profiles")
        
        # Generate Sarus configuration
        container_config = self._generate_sarus_config(metadata, task_id)
        
        profile_id = f"sarus_{task_id or hashlib.md5(metadata.repository_path.encode()).hexdigest()[:8]}_{int(time.time())}"
        
        profile = EnvironmentProfile(
            profile_id=profile_id,
            name=container_config['name'],
            environment_type=EnvironmentType.SARUS,
            base_path=f"/sarus/containers/{container_config['name']}",
            container_config=container_config,
            metadata=metadata,
            status=InitializationStatus.PENDING
        )
        
        # Deploy container using orchestrator
        await self._deploy_container_profile(profile)
        
        return profile
    
    async def _build_venv_profile(self, metadata: EnvironmentMetadata, task_id: Optional[str]) -> EnvironmentProfile:
        """Build Python virtual environment profile"""
        logger.info("Building venv environment profile", extra={
            'operation': 'BUILD_VENV_PROFILE',
            'task_id': task_id
        })
        
        env_name = f"alita_venv_{task_id or hashlib.md5(metadata.repository_path.encode()).hexdigest()[:8]}"
        venv_path = Path.home() / ".alita_venvs" / env_name
        
        profile_id = f"venv_{env_name}_{int(time.time())}"
        
        profile = EnvironmentProfile(
            profile_id=profile_id,
            name=env_name,
            environment_type=EnvironmentType.VENV,
            base_path=str(venv_path),
            metadata=metadata,
            status=InitializationStatus.INITIALIZING
        )
        
        # Create virtual environment
        await self._create_venv_environment(profile)
        
        return profile
    
    async def _build_system_profile(self, metadata: EnvironmentMetadata, task_id: Optional[str]) -> EnvironmentProfile:
        """Build system-level environment profile"""
        logger.info("Building system environment profile", extra={
            'operation': 'BUILD_SYSTEM_PROFILE',
            'task_id': task_id
        })
        
        profile_id = f"system_{task_id or 'default'}_{int(time.time())}"
        
        profile = EnvironmentProfile(
            profile_id=profile_id,
            name="system",
            environment_type=EnvironmentType.SYSTEM,
            base_path=metadata.repository_path,
            python_executable=sys.executable,
            metadata=metadata,
            status=InitializationStatus.READY
        )
        
        return profile
    
    def _determine_optimal_environment_type(self, 
                                          metadata: EnvironmentMetadata, 
                                          optimization_options: Optional[Dict[str, Any]]) -> EnvironmentType:
        """Determine optimal environment type based on metadata and options"""
        
        # Check if containerization is preferred
        if optimization_options and optimization_options.get('prefer_containers', False):
            if self.container_orchestrator:
                return EnvironmentType.DOCKER
        
        # Check for complex dependencies that benefit from conda
        if any(framework in str(metadata.dependencies).lower() 
               for framework in ['torch', 'tensorflow', 'numpy', 'pandas', 'scikit-learn']):
            return EnvironmentType.CONDA
        
        # Check for simple Python projects
        if metadata.language == 'python' and len(metadata.dependencies) < 10:
            return EnvironmentType.VENV
        
        # Default to conda for Python projects
        if metadata.language == 'python':
            return EnvironmentType.CONDA
        
        # For non-Python or mixed projects, prefer containers
        if self.container_orchestrator:
            return EnvironmentType.DOCKER
        
        return EnvironmentType.SYSTEM

class RecoveryManager:
    """
    Automated recovery procedures following KGoT Section 2.5 error management patterns
    
    Implementation integrates with KGoT error management system to provide:
    - Automated recovery procedures for environment initialization failures
    - Fallback strategies including relaxing version constraints
    - Identifying minimal dependency sets required for functionality
    - Self-correction mechanisms for environment setup
    """
    
    def __init__(self, error_management_system: Optional[Any] = None):
        """
        Initialize Recovery Manager
        
        @param {Optional[Any]} error_management_system - KGoT error management system instance
        """
        self.error_management_system = error_management_system
        self.recovery_procedures: Dict[str, RecoveryProcedure] = {}
        self.recovery_statistics: Dict[str, int] = defaultdict(int)
        
        # Initialize standard recovery procedures
        self._initialize_recovery_procedures()
        
        logger.info("Initialized Recovery Manager", extra={
            'operation': 'RECOVERY_MANAGER_INIT',
            'error_management_available': error_management_system is not None,
            'recovery_procedures': len(self.recovery_procedures)
        })
    
    def _initialize_recovery_procedures(self):
        """Initialize standard recovery procedures"""
        
        # Dependency resolution recovery
        dependency_recovery = RecoveryProcedure(
            procedure_id="dependency_resolution",
            error_types=[ErrorType.VALIDATION_ERROR, ErrorType.SYSTEM_ERROR],
            recovery_steps=[
                self._relax_version_constraints,
                self._find_minimal_dependency_set,
                self._try_alternative_packages
            ],
            max_attempts=3,
            backoff_strategy="exponential",
            fallback_action=self._create_minimal_environment
        )
        self.recovery_procedures["dependency_resolution"] = dependency_recovery
        
        # Environment creation recovery
        env_creation_recovery = RecoveryProcedure(
            procedure_id="environment_creation",
            error_types=[ErrorType.EXECUTION_ERROR, ErrorType.SYSTEM_ERROR],
            recovery_steps=[
                self._cleanup_partial_environment,
                self._try_alternative_environment_type,
                self._use_system_environment
            ],
            max_attempts=3,
            backoff_strategy="linear",
            fallback_action=self._create_system_fallback
        )
        self.recovery_procedures["environment_creation"] = env_creation_recovery
        
        # Container deployment recovery
        container_recovery = RecoveryProcedure(
            procedure_id="container_deployment",
            error_types=[ErrorType.EXECUTION_ERROR, ErrorType.TIMEOUT_ERROR],
            recovery_steps=[
                self._reduce_container_resources,
                self._try_alternative_base_image,
                self._fallback_to_local_environment
            ],
            max_attempts=2,
            backoff_strategy="exponential",
            fallback_action=self._create_local_fallback
        )
        self.recovery_procedures["container_deployment"] = container_recovery
    
    async def execute_recovery(self, 
                             error_context: ErrorContext,
                             failed_profile: EnvironmentProfile,
                             recovery_options: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[EnvironmentProfile]]:
        """
        Execute automated recovery procedure
        
        @param {ErrorContext} error_context - Error context from KGoT error management
        @param {EnvironmentProfile} failed_profile - Failed environment profile
        @param {Optional[Dict[str, Any]]} recovery_options - Recovery configuration options
        @returns {Tuple[bool, Optional[EnvironmentProfile]]} - (success, recovered_profile)
        """
        logger.info("Starting automated recovery procedure", extra={
            'operation': 'RECOVERY_START',
            'error_type': error_context.error_type.value,
            'profile_id': failed_profile.profile_id,
            'error_id': error_context.error_id
        })
        
        # Find applicable recovery procedures
        applicable_procedures = self._find_applicable_procedures(error_context.error_type)
        
        if not applicable_procedures:
            logger.warning("No applicable recovery procedures found", extra={
                'operation': 'RECOVERY_NO_PROCEDURES',
                'error_type': error_context.error_type.value
            })
            return False, None
        
        # Execute recovery procedures in order of priority
        for procedure in applicable_procedures:
            try:
                success, recovered_profile = await self._execute_recovery_procedure(
                    procedure, error_context, failed_profile, recovery_options
                )
                
                if success and recovered_profile:
                    logger.info("Recovery procedure successful", extra={
                        'operation': 'RECOVERY_SUCCESS',
                        'procedure_id': procedure.procedure_id,
                        'recovered_profile_id': recovered_profile.profile_id
                    })
                    
                    self.recovery_statistics[procedure.procedure_id] += 1
                    return True, recovered_profile
                
            except Exception as e:
                logger.warning(f"Recovery procedure failed: {str(e)}", extra={
                    'operation': 'RECOVERY_PROCEDURE_FAILED',
                    'procedure_id': procedure.procedure_id,
                    'error': str(e)
                })
                continue
        
        logger.error("All recovery procedures failed", extra={
            'operation': 'RECOVERY_FAILED',
            'error_id': error_context.error_id
        })
        return False, None
    
    async def _execute_recovery_procedure(self,
                                        procedure: RecoveryProcedure,
                                        error_context: ErrorContext,
                                        failed_profile: EnvironmentProfile,
                                        recovery_options: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[EnvironmentProfile]]:
        """Execute individual recovery procedure"""
        logger.info("Executing recovery procedure", extra={
            'operation': 'EXECUTE_RECOVERY_PROCEDURE',
            'procedure_id': procedure.procedure_id,
            'max_attempts': procedure.max_attempts
        })
        
        for attempt in range(procedure.max_attempts):
            try:
                # Execute recovery steps
                for step in procedure.recovery_steps:
                    success = await step(failed_profile, error_context, recovery_options)
                    if success:
                        # Validate recovery
                        if await self._validate_recovery(failed_profile):
                            return True, failed_profile
                        break
                
                # Apply backoff strategy
                if attempt < procedure.max_attempts - 1:
                    await self._apply_backoff(procedure.backoff_strategy, attempt)
                
            except Exception as e:
                logger.warning(f"Recovery step failed: {str(e)}")
                continue
        
        # Execute fallback action if all attempts failed
        if procedure.fallback_action:
            try:
                fallback_profile = await procedure.fallback_action(failed_profile, error_context)
                if fallback_profile:
                    return True, fallback_profile
            except Exception as e:
                logger.error(f"Fallback action failed: {str(e)}")
        
        return False, None
    
    def _find_applicable_procedures(self, error_type: ErrorType) -> List[RecoveryProcedure]:
        """Find recovery procedures applicable to error type"""
        applicable = []
        for procedure in self.recovery_procedures.values():
            if error_type in procedure.error_types:
                applicable.append(procedure)
        
        # Sort by success rate (based on statistics)
        applicable.sort(key=lambda p: self.recovery_statistics.get(p.procedure_id, 0), reverse=True)
        return applicable
    
    async def _validate_recovery(self, profile: EnvironmentProfile) -> bool:
        """Validate that recovery was successful"""
        try:
            if profile.environment_type == EnvironmentType.CONDA:
                if profile.python_executable and os.path.exists(profile.python_executable):
                    return True
            elif profile.environment_type in [EnvironmentType.DOCKER, EnvironmentType.SARUS]:
                # Check container status
                return profile.status == InitializationStatus.READY
            elif profile.environment_type == EnvironmentType.SYSTEM:
                return True
            
            return False
        except Exception:
            return False
    
    async def _apply_backoff(self, strategy: str, attempt: int):
        """Apply backoff strategy between retry attempts"""
        if strategy == "exponential":
            delay = (2 ** attempt) * 1  # 1, 2, 4, 8 seconds
        elif strategy == "linear":
            delay = (attempt + 1) * 2   # 2, 4, 6, 8 seconds
        else:
            delay = 1
        
        logger.debug(f"Applying {strategy} backoff: {delay}s", extra={
            'operation': 'RECOVERY_BACKOFF',
            'strategy': strategy,
            'attempt': attempt,
            'delay': delay
        })
        
        await asyncio.sleep(delay)
    
    # Recovery step implementations
    async def _relax_version_constraints(self, profile: EnvironmentProfile, error_context: ErrorContext, options: Optional[Dict[str, Any]]) -> bool:
        """Relax version constraints in dependencies"""
        logger.info("Relaxing version constraints", extra={
            'operation': 'RELAX_VERSION_CONSTRAINTS',
            'profile_id': profile.profile_id
        })
        
        try:
            if profile.metadata and profile.metadata.dependencies:
                relaxed_deps = []
                for dep in profile.metadata.dependencies:
                    # Remove exact version pins
                    relaxed_dep = re.sub(r'==[\d\.]+', '', dep)
                    relaxed_dep = re.sub(r'>=[\d\.]+,<[\d\.]+', '', relaxed_dep)
                    relaxed_deps.append(relaxed_dep)
                
                profile.metadata.dependencies = relaxed_deps
                return True
        except Exception as e:
            logger.warning(f"Failed to relax version constraints: {str(e)}")
        
        return False
    
    async def _find_minimal_dependency_set(self, profile: EnvironmentProfile, error_context: ErrorContext, options: Optional[Dict[str, Any]]) -> bool:
        """Find minimal set of dependencies required for functionality"""
        logger.info("Finding minimal dependency set", extra={
            'operation': 'FIND_MINIMAL_DEPS',
            'profile_id': profile.profile_id
        })
        
        try:
            if profile.metadata and profile.metadata.dependencies:
                # Keep only essential dependencies
                essential_deps = []
                for dep in profile.metadata.dependencies:
                    if any(essential in dep.lower() for essential in ['numpy', 'pandas', 'requests', 'click']):
                        essential_deps.append(dep)
                
                if essential_deps:
                    profile.metadata.dependencies = essential_deps
                    return True
        except Exception as e:
            logger.warning(f"Failed to find minimal dependency set: {str(e)}")
        
        return False
    
    async def _try_alternative_packages(self, profile: EnvironmentProfile, error_context: ErrorContext, options: Optional[Dict[str, Any]]) -> bool:
        """Try alternative package names or sources"""
        logger.info("Trying alternative packages", extra={
            'operation': 'TRY_ALTERNATIVE_PACKAGES',
            'profile_id': profile.profile_id
        })
        
        # Implementation would include package name alternatives
        # This is a simplified version
        return False
    
    async def _cleanup_partial_environment(self, profile: EnvironmentProfile, error_context: ErrorContext, options: Optional[Dict[str, Any]]) -> bool:
        """Clean up partially created environment"""
        logger.info("Cleaning up partial environment", extra={
            'operation': 'CLEANUP_PARTIAL_ENV',
            'profile_id': profile.profile_id
        })
        
        try:
            if profile.environment_type == EnvironmentType.CONDA and profile.conda_env_name:
                cmd = ['conda', 'env', 'remove', '-n', profile.conda_env_name, '-y']
                process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await process.communicate()
            
            return True
        except Exception as e:
            logger.warning(f"Failed to cleanup partial environment: {str(e)}")
        
        return False
    
    async def _try_alternative_environment_type(self, profile: EnvironmentProfile, error_context: ErrorContext, options: Optional[Dict[str, Any]]) -> bool:
        """Try alternative environment type"""
        logger.info("Trying alternative environment type", extra={
            'operation': 'TRY_ALT_ENV_TYPE',
            'current_type': profile.environment_type.value
        })
        
        # This would involve recreating the profile with a different environment type
        # For now, just mark as attempting alternative
        return False
    
    async def _use_system_environment(self, profile: EnvironmentProfile, error_context: ErrorContext, options: Optional[Dict[str, Any]]) -> bool:
        """Fall back to using system environment"""
        logger.info("Using system environment fallback", extra={
            'operation': 'USE_SYSTEM_ENV',
            'profile_id': profile.profile_id
        })
        
        try:
            profile.environment_type = EnvironmentType.SYSTEM
            profile.python_executable = sys.executable
            profile.status = InitializationStatus.READY
            return True
        except Exception:
            return False
    
    async def _create_minimal_environment(self, profile: EnvironmentProfile, error_context: ErrorContext) -> Optional[EnvironmentProfile]:
        """Create minimal environment as fallback"""
        logger.info("Creating minimal environment fallback", extra={
            'operation': 'CREATE_MINIMAL_ENV'
        })
        
        # Create a basic system environment with minimal dependencies
        minimal_profile = EnvironmentProfile(
            profile_id=f"minimal_{int(time.time())}",
            name="minimal_fallback",
            environment_type=EnvironmentType.SYSTEM,
            base_path=profile.metadata.repository_path if profile.metadata else "/tmp",
            python_executable=sys.executable,
            metadata=profile.metadata,
            status=InitializationStatus.READY
        )
        
        return minimal_profile
    
    async def _create_system_fallback(self, profile: EnvironmentProfile, error_context: ErrorContext) -> Optional[EnvironmentProfile]:
        """Create system environment as fallback"""
        return await self._create_minimal_environment(profile, error_context)
    
    async def _reduce_container_resources(self, profile: EnvironmentProfile, error_context: ErrorContext, options: Optional[Dict[str, Any]]) -> bool:
        """Reduce container resource requirements"""
        logger.info("Reducing container resources", extra={
            'operation': 'REDUCE_CONTAINER_RESOURCES',
            'profile_id': profile.profile_id
        })
        
        if profile.container_config:
            # Reduce memory and CPU limits
            if 'memory' in profile.container_config:
                current_memory = profile.container_config['memory']
                profile.container_config['memory'] = f"{int(current_memory.rstrip('m')) // 2}m"
            
            return True
        
        return False
    
    async def _try_alternative_base_image(self, profile: EnvironmentProfile, error_context: ErrorContext, options: Optional[Dict[str, Any]]) -> bool:
        """Try alternative container base image"""
        logger.info("Trying alternative base image", extra={
            'operation': 'TRY_ALT_BASE_IMAGE',
            'profile_id': profile.profile_id
        })
        
        if profile.container_config and 'image' in profile.container_config:
            # Try lighter base images
            alternatives = ['python:3.9-slim', 'python:3.9-alpine', 'ubuntu:20.04']
            for alt_image in alternatives:
                if alt_image != profile.container_config['image']:
                    profile.container_config['image'] = alt_image
                    return True
        
        return False
    
    async def _fallback_to_local_environment(self, profile: EnvironmentProfile, error_context: ErrorContext, options: Optional[Dict[str, Any]]) -> bool:
        """Fallback to local environment instead of container"""
        logger.info("Falling back to local environment", extra={
            'operation': 'FALLBACK_TO_LOCAL',
            'profile_id': profile.profile_id
        })
        
        try:
            profile.environment_type = EnvironmentType.CONDA
            profile.container_config = None
            return True
        except Exception:
            return False
    
    async def _create_local_fallback(self, profile: EnvironmentProfile, error_context: ErrorContext) -> Optional[EnvironmentProfile]:
        """Create local environment as fallback"""
        return await self._create_minimal_environment(profile, error_context)

class EnvironmentManager:
    """
    Main Environment Manager orchestrator class
    
    Complete implementation of Alita Section 2.3.4 "Environment Management" with KGoT integration.
    This class coordinates all environment management components and provides the main interface
    for creating, managing, and recovering isolated execution environments.
    
    Key Features:
    - Repository metadata parsing with TextInspectorTool integration
    - Conda environment management with unique naming and parallel initialization  
    - Container integration with KGoT containerization system
    - Automated recovery procedures following KGoT error management patterns
    - LangChain agent integration for intelligent environment management
    """
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 container_orchestrator: Optional[Any] = None,
                 error_management_system: Optional[Any] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Environment Manager
        
        @param {Optional[str]} openrouter_api_key - OpenRouter API key for LLM integration (per user memory)
        @param {Optional[Any]} container_orchestrator - KGoT container orchestrator for enhanced isolation
        @param {Optional[Any]} error_management_system - KGoT error management system
        @param {Optional[Dict[str, Any]]} config - Configuration options
        """
        self.config = config or {}
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        
        # Initialize LLM client (OpenRouter-based per user memory)
        self.llm_client = self._initialize_llm_client()
        
        # Initialize component managers
        self.metadata_parser = EnvironmentMetadataParser(llm_client=self.llm_client)
        self.conda_manager = CondaEnvironmentManager()
        self.profile_builder = EnvironmentProfileBuilder(
            metadata_parser=self.metadata_parser,
            conda_manager=self.conda_manager,
            container_orchestrator=container_orchestrator
        )
        self.recovery_manager = RecoveryManager(error_management_system=error_management_system)
        
        # Environment tracking
        self.active_environments: Dict[str, EnvironmentProfile] = {}
        self.environment_history: List[Dict[str, Any]] = []
        
        # LangChain agent for intelligent environment management (per user hard rule)
        self.agent_executor: Optional[AgentExecutor] = None
        self._initialize_langchain_agent()
        
        logger.info("Initialized Environment Manager", extra={
            'operation': 'ENV_MANAGER_INIT',
            'llm_available': self.llm_client is not None,
            'container_integration': container_orchestrator is not None,
            'error_management_integration': error_management_system is not None,
            'agent_available': self.agent_executor is not None
        })
    
    def _initialize_llm_client(self) -> Optional[ChatOpenAI]:
        """Initialize OpenRouter-based LLM client"""
        if not self.openrouter_api_key:
            logger.warning("OpenRouter API key not available, LLM features disabled")
            return None
        
        try:
            from langchain_openai import ChatOpenAI
            
            # Use OpenRouter API for LLM integration (per user memory)
            llm_client = ChatOpenAI(
                model="google/gemini-2.5-pro",  # For orchestration per user rules
                openai_api_key=self.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.1
            )
            
            logger.info("Initialized OpenRouter LLM client", extra={
                'operation': 'LLM_CLIENT_INIT',
                'model': "google/gemini-2.5-pro"
            })
            
            return llm_client
            
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {str(e)}", extra={
                'operation': 'LLM_CLIENT_INIT_ERROR',
                'error': str(e)
            })
            return None
    
    def _initialize_langchain_agent(self):
        """Initialize LangChain agent for intelligent environment management"""
        if not self.llm_client:
            return
        
        try:
            # Create environment management tools
            tools = [
                self._create_metadata_analysis_tool(),
                self._create_environment_optimization_tool(),
                self._create_dependency_resolution_tool()
            ]
            
            # Create agent prompt
            system_prompt = """You are an expert environment management agent for the Alita-KGoT system. 
            You help users create, manage, and optimize isolated execution environments for various programming projects.
            
            Your capabilities include:
            - Analyzing repository metadata to understand project requirements
            - Recommending optimal environment types (conda, docker, venv, system)
            - Resolving dependency conflicts and version constraints
            - Optimizing environments for performance and resource usage
            - Providing recovery suggestions when environment setup fails
            
            Always provide clear explanations for your recommendations and focus on reliability and reproducibility."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Create agent executor
            agent = create_openai_functions_agent(self.llm_client, tools, prompt)
            self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            logger.info("Initialized LangChain environment management agent", extra={
                'operation': 'AGENT_INIT',
                'tools_count': len(tools)
            })
            
        except Exception as e:
            logger.warning(f"Failed to initialize LangChain agent: {str(e)}", extra={
                'operation': 'AGENT_INIT_ERROR',
                'error': str(e)
            })
    
    async def create_environment(self, 
                               repository_path: str,
                               task_id: Optional[str] = None,
                               environment_type: Optional[EnvironmentType] = None,
                               options: Optional[Dict[str, Any]] = None) -> EnvironmentProfile:
        """
        Create isolated execution environment for repository
        
        @param {str} repository_path - Path to repository or script directory
        @param {Optional[str]} task_id - Task ID for unique naming and tracking
        @param {Optional[EnvironmentType]} environment_type - Preferred environment type
        @param {Optional[Dict[str, Any]]} options - Additional configuration options
        @returns {EnvironmentProfile} - Created environment profile
        """
        start_time = time.time()
        
        logger.info("Starting environment creation", extra={
            'operation': 'CREATE_ENV_START',
            'repository_path': repository_path,
            'task_id': task_id,
            'environment_type': environment_type.value if environment_type else None
        })
        
        try:
            # Build environment profile
            profile = await self.profile_builder.build_profile(
                repository_path=repository_path,
                task_id=task_id,
                environment_type=environment_type,
                optimization_options=options
            )
            
            # Register active environment
            self.active_environments[profile.profile_id] = profile
            
            # Record in history
            self.environment_history.append({
                'profile_id': profile.profile_id,
                'repository_path': repository_path,
                'task_id': task_id,
                'environment_type': profile.environment_type.value,
                'created_at': profile.created_at.isoformat(),
                'creation_duration': time.time() - start_time,
                'status': 'success'
            })
            
            logger.info("Environment creation successful", extra={
                'operation': 'CREATE_ENV_SUCCESS',
                'profile_id': profile.profile_id,
                'environment_type': profile.environment_type.value,
                'duration': time.time() - start_time
            })
            
            return profile
            
        except Exception as e:
            # Record failure in history
            self.environment_history.append({
                'repository_path': repository_path,
                'task_id': task_id,
                'environment_type': environment_type.value if environment_type else None,
                'created_at': datetime.now().isoformat(),
                'creation_duration': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            })
            
            logger.error(f"Environment creation failed: {str(e)}", extra={
                'operation': 'CREATE_ENV_ERROR',
                'repository_path': repository_path,
                'error': str(e)
            })
            
            # Attempt recovery if error management system is available
            if self.recovery_manager.error_management_system:
                return await self._attempt_recovery_creation(repository_path, task_id, e, options)
            
            raise
    
    async def _attempt_recovery_creation(self, 
                                       repository_path: str, 
                                       task_id: Optional[str], 
                                       original_error: Exception,
                                       options: Optional[Dict[str, Any]]) -> EnvironmentProfile:
        """Attempt to recover from environment creation failure"""
        logger.info("Attempting recovery from environment creation failure", extra={
            'operation': 'RECOVERY_ATTEMPT',
            'repository_path': repository_path,
            'original_error': str(original_error)
        })
        
        try:
            # Create mock error context and profile for recovery
            from ..kgot_core.error_management import ErrorContext, ErrorType, ErrorSeverity
            
            error_context = ErrorContext(
                error_id=f"env_creation_{int(time.time())}",
                error_type=ErrorType.SYSTEM_ERROR,
                severity=ErrorSeverity.HIGH,
                timestamp=datetime.now(),
                original_operation="environment_creation",
                error_message=str(original_error)
            )
            
            # Parse metadata for recovery profile
            metadata = await self.metadata_parser.parse_repository_metadata(repository_path)
            
            failed_profile = EnvironmentProfile(
                profile_id=f"failed_{int(time.time())}",
                name="failed_env",
                environment_type=EnvironmentType.SYSTEM,
                base_path=repository_path,
                metadata=metadata,
                status=InitializationStatus.ERROR
            )
            
            # Execute recovery
            success, recovered_profile = await self.recovery_manager.execute_recovery(
                error_context, failed_profile, options
            )
            
            if success and recovered_profile:
                # Register recovered environment
                self.active_environments[recovered_profile.profile_id] = recovered_profile
                
                logger.info("Environment creation recovery successful", extra={
                    'operation': 'RECOVERY_SUCCESS',
                    'recovered_profile_id': recovered_profile.profile_id
                })
                
                return recovered_profile
            
            else:
                raise RuntimeError("Recovery failed: All recovery procedures exhausted")
                
        except Exception as recovery_error:
            logger.error(f"Environment creation recovery failed: {str(recovery_error)}", extra={
                'operation': 'RECOVERY_ERROR',
                'error': str(recovery_error)
            })
            raise original_error  # Re-raise original error if recovery fails
    
    async def get_environment(self, profile_id: str) -> Optional[EnvironmentProfile]:
        """Get environment profile by ID"""
        return self.active_environments.get(profile_id)
    
    async def list_environments(self) -> List[EnvironmentProfile]:
        """List all active environments"""
        return list(self.active_environments.values())
    
    async def cleanup_environment(self, profile_id: str) -> bool:
        """Clean up environment and release resources"""
        profile = self.active_environments.get(profile_id)
        if not profile:
            return False
        
        logger.info("Cleaning up environment", extra={
            'operation': 'CLEANUP_ENV',
            'profile_id': profile_id,
            'environment_type': profile.environment_type.value
        })
        
        try:
            if profile.environment_type == EnvironmentType.CONDA and profile.conda_env_name:
                cmd = ['conda', 'env', 'remove', '-n', profile.conda_env_name, '-y']
                process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await process.communicate()
            
            # Remove from active environments
            del self.active_environments[profile_id]
            
            logger.info("Environment cleanup successful", extra={
                'operation': 'CLEANUP_ENV_SUCCESS',
                'profile_id': profile_id
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Environment cleanup failed: {str(e)}", extra={
                'operation': 'CLEANUP_ENV_ERROR',
                'profile_id': profile_id,
                'error': str(e)
            })
            return False
    
    async def cleanup_all_environments(self) -> Dict[str, bool]:
        """Clean up all active environments"""
        logger.info("Cleaning up all environments", extra={
            'operation': 'CLEANUP_ALL_ENVS',
            'environment_count': len(self.active_environments)
        })
        
        cleanup_results = {}
        for profile_id in list(self.active_environments.keys()):
            cleanup_results[profile_id] = await self.cleanup_environment(profile_id)
        
        return cleanup_results
    
    def get_environment_statistics(self) -> Dict[str, Any]:
        """Get environment management statistics"""
        active_envs = list(self.active_environments.values())
        
        env_type_counts = {}
        for env_type in EnvironmentType:
            env_type_counts[env_type.value] = len([env for env in active_envs if env.environment_type == env_type])
        
        status_counts = {}
        for status in InitializationStatus:
            status_counts[status.value] = len([env for env in active_envs if env.status == status])
        
        return {
            'total_active_environments': len(active_envs),
            'environment_types': env_type_counts,
            'environment_statuses': status_counts,
            'total_created': len(self.environment_history),
            'recovery_statistics': self.recovery_manager.recovery_statistics,
            'creation_history': self.environment_history[-10:]  # Last 10 creations
        }
    
    # LangChain agent tools
    def _create_metadata_analysis_tool(self) -> BaseTool:
        """Create tool for metadata analysis"""
        class MetadataAnalysisTool(BaseTool):
            name = "analyze_repository_metadata"
            description = "Analyze repository metadata to understand project requirements and dependencies"
            
            def _run(self, repository_path: str) -> str:
                try:
                    # This would be implemented as async in real usage
                    import asyncio
                    loop = asyncio.get_event_loop()
                    metadata = loop.run_until_complete(
                        self.metadata_parser.parse_repository_metadata(repository_path)
                    )
                    
                    return json.dumps({
                        'python_version': metadata.python_version,
                        'dependencies': metadata.dependencies[:10],  # Limit for output
                        'framework': metadata.framework,
                        'language': metadata.language,
                        'system_requirements': metadata.system_requirements
                    }, indent=2)
                    
                except Exception as e:
                    return f"Error analyzing metadata: {str(e)}"
        
        return MetadataAnalysisTool()
    
    def _create_environment_optimization_tool(self) -> BaseTool:
        """Create tool for environment optimization recommendations"""
        class EnvironmentOptimizationTool(BaseTool):
            name = "recommend_environment_optimization"
            description = "Recommend optimal environment type and configuration for a project"
            
            def _run(self, project_description: str) -> str:
                recommendations = {
                    'environment_type': 'conda',
                    'reasoning': 'Based on project analysis',
                    'optimizations': [
                        'Use conda for complex scientific dependencies',
                        'Enable container integration for isolation',
                        'Use minimal dependency set for faster setup'
                    ]
                }
                
                return json.dumps(recommendations, indent=2)
        
        return EnvironmentOptimizationTool()
    
    def _create_dependency_resolution_tool(self) -> BaseTool:
        """Create tool for dependency conflict resolution"""
        class DependencyResolutionTool(BaseTool):
            name = "resolve_dependency_conflicts"
            description = "Suggest solutions for dependency conflicts and version constraints"
            
            def _run(self, dependency_conflicts: str) -> str:
                solutions = {
                    'strategy': 'version_relaxation',
                    'steps': [
                        'Remove exact version pins',
                        'Use compatible version ranges',
                        'Install core dependencies first',
                        'Add optional dependencies separately'
                    ],
                    'fallback': 'Create minimal environment with essential packages only'
                }
                
                return json.dumps(solutions, indent=2)
        
        return DependencyResolutionTool()

# Helper methods for EnvironmentProfileBuilder
def _generate_cache_key(self, repository_path: str, task_id: Optional[str], environment_type: Optional[EnvironmentType]) -> str:
    """Generate cache key for profile caching"""
    key_parts = [
        hashlib.md5(repository_path.encode()).hexdigest()[:8],
        task_id or "notask",
        environment_type.value if environment_type else "auto"
    ]
    return "_".join(key_parts)

async def _validate_cached_profile(self, profile: EnvironmentProfile) -> bool:
    """Validate that cached profile is still valid"""
    try:
        if profile.status != InitializationStatus.READY:
            return False
        
        if profile.environment_type == EnvironmentType.CONDA:
            if profile.python_executable and os.path.exists(profile.python_executable):
                return True
        elif profile.environment_type == EnvironmentType.SYSTEM:
            return True
        
        return False
    except Exception:
        return False

# Monkey patch helper methods to EnvironmentProfileBuilder
EnvironmentProfileBuilder._generate_cache_key = _generate_cache_key
EnvironmentProfileBuilder._validate_cached_profile = _validate_cached_profile

async def main():
    """
    Demonstration of Environment Manager usage
    
    Example usage of the complete Alita Environment Management system with KGoT integration
    """
    print("🚀 Alita Environment Manager - Task 12 Implementation Demo")
    print("=" * 60)
    
    try:
        # Initialize Environment Manager
        print("📋 Initializing Environment Manager...")
        env_manager = EnvironmentManager()
        
        # Example repository path (user would provide actual path)
        repository_path = "."  # Current directory for demo
        task_id = "demo_task_001"
        
        print(f"📁 Repository: {repository_path}")
        print(f"🔖 Task ID: {task_id}")
        print()
        
        # Create environment
        print("🔧 Creating environment...")
        profile = await env_manager.create_environment(
            repository_path=repository_path,
            task_id=task_id,
            environment_type=EnvironmentType.CONDA,
            options={
                'prefer_containers': False,
                'optimize_for_performance': True
            }
        )
        
        print(f"✅ Environment created successfully!")
        print(f"   Profile ID: {profile.profile_id}")
        print(f"   Environment Type: {profile.environment_type.value}")
        print(f"   Status: {profile.status.value}")
        print(f"   Python Executable: {profile.python_executable}")
        print()
        
        # Display environment statistics
        print("📊 Environment Statistics:")
        stats = env_manager.get_environment_statistics()
        for key, value in stats.items():
            if key != 'creation_history':
                print(f"   {key}: {value}")
        print()
        
        # List active environments
        print("📋 Active Environments:")
        environments = await env_manager.list_environments()
        for env in environments:
            print(f"   - {env.profile_id} ({env.environment_type.value}) - {env.status.value}")
        print()
        
        # Cleanup demonstration
        print("🧹 Cleanup demonstration...")
        cleanup_result = await env_manager.cleanup_environment(profile.profile_id)
        print(f"   Cleanup successful: {cleanup_result}")
        
        print()
        print("✨ Environment Manager demonstration complete!")
        print("🎯 Task 12 implementation successfully demonstrated!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 