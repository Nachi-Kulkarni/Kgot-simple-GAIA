#!/usr/bin/env python3
"""
Alita Code Running Tool with KGoT Execution Environment

Implementation of Alita Section 2.3.3 "CodeRunningTool" complete functionality
integrated with KGoT Section 2.6 "Python Executor tool containerization" and
Section 2.5 "Error Management" for comprehensive script validation and execution.

This module provides:
- Script validation through isolated environment execution (Alita 2.3.3)
- Iterative refinement with error inspection and code regeneration
- Output caching for potential MCP server generation
- Integration with KGoT containerization for secure execution (KGoT 2.6)
- Robust error handling using KGoT error management (KGoT 2.5)
- Cross-validation framework for comprehensive validation testing
- Automatic tool registration as reusable MCPs upon successful execution

Key Components:
1. CodeRunningTool: Main orchestrator for script validation and execution
2. ExecutionEnvironment: Secure containerized execution wrapper
3. IterativeRefinementEngine: LLM-based error correction and improvement
4. MCPServerGenerator: Tool registration and MCP generation
5. ValidationFramework: Cross-validation testing with majority voting
6. ResultCache: Performance optimization through intelligent caching

@module CodeRunner
@author Enhanced Alita KGoT Team
@version 1.0.0
@based_on Alita Section 2.3.3, KGoT Sections 2.5, 2.6
"""

import os
import sys
import json
import time
import uuid
import hashlib
import asyncio
import logging
import tempfile
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter

# LangChain imports for agent development (per user memory)
from langchain.agents import Agent, AgentExecutor
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Import existing KGoT infrastructure components
import docker
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import requests

# Import existing KGoT modules for integration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kgot_core'))
from error_management import (
    KGoTErrorManagementSystem, 
    ErrorType, 
    ErrorSeverity, 
    ErrorContext,
    create_kgot_error_management_system
)
from containerization import (
    ContainerOrchestrator, 
    ContainerConfig, 
    DeploymentEnvironment,
    EnvironmentDetector,
    ResourceManager
)

# Note: Using Claude 4 Sonnet for all code running tasks (per user requirements)

# Winston-compatible logging setup for CodeRunningTool
logger = logging.getLogger('AlitaCodeRunner')
handler = logging.FileHandler('./logs/alita/combined.log')
formatter = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# HTTP logging for API interactions
http_logger = logging.getLogger('AlitaCodeRunnerHTTP')
http_handler = logging.FileHandler('./logs/alita/http.log')
http_handler.setFormatter(formatter)
http_logger.addHandler(http_handler)
http_logger.setLevel(logging.INFO)


class ExecutionResult(Enum):
    """
    Enumeration of execution result statuses for code validation
    """
    SUCCESS = "success"                     # Code executed successfully with expected results
    FAILURE = "failure"                     # Code execution failed with errors
    TIMEOUT = "timeout"                     # Code execution timed out
    SYNTAX_ERROR = "syntax_error"           # Syntax errors detected in code
    RUNTIME_ERROR = "runtime_error"         # Runtime errors during execution
    VALIDATION_FAILED = "validation_failed" # Output validation failed
    SECURITY_VIOLATION = "security_violation" # Security policy violation detected


class RefinementStrategy(Enum):
    """
    Strategies for iterative code refinement and improvement
    """
    SYNTAX_CORRECTION = "syntax_correction"     # Focus on syntax error fixes
    LOGIC_IMPROVEMENT = "logic_improvement"     # Improve code logic and algorithm
    PERFORMANCE_OPTIMIZATION = "performance_optimization" # Optimize for performance
    ERROR_HANDLING = "error_handling"          # Add robust error handling
    SECURITY_HARDENING = "security_hardening"  # Enhance security measures
    COMPREHENSIVE = "comprehensive"            # All-around improvement


@dataclass
class ExecutionContext:
    """
    Comprehensive context information for code execution and validation
    
    Attributes:
        execution_id: Unique identifier for execution session
        code: Source code to execute
        language: Programming language (default: python)
        expected_output: Expected execution output for validation
        timeout: Maximum execution time in seconds
        resource_limits: CPU and memory constraints
        dependencies: Required packages/modules
        environment_vars: Environment variables for execution
        validation_criteria: Custom validation rules
        metadata: Additional execution metadata
    """
    execution_id: str
    code: str
    language: str = "python"
    expected_output: Optional[str] = None
    timeout: int = 30
    resource_limits: Dict[str, str] = field(default_factory=lambda: {"memory": "256m", "cpu": "0.5"})
    dependencies: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    validation_criteria: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution context to dictionary for logging and caching"""
        return {
            'execution_id': self.execution_id,
            'code_hash': hashlib.sha256(self.code.encode()).hexdigest(),
            'language': self.language,
            'timeout': self.timeout,
            'resource_limits': self.resource_limits,
            'dependencies': self.dependencies,
            'environment_vars': self.environment_vars,
            'validation_criteria': self.validation_criteria,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ValidationResult:
    """
    Comprehensive validation result with detailed analysis
    
    Attributes:
        success: Whether validation passed overall
        execution_result: Result of code execution
        output: Actual execution output
        expected_output: Expected output for comparison
        execution_time: Time taken for execution
        resource_usage: Resource consumption during execution
        errors: List of errors encountered
        warnings: List of warnings generated
        validation_score: Numerical validation score (0-1)
        confidence_level: Confidence in validation result (0-1)
        recommendations: Improvement recommendations
    """
    success: bool
    execution_result: ExecutionResult
    output: str
    expected_output: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_score: float = 0.0
    confidence_level: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary for logging and analysis"""
        return {
            'success': self.success,
            'execution_result': self.execution_result.value,
            'output_length': len(self.output) if self.output else 0,
            'execution_time': self.execution_time,
            'resource_usage': self.resource_usage,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'validation_score': self.validation_score,
            'confidence_level': self.confidence_level,
            'recommendations_count': len(self.recommendations),
            'metadata': self.metadata
        }


class ExecutionEnvironment:
    """
    Secure execution environment wrapper using KGoT containerization infrastructure
    
    Provides isolated, secure execution of code with comprehensive resource management,
    timeout handling, and integration with existing KGoT containerization system.
    
    Features:
    - Docker/Sarus containerized execution for security isolation
    - Resource limits and timeout enforcement
    - Integration with KGoT containerization infrastructure
    - Comprehensive logging and monitoring
    - Support for multiple programming languages
    """
    
    def __init__(self, 
                 container_orchestrator: Optional[ContainerOrchestrator] = None,
                 default_image: str = "python:3.9-slim",
                 max_concurrent_executions: int = 5):
        """
        Initialize execution environment with containerization support
        
        @param {ContainerOrchestrator} container_orchestrator - KGoT container orchestrator instance
        @param {str} default_image - Default Docker image for code execution
        @param {int} max_concurrent_executions - Maximum concurrent executions allowed
        """
        self.container_orchestrator = container_orchestrator or self._create_default_orchestrator()
        self.default_image = default_image
        self.max_concurrent_executions = max_concurrent_executions
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_executions)
        
        logger.info("Initialized ExecutionEnvironment", extra={
            'operation': 'EXECUTION_ENVIRONMENT_INIT',
            'default_image': default_image,
            'max_concurrent': max_concurrent_executions
        })
    
    def _create_default_orchestrator(self) -> ContainerOrchestrator:
        """
        Create default container orchestrator using KGoT infrastructure
        
        @returns {ContainerOrchestrator} Configured container orchestrator
        """
        try:
            orchestrator = ContainerOrchestrator()
            # Note: Initialization will be handled later when needed
            return orchestrator
        except Exception as e:
            logger.error(f"Failed to create container orchestrator: {e}")
            # Return None to allow graceful fallback
            return None
    
    async def execute_code_securely(self, 
                                   execution_context: ExecutionContext) -> ValidationResult:
        """
        Execute code in secure containerized environment with comprehensive validation
        
        @param {ExecutionContext} execution_context - Complete execution context and requirements
        @returns {ValidationResult} Detailed validation result with execution analysis
        """
        execution_id = execution_context.execution_id
        
        logger.info("Starting secure code execution", extra={
            'operation': 'SECURE_CODE_EXECUTION_START',
            'execution_id': execution_id,
            'language': execution_context.language,
            'timeout': execution_context.timeout
        })
        
        try:
            # Prepare execution environment container
            container_config = self._prepare_container_config(execution_context)
            
            # Execute code with timeout and resource monitoring
            start_time = time.time()
            execution_result = await self._execute_in_container(container_config, execution_context)
            execution_time = time.time() - start_time
            
            # Validate execution results
            validation_result = self._validate_execution_result(
                execution_result, 
                execution_context, 
                execution_time
            )
            
            # Log execution completion
            logger.info("Code execution completed", extra={
                'operation': 'SECURE_CODE_EXECUTION_COMPLETE',
                'execution_id': execution_id,
                'success': validation_result.success,
                'execution_time': execution_time,
                'result_type': validation_result.execution_result.value
            })
            
            return validation_result
            
        except Exception as e:
            logger.error("Code execution failed", extra={
                'operation': 'SECURE_CODE_EXECUTION_ERROR',
                'execution_id': execution_id,
                'error': str(e)
            })
            
            return ValidationResult(
                success=False,
                execution_result=ExecutionResult.FAILURE,
                output="",
                errors=[str(e)],
                execution_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )
    
    def _prepare_container_config(self, execution_context: ExecutionContext) -> ContainerConfig:
        """
        Prepare container configuration for secure code execution
        
        @param {ExecutionContext} execution_context - Execution requirements and constraints
        @returns {ContainerConfig} Configured container specification
        """
        container_name = f"code_runner_{execution_context.execution_id}"
        
        # Build environment variables with security considerations
        safe_env_vars = {
            "PYTHONPATH": "/app",
            "PYTHONUNBUFFERED": "1",
            **execution_context.environment_vars
        }
        
        # Create container configuration using KGoT infrastructure
        container_config = ContainerConfig(
            name=container_name,
            image=self.default_image,
            environment=safe_env_vars,
            resource_limits=execution_context.resource_limits,
            networks=["code_runner_network"],
            restart_policy="no"  # Single-use containers for security
        )
        
        logger.debug("Prepared container configuration", extra={
            'operation': 'CONTAINER_CONFIG_PREPARE',
            'container_name': container_name,
            'image': self.default_image,
            'resource_limits': execution_context.resource_limits
        })
        
        return container_config 

    async def _execute_in_container(self, 
                                   container_config: ContainerConfig, 
                                   execution_context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute code within containerized environment with monitoring
        
        @param {ContainerConfig} container_config - Container configuration for execution
        @param {ExecutionContext} execution_context - Execution context and code
        @returns {Dict[str, Any]} Raw execution results with output and metrics
        """
        execution_id = execution_context.execution_id
        
        try:
            # Start container using KGoT orchestrator
            container_started = await self.container_orchestrator.deploy_service_stack()
            if not container_started:
                raise RuntimeError("Failed to start execution container")
            
            # Prepare code execution payload
            execution_payload = {
                'required_modules': execution_context.dependencies,
                'code': execution_context.code
            }
            
            # Execute code via container API (using existing Python executor pattern)
            container_url = f"http://localhost:16000/run"
            
            response = requests.post(
                container_url,
                json=execution_payload,
                timeout=execution_context.timeout
            )
            
            # Process execution results
            if response.status_code == 200:
                result_data = response.json()
                return {
                    'success': True,
                    'output': result_data.get('output', ''),
                    'status_code': response.status_code,
                    'execution_time': response.elapsed.total_seconds()
                }
            else:
                error_data = response.json() if response.content else {}
                return {
                    'success': False,
                    'output': '',
                    'error': error_data.get('error', 'Unknown execution error'),
                    'status_code': response.status_code
                }
                
        except requests.exceptions.Timeout:
            logger.warning("Code execution timed out", extra={
                'operation': 'CODE_EXECUTION_TIMEOUT',
                'execution_id': execution_id,
                'timeout': execution_context.timeout
            })
            return {
                'success': False,
                'output': '',
                'error': 'Code execution timed out',
                'timeout': True
            }
        except Exception as e:
            logger.error("Container execution failed", extra={
                'operation': 'CONTAINER_EXECUTION_ERROR',
                'execution_id': execution_id,
                'error': str(e)
            })
            return {
                'success': False,
                'output': '',
                'error': str(e)
            }
    
    def _validate_execution_result(self, 
                                  execution_result: Dict[str, Any], 
                                  execution_context: ExecutionContext,
                                  execution_time: float) -> ValidationResult:
        """
        Validate execution results against expected criteria
        
        @param {Dict[str, Any]} execution_result - Raw execution results from container
        @param {ExecutionContext} execution_context - Original execution context
        @param {float} execution_time - Time taken for execution
        @returns {ValidationResult} Comprehensive validation analysis
        """
        if not execution_result.get('success', False):
            error_msg = execution_result.get('error', 'Unknown error')
            
            # Classify error type for appropriate handling
            if 'timeout' in execution_result:
                result_type = ExecutionResult.TIMEOUT
            elif 'SyntaxError' in error_msg or 'IndentationError' in error_msg:
                result_type = ExecutionResult.SYNTAX_ERROR
            else:
                result_type = ExecutionResult.RUNTIME_ERROR
            
            return ValidationResult(
                success=False,
                execution_result=result_type,
                output=execution_result.get('output', ''),
                execution_time=execution_time,
                errors=[error_msg]
            )
        
        # Successful execution - validate output if expected output provided
        output = execution_result.get('output', '')
        expected_output = execution_context.expected_output
        
        validation_score = 1.0  # Default perfect score for successful execution
        confidence_level = 0.9  # High confidence for successful execution
        warnings = []
        
        # Output validation if expected output specified
        if expected_output:
            validation_score = self._calculate_output_similarity(output, expected_output)
            confidence_level = min(validation_score + 0.1, 1.0)
            
            if validation_score < 0.8:
                warnings.append("Output differs significantly from expected result")
        
        # Performance validation
        if execution_time > execution_context.timeout * 0.8:
            warnings.append("Execution time approaching timeout limit")
        
        return ValidationResult(
            success=True,
            execution_result=ExecutionResult.SUCCESS,
            output=output,
            expected_output=expected_output,
            execution_time=execution_time,
            validation_score=validation_score,
            confidence_level=confidence_level,
            warnings=warnings
        )
    
    def _calculate_output_similarity(self, actual: str, expected: str) -> float:
        """
        Calculate similarity between actual and expected output
        
        @param {str} actual - Actual execution output
        @param {str} expected - Expected execution output
        @returns {float} Similarity score between 0 and 1
        """
        if not actual or not expected:
            return 0.0
        
        # Simple similarity calculation - can be enhanced with more sophisticated algorithms
        actual_lines = set(actual.strip().split('\n'))
        expected_lines = set(expected.strip().split('\n'))
        
        if not expected_lines:
            return 1.0 if not actual_lines else 0.0
        
        intersection = actual_lines.intersection(expected_lines)
        union = actual_lines.union(expected_lines)
        
        return len(intersection) / len(union) if union else 1.0


class IterativeRefinementEngine:
    """
    LLM-based iterative code refinement and improvement engine
    
    Implements iterative refinement with error inspection and code regeneration
    as specified in Alita Section 2.3.3, using LangChain agents for intelligent
    code improvement and OpenRouter for LLM interactions (per user memory).
    
    Features:
    - Multiple refinement strategies for different error types
    - Integration with KGoT error management for robust error handling
    - LangChain agent-based code improvement
    - Iterative refinement with convergence detection
    - Learning from previous refinement patterns
    """
    
    def __init__(self, 
                 llm_client: Optional[ChatOpenAI] = None,
                 error_management_system: Optional[KGoTErrorManagementSystem] = None,
                 max_refinement_iterations: int = 5):
        """
        Initialize iterative refinement engine with LLM and error management
        
        @param {ChatOpenAI} llm_client - OpenRouter-based LLM client (per user memory)
        @param {KGoTErrorManagementSystem} error_management_system - KGoT error management integration
        @param {int} max_refinement_iterations - Maximum number of refinement attempts
        """
        self.llm_client = llm_client or self._create_default_llm_client()
        self.error_management = error_management_system or create_kgot_error_management_system(self.llm_client)
        self.max_refinement_iterations = max_refinement_iterations
        self.refinement_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # LangChain agent for code improvement (per user memory requirement)
        self.refinement_agent = self._create_refinement_agent()
        
        logger.info("Initialized IterativeRefinementEngine", extra={
            'operation': 'REFINEMENT_ENGINE_INIT',
            'max_iterations': max_refinement_iterations,
            'has_llm_client': self.llm_client is not None,
            'has_error_management': self.error_management is not None
        })
    
    def _create_default_llm_client(self) -> ChatOpenAI:
        """
        Create Claude 4 Sonnet client for code running tasks (per user requirements)
        
        @returns {ChatOpenAI} Configured Claude 4 Sonnet client via OpenRouter
        """
        logger.info("Creating Claude 4 Sonnet client for code running")
        
        return ChatOpenAI(
            model="anthropic/claude-4-sonnet",  # Using Claude 4 Sonnet for code running per user requirements
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY", "mock_key_for_testing"),
            temperature=0.1,  # Low temperature for consistent code generation and refinement
            max_tokens=8000,  # Higher token limit for complex code tasks
            model_kwargs={
                "headers": {
                    "HTTP-Referer": "https://github.com/alita-kgot-enhanced",
                    "X-Title": "Alita KGoT Code Runner - Claude 4 Sonnet"
                }
            }
        )
    
    def _create_refinement_agent(self) -> Agent:
        """
        Create LangChain agent for intelligent code refinement (per user memory)
        
        @returns {Agent} Configured LangChain agent for code improvement
        """
        # Refinement prompt template for code improvement
        refinement_prompt = PromptTemplate(
            input_variables=["code", "errors", "strategy", "context"],
            template="""
You are an expert code refinement specialist. Your task is to improve the given code
based on the errors encountered and the specified refinement strategy.

Original Code:
{code}

Errors Encountered:
{errors}

Refinement Strategy: {strategy}

Context: {context}

Please provide an improved version of the code that:
1. Fixes all identified errors
2. Follows the specified refinement strategy
3. Maintains the original functionality
4. Includes proper error handling
5. Is well-documented with comments

Improved Code:
"""
        )
        
        # Create custom LangChain agent for code refinement
        class CodeRefinementAgent(Agent):
            def _get_llm_chain(self):
                return self.llm_client
            
            def plan(self, intermediate_steps, **kwargs):
                # Simple planning for code refinement
                return AgentAction(
                    tool="code_refinement",
                    tool_input=kwargs,
                    log="Refining code based on errors and strategy"
                )
        
        return CodeRefinementAgent(llm_chain=self.llm_client)
    
    async def refine_code_iteratively(self, 
                                     execution_context: ExecutionContext,
                                     validation_result: ValidationResult,
                                     strategy: RefinementStrategy = RefinementStrategy.COMPREHENSIVE) -> Tuple[str, bool]:
        """
        Perform iterative code refinement with error correction and improvement
        
        @param {ExecutionContext} execution_context - Original execution context
        @param {ValidationResult} validation_result - Validation results with errors
        @param {RefinementStrategy} strategy - Refinement approach to apply
        @returns {Tuple[str, bool]} - (refined_code, refinement_successful)
        """
        execution_id = execution_context.execution_id
        original_code = execution_context.code
        current_code = original_code
        
        logger.info("Starting iterative code refinement", extra={
            'operation': 'ITERATIVE_REFINEMENT_START',
            'execution_id': execution_id,
            'strategy': strategy.value,
            'initial_errors': len(validation_result.errors)
        })
        
        for iteration in range(self.max_refinement_iterations):
            try:
                # Generate refined code using LangChain agent and LLM
                refined_code = await self._generate_refined_code(
                    current_code, 
                    validation_result.errors,
                    strategy,
                    execution_context,
                    iteration
                )
                
                if refined_code == current_code:
                    # No changes made - refinement converged
                    logger.info("Code refinement converged", extra={
                        'operation': 'REFINEMENT_CONVERGED',
                        'execution_id': execution_id,
                        'iteration': iteration
                    })
                    break
                
                current_code = refined_code
                
                # Record refinement iteration
                self.refinement_history[execution_id].append({
                    'iteration': iteration,
                    'strategy': strategy.value,
                    'code_length': len(refined_code),
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.debug("Refinement iteration completed", extra={
                    'operation': 'REFINEMENT_ITERATION',
                    'execution_id': execution_id,
                    'iteration': iteration,
                    'code_changed': refined_code != current_code
                })
                
            except Exception as e:
                logger.error("Refinement iteration failed", extra={
                    'operation': 'REFINEMENT_ITERATION_ERROR',
                    'execution_id': execution_id,
                    'iteration': iteration,
                    'error': str(e)
                })
                
                # Use error management system for recovery
                try:
                    error_context = ErrorContext(
                        error_id=f"refinement_{execution_id}_{iteration}",
                        error_type=ErrorType.SYSTEM_ERROR,
                        severity=ErrorSeverity.MEDIUM,
                        timestamp=datetime.now(),
                        original_operation=f"code_refinement_iteration_{iteration}",
                        error_message=str(e)
                    )
                    
                    recovery_result, recovered = await self.error_management.handle_error(
                        e, f"refinement_iteration_{iteration}", ErrorType.SYSTEM_ERROR
                    )
                    
                    if not recovered:
                        # If error management can't recover, return current best attempt
                        logger.warning("Could not recover from refinement error", extra={
                            'operation': 'REFINEMENT_RECOVERY_FAILED',
                            'execution_id': execution_id,
                            'iteration': iteration
                        })
                        break
                        
                except Exception as recovery_error:
                    logger.error("Error management recovery failed", extra={
                        'operation': 'REFINEMENT_ERROR_RECOVERY_FAILED',
                        'execution_id': execution_id,
                        'recovery_error': str(recovery_error)
                    })
                    break
        
        # Determine if refinement was successful
        refinement_successful = current_code != original_code
        
        logger.info("Iterative refinement completed", extra={
            'operation': 'ITERATIVE_REFINEMENT_COMPLETE',
            'execution_id': execution_id,
            'successful': refinement_successful,
            'iterations_used': len(self.refinement_history[execution_id]),
            'final_code_length': len(current_code)
        })
        
        return current_code, refinement_successful 

    async def _generate_refined_code(self, 
                                   current_code: str,
                                   errors: List[str], 
                                   strategy: RefinementStrategy,
                                   execution_context: ExecutionContext,
                                   iteration: int) -> str:
        """
        Generate refined code using LLM and refinement strategy
        
        @param {str} current_code - Current code to refine
        @param {List[str]} errors - List of errors to address
        @param {RefinementStrategy} strategy - Refinement strategy to apply
        @param {ExecutionContext} execution_context - Original execution context
        @param {int} iteration - Current iteration number
        @returns {str} Refined code with improvements
        """
        try:
            # Prepare context for refinement
            context_info = {
                'execution_id': execution_context.execution_id,
                'language': execution_context.language,
                'dependencies': execution_context.dependencies,
                'iteration': iteration,
                'previous_attempts': len(self.refinement_history[execution_context.execution_id])
            }
            
            # Create refinement prompt
            refinement_prompt = self._build_refinement_prompt(
                current_code, errors, strategy, context_info
            )
            
            # Generate refined code using LLM
            response = await self.llm_client.agenerate([refinement_prompt])
            refined_code = response.generations[0][0].text.strip()
            
            # Extract code from response (remove any markdown formatting)
            refined_code = self._extract_code_from_response(refined_code)
            
            logger.debug("Generated refined code", extra={
                'operation': 'REFINED_CODE_GENERATED',
                'execution_id': execution_context.execution_id,
                'iteration': iteration,
                'code_length': len(refined_code),
                'strategy': strategy.value
            })
            
            return refined_code
            
        except Exception as e:
            logger.error("Failed to generate refined code", extra={
                'operation': 'REFINED_CODE_GENERATION_ERROR',
                'execution_id': execution_context.execution_id,
                'iteration': iteration,
                'error': str(e)
            })
            # Return original code if refinement fails
            return current_code
    
    def _build_refinement_prompt(self, 
                               code: str, 
                               errors: List[str], 
                               strategy: RefinementStrategy,
                               context: Dict[str, Any]) -> HumanMessage:
        """
        Build comprehensive refinement prompt for LLM
        
        @param {str} code - Code to refine
        @param {List[str]} errors - Errors to address
        @param {RefinementStrategy} strategy - Refinement approach
        @param {Dict[str, Any]} context - Additional context information
        @returns {HumanMessage} Formatted prompt for LLM
        """
        strategy_instructions = {
            RefinementStrategy.SYNTAX_CORRECTION: "Focus on fixing syntax errors and ensuring code compiles correctly.",
            RefinementStrategy.LOGIC_IMPROVEMENT: "Improve the logic and algorithm implementation for better results.",
            RefinementStrategy.PERFORMANCE_OPTIMIZATION: "Optimize the code for better performance and efficiency.",
            RefinementStrategy.ERROR_HANDLING: "Add comprehensive error handling and exception management.",
            RefinementStrategy.SECURITY_HARDENING: "Enhance security measures and input validation.",
            RefinementStrategy.COMPREHENSIVE: "Apply all improvements: fix errors, improve logic, optimize performance, add error handling, and enhance security."
        }
        
        prompt_text = f"""
You are an expert code refinement specialist. Improve the following code based on the errors and strategy provided.

Current Code:
```{context.get('language', 'python')}
{code}
```

Errors to Address:
{chr(10).join(f"- {error}" for error in errors)}

Refinement Strategy: {strategy.value}
Instructions: {strategy_instructions.get(strategy, strategy_instructions[RefinementStrategy.COMPREHENSIVE])}

Context:
- Language: {context.get('language', 'python')}
- Dependencies: {', '.join(context.get('dependencies', []))}
- Iteration: {context.get('iteration', 0)}
- Previous Attempts: {context.get('previous_attempts', 0)}

Requirements:
1. Fix all identified errors
2. Maintain original functionality
3. Add comprehensive comments using JSDoc3 style
4. Include proper error handling
5. Follow best practices for {context.get('language', 'python')}
6. Ensure code is production-ready

Provide only the improved code without any additional explanation:
"""
        return HumanMessage(content=prompt_text)
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract code from LLM response, removing markdown formatting
        
        @param {str} response - Raw LLM response
        @returns {str} Extracted code content
        """
        # Remove code block markers if present
        if '```' in response:
            lines = response.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    code_lines.append(line)
            
            return '\n'.join(code_lines)
        
        return response.strip()


class MCPServerGenerator:
    """
    MCP Server Generator for converting successful code into reusable tools
    
    Implements output caching for potential MCP server generation as specified
    in Alita Section 2.3.3, with integration to existing MCP infrastructure
    for tool registration and management.
    
    Features:
    - Automatic MCP server configuration generation
    - Tool metadata extraction and documentation
    - Integration with existing MCP brainstorming framework
    - Capability assessment and tool classification
    - Output caching for performance optimization
    """
    
    def __init__(self, 
                 mcp_registry_url: str = "http://localhost:3001/mcp-registry",
                 cache_directory: str = "./cache/mcp_generation"):
        """
        Initialize MCP server generator with registry integration
        
        @param {str} mcp_registry_url - URL for MCP tool registry service
        @param {str} cache_directory - Directory for caching generated MCPs
        """
        self.mcp_registry_url = mcp_registry_url
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.generated_mcps: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized MCPServerGenerator", extra={
            'operation': 'MCP_GENERATOR_INIT',
            'registry_url': mcp_registry_url,
            'cache_directory': str(self.cache_directory)
        })
    
    async def generate_mcp_from_successful_code(self, 
                                              execution_context: ExecutionContext,
                                              validation_result: ValidationResult,
                                              refined_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate MCP server configuration from successfully executed code
        
        @param {ExecutionContext} execution_context - Original execution context
        @param {ValidationResult} validation_result - Successful validation results
        @param {str} refined_code - Optionally refined code (if different from original)
        @returns {Dict[str, Any]} Generated MCP configuration and metadata
        """
        if not validation_result.success:
            raise ValueError("Cannot generate MCP from failed code execution")
        
        execution_id = execution_context.execution_id
        final_code = refined_code or execution_context.code
        
        logger.info("Generating MCP from successful code", extra={
            'operation': 'MCP_GENERATION_START',
            'execution_id': execution_id,
            'validation_score': validation_result.validation_score
        })
        
        try:
            # Extract tool capabilities and metadata
            tool_metadata = await self._analyze_code_capabilities(final_code, execution_context)
            
            # Generate MCP server configuration
            mcp_config = await self._create_mcp_configuration(
                final_code, execution_context, validation_result, tool_metadata
            )
            
            # Generate tool documentation
            documentation = await self._generate_tool_documentation(
                mcp_config, execution_context, validation_result
            )
            
            # Cache generated MCP for future use
            cache_key = self._generate_cache_key(execution_context, final_code)
            await self._cache_mcp_configuration(cache_key, mcp_config, documentation)
            
            # Register with MCP registry if available
            registration_result = await self._register_mcp_tool(mcp_config, documentation)
            
            mcp_result = {
                'mcp_config': mcp_config,
                'documentation': documentation,
                'tool_metadata': tool_metadata,
                'cache_key': cache_key,
                'registration_result': registration_result,
                'generated_at': datetime.now().isoformat()
            }
            
            self.generated_mcps[execution_id] = mcp_result
            
            logger.info("MCP generation completed", extra={
                'operation': 'MCP_GENERATION_COMPLETE',
                'execution_id': execution_id,
                'tool_name': mcp_config.get('name'),
                'capabilities_count': len(tool_metadata.get('capabilities', []))
            })
            
            return mcp_result
            
        except Exception as e:
            logger.error("MCP generation failed", extra={
                'operation': 'MCP_GENERATION_ERROR',
                'execution_id': execution_id,
                'error': str(e)
            })
            raise
    
    async def _analyze_code_capabilities(self, 
                                       code: str, 
                                       execution_context: ExecutionContext) -> Dict[str, Any]:
        """
        Analyze code to extract capabilities and classify tool type
        
        @param {str} code - Code to analyze for capabilities
        @param {ExecutionContext} execution_context - Execution context with metadata
        @returns {Dict[str, Any]} Extracted capabilities and metadata
        """
        # Analyze imports and functions to determine capabilities
        capabilities = []
        tool_type = "utility"
        complexity_score = 0.0
        
        lines = code.split('\n')
        imports = [line.strip() for line in lines if line.strip().startswith('import') or line.strip().startswith('from')]
        functions = [line.strip() for line in lines if line.strip().startswith('def ')]
        classes = [line.strip() for line in lines if line.strip().startswith('class ')]
        
        # Determine capabilities based on imports and code content
        capability_mapping = {
            'requests': ['web_interaction', 'api_client'],
            'pandas': ['data_processing', 'data_analysis'],
            'numpy': ['numerical_computation', 'data_processing'],
            'matplotlib': ['visualization', 'plotting'],
            'json': ['data_serialization', 'json_processing'],
            'sqlite3': ['database_operations', 'data_storage'],
            'os': ['file_operations', 'system_interaction'],
            'subprocess': ['system_execution', 'process_management'],
            'BeautifulSoup': ['web_scraping', 'html_parsing'],
            'selenium': ['browser_automation', 'ui_testing']
        }
        
        for import_line in imports:
            for module, caps in capability_mapping.items():
                if module in import_line:
                    capabilities.extend(caps)
        
        # Calculate complexity score
        complexity_score = (len(functions) * 0.3 + len(classes) * 0.5 + len(imports) * 0.2) / 10
        complexity_score = min(complexity_score, 1.0)
        
        # Determine tool type based on capabilities
        if any(cap in ['web_interaction', 'api_client', 'web_scraping'] for cap in capabilities):
            tool_type = "web_tool"
        elif any(cap in ['data_processing', 'data_analysis'] for cap in capabilities):
            tool_type = "data_tool"
        elif any(cap in ['file_operations', 'system_interaction'] for cap in capabilities):
            tool_type = "system_tool"
        
        return {
            'capabilities': list(set(capabilities)),
            'tool_type': tool_type,
            'complexity_score': complexity_score,
            'function_count': len(functions),
            'class_count': len(classes),
            'import_count': len(imports),
            'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        }
    
    async def _create_mcp_configuration(self, 
                                      code: str,
                                      execution_context: ExecutionContext,
                                      validation_result: ValidationResult,
                                      tool_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive MCP server configuration
        
        @param {str} code - Final executable code
        @param {ExecutionContext} execution_context - Execution context
        @param {ValidationResult} validation_result - Validation results
        @param {Dict[str, Any]} tool_metadata - Extracted tool metadata
        @returns {Dict[str, Any]} Complete MCP configuration
        """
        tool_name = f"generated_tool_{execution_context.execution_id[:8]}"
        
        mcp_config = {
            'name': tool_name,
            'version': '1.0.0',
            'description': f"Auto-generated tool from successful code execution (ID: {execution_context.execution_id})",
            'type': tool_metadata.get('tool_type', 'utility'),
            'capabilities': tool_metadata.get('capabilities', []),
            'code': code,
            'language': execution_context.language,
            'dependencies': execution_context.dependencies,
            'resource_requirements': execution_context.resource_limits,
            'timeout': execution_context.timeout,
            'validation_score': validation_result.validation_score,
            'confidence_level': validation_result.confidence_level,
            'complexity_score': tool_metadata.get('complexity_score', 0.0),
            'performance_metrics': {
                'execution_time': validation_result.execution_time,
                'success_rate': 1.0,  # Based on successful execution
                'reliability_score': validation_result.confidence_level
            },
            'generated_from': {
                'execution_id': execution_context.execution_id,
                'original_context': execution_context.metadata,
                'validation_metadata': validation_result.metadata
            },
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        return mcp_config
    
    async def _generate_tool_documentation(self, 
                                         mcp_config: Dict[str, Any],
                                         execution_context: ExecutionContext,
                                         validation_result: ValidationResult) -> Dict[str, Any]:
        """
        Generate comprehensive documentation for the MCP tool
        
        @param {Dict[str, Any]} mcp_config - MCP configuration
        @param {ExecutionContext} execution_context - Original execution context
        @param {ValidationResult} validation_result - Validation results
        @returns {Dict[str, Any]} Complete tool documentation
        """
        documentation = {
            'title': f"Auto-Generated Tool: {mcp_config['name']}",
            'description': mcp_config['description'],
            'usage': {
                'language': execution_context.language,
                'dependencies': execution_context.dependencies,
                'example_usage': f"# Generated from execution {execution_context.execution_id}\n{mcp_config['code'][:200]}..."
            },
            'capabilities': {
                'primary_functions': mcp_config['capabilities'],
                'tool_type': mcp_config['type'],
                'complexity_level': 'low' if mcp_config['complexity_score'] < 0.3 else 'medium' if mcp_config['complexity_score'] < 0.7 else 'high'
            },
            'performance': {
                'execution_time': validation_result.execution_time,
                'validation_score': validation_result.validation_score,
                'confidence_level': validation_result.confidence_level,
                'resource_usage': validation_result.resource_usage
            },
            'requirements': {
                'timeout': execution_context.timeout,
                'memory_limit': execution_context.resource_limits.get('memory', 'N/A'),
                'cpu_limit': execution_context.resource_limits.get('cpu', 'N/A')
            },
            'generated_metadata': {
                'generated_at': datetime.now().isoformat(),
                'source_execution_id': execution_context.execution_id,
                'validation_warnings': validation_result.warnings
            }
        }
        
        return documentation


class ValidationFramework:
    """
    Cross-validation framework for comprehensive execution testing
    
    Implements comprehensive validation testing using cross-validation framework
    with majority voting approach as specified in KGoT research paper Section 3.5
    for ensuring system robustness and reliability.
    
    Features:
    - Multiple validation rounds with majority voting
    - Cross-validation testing across different environments
    - Statistical analysis of validation results
    - Integration with existing KGoT validation infrastructure
    - Confidence interval calculation and reliability assessment
    """
    
    def __init__(self, 
                 validation_rounds: int = 3,
                 consensus_threshold: float = 0.67,
                 execution_environment: Optional[ExecutionEnvironment] = None):
        """
        Initialize validation framework with cross-validation parameters
        
        @param {int} validation_rounds - Number of validation rounds for majority voting
        @param {float} consensus_threshold - Threshold for consensus agreement (0-1)
        @param {ExecutionEnvironment} execution_environment - Execution environment for testing
        """
        self.validation_rounds = validation_rounds
        self.consensus_threshold = consensus_threshold
        self.execution_environment = execution_environment or ExecutionEnvironment()
        self.validation_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("Initialized ValidationFramework", extra={
            'operation': 'VALIDATION_FRAMEWORK_INIT',
            'validation_rounds': validation_rounds,
            'consensus_threshold': consensus_threshold
        })
    
    async def perform_cross_validation(self, 
                                     execution_context: ExecutionContext,
                                     test_variations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation testing with majority voting
        
        @param {ExecutionContext} execution_context - Base execution context for validation
        @param {List[Dict[str, Any]]} test_variations - Optional test variations for comprehensive testing
        @returns {Dict[str, Any]} Comprehensive validation results with statistical analysis
        """
        execution_id = execution_context.execution_id
        
        logger.info("Starting cross-validation testing", extra={
            'operation': 'CROSS_VALIDATION_START',
            'execution_id': execution_id,
            'validation_rounds': self.validation_rounds
        })
        
        validation_results = []
        
        for round_num in range(self.validation_rounds):
            try:
                # Create variation of execution context for this round
                varied_context = self._create_validation_variation(
                    execution_context, round_num, test_variations
                )
                
                # Execute validation round
                round_result = await self.execution_environment.execute_code_securely(varied_context)
                
                # Record round results
                validation_results.append({
                    'round': round_num,
                    'context_variation': varied_context.metadata.get('variation_type', 'standard'),
                    'result': round_result,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.debug("Validation round completed", extra={
                    'operation': 'VALIDATION_ROUND_COMPLETE',
                    'execution_id': execution_id,
                    'round': round_num,
                    'success': round_result.success
                })
                
            except Exception as e:
                logger.error("Validation round failed", extra={
                    'operation': 'VALIDATION_ROUND_ERROR',
                    'execution_id': execution_id,
                    'round': round_num,
                    'error': str(e)
                })
                
                # Record failed round
                validation_results.append({
                    'round': round_num,
                    'context_variation': 'error',
                    'result': ValidationResult(
                        success=False,
                        execution_result=ExecutionResult.FAILURE,
                        output="",
                        errors=[str(e)]
                    ),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Analyze validation results with majority voting
        analysis_result = self._analyze_validation_results(validation_results, execution_context)
        
        # Store validation history
        self.validation_history[execution_id] = validation_results
        
        logger.info("Cross-validation completed", extra={
            'operation': 'CROSS_VALIDATION_COMPLETE',
            'execution_id': execution_id,
            'consensus_achieved': analysis_result['consensus_achieved'],
            'success_rate': analysis_result['success_rate']
        })
        
        return analysis_result 

    def _create_validation_variation(self, 
                                   base_context: ExecutionContext,
                                   round_num: int,
                                   test_variations: Optional[List[Dict[str, Any]]] = None) -> ExecutionContext:
        """
        Create execution context variation for validation round
        
        @param {ExecutionContext} base_context - Base execution context
        @param {int} round_num - Current validation round number
        @param {List[Dict[str, Any]]} test_variations - Optional predefined variations
        @returns {ExecutionContext} Modified execution context for validation
        """
        # Create copy of base context
        varied_context = ExecutionContext(
            execution_id=f"{base_context.execution_id}_validation_{round_num}",
            code=base_context.code,
            language=base_context.language,
            expected_output=base_context.expected_output,
            timeout=base_context.timeout,
            resource_limits=base_context.resource_limits.copy(),
            dependencies=base_context.dependencies.copy(),
            environment_vars=base_context.environment_vars.copy(),
            validation_criteria=base_context.validation_criteria.copy(),
            metadata=base_context.metadata.copy()
        )
        
        # Apply variation if specified
        if test_variations and round_num < len(test_variations):
            variation = test_variations[round_num]
            varied_context.metadata['variation_type'] = variation.get('type', 'custom')
            
            # Apply specific variations based on type
            if 'timeout_factor' in variation:
                varied_context.timeout = int(base_context.timeout * variation['timeout_factor'])
            if 'memory_factor' in variation:
                memory_mb = int(float(base_context.resource_limits.get('memory', '256m').replace('m', '')) * variation['memory_factor'])
                varied_context.resource_limits['memory'] = f"{memory_mb}m"
            if 'environment_additions' in variation:
                varied_context.environment_vars.update(variation['environment_additions'])
        else:
            # Default variations for different rounds
            varied_context.metadata['variation_type'] = 'default'
            
            if round_num == 1:
                # Stricter resource limits
                varied_context.timeout = max(base_context.timeout // 2, 5)
                memory_mb = max(int(float(base_context.resource_limits.get('memory', '256m').replace('m', '')) * 0.5), 64)
                varied_context.resource_limits['memory'] = f"{memory_mb}m"
            elif round_num == 2:
                # More relaxed limits
                varied_context.timeout = base_context.timeout * 2
                memory_mb = int(float(base_context.resource_limits.get('memory', '256m').replace('m', '')) * 1.5)
                varied_context.resource_limits['memory'] = f"{memory_mb}m"
        
        return varied_context
    
    def _analyze_validation_results(self, 
                                  validation_results: List[Dict[str, Any]],
                                  execution_context: ExecutionContext) -> Dict[str, Any]:
        """
        Analyze validation results using majority voting and statistical analysis
        
        @param {List[Dict[str, Any]]} validation_results - Results from all validation rounds
        @param {ExecutionContext} execution_context - Original execution context
        @returns {Dict[str, Any]} Comprehensive validation analysis
        """
        total_rounds = len(validation_results)
        successful_rounds = sum(1 for result in validation_results if result['result'].success)
        success_rate = successful_rounds / total_rounds if total_rounds > 0 else 0.0
        
        # Majority voting for consensus
        consensus_achieved = success_rate >= self.consensus_threshold
        
        # Calculate average execution times and scores
        successful_results = [r['result'] for r in validation_results if r['result'].success]
        if successful_results:
            avg_execution_time = sum(r.execution_time for r in successful_results) / len(successful_results)
            avg_validation_score = sum(r.validation_score for r in successful_results) / len(successful_results)
            avg_confidence = sum(r.confidence_level for r in successful_results) / len(successful_results)
        else:
            avg_execution_time = 0.0
            avg_validation_score = 0.0
            avg_confidence = 0.0
        
        # Collect all errors across rounds
        all_errors = []
        for result in validation_results:
            all_errors.extend(result['result'].errors)
        
        # Analyze error patterns
        error_frequency = Counter(all_errors)
        common_errors = error_frequency.most_common(3)
        
        # Calculate confidence intervals
        if successful_results:
            execution_times = [r.execution_time for r in successful_results]
            time_variance = sum((t - avg_execution_time) ** 2 for t in execution_times) / len(execution_times)
            time_std_dev = time_variance ** 0.5
        else:
            time_std_dev = 0.0
        
        analysis_result = {
            'consensus_achieved': consensus_achieved,
            'success_rate': success_rate,
            'total_rounds': total_rounds,
            'successful_rounds': successful_rounds,
            'average_metrics': {
                'execution_time': avg_execution_time,
                'validation_score': avg_validation_score,
                'confidence_level': avg_confidence,
                'time_std_deviation': time_std_dev
            },
            'error_analysis': {
                'total_errors': len(all_errors),
                'unique_errors': len(set(all_errors)),
                'common_errors': common_errors,
                'error_rate': (total_rounds - successful_rounds) / total_rounds if total_rounds > 0 else 0.0
            },
            'reliability_assessment': {
                'highly_reliable': success_rate >= 0.9,
                'moderately_reliable': 0.7 <= success_rate < 0.9,
                'low_reliability': success_rate < 0.7,
                'consensus_threshold': self.consensus_threshold
            },
            'validation_rounds': validation_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return analysis_result


class ResultCache:
    """
    Intelligent caching system for execution results and MCP generation
    
    Provides performance optimization through intelligent caching of execution results,
    supporting future MCP server generation and reducing redundant computations.
    
    Features:
    - Hash-based cache key generation for efficient lookup
    - TTL (Time To Live) support for cache expiration
    - LRU (Least Recently Used) eviction policy
    - Integration with file system for persistent caching
    - Cache statistics and performance monitoring
    """
    
    def __init__(self, 
                 cache_directory: str = "./cache/execution_results",
                 max_cache_size: int = 1000,
                 default_ttl: int = 3600):  # 1 hour default TTL
        """
        Initialize result cache with configuration
        
        @param {str} cache_directory - Directory for persistent cache storage
        @param {int} max_cache_size - Maximum number of cached items
        @param {int} default_ttl - Default cache TTL in seconds
        """
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        
        # In-memory cache with metadata
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
        
        logger.info("Initialized ResultCache", extra={
            'operation': 'RESULT_CACHE_INIT',
            'cache_directory': str(self.cache_directory),
            'max_size': max_cache_size,
            'default_ttl': default_ttl
        })
    
    def generate_cache_key(self, execution_context: ExecutionContext) -> str:
        """
        Generate unique cache key for execution context
        
        @param {ExecutionContext} execution_context - Execution context to cache
        @returns {str} Unique cache key
        """
        # Create hash from relevant execution parameters
        key_components = [
            execution_context.code,
            execution_context.language,
            str(execution_context.dependencies),
            str(execution_context.resource_limits),
            str(execution_context.environment_vars)
        ]
        
        key_string = '|'.join(key_components)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{execution_context.language}_{cache_key}"
    
    async def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result if available and not expired
        
        @param {str} cache_key - Cache key to lookup
        @returns {Optional[Dict[str, Any]]} Cached result or None if not found/expired
        """
        if cache_key not in self.cache:
            self.cache_stats['misses'] += 1
            return None
        
        cached_item = self.cache[cache_key]
        
        # Check TTL expiration
        if datetime.now() > cached_item['expires_at']:
            # Remove expired item
            await self._remove_from_cache(cache_key)
            self.cache_stats['misses'] += 1
            return None
        
        # Update access time for LRU
        self.access_times[cache_key] = datetime.now()
        self.cache_stats['hits'] += 1
        
        logger.debug("Cache hit", extra={
            'operation': 'CACHE_HIT',
            'cache_key': cache_key,
            'cached_at': cached_item['cached_at']
        })
        
        return cached_item['data']
    
    async def cache_result(self, 
                          cache_key: str, 
                          result_data: Dict[str, Any],
                          ttl: Optional[int] = None) -> None:
        """
        Cache execution result with TTL and persistence
        
        @param {str} cache_key - Unique cache key
        @param {Dict[str, Any]} result_data - Result data to cache
        @param {int} ttl - Optional custom TTL in seconds
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        # Ensure cache size limit
        if len(self.cache) >= self.max_cache_size:
            await self._evict_lru_item()
        
        # Prepare cache item
        cache_item = {
            'data': result_data,
            'cached_at': datetime.now().isoformat(),
            'expires_at': expires_at,
            'ttl': ttl,
            'access_count': 1
        }
        
        # Store in memory
        self.cache[cache_key] = cache_item
        self.access_times[cache_key] = datetime.now()
        
        # Persist to disk
        await self._persist_cache_item(cache_key, cache_item)
        
        self.cache_stats['size'] = len(self.cache)
        
        logger.debug("Result cached", extra={
            'operation': 'RESULT_CACHED',
            'cache_key': cache_key,
            'ttl': ttl,
            'cache_size': len(self.cache)
        })
    
    async def _evict_lru_item(self) -> None:
        """Evict least recently used item from cache"""
        if not self.access_times:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        await self._remove_from_cache(lru_key)
        self.cache_stats['evictions'] += 1
        
        logger.debug("LRU item evicted", extra={
            'operation': 'CACHE_LRU_EVICTION',
            'evicted_key': lru_key
        })
    
    async def _remove_from_cache(self, cache_key: str) -> None:
        """Remove item from cache and persistent storage"""
        if cache_key in self.cache:
            del self.cache[cache_key]
        if cache_key in self.access_times:
            del self.access_times[cache_key]
        
        # Remove from persistent storage
        cache_file = self.cache_directory / f"{cache_key}.json"
        if cache_file.exists():
            cache_file.unlink()
    
    async def _persist_cache_item(self, cache_key: str, cache_item: Dict[str, Any]) -> None:
        """Persist cache item to disk for durability"""
        try:
            cache_file = self.cache_directory / f"{cache_key}.json"
            
            # Serialize cache item (handling datetime objects)
            serializable_item = cache_item.copy()
            serializable_item['expires_at'] = cache_item['expires_at'].isoformat()
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_item, f, indent=2)
                
        except Exception as e:
            logger.warning("Failed to persist cache item", extra={
                'operation': 'CACHE_PERSISTENCE_ERROR',
                'cache_key': cache_key,
                'error': str(e)
            })
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache performance statistics
        
        @returns {Dict[str, Any]} Cache performance metrics
        """
        hit_rate = self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses']) if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0.0
        
        return {
            'hit_rate': hit_rate,
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses'],
            'total_evictions': self.cache_stats['evictions'],
            'current_size': self.cache_stats['size'],
            'max_size': self.max_cache_size,
            'cache_utilization': self.cache_stats['size'] / self.max_cache_size if self.max_cache_size > 0 else 0.0
        }


class CodeRunningTool:
    """
    Main Alita Code Running Tool with comprehensive KGoT integration
    
    Implementation of Alita Section 2.3.3 "CodeRunningTool" complete functionality
    with full integration of KGoT execution environment, error management, and
    MCP generation capabilities.
    
    Features:
    - Script validation through isolated environment execution
    - Iterative refinement with error inspection and code regeneration
    - Output caching for potential MCP server generation
    - Cross-validation testing with majority voting
    - Integration with KGoT containerization and error management
    - Automatic tool registration as reusable MCPs
    - Comprehensive logging and performance monitoring
    """
    
    def __init__(self, 
                 llm_client: Optional[ChatOpenAI] = None,
                 container_orchestrator: Optional[ContainerOrchestrator] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Alita Code Running Tool with comprehensive integration
        
        @param {ChatOpenAI} llm_client - OpenRouter-based LLM client (per user memory)
        @param {ContainerOrchestrator} container_orchestrator - KGoT container orchestrator
        @param {Dict[str, Any]} config - Configuration options
        """
        self.config = config or {}
        
        # Initialize core components
        self.execution_environment = ExecutionEnvironment(
            container_orchestrator=container_orchestrator,
            max_concurrent_executions=self.config.get('max_concurrent_executions', 5)
        )
        
        self.refinement_engine = IterativeRefinementEngine(
            llm_client=llm_client,
            max_refinement_iterations=self.config.get('max_refinement_iterations', 5)
        )
        
        self.mcp_generator = MCPServerGenerator(
            mcp_registry_url=self.config.get('mcp_registry_url', 'http://localhost:3001/mcp-registry'),
            cache_directory=self.config.get('mcp_cache_directory', './cache/mcp_generation')
        )
        
        self.validation_framework = ValidationFramework(
            validation_rounds=self.config.get('validation_rounds', 3),
            consensus_threshold=self.config.get('consensus_threshold', 0.67),
            execution_environment=self.execution_environment
        )
        
        self.result_cache = ResultCache(
            cache_directory=self.config.get('result_cache_directory', './cache/execution_results'),
            max_cache_size=self.config.get('max_cache_size', 1000),
            default_ttl=self.config.get('cache_ttl', 3600)
        )
        
        # Tool session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.tool_registry: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized CodeRunningTool", extra={
            'operation': 'CODE_RUNNER_INIT',
            'config_keys': list(self.config.keys()),
            'has_llm_client': llm_client is not None,
            'has_container_orchestrator': container_orchestrator is not None
        })
    
    async def validate_and_execute_code(self, 
                                      code: str,
                                      language: str = "python",
                                      expected_output: Optional[str] = None,
                                      dependencies: Optional[List[str]] = None,
                                      timeout: int = 30,
                                      enable_refinement: bool = True,
                                      enable_caching: bool = True,
                                      enable_cross_validation: bool = True,
                                      generate_mcp: bool = True) -> Dict[str, Any]:
        """
        Main entry point for code validation and execution with comprehensive features
        
        @param {str} code - Source code to validate and execute
        @param {str} language - Programming language (default: python)
        @param {str} expected_output - Expected execution output for validation
        @param {List[str]} dependencies - Required packages/modules
        @param {int} timeout - Maximum execution time in seconds
        @param {bool} enable_refinement - Enable iterative refinement on errors
        @param {bool} enable_caching - Enable result caching
        @param {bool} enable_cross_validation - Enable cross-validation testing
        @param {bool} generate_mcp - Generate MCP server for successful code
        @returns {Dict[str, Any]} Comprehensive execution and validation results
        """
        # Create execution context
        execution_context = ExecutionContext(
            execution_id=str(uuid.uuid4()),
            code=code,
            language=language,
            expected_output=expected_output,
            timeout=timeout,
            dependencies=dependencies or [],
            metadata={
                'enable_refinement': enable_refinement,
                'enable_caching': enable_caching,
                'enable_cross_validation': enable_cross_validation,
                'generate_mcp': generate_mcp
            }
        )
        
        session_id = execution_context.execution_id
        self.active_sessions[session_id] = {
            'context': execution_context,
            'started_at': datetime.now(),
            'status': 'initializing'
        }
        
        logger.info("Starting code validation and execution", extra={
            'operation': 'CODE_VALIDATION_START',
            'session_id': session_id,
            'language': language,
            'code_length': len(code),
            'features_enabled': {
                'refinement': enable_refinement,
                'caching': enable_caching,
                'cross_validation': enable_cross_validation,
                'mcp_generation': generate_mcp
            }
        })
        
        try:
            # Check cache first if enabled
            cache_key = None
            if enable_caching:
                cache_key = self.result_cache.generate_cache_key(execution_context)
                cached_result = await self.result_cache.get_cached_result(cache_key)
                
                if cached_result:
                    logger.info("Using cached result", extra={
                        'operation': 'CACHED_RESULT_USED',
                        'session_id': session_id,
                        'cache_key': cache_key
                    })
                    self.active_sessions[session_id]['status'] = 'completed_from_cache'
                    return cached_result
            
            self.active_sessions[session_id]['status'] = 'executing'
            
            # Execute code in secure environment
            initial_result = await self.execution_environment.execute_code_securely(execution_context)
            
            current_code = code
            final_validation_result = initial_result
            refinement_applied = False
            
            # Apply iterative refinement if enabled and needed
            if enable_refinement and not initial_result.success:
                self.active_sessions[session_id]['status'] = 'refining'
                
                refined_code, refinement_successful = await self.refinement_engine.refine_code_iteratively(
                    execution_context, initial_result
                )
                
                if refinement_successful:
                    refinement_applied = True
                    current_code = refined_code
                    
                    # Re-execute refined code
                    refined_context = execution_context
                    refined_context.code = refined_code
                    final_validation_result = await self.execution_environment.execute_code_securely(refined_context)
            
            # Perform cross-validation if enabled and code is successful
            cross_validation_result = None
            if enable_cross_validation and final_validation_result.success:
                self.active_sessions[session_id]['status'] = 'cross_validating'
                
                validation_context = execution_context
                validation_context.code = current_code
                cross_validation_result = await self.validation_framework.perform_cross_validation(validation_context)
            
            # Generate MCP if enabled and validation successful
            mcp_result = None
            if generate_mcp and final_validation_result.success:
                try:
                    self.active_sessions[session_id]['status'] = 'generating_mcp'
                    
                    mcp_result = await self.mcp_generator.generate_mcp_from_successful_code(
                        execution_context, final_validation_result, current_code if refinement_applied else None
                    )
                    
                    # Register tool in local registry
                    tool_id = mcp_result['mcp_config']['name']
                    self.tool_registry[tool_id] = {
                        'session_id': session_id,
                        'mcp_config': mcp_result['mcp_config'],
                        'created_at': datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    logger.warning("MCP generation failed", extra={
                        'operation': 'MCP_GENERATION_WARNING',
                        'session_id': session_id,
                        'error': str(e)
                    })
            
            # Compile comprehensive results
            comprehensive_result = {
                'session_id': session_id,
                'execution_context': execution_context.to_dict(),
                'initial_execution': {
                    'success': initial_result.success,
                    'result': initial_result.to_dict()
                },
                'refinement': {
                    'applied': refinement_applied,
                    'successful': refinement_applied and final_validation_result.success,
                    'refined_code': current_code if refinement_applied else None
                },
                'final_validation': final_validation_result.to_dict(),
                'cross_validation': cross_validation_result,
                'mcp_generation': mcp_result,
                'performance_metrics': {
                    'total_execution_time': final_validation_result.execution_time,
                    'refinement_iterations': len(self.refinement_engine.refinement_history.get(session_id, [])),
                    'validation_rounds': len(cross_validation_result['validation_rounds']) if cross_validation_result else 0
                },
                'final_code': current_code,
                'cache_key': cache_key,
                'completed_at': datetime.now().isoformat(),
                'overall_success': final_validation_result.success
            }
            
            # Cache result if enabled and successful
            if enable_caching and cache_key and final_validation_result.success:
                await self.result_cache.cache_result(cache_key, comprehensive_result)
            
            self.active_sessions[session_id]['status'] = 'completed'
            self.active_sessions[session_id]['result'] = comprehensive_result
            
            logger.info("Code validation and execution completed", extra={
                'operation': 'CODE_VALIDATION_COMPLETE',
                'session_id': session_id,
                'overall_success': comprehensive_result['overall_success'],
                'refinement_applied': refinement_applied,
                'mcp_generated': mcp_result is not None
            })
            
            return comprehensive_result
            
        except Exception as e:
            logger.error("Code validation and execution failed", extra={
                'operation': 'CODE_VALIDATION_ERROR',
                'session_id': session_id,
                'error': str(e)
            })
            
            error_result = {
                'session_id': session_id,
                'overall_success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'failed_at': datetime.now().isoformat()
            }
            
            self.active_sessions[session_id]['status'] = 'failed'
            self.active_sessions[session_id]['error'] = error_result
            
            return error_result
    
    def get_tool_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Get registry of all generated MCP tools
        
        @returns {Dict[str, Dict[str, Any]]} Complete tool registry
        """
        return self.tool_registry.copy()
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of specific execution session
        
        @param {str} session_id - Session ID to query
        @returns {Optional[Dict[str, Any]]} Session status or None if not found
        """
        return self.active_sessions.get(session_id)
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics
        
        @returns {Dict[str, Any]} Performance metrics and statistics
        """
        total_sessions = len(self.active_sessions)
        completed_sessions = sum(1 for s in self.active_sessions.values() if s['status'] == 'completed')
        failed_sessions = sum(1 for s in self.active_sessions.values() if s['status'] == 'failed')
        
        return {
            'total_sessions': total_sessions,
            'completed_sessions': completed_sessions,
            'failed_sessions': failed_sessions,
            'success_rate': completed_sessions / total_sessions if total_sessions > 0 else 0.0,
            'generated_tools': len(self.tool_registry),
            'cache_statistics': self.result_cache.get_cache_statistics(),
            'active_sessions': len([s for s in self.active_sessions.values() if s['status'] not in ['completed', 'failed']]),
            'component_status': {
                'execution_environment': 'active',
                'refinement_engine': 'active',
                'mcp_generator': 'active',
                'validation_framework': 'active',
                'result_cache': 'active'
            }
        }


def create_code_running_tool(config: Optional[Dict[str, Any]] = None) -> CodeRunningTool:
    """
    Factory function to create configured CodeRunningTool instance
    
    @param {Dict[str, Any]} config - Configuration options
    @returns {CodeRunningTool} Configured CodeRunningTool instance
    """
    logger.info("Creating CodeRunningTool instance", extra={
        'operation': 'CODE_RUNNER_FACTORY',
        'config_provided': config is not None
    })
    
    return CodeRunningTool(config=config)


# Example usage and testing
async def main():
    """
    Example usage of the CodeRunningTool with comprehensive features
    """
    # Create code running tool
    code_runner = create_code_running_tool({
        'max_concurrent_executions': 3,
        'validation_rounds': 3,
        'enable_mcp_generation': True
    })
    
    # Example code to validate and execute
    test_code = """
import json
import requests

def fetch_user_data(user_id):
    \"\"\"
    Fetch user data from a mock API
    
    @param {int} user_id - User ID to fetch
    @returns {dict} User data
    \"\"\"
    try:
        # Mock API call simulation
        user_data = {
            'id': user_id,
            'name': f'User {user_id}',
            'email': f'user{user_id}@example.com',
            'status': 'active'
        }
        
        return user_data
        
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return None

# Test the function
result = fetch_user_data(123)
print(json.dumps(result, indent=2))
"""
    
    try:
        # Execute comprehensive validation
        result = await code_runner.validate_and_execute_code(
            code=test_code,
            language="python",
            dependencies=["requests"],
            timeout=30,
            enable_refinement=True,
            enable_cross_validation=True,
            generate_mcp=True
        )
        
        print(f"Validation result: {result['overall_success']}")
        print(f"Session ID: {result['session_id']}")
        
        if result['mcp_generation']:
            print(f"Generated MCP tool: {result['mcp_generation']['mcp_config']['name']}")
        
        # Print performance statistics
        stats = code_runner.get_performance_statistics()
        print(f"Performance stats: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        print(f"Error in example execution: {e}")


if __name__ == "__main__":
    # Run example if script is executed directly
    asyncio.run(main()) 