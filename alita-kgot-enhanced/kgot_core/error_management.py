#!/usr/bin/env python3
"""
KGoT Error Management and Containment System

Implementation of KGoT's comprehensive error management system as specified in the research paper Section 3.6:
"Ensuring Layered Error Containment & Management"

This module provides:
- Layered error containment with LangChain JSON parsers for syntax detection
- Retry mechanisms (three attempts by default) with unicode escape adjustments
- Comprehensive logging systems for error tracking and analysis
- Python Executor tool containerization for secure code execution with timeouts
- Integration with Alita's iterative refinement and error correction processes
- Support for error recovery procedures maintaining KGoT robustness

Key Components:
1. SyntaxErrorManager: Handles LLM-generated syntax errors using LangChain JSON parsers
2. APIErrorManager: Manages API & system errors with exponential backoff using tenacity
3. PythonExecutorManager: Provides secure containerized Python code execution
4. ErrorRecoveryOrchestrator: Coordinates error recovery with majority voting
5. KGoTErrorManagementSystem: Main orchestrator integrating all error management components

@author: Enhanced Alita KGoT Team
@version: 1.0.0
@based_on: KGoT Research Paper Sections 3.6, 3.7, and B.3
"""

import logging
import json
import time
import asyncio
import docker
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Core dependencies for error management
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_random_exponential, 
    retry_if_exception_type,
    before_sleep_log
)
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.schema import OutputParserException
from langchain_core.exceptions import LangChainException
import pydantic
from pydantic import BaseModel, ValidationError

# Winston-compatible logging setup for KGoT error management
logger = logging.getLogger('KGoTErrorManagement')
handler = logging.FileHandler('./logs/kgot/error_management.log')
formatter = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ErrorType(Enum):
    """
    Classification of error types handled by KGoT error management system
    Based on research paper Section 3.6
    """
    SYNTAX_ERROR = "syntax_error"                    # LLM-generated syntax errors
    API_ERROR = "api_error"                         # OpenAI, external API errors  
    SYSTEM_ERROR = "system_error"                   # System-level failures
    VALIDATION_ERROR = "validation_error"           # Data validation errors
    EXECUTION_ERROR = "execution_error"             # Code execution errors
    TIMEOUT_ERROR = "timeout_error"                 # Operation timeout errors
    SECURITY_ERROR = "security_error"               # Security-related errors


class ErrorSeverity(Enum):
    """Error severity levels for prioritized handling"""
    CRITICAL = "critical"                           # System-breaking errors
    HIGH = "high"                                  # Significant impact errors
    MEDIUM = "medium"                              # Moderate impact errors
    LOW = "low"                                    # Minor errors
    INFO = "info"                                  # Informational errors


@dataclass
class ErrorContext:
    """
    Comprehensive error context information for analysis and recovery
    """
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for logging"""
        return {
            'error_id': self.error_id,
            'error_type': self.error_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'original_operation': self.original_operation,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'metadata': self.metadata
        }


class SyntaxErrorManager:
    """
    Manages LLM-generated syntax errors using LangChain JSON parsers
    
    Implementation based on research paper Section B.3.1:
    "To manage LLM-generated syntax errors, KGoT includes LangChain's JSON parsers that detect
    syntax issues. When a syntax error is detected, the system first attempts to correct it by 
    adjusting the problematic syntax using different encoders, such as the 'unicode escape'."
    """
    
    def __init__(self, llm_client, max_retries: int = 3):
        """
        Initialize Syntax Error Manager
        
        @param {Any} llm_client - LLM client for syntax correction (OpenRouter-based per user memory)
        @param {int} max_retries - Maximum retry attempts (default: 3 as per research paper)
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.unicode_encoders = ['unicode_escape', 'utf-8', 'ascii', 'latin-1']
        
        logger.info("Initialized Syntax Error Manager", extra={
            'operation': 'SYNTAX_ERROR_MANAGER_INIT',
            'max_retries': max_retries,
            'available_encoders': len(self.unicode_encoders)
        })
    
    async def handle_syntax_error(self, 
                                 problematic_content: str, 
                                 operation_context: str,
                                 error_details: str) -> Tuple[str, bool]:
        """
        Handle syntax errors with layered correction approach
        
        @param {str} problematic_content - Content with syntax errors
        @param {str} operation_context - Context of the operation that failed
        @param {str} error_details - Detailed error information
        @returns {Tuple[str, bool]} - (corrected_content, success_flag)
        """
        error_id = f"syntax_{int(time.time())}"
        
        logger.info("Starting syntax error handling", extra={
            'operation': 'SYNTAX_ERROR_HANDLE_START',
            'error_id': error_id,
            'operation_context': operation_context
        })
        
        # Step 1: Try unicode escape corrections (as per research paper)
        for encoder in self.unicode_encoders:
            try:
                corrected_content = self._apply_unicode_correction(problematic_content, encoder)
                if await self._validate_syntax(corrected_content):
                    logger.info("Syntax corrected with unicode encoder", extra={
                        'operation': 'SYNTAX_UNICODE_CORRECTION_SUCCESS',
                        'error_id': error_id,
                        'encoder_used': encoder
                    })
                    return corrected_content, True
            except Exception as e:
                logger.debug(f"Unicode encoder {encoder} failed: {str(e)}")
                continue
        
        # Step 2: LangChain JSON parser approach for structured content
        if self._is_json_content(problematic_content):
            corrected_content = await self._langchain_json_correction(problematic_content, error_id)
            if corrected_content:
                return corrected_content, True
        
        # Step 3: LLM-based syntax correction with retry mechanism
        for attempt in range(self.max_retries):
            try:
                corrected_content = await self._llm_syntax_correction(
                    problematic_content, 
                    operation_context, 
                    error_details,
                    attempt + 1
                )
                
                if await self._validate_syntax(corrected_content):
                    logger.info("Syntax corrected with LLM", extra={
                        'operation': 'SYNTAX_LLM_CORRECTION_SUCCESS',
                        'error_id': error_id,
                        'attempt': attempt + 1
                    })
                    return corrected_content, True
                    
            except Exception as e:
                logger.warning(f"LLM syntax correction attempt {attempt + 1} failed: {str(e)}")
                continue
        
        # Step 4: Log failure for further analysis and bypass
        logger.error("All syntax correction methods failed", extra={
            'operation': 'SYNTAX_CORRECTION_FAILED',
            'error_id': error_id,
            'operation_context': operation_context
        })
        
        return problematic_content, False
    
    def _apply_unicode_correction(self, content: str, encoder: str) -> str:
        """Apply unicode escape corrections as specified in research paper"""
        try:
            if encoder == 'unicode_escape':
                return content.encode('unicode_escape').decode('ascii')
            else:
                return content.encode(encoder, errors='ignore').decode(encoder)
        except Exception as e:
            raise ValueError(f"Unicode correction failed with {encoder}: {str(e)}")
    
    async def _langchain_json_correction(self, content: str, error_id: str) -> Optional[str]:
        """Use LangChain JSON parsers for structured content correction"""
        try:
            # Define a simple output model for JSON validation
            class JSONOutput(BaseModel):
                content: Any
            
            parser = PydanticOutputParser(pydantic_object=JSONOutput)
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_client)
            
            # Attempt to parse and fix JSON content
            parsed_result = await fixing_parser.aparse(content)
            return json.dumps(parsed_result.content, indent=2)
            
        except Exception as e:
            logger.debug(f"LangChain JSON correction failed for {error_id}: {str(e)}")
            return None
    
    async def _llm_syntax_correction(self, 
                                   content: str, 
                                   context: str, 
                                   error_details: str,
                                   attempt: int) -> str:
        """Use LLM to rephrase and regenerate corrected content"""
        correction_prompt = f"""
        You are a syntax error correction assistant. The following content has syntax errors:
        
        ORIGINAL CONTENT:
        {content}
        
        OPERATION CONTEXT:
        {context}
        
        ERROR DETAILS:
        {error_details}
        
        ATTEMPT: {attempt}/{self.max_retries}
        
        Please correct the syntax errors and return only the corrected content.
        Focus on:
        1. JSON syntax validation
        2. Proper escaping of special characters
        3. Correct quotation marks and brackets
        4. Proper encoding handling
        
        CORRECTED CONTENT:
        """
        
        # Use the LLM client to generate corrected content
        response = await self.llm_client.acomplete(correction_prompt)
        return response.text.strip()
    
    def _is_json_content(self, content: str) -> bool:
        """Check if content appears to be JSON"""
        try:
            json.loads(content)
            return True
        except:
            return content.strip().startswith(('{', '['))
    
    async def _validate_syntax(self, content: str) -> bool:
        """Validate if corrected content has proper syntax"""
        try:
            # Basic validation - try to parse as JSON if it looks like JSON
            if self._is_json_content(content):
                json.loads(content)
            return True
        except:
            return False


class APIErrorManager:
    """
    Manages API and system-related errors with exponential backoff
    
    Implementation based on research paper Section B.3.2:
    "API-related errors, such as the OpenAI code '500' errors, are a common challenge...
    the primary strategy employed is exponential backoff...implemented using the tenacity library,
    with a retry policy that waits for random intervals ranging from 1 to 60 seconds and 
    allows for up to six retry attempts."
    """
    
    def __init__(self, max_retries: int = 6, min_wait: int = 1, max_wait: int = 60):
        """
        Initialize API Error Manager with exponential backoff configuration
        
        @param {int} max_retries - Maximum retry attempts (default: 6 as per research paper)
        @param {int} min_wait - Minimum wait time in seconds (default: 1)
        @param {int} max_wait - Maximum wait time in seconds (default: 60)
        """
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.error_stats = defaultdict(int)
        
        logger.info("Initialized API Error Manager", extra={
            'operation': 'API_ERROR_MANAGER_INIT',
            'max_retries': max_retries,
            'wait_range': f"{min_wait}-{max_wait}s"
        })
    
    @retry(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(min=1, max=60),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def execute_with_retry(self, 
                               operation: Callable,
                               operation_name: str,
                               *args, **kwargs) -> Any:
        """
        Execute operation with exponential backoff retry mechanism
        
        @param {Callable} operation - Operation to execute with retry
        @param {str} operation_name - Name of operation for logging
        @param {Any} args - Positional arguments for operation
        @param {Any} kwargs - Keyword arguments for operation
        @returns {Any} - Result of successful operation
        """
        error_id = f"api_{int(time.time())}"
        
        logger.info("Executing operation with retry protection", extra={
            'operation': 'API_OPERATION_RETRY_START',
            'error_id': error_id,
            'operation_name': operation_name
        })
        
        try:
            # Execute the operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            logger.info("Operation completed successfully", extra={
                'operation': 'API_OPERATION_SUCCESS',
                'error_id': error_id,
                'operation_name': operation_name
            })
            
            return result
            
        except Exception as e:
            # Track error statistics
            error_type = type(e).__name__
            self.error_stats[error_type] += 1
            
            logger.error("Operation failed, will retry with exponential backoff", extra={
                'operation': 'API_OPERATION_FAILED',
                'error_id': error_id,
                'operation_name': operation_name,
                'error_type': error_type,
                'error_message': str(e)
            })
            
            # Re-raise for tenacity to handle
            raise
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics for analysis"""
        return {
            'total_errors': sum(self.error_stats.values()),
            'error_breakdown': dict(self.error_stats),
            'most_common_error': max(self.error_stats.items(), key=lambda x: x[1]) if self.error_stats else None,
            'timestamp': datetime.now().isoformat()
        }


class PythonExecutorManager:
    """
    Manages secure containerized Python code execution
    
    Implementation based on research paper Section 3.6:
    "The Python Executor tool, a key component of the system, is containerized to ensure 
    secure execution of LLM-generated code. This tool is designed to run code with strict 
    timeouts and safeguards, preventing potential misuse or resource overconsumption."
    """
    
    def __init__(self, 
                 container_image: str = "python:3.9-slim",
                 default_timeout: int = 30,
                 memory_limit: str = "256m",
                 cpu_limit: str = "0.5"):
        """
        Initialize Python Executor Manager with containerization settings
        
        @param {str} container_image - Docker image for Python execution
        @param {int} default_timeout - Default execution timeout in seconds
        @param {str} memory_limit - Container memory limit
        @param {str} cpu_limit - Container CPU limit
        """
        self.container_image = container_image
        self.default_timeout = default_timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.docker_client = docker.from_env()
        self.active_containers = {}
        
        logger.info("Initialized Python Executor Manager", extra={
            'operation': 'PYTHON_EXECUTOR_INIT',
            'container_image': container_image,
            'default_timeout': default_timeout,
            'memory_limit': memory_limit
        })
    
    async def execute_code_safely(self, 
                                 code: str, 
                                 execution_context: str,
                                 timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute Python code in secure containerized environment
        
        @param {str} code - Python code to execute
        @param {str} execution_context - Context description for logging
        @param {Optional[int]} timeout - Custom timeout for execution
        @returns {Dict[str, Any]} - Execution result with output, errors, and metadata
        """
        execution_id = f"exec_{int(time.time())}"
        timeout = timeout or self.default_timeout
        
        logger.info("Starting secure Python code execution", extra={
            'operation': 'PYTHON_EXECUTION_START',
            'execution_id': execution_id,
            'execution_context': execution_context,
            'timeout': timeout
        })
        
        try:
            # Prepare the execution environment
            container_config = {
                'image': self.container_image,
                'command': ['python', '-c', code],
                'mem_limit': self.memory_limit,
                'cpu_count': float(self.cpu_limit),
                'network_disabled': True,  # Security: disable network access
                'read_only': True,         # Security: read-only filesystem
                'remove': True,            # Auto-cleanup
                'stderr': True,
                'stdout': True
            }
            
            # Execute code in container with timeout
            start_time = time.time()
            container = self.docker_client.containers.run(**container_config, detach=True)
            self.active_containers[execution_id] = container
            
            # Wait for execution with timeout
            try:
                exit_code = container.wait(timeout=timeout)['StatusCode']
                output = container.logs(stdout=True, stderr=False).decode('utf-8')
                errors = container.logs(stdout=False, stderr=True).decode('utf-8')
                execution_time = time.time() - start_time
                
                # Clean up
                if execution_id in self.active_containers:
                    del self.active_containers[execution_id]
                
                result = {
                    'success': exit_code == 0,
                    'exit_code': exit_code,
                    'output': output,
                    'errors': errors,
                    'execution_time': execution_time,
                    'execution_id': execution_id
                }
                
                logger.info("Python code execution completed", extra={
                    'operation': 'PYTHON_EXECUTION_COMPLETE',
                    'execution_id': execution_id,
                    'success': result['success'],
                    'execution_time': execution_time
                })
                
                return result
                
            except Exception as timeout_error:
                # Handle timeout
                container.kill()
                logger.warning("Python execution timed out", extra={
                    'operation': 'PYTHON_EXECUTION_TIMEOUT',
                    'execution_id': execution_id,
                    'timeout': timeout
                })
                
                return {
                    'success': False,
                    'exit_code': -1,
                    'output': '',
                    'errors': f'Execution timed out after {timeout} seconds',
                    'execution_time': timeout,
                    'execution_id': execution_id
                }
                
        except Exception as e:
            logger.error("Python execution failed", extra={
                'operation': 'PYTHON_EXECUTION_FAILED',
                'execution_id': execution_id,
                'error': str(e)
            })
            
            return {
                'success': False,
                'exit_code': -1,
                'output': '',
                'errors': f'Container execution failed: {str(e)}',
                'execution_time': 0,
                'execution_id': execution_id
            }
    
    def cleanup_containers(self):
        """Clean up any remaining active containers"""
        for execution_id, container in list(self.active_containers.items()):
            try:
                container.kill()
                del self.active_containers[execution_id]
                logger.info(f"Cleaned up container {execution_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup container {execution_id}: {str(e)}")


class ErrorRecoveryOrchestrator:
    """
    Orchestrates error recovery using majority voting and iterative refinement
    
    Implementation based on research paper Section 3.5:
    "Robustness is ensured with majority voting, also known as self-consistency...
    we query the LLM multiple times when deciding whether to insert more data into the KG
    or retrieve existing data, when deciding which tool to use, and when parsing the final solution."
    """
    
    def __init__(self, 
                 voting_rounds: int = 3,
                 consensus_threshold: float = 0.6,
                 max_refinement_iterations: int = 3):
        """
        Initialize Error Recovery Orchestrator
        
        @param {int} voting_rounds - Number of voting rounds for majority decision
        @param {float} consensus_threshold - Threshold for consensus (0.0-1.0)
        @param {int} max_refinement_iterations - Maximum iterative refinement attempts
        """
        self.voting_rounds = voting_rounds
        self.consensus_threshold = consensus_threshold
        self.max_refinement_iterations = max_refinement_iterations
        self.recovery_history = []
        
        logger.info("Initialized Error Recovery Orchestrator", extra={
            'operation': 'ERROR_RECOVERY_INIT',
            'voting_rounds': voting_rounds,
            'consensus_threshold': consensus_threshold,
            'max_refinement_iterations': max_refinement_iterations
        })
    
    async def execute_with_majority_voting(self,
                                         operation: Callable,
                                         operation_name: str,
                                         *args, **kwargs) -> Tuple[Any, float]:
        """
        Execute operation with majority voting for robustness
        
        @param {Callable} operation - Operation to execute with voting
        @param {str} operation_name - Name of operation for logging
        @param {Any} args - Positional arguments for operation
        @param {Any} kwargs - Keyword arguments for operation
        @returns {Tuple[Any, float]} - (result, confidence_score)
        """
        recovery_id = f"recovery_{int(time.time())}"
        
        logger.info("Starting majority voting execution", extra={
            'operation': 'MAJORITY_VOTING_START',
            'recovery_id': recovery_id,
            'operation_name': operation_name,
            'voting_rounds': self.voting_rounds
        })
        
        results = []
        errors = []
        
        # Execute operation multiple times for voting
        for round_num in range(self.voting_rounds):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                results.append(result)
                
                logger.debug(f"Voting round {round_num + 1} completed", extra={
                    'operation': 'VOTING_ROUND_COMPLETE',
                    'recovery_id': recovery_id,
                    'round': round_num + 1
                })
                
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"Voting round {round_num + 1} failed: {str(e)}")
        
        # Analyze results for majority consensus
        if not results:
            raise Exception(f"All voting rounds failed: {'; '.join(errors)}")
        
        # Find majority consensus
        majority_result, confidence = self._find_majority_consensus(results)
        
        logger.info("Majority voting completed", extra={
            'operation': 'MAJORITY_VOTING_COMPLETE',
            'recovery_id': recovery_id,
            'successful_rounds': len(results),
            'total_rounds': self.voting_rounds,
            'confidence': confidence
        })
        
        return majority_result, confidence
    
    async def iterative_error_refinement(self,
                                       failed_operation: Callable,
                                       error_context: ErrorContext,
                                       refinement_strategy: Callable) -> Tuple[Any, bool]:
        """
        Perform iterative refinement to recover from errors
        
        @param {Callable} failed_operation - Operation that failed
        @param {ErrorContext} error_context - Context of the error
        @param {Callable} refinement_strategy - Strategy for refining the operation
        @returns {Tuple[Any, bool]} - (result, success_flag)
        """
        recovery_id = f"refinement_{int(time.time())}"
        
        logger.info("Starting iterative error refinement", extra={
            'operation': 'ITERATIVE_REFINEMENT_START',
            'recovery_id': recovery_id,
            'error_id': error_context.error_id,
            'max_iterations': self.max_refinement_iterations
        })
        
        current_context = error_context
        
        for iteration in range(self.max_refinement_iterations):
            try:
                # Apply refinement strategy
                refined_params = await refinement_strategy(current_context, iteration)
                
                # Retry operation with refined parameters
                if asyncio.iscoroutinefunction(failed_operation):
                    result = await failed_operation(**refined_params)
                else:
                    result = failed_operation(**refined_params)
                
                logger.info("Iterative refinement succeeded", extra={
                    'operation': 'ITERATIVE_REFINEMENT_SUCCESS',
                    'recovery_id': recovery_id,
                    'iteration': iteration + 1,
                    'error_id': error_context.error_id
                })
                
                # Record successful recovery
                self.recovery_history.append({
                    'recovery_id': recovery_id,
                    'error_context': current_context.to_dict(),
                    'successful_iteration': iteration + 1,
                    'timestamp': datetime.now().isoformat()
                })
                
                return result, True
                
            except Exception as e:
                # Update error context for next iteration
                current_context.retry_count = iteration + 1
                current_context.recovery_attempts.append({
                    'iteration': iteration + 1,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.warning(f"Refinement iteration {iteration + 1} failed: {str(e)}")
        
        # All refinement attempts failed
        logger.error("All iterative refinement attempts failed", extra={
            'operation': 'ITERATIVE_REFINEMENT_FAILED',
            'recovery_id': recovery_id,
            'error_id': error_context.error_id,
            'total_iterations': self.max_refinement_iterations
        })
        
        return None, False
    
    def _find_majority_consensus(self, results: List[Any]) -> Tuple[Any, float]:
        """Find majority consensus from voting results"""
        if not results:
            raise ValueError("No results to analyze for consensus")
        
        # Convert results to strings for comparison
        result_strings = [str(result) for result in results]
        result_counts = Counter(result_strings)
        
        # Find most common result
        most_common = result_counts.most_common(1)[0]
        most_common_result_str, count = most_common
        
        # Calculate confidence
        confidence = count / len(results)
        
        # Find original result object corresponding to most common string
        for i, result_str in enumerate(result_strings):
            if result_str == most_common_result_str:
                majority_result = results[i]
                break
        else:
            majority_result = most_common_result_str
        
        return majority_result, confidence


class KGoTErrorManagementSystem:
    """
    Main orchestrator for KGoT's comprehensive error management system
    
    Integrates all error management components and provides unified interface
    for error handling throughout the KGoT architecture
    """
    
    def __init__(self, llm_client, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the comprehensive KGoT Error Management System
        
        @param {Any} llm_client - LLM client for error correction
        @param {Optional[Dict[str, Any]]} config - Configuration overrides
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        # Initialize all error management components
        self.syntax_manager = SyntaxErrorManager(
            llm_client=llm_client,
            max_retries=self.config.get('syntax_max_retries', 3)
        )
        
        self.api_manager = APIErrorManager(
            max_retries=self.config.get('api_max_retries', 6),
            min_wait=self.config.get('api_min_wait', 1),
            max_wait=self.config.get('api_max_wait', 60)
        )
        
        self.executor_manager = PythonExecutorManager(
            container_image=self.config.get('container_image', 'python:3.9-slim'),
            default_timeout=self.config.get('execution_timeout', 30),
            memory_limit=self.config.get('memory_limit', '256m'),
            cpu_limit=self.config.get('cpu_limit', '0.5')
        )
        
        self.recovery_orchestrator = ErrorRecoveryOrchestrator(
            voting_rounds=self.config.get('voting_rounds', 3),
            consensus_threshold=self.config.get('consensus_threshold', 0.6),
            max_refinement_iterations=self.config.get('max_refinement_iterations', 3)
        )
        
        # Error management statistics
        self.error_statistics = {
            'total_errors_handled': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'error_types': defaultdict(int),
            'recovery_methods': defaultdict(int)
        }
        
        logger.info("Initialized KGoT Error Management System", extra={
            'operation': 'KGOT_ERROR_MANAGEMENT_INIT',
            'components': [
                'SyntaxErrorManager',
                'APIErrorManager', 
                'PythonExecutorManager',
                'ErrorRecoveryOrchestrator'
            ]
        })
    
    async def handle_error(self, 
                          error: Exception,
                          operation_context: str,
                          error_type: Optional[ErrorType] = None,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Tuple[Any, bool]:
        """
        Unified error handling entry point
        
        @param {Exception} error - The error to handle
        @param {str} operation_context - Context where error occurred
        @param {Optional[ErrorType]} error_type - Type of error (auto-detected if None)
        @param {ErrorSeverity} severity - Error severity level
        @returns {Tuple[Any, bool]} - (recovery_result, success_flag)
        """
        error_id = f"error_{int(time.time())}"
        
        # Auto-detect error type if not provided
        if error_type is None:
            error_type = self._classify_error(error)
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            error_type=error_type,
            severity=severity,
            timestamp=datetime.now(),
            original_operation=operation_context,
            error_message=str(error),
            stack_trace=str(error.__traceback__) if hasattr(error, '__traceback__') else None
        )
        
        # Update statistics
        self.error_statistics['total_errors_handled'] += 1
        self.error_statistics['error_types'][error_type.value] += 1
        
        logger.error("Handling error with KGoT Error Management System", extra={
            'operation': 'KGOT_ERROR_HANDLE',
            'error_id': error_id,
            'error_type': error_type.value,
            'severity': severity.value,
            'context': operation_context
        })
        
        # Route to appropriate error handler
        try:
            if error_type == ErrorType.SYNTAX_ERROR:
                result = await self._handle_syntax_error(error, error_context)
            elif error_type in [ErrorType.API_ERROR, ErrorType.SYSTEM_ERROR]:
                result = await self._handle_api_system_error(error, error_context)
            elif error_type == ErrorType.EXECUTION_ERROR:
                result = await self._handle_execution_error(error, error_context)
            else:
                result = await self._handle_generic_error(error, error_context)
            
            if result[1]:  # If successful
                self.error_statistics['successful_recoveries'] += 1
                logger.info("Error successfully recovered", extra={
                    'operation': 'KGOT_ERROR_RECOVERY_SUCCESS',
                    'error_id': error_id
                })
            else:
                self.error_statistics['failed_recoveries'] += 1
                logger.error("Error recovery failed", extra={
                    'operation': 'KGOT_ERROR_RECOVERY_FAILED',
                    'error_id': error_id
                })
            
            return result
            
        except Exception as recovery_error:
            self.error_statistics['failed_recoveries'] += 1
            logger.critical("Error recovery process itself failed", extra={
                'operation': 'KGOT_ERROR_RECOVERY_CRITICAL',
                'error_id': error_id,
                'recovery_error': str(recovery_error)
            })
            return None, False
    
    async def _handle_syntax_error(self, error: Exception, context: ErrorContext) -> Tuple[Any, bool]:
        """Handle syntax errors using SyntaxErrorManager"""
        self.error_statistics['recovery_methods']['syntax_correction'] += 1
        
        # Extract problematic content from error
        problematic_content = getattr(error, 'problematic_content', str(error))
        
        result, success = await self.syntax_manager.handle_syntax_error(
            problematic_content=problematic_content,
            operation_context=context.original_operation,
            error_details=context.error_message
        )
        
        return result, success
    
    async def _handle_api_system_error(self, error: Exception, context: ErrorContext) -> Tuple[Any, bool]:
        """Handle API/system errors using APIErrorManager with exponential backoff"""
        self.error_statistics['recovery_methods']['api_retry'] += 1
        
        # Create a mock operation that would have succeeded
        async def mock_recovery_operation():
            # In real implementation, this would be the original failed operation
            return f"Recovered from {context.error_type.value}: {context.error_message}"
        
        try:
            result = await self.api_manager.execute_with_retry(
                operation=mock_recovery_operation,
                operation_name=context.original_operation
            )
            return result, True
        except Exception:
            return None, False
    
    async def _handle_execution_error(self, error: Exception, context: ErrorContext) -> Tuple[Any, bool]:
        """Handle execution errors using secure containerized execution"""
        self.error_statistics['recovery_methods']['safe_execution'] += 1
        
        # Extract code from error context if available
        code = getattr(error, 'code', 'print("Error recovery execution")')
        
        result = await self.executor_manager.execute_code_safely(
            code=code,
            execution_context=context.original_operation
        )
        
        return result, result['success']
    
    async def _handle_generic_error(self, error: Exception, context: ErrorContext) -> Tuple[Any, bool]:
        """Handle generic errors using majority voting and iterative refinement"""
        self.error_statistics['recovery_methods']['majority_voting'] += 1
        
        # Create mock operation for demonstration
        async def recovery_operation():
            return f"Generic recovery for {context.error_type.value}"
        
        try:
            result, confidence = await self.recovery_orchestrator.execute_with_majority_voting(
                operation=recovery_operation,
                operation_name=context.original_operation
            )
            
            # Consider successful if confidence is above threshold
            success = confidence >= self.recovery_orchestrator.consensus_threshold
            return result, success
            
        except Exception:
            return None, False
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Auto-classify error type based on error characteristics"""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        if 'syntax' in error_str or 'json' in error_str or isinstance(error, (SyntaxError, json.JSONDecodeError)):
            return ErrorType.SYNTAX_ERROR
        elif 'api' in error_str or 'connection' in error_str or 'timeout' in error_str:
            return ErrorType.API_ERROR
        elif 'validation' in error_str or isinstance(error, ValidationError):
            return ErrorType.VALIDATION_ERROR
        elif 'execution' in error_str or 'runtime' in error_str:
            return ErrorType.EXECUTION_ERROR
        elif 'timeout' in error_str or isinstance(error, TimeoutError):
            return ErrorType.TIMEOUT_ERROR
        elif 'security' in error_str or 'permission' in error_str:
            return ErrorType.SECURITY_ERROR
        else:
            return ErrorType.SYSTEM_ERROR
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error management statistics"""
        api_stats = self.api_manager.get_error_statistics()
        
        return {
            'kgot_error_management': self.error_statistics,
            'api_error_stats': api_stats,
            'recovery_history_count': len(self.recovery_orchestrator.recovery_history),
            'active_containers': len(self.executor_manager.active_containers),
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Cleanup all error management resources"""
        logger.info("Cleaning up KGoT Error Management System")
        
        # Cleanup containers
        self.executor_manager.cleanup_containers()
        
        # Log final statistics
        final_stats = self.get_comprehensive_statistics()
        logger.info("KGoT Error Management System cleanup completed", extra={
            'operation': 'KGOT_ERROR_MANAGEMENT_CLEANUP',
            'final_statistics': final_stats
        })


# Factory function for easy initialization
def create_kgot_error_management_system(llm_client, config: Optional[Dict[str, Any]] = None) -> KGoTErrorManagementSystem:
    """
    Factory function to create KGoT Error Management System
    
    @param {Any} llm_client - LLM client for error correction
    @param {Optional[Dict[str, Any]]} config - Configuration overrides
    @returns {KGoTErrorManagementSystem} - Initialized error management system
    """
    return KGoTErrorManagementSystem(llm_client=llm_client, config=config)


# Example usage and testing
if __name__ == "__main__":
    async def test_error_management():
        """Test the KGoT Error Management System"""
        # Mock LLM client for testing
        class MockLLMClient:
            async def acomplete(self, prompt: str):
                class MockResponse:
                    text = '{"corrected": "content"}'
                return MockResponse()
        
        # Initialize error management system
        error_system = create_kgot_error_management_system(
            llm_client=MockLLMClient(),
            config={
                'syntax_max_retries': 2,
                'api_max_retries': 3,
                'voting_rounds': 2
            }
        )
        
        # Test different error types
        test_errors = [
            (SyntaxError("Invalid JSON syntax"), "JSON parsing operation"),
            (ConnectionError("API connection failed"), "External API call"),
            (RuntimeError("Code execution failed"), "Python script execution")
        ]
        
        for error, context in test_errors:
            try:
                result, success = await error_system.handle_error(error, context)
                print(f"Handled {type(error).__name__}: Success={success}")
            except Exception as e:
                print(f"Error handling failed: {str(e)}")
        
        # Print statistics
        stats = error_system.get_comprehensive_statistics()
        print(f"Error Management Statistics: {json.dumps(stats, indent=2)}")
        
        # Cleanup
        error_system.cleanup()
    
    # Run test if executed directly
    asyncio.run(test_error_management()) 