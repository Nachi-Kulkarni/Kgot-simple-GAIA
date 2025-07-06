"""
Sequential Thinking MCP Integration for Unified System Controller

This module provides the Sequential Thinking MCP integration specifically designed
for the Unified System Controller to handle intelligent task routing and system
coordination between Alita and KGoT systems.

Key Features:
- Task complexity assessment and routing decision support
- System coordination strategy determination
- Error resolution and fallback planning
- Performance optimization recommendations
- Cross-system state management insights

@module SequentialThinkingMCPIntegration
@author AI Assistant
@date 2025-01-22
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict

import httpx

# Import logging configuration [[memory:1383804]]
from ..config.logging.winston_config import get_logger

# Create logger instance
logger = get_logger('sequential_thinking_mcp')


class ThinkingMode(Enum):
    """Sequential thinking mode enumeration"""
    TASK_ROUTING = "task_routing"
    SYSTEM_COORDINATION = "system_coordination"
    ERROR_RESOLUTION = "error_resolution"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    FALLBACK_PLANNING = "fallback_planning"


@dataclass
class ThinkingRequest:
    """Request structure for Sequential Thinking MCP"""
    task_description: str
    mode: ThinkingMode
    context: Dict[str, Any]
    max_thoughts: int = 10
    timeout_seconds: int = 30
    require_final_answer: bool = True


@dataclass
class ThinkingStep:
    """Individual thinking step result"""
    thought_number: int
    content: str
    reasoning: str
    confidence: float
    insights: List[str]
    recommendations: List[str]
    timestamp: datetime


@dataclass
class ThinkingResult:
    """Complete Sequential Thinking result"""
    request_id: str
    mode: ThinkingMode
    steps: List[ThinkingStep]
    final_answer: str
    confidence: float
    routing_recommendation: Optional[str]
    system_recommendations: Dict[str, Any]
    performance_insights: Dict[str, Any]
    execution_time_ms: float
    success: bool
    error: Optional[str] = None


class SequentialThinkingMCPIntegration:
    """
    Sequential Thinking MCP integration for the Unified System Controller
    
    This class provides sophisticated reasoning capabilities for:
    - Intelligent task routing between Alita and KGoT systems
    - System coordination strategy planning
    - Error resolution and fallback planning
    - Performance optimization recommendations
    
    The integration communicates with the Sequential Thinking MCP via HTTP API
    and provides specialized prompts and analysis for system orchestration tasks.
    """
    
    def __init__(self, 
                 mcp_endpoint: str = "http://localhost:3000/sequential-thinking",
                 timeout_seconds: int = 60,
                 max_retries: int = 3):
        """
        Initialize Sequential Thinking MCP integration
        
        Args:
            mcp_endpoint: HTTP endpoint for Sequential Thinking MCP
            timeout_seconds: Default timeout for thinking requests
            max_retries: Maximum retry attempts for failed requests
        """
        self.mcp_endpoint = mcp_endpoint
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # HTTP client for MCP communication
        self.http_client = httpx.AsyncClient(timeout=timeout_seconds)
        
        # Thinking templates for different modes
        self.thinking_templates = self._initialize_thinking_templates()
        
        # Performance tracking
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "mode_usage": {mode.value: 0 for mode in ThinkingMode}
        }
        
        logger.info("Sequential Thinking MCP Integration initialized", extra={
            'operation': 'SEQUENTIAL_THINKING_INIT',
            'mcp_endpoint': mcp_endpoint,
            'timeout_seconds': timeout_seconds
        })
    
    def _initialize_thinking_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize thinking templates for different modes"""
        return {
            ThinkingMode.TASK_ROUTING.value: {
                "name": "Task Routing Analysis",
                "description": "Determine optimal routing strategy for task execution",
                "system_prompt": """You are an expert system orchestrator analyzing tasks to determine the optimal routing strategy between Alita and KGoT systems.

Alita System Capabilities:
- Tool creation and MCP generation
- Web interaction and automation
- Code generation and execution
- External system integration

KGoT System Capabilities:
- Knowledge graph reasoning
- Complex analytical processing
- Pattern recognition and inference
- Structured knowledge management

Your goal is to analyze the task and recommend the best routing strategy:
- ALITA_FIRST: Start with Alita system for tool creation/web tasks
- KGOT_FIRST: Start with KGoT system for reasoning/analysis tasks
- HYBRID: Sequential execution using both systems
- PARALLEL: Concurrent execution on both systems

Consider system health, task complexity, and performance requirements.""",
                "thinking_focus": "routing_optimization",
                "expected_output": "routing_strategy_recommendation"
            },
            
            ThinkingMode.SYSTEM_COORDINATION.value: {
                "name": "System Coordination Planning",
                "description": "Plan coordination strategy between multiple systems",
                "system_prompt": """You are a system coordination specialist planning how multiple AI systems should work together effectively.

Available Systems:
- Alita Manager Agent (tool creation, web interaction)
- KGoT Controller (knowledge reasoning, graph processing)
- Validation Services (quality assurance)
- Multimodal Processors (vision, audio, text)

Your goal is to create an optimal coordination plan that considers:
- Data flow between systems
- Resource allocation and timing
- Error handling and fallback strategies
- Performance optimization opportunities
- State management and synchronization

Provide a detailed coordination strategy with specific steps.""",
                "thinking_focus": "coordination_planning",
                "expected_output": "coordination_strategy"
            },
            
            ThinkingMode.ERROR_RESOLUTION.value: {
                "name": "Error Resolution Strategy",
                "description": "Develop systematic error resolution approach",
                "system_prompt": """You are an expert error resolution specialist developing strategies for complex system failures.

Consider these error types:
- System unavailability (circuit breaker open)
- Performance degradation (slow response times)
- Resource exhaustion (memory, CPU, budget)
- Integration failures (API errors, timeouts)
- Data inconsistency (state synchronization issues)

Your goal is to create a comprehensive error resolution strategy that includes:
- Root cause analysis approach
- Immediate mitigation steps
- System recovery procedures
- Prevention measures for future occurrences
- Fallback alternatives

Prioritize system stability and user experience.""",
                "thinking_focus": "error_resolution",
                "expected_output": "resolution_strategy"
            },
            
            ThinkingMode.PERFORMANCE_OPTIMIZATION.value: {
                "name": "Performance Optimization Analysis",
                "description": "Analyze and optimize system performance",
                "system_prompt": """You are a performance optimization expert analyzing system efficiency and recommending improvements.

Performance Dimensions:
- Response time and latency
- Resource utilization (CPU, memory, network)
- Throughput and scalability
- Cost efficiency
- User experience metrics

Consider these optimization strategies:
- Load balancing and distribution
- Caching and data optimization
- Parallel processing opportunities
- Resource allocation adjustments
- Circuit breaker and timeout tuning

Provide specific, actionable optimization recommendations.""",
                "thinking_focus": "performance_optimization",
                "expected_output": "optimization_recommendations"
            },
            
            ThinkingMode.FALLBACK_PLANNING.value: {
                "name": "Fallback Strategy Planning",
                "description": "Develop robust fallback and contingency plans",
                "system_prompt": """You are a resilience engineer designing fallback strategies for system failures.

Failure Scenarios to Consider:
- Primary system complete failure
- Partial system degradation
- Network connectivity issues
- Resource constraints (budget, compute)
- Data corruption or inconsistency

Design fallback strategies that ensure:
- Graceful degradation of functionality
- User experience preservation
- Data integrity maintenance
- Quick recovery capability
- Minimal disruption to ongoing operations

Provide detailed fallback plans with clear triggers and procedures.""",
                "thinking_focus": "fallback_planning",
                "expected_output": "fallback_strategy"
            }
        }
    
    async def analyze_task_routing(self, 
                                 task_description: str,
                                 task_context: Dict[str, Any],
                                 system_health: Dict[str, Any]) -> ThinkingResult:
        """
        Analyze task to determine optimal routing strategy
        
        Args:
            task_description: Description of the task to route
            task_context: Context information about the task
            system_health: Current health status of available systems
            
        Returns:
            ThinkingResult with routing recommendations
        """
        context = {
            "task_description": task_description,
            "task_context": task_context,
            "system_health": system_health,
            "available_systems": ["alita", "kgot"],
            "routing_options": ["ALITA_FIRST", "KGOT_FIRST", "HYBRID", "PARALLEL"]
        }
        
        request = ThinkingRequest(
            task_description=task_description,
            mode=ThinkingMode.TASK_ROUTING,
            context=context,
            max_thoughts=8
        )
        
        return await self._execute_thinking_request(request)
    
    async def plan_system_coordination(self,
                                     systems_involved: List[str],
                                     coordination_context: Dict[str, Any]) -> ThinkingResult:
        """
        Plan coordination strategy between multiple systems
        
        Args:
            systems_involved: List of systems that need coordination
            coordination_context: Context for coordination planning
            
        Returns:
            ThinkingResult with coordination strategy
        """
        context = {
            "systems_involved": systems_involved,
            "coordination_context": coordination_context,
            "coordination_types": ["sequential", "parallel", "hybrid", "conditional"]
        }
        
        request = ThinkingRequest(
            task_description=f"Plan coordination between systems: {', '.join(systems_involved)}",
            mode=ThinkingMode.SYSTEM_COORDINATION,
            context=context,
            max_thoughts=10
        )
        
        return await self._execute_thinking_request(request)
    
    async def resolve_system_errors(self,
                                  error_context: Dict[str, Any],
                                  failed_systems: List[str]) -> ThinkingResult:
        """
        Develop error resolution strategy for system failures
        
        Args:
            error_context: Information about the errors encountered
            failed_systems: List of systems that have failed
            
        Returns:
            ThinkingResult with error resolution strategy
        """
        context = {
            "error_context": error_context,
            "failed_systems": failed_systems,
            "resolution_types": ["restart", "fallback", "bypass", "manual_intervention"]
        }
        
        request = ThinkingRequest(
            task_description=f"Resolve errors in systems: {', '.join(failed_systems)}",
            mode=ThinkingMode.ERROR_RESOLUTION,
            context=context,
            max_thoughts=12
        )
        
        return await self._execute_thinking_request(request)
    
    async def optimize_system_performance(self,
                                        performance_metrics: Dict[str, Any],
                                        optimization_goals: Dict[str, Any]) -> ThinkingResult:
        """
        Analyze performance and recommend optimizations
        
        Args:
            performance_metrics: Current system performance data
            optimization_goals: Target performance goals
            
        Returns:
            ThinkingResult with optimization recommendations
        """
        context = {
            "performance_metrics": performance_metrics,
            "optimization_goals": optimization_goals,
            "optimization_strategies": ["caching", "load_balancing", "resource_scaling", "algorithm_optimization"]
        }
        
        request = ThinkingRequest(
            task_description="Optimize system performance based on current metrics",
            mode=ThinkingMode.PERFORMANCE_OPTIMIZATION,
            context=context,
            max_thoughts=10
        )
        
        return await self._execute_thinking_request(request)
    
    async def plan_fallback_strategy(self,
                                   primary_plan: Dict[str, Any],
                                   risk_factors: List[str]) -> ThinkingResult:
        """
        Develop fallback strategy for primary plan failures
        
        Args:
            primary_plan: The primary execution plan
            risk_factors: Identified risk factors that could cause failures
            
        Returns:
            ThinkingResult with fallback strategy
        """
        context = {
            "primary_plan": primary_plan,
            "risk_factors": risk_factors,
            "fallback_types": ["system_substitution", "feature_reduction", "manual_override", "abort_gracefully"]
        }
        
        request = ThinkingRequest(
            task_description="Develop comprehensive fallback strategy for plan execution",
            mode=ThinkingMode.FALLBACK_PLANNING,
            context=context,
            max_thoughts=8
        )
        
        return await self._execute_thinking_request(request)
    
    async def _execute_thinking_request(self, request: ThinkingRequest) -> ThinkingResult:
        """
        Execute a thinking request with the Sequential Thinking MCP
        
        Args:
            request: The thinking request to execute
            
        Returns:
            ThinkingResult with the analysis results
        """
        request_id = f"thinking_{int(time.time() * 1000)}"
        start_time = time.time()
        
        logger.info(f"Executing Sequential Thinking request: {request.mode.value}", extra={
            'operation': 'SEQUENTIAL_THINKING_REQUEST',
            'request_id': request_id,
            'mode': request.mode.value,
            'max_thoughts': request.max_thoughts
        })
        
        # Update usage statistics
        self.request_stats["total_requests"] += 1
        self.request_stats["mode_usage"][request.mode.value] += 1
        
        try:
            # Prepare the request payload
            template = self.thinking_templates[request.mode.value]
            payload = {
                "task_description": request.task_description,
                "system_prompt": template["system_prompt"],
                "context": request.context,
                "max_thoughts": request.max_thoughts,
                "thinking_focus": template["thinking_focus"],
                "require_final_answer": request.require_final_answer,
                "request_id": request_id
            }
            
            # Execute the request with retries
            response_data = await self._execute_with_retries(payload, request.timeout_seconds)
            
            # Parse the response
            thinking_result = self._parse_thinking_response(
                request_id, request.mode, response_data, start_time
            )
            
            # Update success statistics
            self.request_stats["successful_requests"] += 1
            execution_time = time.time() - start_time
            self._update_average_response_time(execution_time)
            
            logger.info(f"Sequential Thinking completed: {request_id}", extra={
                'operation': 'SEQUENTIAL_THINKING_SUCCESS',
                'request_id': request_id,
                'execution_time_ms': thinking_result.execution_time_ms,
                'confidence': thinking_result.confidence,
                'steps_count': len(thinking_result.steps)
            })
            
            return thinking_result
            
        except Exception as e:
            # Update failure statistics
            self.request_stats["failed_requests"] += 1
            execution_time = (time.time() - start_time) * 1000
            
            logger.error(f"Sequential Thinking failed: {request_id} - {str(e)}", extra={
                'operation': 'SEQUENTIAL_THINKING_ERROR',
                'request_id': request_id,
                'error': str(e),
                'execution_time_ms': execution_time
            })
            
            # Return failed result
            return ThinkingResult(
                request_id=request_id,
                mode=request.mode,
                steps=[],
                final_answer="",
                confidence=0.0,
                routing_recommendation=None,
                system_recommendations={},
                performance_insights={},
                execution_time_ms=execution_time,
                success=False,
                error=str(e)
            )
    
    async def _execute_with_retries(self, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Execute HTTP request with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.http_client.post(
                    self.mcp_endpoint,
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Sequential Thinking attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All Sequential Thinking attempts failed: {str(e)}")
        
        raise last_exception
    
    def _parse_thinking_response(self, 
                               request_id: str,
                               mode: ThinkingMode,
                               response_data: Dict[str, Any],
                               start_time: float) -> ThinkingResult:
        """Parse Sequential Thinking MCP response into ThinkingResult"""
        execution_time = (time.time() - start_time) * 1000
        
        # Extract thinking steps
        steps = []
        raw_thoughts = response_data.get("thoughts", [])
        
        for i, thought_data in enumerate(raw_thoughts):
            step = ThinkingStep(
                thought_number=i + 1,
                content=thought_data.get("content", ""),
                reasoning=thought_data.get("reasoning", ""),
                confidence=thought_data.get("confidence", 0.5),
                insights=thought_data.get("insights", []),
                recommendations=thought_data.get("recommendations", []),
                timestamp=datetime.now()
            )
            steps.append(step)
        
        # Extract final answer and recommendations
        final_answer = response_data.get("final_answer", "")
        confidence = response_data.get("confidence", 0.5)
        
        # Parse mode-specific recommendations
        routing_recommendation = None
        system_recommendations = {}
        performance_insights = {}
        
        if mode == ThinkingMode.TASK_ROUTING:
            routing_recommendation = self._extract_routing_recommendation(final_answer, steps)
            system_recommendations = self._extract_system_recommendations(response_data)
        elif mode == ThinkingMode.SYSTEM_COORDINATION:
            system_recommendations = self._extract_coordination_strategy(response_data)
        elif mode == ThinkingMode.ERROR_RESOLUTION:
            system_recommendations = self._extract_resolution_strategy(response_data)
        elif mode == ThinkingMode.PERFORMANCE_OPTIMIZATION:
            performance_insights = self._extract_performance_insights(response_data)
            system_recommendations = self._extract_optimization_recommendations(response_data)
        elif mode == ThinkingMode.FALLBACK_PLANNING:
            system_recommendations = self._extract_fallback_strategy(response_data)
        
        return ThinkingResult(
            request_id=request_id,
            mode=mode,
            steps=steps,
            final_answer=final_answer,
            confidence=confidence,
            routing_recommendation=routing_recommendation,
            system_recommendations=system_recommendations,
            performance_insights=performance_insights,
            execution_time_ms=execution_time,
            success=True
        )
    
    def _extract_routing_recommendation(self, final_answer: str, steps: List[ThinkingStep]) -> Optional[str]:
        """Extract routing recommendation from thinking results"""
        text_to_analyze = final_answer + " " + " ".join([step.content for step in steps])
        text_lower = text_to_analyze.lower()
        
        if "hybrid" in text_lower or "both systems" in text_lower:
            return "HYBRID"
        elif "parallel" in text_lower or "concurrent" in text_lower:
            return "PARALLEL"
        elif "kgot" in text_lower or "knowledge graph" in text_lower or "reasoning first" in text_lower:
            return "KGOT_FIRST"
        elif "alita" in text_lower or "tool creation" in text_lower or "mcp first" in text_lower:
            return "ALITA_FIRST"
        
        return None
    
    def _extract_system_recommendations(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract general system recommendations"""
        return response_data.get("system_recommendations", {})
    
    def _extract_coordination_strategy(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract coordination strategy from response"""
        return response_data.get("coordination_strategy", {})
    
    def _extract_resolution_strategy(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract error resolution strategy from response"""
        return response_data.get("resolution_strategy", {})
    
    def _extract_performance_insights(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance insights from response"""
        return response_data.get("performance_insights", {})
    
    def _extract_optimization_recommendations(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract optimization recommendations from response"""
        return response_data.get("optimization_recommendations", {})
    
    def _extract_fallback_strategy(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fallback strategy from response"""
        return response_data.get("fallback_strategy", {})
    
    def _update_average_response_time(self, execution_time: float) -> None:
        """Update average response time statistics"""
        total_successful = self.request_stats["successful_requests"]
        current_avg = self.request_stats["average_response_time"]
        
        if total_successful == 1:
            self.request_stats["average_response_time"] = execution_time
        else:
            self.request_stats["average_response_time"] = (
                (current_avg * (total_successful - 1) + execution_time) / total_successful
            )
    
    async def close(self) -> None:
        """Close the HTTP client and cleanup resources"""
        await self.http_client.aclose()
        
        logger.info("Sequential Thinking MCP Integration closed", extra={
            'operation': 'SEQUENTIAL_THINKING_CLOSE',
            'total_requests': self.request_stats["total_requests"],
            'success_rate': (
                self.request_stats["successful_requests"] / max(1, self.request_stats["total_requests"])
            ) * 100
        })
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for the Sequential Thinking integration"""
        total_requests = self.request_stats["total_requests"]
        success_rate = 0.0
        if total_requests > 0:
            success_rate = (self.request_stats["successful_requests"] / total_requests) * 100
        
        return {
            "total_requests": total_requests,
            "successful_requests": self.request_stats["successful_requests"],
            "failed_requests": self.request_stats["failed_requests"],
            "success_rate_percent": success_rate,
            "average_response_time_seconds": self.request_stats["average_response_time"],
            "mode_usage": self.request_stats["mode_usage"].copy()
        } 