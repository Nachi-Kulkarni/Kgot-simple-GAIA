#!/usr/bin/env python3
"""
LangChain Sequential Manager Agent for Alita-KGoT Enhanced System

This module implements the core LangChain-based manager agent with sequential thinking
integration as the primary reasoning engine. It provides advanced memory management,
conversation history tracking, and orchestration capabilities for complex multi-step
operations spanning both Alita and KGoT systems.

Features:
- LangChain AgentExecutor with OpenRouter integration
- Sequential thinking MCP as primary reasoning tool
- Memory and context management for multi-step operations
- Conversation history tracking for thinking processes
- Agent workflow: complexity assessment → sequential thinking → system coordination → validation
- Integration with existing JavaScript components via REST APIs

@module LangChainSequentialManager
@author Alita-KGoT Development Team
@version 1.0.0
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, tool
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler


class ComplexityLevel(Enum):
    """Enumeration for task complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemType(Enum):
    """Enumeration for system types in the architecture"""
    ALITA = "alita"
    KGOT = "kgot"
    BOTH = "both"
    VALIDATION = "validation"
    MULTIMODAL = "multimodal"


@dataclass
class ConversationContext:
    """
    Data class for managing conversation context and memory
    
    Attributes:
        conversation_id: Unique identifier for the conversation
        user_id: Identifier for the user
        session_start: Timestamp when the session started
        last_activity: Timestamp of last activity
        message_count: Number of messages in the conversation
        total_tokens: Total tokens used in the conversation
        complexity_history: History of complexity assessments
        thinking_sessions: Active sequential thinking sessions
        system_interactions: History of system interactions
        context_summary: Summary of conversation context
    """
    conversation_id: str
    user_id: str
    session_start: datetime
    last_activity: datetime
    message_count: int = 0
    total_tokens: int = 0
    complexity_history: List[Dict[str, Any]] = None
    thinking_sessions: List[str] = None
    system_interactions: List[Dict[str, Any]] = None
    context_summary: str = ""
    
    def __post_init__(self):
        if self.complexity_history is None:
            self.complexity_history = []
        if self.thinking_sessions is None:
            self.thinking_sessions = []
        if self.system_interactions is None:
            self.system_interactions = []


@dataclass
class SequentialThinkingSession:
    """
    Data class for tracking sequential thinking sessions
    
    Attributes:
        session_id: Unique identifier for the thinking session
        conversation_id: Associated conversation ID
        task_description: Description of the task being processed
        complexity_score: Calculated complexity score
        template_used: Thought process template used
        start_time: Session start timestamp
        end_time: Session end timestamp (None if active)
        thought_steps: List of thought steps taken
        conclusions: Final conclusions from thinking process
        system_recommendations: Recommendations for system coordination
        status: Current status of the session
    """
    session_id: str
    conversation_id: str
    task_description: str
    complexity_score: float
    template_used: str
    start_time: datetime
    end_time: Optional[datetime] = None
    thought_steps: List[Dict[str, Any]] = None
    conclusions: Dict[str, Any] = None
    system_recommendations: Dict[str, Any] = None
    status: str = "active"
    
    def __post_init__(self):
        if self.thought_steps is None:
            self.thought_steps = []


class MemoryManager:
    """
    Advanced memory management system for the LangChain Sequential Manager
    
    Handles conversation memory, context management, and sequential thinking
    session tracking with intelligent memory optimization and cleanup.
    """
    
    def __init__(self, max_conversations: int = 100, memory_window: int = 50):
        """
        Initialize the memory management system
        
        Args:
            max_conversations: Maximum number of conversations to keep in memory
            memory_window: Number of messages to keep in conversation buffer
        """
        # Core memory storage
        self.conversations: Dict[str, ConversationContext] = {}
        self.thinking_sessions: Dict[str, SequentialThinkingSession] = {}
        
        # LangChain memory instances for each conversation
        self.langchain_memories: Dict[str, Union[ConversationBufferWindowMemory, ConversationSummaryBufferMemory]] = {}
        
        # Configuration
        self.max_conversations = max_conversations
        self.memory_window = memory_window
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.MemoryManager")
        self.logger.info("Memory Manager initialized", extra={
            'operation': 'MEMORY_MANAGER_INIT',
            'max_conversations': max_conversations,
            'memory_window': memory_window
        })
    
    def create_conversation(self, user_id: str, conversation_id: Optional[str] = None) -> str:
        """
        Create a new conversation context with memory management
        
        Args:
            user_id: Identifier for the user
            conversation_id: Optional conversation ID, generated if not provided
            
        Returns:
            str: Conversation ID for the created conversation
        """
        if conversation_id is None:
            conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        
        # Create conversation context
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            session_start=datetime.now(),
            last_activity=datetime.now()
        )
        
        # Create LangChain memory for this conversation
        langchain_memory = ConversationBufferWindowMemory(
            k=self.memory_window,
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        # Store conversation and memory
        self.conversations[conversation_id] = context
        self.langchain_memories[conversation_id] = langchain_memory
        
        # Cleanup old conversations if needed
        self._cleanup_old_conversations()
        
        self.logger.info("Created new conversation", extra={
            'operation': 'CONVERSATION_CREATE',
            'conversation_id': conversation_id,
            'user_id': user_id
        })
        
        return conversation_id
    
    def get_conversation_memory(self, conversation_id: str) -> Optional[ConversationBufferWindowMemory]:
        """
        Get LangChain memory instance for a conversation
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            ConversationBufferWindowMemory: Memory instance for the conversation
        """
        return self.langchain_memories.get(conversation_id)
    
    def update_conversation(self, conversation_id: str, **updates) -> None:
        """
        Update conversation context with new information
        
        Args:
            conversation_id: ID of the conversation to update
            **updates: Fields to update in the conversation context
        """
        if conversation_id in self.conversations:
            context = self.conversations[conversation_id]
            context.last_activity = datetime.now()
            context.message_count += 1
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)
            
            self.logger.debug("Updated conversation context", extra={
                'operation': 'CONVERSATION_UPDATE',
                'conversation_id': conversation_id,
                'updates': list(updates.keys())
            })
    
    def create_thinking_session(self, conversation_id: str, task_description: str, 
                              complexity_score: float, template_used: str) -> str:
        """
        Create a new sequential thinking session
        
        Args:
            conversation_id: Associated conversation ID
            task_description: Description of the task
            complexity_score: Calculated complexity score
            template_used: Thought process template being used
            
        Returns:
            str: Session ID for the thinking session
        """
        session_id = f"thinking_{uuid.uuid4().hex[:12]}"
        
        session = SequentialThinkingSession(
            session_id=session_id,
            conversation_id=conversation_id,
            task_description=task_description,
            complexity_score=complexity_score,
            template_used=template_used,
            start_time=datetime.now()
        )
        
        # Store the session
        self.thinking_sessions[session_id] = session
        
        # Add to conversation's thinking sessions list
        if conversation_id in self.conversations:
            self.conversations[conversation_id].thinking_sessions.append(session_id)
        
        self.logger.info("Created sequential thinking session", extra={
            'operation': 'THINKING_SESSION_CREATE',
            'session_id': session_id,
            'conversation_id': conversation_id,
            'complexity_score': complexity_score,
            'template_used': template_used
        })
        
        return session_id
    
    def update_thinking_session(self, session_id: str, **updates) -> None:
        """
        Update a sequential thinking session with new information
        
        Args:
            session_id: ID of the thinking session
            **updates: Fields to update in the session
        """
        if session_id in self.thinking_sessions:
            session = self.thinking_sessions[session_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            
            self.logger.debug("Updated thinking session", extra={
                'operation': 'THINKING_SESSION_UPDATE',
                'session_id': session_id,
                'updates': list(updates.keys())
            })
    
    def complete_thinking_session(self, session_id: str, conclusions: Dict[str, Any], 
                                system_recommendations: Dict[str, Any]) -> None:
        """
        Mark a thinking session as complete with final results
        
        Args:
            session_id: ID of the thinking session
            conclusions: Final conclusions from the thinking process
            system_recommendations: Recommendations for system coordination
        """
        if session_id in self.thinking_sessions:
            session = self.thinking_sessions[session_id]
            session.end_time = datetime.now()
            session.conclusions = conclusions
            session.system_recommendations = system_recommendations
            session.status = "completed"
            
            self.logger.info("Completed thinking session", extra={
                'operation': 'THINKING_SESSION_COMPLETE',
                'session_id': session_id,
                'duration': (session.end_time - session.start_time).total_seconds(),
                'thought_steps': len(session.thought_steps)
            })
    
    def get_context_summary(self, conversation_id: str, max_length: int = 1000) -> str:
        """
        Generate a context summary for a conversation
        
        Args:
            conversation_id: ID of the conversation
            max_length: Maximum length of the summary
            
        Returns:
            str: Context summary for the conversation
        """
        if conversation_id not in self.conversations:
            return ""
        
        context = self.conversations[conversation_id]
        
        # Build summary from conversation data
        summary_parts = [
            f"Conversation: {conversation_id}",
            f"Messages: {context.message_count}",
            f"Duration: {(context.last_activity - context.session_start).total_seconds() / 3600:.1f}h"
        ]
        
        # Add complexity history
        if context.complexity_history:
            recent_complexity = context.complexity_history[-3:]  # Last 3 assessments
            complexity_summary = ", ".join([f"{c.get('score', 0):.1f}" for c in recent_complexity])
            summary_parts.append(f"Recent complexity: {complexity_summary}")
        
        # Add thinking sessions
        if context.thinking_sessions:
            active_sessions = [s for s in context.thinking_sessions if self.thinking_sessions.get(s, {}).get('status') == 'active']
            completed_sessions = [s for s in context.thinking_sessions if self.thinking_sessions.get(s, {}).get('status') == 'completed']
            summary_parts.append(f"Thinking sessions: {len(completed_sessions)} completed, {len(active_sessions)} active")
        
        summary = " | ".join(summary_parts)
        return summary[:max_length] if len(summary) > max_length else summary
    
    def _cleanup_old_conversations(self) -> None:
        """Clean up old conversations to maintain memory limits"""
        if len(self.conversations) <= self.max_conversations:
            return
        
        # Sort conversations by last activity (oldest first)
        sorted_conversations = sorted(
            self.conversations.items(),
            key=lambda x: x[1].last_activity
        )
        
        # Remove oldest conversations
        conversations_to_remove = len(self.conversations) - self.max_conversations
        for i in range(conversations_to_remove):
            conv_id, _ = sorted_conversations[i]
            
            # Clean up thinking sessions for this conversation
            if conv_id in self.conversations:
                for session_id in self.conversations[conv_id].thinking_sessions:
                    self.thinking_sessions.pop(session_id, None)
            
            # Remove conversation and memory
            self.conversations.pop(conv_id, None)
            self.langchain_memories.pop(conv_id, None)
        
        self.logger.info("Cleaned up old conversations", extra={
            'operation': 'MEMORY_CLEANUP',
            'removed_count': conversations_to_remove,
            'remaining_count': len(self.conversations)
        })
    
    async def start_cleanup_task(self, interval: int = 3600) -> None:
        """
        Start background cleanup task
        
        Args:
            interval: Cleanup interval in seconds (default: 1 hour)
        """
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    self._cleanup_old_conversations()
                    
                    # Clean up completed thinking sessions older than 24 hours
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    sessions_to_remove = [
                        session_id for session_id, session in self.thinking_sessions.items()
                        if session.status == "completed" and session.end_time and session.end_time < cutoff_time
                    ]
                    
                    for session_id in sessions_to_remove:
                        self.thinking_sessions.pop(session_id, None)
                    
                    if sessions_to_remove:
                        self.logger.info("Cleaned up old thinking sessions", extra={
                            'operation': 'THINKING_CLEANUP',
                            'removed_count': len(sessions_to_remove)
                        })
                        
                except Exception as e:
                    self.logger.error("Error in cleanup task", extra={
                        'operation': 'CLEANUP_ERROR',
                        'error': str(e)
                    })
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        self.logger.info("Started memory cleanup task", extra={
            'operation': 'CLEANUP_TASK_START',
            'interval': interval
        })
    
    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            
        self.logger.info("Stopped memory cleanup task", extra={
            'operation': 'CLEANUP_TASK_STOP'
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory manager statistics
        
        Returns:
            Dict containing memory usage statistics
        """
        active_thinking_sessions = sum(1 for s in self.thinking_sessions.values() if s.status == "active")
        completed_thinking_sessions = sum(1 for s in self.thinking_sessions.values() if s.status == "completed")
        
        return {
            "conversations": {
                "total": len(self.conversations),
                "max_allowed": self.max_conversations
            },
            "thinking_sessions": {
                "active": active_thinking_sessions,
                "completed": completed_thinking_sessions,
                "total": len(self.thinking_sessions)
            },
            "memory_window": self.memory_window
        }


class SequentialThinkingCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for sequential thinking integration
    
    Tracks and logs sequential thinking processes within LangChain agent execution
    """
    
    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(f"{__name__}.SequentialThinkingCallbackHandler")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Log when a tool starts execution"""
        tool_name = serialized.get("name", "unknown")
        self.logger.info("Tool execution started", extra={
            'operation': 'TOOL_START',
            'tool_name': tool_name,
            'input_length': len(input_str) if input_str else 0
        })
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Log when a tool completes execution"""
        self.logger.info("Tool execution completed", extra={
            'operation': 'TOOL_END',
            'output_length': len(output) if output else 0
        })
    
    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Log tool execution errors"""
        self.logger.error("Tool execution error", extra={
            'operation': 'TOOL_ERROR',
            'error': str(error),
            'error_type': type(error).__name__
        })


class LangChainSequentialManager:
    """
    Advanced LangChain Manager Agent with Sequential Thinking Integration
    
    This is the core orchestrator that implements the workflow:
    complexity assessment → sequential thinking invocation → system coordination → validation
    
    Features:
    - LangChain AgentExecutor with OpenRouter integration
    - Advanced memory and context management
    - Sequential thinking as primary reasoning tool
    - Multi-system coordination (Alita, KGoT, Validation, Multimodal)
    - Comprehensive conversation history tracking
    - RESTful API for integration with existing JavaScript components
    """
    
    def __init__(self, config_path: str = "config/models/model_config.json"):
        """
        Initialize the LangChain Sequential Manager
        
        Args:
            config_path: Path to the model configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize memory management
        self.memory_manager = MemoryManager(
            max_conversations=self.config.get("memory", {}).get("max_conversations", 100),
            memory_window=self.config.get("memory", {}).get("window_size", 50)
        )
        
        # Initialize LangChain components
        self.llm = None
        self.agent = None
        self.agent_executor = None
        
        # Service endpoints for integration
        self.service_endpoints = {
            "sequential_thinking": "http://localhost:3000/sequential-thinking",
            "web_agent": "http://localhost:3001",
            "mcp_creation": "http://localhost:3002", 
            "kgot_controller": "http://localhost:3003",
            "validation": "http://localhost:3004",
            "multimodal": "http://localhost:3005"
        }
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="LangChain Sequential Manager",
            description="Advanced manager agent with sequential thinking integration",
            version="1.0.0"
        )
        self._setup_fastapi()
        
        # HTTP client for service communication
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize agent
        asyncio.create_task(self._initialize_agent())
        
        self.logger.info("LangChain Sequential Manager initialized", extra={
            'operation': 'MANAGER_INIT',
            'config_loaded': bool(self.config),
            'memory_manager': True,
            'service_endpoints': len(self.service_endpoints)
        })

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration settings
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration if config file is not available
        
        Returns:
            Dict containing default configuration
        """
        return {
            "model_providers": {
                "openrouter": {
                    "base_url": "https://openrouter.ai/api/v1",
                    "models": {
                        "claude-4-sonnet": {
                            "model_id": "anthropic/claude-4-sonnet-thinking"
                        },
                        "o3": {
                            "model_id": "openai/o3"
                        },
                        "gemini-2.5-pro": {
                            "model_id": "google/gemini-2.5-pro"
                        }
                    }
                }
            },
            "alita_config": {
                "manager_agent": {
                    "primary_model": "claude-4-sonnet",
                    "timeout": 30,
                    "max_retries": 3,
                    "temperature": 0.1
                }
            },
            "memory": {
                "max_conversations": 100,
                "window_size": 50
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup Winston-compatible logging configuration
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{__name__}.LangChainSequentialManager")
        
        # Set up handler if not already configured
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(operation)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger

    def _setup_fastapi(self) -> None:
        """Setup FastAPI application with CORS, middleware, and routes"""
        self.logger.info("Setting up FastAPI application", extra={'operation': 'FASTAPI_SETUP'})
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Add routes
        self._setup_routes()
        
        self.logger.info("FastAPI application setup complete", extra={'operation': 'FASTAPI_SETUP_COMPLETE'})
    
    def _setup_routes(self) -> None:
        """Setup API routes for the LangChain Sequential Manager"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "LangChain Sequential Manager",
                "version": "1.0.0",
                "status": "active",
                "agent_initialized": self.agent_executor is not None
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "memory_stats": self.memory_manager.get_stats()
            }
        
        @self.app.post("/conversation")
        async def create_conversation(request: dict):
            """Create a new conversation with memory management"""
            user_id = request.get("user_id", "default_user")
            conversation_id = self.memory_manager.create_conversation(user_id)
            
            return {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "status": "created"
            }
        
        @self.app.post("/chat")
        async def chat(request: dict):
            """Main chat endpoint with full workflow implementation"""
            try:
                conversation_id = request.get("conversation_id")
                user_input = request.get("input", "")
                user_id = request.get("user_id", "default_user")
                
                # Create conversation if not exists
                if not conversation_id or conversation_id not in self.memory_manager.conversations:
                    conversation_id = self.memory_manager.create_conversation(user_id)
                
                # Get conversation memory
                memory = self.memory_manager.get_conversation_memory(conversation_id)
                if not memory:
                    return {"error": "Failed to get conversation memory", "conversation_id": conversation_id}
                
                # Phase 1: Complexity Assessment
                complexity_analysis = await self._assess_complexity(user_input, conversation_id)
                
                # Update conversation context
                self.memory_manager.update_conversation(
                    conversation_id,
                    complexity_history=self.memory_manager.conversations[conversation_id].complexity_history + [complexity_analysis]
                )
                
                # Phase 2: Sequential Thinking Invocation (if needed)
                thinking_result = None
                if complexity_analysis["should_trigger_sequential_thinking"]:
                    thinking_result = await self._invoke_sequential_thinking(
                        user_input, complexity_analysis, conversation_id
                    )
                
                # Phase 3: Agent Execution with Context
                agent_input = {
                    "input": user_input,
                    "conversation_id": conversation_id,
                    "complexity_analysis": complexity_analysis,
                    "thinking_result": thinking_result,
                    "context_summary": self.memory_manager.get_context_summary(conversation_id)
                }
                
                # Execute agent with memory
                if self.agent_executor:
                    result = await self.agent_executor.ainvoke(
                        agent_input,
                        config={"configurable": {"session_id": conversation_id}}
                    )
                    
                    response = result.get("output", "")
                    
                    # Phase 4: Validation (if needed)
                    if complexity_analysis.get("complexity_score", 0) > 7:
                        validation_result = await self._validate_response(response, user_input)
                        if not validation_result.get("is_valid", True):
                            response = f"{response}\n\n[Validation Notice: {validation_result.get('message', 'Quality check flagged for review')}]"
                    
                    # Update conversation memory
                    memory.save_context({"input": user_input}, {"output": response})
                    
                    # Update conversation context
                    self.memory_manager.update_conversation(
                        conversation_id,
                        total_tokens=self.memory_manager.conversations[conversation_id].total_tokens + len(user_input) + len(response)
                    )
                    
                    return {
                        "response": response,
                        "conversation_id": conversation_id,
                        "complexity_analysis": complexity_analysis,
                        "thinking_session_id": thinking_result.get("session_id") if thinking_result else None,
                        "agent_steps": result.get("intermediate_steps", [])
                    }
                else:
                    return {"error": "Agent not initialized", "conversation_id": conversation_id}
                    
            except Exception as e:
                self.logger.error("Error in chat endpoint", extra={
                    'operation': 'CHAT_ERROR',
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                return {"error": f"Chat processing failed: {str(e)}"}
        
        @self.app.get("/conversations/{conversation_id}/memory")
        async def get_conversation_memory(conversation_id: str):
            """Get conversation memory and context"""
            if conversation_id in self.memory_manager.conversations:
                context = self.memory_manager.conversations[conversation_id]
                return {
                    "conversation_id": conversation_id,
                    "context": asdict(context),
                    "memory_summary": self.memory_manager.get_context_summary(conversation_id)
                }
            else:
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        @self.app.get("/thinking-sessions/{session_id}")
        async def get_thinking_session(session_id: str):
            """Get thinking session details"""
            if session_id in self.memory_manager.thinking_sessions:
                session = self.memory_manager.thinking_sessions[session_id]
                return asdict(session)
            else:
                raise HTTPException(status_code=404, detail="Thinking session not found")
        
        @self.app.get("/stats")
        async def get_stats():
            """Get system statistics"""
            return {
                "memory_stats": self.memory_manager.get_stats(),
                "agent_initialized": self.agent_executor is not None,
                "service_endpoints": self.service_endpoints
            }
    
    async def _assess_complexity(self, user_input: str, conversation_id: str) -> Dict[str, Any]:
        """
        Assess task complexity to determine if sequential thinking is needed
        
        Args:
            user_input: User's input message
            conversation_id: Conversation identifier
            
        Returns:
            Dictionary containing complexity analysis
        """
        try:
            # Basic complexity scoring factors
            complexity_score = 0
            factors = []
            
            # Length factor
            if len(user_input) > 500:
                complexity_score += 2
                factors.append("long_input")
            
            # Complexity keywords
            complex_keywords = [
                "analyze", "compare", "evaluate", "integrate", "coordinate", 
                "optimize", "troubleshoot", "debug", "research", "synthesize",
                "multiple", "complex", "advanced", "sophisticated", "comprehensive"
            ]
            keyword_matches = sum(1 for keyword in complex_keywords if keyword.lower() in user_input.lower())
            complexity_score += keyword_matches
            if keyword_matches > 2:
                factors.append("complex_keywords")
            
            # Multi-step indicators
            step_indicators = ["first", "then", "next", "after", "finally", "step", "phase"]
            if sum(1 for indicator in step_indicators if indicator in user_input.lower()) > 2:
                complexity_score += 3
                factors.append("multi_step")
            
            # Question complexity
            question_count = user_input.count("?")
            if question_count > 2:
                complexity_score += 2
                factors.append("multiple_questions")
            
            # Context from conversation history
            if conversation_id in self.memory_manager.conversations:
                context = self.memory_manager.conversations[conversation_id]
                if len(context.complexity_history) > 0:
                    avg_previous_complexity = sum(h.get("complexity_score", 0) for h in context.complexity_history) / len(context.complexity_history)
                    if avg_previous_complexity > 5:
                        complexity_score += 1
                        factors.append("complex_conversation_history")
            
            # System integration needs
            system_keywords = ["web", "scrape", "mcp", "kgot", "knowledge", "graph", "validate", "multimodal"]
            system_matches = sum(1 for keyword in system_keywords if keyword.lower() in user_input.lower())
            if system_matches > 1:
                complexity_score += 2
                factors.append("multi_system_needs")
            
            # Determine if sequential thinking should trigger
            should_trigger = complexity_score > 7
            
            analysis = {
                "complexity_score": complexity_score,
                "factors": factors,
                "should_trigger_sequential_thinking": should_trigger,
                "assessment_timestamp": datetime.now().isoformat(),
                "input_length": len(user_input),
                "keyword_matches": keyword_matches,
                "system_matches": system_matches
            }
            
            self.logger.info("Complexity assessment completed", extra={
                'operation': 'COMPLEXITY_ASSESSMENT',
                'complexity_score': complexity_score,
                'should_trigger': should_trigger,
                'factors': factors
            })
            
            return analysis
            
        except Exception as e:
            self.logger.error("Error in complexity assessment", extra={
                'operation': 'COMPLEXITY_ASSESSMENT_ERROR',
                'error': str(e)
            })
            return {
                "complexity_score": 5,  # Default moderate complexity
                "factors": ["assessment_error"],
                "should_trigger_sequential_thinking": False,
                "error": str(e)
            }
    
    async def _invoke_sequential_thinking(self, user_input: str, complexity_analysis: Dict[str, Any], conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Invoke sequential thinking MCP for complex task processing
        
        Args:
            user_input: User's input message
            complexity_analysis: Complexity assessment results
            conversation_id: Conversation identifier
            
        Returns:
            Sequential thinking results or None if failed
        """
        try:
            self.logger.info("Invoking sequential thinking MCP", extra={
                'operation': 'SEQUENTIAL_THINKING_INVOKE',
                'conversation_id': conversation_id,
                'complexity_score': complexity_analysis.get("complexity_score", 0)
            })
            
            # Prepare task context for sequential thinking
            task_context = {
                "taskId": f"task_{conversation_id}_{int(time.time())}",
                "description": user_input,
                "requirements": [{"description": "Process user request", "priority": "high"}],
                "errors": [],  # Will be populated if there are known errors
                "systemsInvolved": self._determine_systems_needed(user_input),
                "dataTypes": self._determine_data_types(user_input),
                "interactions": [{"type": "user_interaction", "complexity": "high"}],
                "timeline": {"urgency": "medium"},
                "dependencies": [],
                "complexity_analysis": complexity_analysis
            }
            
            # Make request to sequential thinking service
            response = await self.http_client.post(
                self.service_endpoints["sequential_thinking"],
                json=task_context,
                timeout=90.0  # Longer timeout for thinking processes
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Create thinking session in memory
                session_id = self.memory_manager.create_thinking_session(
                    conversation_id=conversation_id,
                    task_description=user_input,
                    complexity_score=complexity_analysis.get("complexity_score", 0),
                    template_used=result.get("template", "unknown")
                )
                
                # Update session with results
                self.memory_manager.update_thinking_session(
                    session_id,
                    thought_steps=result.get("thinking_steps", []),
                    status="completed"
                )
                
                if "conclusions" in result and "systemRecommendations" in result:
                    self.memory_manager.complete_thinking_session(
                        session_id,
                        conclusions=result["conclusions"],
                        system_recommendations=result["systemRecommendations"]
                    )
                
                self.logger.info("Sequential thinking completed successfully", extra={
                    'operation': 'SEQUENTIAL_THINKING_SUCCESS',
                    'session_id': session_id,
                    'template_used': result.get("template", "unknown")
                })
                
                return {
                    "session_id": session_id,
                    "result": result,
                    "status": "completed"
                }
            else:
                self.logger.error("Sequential thinking service error", extra={
                    'operation': 'SEQUENTIAL_THINKING_SERVICE_ERROR',
                    'status_code': response.status_code
                })
                return None
                
        except Exception as e:
            self.logger.error("Error invoking sequential thinking", extra={
                'operation': 'SEQUENTIAL_THINKING_INVOKE_ERROR',
                'error': str(e),
                'error_type': type(e).__name__
            })
            return None
    
    def _determine_systems_needed(self, user_input: str) -> List[str]:
        """Determine which systems are needed based on user input"""
        systems = []
        
        if any(keyword in user_input.lower() for keyword in ["web", "scrape", "browse", "search"]):
            systems.append("web_agent")
        
        if any(keyword in user_input.lower() for keyword in ["tool", "mcp", "create", "generate"]):
            systems.append("mcp_creation")
        
        if any(keyword in user_input.lower() for keyword in ["knowledge", "graph", "reasoning", "analyze"]):
            systems.append("kgot_controller")
        
        if any(keyword in user_input.lower() for keyword in ["validate", "check", "verify", "quality"]):
            systems.append("validation")
        
        if any(keyword in user_input.lower() for keyword in ["image", "audio", "video", "multimodal"]):
            systems.append("multimodal")
        
        return systems if systems else ["general"]
    
    def _determine_data_types(self, user_input: str) -> List[str]:
        """Determine data types involved based on user input"""
        data_types = ["text"]  # Always include text
        
        if any(keyword in user_input.lower() for keyword in ["image", "picture", "photo", "visual"]):
            data_types.append("image")
        
        if any(keyword in user_input.lower() for keyword in ["audio", "sound", "music", "voice"]):
            data_types.append("audio")
        
        if any(keyword in user_input.lower() for keyword in ["video", "movie", "clip"]):
            data_types.append("video")
        
        if any(keyword in user_input.lower() for keyword in ["json", "xml", "csv", "data", "structured"]):
            data_types.append("structured")
        
        if any(keyword in user_input.lower() for keyword in ["graph", "network", "relationship"]):
            data_types.append("graph")
        
        return data_types
    
    async def _validate_response(self, response: str, original_input: str) -> Dict[str, Any]:
        """
        Validate response quality using validation service
        
        Args:
            response: Generated response
            original_input: Original user input
            
        Returns:
            Validation results
        """
        try:
            validation_data = {
                "content": response,
                "original_input": original_input,
                "type": "response_quality",
                "criteria": ["relevance", "completeness", "accuracy"]
            }
            
            validation_response = await self.http_client.post(
                f"{self.service_endpoints['validation']}/validate",
                json=validation_data,
                timeout=30.0
            )
            
            if validation_response.status_code == 200:
                return validation_response.json()
            else:
                return {"is_valid": True, "message": "Validation service unavailable"}
                
        except Exception as e:
            self.logger.error("Error in response validation", extra={
                'operation': 'RESPONSE_VALIDATION_ERROR',
                'error': str(e)
            })
            return {"is_valid": True, "message": "Validation failed"}
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Start the FastAPI server
        
        Args:
            host: Server host
            port: Server port
        """
        # Start memory cleanup task
        await self.memory_manager.start_cleanup_task()
        
        self.logger.info("Starting LangChain Sequential Manager server", extra={
            'operation': 'SERVER_START',
            'host': host,
            'port': port
        })
        
        # Start the server
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the manager"""
        self.logger.info("Shutting down LangChain Sequential Manager", extra={'operation': 'SHUTDOWN'})
        
        # Stop memory cleanup task
        await self.memory_manager.stop_cleanup_task()
        
        # Close HTTP client
        await self.http_client.aclose()

    async def _initialize_agent(self) -> None:
        """
        Initialize the LangChain agent with tools and memory
        
        This method sets up the complete LangChain agent with:
        - OpenRouter LLM integration
        - Agent tools for system coordination  
        - Memory management integration
        - Sequential thinking capabilities
        """
        try:
            self.logger.info("Initializing LangChain agent", extra={'operation': 'AGENT_INIT_START'})
            
            # Initialize OpenRouter LLM
            openrouter_config = self.config.get("model_providers", {}).get("openrouter", {})
            manager_config = self.config.get("alita_config", {}).get("manager_agent", {})
            
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
            # Create LLM with OpenRouter
            self.llm = ChatOpenAI(
                base_url=openrouter_config.get("base_url", "https://openrouter.ai/api/v1"),
                api_key=api_key,
                model=openrouter_config.get("models", {}).get(manager_config.get("primary_model", "claude-4-sonnet-thinking"), {}).get("model_id", "anthropic/claude-4-sonnet-thinking"),
                temperature=manager_config.get("temperature", 0.1),
                timeout=manager_config.get("timeout", 30),
                max_retries=manager_config.get("max_retries", 3)
            )
            
            # Create agent tools
            tools = await self.create_agent_tools()
            
            # Create agent prompt template
            system_prompt = """You are the LangChain Sequential Manager Agent for the Alita-KGoT Enhanced System.

Your primary capabilities:
1. **Sequential Thinking Integration**: Use sequential thinking MCP for complex reasoning and problem-solving
2. **System Coordination**: Coordinate between Alita MCP creation and KGoT knowledge processing systems  
3. **Memory Management**: Maintain conversation context and multi-step operation state
4. **Multi-Modal Processing**: Handle text, code, knowledge graphs, and multimedia data
5. **Validation**: Ensure system outputs meet quality and accuracy standards

**Workflow Process**:
1. Assess task complexity (trigger sequential thinking if score > 7)
2. Invoke sequential thinking for complex problems requiring systematic reasoning
3. Coordinate system execution based on thinking recommendations
4. Validate results and provide comprehensive responses

**Tools Available**:
- sequential_thinking: Advanced reasoning for complex problems
- web_agent: Web scraping, automation, and browser interactions
- mcp_creation: Create and deploy Model Context Protocol tools
- kgot_processing: Knowledge graph operations and reasoning
- validation: Quality assurance and result validation

Always use sequential thinking for complex tasks involving multiple systems, error resolution, or sophisticated reasoning requirements."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Create agent
            self.agent = create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
            
            # Create callback handler for memory integration
            callback_handler = SequentialThinkingCallbackHandler(self.memory_manager)
            
            # Create agent executor with memory
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=tools,
                memory=None,  # We handle memory manually per conversation
                verbose=True,
                callbacks=[callback_handler],
                max_iterations=10,
                max_execution_time=300,  # 5 minutes timeout
                return_intermediate_steps=True
            )
            
            self.logger.info("LangChain agent initialized successfully", extra={
                'operation': 'AGENT_INIT_SUCCESS',
                'tools_count': len(tools),
                'llm_model': manager_config.get("primary_model", "claude-4-sonnet")
            })
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain agent: {str(e)}", extra={
                'operation': 'AGENT_INIT_ERROR',
                'error': str(e)
            })
            raise

    async def create_agent_tools(self) -> List[BaseTool]:
        """
        Create LangChain tools for the agent to interface with different systems
        
        Returns:
            List[BaseTool]: List of configured LangChain tools
        """
        self.logger.info("Creating agent tools", extra={'operation': 'TOOLS_CREATE_START'})
        
        tools = []
        
        # 1. Sequential Thinking Tool
        @tool
        async def sequential_thinking_tool(task_description: str, complexity_factors: str = "") -> str:
            """
            Invoke sequential thinking MCP for complex reasoning and problem-solving.
            
            Use this tool when:
            - Task complexity score > 7
            - Multiple errors need systematic resolution  
            - Cross-system coordination required
            - Multi-step reasoning needed
            
            Args:
                task_description: Detailed description of the task or problem
                complexity_factors: Additional factors that make this task complex
                
            Returns:
                str: Structured thinking results with conclusions and recommendations
            """
            try:
                self.logger.info("Invoking sequential thinking tool", extra={
                    'operation': 'SEQUENTIAL_THINKING_INVOKE',
                    'task_description': task_description[:100]
                })
                
                # Prepare request for sequential thinking MCP
                thinking_request = {
                    "task_description": task_description,
                    "complexity_factors": complexity_factors,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Call sequential thinking service
                response = await self.http_client.post(
                    f"{self.service_endpoints['sequential_thinking']}/invoke",
                    json=thinking_request,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    return f"Sequential thinking failed with status {response.status_code}"
                    
            except Exception as e:
                self.logger.error(f"Sequential thinking tool error: {str(e)}", extra={
                    'operation': 'SEQUENTIAL_THINKING_ERROR',
                    'error': str(e)
                })
                return f"Error in sequential thinking: {str(e)}"
        
        # 2. Web Agent Tool
        @tool
        async def web_agent_tool(action: str, target: str, data: str = "") -> str:
            """
            Interface with the web agent for browser automation and web scraping.
            
            Actions supported:
            - navigate: Navigate to a URL
            - scrape: Extract data from a webpage
            - interact: Interact with web elements
            - screenshot: Take webpage screenshots
            
            Args:
                action: The action to perform (navigate, scrape, interact, screenshot)
                target: URL or element selector
                data: Additional data for the action
                
            Returns:
                str: Result of the web agent operation
            """
            try:
                self.logger.info("Invoking web agent tool", extra={
                    'operation': 'WEB_AGENT_INVOKE',
                    'action': action,
                    'target': target[:100]
                })
                
                web_request = {
                    "action": action,
                    "target": target,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
                
                response = await self.http_client.post(
                    f"{self.service_endpoints['web_agent']}/execute",
                    json=web_request,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    return f"Web agent failed with status {response.status_code}"
                    
            except Exception as e:
                self.logger.error(f"Web agent tool error: {str(e)}", extra={
                    'operation': 'WEB_AGENT_ERROR',
                    'error': str(e)
                })
                return f"Error in web agent: {str(e)}"
        
        # 3. MCP Creation Tool
        @tool
        async def mcp_creation_tool(tool_type: str, requirements: str, specifications: str = "") -> str:
            """
            Create and deploy Model Context Protocol (MCP) tools based on requirements.
            
            Tool types:
            - web_scraper: Custom web scraping capabilities
            - api_client: API integration tools
            - data_processor: Data transformation tools
            - automation: Task automation tools
            
            Args:
                tool_type: Type of MCP tool to create
                requirements: Detailed requirements for the tool
                specifications: Technical specifications and constraints
                
            Returns:
                str: Result of MCP creation including tool ID and deployment status
            """
            try:
                self.logger.info("Invoking MCP creation tool", extra={
                    'operation': 'MCP_CREATION_INVOKE',
                    'tool_type': tool_type,
                    'requirements': requirements[:100]
                })
                
                mcp_request = {
                    "tool_type": tool_type,
                    "requirements": requirements,
                    "specifications": specifications,
                    "timestamp": datetime.now().isoformat()
                }
                
                response = await self.http_client.post(
                    f"{self.service_endpoints['mcp_creation']}/create",
                    json=mcp_request,
                    timeout=120.0  # MCP creation can take longer
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    return f"MCP creation failed with status {response.status_code}"
                    
            except Exception as e:
                self.logger.error(f"MCP creation tool error: {str(e)}", extra={
                    'operation': 'MCP_CREATION_ERROR',
                    'error': str(e)
                })
                return f"Error in MCP creation: {str(e)}"
        
        # 4. KGoT Processing Tool
        @tool
        async def kgot_processing_tool(operation: str, query: str, data: str = "") -> str:
            """
            Interface with Knowledge Graph of Thoughts (KGoT) system for advanced reasoning.
            
            Operations:
            - query: Execute knowledge graph queries
            - reason: Perform graph-based reasoning
            - analyze: Analyze knowledge patterns
            - integrate: Integrate new knowledge
            
            Args:
                operation: KGoT operation to perform
                query: Query or reasoning task
                data: Additional data for the operation
                
            Returns:
                str: Results from KGoT processing
            """
            try:
                self.logger.info("Invoking KGoT processing tool", extra={
                    'operation': 'KGOT_PROCESSING_INVOKE',
                    'kgot_operation': operation,
                    'query': query[:100]
                })
                
                kgot_request = {
                    "operation": operation,
                    "query": query,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
                
                response = await self.http_client.post(
                    f"{self.service_endpoints['kgot_controller']}/process",
                    json=kgot_request,
                    timeout=90.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    return f"KGoT processing failed with status {response.status_code}"
                    
            except Exception as e:
                self.logger.error(f"KGoT processing tool error: {str(e)}", extra={
                    'operation': 'KGOT_PROCESSING_ERROR',
                    'error': str(e)
                })
                return f"Error in KGoT processing: {str(e)}"
        
        # 5. Validation Tool
        @tool
        async def validation_tool(validation_type: str, data: str, criteria: str = "") -> str:
            """
            Validate results and outputs for quality assurance.
            
            Validation types:
            - accuracy: Check data accuracy and correctness
            - completeness: Verify task completion
            - quality: Assess output quality
            - consistency: Check for consistency issues
            
            Args:
                validation_type: Type of validation to perform
                data: Data or results to validate
                criteria: Specific validation criteria
                
            Returns:
                str: Validation results with scores and recommendations
            """
            try:
                self.logger.info("Invoking validation tool", extra={
                    'operation': 'VALIDATION_INVOKE',
                    'validation_type': validation_type,
                    'data_length': len(data)
                })
                
                validation_request = {
                    "validation_type": validation_type,
                    "data": data,
                    "criteria": criteria,
                    "timestamp": datetime.now().isoformat()
                }
                
                response = await self.http_client.post(
                    f"{self.service_endpoints['validation']}/validate",
                    json=validation_request,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    return f"Validation failed with status {response.status_code}"
                    
            except Exception as e:
                self.logger.error(f"Validation tool error: {str(e)}", extra={
                    'operation': 'VALIDATION_ERROR',
                    'error': str(e)
                })
                return f"Error in validation: {str(e)}"
        
        # Add all tools to the list
        tools.extend([
            sequential_thinking_tool,
            web_agent_tool,
            mcp_creation_tool,
            kgot_processing_tool,
            validation_tool
        ])
        
        self.logger.info("Agent tools created successfully", extra={
            'operation': 'TOOLS_CREATE_SUCCESS',
            'tools_count': len(tools)
        })
        
        return tools


# Main entry point
async def main():
    """Main entry point for the LangChain Sequential Manager"""
    import os
    
    # Initialize manager
    manager = LangChainSequentialManager()
    
    # Start server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    try:
        await manager.start_server(host=host, port=port)
    except KeyboardInterrupt:
        await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 