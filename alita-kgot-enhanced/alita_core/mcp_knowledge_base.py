#!/usr/bin/env python3
"""
Alita MCP Knowledge Base Builder - Task 14 Implementation

Advanced MCP Knowledge Base Builder implementing RAG-MCP Section 3.2 external vector index design principles
with enhanced semantic indexing, performance tracking, and intelligent curation following Pareto optimization.

Features:
- RAG-MCP Section 3.2 external vector index architecture with multi-modal embeddings  
- Section 4 experimental metrics tracking for performance optimization
- LangChain agent-based intelligent MCP curation following Pareto principles
- Incremental updates with version control and dependency management
- Enhanced semantic search with capability-aware vector indexing
- Performance analytics and automated quality assessment
- Seamless integration with existing RAG-MCP Engine architecture

@module MCPKnowledgeBase
@author Enhanced Alita KGoT System  
@version 1.0.0
@based_on RAG-MCP Sections 3.2, 4.0 with enhanced knowledge management capabilities
"""

import os
import sys
import json
import logging
import asyncio
import aiohttp
import numpy as np
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import aiosqlite
from collections import defaultdict, deque

# LangChain imports for intelligent MCP curation (per user's hard rule)
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import BaseMessage
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferWindowMemory

# Scientific computing for advanced vector operations
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using fallback similarity search")

try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Pydantic for enhanced data validation
from pydantic import BaseModel, Field, validator

# Import existing RAG-MCP components for integration
try:
    from .rag_mcp_engine import (
        MCPCategory, MCPToolSpec, ParetoMCPRegistry, 
        VectorIndexManager, RAGMCPEngine, logger as rag_logger
    )
except ImportError:
    # Fallback if running standalone
    from rag_mcp_engine import (
        MCPCategory, MCPToolSpec, ParetoMCPRegistry,
        VectorIndexManager, RAGMCPEngine, logger as rag_logger
    )

# Setup Winston-compatible logging with enhanced context
project_root = Path(__file__).parent.parent
log_dir = project_root / 'logs' / 'alita'
log_dir.mkdir(parents=True, exist_ok=True)

# Enhanced logging configuration for knowledge base operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(operation)s] [%(component)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'mcp_knowledge_base.log'),
        logging.FileHandler(log_dir / 'combined.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MCPKnowledgeBase')

class MCPQualityScore(Enum):
    """Quality assessment scores for MCP specifications"""
    EXCELLENT = "excellent"      # 90-100% - Production ready, comprehensive
    GOOD = "good"               # 75-89% - Good quality, minor improvements needed  
    ACCEPTABLE = "acceptable"   # 60-74% - Usable but needs enhancement
    POOR = "poor"              # 40-59% - Significant improvements needed
    UNACCEPTABLE = "unacceptable"  # 0-39% - Major rework required

class MCPUpdateOperation(Enum):
    """Types of incremental update operations"""
    CREATE = "create"
    UPDATE = "update" 
    DELETE = "delete"
    BATCH_UPDATE = "batch_update"
    REINDEX = "reindex"

@dataclass
class EnhancedMCPSpec:
    """
    Enhanced MCP Tool Specification with version control, performance tracking,
    and multi-modal embedding support for advanced knowledge base management
    """
    # Core MCP specification fields (compatible with existing MCPToolSpec)
    name: str
    description: str
    capabilities: List[str]
    category: MCPCategory
    usage_frequency: float = 0.0
    reliability_score: float = 0.0
    cost_efficiency: float = 0.0
    pareto_score: float = 0.0
    
    # Enhanced knowledge base fields
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = "system"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Performance tracking (RAG-MCP Section 4 experimental metrics)
    usage_stats: Dict[str, Any] = field(default_factory=lambda: {
        'total_invocations': 0,
        'successful_invocations': 0,
        'average_response_time_ms': 0.0,
        'last_used': None,
        'usage_trend': 'stable'
    })
    
    performance_metrics: Dict[str, float] = field(default_factory=lambda: {
        'success_rate': 0.0,
        'avg_completion_time': 0.0,
        'cost_per_invocation': 0.0,
        'user_satisfaction': 0.0,
        'error_rate': 0.0,
        'resource_efficiency': 0.0
    })
    
    # Quality assessment and validation
    quality_score: MCPQualityScore = MCPQualityScore.ACCEPTABLE
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    curation_notes: List[str] = field(default_factory=list)
    
    # Multi-modal embeddings for enhanced semantic search
    text_embedding: Optional[np.ndarray] = None          # Description embedding
    capability_embedding: Optional[np.ndarray] = None    # Capabilities embedding  
    metadata_embedding: Optional[np.ndarray] = None      # Metadata embedding
    composite_embedding: Optional[np.ndarray] = None     # Combined multi-modal embedding
    
    # Indexing metadata
    indexed_at: Optional[datetime] = None
    index_version: str = "1.0"
    embedding_model: str = "openai/text-embedding-ada-002"
    
    def to_legacy_spec(self) -> MCPToolSpec:
        """Convert to legacy MCPToolSpec for backwards compatibility"""
        return MCPToolSpec(
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
            category=self.category,
            usage_frequency=self.usage_frequency,
            reliability_score=self.reliability_score,
            cost_efficiency=self.cost_efficiency,
            pareto_score=self.pareto_score,
            metadata=self.get_metadata_dict(),
            embedding=self.composite_embedding,
            indexed_at=self.indexed_at
        )
    
    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get metadata dictionary for integration with existing systems"""
        return {
            'version': self.version,
            'author': self.author,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'quality_score': self.quality_score.value,
            'usage_stats': self.usage_stats,
            'performance_metrics': self.performance_metrics
        }
    
    def update_performance_metrics(self, success: bool, response_time_ms: float, 
                                 cost: float = 0.0, user_rating: float = 0.0):
        """Update performance metrics based on usage results"""
        self.usage_stats['total_invocations'] += 1
        if success:
            self.usage_stats['successful_invocations'] += 1
        
        # Update running averages
        total = self.usage_stats['total_invocations']
        current_avg_time = self.usage_stats['average_response_time_ms']
        self.usage_stats['average_response_time_ms'] = (
            (current_avg_time * (total - 1) + response_time_ms) / total
        )
        
        # Update performance metrics
        self.performance_metrics['success_rate'] = (
            self.usage_stats['successful_invocations'] / total
        )
        self.performance_metrics['avg_completion_time'] = response_time_ms
        self.performance_metrics['error_rate'] = 1.0 - self.performance_metrics['success_rate']
        
        if cost > 0:
            self.performance_metrics['cost_per_invocation'] = cost
        if user_rating > 0:
            current_satisfaction = self.performance_metrics['user_satisfaction']
            self.performance_metrics['user_satisfaction'] = (
                (current_satisfaction * (total - 1) + user_rating) / total
            )
        
        self.usage_stats['last_used'] = datetime.now()
        self.updated_at = datetime.now()
        
        logger.info("Updated performance metrics for MCP", extra={
            'operation': 'PERFORMANCE_UPDATE',
            'component': 'EnhancedMCPSpec',
            'mcp_name': self.name,
            'success_rate': self.performance_metrics['success_rate'],
            'total_invocations': total
        })

@dataclass  
class MCPChangeRecord:
    """Record of changes made to MCP specifications for audit trail"""
    change_id: str
    mcp_name: str
    operation: MCPUpdateOperation
    changes: Dict[str, Any]
    reason: str
    author: str
    timestamp: datetime = field(default_factory=datetime.now)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeBaseMetrics:
    """Comprehensive metrics for knowledge base performance and health"""
    total_mcps: int = 0
    mcps_by_category: Dict[MCPCategory, int] = field(default_factory=dict)
    mcps_by_quality: Dict[MCPQualityScore, int] = field(default_factory=dict)
    average_pareto_score: float = 0.0
    index_size_mb: float = 0.0
    last_update: Optional[datetime] = None
    update_frequency: Dict[str, int] = field(default_factory=dict)
    performance_trends: Dict[str, List[float]] = field(default_factory=dict)
    health_score: float = 0.0

class MCPPerformanceTracker:
    """
    Advanced performance tracking system implementing RAG-MCP Section 4 experimental metrics
    with real-time analytics, trend analysis, and automated quality assessment
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize performance tracker with persistent storage
        
        @param db_path Optional path to SQLite database for metrics storage
        """
        self.db_path = db_path or str(project_root / 'data' / 'mcp_performance.db')
        self.metrics_cache = {}
        self.performance_window = deque(maxlen=1000)  # Sliding window for trend analysis
        self.alert_thresholds = {
            'success_rate_min': 0.85,
            'response_time_max_ms': 5000,
            'error_rate_max': 0.15,
            'satisfaction_min': 3.5
        }
        
        logger.info("Initialized MCP Performance Tracker", extra={
            'operation': 'INIT',
            'component': 'MCPPerformanceTracker',
            'db_path': self.db_path
        })
    
    async def initialize_storage(self):
        """Initialize SQLite database for performance metrics storage"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS mcp_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mcp_name TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        operation_type TEXT,
                        success BOOLEAN,
                        response_time_ms REAL,
                        cost REAL,
                        user_rating REAL,
                        error_message TEXT,
                        context_data TEXT
                    )
                ''')
                
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS mcp_aggregated_metrics (
                        mcp_name TEXT PRIMARY KEY,
                        total_invocations INTEGER DEFAULT 0,
                        successful_invocations INTEGER DEFAULT 0,
                        total_response_time_ms REAL DEFAULT 0,
                        total_cost REAL DEFAULT 0,
                        total_user_rating REAL DEFAULT 0,
                        rating_count INTEGER DEFAULT 0,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                        pareto_score REAL DEFAULT 0
                    )
                ''')
                
                await db.execute('''
                    CREATE INDEX IF NOT EXISTS idx_mcp_name ON mcp_performance(mcp_name)
                ''')
                await db.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON mcp_performance(timestamp)
                ''')
                
                await db.commit()
                
            logger.info("Initialized performance tracking database", extra={
                'operation': 'DB_INIT',
                'component': 'MCPPerformanceTracker'
            })
                
        except Exception as e:
            logger.error(f"Failed to initialize performance tracking database: {str(e)}", extra={
                'operation': 'DB_INIT_ERROR',
                'component': 'MCPPerformanceTracker',
                'error': str(e)
            })
            raise
    
    async def record_usage(self, mcp_name: str, operation_type: str, 
                          success: bool, response_time_ms: float,
                          cost: float = 0.0, user_rating: float = 0.0,
                          error_message: str = None, context: Dict[str, Any] = None):
        """
        Record MCP usage event with comprehensive metrics
        
        @param mcp_name Name of the MCP being tracked
        @param operation_type Type of operation performed 
        @param success Whether the operation was successful
        @param response_time_ms Response time in milliseconds
        @param cost Cost of the operation (optional)
        @param user_rating User satisfaction rating 1-5 (optional)
        @param error_message Error message if operation failed
        @param context Additional context data
        """
        try:
            # Record individual event
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT INTO mcp_performance 
                    (mcp_name, operation_type, success, response_time_ms, cost, 
                     user_rating, error_message, context_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    mcp_name, operation_type, success, response_time_ms,
                    cost, user_rating, error_message, 
                    json.dumps(context) if context else None
                ))
                
                # Update aggregated metrics
                await db.execute('''
                    INSERT OR REPLACE INTO mcp_aggregated_metrics 
                    (mcp_name, total_invocations, successful_invocations, 
                     total_response_time_ms, total_cost, total_user_rating, rating_count)
                    VALUES (
                        ?,
                        COALESCE((SELECT total_invocations FROM mcp_aggregated_metrics WHERE mcp_name = ?), 0) + 1,
                        COALESCE((SELECT successful_invocations FROM mcp_aggregated_metrics WHERE mcp_name = ?), 0) + ?,
                        COALESCE((SELECT total_response_time_ms FROM mcp_aggregated_metrics WHERE mcp_name = ?), 0) + ?,
                        COALESCE((SELECT total_cost FROM mcp_aggregated_metrics WHERE mcp_name = ?), 0) + ?,
                        COALESCE((SELECT total_user_rating FROM mcp_aggregated_metrics WHERE mcp_name = ?), 0) + ?,
                        COALESCE((SELECT rating_count FROM mcp_aggregated_metrics WHERE mcp_name = ?), 0) + ?
                    )
                ''', (
                    mcp_name, mcp_name, mcp_name, 1 if success else 0,
                    mcp_name, response_time_ms, mcp_name, cost,
                    mcp_name, user_rating if user_rating > 0 else 0,
                    mcp_name, 1 if user_rating > 0 else 0
                ))
                
                await db.commit()
            
            # Update in-memory cache
            self._update_cache_metrics(mcp_name, success, response_time_ms, cost, user_rating)
            
            # Add to sliding window for trend analysis
            self.performance_window.append({
                'mcp_name': mcp_name,
                'timestamp': datetime.now(),
                'success': success,
                'response_time_ms': response_time_ms,
                'cost': cost
            })
            
            logger.info("Recorded MCP usage", extra={
                'operation': 'USAGE_RECORD',
                'component': 'MCPPerformanceTracker',
                'mcp_name': mcp_name,
                'success': success,
                'response_time_ms': response_time_ms
            })
            
        except Exception as e:
            logger.error(f"Failed to record MCP usage: {str(e)}", extra={
                'operation': 'USAGE_RECORD_ERROR',
                'component': 'MCPPerformanceTracker',
                'mcp_name': mcp_name,
                'error': str(e)
            })
    
    def _update_cache_metrics(self, mcp_name: str, success: bool, 
                            response_time_ms: float, cost: float, user_rating: float):
        """Update in-memory metrics cache for fast access"""
        if mcp_name not in self.metrics_cache:
            self.metrics_cache[mcp_name] = {
                'total_invocations': 0,
                'successful_invocations': 0,
                'total_response_time': 0.0,
                'total_cost': 0.0,
                'total_rating': 0.0,
                'rating_count': 0
            }
        
        cache = self.metrics_cache[mcp_name]
        cache['total_invocations'] += 1
        if success:
            cache['successful_invocations'] += 1
        cache['total_response_time'] += response_time_ms
        cache['total_cost'] += cost
        if user_rating > 0:
            cache['total_rating'] += user_rating
            cache['rating_count'] += 1
    
    async def get_mcp_metrics(self, mcp_name: str, timeframe_days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for a specific MCP
        
        @param mcp_name Name of the MCP
        @param timeframe_days Number of days to include in analysis
        @return Dictionary containing performance metrics and trends
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=timeframe_days)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Get aggregated metrics
                async with db.execute('''
                    SELECT * FROM mcp_aggregated_metrics WHERE mcp_name = ?
                ''', (mcp_name,)) as cursor:
                    agg_row = await cursor.fetchone()
                
                if not agg_row:
                    return {'error': f'No metrics found for MCP: {mcp_name}'}
                
                # Get recent performance data for trends
                async with db.execute('''
                    SELECT success, response_time_ms, cost, user_rating, timestamp
                    FROM mcp_performance 
                    WHERE mcp_name = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (mcp_name, cutoff_date)) as cursor:
                    recent_data = await cursor.fetchall()
            
            # Calculate derived metrics
            total_invocations = agg_row[1]
            successful_invocations = agg_row[2]
            success_rate = successful_invocations / total_invocations if total_invocations > 0 else 0
            avg_response_time = agg_row[3] / total_invocations if total_invocations > 0 else 0
            avg_cost = agg_row[4] / total_invocations if total_invocations > 0 else 0
            avg_rating = agg_row[5] / agg_row[6] if agg_row[6] > 0 else 0
            
            # Calculate trends from recent data
            if recent_data:
                recent_success_rate = sum(1 for row in recent_data if row[0]) / len(recent_data)
                recent_avg_time = sum(row[1] for row in recent_data) / len(recent_data)
                trend_success = "improving" if recent_success_rate > success_rate else "declining"
                trend_performance = "improving" if recent_avg_time < avg_response_time else "declining"
            else:
                trend_success = "stable"
                trend_performance = "stable"
            
            # Calculate Pareto score based on RAG-MCP Section 4 methodology
            pareto_score = self._calculate_pareto_score(success_rate, avg_response_time, avg_cost, avg_rating)
            
            metrics = {
                'mcp_name': mcp_name,
                'total_invocations': total_invocations,
                'success_rate': success_rate,
                'avg_response_time_ms': avg_response_time,
                'avg_cost': avg_cost,
                'avg_user_rating': avg_rating,
                'pareto_score': pareto_score,
                'trends': {
                    'success_rate_trend': trend_success,
                    'performance_trend': trend_performance
                },
                'recent_data_points': len(recent_data),
                'timeframe_days': timeframe_days,
                'health_score': self._calculate_health_score(success_rate, avg_response_time, avg_rating)
            }
            
            logger.info("Retrieved MCP performance metrics", extra={
                'operation': 'METRICS_RETRIEVAL',
                'component': 'MCPPerformanceTracker',
                'mcp_name': mcp_name,
                'success_rate': success_rate,
                'pareto_score': pareto_score
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get MCP metrics: {str(e)}", extra={
                'operation': 'METRICS_RETRIEVAL_ERROR',
                'component': 'MCPPerformanceTracker',
                'mcp_name': mcp_name,
                'error': str(e)
            })
            return {'error': str(e)}
    
    def _calculate_pareto_score(self, success_rate: float, avg_response_time: float, 
                              avg_cost: float, avg_rating: float) -> float:
        """
        Calculate Pareto efficiency score based on RAG-MCP Section 4 methodology
        Combines success rate, performance, cost efficiency, and user satisfaction
        """
        # Normalize metrics to 0-1 range
        success_weight = 0.4
        performance_weight = 0.3  # Lower response time is better
        cost_weight = 0.2         # Lower cost is better  
        satisfaction_weight = 0.1
        
        # Success rate is already 0-1
        success_component = success_rate * success_weight
        
        # Performance component (normalized, inverted - lower time is better)
        max_acceptable_time = 5000  # 5 seconds
        performance_component = max(0, 1 - (avg_response_time / max_acceptable_time)) * performance_weight
        
        # Cost component (normalized, inverted - lower cost is better)
        max_acceptable_cost = 1.0  # $1 per operation
        cost_component = max(0, 1 - (avg_cost / max_acceptable_cost)) * cost_weight if avg_cost > 0 else cost_weight
        
        # Satisfaction component (scale from 1-5 to 0-1)
        satisfaction_component = max(0, (avg_rating - 1) / 4) * satisfaction_weight if avg_rating > 0 else 0
        
        pareto_score = success_component + performance_component + cost_component + satisfaction_component
        return min(1.0, pareto_score)  # Cap at 1.0
    
    def _calculate_health_score(self, success_rate: float, avg_response_time: float, 
                              avg_rating: float) -> float:
        """Calculate overall health score for MCP based on key performance indicators"""
        health_factors = []
        
        # Success rate factor
        if success_rate >= 0.95:
            health_factors.append(1.0)
        elif success_rate >= 0.85:
            health_factors.append(0.8)
        elif success_rate >= 0.70:
            health_factors.append(0.6)
        else:
            health_factors.append(0.3)
        
        # Response time factor
        if avg_response_time <= 1000:  # < 1 second
            health_factors.append(1.0)
        elif avg_response_time <= 3000:  # < 3 seconds
            health_factors.append(0.8)
        elif avg_response_time <= 5000:  # < 5 seconds
            health_factors.append(0.6)
        else:
            health_factors.append(0.3)
        
        # User satisfaction factor
        if avg_rating >= 4.5:
            health_factors.append(1.0)
        elif avg_rating >= 3.5:
            health_factors.append(0.8)
        elif avg_rating >= 2.5:
            health_factors.append(0.6)
        elif avg_rating > 0:
            health_factors.append(0.3)
        else:
            health_factors.append(0.5)  # No ratings yet
        
        return sum(health_factors) / len(health_factors) 

class EnhancedVectorStore:
    """
    Advanced vector storage implementing RAG-MCP Section 3.2 external vector index design
    with multi-modal embeddings, enhanced semantic search, and optimized retrieval
    """
    
    def __init__(self, openrouter_api_key: Optional[str] = None,
                 vector_dimensions: int = 1536,
                 similarity_threshold: float = 0.7,
                 cache_directory: Optional[str] = None,
                 enable_multi_modal: bool = True):
        """
        Initialize enhanced vector store with multi-modal embedding support
        
        @param openrouter_api_key OpenRouter API key for embeddings
        @param vector_dimensions Dimensions for embedding vectors  
        @param similarity_threshold Minimum similarity for retrieval
        @param cache_directory Directory for caching indices
        @param enable_multi_modal Enable multi-modal embedding strategies
        """
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        self.vector_dimensions = vector_dimensions
        self.similarity_threshold = similarity_threshold
        self.cache_directory = cache_directory or str(project_root / 'cache' / 'enhanced_vectors')
        self.enable_multi_modal = enable_multi_modal
        
        # Initialize embedding clients
        self.embedding_client = self._initialize_embedding_client()
        self.local_embedding_model = self._initialize_local_embedding_model()
        
        # Vector indices for different embedding types
        self.text_index = None          # Description embeddings
        self.capability_index = None    # Capability embeddings  
        self.metadata_index = None      # Metadata embeddings
        self.composite_index = None     # Combined multi-modal embeddings
        
        # In-memory stores for fast lookup
        self.mcp_by_name = {}
        self.embeddings_cache = {}
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time_ms': 0.0
        }
        
        # Ensure cache directory exists
        Path(self.cache_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized Enhanced Vector Store", extra={
            'operation': 'INIT',
            'component': 'EnhancedVectorStore',
            'multi_modal': enable_multi_modal,
            'cache_dir': self.cache_directory
        })
    
    def _initialize_embedding_client(self) -> Optional[OpenAIEmbeddings]:
        """Initialize OpenRouter embedding client with preferred models"""
        if not self.openrouter_api_key:
            logger.warning("No OpenRouter API key provided, using fallback embeddings")
            return None
        
        try:
            # Use OpenRouter with preferred model (per user memory)
            embedding_client = OpenAIEmbeddings(
                model="openai/text-embedding-ada-002",
                openai_api_key=self.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                headers={
                    "HTTP-Referer": os.getenv('APP_URL', 'http://localhost:3000'),
                    "X-Title": "Alita KGoT Enhanced MCP Knowledge Base"
                }
            )
            
            logger.info("Initialized OpenRouter embedding client", extra={
                'operation': 'EMBED_CLIENT_INIT',
                'component': 'EnhancedVectorStore',
                'model': 'openai/text-embedding-ada-002'
            })
            
            return embedding_client
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter embedding client: {str(e)}", extra={
                'operation': 'EMBED_CLIENT_ERROR',
                'component': 'EnhancedVectorStore',
                'error': str(e)
            })
            return None
    
    def _initialize_local_embedding_model(self):
        """Initialize local embedding model as fallback"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
            
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Initialized local embedding model", extra={
                'operation': 'LOCAL_EMBED_INIT',
                'component': 'EnhancedVectorStore',
                'model': 'all-MiniLM-L6-v2'
            })
            
            return model
            
        except Exception as e:
            logger.warning(f"Failed to initialize local embedding model: {str(e)}")
            return None
    
    async def generate_multi_modal_embeddings(self, mcp_spec: EnhancedMCPSpec) -> Dict[str, np.ndarray]:
        """
        Generate multi-modal embeddings for enhanced semantic understanding
        
        @param mcp_spec MCP specification to embed
        @return Dictionary of embedding types and their vectors
        """
        embeddings = {}
        
        try:
            # Text embedding (description + name)
            text_content = f"{mcp_spec.name}: {mcp_spec.description}"
            text_embedding = await self._generate_embedding(text_content, "text")
            if text_embedding is not None:
                embeddings['text'] = text_embedding
            
            # Capability embedding (structured capability representation)
            if self.enable_multi_modal and mcp_spec.capabilities:
                capability_content = f"Capabilities: {', '.join(mcp_spec.capabilities)}"
                capability_embedding = await self._generate_embedding(capability_content, "capability")
                if capability_embedding is not None:
                    embeddings['capability'] = capability_embedding
            
            # Metadata embedding (category, tags, performance metrics)
            if self.enable_multi_modal:
                metadata_parts = [
                    f"Category: {mcp_spec.category.value}",
                    f"Tags: {', '.join(mcp_spec.tags)}" if mcp_spec.tags else "",
                    f"Quality: {mcp_spec.quality_score.value}",
                    f"Pareto Score: {mcp_spec.pareto_score:.2f}"
                ]
                metadata_content = " ".join(filter(None, metadata_parts))
                metadata_embedding = await self._generate_embedding(metadata_content, "metadata")
                if metadata_embedding is not None:
                    embeddings['metadata'] = metadata_embedding
            
            # Composite embedding (weighted combination of all modalities)
            if len(embeddings) > 1:
                composite_embedding = self._create_composite_embedding(embeddings)
                embeddings['composite'] = composite_embedding
            elif 'text' in embeddings:
                embeddings['composite'] = embeddings['text']
            
            logger.info("Generated multi-modal embeddings", extra={
                'operation': 'EMBEDDING_GENERATION',
                'component': 'EnhancedVectorStore',
                'mcp_name': mcp_spec.name,
                'embedding_types': list(embeddings.keys())
            })
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate multi-modal embeddings: {str(e)}", extra={
                'operation': 'EMBEDDING_ERROR',
                'component': 'EnhancedVectorStore',
                'mcp_name': mcp_spec.name,
                'error': str(e)
            })
            return {}
    
    async def _generate_embedding(self, content: str, embedding_type: str) -> Optional[np.ndarray]:
        """Generate embedding for specific content with caching"""
        # Check cache first
        cache_key = hashlib.md5(f"{content}:{embedding_type}".encode()).hexdigest()
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        embedding = None
        
        # Try OpenRouter first (preferred per user memory)
        if self.embedding_client:
            try:
                embedding_result = await asyncio.to_thread(
                    self.embedding_client.embed_query, content
                )
                embedding = np.array(embedding_result, dtype=np.float32)
                
            except Exception as e:
                logger.warning(f"OpenRouter embedding failed, trying fallback: {str(e)}")
        
        # Fallback to local model
        if embedding is None and self.local_embedding_model:
            try:
                embedding_result = self.local_embedding_model.encode(content)
                embedding = np.array(embedding_result, dtype=np.float32)
                
                # Resize to match expected dimensions if needed
                if len(embedding) != self.vector_dimensions:
                    if len(embedding) > self.vector_dimensions:
                        embedding = embedding[:self.vector_dimensions]
                    else:
                        # Pad with zeros if smaller
                        padding = np.zeros(self.vector_dimensions - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                        
            except Exception as e:
                logger.warning(f"Local embedding failed: {str(e)}")
        
        # Final fallback - mock embedding
        if embedding is None:
            logger.warning(f"All embedding methods failed, using mock embedding for: {content[:50]}...")
            embedding = np.random.normal(0, 0.1, self.vector_dimensions).astype(np.float32)
        
        # Cache the result
        self.embeddings_cache[cache_key] = embedding
        return embedding
    
    def _create_composite_embedding(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create composite embedding by combining different modalities with weights
        Based on importance for semantic search effectiveness
        """
        weights = {
            'text': 0.5,       # Description is most important
            'capability': 0.3,  # Capabilities are key for matching
            'metadata': 0.2     # Metadata provides context
        }
        
        composite = np.zeros(self.vector_dimensions, dtype=np.float32)
        total_weight = 0.0
        
        for embed_type, embedding in embeddings.items():
            if embed_type != 'composite' and embed_type in weights:
                weight = weights[embed_type]
                composite += embedding * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            composite /= total_weight
            
        return composite
    
    async def build_enhanced_indices(self, mcp_specs: List[EnhancedMCPSpec]) -> bool:
        """
        Build enhanced vector indices with multi-modal support
        
        @param mcp_specs List of MCP specifications to index
        @return Success status
        """
        try:
            logger.info("Building enhanced vector indices", extra={
                'operation': 'INDEX_BUILD_START',
                'component': 'EnhancedVectorStore',
                'total_mcps': len(mcp_specs)
            })
            
            # Generate embeddings for all MCPs
            embeddings_data = {
                'text': [],
                'capability': [],
                'metadata': [],
                'composite': []
            }
            
            for mcp_spec in mcp_specs:
                embeddings = await self.generate_multi_modal_embeddings(mcp_spec)
                
                # Update MCP spec with embeddings
                mcp_spec.text_embedding = embeddings.get('text')
                mcp_spec.capability_embedding = embeddings.get('capability')
                mcp_spec.metadata_embedding = embeddings.get('metadata')
                mcp_spec.composite_embedding = embeddings.get('composite')
                mcp_spec.indexed_at = datetime.now()
                
                # Collect embeddings for index building
                for embed_type in embeddings_data.keys():
                    if embed_type in embeddings:
                        embeddings_data[embed_type].append((mcp_spec.name, embeddings[embed_type]))
                
                # Store in lookup cache
                self.mcp_by_name[mcp_spec.name] = mcp_spec
            
            # Build FAISS indices if available
            if FAISS_AVAILABLE:
                await self._build_faiss_indices(embeddings_data)
            else:
                await self._build_fallback_indices(embeddings_data)
            
            # Cache indices to disk
            await self._cache_indices(mcp_specs)
            
            logger.info("Successfully built enhanced vector indices", extra={
                'operation': 'INDEX_BUILD_COMPLETE',
                'component': 'EnhancedVectorStore',
                'indexed_mcps': len(mcp_specs),
                'embedding_types': list(embeddings_data.keys())
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build enhanced indices: {str(e)}", extra={
                'operation': 'INDEX_BUILD_ERROR',
                'component': 'EnhancedVectorStore',
                'error': str(e)
            })
            return False
    
    async def _build_faiss_indices(self, embeddings_data: Dict[str, List[Tuple[str, np.ndarray]]]):
        """Build FAISS indices for each embedding type"""
        for embed_type, embeddings in embeddings_data.items():
            if not embeddings:
                continue
                
            # Extract embeddings and create index
            embedding_vectors = np.array([emb[1] for emb in embeddings], dtype=np.float32)
            embedding_names = [emb[0] for emb in embeddings]
            
            # Create FAISS index
            index = faiss.IndexFlatIP(self.vector_dimensions)  # Inner product for cosine similarity
            faiss.normalize_L2(embedding_vectors)  # Normalize for cosine similarity
            index.add(embedding_vectors)
            
            # Store index and name mapping
            setattr(self, f"{embed_type}_index", {
                'faiss_index': index,
                'name_mapping': embedding_names,
                'type': 'faiss'
            })
            
            logger.info(f"Built FAISS index for {embed_type}", extra={
                'operation': 'FAISS_INDEX_BUILD',
                'component': 'EnhancedVectorStore',
                'embed_type': embed_type,
                'size': len(embeddings)
            })
    
    async def _build_fallback_indices(self, embeddings_data: Dict[str, List[Tuple[str, np.ndarray]]]):
        """Build fallback indices using numpy for similarity search"""
        for embed_type, embeddings in embeddings_data.items():
            if not embeddings:
                continue
                
            embedding_vectors = np.array([emb[1] for emb in embeddings], dtype=np.float32)
            embedding_names = [emb[0] for emb in embeddings]
            
            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(embedding_vectors, axis=1, keepdims=True)
            embedding_vectors = embedding_vectors / np.maximum(norms, 1e-8)
            
            setattr(self, f"{embed_type}_index", {
                'vectors': embedding_vectors,
                'name_mapping': embedding_names,
                'type': 'fallback'
            })
            
            logger.info(f"Built fallback index for {embed_type}", extra={
                'operation': 'FALLBACK_INDEX_BUILD',
                'component': 'EnhancedVectorStore',
                'embed_type': embed_type,
                'size': len(embeddings)
            })

    async def _cache_indices(self, mcp_specs: List[EnhancedMCPSpec]):
        """Cache vector indices to disk"""
        try:
            # Serialize and save vector indices
            for mcp_spec in mcp_specs:
                for embed_type, index_data in self.__dict__.items():
                    if embed_type.endswith('_index'):
                        index_name = embed_type[:-5]  # Extract index name from attribute name
                        if index_name in mcp_spec.__dict__:
                            continue  # Skip if already set
                        mcp_spec.__dict__[index_name] = index_data['name_mapping']
            
            # Save to file
            index_file = f"{self.cache_directory}/{'_'.join([mcp_spec.name for mcp_spec in mcp_specs])}_indices.pkl"
            with open(index_file, 'wb') as f:
                pickle.dump(mcp_specs, f)
            
            logger.info("Successfully cached vector indices", extra={
                'operation': 'INDEX_CACHE',
                'component': 'EnhancedVectorStore',
                'total_mcps': len(mcp_specs)
            })
            
        except Exception as e:
            logger.error(f"Failed to cache vector indices: {str(e)}", extra={
                'operation': 'INDEX_CACHE_ERROR',
                'component': 'EnhancedVectorStore',
                'error': str(e)
            })
    
    async def enhanced_semantic_search(self, query: str, top_k: int = 5,
                                     search_modes: List[str] = None,
                                     filter_category: Optional[MCPCategory] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform enhanced semantic search across multiple embedding modalities
        
        @param query Search query string
        @param top_k Number of top results to return
        @param search_modes List of embedding types to search ['text', 'capability', 'metadata', 'composite']
        @param filter_category Optional category filter
        @return List of (mcp_name, similarity_score, search_metadata) tuples
        """
        search_start = time.time()
        
        try:
            if search_modes is None:
                search_modes = ['composite', 'text', 'capability']  # Default search modes
            
            # Generate query embeddings
            query_embeddings = {}
            for mode in search_modes:
                if mode == 'composite':
                    # For composite search, combine query representations
                    text_emb = await self._generate_embedding(query, "text")
                    cap_emb = await self._generate_embedding(f"Capabilities needed: {query}", "capability")
                    meta_emb = await self._generate_embedding(f"Task type: {query}", "metadata")
                    
                    composite_emb = self._create_composite_embedding({
                        'text': text_emb, 'capability': cap_emb, 'metadata': meta_emb
                    })
                    query_embeddings['composite'] = composite_emb
                else:
                    query_embeddings[mode] = await self._generate_embedding(query, mode)
            
            # Perform search across different modalities
            all_results = {}
            
            for mode, query_embedding in query_embeddings.items():
                if query_embedding is None:
                    continue
                    
                index_attr = f"{mode}_index"
                if hasattr(self, index_attr):
                    index_data = getattr(self, index_attr)
                    mode_results = await self._search_index(query_embedding, index_data, top_k, mode)
                    
                    # Weight results by mode importance
                    mode_weights = {'composite': 1.0, 'text': 0.8, 'capability': 0.9, 'metadata': 0.6}
                    weight = mode_weights.get(mode, 1.0)
                    
                    for mcp_name, similarity in mode_results:
                        if mcp_name not in all_results:
                            all_results[mcp_name] = {
                                'scores': {},
                                'weighted_score': 0.0,
                                'search_metadata': {'modes_found': []}
                            }
                        
                        all_results[mcp_name]['scores'][mode] = similarity
                        all_results[mcp_name]['search_metadata']['modes_found'].append(mode)
                        
                        # Update weighted score
                        current_weight = all_results[mcp_name]['weighted_score']
                        all_results[mcp_name]['weighted_score'] = max(current_weight, similarity * weight)
            
            # Apply category filter if specified
            if filter_category:
                filtered_results = {}
                for mcp_name, result_data in all_results.items():
                    if mcp_name in self.mcp_by_name:
                        mcp_spec = self.mcp_by_name[mcp_name]
                        if mcp_spec.category == filter_category:
                            filtered_results[mcp_name] = result_data
                all_results = filtered_results
            
            # Sort by weighted score and return top_k
            sorted_results = sorted(
                all_results.items(), 
                key=lambda x: x[1]['weighted_score'], 
                reverse=True
            )[:top_k]
            
            # Format results
            final_results = []
            for mcp_name, result_data in sorted_results:
                similarity_score = result_data['weighted_score']
                search_metadata = result_data['search_metadata']
                search_metadata['individual_scores'] = result_data['scores']
                
                final_results.append((mcp_name, similarity_score, search_metadata))
            
            # Update search statistics
            search_time = (time.time() - search_start) * 1000
            self.search_stats['total_searches'] += 1
            current_avg = self.search_stats['avg_search_time_ms']
            total_searches = self.search_stats['total_searches']
            self.search_stats['avg_search_time_ms'] = (
                (current_avg * (total_searches - 1) + search_time) / total_searches
            )
            
            logger.info("Completed enhanced semantic search", extra={
                'operation': 'SEMANTIC_SEARCH',
                'component': 'EnhancedVectorStore',
                'query': query[:50],
                'results_count': len(final_results),
                'search_time_ms': search_time,
                'search_modes': search_modes
            })
            
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced semantic search failed: {str(e)}", extra={
                'operation': 'SEMANTIC_SEARCH_ERROR',
                'component': 'EnhancedVectorStore',
                'query': query[:50],
                'error': str(e)
            })
            return []
    
    async def _search_index(self, query_embedding: np.ndarray, index_data: Dict[str, Any], 
                          top_k: int, search_mode: str) -> List[Tuple[str, float]]:
        """Search specific index for similar vectors"""
        try:
            if index_data['type'] == 'faiss':
                return await self._search_faiss_index(query_embedding, index_data, top_k)
            else:
                return await self._search_fallback_index(query_embedding, index_data, top_k)
                
        except Exception as e:
            logger.warning(f"Index search failed for mode {search_mode}: {str(e)}")
            return []
    
    async def _search_faiss_index(self, query_embedding: np.ndarray, index_data: Dict[str, Any], 
                                top_k: int) -> List[Tuple[str, float]]:
        """Search FAISS index"""
        faiss_index = index_data['faiss_index']
        name_mapping = index_data['name_mapping']
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search index
        similarities, indices = faiss_index.search(query_vector, min(top_k, len(name_mapping)))
        
        results = []
        for i, similarity in zip(indices[0], similarities[0]):
            if i >= 0 and similarity >= self.similarity_threshold:
                mcp_name = name_mapping[i]
                results.append((mcp_name, float(similarity)))
        
        return results
    
    async def _search_fallback_index(self, query_embedding: np.ndarray, index_data: Dict[str, Any], 
                                   top_k: int) -> List[Tuple[str, float]]:
        """Search fallback numpy index"""
        vectors = index_data['vectors']
        name_mapping = index_data['name_mapping']
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Calculate cosine similarities
        similarities = np.dot(vectors, query_embedding)
        
        # Get top_k results above threshold
        valid_indices = np.where(similarities >= self.similarity_threshold)[0]
        if len(valid_indices) == 0:
            return []
        
        valid_similarities = similarities[valid_indices]
        sorted_indices = valid_indices[np.argsort(valid_similarities)[::-1]][:top_k]
        
        results = []
        for idx in sorted_indices:
            mcp_name = name_mapping[idx]
            similarity = similarities[idx]
            results.append((mcp_name, float(similarity)))
        
        return results

class MCPCuratorAgent:
    """
    LangChain-based intelligent MCP curation agent implementing Pareto optimization
    and quality assessment following RAG-MCP principles
    """
    
    def __init__(self, openrouter_api_key: Optional[str] = None):
        """
        Initialize MCP Curator Agent with LangChain tools and OpenRouter LLM
        
        @param openrouter_api_key OpenRouter API key for LLM operations
        """
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        self.llm_client = self._initialize_llm_client()
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 interactions
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize curation agent with specialized tools
        self.curation_agent = self._initialize_curation_agent()
        
        # Curation statistics
        self.curation_stats = {
            'total_evaluations': 0,
            'quality_improvements': 0,
            'pareto_optimizations': 0,
            'duplicates_detected': 0
        }
        
        logger.info("Initialized MCP Curator Agent", extra={
            'operation': 'INIT',
            'component': 'MCPCuratorAgent',
            'has_llm': self.llm_client is not None
        })
    
    def _initialize_llm_client(self) -> Optional[ChatOpenAI]:
        """Initialize OpenRouter LLM client for curation operations"""
        if not self.openrouter_api_key:
            logger.warning("No OpenRouter API key provided for MCP Curator Agent")
            return None
        
        try:
            llm_client = ChatOpenAI(
                model="anthropic/claude-sonnet-4",  # Use Claude-3-Sonnet for validation
                openai_api_key=self.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                headers={
                    "HTTP-Referer": os.getenv('APP_URL', 'http://localhost:3000'),
                    "X-Title": "Alita KGoT Enhanced MCP Curator"
                },
                temperature=0.1,  # Low temperature for consistent curation
                max_tokens=2000
            )
            
            logger.info("Initialized claude-sonnet-4 for MCP curation", extra={
                'operation': 'LLM_INIT',
                'component': 'MCPCuratorAgent',
                'model': 'anthropic/claude-sonnet-4'
            })
            
            return llm_client
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}", extra={
                'operation': 'LLM_INIT_ERROR',
                'component': 'MCPCuratorAgent',
                'error': str(e)
            })
            return None
    
    def _initialize_curation_agent(self) -> Optional[AgentExecutor]:
        """Initialize LangChain agent with specialized MCP curation tools"""
        if not self.llm_client:
            return None
        
        try:
            # Create specialized tools for MCP curation
            tools = [
                self._create_quality_assessment_tool(),
                self._create_capability_analysis_tool(), 
                self._create_pareto_evaluation_tool(),
                self._create_duplicate_detection_tool(),
                self._create_category_classification_tool()
            ]
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert MCP (Model Control Protocol) curator responsible for maintaining a high-quality knowledge base of MCP specifications. Your role is to:

1. Assess MCP quality and completeness following RAG-MCP Section 3.2 standards
2. Analyze capabilities and ensure proper categorization  
3. Apply Pareto optimization principles from RAG-MCP Section 4
4. Detect duplicate or overlapping MCPs for consolidation
5. Provide actionable recommendations for improvement

When evaluating MCPs, consider:
- Description clarity and completeness
- Capability specification accuracy
- Category alignment with functionality
- Pareto efficiency (80/20 principle)
- Performance metrics and reliability
- Integration potential with existing MCPs

Always provide specific, actionable feedback with reasoning based on RAG-MCP framework principles."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Create agent
            agent = create_openai_functions_agent(
                llm=self.llm_client,
                tools=tools,
                prompt=prompt
            )
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=self.memory,
                verbose=True,
                max_iterations=5,
                handle_parsing_errors=True
            )
            
            logger.info("Initialized MCP curation agent", extra={
                'operation': 'AGENT_INIT',
                'component': 'MCPCuratorAgent',
                'tools_count': len(tools)
            })
            
            return agent_executor
            
        except Exception as e:
            logger.error(f"Failed to initialize curation agent: {str(e)}", extra={
                'operation': 'AGENT_INIT_ERROR',
                'component': 'MCPCuratorAgent',
                'error': str(e)
            })
            return None
    
    def _create_quality_assessment_tool(self) -> Tool:
        """Create tool for assessing MCP quality and completeness"""
        def assess_quality(mcp_data: str) -> str:
            """Assess MCP quality based on description, capabilities, and metadata completeness"""
            try:
                mcp_info = json.loads(mcp_data) if isinstance(mcp_data, str) else mcp_data
                
                quality_factors = []
                
                # Description quality (30%)
                description = mcp_info.get('description', '')
                if len(description) >= 50:
                    quality_factors.append(0.9 if len(description) >= 100 else 0.7)
                else:
                    quality_factors.append(0.3)
                
                # Capabilities completeness (25%)
                capabilities = mcp_info.get('capabilities', [])
                if len(capabilities) >= 3:
                    quality_factors.append(0.9)
                elif len(capabilities) >= 1:
                    quality_factors.append(0.6)
                else:
                    quality_factors.append(0.2)
                
                # Category appropriateness (15%)
                category = mcp_info.get('category', '')
                if category in ['web_information', 'data_processing', 'communication', 'development']:
                    quality_factors.append(0.8)
                else:
                    quality_factors.append(0.4)
                
                # Metadata completeness (15%)
                has_tags = bool(mcp_info.get('tags', []))
                has_author = bool(mcp_info.get('author', ''))
                metadata_score = (has_tags + has_author) / 2
                quality_factors.append(metadata_score)
                
                # Performance metrics availability (15%)
                performance_metrics = mcp_info.get('performance_metrics', {})
                if performance_metrics and len(performance_metrics) >= 3:
                    quality_factors.append(0.8)
                else:
                    quality_factors.append(0.3)
                
                overall_score = sum(quality_factors) / len(quality_factors)
                
                # Determine quality rating
                if overall_score >= 0.9:
                    rating = "EXCELLENT"
                elif overall_score >= 0.75:
                    rating = "GOOD"
                elif overall_score >= 0.6:
                    rating = "ACCEPTABLE"
                elif overall_score >= 0.4:
                    rating = "POOR"
                else:
                    rating = "UNACCEPTABLE"
                
                return json.dumps({
                    'quality_score': overall_score,
                    'quality_rating': rating,
                    'assessment_factors': {
                        'description_quality': quality_factors[0],
                        'capabilities_completeness': quality_factors[1],
                        'category_appropriateness': quality_factors[2],
                        'metadata_completeness': quality_factors[3],
                        'performance_availability': quality_factors[4]
                    },
                    'recommendations': self._generate_quality_recommendations(mcp_info, quality_factors)
                })
                
            except Exception as e:
                return json.dumps({'error': f'Quality assessment failed: {str(e)}'})
        
        return Tool(
            name="assess_mcp_quality",
            description="Assess the quality and completeness of an MCP specification",
            func=assess_quality
        )
    
    def _generate_quality_recommendations(self, mcp_info: Dict[str, Any], 
                                        quality_factors: List[float]) -> List[str]:
        """Generate specific recommendations for improving MCP quality"""
        recommendations = []
        
        if quality_factors[0] < 0.7:  # Description quality
            recommendations.append("Improve description clarity and add more detail about functionality")
        
        if quality_factors[1] < 0.6:  # Capabilities
            recommendations.append("Add more specific capabilities or improve capability descriptions")
        
        if quality_factors[2] < 0.6:  # Category
            recommendations.append("Review and correct MCP category assignment")
        
        if quality_factors[3] < 0.5:  # Metadata
            recommendations.append("Add tags and author information for better discoverability")
        
        if quality_factors[4] < 0.5:  # Performance
            recommendations.append("Collect and add performance metrics data")
        
        return recommendations

    def _create_capability_analysis_tool(self) -> Tool:
        """Create tool for analyzing MCP capabilities"""
        def analyze_capabilities(mcp_data: str) -> str:
            """Analyze MCP capabilities and provide recommendations"""
            try:
                mcp_info = json.loads(mcp_data) if isinstance(mcp_data, str) else mcp_data
                
                # Implementation of capability analysis logic
                # This is a placeholder and should be replaced with actual implementation
                analysis_result = "Capability analysis logic not implemented yet."
                
                return json.dumps({
                    'analysis_result': analysis_result
                })
                
            except Exception as e:
                return json.dumps({'error': f'Capability analysis failed: {str(e)}'})
        
        return Tool(
            name="analyze_mcp_capabilities",
            description="Analyze the capabilities of an MCP specification",
            func=analyze_capabilities
        )

    def _create_pareto_evaluation_tool(self) -> Tool:
        """Create tool for evaluating MCP Pareto efficiency"""
        def evaluate_pareto(mcp_data: str) -> str:
            """Evaluate MCP Pareto efficiency and provide recommendations"""
            try:
                mcp_info = json.loads(mcp_data) if isinstance(mcp_data, str) else mcp_data
                
                # Implementation of Pareto evaluation logic
                # This is a placeholder and should be replaced with actual implementation
                evaluation_result = "Pareto evaluation logic not implemented yet."
                
                return json.dumps({
                    'evaluation_result': evaluation_result
                })
                
            except Exception as e:
                return json.dumps({'error': f'Pareto evaluation failed: {str(e)}'})
        
        return Tool(
            name="evaluate_mcp_pareto",
            description="Evaluate the Pareto efficiency of an MCP specification",
            func=evaluate_pareto
        )

    def _create_duplicate_detection_tool(self) -> Tool:
        """Create tool for detecting duplicate MCPs"""
        def detect_duplicates(mcp_data: str) -> str:
            """Detect duplicate MCPs and provide recommendations"""
            try:
                mcp_info = json.loads(mcp_data) if isinstance(mcp_data, str) else mcp_data
                
                # Implementation of duplicate detection logic
                # This is a placeholder and should be replaced with actual implementation
                detection_result = "Duplicate detection logic not implemented yet."
                
                return json.dumps({
                    'detection_result': detection_result
                })
                
            except Exception as e:
                return json.dumps({'error': f'Duplicate detection failed: {str(e)}'})
        
        return Tool(
            name="detect_mcp_duplicates",
            description="Detect duplicate MCPs in the knowledge base",
            func=detect_duplicates
        )

    def _create_category_classification_tool(self) -> Tool:
        """Create tool for classifying MCP category"""
        def classify_category(mcp_data: str) -> str:
            """Classify MCP category and provide recommendations"""
            try:
                mcp_info = json.loads(mcp_data) if isinstance(mcp_data, str) else mcp_data
                
                # Implementation of category classification logic
                # This is a placeholder and should be replaced with actual implementation
                classification_result = "Category classification logic not implemented yet."
                
                return json.dumps({
                    'classification_result': classification_result
                })
                
            except Exception as e:
                return json.dumps({'error': f'Category classification failed: {str(e)}'})
        
        return Tool(
            name="classify_mcp_category",
            description="Classify the category of an MCP specification",
            func=classify_category
        )

class MCPKnowledgeBase:
    """
    Main MCP Knowledge Base Builder implementing RAG-MCP Section 3.2 framework
    with enhanced semantic indexing, intelligent curation, and performance tracking
    """
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 vector_dimensions: int = 1536,
                 similarity_threshold: float = 0.7,
                 cache_directory: Optional[str] = None,
                 enable_multi_modal: bool = True,
                 enable_curation: bool = True,
                 rag_engine: Optional[RAGMCPEngine] = None):
        """
        Initialize MCP Knowledge Base with all enhanced components
        
        @param openrouter_api_key OpenRouter API key for embeddings and LLM
        @param vector_dimensions Vector dimensions for embeddings
        @param similarity_threshold Minimum similarity for retrieval
        @param cache_directory Cache directory for indices and data
        @param enable_multi_modal Enable multi-modal embedding strategies
        @param enable_curation Enable intelligent MCP curation
        @param rag_engine Optional existing RAG-MCP Engine for integration
        """
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        self.cache_directory = cache_directory or str(project_root / 'cache' / 'mcp_knowledge_base')
        self.enable_curation = enable_curation
        self.rag_engine = rag_engine
        
        # Ensure cache directory exists
        Path(self.cache_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.performance_tracker = MCPPerformanceTracker(
            db_path=str(Path(self.cache_directory) / 'performance.db')
        )
        
        self.vector_store = EnhancedVectorStore(
            openrouter_api_key=self.openrouter_api_key,
            vector_dimensions=vector_dimensions,
            similarity_threshold=similarity_threshold,
            cache_directory=str(Path(self.cache_directory) / 'vectors'),
            enable_multi_modal=enable_multi_modal
        )
        
        self.curator_agent = MCPCuratorAgent(
            openrouter_api_key=self.openrouter_api_key
        ) if enable_curation else None
        
        # Knowledge base storage
        self.mcp_registry: Dict[str, EnhancedMCPSpec] = {}
        self.change_log: List[MCPChangeRecord] = []
        
        # Performance tracking
        self.kb_metrics = KnowledgeBaseMetrics()
        self.initialization_time = None
        
        logger.info("Initialized MCP Knowledge Base", extra={
            'operation': 'INIT',
            'component': 'MCPKnowledgeBase',
            'cache_dir': self.cache_directory,
            'multi_modal': enable_multi_modal,
            'curation_enabled': enable_curation
        })
    
    async def initialize(self) -> bool:
        """
        Initialize the knowledge base with all components and indices
        
        @return Success status
        """
        start_time = time.time()
        
        try:
            logger.info("Initializing MCP Knowledge Base", extra={
                'operation': 'KB_INIT_START',
                'component': 'MCPKnowledgeBase'
            })
            
            # Initialize performance tracker storage
            await self.performance_tracker.initialize_storage()
            
            # Load existing MCPs from cache or create initial set
            await self._load_or_create_initial_mcps()
            
            # Build vector indices
            if self.mcp_registry:
                mcp_specs = list(self.mcp_registry.values())
                success = await self.vector_store.build_enhanced_indices(mcp_specs)
                if not success:
                    logger.warning("Failed to build vector indices, continuing with limited functionality")
            
            # Update knowledge base metrics
            await self._update_kb_metrics()
            
            # Integration with existing RAG-MCP Engine if provided
            if self.rag_engine:
                await self._integrate_with_rag_engine()
            
            self.initialization_time = time.time() - start_time
            
            logger.info("Successfully initialized MCP Knowledge Base", extra={
                'operation': 'KB_INIT_COMPLETE',
                'component': 'MCPKnowledgeBase',
                'total_mcps': len(self.mcp_registry),
                'init_time_ms': self.initialization_time * 1000
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Knowledge Base: {str(e)}", extra={
                'operation': 'KB_INIT_ERROR',
                'component': 'MCPKnowledgeBase',
                'error': str(e)
            })
            return False
    
    async def _load_or_create_initial_mcps(self):
        """Load existing MCPs from cache or create initial high-value set"""
        cache_file = Path(self.cache_directory) / 'mcp_registry.json'
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                for mcp_data in cached_data:
                    mcp_spec = self._deserialize_mcp_spec(mcp_data)
                    self.mcp_registry[mcp_spec.name] = mcp_spec
                
                logger.info(f"Loaded {len(self.mcp_registry)} MCPs from cache", extra={
                    'operation': 'CACHE_LOAD',
                    'component': 'MCPKnowledgeBase'
                })
                
            except Exception as e:
                logger.warning(f"Failed to load cache, creating initial MCPs: {str(e)}")
                await self._create_initial_high_value_mcps()
        else:
            await self._create_initial_high_value_mcps()
    
    async def _create_initial_high_value_mcps(self):
        """Create initial set of high-value MCPs based on Pareto principles"""
        # Import from existing RAG-MCP Engine if available
        if self.rag_engine and hasattr(self.rag_engine, 'pareto_registry'):
            existing_mcps = self.rag_engine.pareto_registry.mcps
            
            for mcp_spec in existing_mcps:
                enhanced_spec = self._convert_to_enhanced_spec(mcp_spec)
                self.mcp_registry[enhanced_spec.name] = enhanced_spec
            
            logger.info(f"Imported {len(existing_mcps)} MCPs from RAG-MCP Engine", extra={
                'operation': 'IMPORT_FROM_RAG',
                'component': 'MCPKnowledgeBase'
            })
        else:
            # Create basic high-value MCPs if no existing registry
            initial_mcps = self._get_default_high_value_mcps()
            for mcp_spec in initial_mcps:
                self.mcp_registry[mcp_spec.name] = mcp_spec
            
            logger.info(f"Created {len(initial_mcps)} default high-value MCPs", extra={
                'operation': 'CREATE_DEFAULT',
                'component': 'MCPKnowledgeBase'
            })
    
    def _convert_to_enhanced_spec(self, legacy_spec: MCPToolSpec) -> EnhancedMCPSpec:
        """Convert legacy MCPToolSpec to enhanced specification"""
        return EnhancedMCPSpec(
            name=legacy_spec.name,
            description=legacy_spec.description,
            capabilities=legacy_spec.capabilities,
            category=legacy_spec.category,
            usage_frequency=legacy_spec.usage_frequency,
            reliability_score=legacy_spec.reliability_score,
            cost_efficiency=legacy_spec.cost_efficiency,
            pareto_score=legacy_spec.pareto_score,
            tags=legacy_spec.metadata.get('tags', []) if legacy_spec.metadata else [],
            composite_embedding=legacy_spec.embedding,
            indexed_at=legacy_spec.indexed_at
        )
    
    def _get_default_high_value_mcps(self) -> List[EnhancedMCPSpec]:
        """Get default set of high-value MCPs following Pareto principles"""
        return [
            EnhancedMCPSpec(
                name="web_scraper_enhanced",
                description="Advanced web scraping with AI-powered content extraction and structure analysis",
                capabilities=["html_parsing", "content_extraction", "data_cleaning", "structure_analysis"],
                category=MCPCategory.WEB_INFORMATION,
                pareto_score=0.92,
                tags=["web", "scraping", "data_extraction", "high_value"],
                quality_score=MCPQualityScore.EXCELLENT
            ),
            EnhancedMCPSpec(
                name="python_executor_secure",
                description="Secure Python code execution with sandboxing and comprehensive error handling",
                capabilities=["code_execution", "sandbox_security", "error_handling", "result_validation"],
                category=MCPCategory.DEVELOPMENT,
                pareto_score=0.95,
                tags=["python", "execution", "security", "development"],
                quality_score=MCPQualityScore.EXCELLENT
            ),
            EnhancedMCPSpec(
                name="data_analyzer_smart", 
                description="Intelligent data analysis with automated insights and visualization generation",
                capabilities=["data_analysis", "pattern_detection", "visualization", "insight_generation"],
                category=MCPCategory.DATA_PROCESSING,
                pareto_score=0.88,
                tags=["data", "analysis", "visualization", "insights"],
                quality_score=MCPQualityScore.GOOD
            )
        ]
    
    async def add_mcp(self, mcp_spec: Union[EnhancedMCPSpec, Dict[str, Any]], 
                     auto_curate: bool = True) -> bool:
        """
        Add new MCP to knowledge base with optional intelligent curation
        
        @param mcp_spec MCP specification (enhanced or dictionary)
        @param auto_curate Enable automatic curation and quality assessment
        @return Success status
        """
        try:
            # Convert dictionary to EnhancedMCPSpec if needed
            if isinstance(mcp_spec, dict):
                mcp_spec = EnhancedMCPSpec(**mcp_spec)
            
            # Check for duplicates
            if mcp_spec.name in self.mcp_registry:
                logger.warning(f"MCP {mcp_spec.name} already exists", extra={
                    'operation': 'ADD_MCP_DUPLICATE',
                    'component': 'MCPKnowledgeBase',
                    'mcp_name': mcp_spec.name
                })
                return False
            
            # Perform intelligent curation if enabled
            if auto_curate and self.curator_agent and self.curator_agent.curation_agent:
                curation_result = await self._curate_mcp(mcp_spec)
                if curation_result:
                    # Apply curation improvements
                    mcp_spec = self._apply_curation_improvements(mcp_spec, curation_result)
            
            # Generate embeddings
            embeddings = await self.vector_store.generate_multi_modal_embeddings(mcp_spec)
            mcp_spec.text_embedding = embeddings.get('text')
            mcp_spec.capability_embedding = embeddings.get('capability')
            mcp_spec.metadata_embedding = embeddings.get('metadata')
            mcp_spec.composite_embedding = embeddings.get('composite')
            mcp_spec.indexed_at = datetime.now()
            
            # Add to registry
            self.mcp_registry[mcp_spec.name] = mcp_spec
            
            # Update vector store
            await self.vector_store.build_enhanced_indices([mcp_spec])
            
            # Record change
            change_record = MCPChangeRecord(
                change_id=str(uuid.uuid4()),
                mcp_name=mcp_spec.name,
                operation=MCPUpdateOperation.CREATE,
                changes=asdict(mcp_spec),
                reason="New MCP addition",
                author="system"
            )
            self.change_log.append(change_record)
            
            # Update metrics
            await self._update_kb_metrics()
            
            # Save to cache
            await self._save_registry_cache()
            
            logger.info(f"Successfully added MCP {mcp_spec.name}", extra={
                'operation': 'ADD_MCP_SUCCESS',
                'component': 'MCPKnowledgeBase',
                'mcp_name': mcp_spec.name,
                'quality_score': mcp_spec.quality_score.value
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add MCP: {str(e)}", extra={
                'operation': 'ADD_MCP_ERROR',
                'component': 'MCPKnowledgeBase',
                'error': str(e)
            })
            return False
    
    async def search_mcps(self, query: str, max_results: int = 5,
                         filter_category: Optional[MCPCategory] = None,
                         include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Search MCPs using enhanced semantic search with multi-modal embeddings
        
        @param query Search query string
        @param max_results Maximum number of results to return
        @param filter_category Optional category filter
        @param include_metadata Include detailed search metadata
        @return List of search results with MCP details and scores
        """
        try:
            # Perform enhanced semantic search
            search_results = await self.vector_store.enhanced_semantic_search(
                query=query,
                top_k=max_results,
                filter_category=filter_category
            )
            
            # Format results with MCP details
            formatted_results = []
            for mcp_name, similarity_score, search_metadata in search_results:
                if mcp_name in self.mcp_registry:
                    mcp_spec = self.mcp_registry[mcp_name]
                    
                    # Get performance metrics
                    performance_metrics = await self.performance_tracker.get_mcp_metrics(mcp_name)
                    
                    result = {
                        'name': mcp_spec.name,
                        'description': mcp_spec.description,
                        'capabilities': mcp_spec.capabilities,
                        'category': mcp_spec.category.value,
                        'quality_score': mcp_spec.quality_score.value,
                        'pareto_score': mcp_spec.pareto_score,
                        'similarity_score': similarity_score,
                        'tags': mcp_spec.tags,
                        'performance_summary': {
                            'success_rate': performance_metrics.get('success_rate', 0.0),
                            'avg_response_time_ms': performance_metrics.get('avg_response_time_ms', 0.0),
                            'health_score': performance_metrics.get('health_score', 0.0)
                        }
                    }
                    
                    if include_metadata:
                        result['search_metadata'] = search_metadata
                        result['version'] = mcp_spec.version
                        result['author'] = mcp_spec.author
                        result['created_at'] = mcp_spec.created_at.isoformat()
                        result['updated_at'] = mcp_spec.updated_at.isoformat()
                    
                    formatted_results.append(result)
            
            logger.info(f"Search completed for query: {query[:50]}", extra={
                'operation': 'SEARCH_MCPS',
                'component': 'MCPKnowledgeBase',
                'query': query[:50],
                'results_count': len(formatted_results)
            })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"MCP search failed: {str(e)}", extra={
                'operation': 'SEARCH_ERROR',
                'component': 'MCPKnowledgeBase',
                'query': query[:50],
                'error': str(e)
            })
            return []
    
    async def get_knowledge_base_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics about the knowledge base performance and health
        
        @return Analytics dictionary with metrics and insights
        """
        try:
            # Update current metrics
            await self._update_kb_metrics()
            
            # Get top performing MCPs
            top_mcps = []
            for mcp_name in self.mcp_registry.keys():
                metrics = await self.performance_tracker.get_mcp_metrics(mcp_name)
                if 'error' not in metrics:
                    top_mcps.append({
                        'name': mcp_name,
                        'pareto_score': metrics.get('pareto_score', 0.0),
                        'health_score': metrics.get('health_score', 0.0),
                        'success_rate': metrics.get('success_rate', 0.0)
                    })
            
            top_mcps.sort(key=lambda x: x['pareto_score'], reverse=True)
            
            # Calculate quality distribution
            quality_distribution = {}
            for mcp_spec in self.mcp_registry.values():
                quality = mcp_spec.quality_score.value
                quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
            
            # Search performance statistics
            search_stats = self.vector_store.search_stats.copy()
            
            analytics = {
                'knowledge_base_metrics': asdict(self.kb_metrics),
                'top_performing_mcps': top_mcps[:10],
                'quality_distribution': quality_distribution,
                'search_performance': search_stats,
                'curation_statistics': self.curator_agent.curation_stats if self.curator_agent else {},
                'total_change_records': len(self.change_log),
                'initialization_time_ms': (self.initialization_time * 1000) if self.initialization_time else 0,
                'cache_directory': self.cache_directory,
                'multi_modal_enabled': self.vector_store.enable_multi_modal
            }
            
            logger.info("Generated knowledge base analytics", extra={
                'operation': 'ANALYTICS',
                'component': 'MCPKnowledgeBase',
                'total_mcps': len(self.mcp_registry)
            })
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate analytics: {str(e)}", extra={
                'operation': 'ANALYTICS_ERROR',
                'component': 'MCPKnowledgeBase',
                'error': str(e)
            })
            return {'error': str(e)}
    
    async def _update_kb_metrics(self):
        """Update knowledge base metrics"""
        try:
            self.kb_metrics.total_mcps = len(self.mcp_registry)
            self.kb_metrics.last_update = datetime.now()
            
            # Count by category
            self.kb_metrics.mcps_by_category = {}
            for mcp_spec in self.mcp_registry.values():
                category = mcp_spec.category
                self.kb_metrics.mcps_by_category[category] = (
                    self.kb_metrics.mcps_by_category.get(category, 0) + 1
                )
            
            # Count by quality
            self.kb_metrics.mcps_by_quality = {}
            for mcp_spec in self.mcp_registry.values():
                quality = mcp_spec.quality_score
                self.kb_metrics.mcps_by_quality[quality] = (
                    self.kb_metrics.mcps_by_quality.get(quality, 0) + 1
                )
            
            # Calculate average Pareto score
            if self.mcp_registry:
                total_pareto = sum(mcp.pareto_score for mcp in self.mcp_registry.values())
                self.kb_metrics.average_pareto_score = total_pareto / len(self.mcp_registry)
            
            # Calculate health score
            self.kb_metrics.health_score = self._calculate_kb_health_score()
            
        except Exception as e:
            logger.warning(f"Failed to update KB metrics: {str(e)}")
    
    def _calculate_kb_health_score(self) -> float:
        """Calculate overall knowledge base health score"""
        if not self.mcp_registry:
            return 0.0
        
        health_factors = []
        
        # Quality distribution factor
        excellent_count = sum(1 for mcp in self.mcp_registry.values() 
                            if mcp.quality_score == MCPQualityScore.EXCELLENT)
        quality_ratio = excellent_count / len(self.mcp_registry)
        health_factors.append(quality_ratio)
        
        # Average Pareto score factor
        health_factors.append(self.kb_metrics.average_pareto_score)
        
        # Category coverage factor (balanced distribution is good)
        categories_covered = len(self.kb_metrics.mcps_by_category)
        max_categories = len(MCPCategory)
        coverage_factor = min(categories_covered / max_categories, 1.0)
        health_factors.append(coverage_factor)
        
        return sum(health_factors) / len(health_factors)
    
    async def _save_registry_cache(self):
        """Save MCP registry to cache"""
        try:
            cache_file = Path(self.cache_directory) / 'mcp_registry.json'
            
            serialized_mcps = []
            for mcp_spec in self.mcp_registry.values():
                serialized_mcps.append(self._serialize_mcp_spec(mcp_spec))
            
            with open(cache_file, 'w') as f:
                json.dump(serialized_mcps, f, indent=2, default=str)
            
        except Exception as e:
            logger.warning(f"Failed to save registry cache: {str(e)}")
    
    def _serialize_mcp_spec(self, mcp_spec: EnhancedMCPSpec) -> Dict[str, Any]:
        """Serialize MCP spec for caching (excluding numpy arrays)"""
        data = asdict(mcp_spec)
        
        # Remove numpy arrays (they're cached separately)
        for key in ['text_embedding', 'capability_embedding', 'metadata_embedding', 'composite_embedding']:
            if key in data:
                data[key] = None
        
        # Convert enums to strings
        data['category'] = data['category'].value if hasattr(data['category'], 'value') else str(data['category'])
        data['quality_score'] = data['quality_score'].value if hasattr(data['quality_score'], 'value') else str(data['quality_score'])
        
        return data
    
    def _deserialize_mcp_spec(self, data: Dict[str, Any]) -> EnhancedMCPSpec:
        """Deserialize MCP spec from cache"""
        # Convert string enums back to enum objects
        if isinstance(data.get('category'), str):
            data['category'] = MCPCategory(data['category'])
        
        if isinstance(data.get('quality_score'), str):
            data['quality_score'] = MCPQualityScore(data['quality_score'])
        
        # Convert datetime strings
        for key in ['created_at', 'updated_at', 'indexed_at']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        
        return EnhancedMCPSpec(**data)

# Example usage and integration functions
async def create_enhanced_knowledge_base(openrouter_api_key: Optional[str] = None) -> MCPKnowledgeBase:
    """
    Create and initialize an enhanced MCP Knowledge Base
    
    @param openrouter_api_key OpenRouter API key
    @return Initialized MCPKnowledgeBase instance
    """
    kb = MCPKnowledgeBase(
        openrouter_api_key=openrouter_api_key,
        enable_multi_modal=True,
        enable_curation=True
    )
    
    success = await kb.initialize()
    if not success:
        raise RuntimeError("Failed to initialize MCP Knowledge Base")
    
    return kb

async def integrate_with_existing_rag_engine(rag_engine: RAGMCPEngine) -> MCPKnowledgeBase:
    """
    Create knowledge base that integrates with existing RAG-MCP Engine
    
    @param rag_engine Existing RAG-MCP Engine instance
    @return Integrated MCPKnowledgeBase instance
    """
    kb = MCPKnowledgeBase(
        openrouter_api_key=rag_engine.openrouter_api_key,
        rag_engine=rag_engine
    )
    
    success = await kb.initialize()
    if not success:
        raise RuntimeError("Failed to initialize integrated MCP Knowledge Base")
    
    return kb

# Main execution example
async def main():
    """Example usage of the MCP Knowledge Base Builder"""
    try:
        # Create and initialize knowledge base
        logger.info("Creating MCP Knowledge Base...")
        kb = await create_enhanced_knowledge_base()
        
        # Example: Add a new MCP
        new_mcp = EnhancedMCPSpec(
            name="example_ai_agent",
            description="AI-powered agent for automated task execution with learning capabilities",
            capabilities=["task_automation", "machine_learning", "adaptive_behavior", "decision_making"],
            category=MCPCategory.DEVELOPMENT,
            tags=["ai", "automation", "learning", "experimental"],
            author="example_user"
        )
        
        success = await kb.add_mcp(new_mcp, auto_curate=True)
        if success:
            logger.info(f"Successfully added MCP: {new_mcp.name}")
        
        # Example: Search for MCPs
        results = await kb.search_mcps(
            "automate data processing tasks",
            max_results=3,
            filter_category=MCPCategory.DATA_PROCESSING
        )
        
        logger.info(f"Found {len(results)} MCPs for data processing automation")
        for result in results:
            logger.info(f"- {result['name']}: {result['similarity_score']:.3f} similarity")
        
        # Example: Get analytics
        analytics = await kb.get_knowledge_base_analytics()
        logger.info(f"Knowledge Base Health Score: {analytics['knowledge_base_metrics']['health_score']:.3f}")
        
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 