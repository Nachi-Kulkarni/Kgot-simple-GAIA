#!/usr/bin/env python3
"""
RAG-MCP Coordinator - Task 17 Implementation

Advanced RAG-MCP orchestration system implementing the RAG-MCP Framework from Section 3.2
as specified in the 5-Phase Implementation Plan for Enhanced Alita.

This module provides:
- RAG-MCP Section 3.2 "RAG-MCP Framework" orchestration strategy
- Retrieval-first strategy before MCP creation following RAG-MCP methodology
- Intelligent fallback to existing MCP builder when RAG-MCP retrieval insufficient
- Usage pattern tracking based on RAG-MCP Section 4 experimental findings
- Integration with cross-validation framework for comprehensive MCP validation

Key Components:
1. RAGMCPCoordinator: Main orchestrator for RAG-MCP pipeline management
2. RetrievalFirstStrategy: Semantic search and MCP candidate selection
3. MCPValidationLayer: Sanity check validation before MCP invocation
4. IntelligentFallback: Fallback to MCP creation when retrieval insufficient
5. UsagePatternTracker: Analytics and optimization based on experimental findings
6. CrossValidationBridge: Integration with existing validation framework

Architecture follows RAG-MCP methodology:
Phase 1: Retrieval → Phase 2: Validation → Phase 3: Invocation → Phase 4: Fallback

@author: Enhanced Alita KGoT Team
@version: 1.0.0
@task: Task 17 - Create RAG-MCP Coordinator
"""

import asyncio
import json
import logging
import time
import uuid
import os
import sys
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
from pathlib import Path
import pickle

# Core scientific computing and ML libraries
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
import warnings

# LangChain for agent development (per user memory requirement)
# Using specific imports to avoid metaclass conflicts
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field

# OpenRouter integration (per user memory requirement)
import openai
from langchain_openai import ChatOpenAI

# Integration with existing systems
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from validation.mcp_cross_validator import (
    MCPCrossValidationEngine,
    MCPValidationSpec, 
    ValidationMetrics,
    CrossValidationResult,
    TaskType,
    ValidationMetricType
)
from kgot_core.error_management import (
    KGoTErrorManagementSystem, 
    ErrorType, 
    ErrorSeverity, 
    ErrorContext
)

# Winston-compatible logging setup
logger = logging.getLogger('RAGMCPCoordinator')
handler = logging.FileHandler('./logs/validation/combined.log')
formatter = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class RetrievalStrategy(Enum):
    """MCP retrieval strategy types for RAG-MCP framework"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    PARETO_OPTIMIZED = "pareto_optimized"
    HYBRID_APPROACH = "hybrid_approach"
    DIVERSITY_SAMPLING = "diversity_sampling"


class ValidationConfidence(Enum):
    """Validation confidence levels for sanity check results"""
    HIGH = "high"           # > 0.8 confidence
    MEDIUM = "medium"       # 0.5 - 0.8 confidence  
    LOW = "low"             # 0.2 - 0.5 confidence
    INSUFFICIENT = "insufficient"  # < 0.2 confidence


class FallbackTrigger(Enum):
    """Triggers for intelligent fallback to MCP creation"""
    LOW_SIMILARITY = "low_similarity"
    VALIDATION_FAILURE = "validation_failure"
    PERFORMANCE_INADEQUATE = "performance_inadequate"
    USER_REJECTION = "user_rejection"
    CAPABILITY_GAP = "capability_gap"


@dataclass
class RAGMCPRequest:
    """
    Input specification for RAG-MCP coordination request
    Based on RAG-MCP Section 3.2 framework requirements
    """
    request_id: str
    task_description: str
    task_type: Optional[TaskType] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_APPROACH
    max_candidates: int = 5
    min_confidence: float = 0.6
    timeout_seconds: int = 30
    user_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass  
class MCPCandidate:
    """
    Retrieved MCP candidate with similarity scores and metadata
    Enhanced with RAG-MCP framework validation information
    """
    mcp_id: str
    mcp_spec: MCPValidationSpec
    similarity_score: float
    pareto_score: float
    usage_frequency: float
    reliability_score: float
    cost_efficiency: float
    capability_match: Dict[str, float]
    retrieval_rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """
    Sanity check validation results for MCP candidates  
    Implements RAG-MCP Section 3.2 validation requirements
    """
    candidate_id: str
    validation_confidence: ValidationConfidence
    sanity_check_results: List[Dict[str, Any]]
    compatibility_score: float
    performance_estimate: Dict[str, float]
    error_indicators: List[str]
    validation_time: float
    recommendations: List[str]
    is_approved: bool


@dataclass
class CoordinationResult:
    """
    Final result of RAG-MCP coordination process
    Contains selected MCP and comprehensive analytics
    """
    request_id: str
    selected_mcp: Optional[MCPCandidate]
    coordination_strategy: str
    confidence_score: float
    fallback_triggered: bool
    fallback_reason: Optional[FallbackTrigger]
    validation_summary: Optional[ValidationSummary]
    retrieval_analytics: Dict[str, Any]
    execution_time: float
    resource_usage: Dict[str, Any]
    recommendations: List[str]
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UsagePattern:
    """
    Usage pattern tracking for RAG-MCP Section 4 experimental analysis
    Implements comprehensive analytics and optimization insights
    """
    pattern_id: str
    task_type: TaskType
    retrieval_strategy: RetrievalStrategy
    success_rate: float
    avg_confidence: float
    fallback_frequency: float
    preferred_mcps: List[str]
    performance_metrics: Dict[str, float]
    user_feedback: Dict[str, Any]
    optimization_opportunities: List[str]
    sample_size: int
    analysis_period: Tuple[datetime, datetime]


class RetrievalFirstStrategy:
    """
    Core retrieval-first strategy implementation
    Implements RAG-MCP Section 3.2 retrieval methodology with semantic search
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize retrieval strategy with configuration
        
        Args:
            config: Configuration dictionary containing:
                - openrouter_api_key: OpenRouter API key for embeddings
                - embedding_model: Model name for generating embeddings
                - similarity_threshold: Minimum similarity score for candidates
                - pareto_weights: Weights for Pareto optimization scoring
        """
        self.config = config
        self.openrouter_client = openai.OpenAI(
            api_key=config.get('openrouter_api_key'),
            base_url="https://openrouter.ai/api/v1"
        )
        self.embedding_model = config.get('embedding_model', 'text-embedding-ada-002')
        self.similarity_threshold = config.get('similarity_threshold', 0.3)
        self.pareto_weights = config.get('pareto_weights', {
            'usage_frequency': 0.4,
            'reliability_score': 0.35, 
            'cost_efficiency': 0.25
        })
        
        # MCP index storage and caching
        self.mcp_index: Dict[str, Dict[str, Any]] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.load_mcp_index()
        
        logger.info("RetrievalFirstStrategy initialized with OpenRouter integration")

    def load_mcp_index(self):
        """
        Load MCP index from storage or initialize with Pareto MCP toolbox
        Implements high-value Pareto MCPs covering 80% of GAIA benchmark tasks
        """
        try:
            index_path = self.config.get('mcp_index_path', './data/mcp_index.json')
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    self.mcp_index = json.load(f)
                logger.info(f"Loaded MCP index with {len(self.mcp_index)} entries")
            else:
                # Initialize with Pareto MCP toolbox from existing brainstorming engine
                self._initialize_pareto_mcp_index()
                logger.info("Initialized MCP index with Pareto MCP toolbox")
        except Exception as e:
            logger.error(f"Error loading MCP index: {e}")
            self._initialize_pareto_mcp_index()

    def _initialize_pareto_mcp_index(self):
        """Initialize MCP index with high-value Pareto MCPs from existing system"""
        # Pareto MCP toolbox based on RAG-MCP Section 4.1 stress test findings
        pareto_mcps = {
            # Web & Information Retrieval MCPs (Core 20% providing 80% coverage)
            'web_scraper_mcp': {
                'name': 'web_scraper_mcp',
                'description': 'Advanced web scraping with Beautiful Soup integration',
                'capabilities': ['html_parsing', 'data_extraction', 'content_analysis'],
                'usage_frequency': 0.25,
                'reliability_score': 0.92,
                'cost_efficiency': 0.88,
                'task_types': [TaskType.WEB_SCRAPING, TaskType.DATA_PROCESSING]
            },
            'browser_automation_mcp': {
                'name': 'browser_automation_mcp',
                'description': 'Automated browser interaction using Playwright/Puppeteer',
                'capabilities': ['ui_automation', 'form_filling', 'screenshot_capture'],
                'usage_frequency': 0.22,
                'reliability_score': 0.89,
                'cost_efficiency': 0.85,
                'task_types': [TaskType.WEB_SCRAPING, TaskType.API_INTEGRATION]
            },
            'search_engine_mcp': {
                'name': 'search_engine_mcp', 
                'description': 'Multi-provider search with Google, Bing, DuckDuckGo integration',
                'capabilities': ['web_search', 'result_ranking', 'content_filtering'],
                'usage_frequency': 0.18,
                'reliability_score': 0.94,
                'cost_efficiency': 0.91,
                'task_types': [TaskType.KNOWLEDGE_EXTRACTION, TaskType.WEB_SCRAPING]
            },
            # Data Processing MCPs
            'pandas_toolkit_mcp': {
                'name': 'pandas_toolkit_mcp',
                'description': 'Comprehensive data analysis and manipulation toolkit',
                'capabilities': ['data_analysis', 'statistical_computation', 'visualization'],
                'usage_frequency': 0.20,
                'reliability_score': 0.91,
                'cost_efficiency': 0.87,
                'task_types': [TaskType.DATA_PROCESSING, TaskType.ANALYSIS_COMPUTATION]
            },
            # Communication & Integration MCPs
            'api_client_mcp': {
                'name': 'api_client_mcp',
                'description': 'REST/GraphQL API interaction with authentication support',
                'capabilities': ['api_integration', 'authentication', 'data_sync'],
                'usage_frequency': 0.19,
                'reliability_score': 0.92,
                'cost_efficiency': 0.88,
                'task_types': [TaskType.API_INTEGRATION, TaskType.COMMUNICATION]
            },
            # Development & System MCPs
            'code_execution_mcp': {
                'name': 'code_execution_mcp',
                'description': 'Secure code execution with containerization support',
                'capabilities': ['code_execution', 'debugging', 'security_sandboxing'],
                'usage_frequency': 0.21,
                'reliability_score': 0.90,
                'cost_efficiency': 0.86,
                'task_types': [TaskType.CODE_GENERATION, TaskType.DATA_PROCESSING]
            }
        }
        
        self.mcp_index = pareto_mcps
        self._save_mcp_index()

    def _save_mcp_index(self):
        """Save MCP index to persistent storage"""
        try:
            index_path = self.config.get('mcp_index_path', './data/mcp_index.json')
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            with open(index_path, 'w') as f:
                json.dump(self.mcp_index, f, indent=2, default=str)
            logger.info(f"Saved MCP index to {index_path}")
        except Exception as e:
            logger.error(f"Error saving MCP index: {e}")

    async def retrieve_mcp_candidates(self, request: RAGMCPRequest) -> List[MCPCandidate]:
        """
        Retrieve MCP candidates using retrieval-first strategy
        Implements RAG-MCP Section 3.2 retrieval methodology
        
        Args:
            request: RAG-MCP coordination request
            
        Returns:
            List of ranked MCP candidates with similarity scores
        """
        logger.info(f"Starting MCP retrieval for request: {request.request_id}")
        start_time = time.time()
        
        try:
            # Phase 1: Generate task embedding
            task_embedding = await self.generate_task_embedding(request.task_description)
            
            # Phase 2: Calculate similarity scores for all MCPs
            similarity_scores = await self.calculate_similarity_scores(
                task_embedding, request.task_type
            )
            
            # Phase 3: Apply strategy-specific selection
            candidates = await self.apply_retrieval_strategy(
                similarity_scores, request
            )
            
            # Phase 4: Rank and filter candidates
            final_candidates = self.rank_and_filter_candidates(
                candidates, request
            )
            
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieved {len(final_candidates)} candidates in {retrieval_time:.2f}s")
            
            return final_candidates
            
        except Exception as e:
            logger.error(f"Error in MCP retrieval: {e}")
            return []

    async def generate_task_embedding(self, task_description: str) -> np.ndarray:
        """
        Generate embedding for task description using OpenRouter
        Implements caching for performance optimization
        
        Args:
            task_description: User task description
            
        Returns:
            Task embedding vector
        """
        # Check cache first
        cache_key = hash(task_description)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Generate embedding using OpenRouter
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openrouter_client.embeddings.create(
                    input=task_description,
                    model=self.embedding_model
                )
            )
            
            embedding = np.array(response.data[0].embedding)
            
            # Cache the embedding
            self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating task embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(1536)  # Standard embedding dimension

    async def calculate_similarity_scores(self, task_embedding: np.ndarray, 
                                        task_type: Optional[TaskType] = None) -> Dict[str, float]:
        """
        Calculate similarity scores between task and all MCPs
        Uses cosine similarity with task type filtering
        
        Args:
            task_embedding: Task embedding vector
            task_type: Optional task type for filtering
            
        Returns:
            Dictionary mapping MCP IDs to similarity scores
        """
        similarity_scores = {}
        
        for mcp_id, mcp_data in self.mcp_index.items():
            try:
                # Filter by task type if specified
                if task_type and task_type not in mcp_data.get('task_types', []):
                    continue
                
                # Generate MCP embedding (cached)
                mcp_embedding = await self.get_mcp_embedding(mcp_data)
                
                # Calculate cosine similarity
                similarity = 1 - cosine(task_embedding, mcp_embedding)
                similarity_scores[mcp_id] = max(0, similarity)  # Ensure non-negative
                
            except Exception as e:
                logger.warning(f"Error calculating similarity for MCP {mcp_id}: {e}")
                similarity_scores[mcp_id] = 0.0
        
        return similarity_scores

    async def get_mcp_embedding(self, mcp_data: Dict[str, Any]) -> np.ndarray:
        """
        Get or generate embedding for MCP description
        Implements caching for performance
        """
        # Create embedding text from MCP data
        embedding_text = f"{mcp_data['description']} {' '.join(mcp_data['capabilities'])}"
        cache_key = hash(embedding_text)
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openrouter_client.embeddings.create(
                    input=embedding_text,
                    model=self.embedding_model
                )
            )
            
            embedding = np.array(response.data[0].embedding)
            self.embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating MCP embedding: {e}")
            return np.random.normal(0, 0.1, 1536)  # Random fallback

    async def apply_retrieval_strategy(self, similarity_scores: Dict[str, float], 
                                     request: RAGMCPRequest) -> List[MCPCandidate]:
        """
        Apply specified retrieval strategy to select candidates
        Implements multiple strategies: semantic, Pareto, hybrid, diversity
        """
        if request.retrieval_strategy == RetrievalStrategy.SEMANTIC_SIMILARITY:
            return await self._apply_semantic_strategy(similarity_scores, request)
        elif request.retrieval_strategy == RetrievalStrategy.PARETO_OPTIMIZED:
            return await self._apply_pareto_strategy(similarity_scores, request)
        elif request.retrieval_strategy == RetrievalStrategy.HYBRID_APPROACH:
            return await self._apply_hybrid_strategy(similarity_scores, request)
        elif request.retrieval_strategy == RetrievalStrategy.DIVERSITY_SAMPLING:
            return await self._apply_diversity_strategy(similarity_scores, request)
        else:
            # Default to semantic similarity
            return await self._apply_semantic_strategy(similarity_scores, request)

    async def _apply_semantic_strategy(self, similarity_scores: Dict[str, float], 
                                     request: RAGMCPRequest) -> List[MCPCandidate]:
        """Apply pure semantic similarity-based selection"""
        candidates = []
        
        # Sort by similarity score
        sorted_mcps = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (mcp_id, similarity) in enumerate(sorted_mcps[:request.max_candidates]):
            if similarity >= self.similarity_threshold:
                mcp_data = self.mcp_index[mcp_id]
                
                candidate = MCPCandidate(
                    mcp_id=mcp_id,
                    mcp_spec=self._create_mcp_spec(mcp_id, mcp_data),
                    similarity_score=similarity,
                    pareto_score=0.0,  # Not used in semantic strategy
                    usage_frequency=mcp_data.get('usage_frequency', 0.0),
                    reliability_score=mcp_data.get('reliability_score', 0.0),
                    cost_efficiency=mcp_data.get('cost_efficiency', 0.0),
                    capability_match=self._calculate_capability_match(mcp_data, request),
                    retrieval_rank=i + 1
                )
                candidates.append(candidate)
        
        return candidates

    async def _apply_pareto_strategy(self, similarity_scores: Dict[str, float], 
                                   request: RAGMCPRequest) -> List[MCPCandidate]:
        """Apply Pareto-optimized selection based on RAG-MCP stress test findings"""
        candidates = []
        
        # Calculate Pareto scores for all candidates
        pareto_scores = {}
        for mcp_id, similarity in similarity_scores.items():
            if similarity >= self.similarity_threshold:
                mcp_data = self.mcp_index[mcp_id]
                pareto_score = self._calculate_pareto_score(mcp_data, similarity)
                pareto_scores[mcp_id] = pareto_score
        
        # Sort by Pareto score
        sorted_mcps = sorted(pareto_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (mcp_id, pareto_score) in enumerate(sorted_mcps[:request.max_candidates]):
            mcp_data = self.mcp_index[mcp_id]
            
            candidate = MCPCandidate(
                mcp_id=mcp_id,
                mcp_spec=self._create_mcp_spec(mcp_id, mcp_data),
                similarity_score=similarity_scores[mcp_id],
                pareto_score=pareto_score,
                usage_frequency=mcp_data.get('usage_frequency', 0.0),
                reliability_score=mcp_data.get('reliability_score', 0.0),
                cost_efficiency=mcp_data.get('cost_efficiency', 0.0),
                capability_match=self._calculate_capability_match(mcp_data, request),
                retrieval_rank=i + 1
            )
            candidates.append(candidate)
        
        return candidates

    def _calculate_pareto_score(self, mcp_data: Dict[str, Any], similarity: float) -> float:
        """
        Calculate Pareto optimization score
        Combines similarity with usage frequency, reliability, and cost efficiency
        """
        weights = self.pareto_weights
        
        score = (
            weights['usage_frequency'] * mcp_data.get('usage_frequency', 0.0) +
            weights['reliability_score'] * mcp_data.get('reliability_score', 0.0) +
            weights['cost_efficiency'] * mcp_data.get('cost_efficiency', 0.0)
        ) * similarity  # Weight by similarity
        
        return score

    async def _apply_hybrid_strategy(self, similarity_scores: Dict[str, float], 
                                   request: RAGMCPRequest) -> List[MCPCandidate]:
        """Apply hybrid approach combining semantic and Pareto strategies"""
        # Get candidates from both strategies
        semantic_candidates = await self._apply_semantic_strategy(similarity_scores, request)
        pareto_candidates = await self._apply_pareto_strategy(similarity_scores, request)
        
        # Combine and deduplicate
        combined_candidates = {}
        
        # Add semantic candidates with weight
        for candidate in semantic_candidates:
            combined_candidates[candidate.mcp_id] = candidate
        
        # Merge with Pareto candidates, averaging scores
        for candidate in pareto_candidates:
            if candidate.mcp_id in combined_candidates:
                # Average the rankings
                existing = combined_candidates[candidate.mcp_id]
                existing.pareto_score = candidate.pareto_score
                existing.retrieval_rank = (existing.retrieval_rank + candidate.retrieval_rank) / 2
            else:
                combined_candidates[candidate.mcp_id] = candidate
        
        # Sort by combined score (similarity + pareto)
        hybrid_candidates = list(combined_candidates.values())
        hybrid_candidates.sort(
            key=lambda x: 0.6 * x.similarity_score + 0.4 * x.pareto_score, 
            reverse=True
        )
        
        return hybrid_candidates[:request.max_candidates]

    async def _apply_diversity_strategy(self, similarity_scores: Dict[str, float], 
                                      request: RAGMCPRequest) -> List[MCPCandidate]:
        """Apply diversity sampling to ensure capability coverage"""
        candidates = []
        selected_capabilities = set()
        
        # Sort by similarity
        sorted_mcps = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        
        for mcp_id, similarity in sorted_mcps:
            if len(candidates) >= request.max_candidates:
                break
            
            if similarity < self.similarity_threshold:
                continue
            
            mcp_data = self.mcp_index[mcp_id]
            mcp_capabilities = set(mcp_data.get('capabilities', []))
            
            # Check if this MCP adds new capabilities
            if not selected_capabilities or not mcp_capabilities.issubset(selected_capabilities):
                candidate = MCPCandidate(
                    mcp_id=mcp_id,
                    mcp_spec=self._create_mcp_spec(mcp_id, mcp_data),
                    similarity_score=similarity,
                    pareto_score=self._calculate_pareto_score(mcp_data, similarity),
                    usage_frequency=mcp_data.get('usage_frequency', 0.0),
                    reliability_score=mcp_data.get('reliability_score', 0.0),
                    cost_efficiency=mcp_data.get('cost_efficiency', 0.0),
                    capability_match=self._calculate_capability_match(mcp_data, request),
                    retrieval_rank=len(candidates) + 1
                )
                candidates.append(candidate)
                selected_capabilities.update(mcp_capabilities)
        
        return candidates

    def _create_mcp_spec(self, mcp_id: str, mcp_data: Dict[str, Any]) -> MCPValidationSpec:
        """Create MCPValidationSpec from MCP index data"""
        return MCPValidationSpec(
            mcp_id=mcp_id,
            name=mcp_data['name'],
            description=mcp_data['description'],
            task_type=mcp_data.get('task_types', [TaskType.DATA_PROCESSING])[0],
            capabilities=mcp_data.get('capabilities', []),
            implementation_code="",  # Would be loaded from actual implementation
            test_cases=[],  # Would be loaded from test suite
            expected_outputs=[],
            performance_requirements={},
            reliability_requirements={}
        )

    def _calculate_capability_match(self, mcp_data: Dict[str, Any], 
                                  request: RAGMCPRequest) -> Dict[str, float]:
        """Calculate how well MCP capabilities match request requirements"""
        # Extract capabilities from request description (simplified)
        request_text = request.task_description.lower()
        mcp_capabilities = mcp_data.get('capabilities', [])
        
        capability_scores = {}
        for capability in mcp_capabilities:
            # Simple keyword matching (could be enhanced with NLP)
            if capability.lower() in request_text:
                capability_scores[capability] = 1.0
            else:
                # Fuzzy matching based on semantic similarity
                capability_scores[capability] = 0.5  # Placeholder
        
        return capability_scores

    def rank_and_filter_candidates(self, candidates: List[MCPCandidate], 
                                 request: RAGMCPRequest) -> List[MCPCandidate]:
        """Final ranking and filtering of candidates"""
        # Filter by minimum confidence
        filtered_candidates = [
            c for c in candidates 
            if c.similarity_score >= request.min_confidence
        ]
        
        # Sort by final ranking score
        filtered_candidates.sort(
            key=lambda x: (x.similarity_score * 0.4 + 
                          x.pareto_score * 0.3 + 
                          x.reliability_score * 0.3),
            reverse=True
        )
        
        return filtered_candidates[:request.max_candidates]


class MCPValidationLayer:
    """
    Sanity check validation layer for MCP candidates
    Implements RAG-MCP Section 3.2 validation requirements before invocation
    """
    
    def __init__(self, config: Dict[str, Any], llm_client: Optional[Any] = None):
        """
        Initialize validation layer with LLM client for testing
        
        Args:
            config: Validation configuration parameters
            llm_client: LangChain LLM client for validation queries
        """
        self.config = config
        self.llm_client = llm_client or ChatOpenAI(
            openai_api_key=config.get('openrouter_api_key'),
            openai_api_base="https://openrouter.ai/api/v1",
            model_name=config.get('validation_model', 'anthropic/claude-sonnet-4'),
            temperature=0.1
        )
        
        self.validation_timeout = config.get('validation_timeout', 10)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        logger.info("MCPValidationLayer initialized with OpenRouter LLM client")

    async def validate_mcp_candidates(self, candidates: List[MCPCandidate], 
                                    request: RAGMCPRequest) -> List[ValidationSummary]:
        """
        Perform sanity check validation on MCP candidates
        Generates validation queries and tests compatibility
        
        Args:
            candidates: List of MCP candidates to validate
            request: Original RAG-MCP request for context
            
        Returns:
            List of validation summaries for each candidate
        """
        logger.info(f"Starting validation for {len(candidates)} MCP candidates")
        validation_results = []
        
        for candidate in candidates:
            try:
                validation_summary = await self._validate_single_candidate(
                    candidate, request
                )
                validation_results.append(validation_summary)
                
            except Exception as e:
                logger.error(f"Error validating candidate {candidate.mcp_id}: {e}")
                # Create failed validation summary
                failed_summary = ValidationSummary(
                    candidate_id=candidate.mcp_id,
                    validation_confidence=ValidationConfidence.INSUFFICIENT,
                    sanity_check_results=[],
                    compatibility_score=0.0,
                    performance_estimate={},
                    error_indicators=[str(e)],
                    validation_time=0.0,
                    recommendations=["Validation failed - exclude from selection"],
                    is_approved=False
                )
                validation_results.append(failed_summary)
        
        return validation_results

    async def _validate_single_candidate(self, candidate: MCPCandidate, 
                                       request: RAGMCPRequest) -> ValidationSummary:
        """
        Validate a single MCP candidate with comprehensive testing
        Implements sanity check methodology from RAG-MCP framework
        """
        start_time = time.time()
        logger.info(f"Validating candidate: {candidate.mcp_id}")
        
        # Generate validation queries
        validation_queries = await self._generate_validation_queries(candidate, request)
        
        # Execute sanity checks
        sanity_check_results = []
        for query in validation_queries:
            result = await self._execute_sanity_check(candidate, query)
            sanity_check_results.append(result)
        
        # Calculate compatibility score
        compatibility_score = self._calculate_compatibility_score(sanity_check_results)
        
        # Estimate performance metrics
        performance_estimate = self._estimate_performance_metrics(
            candidate, sanity_check_results
        )
        
        # Identify error indicators
        error_indicators = self._identify_error_indicators(sanity_check_results)
        
        # Determine validation confidence
        validation_confidence = self._determine_validation_confidence(
            compatibility_score, error_indicators
        )
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations(
            candidate, validation_confidence, error_indicators
        )
        
        validation_time = time.time() - start_time
        is_approved = (validation_confidence in [ValidationConfidence.HIGH, ValidationConfidence.MEDIUM] 
                      and compatibility_score >= self.confidence_threshold)
        
        return ValidationSummary(
            candidate_id=candidate.mcp_id,
            validation_confidence=validation_confidence,
            sanity_check_results=sanity_check_results,
            compatibility_score=compatibility_score,
            performance_estimate=performance_estimate,
            error_indicators=error_indicators,
            validation_time=validation_time,
            recommendations=recommendations,
            is_approved=is_approved
        )

    async def _generate_validation_queries(self, candidate: MCPCandidate, 
                                         request: RAGMCPRequest) -> List[Dict[str, Any]]:
        """
        Generate few-shot validation queries for MCP testing
        Creates simple test cases to verify basic functionality
        """
        queries = []
        
        # Generate basic capability test
        capability_query = {
            'type': 'capability_test',
            'description': f"Test basic {candidate.mcp_spec.name} functionality",
            'input': self._create_simple_test_input(candidate, request),
            'expected_behavior': 'successful_execution'
        }
        queries.append(capability_query)
        
        # Generate edge case test
        edge_case_query = {
            'type': 'edge_case_test',
            'description': f"Test {candidate.mcp_spec.name} with edge case input",
            'input': self._create_edge_case_input(candidate),
            'expected_behavior': 'graceful_handling'
        }
        queries.append(edge_case_query)
        
        # Generate error handling test
        error_handling_query = {
            'type': 'error_handling_test',
            'description': f"Test {candidate.mcp_spec.name} error handling",
            'input': self._create_invalid_input(candidate),
            'expected_behavior': 'error_recovery'
        }
        queries.append(error_handling_query)
        
        return queries

    def _create_simple_test_input(self, candidate: MCPCandidate, request: RAGMCPRequest) -> Dict[str, Any]:
        """Create simple test input based on MCP capabilities and task description"""
        # Simplified test input generation based on MCP type
        if 'web_scraping' in candidate.mcp_id:
            return {'url': 'https://httpbin.org/html', 'selector': 'h1'}
        elif 'api_client' in candidate.mcp_id:
            return {'endpoint': 'https://httpbin.org/get', 'method': 'GET'}
        elif 'data_processing' in candidate.mcp_id:
            return {'data': [1, 2, 3, 4, 5], 'operation': 'sum'}
        else:
            return {'test_data': 'sample input for validation'}

    def _create_edge_case_input(self, candidate: MCPCandidate) -> Dict[str, Any]:
        """Create edge case test input"""
        return {'edge_case': True, 'empty_input': '', 'large_input': 'x' * 1000}

    def _create_invalid_input(self, candidate: MCPCandidate) -> Dict[str, Any]:
        """Create invalid input for error handling test"""
        return {'invalid': True, 'malformed_data': None}

    async def _execute_sanity_check(self, candidate: MCPCandidate, 
                                  query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sanity check query using LLM simulation
        In production, this would execute against actual MCP implementation
        """
        try:
            # Create validation prompt
            validation_prompt = self._create_validation_prompt(candidate, query)
            
            # Execute with LLM (simulating MCP execution)
            response = await asyncio.wait_for(
                self._query_llm_for_validation(validation_prompt),
                timeout=self.validation_timeout
            )
            
            # Parse and evaluate response
            result = {
                'query_type': query['type'],
                'input': query['input'],
                'response': response,
                'success': self._evaluate_response_success(response, query),
                'execution_time': time.time(),
                'error_indicators': self._extract_error_indicators(response)
            }
            
            return result
            
        except asyncio.TimeoutError:
            return {
                'query_type': query['type'],
                'input': query['input'],
                'response': None,
                'success': False,
                'execution_time': self.validation_timeout,
                'error_indicators': ['timeout']
            }
        except Exception as e:
            return {
                'query_type': query['type'],
                'input': query['input'],
                'response': None,
                'success': False,
                'execution_time': 0.0,
                'error_indicators': [str(e)]
            }

    def _create_validation_prompt(self, candidate: MCPCandidate, query: Dict[str, Any]) -> str:
        """Create prompt for LLM-based validation simulation"""
        return f"""
You are simulating the execution of an MCP (Model Context Protocol) tool for validation purposes.

MCP Details:
- Name: {candidate.mcp_spec.name}
- Description: {candidate.mcp_spec.description}
- Capabilities: {', '.join(candidate.mcp_spec.capabilities)}

Validation Query:
- Type: {query['type']}
- Description: {query['description']}
- Input: {json.dumps(query['input'])}
- Expected Behavior: {query['expected_behavior']}

Please simulate how this MCP would respond to the given input. Consider:
1. Would the MCP successfully process this input?
2. What would be the expected output format?
3. Are there any potential errors or issues?
4. How would the MCP handle edge cases or invalid inputs?

Respond with a JSON object containing:
- success: boolean indicating if execution would succeed
- output: the expected output from the MCP
- errors: any errors or warnings that might occur
- confidence: your confidence level (0.0-1.0) in this simulation
"""

    async def _query_llm_for_validation(self, prompt: str) -> str:
        """Query LLM for validation simulation"""
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm_client(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error querying LLM for validation: {e}")
            return json.dumps({
                'success': False,
                'output': None,
                'errors': [str(e)],
                'confidence': 0.0
            })

    def _evaluate_response_success(self, response: str, query: Dict[str, Any]) -> bool:
        """Evaluate if the validation response indicates success"""
        try:
            # Parse LLM response
            parsed_response = json.loads(response)
            return parsed_response.get('success', False)
        except:
            # Fallback to text analysis
            return 'success' in response.lower() and 'error' not in response.lower()

    def _extract_error_indicators(self, response: str) -> List[str]:
        """Extract error indicators from validation response"""
        error_indicators = []
        try:
            parsed_response = json.loads(response)
            errors = parsed_response.get('errors', [])
            error_indicators.extend(errors)
        except:
            # Fallback to text analysis
            if 'error' in response.lower():
                error_indicators.append('execution_error')
            if 'timeout' in response.lower():
                error_indicators.append('timeout')
            if 'failed' in response.lower():
                error_indicators.append('failure')
        
        return error_indicators

    def _calculate_compatibility_score(self, sanity_check_results: List[Dict[str, Any]]) -> float:
        """Calculate overall compatibility score from sanity check results"""
        if not sanity_check_results:
            return 0.0
        
        success_count = sum(1 for result in sanity_check_results if result.get('success', False))
        compatibility_score = success_count / len(sanity_check_results)
        
        return compatibility_score

    def _estimate_performance_metrics(self, candidate: MCPCandidate, 
                                    sanity_check_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate performance metrics from validation results"""
        execution_times = [
            result.get('execution_time', 0.0) 
            for result in sanity_check_results
        ]
        
        return {
            'avg_execution_time': np.mean(execution_times) if execution_times else 0.0,
            'max_execution_time': np.max(execution_times) if execution_times else 0.0,
            'reliability_estimate': candidate.reliability_score,
            'efficiency_estimate': candidate.cost_efficiency
        }

    def _identify_error_indicators(self, sanity_check_results: List[Dict[str, Any]]) -> List[str]:
        """Identify common error patterns from validation results"""
        all_errors = []
        for result in sanity_check_results:
            all_errors.extend(result.get('error_indicators', []))
        
        # Count and filter significant errors
        error_counts = Counter(all_errors)
        significant_errors = [
            error for error, count in error_counts.items() 
            if count >= len(sanity_check_results) * 0.3  # 30% threshold
        ]
        
        return significant_errors

    def _determine_validation_confidence(self, compatibility_score: float, 
                                       error_indicators: List[str]) -> ValidationConfidence:
        """Determine validation confidence level based on results"""
        if compatibility_score >= 0.8 and len(error_indicators) == 0:
            return ValidationConfidence.HIGH
        elif compatibility_score >= 0.6 and len(error_indicators) <= 1:
            return ValidationConfidence.MEDIUM
        elif compatibility_score >= 0.3:
            return ValidationConfidence.LOW
        else:
            return ValidationConfidence.INSUFFICIENT

    def _generate_validation_recommendations(self, candidate: MCPCandidate,
                                           confidence: ValidationConfidence,
                                           error_indicators: List[str]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if confidence == ValidationConfidence.HIGH:
            recommendations.append("Candidate approved for execution")
        elif confidence == ValidationConfidence.MEDIUM:
            recommendations.append("Candidate approved with monitoring")
            if error_indicators:
                recommendations.append(f"Monitor for: {', '.join(error_indicators)}")
        elif confidence == ValidationConfidence.LOW:
            recommendations.append("Candidate requires additional validation")
            recommendations.append("Consider fallback options")
        else:
            recommendations.append("Candidate rejected - insufficient validation")
            recommendations.append("Trigger fallback to MCP creation")
        
        return recommendations


class IntelligentFallback:
    """
    Intelligent fallback system for MCP creation when retrieval is insufficient
    Integrates with existing MCP brainstorming engine for gap-based creation
    """
    
    def __init__(self, config: Dict[str, Any], brainstorming_engine_path: str = None):
        """
        Initialize fallback system with connection to MCP brainstorming engine
        
        Args:
            config: Fallback configuration parameters
            brainstorming_engine_path: Path to MCP brainstorming engine
        """
        self.config = config
        self.brainstorming_engine_path = (brainstorming_engine_path or 
                                        '../alita_core/mcp_brainstorming.js')
        
        # Fallback trigger thresholds
        self.similarity_threshold = config.get('fallback_similarity_threshold', 0.3)
        self.validation_threshold = config.get('fallback_validation_threshold', 0.5)
        self.performance_threshold = config.get('fallback_performance_threshold', 0.4)
        
        # Fallback tracking
        self.fallback_history: deque = deque(maxlen=100)
        
        logger.info("IntelligentFallback initialized with MCP brainstorming integration")

    async def detect_retrieval_insufficiency(self, retrieval_results: List[MCPCandidate],
                                           validation_results: List[ValidationSummary],
                                           request: RAGMCPRequest) -> Optional[FallbackTrigger]:
        """
        Detect if retrieval results are insufficient and determine fallback trigger
        Implements multi-criteria decision making for fallback activation
        
        Args:
            retrieval_results: Retrieved MCP candidates
            validation_results: Validation results for candidates
            request: Original RAG-MCP request
            
        Returns:
            FallbackTrigger if fallback should be triggered, None otherwise
        """
        logger.info("Analyzing retrieval results for fallback triggers")
        
        # Check for low similarity scores
        if not retrieval_results or all(c.similarity_score < self.similarity_threshold 
                                      for c in retrieval_results):
            return FallbackTrigger.LOW_SIMILARITY
        
        # Check for validation failures
        approved_validations = [v for v in validation_results if v.is_approved]
        if not approved_validations:
            return FallbackTrigger.VALIDATION_FAILURE
        
        # Check for performance inadequacy
        performance_adequate = any(
            v.performance_estimate.get('efficiency_estimate', 0) >= self.performance_threshold
            for v in validation_results
        )
        if not performance_adequate:
            return FallbackTrigger.PERFORMANCE_INADEQUATE
        
        # Check for capability gaps
        capability_gap = await self._detect_capability_gap(
            retrieval_results, request
        )
        if capability_gap:
            return FallbackTrigger.CAPABILITY_GAP
        
        return None  # No fallback needed

    async def _detect_capability_gap(self, candidates: List[MCPCandidate], 
                                   request: RAGMCPRequest) -> bool:
        """Detect if there's a significant capability gap in retrieved candidates"""
        # Extract required capabilities from request (simplified)
        required_capabilities = self._extract_required_capabilities(request.task_description)
        
        # Get all available capabilities from candidates
        available_capabilities = set()
        for candidate in candidates:
            available_capabilities.update(candidate.mcp_spec.capabilities)
        
        # Calculate coverage
        coverage = len(available_capabilities.intersection(required_capabilities))
        coverage_ratio = coverage / len(required_capabilities) if required_capabilities else 1.0
        
        return coverage_ratio < 0.7  # 70% coverage threshold

    def _extract_required_capabilities(self, task_description: str) -> Set[str]:
        """Extract required capabilities from task description (simplified)"""
        # Simple keyword mapping - could be enhanced with NLP
        capability_keywords = {
            'scrape': 'web_scraping',
            'api': 'api_integration', 
            'data': 'data_processing',
            'analyze': 'data_analysis',
            'code': 'code_execution',
            'search': 'web_search',
            'email': 'email_automation'
        }
        
        required_capabilities = set()
        task_lower = task_description.lower()
        
        for keyword, capability in capability_keywords.items():
            if keyword in task_lower:
                required_capabilities.add(capability)
        
        return required_capabilities

    async def fallback_to_mcp_creation(self, request: RAGMCPRequest, 
                                     trigger: FallbackTrigger,
                                     gap_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fallback to MCP creation using existing brainstorming engine
        Integrates with Node.js brainstorming engine via subprocess
        
        Args:
            request: Original RAG-MCP request
            trigger: Reason for fallback activation
            gap_analysis: Optional capability gap analysis
            
        Returns:
            MCP creation results from brainstorming engine
        """
        logger.info(f"Initiating fallback to MCP creation - trigger: {trigger.value}")
        
        try:
            # Record fallback event
            fallback_event = {
                'request_id': request.request_id,
                'trigger': trigger.value,
                'timestamp': datetime.now(),
                'task_description': request.task_description,
                'gap_analysis': gap_analysis
            }
            self.fallback_history.append(fallback_event)
            
            # Prepare input for brainstorming engine
            brainstorming_input = {
                'taskDescription': request.task_description,
                'taskType': request.task_type.value if request.task_type else None,
                'constraints': request.constraints,
                'preferences': request.preferences,
                'fallbackTrigger': trigger.value,
                'gapAnalysis': gap_analysis,
                'requestId': request.request_id
            }
            
            # Execute brainstorming engine
            result = await self._execute_brainstorming_engine(brainstorming_input)
            
            logger.info(f"Fallback MCP creation completed for request: {request.request_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in fallback MCP creation: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_trigger': trigger.value,
                'request_id': request.request_id
            }

    async def _execute_brainstorming_engine(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP brainstorming engine via subprocess"""
        try:
            # Write input to temporary file
            input_file = f"/tmp/mcp_brainstorming_input_{uuid.uuid4().hex}.json"
            with open(input_file, 'w') as f:
                json.dump(input_data, f, indent=2, default=str)
            
            # Execute Node.js brainstorming engine
            cmd = [
                'node', 
                self.brainstorming_engine_path,
                '--input', input_file,
                '--mode', 'fallback'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Clean up input file
            os.unlink(input_file)
            
            if process.returncode == 0:
                result = json.loads(stdout.decode())
                return result
            else:
                logger.error(f"Brainstorming engine error: {stderr.decode()}")
                return {
                    'success': False,
                    'error': stderr.decode(),
                    'stdout': stdout.decode()
                }
                
        except Exception as e:
            logger.error(f"Error executing brainstorming engine: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def integrate_new_mcp_to_index(self, new_mcp_spec: Dict[str, Any],
                                       retrieval_strategy: RetrievalFirstStrategy) -> bool:
        """
        Integrate newly created MCP back into retrieval index
        Updates MCP index with new capabilities for future retrieval
        """
        try:
            # Extract MCP data from brainstorming result
            mcp_data = {
                'name': new_mcp_spec.get('name', ''),
                'description': new_mcp_spec.get('description', ''),
                'capabilities': new_mcp_spec.get('capabilities', []),
                'usage_frequency': 0.01,  # Start with low frequency
                'reliability_score': 0.7,  # Default reliability
                'cost_efficiency': 0.6,   # Default efficiency
                'task_types': new_mcp_spec.get('task_types', [])
            }
            
            # Add to retrieval index
            mcp_id = new_mcp_spec.get('id', f"generated_{uuid.uuid4().hex[:8]}")
            retrieval_strategy.mcp_index[mcp_id] = mcp_data
            
            # Save updated index
            retrieval_strategy._save_mcp_index()
            
            logger.info(f"Integrated new MCP {mcp_id} into retrieval index")
            return True
            
        except Exception as e:
            logger.error(f"Error integrating new MCP to index: {e}")
            return False

    def get_fallback_analytics(self) -> Dict[str, Any]:
        """Get analytics on fallback patterns and triggers"""
        if not self.fallback_history:
            return {
                'total_fallbacks': 0,
                'trigger_distribution': {},
                'success_rate': 0.0,
                'common_gaps': []
            }
        
        # Analyze trigger distribution
        triggers = [event['trigger'] for event in self.fallback_history]
        trigger_distribution = dict(Counter(triggers))
        
        # Calculate recent success rate (placeholder - would need actual tracking)
        success_rate = 0.75  # Placeholder
        
        # Identify common capability gaps
        common_gaps = []  # Would be extracted from gap_analysis data
        
        return {
            'total_fallbacks': len(self.fallback_history),
            'trigger_distribution': trigger_distribution,
            'success_rate': success_rate,
            'common_gaps': common_gaps,
            'recent_fallbacks': list(self.fallback_history)[-10:]  # Last 10 events
        }


class UsagePatternTracker:
    """
    Usage pattern tracking and analysis system
    Implements RAG-MCP Section 4 experimental findings analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize usage pattern tracker with analytics configuration
        
        Args:
            config: Tracking configuration parameters
        """
        self.config = config
        self.usage_data_path = config.get('usage_data_path', './data/usage_patterns.json')
        self.analytics_window = config.get('analytics_window_days', 30)
        
        # In-memory tracking
        self.usage_events: deque = deque(maxlen=1000)
        self.pattern_cache = {}
        
        # Load existing usage data
        self.load_usage_data()
        
        logger.info("UsagePatternTracker initialized for RAG-MCP analytics")

    def load_usage_data(self):
        """Load existing usage data from persistent storage"""
        try:
            if os.path.exists(self.usage_data_path):
                with open(self.usage_data_path, 'r') as f:
                    data = json.load(f)
                    # Load recent events into memory
                    for event in data.get('recent_events', [])[-1000:]:
                        event['timestamp'] = datetime.fromisoformat(event['timestamp'])
                        self.usage_events.append(event)
                logger.info(f"Loaded {len(self.usage_events)} usage events")
        except Exception as e:
            logger.error(f"Error loading usage data: {e}")

    def save_usage_data(self):
        """Save usage data to persistent storage"""
        try:
            os.makedirs(os.path.dirname(self.usage_data_path), exist_ok=True)
            data = {
                'recent_events': [
                    {**event, 'timestamp': event['timestamp'].isoformat()}
                    for event in self.usage_events
                ],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.usage_data_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving usage data: {e}")

    async def track_retrieval_event(self, request: RAGMCPRequest, 
                                  result: CoordinationResult) -> None:
        """
        Track a retrieval event for pattern analysis
        Records comprehensive event data for experimental analysis
        """
        event = {
            'event_id': str(uuid.uuid4()),
            'request_id': request.request_id,
            'timestamp': datetime.now(),
            'task_type': request.task_type.value if request.task_type else None,
            'retrieval_strategy': request.retrieval_strategy.value,
            'success': result.success,
            'selected_mcp': result.selected_mcp.mcp_id if result.selected_mcp else None,
            'confidence_score': result.confidence_score,
            'fallback_triggered': result.fallback_triggered,
            'fallback_reason': result.fallback_reason.value if result.fallback_reason else None,
            'execution_time': result.execution_time,
            'resource_usage': result.resource_usage,
            'user_context': request.user_context
        }
        
        self.usage_events.append(event)
        
        # Periodically save to persistent storage
        if len(self.usage_events) % 10 == 0:
            self.save_usage_data()

    async def analyze_usage_patterns(self, time_window_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze usage patterns over specified time window
        Implements comprehensive pattern analysis based on RAG-MCP experimental methodology
        """
        window_days = time_window_days or self.analytics_window
        cutoff_date = datetime.now() - timedelta(days=window_days)
        
        # Filter events within time window
        recent_events = [
            event for event in self.usage_events
            if event['timestamp'] >= cutoff_date
        ]
        
        if not recent_events:
            return {
                'analysis_period': {
                    'start_date': cutoff_date.isoformat(),
                    'end_date': datetime.now().isoformat(),
                    'event_count': 0
                },
                'patterns': {},
                'recommendations': ["Insufficient data for pattern analysis"]
            }
        
        # Perform comprehensive pattern analysis
        patterns = {
            'task_type_distribution': self._analyze_task_type_patterns(recent_events),
            'strategy_effectiveness': self._analyze_strategy_effectiveness(recent_events),
            'success_rate_trends': self._analyze_success_rate_trends(recent_events),
            'fallback_patterns': self._analyze_fallback_patterns(recent_events),
            'performance_metrics': self._analyze_performance_metrics(recent_events),
            'mcp_selection_patterns': self._analyze_mcp_selection_patterns(recent_events),
            'user_behavior_patterns': self._analyze_user_behavior_patterns(recent_events)
        }
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(patterns)
        
        return {
            'analysis_period': {
                'start_date': cutoff_date.isoformat(),
                'end_date': datetime.now().isoformat(),
                'event_count': len(recent_events)
            },
            'patterns': patterns,
            'recommendations': recommendations
        }

    def _analyze_task_type_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns by task type"""
        task_types = [event.get('task_type') for event in events if event.get('task_type')]
        task_type_counts = dict(Counter(task_types))
        
        # Calculate success rates by task type
        task_type_success = {}
        for task_type in task_type_counts:
            task_events = [e for e in events if e.get('task_type') == task_type]
            successes = sum(1 for e in task_events if e.get('success', False))
            task_type_success[task_type] = successes / len(task_events) if task_events else 0
        
        return {
            'distribution': task_type_counts,
            'success_rates': task_type_success,
            'most_common': max(task_type_counts.items(), key=lambda x: x[1])[0] if task_type_counts else None
        }

    def _analyze_strategy_effectiveness(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze effectiveness of different retrieval strategies"""
        strategy_performance = defaultdict(list)
        
        for event in events:
            strategy = event.get('retrieval_strategy')
            if strategy:
                strategy_performance[strategy].append({
                    'success': event.get('success', False),
                    'confidence': event.get('confidence_score', 0.0),
                    'execution_time': event.get('execution_time', 0.0)
                })
        
        # Calculate metrics for each strategy
        strategy_metrics = {}
        for strategy, performances in strategy_performance.items():
            success_rate = np.mean([p['success'] for p in performances])
            avg_confidence = np.mean([p['confidence'] for p in performances])
            avg_execution_time = np.mean([p['execution_time'] for p in performances])
            
            strategy_metrics[strategy] = {
                'success_rate': success_rate,
                'avg_confidence': avg_confidence,
                'avg_execution_time': avg_execution_time,
                'sample_size': len(performances)
            }
        
        return strategy_metrics

    def _analyze_success_rate_trends(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze success rate trends over time"""
        # Group events by day
        daily_success = defaultdict(list)
        for event in events:
            day = event['timestamp'].date()
            daily_success[day].append(event.get('success', False))
        
        # Calculate daily success rates
        daily_rates = {}
        for day, successes in daily_success.items():
            daily_rates[day.isoformat()] = np.mean(successes)
        
        # Calculate trend
        if len(daily_rates) >= 2:
            dates = list(daily_rates.keys())
            rates = list(daily_rates.values())
            
            # Simple linear trend calculation
            x = np.arange(len(rates))
            trend_slope = np.corrcoef(x, rates)[0, 1] if len(rates) > 1 else 0
        else:
            trend_slope = 0
        
        return {
            'daily_success_rates': daily_rates,
            'overall_success_rate': np.mean([event.get('success', False) for event in events]),
            'trend_slope': trend_slope,
            'trend_direction': 'improving' if trend_slope > 0.1 else 'declining' if trend_slope < -0.1 else 'stable'
        }

    def _analyze_fallback_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze fallback trigger patterns"""
        fallback_events = [e for e in events if e.get('fallback_triggered', False)]
        
        if not fallback_events:
            return {
                'fallback_rate': 0.0,
                'trigger_distribution': {},
                'common_triggers': []
            }
        
        fallback_rate = len(fallback_events) / len(events)
        triggers = [e.get('fallback_reason') for e in fallback_events if e.get('fallback_reason')]
        trigger_distribution = dict(Counter(triggers))
        
        return {
            'fallback_rate': fallback_rate,
            'trigger_distribution': trigger_distribution,
            'common_triggers': list(trigger_distribution.keys())[:3]
        }

    def _analyze_performance_metrics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        execution_times = [e.get('execution_time', 0) for e in events if e.get('execution_time')]
        confidence_scores = [e.get('confidence_score', 0) for e in events if e.get('confidence_score')]
        
        return {
            'avg_execution_time': np.mean(execution_times) if execution_times else 0,
            'execution_time_std': np.std(execution_times) if execution_times else 0,
            'avg_confidence_score': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_score_std': np.std(confidence_scores) if confidence_scores else 0
        }

    def _analyze_mcp_selection_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze MCP selection patterns"""
        selected_mcps = [e.get('selected_mcp') for e in events if e.get('selected_mcp')]
        mcp_counts = dict(Counter(selected_mcps))
        
        return {
            'most_selected_mcps': dict(sorted(mcp_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'total_unique_mcps': len(mcp_counts),
            'selection_diversity': len(mcp_counts) / len(selected_mcps) if selected_mcps else 0
        }

    def _analyze_user_behavior_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user behavior patterns from context data"""
        # Simplified user behavior analysis
        user_contexts = [e.get('user_context', {}) for e in events]
        
        return {
            'unique_users': len(set(ctx.get('user_id') for ctx in user_contexts if ctx.get('user_id'))),
            'avg_session_length': 1.0,  # Placeholder - would need session tracking
            'common_preferences': {}  # Placeholder - would analyze preference patterns
        }

    def _generate_optimization_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on pattern analysis"""
        recommendations = []
        
        # Strategy effectiveness recommendations
        strategy_metrics = patterns.get('strategy_effectiveness', {})
        if strategy_metrics:
            best_strategy = max(strategy_metrics.items(), key=lambda x: x[1]['success_rate'])
            recommendations.append(f"Consider prioritizing {best_strategy[0]} strategy (success rate: {best_strategy[1]['success_rate']:.2f})")
        
        # Fallback pattern recommendations
        fallback_patterns = patterns.get('fallback_patterns', {})
        if fallback_patterns.get('fallback_rate', 0) > 0.3:
            recommendations.append("High fallback rate detected - consider expanding MCP index")
        
        # Performance recommendations
        performance_metrics = patterns.get('performance_metrics', {})
        if performance_metrics.get('avg_execution_time', 0) > 10:
            recommendations.append("High execution times detected - consider performance optimization")
        
        # Success rate recommendations
        success_trends = patterns.get('success_rate_trends', {})
        if success_trends.get('trend_direction') == 'declining':
            recommendations.append("Declining success rate trend - investigate recent changes")
        
        return recommendations

    async def export_analytics_report(self, output_path: Optional[str] = None) -> str:
        """Export comprehensive analytics report"""
        analysis = await self.analyze_usage_patterns()
        
        report_path = output_path or f"./reports/rag_mcp_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Analytics report exported to: {report_path}")
        return report_path


class CrossValidationBridge:
    """
    Bridge to existing cross-validation framework for comprehensive MCP validation
    Integrates RAG-MCP coordination with existing MCPCrossValidationEngine
    """
    
    def __init__(self, cross_validation_engine: MCPCrossValidationEngine):
        """
        Initialize bridge with existing cross-validation engine
        
        Args:
            cross_validation_engine: Existing MCPCrossValidationEngine instance
        """
        self.cross_validation_engine = cross_validation_engine
        self.validation_history: Dict[str, List[CrossValidationResult]] = defaultdict(list)
        self.reliability_scores: Dict[str, float] = {}
        
        logger.info("CrossValidationBridge initialized with existing validation engine")

    async def validate_selected_mcp(self, selected_candidate: MCPCandidate,
                                  validation_config: Optional[Dict[str, Any]] = None) -> CrossValidationResult:
        """
        Perform comprehensive validation on selected MCP using existing framework
        Extends basic sanity checks with full cross-validation methodology
        
        Args:
            selected_candidate: Selected MCP candidate from RAG-MCP coordination
            validation_config: Optional validation configuration
            
        Returns:
            Comprehensive cross-validation results
        """
        logger.info(f"Performing comprehensive validation for MCP: {selected_candidate.mcp_id}")
        
        try:
            # Execute comprehensive validation
            validation_result = await self.cross_validation_engine.validate_mcp_comprehensive(
                selected_candidate.mcp_spec,
                validation_config
            )
            
            # Store in validation history
            self.validation_history[selected_candidate.mcp_id].append(validation_result)
            
            # Update reliability scores
            self._update_reliability_score(selected_candidate.mcp_id, validation_result)
            
            logger.info(f"Comprehensive validation completed for MCP: {selected_candidate.mcp_id}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive validation: {e}")
            # Return failed validation result
            return CrossValidationResult(
                validation_id=str(uuid.uuid4()),
                mcp_spec=selected_candidate.mcp_spec,
                k_value=0,
                fold_results=[],
                aggregated_metrics=ValidationMetrics(
                    consistency_score=0.0,
                    error_rate=1.0,
                    failure_recovery_rate=0.0,
                    uptime_percentage=0.0,
                    output_consistency=0.0,
                    logic_consistency=0.0,
                    temporal_consistency=0.0,
                    cross_model_agreement=0.0,
                    execution_time_avg=0.0,
                    execution_time_std=0.0,
                    throughput_ops_per_sec=0.0,
                    resource_efficiency=0.0,
                    scalability_score=0.0,
                    ground_truth_accuracy=0.0,
                    expert_validation_score=0.0,
                    benchmark_performance=0.0,
                    cross_validation_accuracy=0.0,
                    confidence_interval=(0.0, 0.0),
                    p_value=1.0,
                    statistical_significance=False,
                    sample_size=0,
                    validation_duration=0.0
                ),
                statistical_analysis={'error': str(e)},
                error_analysis={'validation_error': str(e)},
                recommendations=["Validation failed - exclude from use"],
                is_valid=False,
                confidence_score=0.0
            )

    async def get_validation_history(self, mcp_id: str) -> List[CrossValidationResult]:
        """
        Get validation history for specific MCP
        
        Args:
            mcp_id: MCP identifier
            
        Returns:
            List of historical validation results
        """
        return self.validation_history.get(mcp_id, [])

    def get_reliability_score(self, mcp_id: str) -> float:
        """
        Get current reliability score for MCP based on validation history
        
        Args:
            mcp_id: MCP identifier
            
        Returns:
            Reliability score (0.0 to 1.0)
        """
        return self.reliability_scores.get(mcp_id, 0.5)  # Default to neutral

    def _update_reliability_score(self, mcp_id: str, validation_result: CrossValidationResult):
        """Update reliability score based on validation results"""
        try:
            # Calculate reliability based on validation metrics
            metrics = validation_result.aggregated_metrics
            
            # Weighted combination of key reliability indicators
            reliability_components = [
                metrics.consistency_score * 0.25,
                (1 - metrics.error_rate) * 0.25,
                metrics.failure_recovery_rate * 0.15,
                metrics.uptime_percentage * 0.10,
                metrics.cross_validation_accuracy * 0.25
            ]
            
            new_reliability = sum(reliability_components)
            
            # Exponential moving average with previous scores
            if mcp_id in self.reliability_scores:
                alpha = 0.3  # Learning rate
                self.reliability_scores[mcp_id] = (
                    alpha * new_reliability + 
                    (1 - alpha) * self.reliability_scores[mcp_id]
                )
            else:
                self.reliability_scores[mcp_id] = new_reliability
                
        except Exception as e:
            logger.error(f"Error updating reliability score for {mcp_id}: {e}")

    async def update_mcp_reliability_scores(self, batch_results: List[CrossValidationResult]):
        """
        Batch update reliability scores from multiple validation results
        
        Args:
            batch_results: List of validation results to process
        """
        for result in batch_results:
            self._update_reliability_score(result.mcp_spec.mcp_id, result)
        
        logger.info(f"Updated reliability scores for {len(batch_results)} MCPs")

    def get_validation_analytics(self) -> Dict[str, Any]:
        """Get analytics on validation patterns and performance"""
        total_validations = sum(len(history) for history in self.validation_history.values())
        
        if total_validations == 0:
            return {
                'total_validations': 0,
                'validated_mcps': 0,
                'avg_reliability': 0.0,
                'validation_success_rate': 0.0
            }
        
        # Calculate success rate
        successful_validations = 0
        for history in self.validation_history.values():
            successful_validations += sum(1 for result in history if result.is_valid)
        
        success_rate = successful_validations / total_validations
        avg_reliability = np.mean(list(self.reliability_scores.values())) if self.reliability_scores else 0.0
        
        return {
            'total_validations': total_validations,
            'validated_mcps': len(self.validation_history),
            'avg_reliability': avg_reliability,
            'validation_success_rate': success_rate,
            'reliability_distribution': dict(self.reliability_scores)
        }


class RAGMCPCoordinator:
    """
    Main RAG-MCP Coordinator implementing comprehensive orchestration
    Integrates all components for complete RAG-MCP framework implementation
    """
    
    def __init__(self, config: Dict[str, Any], 
                 llm_client: Optional[Any] = None,
                 knowledge_graph_client: Optional[Any] = None,
                 cross_validation_engine: Optional[MCPCrossValidationEngine] = None,
                 error_management_system: Optional[KGoTErrorManagementSystem] = None):
        """
        Initialize RAG-MCP Coordinator with all required components
        
        Args:
            config: Configuration dictionary containing all system parameters
            llm_client: LangChain LLM client for AI operations
            knowledge_graph_client: Knowledge graph interface
            cross_validation_engine: Cross-validation engine for comprehensive testing
            error_management_system: Error management system for resilience
        """
        self.config = config
        self.llm_client = llm_client
        self.knowledge_graph_client = knowledge_graph_client
        self.error_management_system = error_management_system
        
        # Initialize core components
        self.retrieval_strategy = RetrievalFirstStrategy(config.get('retrieval', {}))
        self.validation_layer = MCPValidationLayer(
            config.get('validation', {}), 
            llm_client
        )
        self.intelligent_fallback = IntelligentFallback(
            config.get('fallback', {}),
            config.get('brainstorming_engine_path')
        )
        self.usage_tracker = UsagePatternTracker(config.get('usage_tracking', {}))
        
        # Initialize cross-validation bridge if engine provided
        self.cross_validation_bridge = None
        if cross_validation_engine:
            self.cross_validation_bridge = CrossValidationBridge(cross_validation_engine)
        
        # Coordination state
        self.coordination_history: deque = deque(maxlen=500)
        self.performance_metrics = {
            'total_requests': 0,
            'successful_coordinations': 0,
            'fallback_triggered': 0,
            'avg_execution_time': 0.0
        }
        
        logger.info("RAGMCPCoordinator initialized with complete component integration")

    async def coordinate_mcp_selection(self, request: RAGMCPRequest,
                                     options: Optional[Dict[str, Any]] = None) -> CoordinationResult:
        """
        Main coordination method implementing complete RAG-MCP pipeline
        Orchestrates: Retrieval → Validation → Selection → Fallback (if needed)
        
        Args:
            request: RAG-MCP coordination request
            options: Optional coordination parameters
            
        Returns:
            Complete coordination result with selected MCP and analytics
        """
        start_time = time.time()
        logger.info(f"Starting RAG-MCP coordination for request: {request.request_id}")
        
        try:
            # Phase 1: Retrieval-First Strategy
            logger.info("Phase 1: Executing retrieval-first strategy")
            candidates = await self.retrieval_strategy.retrieve_mcp_candidates(request)
            
            if not candidates:
                logger.warning("No candidates retrieved - triggering fallback")
                return await self._handle_no_candidates_fallback(request, start_time)
            
            # Phase 2: Validation Layer  
            logger.info(f"Phase 2: Validating {len(candidates)} candidates")
            validation_results = await self.validation_layer.validate_mcp_candidates(
                candidates, request
            )
            
            # Phase 3: Intelligent Selection
            logger.info("Phase 3: Performing intelligent selection")
            selection_result = await self._perform_intelligent_selection(
                candidates, validation_results, request
            )
            
            # Phase 4: Fallback Detection and Handling
            fallback_trigger = await self.intelligent_fallback.detect_retrieval_insufficiency(
                candidates, validation_results, request
            )
            
            if fallback_trigger:
                logger.info(f"Phase 4: Fallback triggered - {fallback_trigger.value}")
                return await self._handle_fallback(request, fallback_trigger, start_time)
            
            # Phase 5: Comprehensive Validation (if available)
            if self.cross_validation_bridge and selection_result['selected_candidate']:
                logger.info("Phase 5: Performing comprehensive validation")
                comprehensive_validation = await self.cross_validation_bridge.validate_selected_mcp(
                    selection_result['selected_candidate']
                )
                selection_result['comprehensive_validation'] = comprehensive_validation
            
            # Generate final coordination result
            execution_time = time.time() - start_time
            coordination_result = CoordinationResult(
                request_id=request.request_id,
                selected_mcp=selection_result['selected_candidate'],
                coordination_strategy="rag_mcp_pipeline",
                confidence_score=selection_result['confidence_score'],
                fallback_triggered=False,
                fallback_reason=None,
                validation_summary=selection_result['best_validation'],
                retrieval_analytics={
                    'candidates_retrieved': len(candidates),
                    'validation_success_rate': len([v for v in validation_results if v.is_approved]) / len(validation_results),
                    'avg_similarity_score': np.mean([c.similarity_score for c in candidates]),
                    'retrieval_strategy': request.retrieval_strategy.value
                },
                execution_time=execution_time,
                resource_usage=self._calculate_resource_usage(execution_time),
                recommendations=selection_result['recommendations'],
                success=True
            )
            
            # Track usage patterns
            await self.usage_tracker.track_retrieval_event(request, coordination_result)
            
            # Update performance metrics
            self._update_performance_metrics(coordination_result)
            
            # Store in coordination history
            self.coordination_history.append(coordination_result)
            
            logger.info(f"RAG-MCP coordination completed successfully: {request.request_id}")
            return coordination_result
            
        except Exception as e:
            logger.error(f"Error in RAG-MCP coordination: {e}")
            return await self._handle_coordination_error(request, e, start_time)

    async def _perform_intelligent_selection(self, candidates: List[MCPCandidate],
                                           validation_results: List[ValidationSummary],
                                           request: RAGMCPRequest) -> Dict[str, Any]:
        """
        Perform intelligent selection from validated candidates
        Combines similarity scores, validation results, and Pareto optimization
        """
        approved_candidates = []
        approved_validations = []
        
        # Filter approved candidates
        for candidate, validation in zip(candidates, validation_results):
            if validation.is_approved:
                approved_candidates.append(candidate)
                approved_validations.append(validation)
        
        if not approved_candidates:
            return {
                'selected_candidate': None,
                'confidence_score': 0.0,
                'best_validation': None,
                'recommendations': ["No approved candidates - fallback required"]
            }
        
        # Calculate final selection scores
        candidate_scores = []
        for candidate, validation in zip(approved_candidates, approved_validations):
            # Multi-criteria scoring
            final_score = (
                candidate.similarity_score * 0.3 +
                candidate.pareto_score * 0.25 +
                validation.compatibility_score * 0.25 +
                candidate.reliability_score * 0.2
            )
            
            candidate_scores.append({
                'candidate': candidate,
                'validation': validation,
                'final_score': final_score
            })
        
        # Select best candidate
        best_result = max(candidate_scores, key=lambda x: x['final_score'])
        
        # Generate recommendations
        recommendations = [
            f"Selected {best_result['candidate'].mcp_spec.name} with confidence {best_result['final_score']:.2f}",
            f"Similarity score: {best_result['candidate'].similarity_score:.2f}",
            f"Validation confidence: {best_result['validation'].validation_confidence.value}"
        ]
        
        return {
            'selected_candidate': best_result['candidate'],
            'confidence_score': best_result['final_score'],
            'best_validation': best_result['validation'],
            'recommendations': recommendations
        }

    async def _handle_no_candidates_fallback(self, request: RAGMCPRequest, start_time: float) -> CoordinationResult:
        """Handle fallback when no candidates are retrieved"""
        execution_time = time.time() - start_time
        
        # Trigger fallback to MCP creation
        fallback_result = await self.intelligent_fallback.fallback_to_mcp_creation(
            request, FallbackTrigger.LOW_SIMILARITY
        )
        
        coordination_result = CoordinationResult(
            request_id=request.request_id,
            selected_mcp=None,
            coordination_strategy="fallback_creation",
            confidence_score=0.0,
            fallback_triggered=True,
            fallback_reason=FallbackTrigger.LOW_SIMILARITY,
            validation_summary=None,
            retrieval_analytics={
                'candidates_retrieved': 0,
                'fallback_result': fallback_result
            },
            execution_time=execution_time,
            resource_usage=self._calculate_resource_usage(execution_time),
            recommendations=["No suitable MCPs found - created new MCP via fallback"],
            success=fallback_result.get('success', False)
        )
        
        return coordination_result

    async def _handle_fallback(self, request: RAGMCPRequest, 
                             trigger: FallbackTrigger, start_time: float) -> CoordinationResult:
        """Handle fallback scenarios"""
        execution_time = time.time() - start_time
        
        # Execute fallback to MCP creation
        fallback_result = await self.intelligent_fallback.fallback_to_mcp_creation(
            request, trigger
        )
        
        coordination_result = CoordinationResult(
            request_id=request.request_id,
            selected_mcp=None,
            coordination_strategy="intelligent_fallback",
            confidence_score=0.0,
            fallback_triggered=True,
            fallback_reason=trigger,
            validation_summary=None,
            retrieval_analytics={
                'fallback_trigger': trigger.value,
                'fallback_result': fallback_result
            },
            execution_time=execution_time,
            resource_usage=self._calculate_resource_usage(execution_time),
            recommendations=[f"Fallback triggered due to {trigger.value}"],
            success=fallback_result.get('success', False)
        )
        
        return coordination_result

    async def _handle_coordination_error(self, request: RAGMCPRequest, 
                                       error: Exception, start_time: float) -> CoordinationResult:
        """Handle coordination errors"""
        execution_time = time.time() - start_time
        
        # Log error with error management system if available
        if self.error_management_system:
            try:
                await self.error_management_system.handle_error(
                    error,
                    ErrorContext(
                        operation="rag_mcp_coordination",
                        component="RAGMCPCoordinator",
                        request_id=request.request_id,
                        metadata={'task_description': request.task_description}
                    )
                )
            except Exception as e:
                logger.error(f"Error management system failed: {e}")
        
        return CoordinationResult(
            request_id=request.request_id,
            selected_mcp=None,
            coordination_strategy="error_handling",
            confidence_score=0.0,
            fallback_triggered=False,
            fallback_reason=None,
            validation_summary=None,
            retrieval_analytics={'error': str(error)},
            execution_time=execution_time,
            resource_usage=self._calculate_resource_usage(execution_time),
            recommendations=["Coordination failed - retry with different parameters"],
            success=False
        )

    def _calculate_resource_usage(self, execution_time: float) -> Dict[str, Any]:
        """Calculate resource usage metrics"""
        return {
            'execution_time': execution_time,
            'api_calls': 1,  # Placeholder - would track actual API calls
            'memory_usage': 0.0,  # Placeholder - would track actual memory
            'cost_estimate': execution_time * 0.001  # Placeholder cost calculation
        }

    def _update_performance_metrics(self, result: CoordinationResult):
        """Update coordinator performance metrics"""
        self.performance_metrics['total_requests'] += 1
        
        if result.success:
            self.performance_metrics['successful_coordinations'] += 1
        
        if result.fallback_triggered:
            self.performance_metrics['fallback_triggered'] += 1
        
        # Update average execution time
        current_avg = self.performance_metrics['avg_execution_time']
        total_requests = self.performance_metrics['total_requests']
        self.performance_metrics['avg_execution_time'] = (
            (current_avg * (total_requests - 1) + result.execution_time) / total_requests
        )

    async def get_coordination_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive coordination analytics
        Combines metrics from all components for system-wide insights
        """
        # Get usage pattern analysis
        usage_analytics = await self.usage_tracker.analyze_usage_patterns()
        
        # Get fallback analytics
        fallback_analytics = self.intelligent_fallback.get_fallback_analytics()
        
        # Get cross-validation analytics if available
        cross_validation_analytics = {}
        if self.cross_validation_bridge:
            cross_validation_analytics = self.cross_validation_bridge.get_validation_analytics()
        
        # Combine all analytics
        return {
            'coordinator_performance': self.performance_metrics,
            'usage_patterns': usage_analytics,
            'fallback_patterns': fallback_analytics,
            'cross_validation_metrics': cross_validation_analytics,
            'recent_coordinations': [
                asdict(result) for result in list(self.coordination_history)[-10:]
            ],
            'mcp_index_stats': {
                'total_mcps': len(self.retrieval_strategy.mcp_index),
                'cached_embeddings': len(self.retrieval_strategy.embedding_cache)
            }
        }

    async def initialize_mcp_index(self, mcp_data: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize or update MCP index with new data
        
        Args:
            mcp_data: Optional list of MCP data to add to index
        """
        if mcp_data:
            for mcp_info in mcp_data:
                mcp_id = mcp_info.get('id', f"mcp_{uuid.uuid4().hex[:8]}")
                self.retrieval_strategy.mcp_index[mcp_id] = mcp_info
            
            self.retrieval_strategy._save_mcp_index()
            logger.info(f"Added {len(mcp_data)} MCPs to index")
        
        logger.info(f"MCP index initialized with {len(self.retrieval_strategy.mcp_index)} entries")


# Utility Functions and Factory Methods

def create_rag_mcp_coordinator(config_path: Optional[str] = None,
                             **kwargs) -> RAGMCPCoordinator:
    """
    Factory function to create RAG-MCP Coordinator with default configuration
    
    Args:
        config_path: Optional path to configuration file
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured RAGMCPCoordinator instance
    """
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Apply overrides
    config.update(kwargs)
    
    # Create LLM client with OpenRouter (per user memory)
    llm_client = ChatOpenAI(
        openai_api_key=config.get('openrouter_api_key'),
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=config.get('llm_model', 'anthropic/claude-sonnet-4'),
        temperature=0.1
    )
    
    # Create cross-validation engine if enabled
    cross_validation_engine = None
    if config.get('enable_cross_validation', True):
        try:
            from validation.mcp_cross_validator import create_mcp_cross_validation_engine
            cross_validation_engine = create_mcp_cross_validation_engine(
                config.get('cross_validation', {}),
                llm_client=llm_client
            )
        except ImportError:
            logger.warning("Cross-validation engine not available")
    
    # Create error management system if available
    error_management_system = None
    try:
        error_management_system = KGoTErrorManagementSystem(
            config.get('error_management', {})
        )
    except Exception as e:
        logger.warning(f"Error management system not available: {e}")
    
    return RAGMCPCoordinator(
        config=config,
        llm_client=llm_client,
        cross_validation_engine=cross_validation_engine,
        error_management_system=error_management_system
    )


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for RAG-MCP Coordinator"""
    return {
        'retrieval': {
            'similarity_threshold': 0.3,
            'pareto_weights': {
                'usage_frequency': 0.4,
                'reliability_score': 0.35,
                'cost_efficiency': 0.25
            },
            'mcp_index_path': './data/mcp_index.json',
            'embedding_model': 'text-embedding-ada-002'
        },
        'validation': {
            'validation_timeout': 10,
            'confidence_threshold': 0.6,
            'validation_model': 'anthropic/claude-sonnet-4'
        },
        'fallback': {
            'fallback_similarity_threshold': 0.3,
            'fallback_validation_threshold': 0.5,
            'fallback_performance_threshold': 0.4
        },
        'usage_tracking': {
            'usage_data_path': './data/usage_patterns.json',
            'analytics_window_days': 30
        },
        'cross_validation': {
            'k_fold': 5,
            'significance_level': 0.05
        },
        'error_management': {
            'max_retries': 3,
            'retry_delay': 1.0
        }
    }


# Example Usage and Testing

async def example_rag_mcp_coordination():
    """
    Example usage of RAG-MCP Coordinator
    Demonstrates complete pipeline for MCP selection and coordination
    """
    print("🚀 RAG-MCP Coordinator Example Usage")
    print("=" * 50)
    
    # Create coordinator
    coordinator = create_rag_mcp_coordinator(
        openrouter_api_key="your-openrouter-key",
        enable_cross_validation=True
    )
    
    # Initialize MCP index
    await coordinator.initialize_mcp_index()
    
    # Create sample request
    request = RAGMCPRequest(
        request_id=str(uuid.uuid4()),
        task_description="I need to scrape product data from an e-commerce website and analyze pricing trends",
        task_type=TaskType.WEB_SCRAPING,
        retrieval_strategy=RetrievalStrategy.HYBRID_APPROACH,
        max_candidates=5,
        min_confidence=0.6,
        user_context={'user_id': 'demo_user', 'session_id': 'demo_session'}
    )
    
    print(f"📝 Processing request: {request.task_description}")
    print(f"🔍 Strategy: {request.retrieval_strategy.value}")
    
    # Execute coordination
    result = await coordinator.coordinate_mcp_selection(request)
    
    # Display results
    print(f"\n✅ Coordination Result:")
    print(f"   Success: {result.success}")
    print(f"   Selected MCP: {result.selected_mcp.mcp_id if result.selected_mcp else 'None'}")
    print(f"   Confidence: {result.confidence_score:.2f}")
    print(f"   Fallback Triggered: {result.fallback_triggered}")
    print(f"   Execution Time: {result.execution_time:.2f}s")
    
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   • {rec}")
    
    # Get analytics
    analytics = await coordinator.get_coordination_analytics()
    print(f"\n📊 System Analytics:")
    print(f"   Total Requests: {analytics['coordinator_performance']['total_requests']}")
    print(f"   Success Rate: {analytics['coordinator_performance']['successful_coordinations'] / max(analytics['coordinator_performance']['total_requests'], 1):.2f}")
    print(f"   Average Execution Time: {analytics['coordinator_performance']['avg_execution_time']:.2f}s")
    
    return result


if __name__ == "__main__":
    """
    Main execution for testing and demonstration
    Run this module directly to see RAG-MCP Coordinator in action
    """
    import asyncio
    
    # Set up logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_rag_mcp_coordination()) 