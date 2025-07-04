#!/usr/bin/env python3
"""
KGoT-Alita Cross-Modal Validator - Task 28 Implementation

Advanced cross-modal validation system implementing comprehensive consistency checking,
contradiction detection, confidence scoring, and quality assurance as specified in the
5-Phase Implementation Plan for Enhanced Alita.

This module provides:
- Consistency checking using KGoT Section 2.1 knowledge validation
- Contradiction detection leveraging KGoT Section 2.5 "Error Management"  
- Confidence scoring using KGoT analytical capabilities
- Quality assurance using both KGoT and Alita validation frameworks
- Multi-modal input validation (text, images, audio, video, structured data)
- Statistical significance testing and comprehensive metrics
- Integration with existing validation and error management systems

Model Specialization (@modelsrule.mdc):
- o3(vision): Visual analysis, image processing, and visual content understanding
- claude-4-sonnet(webagent): Reasoning, consistency checking, knowledge validation, contradiction detection
- gemini-2.5-pro(orchestration): Coordination, confidence scoring, statistical analysis, and orchestration

Key Components:
1. ModalityConsistencyChecker: Cross-modal consistency validation
2. KGoTKnowledgeValidator: Knowledge graph-based validation
3. ContradictionDetector: Logical and factual contradiction detection
4. ConfidenceScorer: Multi-modal confidence assessment
5. QualityAssuranceEngine: Unified quality assurance framework
6. KGoTAlitaCrossModalValidator: Main orchestrator for comprehensive validation

@author: Enhanced Alita KGoT Team
@version: 1.0.0
@task: Task 28 - Build KGoT-Alita Cross-Modal Validator
"""

import asyncio
import json
import logging
import time
import uuid
import base64
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
import warnings

# Statistical analysis and machine learning
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, kruskal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangChain for agent development (per user memory requirement)
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Integration with existing systems
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import KGoT components
from kgot_core.error_management import (
    KGoTErrorManagementSystem, 
    ErrorType, 
    ErrorSeverity, 
    ErrorContext
)
from kgot_core.knowledge_extraction import (
    KnowledgeExtractionManager,
    ExtractionContext,
    ExtractionMetrics
)
from kgot_core.graph_store.kg_interface import KnowledgeGraphInterface

# Import existing validation frameworks
from validation.mcp_cross_validator import (
    ValidationMetrics,
    StatisticalSignificanceAnalyzer,
    MCPValidationSpec
)

# Import multimodal components
from multimodal.kgot_visual_analyzer import KGoTVisualAnalyzer, VisualAnalysisConfig

# Winston-compatible logging setup
logger = logging.getLogger('CrossModalValidator')
handler = logging.FileHandler('./logs/multimodal/combined.log')
formatter = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ModalityType(Enum):
    """
    Enumeration of supported input modality types for cross-modal validation
    """
    TEXT = "text"                           # Natural language text content
    IMAGE = "image"                         # Visual content (photos, diagrams, etc.)
    AUDIO = "audio"                         # Audio content (speech, music, sounds)
    VIDEO = "video"                         # Video content with visual and audio
    STRUCTURED_DATA = "structured_data"     # JSON, CSV, XML, databases
    CODE = "code"                          # Programming code in various languages
    GRAPH_DATA = "graph_data"              # Knowledge graph data (RDF, property graphs)


class ValidationLevel(Enum):
    """
    Validation depth levels for different use cases
    """
    BASIC = "basic"                        # Basic consistency checks
    STANDARD = "standard"                  # Standard validation with knowledge checks
    COMPREHENSIVE = "comprehensive"        # Full validation with all features
    EXPERT = "expert"                      # Expert-level validation with detailed analysis


class ContradictionType(Enum):
    """
    Types of contradictions that can be detected across modalities
    """
    LOGICAL = "logical"                    # Logical inconsistencies
    FACTUAL = "factual"                    # Factual contradictions
    TEMPORAL = "temporal"                  # Time-related conflicts
    SEMANTIC = "semantic"                  # Meaning-related conflicts
    STRUCTURAL = "structural"              # Structure or format conflicts


@dataclass
class CrossModalInput:
    """
    Represents a single input item with its modality type and content
    """
    input_id: str
    modality_type: ModalityType
    content: Any                           # Content varies by modality type
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: Optional[float] = None     # Optional confidence from source


@dataclass
class CrossModalValidationSpec:
    """
    Comprehensive specification for cross-modal validation tasks
    Extends existing validation specs with multi-modal capabilities
    """
    validation_id: str
    name: str
    description: str
    inputs: List[CrossModalInput]
    validation_level: ValidationLevel
    expected_consistency: bool = True
    knowledge_context: Optional[Dict[str, Any]] = None
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConsistencyScore:
    """
    Represents consistency scores between different modalities
    """
    modality_pair: Tuple[ModalityType, ModalityType]
    semantic_consistency: float            # Semantic alignment score (0.0-1.0)
    factual_consistency: float             # Factual agreement score (0.0-1.0)
    temporal_consistency: float            # Temporal alignment score (0.0-1.0)
    overall_consistency: float             # Weighted overall score (0.0-1.0)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ContradictionReport:
    """
    Detailed report of detected contradictions
    """
    contradiction_id: str
    contradiction_type: ContradictionType
    severity: ErrorSeverity
    description: str
    affected_modalities: List[ModalityType]
    evidence: List[str]
    resolution_suggestions: List[str]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CrossModalMetrics:
    """
    Comprehensive metrics for cross-modal validation extending ValidationMetrics
    """
    # Cross-modal specific metrics
    modality_consistency_scores: Dict[str, ConsistencyScore]
    knowledge_validation_score: float
    contradiction_count: int
    contradiction_severity_distribution: Dict[str, int]
    
    # Confidence metrics
    individual_modality_confidence: Dict[ModalityType, float]
    cross_modal_agreement_score: float
    knowledge_supported_confidence: float
    overall_confidence: float
    
    # Quality assurance metrics  
    kgot_quality_score: float
    alita_quality_score: float
    unified_quality_score: float
    
    # Performance metrics
    validation_duration: float
    processing_time_per_modality: Dict[ModalityType, float]
    resource_usage: Dict[str, float]
    
    # Statistical metrics
    statistical_significance: bool
    confidence_interval: Tuple[float, float]
    p_value: float
    
    # Base validation metrics (inherited concepts)
    reliability_score: float = 0.0
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    performance_score: float = 0.0
    
    # Metadata
    sample_size: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """
    Complete result of cross-modal validation process
    """
    validation_id: str
    spec: CrossModalValidationSpec
    metrics: CrossModalMetrics
    contradictions: List[ContradictionReport]
    recommendations: List[str]
    is_valid: bool
    overall_score: float
    detailed_analysis: Dict[str, Any]
    processing_logs: List[Dict[str, Any]] = field(default_factory=list)


class ModalityConsistencyChecker:
    """
    Handles consistency checking between different input modalities
    
    This class implements cross-modal consistency validation by comparing semantic
    content, factual information, and temporal alignment across different modalities.
    Uses advanced NLP and vision models to assess consistency and detect conflicts.
    """
    
    def __init__(self, 
                 llm_client: Any,
                 visual_analyzer: Optional[KGoTVisualAnalyzer] = None,
                 similarity_threshold: float = 0.7,
                 vision_client: Optional[Any] = None):
        """
        Initialize the modality consistency checker
        Uses @modelsrule.mdc: claude-4-sonnet(webagent) for reasoning, o3(vision) for visual tasks
        
        @param {Any} llm_client - LLM client for reasoning (claude-4-sonnet)
        @param {KGoTVisualAnalyzer} visual_analyzer - Visual analysis component
        @param {float} similarity_threshold - Threshold for consistency determination
        @param {Any} vision_client - Specialized vision client (o3-vision)
        """
        self.llm_client = llm_client  # claude-4-sonnet for reasoning
        self.vision_client = vision_client or llm_client  # o3(vision) for visual tasks
        self.visual_analyzer = visual_analyzer or KGoTVisualAnalyzer()
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        logger.info("Initialized Modality Consistency Checker", extra={
            'operation': 'CONSISTENCY_CHECKER_INIT',
            'similarity_threshold': similarity_threshold,
            'has_visual_analyzer': visual_analyzer is not None
        })
    
    async def check_cross_modal_consistency(self, 
                                          inputs: List[CrossModalInput]) -> Dict[str, ConsistencyScore]:
        """
        Check consistency across all provided modalities
        
        @param {List[CrossModalInput]} inputs - List of inputs to validate for consistency
        @returns {Dict[str, ConsistencyScore]} - Consistency scores for all modality pairs
        """
        logger.info("Starting cross-modal consistency check", extra={
            'operation': 'CROSS_MODAL_CONSISTENCY_START',
            'input_count': len(inputs),
            'modalities': [inp.modality_type.value for inp in inputs]
        })
        
        consistency_scores = {}
        modality_pairs = []
        
        # Generate all unique pairs of modalities
        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                pair_key = f"{inputs[i].modality_type.value}_{inputs[j].modality_type.value}"
                modality_pairs.append((inputs[i], inputs[j], pair_key))
        
        # Check consistency for each pair
        for input1, input2, pair_key in modality_pairs:
            try:
                consistency_score = await self._check_pair_consistency(input1, input2)
                consistency_scores[pair_key] = consistency_score
                
                logger.info("Completed pair consistency check", extra={
                    'operation': 'PAIR_CONSISTENCY_COMPLETE',
                    'pair': pair_key,
                    'overall_score': consistency_score.overall_consistency
                })
                
            except Exception as e:
                logger.error("Failed consistency check for pair", extra={
                    'operation': 'PAIR_CONSISTENCY_FAILED',
                    'pair': pair_key,
                    'error': str(e)
                })
                # Create failed consistency score
                consistency_scores[pair_key] = ConsistencyScore(
                    modality_pair=(input1.modality_type, input2.modality_type),
                    semantic_consistency=0.0,
                    factual_consistency=0.0,
                    temporal_consistency=0.0,
                    overall_consistency=0.0,
                    contradictions=[{
                        'type': 'processing_error',
                        'description': f"Failed to process pair: {str(e)}"
                    }]
                )
        
        return consistency_scores
    
    async def _check_pair_consistency(self, 
                                    input1: CrossModalInput, 
                                    input2: CrossModalInput) -> ConsistencyScore:
        """
        Check consistency between a specific pair of modality inputs
        
        @param {CrossModalInput} input1 - First input for comparison
        @param {CrossModalInput} input2 - Second input for comparison
        @returns {ConsistencyScore} - Detailed consistency assessment
        """
        modality_pair = (input1.modality_type, input2.modality_type)
        
        # Extract semantic content from both modalities
        content1 = await self._extract_semantic_content(input1)
        content2 = await self._extract_semantic_content(input2)
        
        # Calculate different types of consistency
        semantic_score = await self._calculate_semantic_consistency(content1, content2)
        factual_score = await self._calculate_factual_consistency(content1, content2)
        temporal_score = await self._calculate_temporal_consistency(input1, input2)
        
        # Detect contradictions
        contradictions = await self._detect_pair_contradictions(content1, content2, modality_pair)
        
        # Calculate overall consistency with weights
        weights = {'semantic': 0.4, 'factual': 0.4, 'temporal': 0.2}
        overall_score = (
            semantic_score * weights['semantic'] +
            factual_score * weights['factual'] +
            temporal_score * weights['temporal']
        )
        
        # Calculate confidence based on individual modality confidences and agreement
        confidence = self._calculate_pair_confidence(input1, input2, overall_score)
        
        return ConsistencyScore(
            modality_pair=modality_pair,
            semantic_consistency=semantic_score,
            factual_consistency=factual_score,
            temporal_consistency=temporal_score,
            overall_consistency=overall_score,
            contradictions=contradictions,
            confidence=confidence
        )
    
    async def _extract_semantic_content(self, input_item: CrossModalInput) -> str:
        """
        Extract semantic content from different modality types
        
        @param {CrossModalInput} input_item - Input item to extract content from
        @returns {str} - Extracted semantic content as text
        """
        try:
            if input_item.modality_type == ModalityType.TEXT:
                return str(input_item.content)
                
            elif input_item.modality_type == ModalityType.IMAGE:
                # Use o3(vision) model for image analysis via vision_client
                try:
                    if self.visual_analyzer:
                        # Use existing visual analyzer with o3(vision) model
                        analysis_result = await self.visual_analyzer.analyze_image_with_graph_context(
                            input_item.content,
                            {'extract_semantic_description': True}
                        )
                        return analysis_result.get('semantic_description', '')
                    else:
                        # Direct o3(vision) analysis if visual analyzer not available
                        vision_prompt = f"""
                        Analyze this image and provide a detailed semantic description:
                        
                        Image: {input_item.content}
                        
                        Describe what you see, focusing on:
                        - Objects and their properties
                        - Spatial relationships
                        - Activities or actions
                        - Context and setting
                        - Any text or symbols visible
                        
                        Provide a comprehensive semantic description.
                        """
                        response = await self.vision_client.acomplete(vision_prompt)
                        return response.text if hasattr(response, 'text') else str(response)
                except Exception as e:
                    logger.error("Failed to analyze image with o3(vision)", extra={
                        'operation': 'IMAGE_SEMANTIC_EXTRACTION_FAILED',
                        'error': str(e)
                    })
                    return f"Image analysis failed: {str(e)}"
                
            elif input_item.modality_type == ModalityType.CODE:
                # Extract semantic meaning from code
                code_analysis_prompt = f"""
                Analyze this code and extract its semantic meaning and purpose:
                
                Code:
                {input_item.content}
                
                Provide a semantic description of what this code does, its purpose, 
                and key functionality in natural language.
                """
                response = await self.llm_client.acomplete(code_analysis_prompt)
                return response.text if hasattr(response, 'text') else str(response)
                
            elif input_item.modality_type == ModalityType.STRUCTURED_DATA:
                # Extract semantic content from structured data
                data_str = json.dumps(input_item.content) if isinstance(input_item.content, dict) else str(input_item.content)
                analysis_prompt = f"""
                Analyze this structured data and extract its semantic meaning:
                
                Data:
                {data_str}
                
                Provide a semantic description of what this data represents,
                its key information, and meaning.
                """
                response = await self.llm_client.acomplete(analysis_prompt)
                return response.text if hasattr(response, 'text') else str(response)
                
            else:
                logger.warning("Unsupported modality type for semantic extraction", extra={
                    'operation': 'SEMANTIC_EXTRACTION_UNSUPPORTED',
                    'modality_type': input_item.modality_type.value
                })
                return f"Unsupported modality: {input_item.modality_type.value}"
                
        except Exception as e:
            logger.error("Failed to extract semantic content", extra={
                'operation': 'SEMANTIC_EXTRACTION_FAILED',
                'modality_type': input_item.modality_type.value,
                'error': str(e)
            })
            return f"Extraction failed: {str(e)}"
    
    async def _calculate_semantic_consistency(self, content1: str, content2: str) -> float:
        """
        Calculate semantic consistency between two text contents
        
        @param {str} content1 - First content for comparison
        @param {str} content2 - Second content for comparison
        @returns {float} - Semantic consistency score (0.0-1.0)
        """
        try:
            # Use TF-IDF vectorization for basic similarity
            documents = [content1, content2]
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Use LLM for deeper semantic analysis
            semantic_analysis_prompt = f"""
            Compare these two pieces of content for semantic consistency:
            
            Content 1: {content1}
            Content 2: {content2}
            
            Rate the semantic consistency on a scale of 0.0 to 1.0 where:
            - 1.0: Completely consistent, same meaning
            - 0.5: Partially consistent, some overlap
            - 0.0: Completely inconsistent, contradictory meanings
            
            Provide only the numerical score as a decimal.
            """
            
            response = await self.llm_client.acomplete(semantic_analysis_prompt)
            try:
                llm_score = float(response.text.strip() if hasattr(response, 'text') else str(response).strip())
                llm_score = max(0.0, min(1.0, llm_score))  # Clamp to valid range
            except (ValueError, AttributeError):
                llm_score = 0.5  # Default if parsing fails
            
            # Combine TF-IDF and LLM scores with weights
            combined_score = (cosine_sim * 0.3) + (llm_score * 0.7)
            
            logger.debug("Calculated semantic consistency", extra={
                'operation': 'SEMANTIC_CONSISTENCY_CALC',
                'tfidf_score': cosine_sim,
                'llm_score': llm_score,
                'combined_score': combined_score
            })
            
            return combined_score
            
        except Exception as e:
            logger.error("Failed to calculate semantic consistency", extra={
                'operation': 'SEMANTIC_CONSISTENCY_FAILED',
                'error': str(e)
            })
            return 0.0
    
    async def _calculate_factual_consistency(self, content1: str, content2: str) -> float:
        """
        Calculate factual consistency between two contents
        
        @param {str} content1 - First content for comparison
        @param {str} content2 - Second content for comparison
        @returns {float} - Factual consistency score (0.0-1.0)
        """
        try:
            factual_analysis_prompt = f"""
            Analyze these two pieces of content for factual consistency:
            
            Content 1: {content1}
            Content 2: {content2}
            
            Check for:
            1. Contradictory facts or claims
            2. Inconsistent numbers, dates, or measurements
            3. Conflicting statements about entities or events
            
            Rate the factual consistency on a scale of 0.0 to 1.0 where:
            - 1.0: All facts are consistent, no contradictions
            - 0.5: Some facts are consistent, minor contradictions
            - 0.0: Major factual contradictions present
            
            Provide only the numerical score as a decimal.
            """
            
            response = await self.llm_client.acomplete(factual_analysis_prompt)
            try:
                factual_score = float(response.text.strip() if hasattr(response, 'text') else str(response).strip())
                factual_score = max(0.0, min(1.0, factual_score))  # Clamp to valid range
            except (ValueError, AttributeError):
                factual_score = 0.5  # Default if parsing fails
            
            logger.debug("Calculated factual consistency", extra={
                'operation': 'FACTUAL_CONSISTENCY_CALC',
                'score': factual_score
            })
            
            return factual_score
            
        except Exception as e:
            logger.error("Failed to calculate factual consistency", extra={
                'operation': 'FACTUAL_CONSISTENCY_FAILED',
                'error': str(e)
            })
            return 0.0
    
    async def _calculate_temporal_consistency(self, 
                                           input1: CrossModalInput, 
                                           input2: CrossModalInput) -> float:
        """
        Calculate temporal consistency between two inputs
        
        @param {CrossModalInput} input1 - First input for comparison
        @param {CrossModalInput} input2 - Second input for comparison
        @returns {float} - Temporal consistency score (0.0-1.0)
        """
        try:
            # Check timestamp consistency
            time_diff = abs((input1.timestamp - input2.timestamp).total_seconds())
            
            # If inputs are very close in time, high temporal consistency
            if time_diff < 60:  # Within 1 minute
                timestamp_score = 1.0
            elif time_diff < 3600:  # Within 1 hour
                timestamp_score = 0.8
            elif time_diff < 86400:  # Within 1 day
                timestamp_score = 0.6
            else:
                timestamp_score = 0.4
            
            # Extract and compare temporal content from both inputs
            content1 = await self._extract_semantic_content(input1)
            content2 = await self._extract_semantic_content(input2)
            
            temporal_analysis_prompt = f"""
            Analyze these contents for temporal consistency:
            
            Content 1: {content1}
            Content 2: {content2}
            
            Check for:
            1. Consistent timeline references
            2. Logical temporal ordering
            3. No temporal contradictions
            
            Rate temporal consistency (0.0-1.0):
            """
            
            response = await self.llm_client.acomplete(temporal_analysis_prompt)
            try:
                content_temporal_score = float(response.text.strip() if hasattr(response, 'text') else str(response).strip())
                content_temporal_score = max(0.0, min(1.0, content_temporal_score))
            except (ValueError, AttributeError):
                content_temporal_score = 0.5
            
            # Combine timestamp and content temporal scores
            temporal_score = (timestamp_score * 0.3) + (content_temporal_score * 0.7)
            
            logger.debug("Calculated temporal consistency", extra={
                'operation': 'TEMPORAL_CONSISTENCY_CALC',
                'timestamp_score': timestamp_score,
                'content_score': content_temporal_score,
                'combined_score': temporal_score
            })
            
            return temporal_score
            
        except Exception as e:
            logger.error("Failed to calculate temporal consistency", extra={
                'operation': 'TEMPORAL_CONSISTENCY_FAILED',
                'error': str(e)
            })
            return 0.5  # Default neutral score
    
    async def _detect_pair_contradictions(self, 
                                        content1: str, 
                                        content2: str,
                                        modality_pair: Tuple[ModalityType, ModalityType]) -> List[Dict[str, Any]]:
        """
        Detect contradictions between a pair of contents
        
        @param {str} content1 - First content to analyze
        @param {str} content2 - Second content to analyze
        @param {Tuple[ModalityType, ModalityType]} modality_pair - Pair of modality types
        @returns {List[Dict[str, Any]]} - List of detected contradictions
        """
        try:
            contradiction_analysis_prompt = f"""
            Analyze these contents for contradictions:
            
            Content 1 ({modality_pair[0].value}): {content1}
            Content 2 ({modality_pair[1].value}): {content2}
            
            Identify any contradictions including:
            1. Logical contradictions (A and not-A)
            2. Factual contradictions (conflicting facts)
            3. Semantic contradictions (contradictory meanings)
            
            For each contradiction found, provide:
            - Type: logical/factual/semantic
            - Description: Clear explanation
            - Evidence: Specific text showing contradiction
            
            Format as JSON list: [{"type": "...", "description": "...", "evidence": "..."}]
            If no contradictions, return empty list: []
            """
            
            response = await self.llm_client.acomplete(contradiction_analysis_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                contradictions = json.loads(response_text.strip())
                if not isinstance(contradictions, list):
                    contradictions = []
            except json.JSONDecodeError:
                logger.warning("Failed to parse contradiction analysis response", extra={
                    'operation': 'CONTRADICTION_PARSE_FAILED',
                    'response': response_text[:200]
                })
                contradictions = []
            
            logger.debug("Detected contradictions", extra={
                'operation': 'CONTRADICTION_DETECTION',
                'contradiction_count': len(contradictions),
                'modality_pair': f"{modality_pair[0].value}_{modality_pair[1].value}"
            })
            
            return contradictions
            
        except Exception as e:
            logger.error("Failed to detect contradictions", extra={
                'operation': 'CONTRADICTION_DETECTION_FAILED',
                'error': str(e)
            })
            return []
    
    def _calculate_pair_confidence(self, 
                                 input1: CrossModalInput, 
                                 input2: CrossModalInput,
                                 consistency_score: float) -> float:
        """
        Calculate confidence in the consistency assessment for a pair
        
        @param {CrossModalInput} input1 - First input
        @param {CrossModalInput} input2 - Second input
        @param {float} consistency_score - Calculated consistency score
        @returns {float} - Confidence score (0.0-1.0)
        """
        # Base confidence from individual input confidences
        conf1 = input1.confidence or 0.5
        conf2 = input2.confidence or 0.5
        base_confidence = (conf1 + conf2) / 2
        
        # Adjust confidence based on consistency score
        # High consistency or low consistency both increase confidence in the assessment
        consistency_factor = 1.0 - abs(consistency_score - 0.5) * 2
        
        # Combine factors
        final_confidence = (base_confidence * 0.6) + (consistency_factor * 0.4)
        
        return max(0.0, min(1.0, final_confidence))


class KGoTKnowledgeValidator:
    """
    Implements KGoT Section 2.1 knowledge validation using graph-based approaches
    
    This class validates cross-modal inputs against existing knowledge graphs,
    checks for knowledge consistency, and provides knowledge-supported confidence scoring.
    Integrates with KGoT's knowledge extraction and graph store systems.
    """
    
    def __init__(self, 
                 knowledge_extraction_manager: KnowledgeExtractionManager,
                 graph_store: KnowledgeGraphInterface,
                 llm_client: Any,
                 validation_threshold: float = 0.6):
        """
        Initialize KGoT knowledge validator
        
        @param {KnowledgeExtractionManager} knowledge_extraction_manager - KGoT knowledge extraction system
        @param {KnowledgeGraphInterface} graph_store - Knowledge graph store interface
        @param {Any} llm_client - LLM client for reasoning
        @param {float} validation_threshold - Threshold for knowledge validation acceptance
        """
        self.knowledge_extractor = knowledge_extraction_manager
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.validation_threshold = validation_threshold
        
        logger.info("Initialized KGoT Knowledge Validator", extra={
            'operation': 'KGOT_KNOWLEDGE_VALIDATOR_INIT',
            'validation_threshold': validation_threshold,
            'has_graph_store': graph_store is not None
        })
    
    async def validate_against_knowledge_graph(self, 
                                             inputs: List[CrossModalInput],
                                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate inputs against existing knowledge graph using KGoT Section 2.1 methods
        
        @param {List[CrossModalInput]} inputs - Cross-modal inputs to validate
        @param {Dict[str, Any]} context - Optional validation context
        @returns {Dict[str, Any]} - Comprehensive knowledge validation results
        """
        logger.info("Starting knowledge graph validation", extra={
            'operation': 'KNOWLEDGE_VALIDATION_START',
            'input_count': len(inputs),
            'has_context': context is not None
        })
        
        validation_results = {
            'overall_knowledge_score': 0.0,
            'individual_validations': {},
            'knowledge_contradictions': [],
            'supported_facts': [],
            'unsupported_claims': [],
            'confidence_scores': {},
            'graph_evidence': {}
        }
        
        # Validate each input against knowledge graph
        for input_item in inputs:
            try:
                individual_result = await self._validate_single_input_knowledge(input_item, context)
                validation_results['individual_validations'][input_item.input_id] = individual_result
                
                logger.debug("Completed individual knowledge validation", extra={
                    'operation': 'INDIVIDUAL_KNOWLEDGE_VALIDATION',
                    'input_id': input_item.input_id,
                    'knowledge_score': individual_result.get('knowledge_score', 0.0)
                })
                
            except Exception as e:
                logger.error("Failed individual knowledge validation", extra={
                    'operation': 'INDIVIDUAL_KNOWLEDGE_VALIDATION_FAILED',
                    'input_id': input_item.input_id,
                    'error': str(e)
                })
                validation_results['individual_validations'][input_item.input_id] = {
                    'knowledge_score': 0.0,
                    'error': str(e)
                }
        
        # Cross-input knowledge consistency analysis
        validation_results.update(await self._analyze_cross_input_knowledge_consistency(inputs, context))
        
        # Calculate overall knowledge validation score
        individual_scores = [
            result.get('knowledge_score', 0.0) 
            for result in validation_results['individual_validations'].values()
        ]
        validation_results['overall_knowledge_score'] = np.mean(individual_scores) if individual_scores else 0.0
        
        logger.info("Completed knowledge graph validation", extra={
            'operation': 'KNOWLEDGE_VALIDATION_COMPLETE',
            'overall_score': validation_results['overall_knowledge_score'],
            'contradiction_count': len(validation_results['knowledge_contradictions'])
        })
        
        return validation_results
    
    async def _validate_single_input_knowledge(self, 
                                             input_item: CrossModalInput,
                                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate a single input against knowledge graph
        
        @param {CrossModalInput} input_item - Input to validate
        @param {Dict[str, Any]} context - Optional validation context
        @returns {Dict[str, Any]} - Single input validation results
        """
        # Extract factual claims from the input
        factual_claims = await self._extract_factual_claims(input_item)
        
        # Create extraction context for knowledge graph queries
        extraction_context = ExtractionContext(
            task_description=f"Validate factual claims from {input_item.modality_type.value} input",
            current_graph_state="",
            available_tools=[],
            user_preferences=context or {},
            optimization_target="accuracy"
        )
        
        validation_result = {
            'input_id': input_item.input_id,
            'factual_claims': factual_claims,
            'supported_facts': [],
            'unsupported_claims': [],
            'contradictions': [],
            'knowledge_score': 0.0,
            'confidence': 0.0,
            'graph_evidence': {}
        }
        
        # Validate each factual claim against knowledge graph
        for claim in factual_claims:
            try:
                claim_validation = await self._validate_factual_claim(claim, extraction_context)
                
                if claim_validation['is_supported']:
                    validation_result['supported_facts'].append({
                        'claim': claim,
                        'evidence': claim_validation['evidence'],
                        'confidence': claim_validation['confidence']
                    })
                elif claim_validation['is_contradiction']:
                    validation_result['contradictions'].append({
                        'claim': claim,
                        'contradictory_evidence': claim_validation['evidence'],
                        'confidence': claim_validation['confidence']
                    })
                else:
                    validation_result['unsupported_claims'].append({
                        'claim': claim,
                        'reason': claim_validation.get('reason', 'No supporting evidence found'),
                        'confidence': claim_validation['confidence']
                    })
                    
            except Exception as e:
                logger.error("Failed to validate factual claim", extra={
                    'operation': 'FACTUAL_CLAIM_VALIDATION_FAILED',
                    'claim': claim[:100],
                    'error': str(e)
                })
                validation_result['unsupported_claims'].append({
                    'claim': claim,
                    'reason': f"Validation error: {str(e)}",
                    'confidence': 0.0
                })
        
        # Calculate knowledge score based on validation results
        validation_result['knowledge_score'] = self._calculate_knowledge_score(validation_result)
        validation_result['confidence'] = self._calculate_knowledge_confidence(validation_result)
        
        return validation_result
    
    async def _extract_factual_claims(self, input_item: CrossModalInput) -> List[str]:
        """
        Extract factual claims from cross-modal input
        
        @param {CrossModalInput} input_item - Input to extract claims from
        @returns {List[str]} - List of factual claims
        """
        try:
            # Get semantic content first
            if input_item.modality_type == ModalityType.TEXT:
                content = str(input_item.content)
            elif input_item.modality_type == ModalityType.IMAGE:
                # For images, get description first
                content = f"Visual content analysis needed for: {input_item.content}"
            else:
                content = str(input_item.content)
            
            # Use LLM to extract factual claims
            extraction_prompt = f"""
            Extract factual claims from this content:
            
            Content ({input_item.modality_type.value}): {content}
            
            Identify specific factual statements that can be verified, such as:
            - Statements about entities and their properties
            - Relationships between entities  
            - Temporal facts (dates, sequences)
            - Quantitative facts (numbers, measurements)
            - Categorical facts (classifications, types)
            
            Return as a JSON list of claim strings:
            ["claim 1", "claim 2", ...]
            
            If no factual claims found, return empty list: []
            """
            
            response = await self.llm_client.acomplete(extraction_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                claims = json.loads(response_text.strip())
                if not isinstance(claims, list):
                    claims = []
            except json.JSONDecodeError:
                logger.warning("Failed to parse factual claims response", extra={
                    'operation': 'FACTUAL_CLAIMS_PARSE_FAILED',
                    'response': response_text[:200]
                })
                claims = []
            
            logger.debug("Extracted factual claims", extra={
                'operation': 'FACTUAL_CLAIMS_EXTRACTION',
                'input_id': input_item.input_id,
                'claim_count': len(claims)
            })
            
            return claims
            
        except Exception as e:
            logger.error("Failed to extract factual claims", extra={
                'operation': 'FACTUAL_CLAIMS_EXTRACTION_FAILED',
                'input_id': input_item.input_id,
                'error': str(e)
            })
            return []
    
    async def _validate_factual_claim(self, 
                                    claim: str, 
                                    extraction_context: ExtractionContext) -> Dict[str, Any]:
        """
        Validate a single factual claim against knowledge graph
        
        @param {str} claim - Factual claim to validate
        @param {ExtractionContext} extraction_context - Context for knowledge extraction
        @returns {Dict[str, Any]} - Claim validation results
        """
        try:
            # Use KGoT knowledge extraction to find relevant information
            query = f"Find evidence for or against this claim: {claim}"
            
            knowledge_result, extraction_metrics = await self.knowledge_extractor.extract_knowledge(
                query, extraction_context
            )
            
            # Analyze the extracted knowledge against the claim
            validation_prompt = f"""
            Validate this claim against the provided knowledge:
            
            Claim: {claim}
            
            Knowledge from graph: {knowledge_result}
            
            Determine:
            1. Is the claim supported by the knowledge? (true/false)
            2. Is the claim contradicted by the knowledge? (true/false)  
            3. Confidence in validation (0.0-1.0)
            4. Brief explanation
            
            Respond in JSON format:
            {{
                "is_supported": boolean,
                "is_contradiction": boolean,
                "confidence": float,
                "explanation": "string"
            }}
            """
            
            response = await self.llm_client.acomplete(validation_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                validation_result = json.loads(response_text.strip())
                validation_result['evidence'] = knowledge_result
                validation_result['extraction_metrics'] = extraction_metrics
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse claim validation response", extra={
                    'operation': 'CLAIM_VALIDATION_PARSE_FAILED',
                    'claim': claim[:100],
                    'response': response_text[:200]
                })
                validation_result = {
                    'is_supported': False,
                    'is_contradiction': False,
                    'confidence': 0.0,
                    'explanation': 'Parsing failed',
                    'evidence': knowledge_result
                }
            
            return validation_result
            
        except Exception as e:
            logger.error("Failed to validate factual claim", extra={
                'operation': 'CLAIM_VALIDATION_FAILED',
                'claim': claim[:100],
                'error': str(e)
            })
            return {
                'is_supported': False,
                'is_contradiction': False,
                'confidence': 0.0,
                'explanation': f'Validation error: {str(e)}',
                'evidence': ''
            }
    
    async def _analyze_cross_input_knowledge_consistency(self, 
                                                       inputs: List[CrossModalInput],
                                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze knowledge consistency across multiple inputs
        
        @param {List[CrossModalInput]} inputs - Inputs to analyze
        @param {Dict[str, Any]} context - Optional analysis context
        @returns {Dict[str, Any]} - Cross-input consistency analysis
        """
        try:
            # Extract all factual claims from all inputs
            all_claims = []
            for input_item in inputs:
                claims = await self._extract_factual_claims(input_item)
                for claim in claims:
                    all_claims.append({
                        'claim': claim,
                        'source_input': input_item.input_id,
                        'modality': input_item.modality_type.value
                    })
            
            # Analyze claims for consistency
            consistency_analysis_prompt = f"""
            Analyze these factual claims from multiple sources for consistency:
            
            Claims: {json.dumps(all_claims, indent=2)}
            
            Identify:
            1. Contradictory claims between sources
            2. Mutually supporting claims
            3. Claims that need external validation
            
            Return analysis in JSON format:
            {{
                "contradictory_pairs": [
                    {{
                        "claim1": "...",
                        "claim2": "...", 
                        "source1": "...",
                        "source2": "...",
                        "explanation": "..."
                    }}
                ],
                "supporting_clusters": [
                    {{
                        "claims": ["...", "..."],
                        "sources": ["...", "..."],
                        "consensus": "..."
                    }}
                ],
                "overall_consistency_score": float
            }}
            """
            
            response = await self.llm_client.acomplete(consistency_analysis_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                analysis_result = json.loads(response_text.strip())
            except json.JSONDecodeError:
                logger.warning("Failed to parse consistency analysis", extra={
                    'operation': 'CONSISTENCY_ANALYSIS_PARSE_FAILED',
                    'response': response_text[:200]
                })
                analysis_result = {
                    'contradictory_pairs': [],
                    'supporting_clusters': [],
                    'overall_consistency_score': 0.5
                }
            
            return {
                'cross_input_consistency': analysis_result,
                'total_claims_analyzed': len(all_claims)
            }
            
        except Exception as e:
            logger.error("Failed cross-input knowledge consistency analysis", extra={
                'operation': 'CROSS_INPUT_CONSISTENCY_FAILED',
                'error': str(e)
            })
            return {
                'cross_input_consistency': {
                    'contradictory_pairs': [],
                    'supporting_clusters': [],
                    'overall_consistency_score': 0.0
                },
                'total_claims_analyzed': 0,
                'error': str(e)
            }
    
    def _calculate_knowledge_score(self, validation_result: Dict[str, Any]) -> float:
        """
        Calculate knowledge validation score for a single input
        
        @param {Dict[str, Any]} validation_result - Validation results to score
        @returns {float} - Knowledge score (0.0-1.0)
        """
        supported_count = len(validation_result['supported_facts'])
        unsupported_count = len(validation_result['unsupported_claims'])
        contradiction_count = len(validation_result['contradictions'])
        total_claims = supported_count + unsupported_count + contradiction_count
        
        if total_claims == 0:
            return 0.5  # Neutral score if no claims
        
        # Calculate weighted score
        # Supported facts contribute positively
        # Contradictions contribute negatively  
        # Unsupported claims are neutral but reduce confidence
        supported_weight = 1.0
        contradiction_weight = -2.0  # Contradictions are worse than just unsupported
        unsupported_weight = 0.0
        
        weighted_score = (
            (supported_count * supported_weight) +
            (contradiction_count * contradiction_weight) +
            (unsupported_count * unsupported_weight)
        ) / total_claims
        
        # Normalize to 0.0-1.0 range
        normalized_score = max(0.0, min(1.0, (weighted_score + 2.0) / 3.0))
        
        return normalized_score
    
    def _calculate_knowledge_confidence(self, validation_result: Dict[str, Any]) -> float:
        """
        Calculate confidence in knowledge validation
        
        @param {Dict[str, Any]} validation_result - Validation results
        @returns {float} - Confidence score (0.0-1.0)
        """
        # Gather confidence scores from individual validations
        confidences = []
        
        for fact in validation_result['supported_facts']:
            confidences.append(fact.get('confidence', 0.5))
        
        for contradiction in validation_result['contradictions']:
            confidences.append(contradiction.get('confidence', 0.5))
        
        for unsupported in validation_result['unsupported_claims']:
            confidences.append(unsupported.get('confidence', 0.5))
        
        # Calculate mean confidence
        if confidences:
            base_confidence = np.mean(confidences)
        else:
            base_confidence = 0.5
        
        # Adjust based on validation coverage
        total_claims = len(validation_result['factual_claims'])
        validated_claims = len(validation_result['supported_facts']) + len(validation_result['contradictions'])
        
        if total_claims > 0:
            coverage_factor = validated_claims / total_claims
        else:
            coverage_factor = 1.0
        
        # Combine base confidence with coverage
        final_confidence = (base_confidence * 0.7) + (coverage_factor * 0.3)
        
        return max(0.0, min(1.0, final_confidence)) 


class ContradictionDetector:
    """
    Implements contradiction detection leveraging KGoT Section 2.5 "Error Management"
    
    This class detects logical, factual, temporal, and semantic contradictions across
    cross-modal inputs using KGoT's error management framework for structured error
    handling and recovery mechanisms.
    """
    
    def __init__(self, 
                 error_management_system: KGoTErrorManagementSystem,
                 llm_client: Any,
                 contradiction_threshold: float = 0.7):
        """
        Initialize contradiction detector with KGoT error management integration
        
        @param {KGoTErrorManagementSystem} error_management_system - KGoT error management system
        @param {Any} llm_client - LLM client for reasoning
        @param {float} contradiction_threshold - Threshold for contradiction detection
        """
        self.error_management = error_management_system
        self.llm_client = llm_client
        self.contradiction_threshold = contradiction_threshold
        self.detected_contradictions = []
        
        logger.info("Initialized Contradiction Detector", extra={
            'operation': 'CONTRADICTION_DETECTOR_INIT',
            'contradiction_threshold': contradiction_threshold,
            'has_error_management': error_management_system is not None
        })
    
    async def detect_comprehensive_contradictions(self, 
                                                inputs: List[CrossModalInput],
                                                consistency_scores: Dict[str, ConsistencyScore],
                                                context: Optional[Dict[str, Any]] = None) -> List[ContradictionReport]:
        """
        Detect comprehensive contradictions across all modalities and consistency scores
        
        @param {List[CrossModalInput]} inputs - Cross-modal inputs to analyze
        @param {Dict[str, ConsistencyScore]} consistency_scores - Consistency scores from previous analysis
        @param {Dict[str, Any]} context - Optional detection context
        @returns {List[ContradictionReport]} - Detailed contradiction reports
        """
        logger.info("Starting comprehensive contradiction detection", extra={
            'operation': 'COMPREHENSIVE_CONTRADICTION_DETECTION_START',
            'input_count': len(inputs),
            'consistency_score_count': len(consistency_scores)
        })
        
        all_contradictions = []
        
        try:
            # Detect different types of contradictions
            logical_contradictions = await self._detect_logical_contradictions(inputs, context)
            factual_contradictions = await self._detect_factual_contradictions(inputs, consistency_scores, context)
            temporal_contradictions = await self._detect_temporal_contradictions(inputs, context)
            semantic_contradictions = await self._detect_semantic_contradictions(inputs, consistency_scores, context)
            
            # Combine all contradiction types
            all_contradictions.extend(logical_contradictions)
            all_contradictions.extend(factual_contradictions)
            all_contradictions.extend(temporal_contradictions)
            all_contradictions.extend(semantic_contradictions)
            
            # Process contradictions through error management system
            processed_contradictions = await self._process_contradictions_through_error_management(all_contradictions)
            
            # Store detected contradictions for analysis
            self.detected_contradictions = processed_contradictions
            
            logger.info("Completed comprehensive contradiction detection", extra={
                'operation': 'COMPREHENSIVE_CONTRADICTION_DETECTION_COMPLETE',
                'total_contradictions': len(processed_contradictions),
                'by_type': {
                    'logical': len(logical_contradictions),
                    'factual': len(factual_contradictions),
                    'temporal': len(temporal_contradictions),
                    'semantic': len(semantic_contradictions)
                }
            })
            
            return processed_contradictions
            
        except Exception as e:
            # Use KGoT error management for handling detection failures
            error_context = ErrorContext(
                error_id=f"contradiction_detection_{int(time.time())}",
                error_type=ErrorType.SYSTEM_ERROR,
                severity=ErrorSeverity.HIGH,
                timestamp=datetime.now(),
                original_operation="comprehensive_contradiction_detection",
                error_message=str(e)
            )
            
            recovery_result, success = await self.error_management.handle_error(
                e, "contradiction_detected", ErrorType.SYSTEM_ERROR, ErrorSeverity.HIGH
            )
            
            if not success:
                logger.error("Failed contradiction detection with error management failure", extra={
                    'operation': 'CONTRADICTION_DETECTION_CRITICAL_FAILURE',
                    'error': str(e)
                })
            
            return []
    
    async def _detect_logical_contradictions(self, 
                                           inputs: List[CrossModalInput],
                                           context: Optional[Dict[str, Any]] = None) -> List[ContradictionReport]:
        """
        Detect logical contradictions using formal logic analysis
        
        @param {List[CrossModalInput]} inputs - Inputs to analyze
        @param {Dict[str, Any]} context - Optional detection context
        @returns {List[ContradictionReport]} - Logical contradiction reports
        """
        logical_contradictions = []
        
        try:
            # Extract logical statements from all inputs
            logical_statements = []
            for input_item in inputs:
                statements = await self._extract_logical_statements(input_item)
                logical_statements.extend([
                    {'statement': stmt, 'source': input_item.input_id, 'modality': input_item.modality_type}
                    for stmt in statements
                ])
            
            # Analyze for logical contradictions
            contradiction_analysis_prompt = f"""
            Analyze these logical statements for formal logical contradictions:
            
            Statements: {json.dumps(logical_statements, indent=2)}
            
            Identify pairs of statements that form logical contradictions such as:
            1. Direct contradictions (A and ¬A)
            2. Conditional contradictions (If A then B, and A and ¬B)
            3. Categorical contradictions (All X are Y, and Some X are ¬Y)
            
            For each contradiction found, provide:
            - Statement 1 and its source
            - Statement 2 and its source  
            - Type of logical contradiction
            - Explanation of why it's contradictory
            - Confidence level (0.0-1.0)
            
            Return as JSON list:
            [{{
                "statement1": "...",
                "source1": "...",
                "statement2": "...",
                "source2": "...",
                "contradiction_type": "...",
                "explanation": "...",
                "confidence": float
            }}]
            """
            
            response = await self.llm_client.acomplete(contradiction_analysis_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                contradictions_data = json.loads(response_text.strip())
                if not isinstance(contradictions_data, list):
                    contradictions_data = []
            except json.JSONDecodeError:
                logger.warning("Failed to parse logical contradiction analysis", extra={
                    'operation': 'LOGICAL_CONTRADICTION_PARSE_FAILED',
                    'response': response_text[:200]
                })
                contradictions_data = []
            
            # Convert to ContradictionReport objects
            for contradiction_data in contradictions_data:
                if contradiction_data.get('confidence', 0.0) >= self.contradiction_threshold:
                    report = ContradictionReport(
                        contradiction_id=f"logical_{uuid.uuid4().hex[:8]}",
                        contradiction_type=ContradictionType.LOGICAL,
                        severity=self._determine_contradiction_severity(contradiction_data.get('confidence', 0.0)),
                        description=contradiction_data.get('explanation', ''),
                        affected_modalities=[
                            next((inp.modality_type for inp in inputs if inp.input_id == contradiction_data.get('source1')), ModalityType.TEXT),
                            next((inp.modality_type for inp in inputs if inp.input_id == contradiction_data.get('source2')), ModalityType.TEXT)
                        ],
                        evidence=[
                            contradiction_data.get('statement1', ''),
                            contradiction_data.get('statement2', '')
                        ],
                        resolution_suggestions=await self._generate_resolution_suggestions(contradiction_data),
                        confidence=contradiction_data.get('confidence', 0.0)
                    )
                    logical_contradictions.append(report)
            
            logger.debug("Detected logical contradictions", extra={
                'operation': 'LOGICAL_CONTRADICTION_DETECTION',
                'contradiction_count': len(logical_contradictions)
            })
            
            return logical_contradictions
            
        except Exception as e:
            logger.error("Failed logical contradiction detection", extra={
                'operation': 'LOGICAL_CONTRADICTION_DETECTION_FAILED',
                'error': str(e)
            })
            return []
    
    async def _detect_factual_contradictions(self, 
                                           inputs: List[CrossModalInput],
                                           consistency_scores: Dict[str, ConsistencyScore],
                                           context: Optional[Dict[str, Any]] = None) -> List[ContradictionReport]:
        """
        Detect factual contradictions using consistency scores and factual analysis
        
        @param {List[CrossModalInput]} inputs - Inputs to analyze
        @param {Dict[str, ConsistencyScore]} consistency_scores - Previous consistency analysis
        @param {Dict[str, Any]} context - Optional detection context
        @returns {List[ContradictionReport]} - Factual contradiction reports
        """
        factual_contradictions = []
        
        try:
            # Analyze consistency scores for factual contradictions
            for pair_key, consistency_score in consistency_scores.items():
                if consistency_score.factual_consistency < self.contradiction_threshold:
                    # Extract more detailed factual contradiction information
                    modality1, modality2 = consistency_score.modality_pair
                    
                    # Find the actual inputs for this pair
                    input1 = next((inp for inp in inputs if inp.modality_type == modality1), None)
                    input2 = next((inp for inp in inputs if inp.modality_type == modality2), None)
                    
                    if input1 and input2:
                        detailed_analysis = await self._analyze_detailed_factual_contradictions(input1, input2)
                        
                        for contradiction in detailed_analysis:
                            report = ContradictionReport(
                                contradiction_id=f"factual_{uuid.uuid4().hex[:8]}",
                                contradiction_type=ContradictionType.FACTUAL,
                                severity=self._determine_contradiction_severity(1.0 - consistency_score.factual_consistency),
                                description=contradiction.get('description', ''),
                                affected_modalities=[modality1, modality2],
                                evidence=contradiction.get('evidence', []),
                                resolution_suggestions=contradiction.get('resolution_suggestions', []),
                                confidence=1.0 - consistency_score.factual_consistency
                            )
                            factual_contradictions.append(report)
            
            logger.debug("Detected factual contradictions", extra={
                'operation': 'FACTUAL_CONTRADICTION_DETECTION',
                'contradiction_count': len(factual_contradictions)
            })
            
            return factual_contradictions
            
        except Exception as e:
            logger.error("Failed factual contradiction detection", extra={
                'operation': 'FACTUAL_CONTRADICTION_DETECTION_FAILED',
                'error': str(e)
            })
            return []
    
    async def _detect_temporal_contradictions(self, 
                                            inputs: List[CrossModalInput],
                                            context: Optional[Dict[str, Any]] = None) -> List[ContradictionReport]:
        """
        Detect temporal contradictions in timelines and sequences
        
        @param {List[CrossModalInput]} inputs - Inputs to analyze  
        @param {Dict[str, Any]} context - Optional detection context
        @returns {List[ContradictionReport]} - Temporal contradiction reports
        """
        temporal_contradictions = []
        
        try:
            # Extract temporal information from all inputs
            temporal_info = []
            for input_item in inputs:
                temp_info = await self._extract_temporal_information(input_item)
                temporal_info.extend([
                    {
                        'info': info,
                        'source': input_item.input_id,
                        'modality': input_item.modality_type.value,
                        'timestamp': input_item.timestamp.isoformat()
                    }
                    for info in temp_info
                ])
            
            # Analyze for temporal contradictions
            temporal_analysis_prompt = f"""
            Analyze this temporal information for contradictions:
            
            Temporal Information: {json.dumps(temporal_info, indent=2)}
            
            Identify temporal contradictions such as:
            1. Impossible sequences (effect before cause)
            2. Conflicting dates or times
            3. Contradictory durations or periods
            4. Inconsistent temporal relationships
            
            Return analysis as JSON list of contradictions.
            """
            
            response = await self.llm_client.acomplete(temporal_analysis_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                contradictions_data = json.loads(response_text.strip())
                if not isinstance(contradictions_data, list):
                    contradictions_data = []
            except json.JSONDecodeError:
                contradictions_data = []
            
            # Convert to ContradictionReport objects
            for contradiction_data in contradictions_data:
                if contradiction_data.get('confidence', 0.0) >= self.contradiction_threshold:
                    report = ContradictionReport(
                        contradiction_id=f"temporal_{uuid.uuid4().hex[:8]}",
                        contradiction_type=ContradictionType.TEMPORAL,
                        severity=self._determine_contradiction_severity(contradiction_data.get('confidence', 0.0)),
                        description=contradiction_data.get('description', ''),
                        affected_modalities=[
                            next((inp.modality_type for inp in inputs if inp.input_id == contradiction_data.get('source1')), ModalityType.TEXT),
                            next((inp.modality_type for inp in inputs if inp.input_id == contradiction_data.get('source2')), ModalityType.TEXT)
                        ],
                        evidence=contradiction_data.get('evidence', []),
                        resolution_suggestions=await self._generate_resolution_suggestions(contradiction_data),
                        confidence=contradiction_data.get('confidence', 0.0)
                    )
                    temporal_contradictions.append(report)
            
            return temporal_contradictions
            
        except Exception as e:
            logger.error("Failed temporal contradiction detection", extra={
                'operation': 'TEMPORAL_CONTRADICTION_DETECTION_FAILED',
                'error': str(e)
            })
            return []
    
    async def _detect_semantic_contradictions(self, 
                                            inputs: List[CrossModalInput],
                                            consistency_scores: Dict[str, ConsistencyScore],
                                            context: Optional[Dict[str, Any]] = None) -> List[ContradictionReport]:
        """
        Detect semantic contradictions in meaning and interpretation
        
        @param {List[CrossModalInput]} inputs - Inputs to analyze
        @param {Dict[str, ConsistencyScore]} consistency_scores - Previous consistency analysis
        @param {Dict[str, Any]} context - Optional detection context
        @returns {List[ContradictionReport]} - Semantic contradiction reports
        """
        semantic_contradictions = []
        
        try:
            # Use consistency scores to identify semantic contradictions
            for pair_key, consistency_score in consistency_scores.items():
                if consistency_score.semantic_consistency < self.contradiction_threshold:
                    modality1, modality2 = consistency_score.modality_pair
                    
                    # Find contradictions already identified in consistency scores
                    existing_contradictions = consistency_score.contradictions or []
                    
                    for contradiction in existing_contradictions:
                        if contradiction.get('type') == 'semantic':
                            report = ContradictionReport(
                                contradiction_id=f"semantic_{uuid.uuid4().hex[:8]}",
                                contradiction_type=ContradictionType.SEMANTIC,
                                severity=self._determine_contradiction_severity(1.0 - consistency_score.semantic_consistency),
                                description=contradiction.get('description', ''),
                                affected_modalities=[modality1, modality2],
                                evidence=contradiction.get('evidence', [contradiction.get('description', '')]),
                                resolution_suggestions=await self._generate_semantic_resolution_suggestions(contradiction),
                                confidence=1.0 - consistency_score.semantic_consistency
                            )
                            semantic_contradictions.append(report)
            
            return semantic_contradictions
            
        except Exception as e:
            logger.error("Failed semantic contradiction detection", extra={
                'operation': 'SEMANTIC_CONTRADICTION_DETECTION_FAILED',
                'error': str(e)
            })
            return []
    
    async def _process_contradictions_through_error_management(self, 
                                                             contradictions: List[ContradictionReport]) -> List[ContradictionReport]:
        """
        Process detected contradictions through KGoT error management system
        
        @param {List[ContradictionReport]} contradictions - Raw contradiction reports
        @returns {List[ContradictionReport]} - Processed contradiction reports
        """
        processed_contradictions = []
        
        for contradiction in contradictions:
            try:
                # Create error context for each contradiction
                error_context = ErrorContext(
                    error_id=contradiction.contradiction_id,
                    error_type=ErrorType.VALIDATION_ERROR,
                    severity=contradiction.severity,
                    timestamp=contradiction.timestamp,
                    original_operation="cross_modal_validation",
                    error_message=contradiction.description,
                    metadata={
                        'contradiction_type': contradiction.contradiction_type.value,
                        'affected_modalities': [mod.value for mod in contradiction.affected_modalities],
                        'confidence': contradiction.confidence
                    }
                )
                
                # Process through error management system
                recovery_result, success = await self.error_management.handle_error(
                    Exception(contradiction.description),
                    "contradiction_detected",
                    ErrorType.VALIDATION_ERROR,
                    contradiction.severity
                )
                
                # Update contradiction with error management insights
                contradiction.resolution_suggestions.extend(
                    await self._extract_error_management_suggestions(recovery_result, success)
                )
                
                processed_contradictions.append(contradiction)
                
            except Exception as e:
                logger.error("Failed to process contradiction through error management", extra={
                    'operation': 'CONTRADICTION_ERROR_MANAGEMENT_FAILED',
                    'contradiction_id': contradiction.contradiction_id,
                    'error': str(e)
                })
                # Still include the contradiction even if error management processing fails
                processed_contradictions.append(contradiction)
        
        return processed_contradictions
    
    async def _extract_logical_statements(self, input_item: CrossModalInput) -> List[str]:
        """
        Extract logical statements from input content
        
        @param {CrossModalInput} input_item - Input to extract statements from
        @returns {List[str]} - List of logical statements
        """
        try:
            content = str(input_item.content)
            
            extraction_prompt = f"""
            Extract logical statements from this content that can be analyzed for formal logic:
            
            Content ({input_item.modality_type.value}): {content}
            
            Identify statements that:
            - Make definite claims (All X are Y, No X are Y, Some X are Y)
            - Express conditionals (If A then B)
            - State facts that can be true or false
            - Express relationships between entities
            
            Return as JSON list of statement strings:
            ["statement 1", "statement 2", ...]
            """
            
            response = await self.llm_client.acomplete(extraction_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                statements = json.loads(response_text.strip())
                if not isinstance(statements, list):
                    statements = []
            except json.JSONDecodeError:
                statements = []
            
            return statements
            
        except Exception as e:
            logger.error("Failed to extract logical statements", extra={
                'operation': 'LOGICAL_STATEMENT_EXTRACTION_FAILED',
                'input_id': input_item.input_id,
                'error': str(e)
            })
            return []
    
    def _determine_contradiction_severity(self, confidence: float) -> ErrorSeverity:
        """
        Determine error severity based on contradiction confidence
        
        @param {float} confidence - Confidence in contradiction (0.0-1.0)
        @returns {ErrorSeverity} - Appropriate error severity level
        """
        if confidence >= 0.9:
            return ErrorSeverity.CRITICAL
        elif confidence >= 0.7:
            return ErrorSeverity.HIGH
        elif confidence >= 0.5:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    async def _generate_resolution_suggestions(self, contradiction_data: Dict[str, Any]) -> List[str]:
        """
        Generate resolution suggestions for detected contradictions
        
        @param {Dict[str, Any]} contradiction_data - Contradiction analysis data
        @returns {List[str]} - List of resolution suggestions
        """
        try:
            suggestion_prompt = f"""
            Generate resolution suggestions for this contradiction:
            
            Contradiction: {json.dumps(contradiction_data, indent=2)}
            
            Provide practical suggestions for resolving this contradiction such as:
            - Additional verification needed
            - Source prioritization guidance
            - Context clarification requirements
            - Data correction recommendations
            
            Return as JSON list of suggestion strings:
            ["suggestion 1", "suggestion 2", ...]
            """
            
            response = await self.llm_client.acomplete(suggestion_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                suggestions = json.loads(response_text.strip())
                if not isinstance(suggestions, list):
                    suggestions = [response_text.strip()]
            except json.JSONDecodeError:
                suggestions = [response_text.strip()]
            
            return suggestions
            
        except Exception as e:
            logger.error("Failed to generate resolution suggestions", extra={
                'operation': 'RESOLUTION_SUGGESTION_FAILED',
                'error': str(e)
            })
            return ["Unable to generate resolution suggestions due to processing error"]

    async def _extract_error_management_suggestions(self, recovery_result: Dict[str, Any], success: bool) -> List[str]:
        """
        Extract error management suggestions from recovery result
        
        @param {Dict[str, Any]} recovery_result - Result from error management
        @param {bool} success - Indicates if error was handled successfully
        @returns {List[str]} - List of error management suggestions
        """
        try:
            if success:
                return recovery_result.get('suggestions', [])
            else:
                return ["No recovery suggestions available due to error handling failure"]
        except Exception as e:
            logger.error("Failed to extract error management suggestions", extra={
                'operation': 'ERROR_MANAGEMENT_SUGGESTION_EXTRACTION_FAILED',
                'error': str(e)
            })
            return ["Unable to extract error management suggestions due to processing error"]

    async def _analyze_detailed_factual_contradictions(self, input1: CrossModalInput, input2: CrossModalInput) -> List[Dict[str, Any]]:
        """
        Analyze detailed factual contradictions between two inputs
        
        @param {CrossModalInput} input1 - First input for comparison
        @param {CrossModalInput} input2 - Second input for comparison
        @returns {List[Dict[str, Any]]} - List of detailed contradiction analysis
        """
        try:
            # Extract detailed factual contradictions
            contradiction_analysis_prompt = f"""
            Analyze these two factual contents for contradictions:
            
            Content 1 ({input1.modality_type.value}): {input1.content}
            Content 2 ({input2.modality_type.value}): {input2.content}
            
            Identify contradictions including:
            1. Conflicting facts or claims
            2. Inconsistent numbers, dates, or measurements
            3. Conflicting statements about entities or events
            
            Return analysis as JSON list of contradictions.
            """
            
            response = await self.llm_client.acomplete(contradiction_analysis_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                contradictions_data = json.loads(response_text.strip())
                if not isinstance(contradictions_data, list):
                    contradictions_data = []
            except json.JSONDecodeError:
                contradictions_data = []
            
            return contradictions_data
            
        except Exception as e:
            logger.error("Failed to analyze detailed factual contradictions", extra={
                'operation': 'FACTUAL_CONTRADICTION_ANALYSIS_FAILED',
                'error': str(e)
            })
            return []

    async def _extract_temporal_information(self, input_item: CrossModalInput) -> List[str]:
        """
        Extract temporal information from input content
        
        @param {CrossModalInput} input_item - Input to extract temporal information from
        @returns {List[str]} - List of temporal information
        """
        try:
            content = str(input_item.content)
            
            extraction_prompt = f"""
            Extract temporal information from this content:
            
            Content ({input_item.modality_type.value}): {content}
            
            Identify temporal facts including:
            - Dates and sequences
            - Temporal relationships
            - Duration or period
            
            Return as JSON list of information strings:
            ["information 1", "information 2", ...]
            """
            
            response = await self.llm_client.acomplete(extraction_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                information = json.loads(response_text.strip())
                if not isinstance(information, list):
                    information = []
            except json.JSONDecodeError:
                information = []
            
            return information
            
        except Exception as e:
            logger.error("Failed to extract temporal information", extra={
                'operation': 'TEMPORAL_INFORMATION_EXTRACTION_FAILED',
                'input_id': input_item.input_id,
                'error': str(e)
            })
            return []

    async def _generate_semantic_resolution_suggestions(self, contradiction: Dict[str, Any]) -> List[str]:
        """
        Generate semantic resolution suggestions for a contradiction
        
        @param {Dict[str, Any]} contradiction - Contradiction analysis data
        @returns {List[str]} - List of semantic resolution suggestions
        """
        try:
            suggestion_prompt = f"""
            Generate semantic resolution suggestions for this contradiction:
            
            Contradiction: {json.dumps(contradiction, indent=2)}
            
            Provide practical suggestions for resolving this contradiction such as:
            - Additional verification needed
            - Source prioritization guidance
            - Context clarification requirements
            - Data correction recommendations
            
            Return as JSON list of suggestion strings:
            ["suggestion 1", "suggestion 2", ...]
            """
            
            response = await self.llm_client.acomplete(suggestion_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            try:
                suggestions = json.loads(response_text.strip())
                if not isinstance(suggestions, list):
                    suggestions = [response_text.strip()]
            except json.JSONDecodeError:
                suggestions = [response_text.strip()]
            
            return suggestions
            
        except Exception as e:
            logger.error("Failed to generate semantic resolution suggestions", extra={
                'operation': 'SEMANTIC_RESOLUTION_SUGGESTION_FAILED',
                'error': str(e)
            })
            return ["Unable to generate semantic resolution suggestions due to processing error"] 


class ConfidenceScorer:
    """
    Implements confidence scoring using KGoT analytical capabilities
    
    This class calculates comprehensive confidence scores for cross-modal validation
    based on individual modality confidence, cross-modal agreement, knowledge support,
    and statistical analysis using KGoT's analytical framework.
    """
    
    def __init__(self, 
                 statistical_analyzer: StatisticalSignificanceAnalyzer,
                 llm_client: Any,
                 confidence_weights: Optional[Dict[str, float]] = None):
        """
        Initialize confidence scorer with KGoT analytical capabilities
        
        @param {StatisticalSignificanceAnalyzer} statistical_analyzer - Statistical analysis component
        @param {Any} llm_client - LLM client for analytical reasoning
        @param {Dict[str, float]} confidence_weights - Weights for different confidence factors
        """
        self.statistical_analyzer = statistical_analyzer
        self.llm_client = llm_client
        self.confidence_weights = confidence_weights or {
            'individual_modality': 0.3,
            'cross_modal_agreement': 0.25,
            'knowledge_support': 0.25,
            'statistical_significance': 0.2
        }
        
        logger.info("Initialized Confidence Scorer", extra={
            'operation': 'CONFIDENCE_SCORER_INIT',
            'confidence_weights': self.confidence_weights
        })
    
    async def calculate_comprehensive_confidence(self, 
                                               inputs: List[CrossModalInput],
                                               consistency_scores: Dict[str, ConsistencyScore],
                                               knowledge_validation: Dict[str, Any],
                                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive confidence scores using multiple analytical approaches
        
        @param {List[CrossModalInput]} inputs - Cross-modal inputs
        @param {Dict[str, ConsistencyScore]} consistency_scores - Consistency analysis results
        @param {Dict[str, Any]} knowledge_validation - Knowledge validation results
        @param {Dict[str, Any]} context - Optional scoring context
        @returns {Dict[str, Any]} - Comprehensive confidence analysis
        """
        logger.info("Starting comprehensive confidence calculation", extra={
            'operation': 'COMPREHENSIVE_CONFIDENCE_START',
            'input_count': len(inputs),
            'consistency_score_count': len(consistency_scores)
        })
        
        try:
            # Calculate individual confidence components
            individual_confidence = await self._calculate_individual_modality_confidence(inputs)
            cross_modal_agreement = await self._calculate_cross_modal_agreement_score(consistency_scores)
            knowledge_confidence = await self._calculate_knowledge_supported_confidence(knowledge_validation)
            statistical_confidence = await self._calculate_statistical_significance_confidence(consistency_scores)
            
            # Calculate weighted overall confidence
            overall_confidence = (
                individual_confidence * self.confidence_weights['individual_modality'] +
                cross_modal_agreement * self.confidence_weights['cross_modal_agreement'] +
                knowledge_confidence * self.confidence_weights['knowledge_support'] +
                statistical_confidence * self.confidence_weights['statistical_significance']
            )
            
            # Generate confidence intervals using statistical analysis
            confidence_interval = await self._calculate_confidence_interval(consistency_scores)
            
            # Analyze confidence reliability
            confidence_reliability = await self._analyze_confidence_reliability(
                individual_confidence, cross_modal_agreement, knowledge_confidence, statistical_confidence
            )
            
            confidence_analysis = {
                'overall_confidence': overall_confidence,
                'individual_modality_confidence': individual_confidence,
                'cross_modal_agreement_score': cross_modal_agreement,
                'knowledge_supported_confidence': knowledge_confidence,
                'statistical_significance_confidence': statistical_confidence,
                'confidence_interval': confidence_interval,
                'confidence_reliability': confidence_reliability,
                'confidence_weights_used': self.confidence_weights,
                'confidence_breakdown': {
                    'high_confidence_threshold': 0.8,
                    'medium_confidence_threshold': 0.6,
                    'low_confidence_threshold': 0.4,
                    'confidence_level': self._determine_confidence_level(overall_confidence)
                }
            }
            
            logger.info("Completed comprehensive confidence calculation", extra={
                'operation': 'COMPREHENSIVE_CONFIDENCE_COMPLETE',
                'overall_confidence': overall_confidence,
                'confidence_level': confidence_analysis['confidence_breakdown']['confidence_level']
            })
            
            return confidence_analysis
            
        except Exception as e:
            logger.error("Failed comprehensive confidence calculation", extra={
                'operation': 'COMPREHENSIVE_CONFIDENCE_FAILED',
                'error': str(e)
            })
            return {
                'overall_confidence': 0.0,
                'error': str(e),
                'confidence_level': 'unknown'
            }
    
    async def _calculate_individual_modality_confidence(self, inputs: List[CrossModalInput]) -> float:
        """
        Calculate confidence based on individual modality confidence scores
        
        @param {List[CrossModalInput]} inputs - Cross-modal inputs
        @returns {float} - Individual modality confidence score
        """
        try:
            confidences = [input_item.confidence or 0.5 for input_item in inputs]
            
            if not confidences:
                return 0.5
            
            # Calculate weighted average based on modality reliability
            modality_weights = await self._get_modality_reliability_weights(inputs)
            
            weighted_confidence = sum(
                conf * modality_weights.get(inp.modality_type, 1.0)
                for conf, inp in zip(confidences, inputs)
            ) / sum(modality_weights.get(inp.modality_type, 1.0) for inp in inputs)
            
            return max(0.0, min(1.0, weighted_confidence))
            
        except Exception as e:
            logger.error("Failed individual modality confidence calculation", extra={
                'operation': 'INDIVIDUAL_MODALITY_CONFIDENCE_FAILED',
                'error': str(e)
            })
            return 0.5
    
    async def _calculate_cross_modal_agreement_score(self, consistency_scores: Dict[str, ConsistencyScore]) -> float:
        """
        Calculate confidence based on cross-modal agreement
        
        @param {Dict[str, ConsistencyScore]} consistency_scores - Consistency analysis results
        @returns {float} - Cross-modal agreement confidence score
        """
        try:
            if not consistency_scores:
                return 0.5
            
            # Calculate agreement based on consistency scores
            overall_consistencies = [score.overall_consistency for score in consistency_scores.values()]
            agreement_score = np.mean(overall_consistencies)
            
            # Adjust for confidence in individual consistency scores
            confidence_adjustments = [score.confidence for score in consistency_scores.values()]
            confidence_factor = np.mean(confidence_adjustments) if confidence_adjustments else 1.0
            
            adjusted_agreement = agreement_score * confidence_factor
            
            return max(0.0, min(1.0, adjusted_agreement))
            
        except Exception as e:
            logger.error("Failed cross-modal agreement calculation", extra={
                'operation': 'CROSS_MODAL_AGREEMENT_FAILED',
                'error': str(e)
            })
            return 0.5
    
    async def _calculate_knowledge_supported_confidence(self, knowledge_validation: Dict[str, Any]) -> float:
        """
        Calculate confidence based on knowledge graph support
        
        @param {Dict[str, Any]} knowledge_validation - Knowledge validation results
        @returns {float} - Knowledge-supported confidence score
        """
        try:
            overall_knowledge_score = knowledge_validation.get('overall_knowledge_score', 0.0)
            
            # Adjust based on knowledge validation completeness
            individual_validations = knowledge_validation.get('individual_validations', {})
            validation_coverage = len([v for v in individual_validations.values() if v.get('knowledge_score', 0) > 0]) / max(len(individual_validations), 1)
            
            # Factor in cross-input consistency
            cross_input_consistency = knowledge_validation.get('cross_input_consistency', {}).get('overall_consistency_score', 0.5)
            
            # Combine factors
            knowledge_confidence = (
                overall_knowledge_score * 0.5 +
                validation_coverage * 0.3 +
                cross_input_consistency * 0.2
            )
            
            return max(0.0, min(1.0, knowledge_confidence))
            
        except Exception as e:
            logger.error("Failed knowledge-supported confidence calculation", extra={
                'operation': 'KNOWLEDGE_SUPPORTED_CONFIDENCE_FAILED',
                'error': str(e)
            })
            return 0.5
    
    async def _calculate_statistical_significance_confidence(self, consistency_scores: Dict[str, ConsistencyScore]) -> float:
        """
        Calculate confidence based on statistical significance using KGoT analytical capabilities
        
        @param {Dict[str, ConsistencyScore]} consistency_scores - Consistency analysis results
        @returns {float} - Statistical significance confidence score
        """
        try:
            if not consistency_scores:
                return 0.5
            
            # Create mock validation metrics for statistical analysis
            mock_metrics = []
            for score in consistency_scores.values():
                mock_metric = ValidationMetrics(
                    consistency_score=score.overall_consistency,
                    error_rate=1.0 - score.overall_consistency,
                    failure_recovery_rate=score.confidence,
                    uptime_percentage=score.confidence * 100,
                    output_consistency=score.semantic_consistency,
                    logic_consistency=score.factual_consistency,
                    temporal_consistency=score.temporal_consistency,
                    cross_model_agreement=score.confidence,
                    execution_time_avg=1.0,
                    execution_time_std=0.1,
                    throughput_ops_per_sec=1.0,
                    resource_efficiency=score.confidence,
                    scalability_score=score.confidence,
                    ground_truth_accuracy=score.overall_consistency,
                    expert_validation_score=score.confidence,
                    benchmark_performance=score.overall_consistency,
                    cross_validation_accuracy=score.overall_consistency,
                    confidence_interval=(score.overall_consistency - 0.1, score.overall_consistency + 0.1),
                    p_value=0.05,
                    statistical_significance=score.overall_consistency > 0.6,
                    sample_size=1,
                    validation_duration=1.0
                )
                mock_metrics.append(mock_metric)
            
            # Perform statistical significance analysis
            statistical_analysis = await self.statistical_analyzer.analyze_validation_significance(mock_metrics)
            
            # Extract confidence from statistical analysis
            significance_confidence = 1.0 if statistical_analysis.get('overall_significance', False) else 0.5
            
            # Adjust based on p-values and confidence intervals
            p_value_confidence = 1.0 - statistical_analysis.get('average_p_value', 0.5)
            
            statistical_confidence = (significance_confidence * 0.6) + (p_value_confidence * 0.4)
            
            return max(0.0, min(1.0, statistical_confidence))
            
        except Exception as e:
            logger.error("Failed statistical significance confidence calculation", extra={
                'operation': 'STATISTICAL_SIGNIFICANCE_CONFIDENCE_FAILED',
                'error': str(e)
            })
            return 0.5
    
    def _determine_confidence_level(self, confidence_score: float) -> str:
        """
        Determine qualitative confidence level from numerical score
        
        @param {float} confidence_score - Numerical confidence score (0.0-1.0)
        @returns {str} - Qualitative confidence level
        """
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        elif confidence_score >= 0.4:
            return "low"
        else:
            return "very_low"


class KGoTAlitaCrossModalValidator:
    """
    Main orchestrator for KGoT-Alita Cross-Modal Validation
    
    This class integrates all validation components to provide comprehensive
    cross-modal validation with consistency checking, contradiction detection,
    confidence scoring, and quality assurance using both KGoT and Alita frameworks.
    """
    
    def __init__(self, 
                 llm_client: Any,
                 knowledge_extraction_manager: Optional[KnowledgeExtractionManager] = None,
                 graph_store: Optional[KnowledgeGraphInterface] = None,
                 error_management_system: Optional[KGoTErrorManagementSystem] = None,
                 visual_analyzer: Optional[KGoTVisualAnalyzer] = None,
                 config: Optional[Dict[str, Any]] = None,
                 llm_clients: Optional[Dict[str, Any]] = None):
        """
        Initialize the comprehensive cross-modal validator
        Uses @modelsrule.mdc specialized models for different tasks
        
        @param {Any} llm_client - Default LLM client for reasoning
        @param {KnowledgeExtractionManager} knowledge_extraction_manager - KGoT knowledge extraction
        @param {KnowledgeGraphInterface} graph_store - Knowledge graph store
        @param {KGoTErrorManagementSystem} error_management_system - KGoT error management
        @param {KGoTVisualAnalyzer} visual_analyzer - Visual analysis component
        @param {Dict[str, Any]} config - Configuration options
        @param {Dict[str, Any]} llm_clients - Specialized LLM clients by task type
        """
        self.llm_client = llm_client
        self.llm_clients = llm_clients or {'default': llm_client}
        self.config = config or {}
        
        # Get specialized clients based on @modelsrule.mdc
        vision_client = self.llm_clients.get('vision', llm_client)  # o3(vision)
        webagent_client = self.llm_clients.get('webagent', llm_client)  # claude-4-sonnet
        orchestration_client = self.llm_clients.get('orchestration', llm_client)  # gemini-2.5-pro
        
        # Initialize core components with appropriate models
        self.consistency_checker = ModalityConsistencyChecker(
            webagent_client, visual_analyzer, vision_client=vision_client
        )
        
        if knowledge_extraction_manager and graph_store:
            self.knowledge_validator = KGoTKnowledgeValidator(
                knowledge_extraction_manager, graph_store, webagent_client
            )
        else:
            self.knowledge_validator = None
        
        if error_management_system:
            self.contradiction_detector = ContradictionDetector(error_management_system, webagent_client)
        else:
            self.contradiction_detector = None
        
        self.confidence_scorer = ConfidenceScorer(StatisticalSignificanceAnalyzer(), orchestration_client)
        
        logger.info("Initialized KGoT-Alita Cross-Modal Validator with @modelsrule.mdc", extra={
            'operation': 'CROSS_MODAL_VALIDATOR_INIT',
            'has_knowledge_validator': self.knowledge_validator is not None,
            'has_contradiction_detector': self.contradiction_detector is not None,
            'model_assignments': {
                'consistency_checking': 'claude-4-sonnet(webagent)',
                'knowledge_validation': 'claude-4-sonnet(webagent)', 
                'contradiction_detection': 'claude-4-sonnet(webagent)',
                'confidence_scoring': 'gemini-2.5-pro(orchestration)',
                'vision_processing': 'o3(vision)'
            },
            'config': self.config
        })
    
    async def validate_cross_modal_input(self, 
                                       validation_spec: CrossModalValidationSpec) -> ValidationResult:
        """
        Main entry point for comprehensive cross-modal validation
        Uses @modelsrule.mdc specialized models for optimal performance:
        - o3(vision) for image/visual processing
        - claude-4-sonnet(webagent) for reasoning and analysis
        - gemini-2.5-pro(orchestration) for coordination and confidence scoring
        
        @param {CrossModalValidationSpec} validation_spec - Specification for validation
        @returns {ValidationResult} - Complete validation results
        """
        logger.info("Starting comprehensive cross-modal validation", extra={
            'operation': 'CROSS_MODAL_VALIDATION_START',
            'validation_id': validation_spec.validation_id,
            'input_count': len(validation_spec.inputs),
            'validation_level': validation_spec.validation_level.value
        })
        
        start_time = time.time()
        processing_logs = []
        
        try:
            # Step 1: Consistency checking using KGoT Section 2.1 knowledge validation
            processing_logs.append({
                'step': 'consistency_checking',
                'timestamp': datetime.now().isoformat(),
                'status': 'started'
            })
            
            consistency_scores = await self.consistency_checker.check_cross_modal_consistency(
                validation_spec.inputs
            )
            
            processing_logs.append({
                'step': 'consistency_checking',
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'result_count': len(consistency_scores)
            })
            
            # Step 2: Knowledge validation (if available)
            knowledge_validation = {}
            if self.knowledge_validator:
                processing_logs.append({
                    'step': 'knowledge_validation',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'started'
                })
                
                knowledge_validation = await self.knowledge_validator.validate_against_knowledge_graph(
                    validation_spec.inputs,
                    validation_spec.knowledge_context
                )
                
                processing_logs.append({
                    'step': 'knowledge_validation',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed',
                    'overall_score': knowledge_validation.get('overall_knowledge_score', 0.0)
                })
            
            # Step 3: Contradiction detection leveraging KGoT Section 2.5 "Error Management"
            contradictions = []
            if self.contradiction_detector:
                processing_logs.append({
                    'step': 'contradiction_detection',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'started'
                })
                
                contradictions = await self.contradiction_detector.detect_comprehensive_contradictions(
                    validation_spec.inputs,
                    consistency_scores,
                    validation_spec.knowledge_context
                )
                
                processing_logs.append({
                    'step': 'contradiction_detection',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed',
                    'contradiction_count': len(contradictions)
                })
            
            # Step 4: Confidence scoring using KGoT analytical capabilities
            processing_logs.append({
                'step': 'confidence_scoring',
                'timestamp': datetime.now().isoformat(),
                'status': 'started'
            })
            
            confidence_analysis = await self.confidence_scorer.calculate_comprehensive_confidence(
                validation_spec.inputs,
                consistency_scores,
                knowledge_validation,
                validation_spec.knowledge_context
            )
            
            processing_logs.append({
                'step': 'confidence_scoring',
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'overall_confidence': confidence_analysis.get('overall_confidence', 0.0)
            })
            
            # Step 5: Generate comprehensive metrics
            metrics = await self._generate_comprehensive_metrics(
                validation_spec,
                consistency_scores,
                knowledge_validation,
                contradictions,
                confidence_analysis,
                time.time() - start_time
            )
            
            # Step 6: Generate recommendations and determine validity
            recommendations = await self._generate_comprehensive_recommendations(
                validation_spec, consistency_scores, knowledge_validation, contradictions, confidence_analysis
            )
            
            is_valid = await self._determine_overall_validity(metrics, contradictions)
            overall_score = await self._calculate_overall_score(metrics, confidence_analysis)
            
            # Create detailed analysis
            detailed_analysis = {
                'consistency_analysis': consistency_scores,
                'knowledge_validation': knowledge_validation,
                'contradiction_analysis': contradictions,
                'confidence_analysis': confidence_analysis,
                'processing_duration': time.time() - start_time,
                'validation_level': validation_spec.validation_level.value
            }
            
            # Create validation result
            result = ValidationResult(
                validation_id=validation_spec.validation_id,
                spec=validation_spec,
                metrics=metrics,
                contradictions=contradictions,
                recommendations=recommendations,
                is_valid=is_valid,
                overall_score=overall_score,
                detailed_analysis=detailed_analysis,
                processing_logs=processing_logs
            )
            
            logger.info("Completed comprehensive cross-modal validation", extra={
                'operation': 'CROSS_MODAL_VALIDATION_COMPLETE',
                'validation_id': validation_spec.validation_id,
                'is_valid': is_valid,
                'overall_score': overall_score,
                'processing_duration': time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            logger.error("Failed comprehensive cross-modal validation", extra={
                'operation': 'CROSS_MODAL_VALIDATION_FAILED',
                'validation_id': validation_spec.validation_id,
                'error': str(e)
            })
            
            # Return failed validation result
            return ValidationResult(
                validation_id=validation_spec.validation_id,
                spec=validation_spec,
                metrics=CrossModalMetrics(
                    modality_consistency_scores={},
                    knowledge_validation_score=0.0,
                    contradiction_count=0,
                    contradiction_severity_distribution={},
                    individual_modality_confidence={},
                    cross_modal_agreement_score=0.0,
                    knowledge_supported_confidence=0.0,
                    overall_confidence=0.0,
                    kgot_quality_score=0.0,
                    alita_quality_score=0.0,
                    unified_quality_score=0.0,
                    validation_duration=time.time() - start_time,
                    processing_time_per_modality={},
                    resource_usage={},
                    statistical_significance=False,
                    confidence_interval=(0.0, 0.0),
                    p_value=1.0
                ),
                contradictions=[],
                recommendations=[f"Validation failed due to error: {str(e)}"],
                is_valid=False,
                overall_score=0.0,
                detailed_analysis={'error': str(e)},
                processing_logs=processing_logs
            )
    
    async def _generate_comprehensive_metrics(self,
                                            validation_spec: CrossModalValidationSpec,
                                            consistency_scores: Dict[str, ConsistencyScore],
                                            knowledge_validation: Dict[str, Any],
                                            contradictions: List[ContradictionReport],
                                            confidence_analysis: Dict[str, Any],
                                            processing_duration: float) -> CrossModalMetrics:
        """
        Generate comprehensive metrics from all validation results
        
        @param {CrossModalValidationSpec} validation_spec - Original validation specification
        @param {Dict[str, ConsistencyScore]} consistency_scores - Cross-modal consistency results
        @param {Dict[str, Any]} knowledge_validation - Knowledge graph validation results
        @param {List[ContradictionReport]} contradictions - Detected contradictions
        @param {Dict[str, Any]} confidence_analysis - Confidence scoring results
        @param {float} processing_duration - Total processing time
        @returns {CrossModalMetrics} - Comprehensive validation metrics
        """
        try:
            # Calculate individual modality confidence scores
            individual_modality_confidence = {}
            processing_time_per_modality = {}
            
            for input_item in validation_spec.inputs:
                individual_modality_confidence[input_item.modality_type] = input_item.confidence or 0.5
                processing_time_per_modality[input_item.modality_type] = processing_duration / len(validation_spec.inputs)
            
            # Calculate contradiction severity distribution
            contradiction_severity_distribution = Counter([
                c.severity.value for c in contradictions
            ])
            
            # Extract confidence metrics from confidence analysis
            cross_modal_agreement_score = confidence_analysis.get('cross_modal_agreement_score', 0.0)
            knowledge_supported_confidence = confidence_analysis.get('knowledge_supported_confidence', 0.0)
            overall_confidence = confidence_analysis.get('overall_confidence', 0.0)
            
            # Calculate quality scores using both KGoT and Alita frameworks
            kgot_quality_score = self._calculate_kgot_quality_score(
                knowledge_validation, consistency_scores, contradictions
            )
            alita_quality_score = self._calculate_alita_quality_score(
                consistency_scores, contradictions, confidence_analysis
            )
            unified_quality_score = (kgot_quality_score + alita_quality_score) / 2
            
            # Calculate statistical significance
            statistical_significance = confidence_analysis.get('statistical_significance', False)
            confidence_interval = confidence_analysis.get('confidence_interval', (0.0, 1.0))
            p_value = confidence_analysis.get('p_value', 1.0)
            
            # Calculate base validation metrics
            reliability_score = self._calculate_reliability_score(consistency_scores, contradictions)
            accuracy_score = knowledge_validation.get('overall_knowledge_score', 0.0)
            consistency_score = np.mean([cs.overall_consistency for cs in consistency_scores.values()]) if consistency_scores else 0.0
            performance_score = self._calculate_performance_score(processing_duration, len(validation_spec.inputs))
            
            # Create comprehensive metrics
            metrics = CrossModalMetrics(
                modality_consistency_scores=consistency_scores,
                knowledge_validation_score=knowledge_validation.get('overall_knowledge_score', 0.0),
                contradiction_count=len(contradictions),
                contradiction_severity_distribution=dict(contradiction_severity_distribution),
                individual_modality_confidence=individual_modality_confidence,
                cross_modal_agreement_score=cross_modal_agreement_score,
                knowledge_supported_confidence=knowledge_supported_confidence,
                overall_confidence=overall_confidence,
                kgot_quality_score=kgot_quality_score,
                alita_quality_score=alita_quality_score,
                unified_quality_score=unified_quality_score,
                validation_duration=processing_duration,
                processing_time_per_modality=processing_time_per_modality,
                resource_usage={'memory_mb': 0.0, 'cpu_percent': 0.0},  # Placeholder for resource monitoring
                statistical_significance=statistical_significance,
                confidence_interval=confidence_interval,
                p_value=p_value,
                reliability_score=reliability_score,
                accuracy_score=accuracy_score,
                consistency_score=consistency_score,
                performance_score=performance_score,
                sample_size=len(validation_spec.inputs)
            )
            
            logger.debug("Generated comprehensive metrics", extra={
                'operation': 'METRICS_GENERATION',
                'overall_confidence': overall_confidence,
                'contradiction_count': len(contradictions),
                'unified_quality_score': unified_quality_score
            })
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to generate comprehensive metrics", extra={
                'operation': 'METRICS_GENERATION_FAILED',
                'error': str(e)
            })
            # Return default metrics on failure
            return CrossModalMetrics(
                modality_consistency_scores={},
                knowledge_validation_score=0.0,
                contradiction_count=0,
                contradiction_severity_distribution={},
                individual_modality_confidence={},
                cross_modal_agreement_score=0.0,
                knowledge_supported_confidence=0.0,
                overall_confidence=0.0,
                kgot_quality_score=0.0,
                alita_quality_score=0.0,
                unified_quality_score=0.0,
                validation_duration=processing_duration,
                processing_time_per_modality={},
                resource_usage={},
                statistical_significance=False,
                confidence_interval=(0.0, 0.0),
                p_value=1.0
            )
    
    async def _generate_comprehensive_recommendations(self,
                                                   validation_spec: CrossModalValidationSpec,
                                                   consistency_scores: Dict[str, ConsistencyScore],
                                                   knowledge_validation: Dict[str, Any],
                                                   contradictions: List[ContradictionReport],
                                                   confidence_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate comprehensive recommendations based on validation results
        
        @param {CrossModalValidationSpec} validation_spec - Original validation specification
        @param {Dict[str, ConsistencyScore]} consistency_scores - Cross-modal consistency results
        @param {Dict[str, Any]} knowledge_validation - Knowledge graph validation results
        @param {List[ContradictionReport]} contradictions - Detected contradictions
        @param {Dict[str, Any]} confidence_analysis - Confidence scoring results
        @returns {List[str]} - List of actionable recommendations
        """
        recommendations = []
        
        try:
            # Consistency-based recommendations
            low_consistency_pairs = [
                pair for pair, score in consistency_scores.items() 
                if score.overall_consistency < 0.7
            ]
            
            if low_consistency_pairs:
                recommendations.append(
                    f"Low consistency detected in {len(low_consistency_pairs)} modality pairs. "
                    f"Consider reviewing input quality and alignment for: {', '.join(low_consistency_pairs)}"
                )
            
            # Knowledge validation recommendations
            knowledge_score = knowledge_validation.get('overall_knowledge_score', 0.0)
            if knowledge_score < 0.6:
                unsupported_claims = len(knowledge_validation.get('unsupported_claims', []))
                recommendations.append(
                    f"Knowledge validation score is low ({knowledge_score:.2f}). "
                    f"Consider verifying {unsupported_claims} unsupported claims against authoritative sources."
                )
            
            # Contradiction-based recommendations
            if contradictions:
                critical_contradictions = [c for c in contradictions if c.severity == ErrorSeverity.CRITICAL]
                high_contradictions = [c for c in contradictions if c.severity == ErrorSeverity.HIGH]
                
                if critical_contradictions:
                    recommendations.append(
                        f"URGENT: {len(critical_contradictions)} critical contradictions detected. "
                        f"These require immediate resolution before proceeding."
                    )
                
                if high_contradictions:
                    recommendations.append(
                        f"High-priority contradictions found ({len(high_contradictions)}). "
                        f"Review and resolve these contradictions to improve validation quality."
                    )
                
                # Add specific resolution suggestions from contradictions
                for contradiction in contradictions[:3]:  # Top 3 contradictions
                    if contradiction.resolution_suggestions:
                        recommendations.extend(contradiction.resolution_suggestions[:1])  # One suggestion per contradiction
            
            # Confidence-based recommendations
            overall_confidence = confidence_analysis.get('overall_confidence', 0.0)
            if overall_confidence < 0.7:
                recommendations.append(
                    f"Overall confidence is low ({overall_confidence:.2f}). "
                    f"Consider gathering additional evidence or improving input quality."
                )
            
            # Statistical significance recommendations
            if not confidence_analysis.get('statistical_significance', False):
                sample_size = len(validation_spec.inputs)
                recommendations.append(
                    f"Results may not be statistically significant with {sample_size} inputs. "
                    f"Consider increasing sample size for more reliable validation."
                )
            
            # Validation level recommendations
            if validation_spec.validation_level == ValidationLevel.BASIC:
                recommendations.append(
                    "Consider upgrading to STANDARD or COMPREHENSIVE validation level "
                    "for more thorough analysis and better reliability."
                )
            
            # General quality improvements
            if not recommendations:
                recommendations.append(
                    "Validation completed successfully. All metrics are within acceptable ranges. "
                    "Continue monitoring for consistency and quality maintenance."
                )
            
            # Limit recommendations to prevent overwhelming output
            recommendations = recommendations[:10]
            
            logger.debug("Generated validation recommendations", extra={
                'operation': 'RECOMMENDATIONS_GENERATION',
                'recommendation_count': len(recommendations),
                'contradiction_count': len(contradictions)
            })
            
            return recommendations
            
        except Exception as e:
            logger.error("Failed to generate recommendations", extra={
                'operation': 'RECOMMENDATIONS_GENERATION_FAILED',
                'error': str(e)
            })
            return [f"Recommendation generation failed: {str(e)}"]
    
    async def _determine_overall_validity(self,
                                        metrics: CrossModalMetrics,
                                        contradictions: List[ContradictionReport]) -> bool:
        """
        Determine overall validity of cross-modal inputs based on comprehensive analysis
        
        @param {CrossModalMetrics} metrics - Comprehensive validation metrics
        @param {List[ContradictionReport]} contradictions - Detected contradictions
        @returns {bool} - True if validation passes, False otherwise
        """
        try:
            # Critical failure conditions
            critical_contradictions = [c for c in contradictions if c.severity == ErrorSeverity.CRITICAL]
            if critical_contradictions:
                logger.info("Validation failed due to critical contradictions", extra={
                    'operation': 'VALIDITY_CHECK',
                    'critical_contradictions': len(critical_contradictions)
                })
                return False
            
            # High-level contradiction threshold
            high_contradictions = [c for c in contradictions if c.severity == ErrorSeverity.HIGH]
            if len(high_contradictions) > 2:  # Allow up to 2 high-severity contradictions
                logger.info("Validation failed due to excessive high-severity contradictions", extra={
                    'operation': 'VALIDITY_CHECK',
                    'high_contradictions': len(high_contradictions)
                })
                return False
            
            # Minimum thresholds for key metrics
            thresholds = {
                'overall_confidence': 0.6,
                'consistency_score': 0.65,
                'knowledge_validation_score': 0.5,
                'unified_quality_score': 0.6
            }
            
            # Check each threshold
            if metrics.overall_confidence < thresholds['overall_confidence']:
                logger.info("Validation failed due to low overall confidence", extra={
                    'operation': 'VALIDITY_CHECK',
                    'confidence': metrics.overall_confidence,
                    'threshold': thresholds['overall_confidence']
                })
                return False
            
            if metrics.consistency_score < thresholds['consistency_score']:
                logger.info("Validation failed due to low consistency", extra={
                    'operation': 'VALIDITY_CHECK',
                    'consistency': metrics.consistency_score,
                    'threshold': thresholds['consistency_score']
                })
                return False
            
            if metrics.knowledge_validation_score < thresholds['knowledge_validation_score']:
                logger.info("Validation failed due to low knowledge validation", extra={
                    'operation': 'VALIDITY_CHECK',
                    'knowledge_score': metrics.knowledge_validation_score,
                    'threshold': thresholds['knowledge_validation_score']
                })
                return False
            
            if metrics.unified_quality_score < thresholds['unified_quality_score']:
                logger.info("Validation failed due to low unified quality", extra={
                    'operation': 'VALIDITY_CHECK',
                    'quality_score': metrics.unified_quality_score,
                    'threshold': thresholds['unified_quality_score']
                })
                return False
            
            # Additional checks for comprehensive validation
            if metrics.cross_modal_agreement_score < 0.5:
                logger.info("Validation failed due to poor cross-modal agreement", extra={
                    'operation': 'VALIDITY_CHECK',
                    'agreement_score': metrics.cross_modal_agreement_score
                })
                return False
            
            # All checks passed
            logger.info("Validation passed all validity checks", extra={
                'operation': 'VALIDITY_CHECK',
                'overall_confidence': metrics.overall_confidence,
                'consistency_score': metrics.consistency_score,
                'quality_score': metrics.unified_quality_score
            })
            return True
            
        except Exception as e:
            logger.error("Failed to determine overall validity", extra={
                'operation': 'VALIDITY_CHECK_FAILED',
                'error': str(e)
            })
            return False  # Default to invalid on error
    
    async def _calculate_overall_score(self,
                                     metrics: CrossModalMetrics,
                                     confidence_analysis: Dict[str, Any]) -> float:
        """
        Calculate unified overall validation score
        
        @param {CrossModalMetrics} metrics - Comprehensive validation metrics
        @param {Dict[str, Any]} confidence_analysis - Confidence scoring results
        @returns {float} - Overall score (0.0-1.0)
        """
        try:
            # Weighted components for overall score
            weights = {
                'consistency': 0.25,
                'knowledge': 0.20,
                'confidence': 0.20,
                'quality': 0.20,
                'reliability': 0.15
            }
            
            # Component scores
            consistency_component = metrics.consistency_score * weights['consistency']
            knowledge_component = metrics.knowledge_validation_score * weights['knowledge']
            confidence_component = metrics.overall_confidence * weights['confidence']
            quality_component = metrics.unified_quality_score * weights['quality']
            reliability_component = metrics.reliability_score * weights['reliability']
            
            # Calculate overall score
            overall_score = (
                consistency_component +
                knowledge_component +
                confidence_component +
                quality_component +
                reliability_component
            )
            
            # Apply penalties for contradictions
            contradiction_penalty = 0.0
            if metrics.contradiction_count > 0:
                # Penalty based on severity distribution
                severity_weights = {
                    'critical': 0.3,
                    'high': 0.15,
                    'medium': 0.05,
                    'low': 0.01
                }
                
                for severity, count in metrics.contradiction_severity_distribution.items():
                    penalty_weight = severity_weights.get(severity, 0.01)
                    contradiction_penalty += count * penalty_weight
            
            # Apply penalty (capped at 0.5 to avoid excessively low scores)
            overall_score = max(0.0, overall_score - min(contradiction_penalty, 0.5))
            
            # Ensure score is within valid range
            overall_score = max(0.0, min(1.0, overall_score))
            
            logger.debug("Calculated overall validation score", extra={
                'operation': 'OVERALL_SCORE_CALCULATION',
                'overall_score': overall_score,
                'components': {
                    'consistency': consistency_component,
                    'knowledge': knowledge_component,
                    'confidence': confidence_component,
                    'quality': quality_component,
                    'reliability': reliability_component
                },
                'contradiction_penalty': contradiction_penalty
            })
            
            return overall_score
            
        except Exception as e:
            logger.error("Failed to calculate overall score", extra={
                'operation': 'OVERALL_SCORE_CALCULATION_FAILED',
                'error': str(e)
            })
            return 0.0  # Default to 0 on error
    
    def _calculate_kgot_quality_score(self,
                                    knowledge_validation: Dict[str, Any],
                                    consistency_scores: Dict[str, ConsistencyScore],
                                    contradictions: List[ContradictionReport]) -> float:
        """Calculate quality score using KGoT validation frameworks"""
        # KGoT emphasizes knowledge graph validation and consistency
        knowledge_score = knowledge_validation.get('overall_knowledge_score', 0.0)
        consistency_score = np.mean([cs.overall_consistency for cs in consistency_scores.values()]) if consistency_scores else 0.0
        
        # KGoT penalizes logical and factual contradictions more heavily
        kgot_contradiction_penalty = 0.0
        for contradiction in contradictions:
            if contradiction.contradiction_type in [ContradictionType.LOGICAL, ContradictionType.FACTUAL]:
                kgot_contradiction_penalty += 0.1
        
        kgot_score = (knowledge_score * 0.6 + consistency_score * 0.4) - kgot_contradiction_penalty
        return max(0.0, min(1.0, kgot_score))
    
    def _calculate_alita_quality_score(self,
                                     consistency_scores: Dict[str, ConsistencyScore],
                                     contradictions: List[ContradictionReport],
                                     confidence_analysis: Dict[str, Any]) -> float:
        """Calculate quality score using Alita validation frameworks"""
        # Alita emphasizes cross-modal consistency and confidence
        consistency_score = np.mean([cs.overall_consistency for cs in consistency_scores.values()]) if consistency_scores else 0.0
        confidence_score = confidence_analysis.get('overall_confidence', 0.0)
        
        # Alita penalizes semantic and temporal contradictions
        alita_contradiction_penalty = 0.0
        for contradiction in contradictions:
            if contradiction.contradiction_type in [ContradictionType.SEMANTIC, ContradictionType.TEMPORAL]:
                alita_contradiction_penalty += 0.1
        
        alita_score = (consistency_score * 0.5 + confidence_score * 0.5) - alita_contradiction_penalty
        return max(0.0, min(1.0, alita_score))
    
    def _calculate_reliability_score(self,
                                   consistency_scores: Dict[str, ConsistencyScore],
                                   contradictions: List[ContradictionReport]) -> float:
        """Calculate reliability score based on consistency and contradictions"""
        if not consistency_scores:
            return 0.0
        
        # Base reliability from consistency scores
        consistency_reliability = np.mean([cs.overall_consistency for cs in consistency_scores.values()])
        
        # Reliability penalty from contradictions
        contradiction_penalty = len(contradictions) * 0.05  # 5% penalty per contradiction
        
        reliability = max(0.0, consistency_reliability - contradiction_penalty)
        return min(1.0, reliability)
    
    def _calculate_performance_score(self, processing_duration: float, input_count: int) -> float:
        """Calculate performance score based on processing efficiency"""
        # Target: 2 seconds per input for good performance
        target_time_per_input = 2.0
        actual_time_per_input = processing_duration / max(input_count, 1)
        
        # Performance score decreases as time increases beyond target
        if actual_time_per_input <= target_time_per_input:
            return 1.0
        else:
            # Exponential decay for performance score
            performance_ratio = target_time_per_input / actual_time_per_input
            return max(0.1, performance_ratio)  # Minimum 0.1 score

    def get_model_assignments(self) -> Dict[str, str]:
        """
        Get current model assignments for debugging and monitoring
        
        @returns {Dict[str, str]} - Current model assignments by component
        """
        return {
            'consistency_checking': 'claude-4-sonnet(webagent)',
            'knowledge_validation': 'claude-4-sonnet(webagent)', 
            'contradiction_detection': 'claude-4-sonnet(webagent)',
            'confidence_scoring': 'gemini-2.5-pro(orchestration)',
            'vision_processing': 'o3(vision)',
            'orchestration': 'gemini-2.5-pro(orchestration)'
        }


def create_kgot_alita_cross_modal_validator(config: Dict[str, Any]) -> KGoTAlitaCrossModalValidator:
    """
    Factory function to create KGoT-Alita Cross-Modal Validator with configuration
    Uses @modelsrule.mdc models: o3(vision), claude-4-sonnet(webagent), gemini-2.5-pro(orchestration)
    
    @param {Dict[str, Any]} config - Configuration for validator components
    @returns {KGoTAlitaCrossModalValidator} - Configured validator instance
    """
    # Initialize specialized LLM clients based on @modelsrule.mdc
    llm_clients = config.get('llm_clients', {})
    
    if not llm_clients:
        # Create model-specific clients using OpenRouter endpoints
        try:
            from langchain_openai import ChatOpenAI
            
            # o3(vision) for vision-related tasks
            vision_client = ChatOpenAI(
                model_name='openai/o3-mini',  # o3(vision) model
                temperature=config.get('temperature', 0.3),
                openai_api_base='https://openrouter.ai/api/v1',
                openai_api_key=config.get('openrouter_api_key', os.getenv('OPENROUTER_API_KEY'))
            )
            
            # claude-4-sonnet(webagent) for web agent and reasoning tasks
            webagent_client = ChatOpenAI(
                model_name='anthropic/claude-4-sonnet',  # claude-4-sonnet(webagent) model
                temperature=config.get('temperature', 0.3),
                openai_api_base='https://openrouter.ai/api/v1',
                openai_api_key=config.get('openrouter_api_key', os.getenv('OPENROUTER_API_KEY'))
            )
            
            # gemini-2.5-pro(orchestration) for orchestration and coordination
            orchestration_client = ChatOpenAI(
                model_name='google/gemini-2.5-pro',  # gemini-2.5-pro(orchestration) model
                temperature=config.get('temperature', 0.3),
                openai_api_base='https://openrouter.ai/api/v1',
                openai_api_key=config.get('openrouter_api_key', os.getenv('OPENROUTER_API_KEY'))
            )
            
            llm_clients = {
                'vision': vision_client,
                'webagent': webagent_client,
                'orchestration': orchestration_client,
                'default': orchestration_client  # Use orchestration as default
            }
            
            logger.info("Initialized specialized LLM clients using @modelsrule.mdc", extra={
                'operation': 'LLM_CLIENTS_INIT',
                'models': {
                    'vision': 'openai/o3-mini',
                    'webagent': 'anthropic/claude-4-sonnet', 
                    'orchestration': 'google/gemini-2.5-pro'
                }
            })
            
        except ImportError:
            logger.warning("Failed to create specialized LLM clients")
            llm_clients = {'default': None}
    
    # Fallback to single client if needed
    llm_client = llm_clients.get('default') or list(llm_clients.values())[0] if llm_clients else None
    
    # Initialize optional components based on configuration
    knowledge_extraction_manager = config.get('knowledge_extraction_manager')
    graph_store = config.get('graph_store')
    error_management_system = config.get('error_management_system')
    visual_analyzer = config.get('visual_analyzer')
    
    return KGoTAlitaCrossModalValidator(
        llm_client=llm_client,
        knowledge_extraction_manager=knowledge_extraction_manager,
        graph_store=graph_store,
        error_management_system=error_management_system,
        visual_analyzer=visual_analyzer,
        config=config,
        llm_clients=llm_clients
    )


if __name__ == "__main__":
    async def example_usage():
        """
        Example usage of KGoT-Alita Cross-Modal Validator with @modelsrule.mdc models
        """
        # Example configuration using @modelsrule.mdc specialized models
        config = {
            'temperature': 0.3,
            'openrouter_api_key': 'your-openrouter-api-key',  # Set your OpenRouter API key
            # The factory will automatically create specialized clients:
            # - o3(vision) for visual tasks
            # - claude-4-sonnet(webagent) for reasoning and web agent tasks  
            # - gemini-2.5-pro(orchestration) for orchestration and coordination
        }
        
        # Create validator with specialized model clients
        validator = create_kgot_alita_cross_modal_validator(config)
        
        # Show model assignments
        print("Model Assignments (@modelsrule.mdc):")
        for component, model in validator.get_model_assignments().items():
            print(f"  {component}: {model}")
        print()
        
        # Create example cross-modal inputs
        inputs = [
            CrossModalInput(
                input_id="text_input_1",
                modality_type=ModalityType.TEXT,
                content="The cat is sitting on the mat.",
                confidence=0.9
            ),
            CrossModalInput(
                input_id="image_input_1", 
                modality_type=ModalityType.IMAGE,
                content="/path/to/cat_image.jpg",
                confidence=0.8
            )
        ]
        
        # Create validation specification
        validation_spec = CrossModalValidationSpec(
            validation_id="example_validation_001",
            name="Example Cross-Modal Validation",
            description="Validate consistency between text description and image content",
            inputs=inputs,
            validation_level=ValidationLevel.COMPREHENSIVE,
            expected_consistency=True
        )
        
        # Perform validation
        try:
            result = await validator.validate_cross_modal_input(validation_spec)
            
            print(f"Validation Result:")
            print(f"  Validation ID: {result.validation_id}")
            print(f"  Is Valid: {result.is_valid}")
            print(f"  Overall Score: {result.overall_score:.3f}")
            print(f"  Contradictions Found: {len(result.contradictions)}")
            print(f"  Overall Confidence: {result.metrics.overall_confidence:.3f}")
            print(f"  Recommendations: {len(result.recommendations)}")
            
        except Exception as e:
            print(f"Validation failed: {str(e)}")
    
    # Run example
    import asyncio
    asyncio.run(example_usage()) 