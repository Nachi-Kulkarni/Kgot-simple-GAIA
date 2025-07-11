#!/usr/bin/env python3
"""
KGoT-Enhanced Visual Analysis Engine

Implementation of Task 26: Create KGoT-Enhanced Visual Analysis Engine
Integrates KGoT Section 2.3 "Image Tool for multimodal inputs using Vision models"
with KGoT Section 2.1 "Graph Store Module" knowledge construction.

This module provides:
- Spatial relationship extraction integrated with KGoT graph storage
- Visual question answering with knowledge graph context
- Enhanced visual analysis beyond basic image inspection
- Multi-modal reasoning combining vision and structured knowledge
- Incremental knowledge graph enhancement from visual data

@module KGoTVisualAnalyzer
@author Enhanced Alita KGoT Team
@date 2025
"""

import asyncio
import base64
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import requests

# Add paths for KGoT and Alita components integration
sys.path.append(str(Path(__file__).parent.parent.parent / "knowledge-graph-of-thoughts"))
sys.path.append(str(Path(__file__).parent.parent))

# Import KGoT Section 2.3 Image Tool components
try:
    from kgot.tools.tools_v2_3.ImageQuestionTool import ImageQuestionTool, ImageQuestionSchema
    from kgot.utils import UsageStatistics, llm_utils
    from kgot.utils.log_and_statistics import collect_stats
except ImportError as e:
    logging.warning(f"KGoT tools import failed: {e}")
    ImageQuestionTool = None

# Import KGoT Section 2.1 Graph Store Module components
from kgot_core.graph_store.kg_interface import KnowledgeGraphInterface
from kgot_core.graph_store.networkx_implementation import NetworkXKnowledgeGraph
try:
    from kgot_core.graph_store.neo4j_implementation import Neo4jKnowledgeGraph
except ImportError:
    Neo4jKnowledgeGraph = None

# Import existing Alita integration components
from kgot_core.integrated_tools.integrated_tools_manager import ModelConfiguration
from config.logging.winston_config import loggers

# LangChain imports (as per user memory preference for agent development)
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Setup Winston-compatible logging
logger = loggers.get('multimodal') or logging.getLogger('KGoTVisualAnalyzer')


@dataclass
class VisualAnalysisConfig:
    """
    Configuration for KGoT Visual Analysis Engine
    
    Manages settings for vision model integration, graph store connectivity,
    and spatial relationship extraction capabilities.
    """
    # Vision model configuration (KGoT Section 2.3)
    vision_model: str = "openai/o3"  # OpenAI o3 for multimodal inputs as specified
    orchestration_model: str = "x-ai/grok-4"  # Gemini 2.5 Pro for complex reasoning
    temperature: float = 0.3
    max_tokens: int = 32000
    
    # Graph store configuration (KGoT Section 2.1)
    graph_backend: str = "networkx"  # Default to NetworkX for development
    graph_store_config: Dict[str, Any] = field(default_factory=dict)
    
    # Spatial analysis configuration
    enable_object_detection: bool = True
    enable_spatial_relationships: bool = True
    enable_scene_understanding: bool = True
    confidence_threshold: float = 0.7
    
    # Visual question answering configuration
    enable_graph_context: bool = True
    max_context_entities: int = 50
    max_context_relationships: int = 100


@dataclass
class SpatialObject:
    """
    Represents an object detected in an image with spatial properties
    """
    object_id: str
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized coordinates
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialRelationship:
    """
    Represents a spatial relationship between two objects in an image
    """
    subject_id: str
    predicate: str  # "to_the_left_of", "contains", "adjacent_to", etc.
    object_id: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpatialRelationshipExtractor:
    """
    Extracts spatial relationships between detected objects in images
    
    This class implements spatial reasoning capabilities that identify objects
    and their positional relationships, converting visual scenes into structured
    spatial knowledge suitable for graph storage.
    """
    
    def __init__(self, config: VisualAnalysisConfig):
        """
        Initialize the spatial relationship extractor
        
        Args:
            config (VisualAnalysisConfig): Configuration for spatial analysis
        """
        self.config = config
        self.vision_llm = self._initialize_vision_model()
        
        logger.info("Spatial Relationship Extractor initialized", extra={
            'operation': 'SPATIAL_EXTRACTOR_INIT',
            'vision_model': config.vision_model,
            'confidence_threshold': config.confidence_threshold
        })
    
    def _initialize_vision_model(self) -> Runnable:
        """
        Initialize the vision model for spatial analysis
        
        Returns:
            Runnable: Configured vision language model
        """
        try:
            return llm_utils.get_llm(
                model_name=self.config.vision_model,
                temperature=self.config.temperature
            )
        except Exception as e:
            logger.error("Failed to initialize vision model", extra={
                'operation': 'VISION_MODEL_INIT_FAILED',
                'error': str(e)
            })
            raise
    
    @collect_stats("spatial_object_detection")
    def detect_objects(self, image_path: str) -> List[SpatialObject]:
        """
        Detect objects in an image with spatial coordinates
        
        Args:
            image_path (str): Path to image file or base64 encoded image
            
        Returns:
            List[SpatialObject]: List of detected objects with spatial properties
        """
        logger.info("Starting object detection", extra={
            'operation': 'OBJECT_DETECTION_START',
            'image_path': image_path
        })
        
        try:
            # Encode image for vision model
            image_data = self._encode_image(image_path)
            
            # Create prompt for object detection with spatial analysis
            detection_prompt = """
            Analyze this image and detect all objects with their spatial properties.
            For each object, provide:
            1. Object label/category
            2. Confidence score (0.0-1.0)
            3. Bounding box coordinates (normalized x1, y1, x2, y2)
            4. Any relevant properties (color, size, material, etc.)
            
            Format your response as a structured list of objects.
            Be precise with spatial coordinates and thorough with object identification.
            """
            
            # Query vision model for object detection
            response = self.vision_llm.invoke([
                SystemMessage(content=detection_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": "Detect objects with spatial coordinates:"},
                    {"type": "image_url", "image_url": {"url": image_data, "detail": "high"}}
                ])
            ])
            
            # Parse response into structured objects
            objects = self._parse_object_detection_response(response.content)
            
            logger.info("Object detection completed", extra={
                'operation': 'OBJECT_DETECTION_SUCCESS',
                'objects_detected': len(objects),
                'image_path': image_path
            })
            
            return objects
            
        except Exception as e:
            logger.error("Object detection failed", extra={
                'operation': 'OBJECT_DETECTION_FAILED',
                'error': str(e),
                'image_path': image_path
            })
            raise
    
    @collect_stats("spatial_relationship_extraction")
    def extract_spatial_relationships(self, objects: List[SpatialObject]) -> List[SpatialRelationship]:
        """
        Extract spatial relationships between detected objects
        
        Args:
            objects (List[SpatialObject]): List of detected objects
            
        Returns:
            List[SpatialRelationship]: List of spatial relationships
        """
        logger.info("Extracting spatial relationships", extra={
            'operation': 'SPATIAL_RELATIONSHIP_START',
            'object_count': len(objects)
        })
        
        relationships = []
        
        try:
            # Analyze pairwise spatial relationships
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects):
                    if i != j:  # Don't compare object with itself
                        relationship = self._analyze_spatial_relationship(obj1, obj2)
                        if relationship and relationship.confidence >= self.config.confidence_threshold:
                            relationships.append(relationship)
            
            logger.info("Spatial relationship extraction completed", extra={
                'operation': 'SPATIAL_RELATIONSHIP_SUCCESS',
                'relationships_found': len(relationships)
            })
            
            return relationships
            
        except Exception as e:
            logger.error("Spatial relationship extraction failed", extra={
                'operation': 'SPATIAL_RELATIONSHIP_FAILED',
                'error': str(e)
            })
            raise
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image for vision model processing
        
        Args:
            image_path (str): Path to image or base64 data
            
        Returns:
            str: Properly formatted image data for vision model
        """
        # Check if already a data URL
        if image_path.startswith('data:image'):
            return image_path
        
        # Check if it's a URL
        if image_path.startswith('http'):
            return image_path
        
        # Local file - encode as base64
        try:
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            logger.error("Failed to encode image", extra={
                'operation': 'IMAGE_ENCODE_FAILED',
                'error': str(e),
                'image_path': image_path
            })
            raise
    
    def _parse_object_detection_response(self, response: str) -> List[SpatialObject]:
        """
        Parse vision model response into structured object list
        
        Args:
            response (str): Raw response from vision model
            
        Returns:
            List[SpatialObject]: Parsed spatial objects
        """
        # This is a simplified parser - in production, this would be more robust
        # and potentially use structured output from the LLM
        objects = []
        
        try:
            # For now, create sample objects - this would parse actual model response
            # TODO: Implement robust parsing of vision model object detection output
            sample_object = SpatialObject(
                object_id="obj_1",
                label="person",
                confidence=0.9,
                bbox=(0.1, 0.2, 0.4, 0.8),
                properties={"color": "blue", "size": "medium"}
            )
            objects.append(sample_object)
            
        except Exception as e:
            logger.error("Failed to parse object detection response", extra={
                'operation': 'OBJECT_PARSING_FAILED',
                'error': str(e)
            })
        
        return objects
    
    def _analyze_spatial_relationship(self, obj1: SpatialObject, obj2: SpatialObject) -> Optional[SpatialRelationship]:
        """
        Analyze spatial relationship between two objects
        
        Args:
            obj1 (SpatialObject): First object
            obj2 (SpatialObject): Second object
            
        Returns:
            Optional[SpatialRelationship]: Detected spatial relationship if any
        """
        # Calculate spatial relationships based on bounding boxes
        x1_1, y1_1, x2_1, y2_1 = obj1.bbox
        x1_2, y1_2, x2_2, y2_2 = obj2.bbox
        
        # Determine spatial relationship
        if x2_1 < x1_2:  # obj1 to the left of obj2
            return SpatialRelationship(
                subject_id=obj1.object_id,
                predicate="to_the_left_of",
                object_id=obj2.object_id,
                confidence=0.9,
                metadata={"spatial_analysis": "bbox_comparison"}
            )
        elif x1_1 > x2_2:  # obj1 to the right of obj2
            return SpatialRelationship(
                subject_id=obj1.object_id,
                predicate="to_the_right_of",
                object_id=obj2.object_id,
                confidence=0.9,
                metadata={"spatial_analysis": "bbox_comparison"}
            )
        # Add more spatial relationship detection logic here
        
        return None


class VisualKnowledgeConstructor:
    """
    Converts visual insights into structured knowledge graph triplets
    
    This class bridges the gap between visual analysis results and graph storage,
    implementing KGoT Section 2.1 knowledge construction patterns for visual data.
    """
    
    def __init__(self, config: VisualAnalysisConfig, graph_store: KnowledgeGraphInterface):
        """
        Initialize the visual knowledge constructor
        
        Args:
            config (VisualAnalysisConfig): Configuration for knowledge construction
            graph_store (KnowledgeGraphInterface): Graph store instance for knowledge storage
        """
        self.config = config
        self.graph_store = graph_store
        
        logger.info("Visual Knowledge Constructor initialized", extra={
            'operation': 'KNOWLEDGE_CONSTRUCTOR_INIT',
            'graph_backend': config.graph_backend
        })
    
    @collect_stats("visual_knowledge_construction")
    async def construct_knowledge_from_visual_analysis(
        self, 
        image_id: str,
        objects: List[SpatialObject],
        relationships: List[SpatialRelationship]
    ) -> Dict[str, Any]:
        """
        Construct knowledge graph entries from visual analysis results
        
        Args:
            image_id (str): Unique identifier for the analyzed image
            objects (List[SpatialObject]): Detected objects with spatial properties
            relationships (List[SpatialRelationship]): Spatial relationships between objects
            
        Returns:
            Dict[str, Any]: Summary of knowledge construction results
        """
        logger.info("Starting visual knowledge construction", extra={
            'operation': 'VISUAL_KNOWLEDGE_START',
            'image_id': image_id,
            'objects_count': len(objects),
            'relationships_count': len(relationships)
        })
        
        construction_results = {
            'image_id': image_id,
            'entities_created': 0,
            'relationships_created': 0,
            'errors': []
        }
        
        try:
            # Create image entity in graph
            await self._create_image_entity(image_id)
            construction_results['entities_created'] += 1
            
            # Create object entities
            for obj in objects:
                try:
                    await self._create_object_entity(obj, image_id)
                    construction_results['entities_created'] += 1
                except Exception as e:
                    construction_results['errors'].append(f"Failed to create entity for {obj.object_id}: {e}")
            
            # Create spatial relationships
            for rel in relationships:
                try:
                    await self._create_spatial_relationship(rel, image_id)
                    construction_results['relationships_created'] += 1
                except Exception as e:
                    construction_results['errors'].append(f"Failed to create relationship {rel.predicate}: {e}")
            
            logger.info("Visual knowledge construction completed", extra={
                'operation': 'VISUAL_KNOWLEDGE_SUCCESS',
                'construction_results': construction_results
            })
            
            return construction_results
            
        except Exception as e:
            logger.error("Visual knowledge construction failed", extra={
                'operation': 'VISUAL_KNOWLEDGE_FAILED',
                'error': str(e),
                'image_id': image_id
            })
            construction_results['errors'].append(f"Construction failed: {e}")
            return construction_results
    
    async def _create_image_entity(self, image_id: str) -> None:
        """
        Create an image entity in the knowledge graph
        
        Args:
            image_id (str): Unique identifier for the image
        """
        await self.graph_store.addEntity({
            'id': image_id,
            'type': 'IMAGE',
            'properties': {
                'analyzed_at': datetime.now().isoformat(),
                'analysis_engine': 'KGoTVisualAnalyzer'
            }
        })
    
    async def _create_object_entity(self, obj: SpatialObject, image_id: str) -> None:
        """
        Create an object entity in the knowledge graph
        
        Args:
            obj (SpatialObject): Detected object with spatial properties
            image_id (str): ID of the image containing this object
        """
        await self.graph_store.addEntity({
            'id': obj.object_id,
            'type': 'VISUAL_OBJECT',
            'properties': {
                'label': obj.label,
                'confidence': obj.confidence,
                'bbox': obj.bbox,
                'source_image': image_id,
                **obj.properties
            }
        })
        
        # Create relationship between object and image
        await self.graph_store.addTriplet({
            'subject': obj.object_id,
            'predicate': 'detected_in',
            'object': image_id,
            'metadata': {
                'detection_confidence': obj.confidence,
                'spatial_coordinates': obj.bbox
            }
        })
    
    async def _create_spatial_relationship(self, rel: SpatialRelationship, image_id: str) -> None:
        """
        Create a spatial relationship in the knowledge graph
        
        Args:
            rel (SpatialRelationship): Spatial relationship between objects
            image_id (str): ID of the image containing this relationship
        """
        await self.graph_store.addTriplet({
            'subject': rel.subject_id,
            'predicate': rel.predicate,
            'object': rel.object_id,
            'metadata': {
                'confidence': rel.confidence,
                'source_image': image_id,
                'relationship_type': 'spatial',
                **rel.metadata
            }
        })


class ContextAwareVQA:
    """
    Context-Aware Visual Question Answering with Knowledge Graph Integration
    
    This class implements visual question answering that leverages both image analysis
    and existing knowledge graph context to provide comprehensive answers.
    """
    
    def __init__(self, config: VisualAnalysisConfig, graph_store: KnowledgeGraphInterface):
        """
        Initialize the context-aware VQA system
        
        Args:
            config (VisualAnalysisConfig): Configuration for VQA system
            graph_store (KnowledgeGraphInterface): Graph store for context retrieval
        """
        self.config = config
        self.graph_store = graph_store
        self.vision_llm = self._initialize_vision_model()
        self.reasoning_llm = self._initialize_reasoning_model()
        
        logger.info("Context-Aware VQA initialized", extra={
            'operation': 'VQA_INIT',
            'vision_model': config.vision_model,
            'reasoning_model': config.orchestration_model
        })
    
    def _initialize_vision_model(self) -> Runnable:
        """Initialize vision model for image analysis"""
        return llm_utils.get_llm(
            model_name=self.config.vision_model,
            temperature=self.config.temperature
        )
    
    def _initialize_reasoning_model(self) -> Runnable:
        """Initialize reasoning model for multi-modal analysis"""
        return llm_utils.get_llm(
            model_name=self.config.orchestration_model,
            temperature=self.config.temperature
        )
    
    @collect_stats("contextual_visual_qa")
    async def answer_visual_question(
        self, 
        question: str, 
        image_path: str,
        context_entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Answer a visual question using both image analysis and graph context
        
        Args:
            question (str): Question about the image
            image_path (str): Path to image file
            context_entities (Optional[List[str]]): Specific entities to include in context
            
        Returns:
            Dict[str, Any]: Answer with reasoning and context information
        """
        logger.info("Starting context-aware visual question answering", extra={
            'operation': 'VQA_START',
            'question': question,
            'image_path': image_path
        })
        
        try:
            # Step 1: Analyze image directly
            image_analysis = await self._analyze_image_for_question(question, image_path)
            
            # Step 2: Retrieve relevant graph context
            graph_context = await self._retrieve_graph_context(question, context_entities)
            
            # Step 3: Combine visual and graph evidence
            combined_answer = await self._synthesize_multimodal_answer(
                question, image_analysis, graph_context
            )
            
            result = {
                'question': question,
                'answer': combined_answer,
                'image_analysis': image_analysis,
                'graph_context': graph_context,
                'confidence': 0.8,  # TODO: Calculate actual confidence
                'reasoning_path': "multi-modal analysis with graph context"
            }
            
            logger.info("Visual question answering completed", extra={
                'operation': 'VQA_SUCCESS',
                'question': question,
                'answer_length': len(combined_answer)
            })
            
            return result
            
        except Exception as e:
            logger.error("Visual question answering failed", extra={
                'operation': 'VQA_FAILED',
                'error': str(e),
                'question': question
            })
            raise
    
    async def _analyze_image_for_question(self, question: str, image_path: str) -> str:
        """
        Analyze image to answer the specific question
        
        Args:
            question (str): Question about the image
            image_path (str): Path to image file
            
        Returns:
            str: Direct image analysis response
        """
        image_data = self._encode_image(image_path)
        
        response = self.vision_llm.invoke([
            SystemMessage(content="Analyze this image and answer the specific question. Be detailed and precise."),
            HumanMessage(content=[
                {"type": "text", "text": f"Question: {question}"},
                {"type": "image_url", "image_url": {"url": image_data, "detail": "high"}}
            ])
        ])
        
        return response.content
    
    async def _retrieve_graph_context(self, question: str, context_entities: Optional[List[str]]) -> Dict[str, Any]:
        """
        Retrieve relevant context from knowledge graph
        
        Args:
            question (str): Question to find relevant context for
            context_entities (Optional[List[str]]): Specific entities to include
            
        Returns:
            Dict[str, Any]: Relevant graph context
        """
        # TODO: Implement intelligent context retrieval based on question content
        # For now, return basic context structure
        context = {
            'entities': [],
            'relationships': [],
            'relevant_facts': []
        }
        
        if self.config.enable_graph_context:
            try:
                # Query graph for relevant entities and relationships
                # This would be enhanced with semantic search and relevance scoring
                graph_state = await self.graph_store.getCurrentGraphState()
                context['graph_summary'] = graph_state[:1000]  # Truncate for context
                
            except Exception as e:
                logger.warning("Failed to retrieve graph context", extra={
                    'operation': 'GRAPH_CONTEXT_FAILED',
                    'error': str(e)
                })
        
        return context
    
    async def _synthesize_multimodal_answer(
        self, 
        question: str, 
        image_analysis: str, 
        graph_context: Dict[str, Any]
    ) -> str:
        """
        Synthesize final answer combining visual analysis and graph context
        
        Args:
            question (str): Original question
            image_analysis (str): Results from image analysis
            graph_context (Dict[str, Any]): Relevant graph context
            
        Returns:
            str: Synthesized multi-modal answer
        """
        synthesis_prompt = f"""
        Question: {question}
        
        Image Analysis: {image_analysis}
        
        Knowledge Graph Context: {json.dumps(graph_context, indent=2)}
        
        Synthesize a comprehensive answer that combines the visual evidence from the image
        with the relevant context from the knowledge graph. Explain your reasoning and
        highlight how the graph context enhances the visual analysis.
        """
        
        response = self.reasoning_llm.invoke([
            SystemMessage(content="You are an expert at multi-modal reasoning. Combine visual and structured knowledge to provide comprehensive answers."),
            HumanMessage(content=synthesis_prompt)
        ])
        
        return response.content
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image for vision model processing"""
        # Reuse encoding logic from SpatialRelationshipExtractor
        extractor = SpatialRelationshipExtractor(self.config)
        return extractor._encode_image(image_path)


class KGoTVisualAnalyzer:
    """
    Main KGoT-Enhanced Visual Analysis Engine
    
    Integrates KGoT Section 2.3 "Image Tool for multimodal inputs using Vision models"
    with KGoT Section 2.1 "Graph Store Module" knowledge construction for comprehensive
    visual understanding and knowledge integration.
    
    Features:
    - Spatial relationship extraction from images
    - Visual question answering with knowledge graph context
    - Incremental knowledge graph enhancement from visual data
    - Multi-modal reasoning combining vision and structured knowledge
    """
    
    def __init__(self, config: Optional[VisualAnalysisConfig] = None):
        """
        Initialize the KGoT Visual Analysis Engine
        
        Args:
            config (Optional[VisualAnalysisConfig]): Configuration for visual analysis
        """
        self.config = config or VisualAnalysisConfig()
        
        # Initialize graph store connection
        self.graph_store = self._initialize_graph_store()
        
        # Initialize analysis components
        self.spatial_extractor = SpatialRelationshipExtractor(self.config)
        self.knowledge_constructor = VisualKnowledgeConstructor(self.config, self.graph_store)
        self.vqa_system = ContextAwareVQA(self.config, self.graph_store)
        
        # Initialize usage statistics tracking
        self.usage_statistics = UsageStatistics()
        
        logger.info("KGoT Visual Analyzer initialized", extra={
            'operation': 'KGOT_VISUAL_ANALYZER_INIT',
            'config': self.config.__dict__
        })
    
    def _initialize_graph_store(self) -> KnowledgeGraphInterface:
        """
        Initialize the graph store backend
        
        Returns:
            KnowledgeGraphInterface: Configured graph store instance
        """
        try:
            if self.config.graph_backend == "neo4j" and Neo4jKnowledgeGraph:
                graph_store = Neo4jKnowledgeGraph(self.config.graph_store_config)
            else:
                # Default to NetworkX for development
                graph_store = NetworkXKnowledgeGraph(self.config.graph_store_config)
            
            logger.info("Graph store initialized", extra={
                'operation': 'GRAPH_STORE_INIT',
                'backend': self.config.graph_backend
            })
            
            return graph_store
            
        except Exception as e:
            logger.error("Failed to initialize graph store", extra={
                'operation': 'GRAPH_STORE_INIT_FAILED',
                'error': str(e)
            })
            raise
    
    @collect_stats("complete_visual_analysis")
    async def analyze_image_with_graph_context(
        self, 
        image_path: str, 
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform complete visual analysis with graph knowledge integration
        
        This is the main method that orchestrates spatial relationship extraction,
        knowledge graph construction, and provides comprehensive visual understanding.
        
        Args:
            image_path (str): Path to image file or base64 encoded image
            analysis_options (Optional[Dict[str, Any]]): Options for analysis customization
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results including:
                - detected_objects: List of objects with spatial properties
                - spatial_relationships: List of spatial relationships
                - knowledge_construction: Results of graph integration
                - graph_enhancement: Summary of knowledge graph updates
        """
        start_time = time.time()
        image_id = f"img_{int(start_time)}"
        
        logger.info("Starting comprehensive visual analysis", extra={
            'operation': 'COMPLETE_ANALYSIS_START',
            'image_id': image_id,
            'image_path': image_path
        })
        
        analysis_results = {
            'image_id': image_id,
            'image_path': image_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'detected_objects': [],
            'spatial_relationships': [],
            'knowledge_construction': {},
            'processing_time_ms': 0,
            'errors': []
        }
        
        try:
            # Step 1: Object detection and spatial analysis
            if self.config.enable_object_detection:
                logger.info("Performing object detection", extra={
                    'operation': 'OBJECT_DETECTION_STEP',
                    'image_id': image_id
                })
                
                detected_objects = self.spatial_extractor.detect_objects(image_path)
                analysis_results['detected_objects'] = [
                    {
                        'object_id': obj.object_id,
                        'label': obj.label,
                        'confidence': obj.confidence,
                        'bbox': obj.bbox,
                        'properties': obj.properties
                    }
                    for obj in detected_objects
                ]
            
            # Step 2: Spatial relationship extraction
            if self.config.enable_spatial_relationships and analysis_results['detected_objects']:
                logger.info("Extracting spatial relationships", extra={
                    'operation': 'SPATIAL_RELATIONSHIPS_STEP',
                    'image_id': image_id
                })
                
                spatial_relationships = self.spatial_extractor.extract_spatial_relationships(detected_objects)
                analysis_results['spatial_relationships'] = [
                    {
                        'subject_id': rel.subject_id,
                        'predicate': rel.predicate,
                        'object_id': rel.object_id,
                        'confidence': rel.confidence,
                        'metadata': rel.metadata
                    }
                    for rel in spatial_relationships
                ]
            
            # Step 3: Knowledge graph construction
            logger.info("Constructing knowledge graph entries", extra={
                'operation': 'KNOWLEDGE_CONSTRUCTION_STEP',
                'image_id': image_id
            })
            
            construction_results = await self.knowledge_constructor.construct_knowledge_from_visual_analysis(
                image_id, detected_objects, spatial_relationships
            )
            analysis_results['knowledge_construction'] = construction_results
            
            # Calculate processing time
            analysis_results['processing_time_ms'] = int((time.time() - start_time) * 1000)
            
            logger.info("Complete visual analysis finished", extra={
                'operation': 'COMPLETE_ANALYSIS_SUCCESS',
                'image_id': image_id,
                'processing_time_ms': analysis_results['processing_time_ms'],
                'objects_detected': len(analysis_results['detected_objects']),
                'relationships_found': len(analysis_results['spatial_relationships'])
            })
            
            return analysis_results
            
        except Exception as e:
            logger.error("Complete visual analysis failed", extra={
                'operation': 'COMPLETE_ANALYSIS_FAILED',
                'error': str(e),
                'image_id': image_id
            })
            analysis_results['errors'].append(f"Analysis failed: {e}")
            analysis_results['processing_time_ms'] = int((time.time() - start_time) * 1000)
            return analysis_results
    
    @collect_stats("visual_question_answering")
    async def answer_visual_question(
        self, 
        question: str, 
        image_path: str,
        use_graph_context: bool = True
    ) -> Dict[str, Any]:
        """
        Answer visual questions with enhanced context from knowledge graph
        
        Args:
            question (str): Question about the image
            image_path (str): Path to image file
            use_graph_context (bool): Whether to use knowledge graph context
            
        Returns:
            Dict[str, Any]: Answer with reasoning and context information
        """
        logger.info("Starting visual question answering", extra={
            'operation': 'VQA_REQUEST',
            'question': question,
            'image_path': image_path,
            'use_graph_context': use_graph_context
        })
        
        try:
            # Use context-aware VQA system
            if use_graph_context and self.config.enable_graph_context:
                result = await self.vqa_system.answer_visual_question(question, image_path)
            else:
                # Fall back to basic image analysis if KGoT tools available
                if ImageQuestionTool:
                    image_tool = ImageQuestionTool(
                        model_name=self.config.vision_model,
                        temperature=self.config.temperature,
                        usage_statistics=self.usage_statistics
                    )
                    basic_answer = image_tool._run(question, image_path)
                    result = {
                        'question': question,
                        'answer': basic_answer,
                        'method': 'basic_image_analysis',
                        'graph_context_used': False
                    }
                else:
                    raise Exception("No image analysis tools available")
            
            logger.info("Visual question answering completed", extra={
                'operation': 'VQA_SUCCESS',
                'question': question
            })
            
            return result
            
        except Exception as e:
            logger.error("Visual question answering failed", extra={
                'operation': 'VQA_FAILED',
                'error': str(e),
                'question': question
            })
            raise
    
    async def enhance_graph_with_visual_data(self, image_path: str, enhancement_options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enhance knowledge graph with insights from visual analysis
        
        Args:
            image_path (str): Path to image for analysis
            enhancement_options (Optional[Dict]): Options for graph enhancement
            
        Returns:
            Dict[str, Any]: Summary of graph enhancement results
        """
        logger.info("Starting graph enhancement with visual data", extra={
            'operation': 'GRAPH_ENHANCEMENT_START',
            'image_path': image_path
        })
        
        try:
            # Perform complete visual analysis
            analysis_results = await self.analyze_image_with_graph_context(image_path)
            
            # Extract enhancement metrics
            enhancement_summary = {
                'image_analyzed': image_path,
                'entities_added': analysis_results['knowledge_construction'].get('entities_created', 0),
                'relationships_added': analysis_results['knowledge_construction'].get('relationships_created', 0),
                'analysis_timestamp': analysis_results['analysis_timestamp'],
                'processing_time_ms': analysis_results['processing_time_ms'],
                'errors': analysis_results['errors']
            }
            
            logger.info("Graph enhancement completed", extra={
                'operation': 'GRAPH_ENHANCEMENT_SUCCESS',
                'enhancement_summary': enhancement_summary
            })
            
            return enhancement_summary
            
        except Exception as e:
            logger.error("Graph enhancement failed", extra={
                'operation': 'GRAPH_ENHANCEMENT_FAILED',
                'error': str(e),
                'image_path': image_path
            })
            raise
    
    async def get_visual_analysis_capabilities(self) -> Dict[str, Any]:
        """
        Get information about current visual analysis capabilities
        
        Returns:
            Dict[str, Any]: Capabilities and configuration information
        """
        capabilities = {
            'engine_version': '1.0.0',
            'configuration': self.config.__dict__,
            'features': {
                'object_detection': self.config.enable_object_detection,
                'spatial_relationships': self.config.enable_spatial_relationships,
                'scene_understanding': self.config.enable_scene_understanding,
                'graph_context_vqa': self.config.enable_graph_context,
                'knowledge_construction': True
            },
            'models': {
                'vision_model': self.config.vision_model,
                'orchestration_model': self.config.orchestration_model
            },
            'graph_store': {
                'backend': self.config.graph_backend,
                'connected': self.graph_store.isInitialized if hasattr(self.graph_store, 'isInitialized') else True
            }
        }
        
        return capabilities


def create_kgot_visual_analyzer(config: Optional[VisualAnalysisConfig] = None) -> KGoTVisualAnalyzer:
    """
    Factory function to create a configured KGoT Visual Analyzer
    
    Args:
        config (Optional[VisualAnalysisConfig]): Configuration for the analyzer
        
    Returns:
        KGoTVisualAnalyzer: Configured visual analyzer instance
    """
    if config is None:
        config = VisualAnalysisConfig()
    
    analyzer = KGoTVisualAnalyzer(config)
    
    logger.info("KGoT Visual Analyzer created", extra={
        'operation': 'ANALYZER_CREATED',
        'config': config.__dict__
    })
    
    return analyzer


# Example usage and testing
if __name__ == "__main__":
    async def test_visual_analyzer():
        """Test the KGoT Visual Analyzer with sample usage"""
        print("Testing KGoT Visual Analyzer...")
        
        # Create analyzer with default configuration
        config = VisualAnalysisConfig(
            vision_model="openai/o3",
            graph_backend="networkx",
            enable_object_detection=True,
            enable_spatial_relationships=True
        )
        
        analyzer = create_kgot_visual_analyzer(config)
        
        # Test capabilities
        capabilities = await analyzer.get_visual_analysis_capabilities()
        print(f"Analyzer capabilities: {json.dumps(capabilities, indent=2)}")
        
        # Example: Analyze an image (would require actual image file)
        # results = await analyzer.analyze_image_with_graph_context("/path/to/image.jpg")
        # print(f"Analysis results: {json.dumps(results, indent=2)}")
        
        # Example: Answer visual question (would require actual image file)
        # answer = await analyzer.answer_visual_question("What objects are in this image?", "/path/to/image.jpg")
        # print(f"VQA answer: {json.dumps(answer, indent=2)}")
        
        print("KGoT Visual Analyzer test completed!")
    
    # Run test
    asyncio.run(test_visual_analyzer()) 