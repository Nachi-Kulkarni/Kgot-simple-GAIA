#!/usr/bin/env python3
"""
KGoT-Alita Screenshot Analyzer - Task 27 Implementation

Implementation of Task 27: Implement KGoT-Alita Screenshot Analyzer
- Integrate KGoT Section 2.3 web navigation with Alita Web Agent screenshot capabilities
- Design webpage layout analysis feeding KGoT Section 2.1 knowledge graph
- Implement UI element classification stored as KGoT entities and relationships
- Add accessibility feature identification with knowledge graph annotation

Model Configuration (as per modelsrule):
- o3 for vision tasks (screenshot analysis, UI classification, accessibility)
- claude-4-sonnet for web agent tasks (browser automation)
- gemini-2.5-pro for orchestration (complex reasoning)
- All models accessed via OpenRouter endpoints

This module provides:
- Web page screenshot analysis with UI element detection
- Accessibility feature identification and scoring
- Layout structure analysis and spatial relationships
- Integration with KGoT knowledge graph for webpage knowledge storage
- Bridge between Alita Web Agent browser automation and KGoT visual analysis

@module KGoTAlitaScreenshotAnalyzer
@author Enhanced Alita KGoT Team  
@date 2025
"""

import asyncio
import base64
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw
import subprocess

# Add paths for KGoT and Alita components integration
sys.path.append(str(Path(__file__).parent.parent.parent / "knowledge-graph-of-thoughts"))
sys.path.append(str(Path(__file__).parent.parent))

# Import KGoT components for knowledge graph integration
try:
    from kgot.tools.tools_v2_3.ImageQuestionTool import ImageQuestionTool, ImageQuestionSchema
    from kgot.utils import UsageStatistics, llm_utils
    from kgot.utils.log_and_statistics import collect_stats
except ImportError as e:
    logging.warning(f"KGoT tools import failed: {e}")
    ImageQuestionTool = None

# Import KGoT graph store components
from kgot_core.graph_store.kg_interface import KnowledgeGraphInterface
from kgot_core.graph_store.networkx_implementation import NetworkXKnowledgeGraph
try:
    from kgot_core.graph_store.neo4j_implementation import Neo4jKnowledgeGraph
except ImportError:
    Neo4jKnowledgeGraph = None

# Import existing visual analyzer components
from kgot_visual_analyzer import (
    KGoTVisualAnalyzer, VisualAnalysisConfig, SpatialObject, SpatialRelationship,
    SpatialRelationshipExtractor, VisualKnowledgeConstructor
)

# Import configuration and logging
from config.logging.winston_config import loggers

# LangChain imports for agent development (user memory preference)
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Setup Winston-compatible logging
logger = loggers.get('multimodal') or logging.getLogger('KGoTAlitaScreenshotAnalyzer')


class UIElementType(Enum):
    """Enumeration of UI element types for classification"""
    BUTTON = "button"
    INPUT_FIELD = "input_field"
    DROPDOWN = "dropdown"
    LINK = "link"
    IMAGE = "image"
    TEXT = "text"
    HEADING = "heading"
    NAVIGATION = "navigation"
    FORM = "form"
    TABLE = "table"
    LIST = "list"
    MODAL = "modal"
    MENU = "menu"
    FOOTER = "footer"
    HEADER = "header"
    SIDEBAR = "sidebar"
    CONTENT_AREA = "content_area"
    ADVERTISEMENT = "advertisement"
    SEARCH_BOX = "search_box"
    BREADCRUMB = "breadcrumb"
    PAGINATION = "pagination"
    TABS = "tabs"
    ACCORDION = "accordion"
    CAROUSEL = "carousel"
    VIDEO_PLAYER = "video_player"
    AUDIO_PLAYER = "audio_player"
    PROGRESS_BAR = "progress_bar"
    TOOLTIP = "tooltip"
    NOTIFICATION = "notification"
    POPUP = "popup"
    OVERLAY = "overlay"
    UNKNOWN = "unknown"


class AccessibilityFeature(Enum):
    """Enumeration of accessibility features for identification"""
    ALT_TEXT = "alt_text"
    ARIA_LABEL = "aria_label"
    HEADING_STRUCTURE = "heading_structure"
    KEYBOARD_NAVIGATION = "keyboard_navigation"
    COLOR_CONTRAST = "color_contrast"
    FOCUS_INDICATORS = "focus_indicators"
    SCREEN_READER_SUPPORT = "screen_reader_support"
    SKIP_LINKS = "skip_links"
    DESCRIPTIVE_LINKS = "descriptive_links"
    FORM_LABELS = "form_labels"
    ERROR_MESSAGES = "error_messages"
    LANGUAGE_ATTRIBUTES = "language_attributes"
    RESPONSIVE_DESIGN = "responsive_design"
    CAPTIONS_SUBTITLES = "captions_subtitles"
    AUDIO_DESCRIPTIONS = "audio_descriptions"
    TEXT_SCALING = "text_scaling"
    MOTION_CONTROL = "motion_control"
    TIMEOUT_CONTROLS = "timeout_controls"


@dataclass
class UIElement:
    """
    Represents a UI element detected in a webpage screenshot
    """
    element_id: str
    element_type: UIElementType
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coordinates
    properties: Dict[str, Any] = field(default_factory=dict)
    accessibility_features: List[AccessibilityFeature] = field(default_factory=list)
    text_content: Optional[str] = None
    html_attributes: Dict[str, str] = field(default_factory=dict)
    css_properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class LayoutStructure:
    """
    Represents the layout structure of a webpage
    """
    layout_id: str
    page_title: str
    url: str
    viewport_size: Tuple[int, int]  # (width, height)
    elements: List[UIElement] = field(default_factory=list)
    spatial_relationships: List[SpatialRelationship] = field(default_factory=list)
    accessibility_score: float = 0.0
    layout_patterns: List[str] = field(default_factory=list)
    color_scheme: Dict[str, str] = field(default_factory=dict)
    typography: Dict[str, str] = field(default_factory=dict)
    responsive_breakpoints: List[int] = field(default_factory=list)


@dataclass 
class AccessibilityAssessment:
    """
    Represents accessibility assessment results for a webpage
    """
    assessment_id: str
    overall_score: float  # 0.0 to 1.0
    wcag_compliance_level: str  # AA, AAA, etc.
    identified_features: List[AccessibilityFeature]
    missing_features: List[AccessibilityFeature]
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    color_contrast_ratios: Dict[str, float] = field(default_factory=dict)
    keyboard_navigation_score: float = 0.0
    screen_reader_compatibility: float = 0.0


@dataclass
class ScreenshotAnalysisConfig:
    """
    Configuration for KGoT-Alita Screenshot Analyzer
    """
    # Vision model configuration (using OpenRouter endpoints as per modelsrule)
    vision_model: str = "openai/o3"  # o3 for vision tasks
    ui_classification_model: str = "openai/o3"  # o3 for UI element classification  
    accessibility_model: str = "openai/o3"  # o3 for accessibility analysis
    web_agent_model: str = "anthropic/claude-4-sonnet"  # claude-4-sonnet for web agent tasks (using 3.5 as placeholder)
    orchestration_model: str = "google/gemini-2.5-pro"  # gemini-2.5-pro for orchestration
    openrouter_base_url: str = "https://openrouter.ai/api/v1"  # OpenRouter endpoint
    temperature: float = 0.2
    max_tokens: int = 32000
    
    # Graph store configuration
    graph_backend: str = "networkx"
    graph_store_config: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis configuration
    enable_ui_classification: bool = True
    enable_accessibility_analysis: bool = True
    enable_layout_analysis: bool = True
    enable_spatial_relationships: bool = True
    confidence_threshold: float = 0.7
    
    # Web agent integration
    alita_web_agent_url: str = "http://localhost:3001"
    playwright_config: Dict[str, Any] = field(default_factory=dict)
    
    # Accessibility analysis configuration
    wcag_compliance_target: str = "AA"  # AA or AAA
    color_contrast_threshold: float = 4.5  # WCAG AA standard
    
    # Performance configuration
    batch_processing: bool = True
    max_concurrent_analyses: int = 5
    cache_results: bool = True
    cache_expiry_hours: int = 24


class UIElementClassifier:
    """
    Classifies UI elements in webpage screenshots using computer vision and AI models
    
    This class implements advanced UI element detection and classification capabilities,
    identifying various types of interface components, their properties, and spatial
    relationships within webpage layouts.
    """
    
    def __init__(self, config: ScreenshotAnalysisConfig):
        """
        Initialize the UI element classifier
        
        Args:
            config (ScreenshotAnalysisConfig): Configuration for screenshot analysis
        """
        self.config = config
        self.vision_llm = self._initialize_vision_model()
        self.ui_classification_llm = self._initialize_ui_classification_model()
        
        # Initialize computer vision components
        self._initialize_cv_components()
        
        logger.info("UI Element Classifier initialized", extra={
            'operation': 'UI_CLASSIFIER_INIT',
            'vision_model': config.vision_model,
            'ui_classification_model': config.ui_classification_model,
            'confidence_threshold': config.confidence_threshold
        })
    
    def _initialize_vision_model(self) -> Runnable:
        """Initialize the vision model for UI analysis"""
        try:
            return llm_utils.get_llm(
                model_name=self.config.vision_model,
                temperature=self.config.temperature,
                base_url=self.config.openrouter_base_url  # OpenRouter endpoint
            )
        except Exception as e:
            logger.error("Failed to initialize vision model", extra={
                'operation': 'VISION_MODEL_INIT_FAILED',
                'error': str(e)
            })
            raise
    
    def _initialize_ui_classification_model(self) -> Runnable:
        """Initialize the UI classification model"""
        try:
            return llm_utils.get_llm(
                model_name=self.config.ui_classification_model,
                temperature=self.config.temperature,
                base_url=self.config.openrouter_base_url  # OpenRouter endpoint
            )
        except Exception as e:
            logger.error("Failed to initialize UI classification model", extra={
                'operation': 'UI_CLASSIFICATION_MODEL_INIT_FAILED',
                'error': str(e)
            })
            raise
    
    def _initialize_cv_components(self):
        """Initialize computer vision components for element detection"""
        try:
            # Initialize OpenCV components for edge detection and contour analysis
            self.edge_detector = cv2.Canny
            self.contour_detector = cv2.findContours
            
            logger.info("Computer vision components initialized", extra={
                'operation': 'CV_COMPONENTS_INIT'
            })
        except Exception as e:
            logger.error("Failed to initialize CV components", extra={
                'operation': 'CV_COMPONENTS_INIT_FAILED',
                'error': str(e)
            })
            raise
    
    @collect_stats("ui_element_classification")
    async def classify_ui_elements(self, screenshot_path: str, html_context: Optional[str] = None) -> List[UIElement]:
        """
        Classify UI elements in a webpage screenshot
        
        Args:
            screenshot_path (str): Path to the screenshot image
            html_context (Optional[str]): HTML source for additional context
            
        Returns:
            List[UIElement]: List of classified UI elements
        """
        logger.info("Starting UI element classification", extra={
            'operation': 'UI_ELEMENT_CLASSIFICATION_START',
            'screenshot_path': screenshot_path
        })
        
        try:
            # Load and preprocess the screenshot
            image = await self._load_and_preprocess_screenshot(screenshot_path)
            
            # Detect UI elements using computer vision
            cv_elements = await self._detect_elements_cv(image)
            
            # Classify elements using AI vision model
            ai_classified_elements = await self._classify_elements_ai(image, cv_elements, html_context)
            
            # Merge and refine classifications
            refined_elements = await self._refine_classifications(cv_elements, ai_classified_elements)
            
            # Extract additional properties
            enhanced_elements = await self._extract_element_properties(image, refined_elements, html_context)
            
            logger.info("UI element classification completed", extra={
                'operation': 'UI_ELEMENT_CLASSIFICATION_SUCCESS',
                'screenshot_path': screenshot_path,
                'element_count': len(enhanced_elements)
            })
            
            return enhanced_elements
            
        except Exception as e:
            logger.error("UI element classification failed", extra={
                'operation': 'UI_ELEMENT_CLASSIFICATION_ERROR',
                'screenshot_path': screenshot_path,
                'error': str(e)
            })
            raise
    
    async def _load_and_preprocess_screenshot(self, screenshot_path: str) -> np.ndarray:
        """Load and preprocess screenshot for analysis"""
        try:
            # Load image using PIL and convert to OpenCV format
            pil_image = Image.open(screenshot_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            logger.debug("Screenshot loaded and preprocessed", extra={
                'operation': 'SCREENSHOT_PREPROCESS',
                'image_shape': image.shape,
                'screenshot_path': screenshot_path
            })
            
            return image
            
        except Exception as e:
            logger.error("Failed to load and preprocess screenshot", extra={
                'operation': 'SCREENSHOT_PREPROCESS_ERROR',
                'screenshot_path': screenshot_path,
                'error': str(e)
            })
            raise
    
    async def _detect_elements_cv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect UI elements using computer vision techniques"""
        elements = []
        
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and process contours
            for i, contour in enumerate(contours):
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 20 or h < 20:
                    continue
                
                # Calculate area and aspect ratio
                area = cv2.contourArea(contour)
                aspect_ratio = float(w) / h
                
                # Create element data
                element_data = {
                    'cv_id': f"cv_element_{i}",
                    'bbox': (x, y, x + w, y + h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'contour': contour,
                    'detection_method': 'computer_vision'
                }
                
                elements.append(element_data)
            
            logger.debug("Computer vision element detection completed", extra={
                'operation': 'CV_ELEMENT_DETECTION',
                'element_count': len(elements)
            })
            
            return elements
            
        except Exception as e:
            logger.error("Computer vision element detection failed", extra={
                'operation': 'CV_ELEMENT_DETECTION_ERROR',
                'error': str(e)
            })
            raise
    
    async def _classify_elements_ai(self, image: np.ndarray, cv_elements: List[Dict], html_context: Optional[str]) -> List[UIElement]:
        """Classify elements using AI vision model"""
        try:
            # Encode image for AI model
            image_base64 = self._encode_image_for_ai(image)
            
            # Create classification prompt
            classification_prompt = self._create_classification_prompt(cv_elements, html_context)
            
            # Call AI model for classification
            classification_response = await self._call_ai_classification(image_base64, classification_prompt)
            
            # Parse AI response into UIElement objects
            classified_elements = self._parse_ai_classification_response(classification_response, cv_elements)
            
            logger.debug("AI element classification completed", extra={
                'operation': 'AI_ELEMENT_CLASSIFICATION',
                'classified_count': len(classified_elements)
            })
            
            return classified_elements
            
        except Exception as e:
            logger.error("AI element classification failed", extra={
                'operation': 'AI_ELEMENT_CLASSIFICATION_ERROR',
                'error': str(e)
            })
            raise
    
    def _encode_image_for_ai(self, image: np.ndarray) -> str:
        """Encode image for AI model input"""
        try:
            # Convert OpenCV image to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Encode as base64
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            logger.error("Image encoding for AI failed", extra={
                'operation': 'IMAGE_ENCODING_ERROR',
                'error': str(e)
            })
            raise
    
    def _create_classification_prompt(self, cv_elements: List[Dict], html_context: Optional[str]) -> str:
        """Create prompt for AI classification"""
        prompt = """
        Analyze this webpage screenshot and classify the UI elements. For each detected element region,
        identify the type of UI component and its properties.
        
        UI Element Types to identify:
        - Buttons (clickable elements)
        - Input fields (text boxes, forms)
        - Navigation elements (menus, links)
        - Content areas (text blocks, images)
        - Interactive elements (dropdowns, checkboxes)
        - Layout containers (headers, footers, sidebars)
        
        For each element, provide:
        1. Element type from the list above
        2. Confidence score (0.0-1.0)
        3. Descriptive label
        4. Key properties (color, size, interactive state)
        5. Accessibility features present
        
        Format your response as a structured JSON array with detailed element information.
        Be thorough in identifying all visible UI components and their relationships.
        """
        
        if html_context:
            prompt += f"\n\nAdditional HTML context:\n{html_context[:2000]}"
        
        if cv_elements:
            prompt += f"\n\nDetected element regions: {len(cv_elements)} potential UI elements found"
        
        return prompt
    
    async def _call_ai_classification(self, image_base64: str, prompt: str) -> str:
        """Call AI model for element classification"""
        try:
            # Create multimodal input with image and text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        }
                    ]
                }
            ]
            
            # Call the vision model
            response = await self.vision_llm.ainvoke(messages)
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error("AI classification call failed", extra={
                'operation': 'AI_CLASSIFICATION_CALL_ERROR',
                'error': str(e)
            })
            raise
    
    def _parse_ai_classification_response(self, response: str, cv_elements: List[Dict]) -> List[UIElement]:
        """Parse AI classification response into UIElement objects"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                elements_data = json.loads(json_match.group())
            else:
                # Fallback parsing
                elements_data = self._fallback_parse_response(response)
            
            # Convert to UIElement objects
            ui_elements = []
            for i, element_data in enumerate(elements_data):
                # Map to cv_element if available
                cv_element = cv_elements[i] if i < len(cv_elements) else None
                
                ui_element = UIElement(
                    element_id=str(uuid.uuid4()),
                    element_type=self._map_element_type(element_data.get('type', 'unknown')),
                    label=element_data.get('label', 'Unknown Element'),
                    confidence=element_data.get('confidence', 0.5),
                    bbox=cv_element['bbox'] if cv_element else (0, 0, 100, 100),
                    properties=element_data.get('properties', {}),
                    text_content=element_data.get('text_content'),
                    html_attributes=element_data.get('html_attributes', {}),
                    css_properties=element_data.get('css_properties', {})
                )
                
                ui_elements.append(ui_element)
            
            return ui_elements
            
        except Exception as e:
            logger.error("Failed to parse AI classification response", extra={
                'operation': 'AI_RESPONSE_PARSE_ERROR',
                'error': str(e),
                'response': response[:500]
            })
            # Return empty list as fallback
            return []
    
    def _map_element_type(self, type_string: str) -> UIElementType:
        """Map string type to UIElementType enum"""
        type_mapping = {
            'button': UIElementType.BUTTON,
            'input': UIElementType.INPUT_FIELD,
            'input_field': UIElementType.INPUT_FIELD,
            'link': UIElementType.LINK,
            'navigation': UIElementType.NAVIGATION,
            'text': UIElementType.TEXT,
            'heading': UIElementType.HEADING,
            'image': UIElementType.IMAGE,
            'form': UIElementType.FORM,
            'dropdown': UIElementType.DROPDOWN,
            'menu': UIElementType.MENU,
            'header': UIElementType.HEADER,
            'footer': UIElementType.FOOTER,
            'sidebar': UIElementType.SIDEBAR,
            'content': UIElementType.CONTENT_AREA,
            'search': UIElementType.SEARCH_BOX,
            'table': UIElementType.TABLE,
            'list': UIElementType.LIST
        }
        
        return type_mapping.get(type_string.lower(), UIElementType.UNKNOWN)
    
    def _fallback_parse_response(self, response: str) -> List[Dict]:
        """Fallback parsing when JSON extraction fails"""
        # Simple fallback that creates basic element structures
        return [
            {
                'type': 'unknown',
                'label': 'Detected Element',
                'confidence': 0.3,
                'properties': {}
            }
        ]
    
    async def _refine_classifications(self, cv_elements: List[Dict], ai_elements: List[UIElement]) -> List[UIElement]:
        """Refine classifications by combining CV and AI results"""
        # Merge computer vision detection with AI classification
        refined_elements = []
        
        for ai_element in ai_elements:
            # Find matching CV element
            matching_cv = self._find_matching_cv_element(ai_element, cv_elements)
            
            if matching_cv:
                # Update bbox with CV data
                ai_element.bbox = matching_cv['bbox']
                # Add CV properties
                ai_element.properties.update({
                    'area': matching_cv['area'],
                    'aspect_ratio': matching_cv['aspect_ratio'],
                    'detection_confidence': ai_element.confidence * 0.9  # Slight boost for CV confirmation
                })
            
            refined_elements.append(ai_element)
        
        return refined_elements
    
    def _find_matching_cv_element(self, ai_element: UIElement, cv_elements: List[Dict]) -> Optional[Dict]:
        """Find matching computer vision element for AI classification"""
        ai_bbox = ai_element.bbox
        ai_center = ((ai_bbox[0] + ai_bbox[2]) / 2, (ai_bbox[1] + ai_bbox[3]) / 2)
        
        best_match = None
        min_distance = float('inf')
        
        for cv_element in cv_elements:
            cv_bbox = cv_element['bbox']
            cv_center = ((cv_bbox[0] + cv_bbox[2]) / 2, (cv_bbox[1] + cv_bbox[3]) / 2)
            
            # Calculate distance between centers
            distance = ((ai_center[0] - cv_center[0]) ** 2 + (ai_center[1] - cv_center[1]) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_match = cv_element
        
        # Return match if distance is reasonable
        return best_match if min_distance < 100 else None
    
    async def _extract_element_properties(self, image: np.ndarray, elements: List[UIElement], html_context: Optional[str]) -> List[UIElement]:
        """Extract additional properties for UI elements"""
        enhanced_elements = []
        
        for element in elements:
            # Extract color information
            element_colors = self._extract_element_colors(image, element.bbox)
            element.properties.update(element_colors)
            
            # Extract text content using OCR if needed
            if element.element_type in [UIElementType.BUTTON, UIElementType.LINK, UIElementType.TEXT]:
                text_content = await self._extract_text_content(image, element.bbox)
                if text_content:
                    element.text_content = text_content
            
            # Add accessibility analysis
            accessibility_features = await self._analyze_element_accessibility(element, html_context)
            element.accessibility_features.extend(accessibility_features)
            
            enhanced_elements.append(element)
        
        return enhanced_elements
    
    def _extract_element_colors(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, str]:
        """Extract dominant colors from element region"""
        try:
            x1, y1, x2, y2 = bbox
            element_region = image[y1:y2, x1:x2]
            
            # Calculate average color
            avg_color = np.mean(element_region.reshape(-1, 3), axis=0)
            
            # Convert to hex
            hex_color = "#{:02x}{:02x}{:02x}".format(int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
            
            return {
                'dominant_color': hex_color,
                'avg_rgb': avg_color.tolist()
            }
        except Exception:
            return {'dominant_color': '#000000', 'avg_rgb': [0, 0, 0]}
    
    async def _extract_text_content(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Extract text content from element region using OCR"""
        try:
            # Extract element region
            x1, y1, x2, y2 = bbox
            element_region = image[y1:y2, x1:x2]
            
            # Use simple OCR approach (could be enhanced with Tesseract)
            # For now, return None as OCR integration would require additional dependencies
            return None
            
        except Exception as e:
            logger.debug("Text extraction failed", extra={
                'operation': 'TEXT_EXTRACTION_ERROR',
                'error': str(e)
            })
            return None
    
    async def _analyze_element_accessibility(self, element: UIElement, html_context: Optional[str]) -> List[AccessibilityFeature]:
        """Analyze accessibility features for a UI element"""
        features = []
        
        # Basic accessibility analysis based on element type
        if element.element_type == UIElementType.IMAGE:
            # Check for alt text (would need HTML context)
            if html_context and 'alt=' in html_context:
                features.append(AccessibilityFeature.ALT_TEXT)
        
        if element.element_type in [UIElementType.BUTTON, UIElementType.LINK]:
            # Check for descriptive text
            if element.text_content and len(element.text_content.strip()) > 3:
                features.append(AccessibilityFeature.DESCRIPTIVE_LINKS)
        
        if element.element_type == UIElementType.HEADING:
            features.append(AccessibilityFeature.HEADING_STRUCTURE)
        
        if element.element_type == UIElementType.INPUT_FIELD:
            # Check for form labels (would need HTML context)
            if html_context and any(tag in html_context for tag in ['<label', 'aria-label']):
                features.append(AccessibilityFeature.FORM_LABELS)
        
        return features 


class AccessibilityAnalyzer:
    """
    Analyzes accessibility features in webpage screenshots
    
    This class implements comprehensive accessibility assessment capabilities,
    including WCAG compliance analysis, color contrast evaluation, and
    accessibility feature identification.
    """
    
    def __init__(self, config: ScreenshotAnalysisConfig):
        """
        Initialize the accessibility analyzer
        
        Args:
            config (ScreenshotAnalysisConfig): Configuration for accessibility analysis
        """
        self.config = config
        self.accessibility_llm = self._initialize_accessibility_model()
        
        logger.info("Accessibility Analyzer initialized", extra={
            'operation': 'ACCESSIBILITY_ANALYZER_INIT',
            'model': config.accessibility_model,
            'wcag_target': config.wcag_compliance_target,
            'contrast_threshold': config.color_contrast_threshold
        })
    
    def _initialize_accessibility_model(self) -> Runnable:
        """Initialize the accessibility analysis model"""
        try:
            return llm_utils.get_llm(
                model_name=self.config.accessibility_model,
                temperature=self.config.temperature,
                base_url=self.config.openrouter_base_url  # OpenRouter endpoint
            )
        except Exception as e:
            logger.error("Failed to initialize accessibility model", extra={
                'operation': 'ACCESSIBILITY_MODEL_INIT_FAILED',
                'error': str(e)
            })
            raise
    
    @collect_stats("accessibility_analysis")
    async def analyze_accessibility(
        self, 
        screenshot_path: str, 
        ui_elements: List[UIElement],
        html_context: Optional[str] = None
    ) -> AccessibilityAssessment:
        """
        Analyze accessibility features in webpage screenshot
        
        Args:
            screenshot_path (str): Path to screenshot image
            ui_elements (List[UIElement]): Detected UI elements
            html_context (Optional[str]): HTML source for additional context
            
        Returns:
            AccessibilityAssessment: Comprehensive accessibility assessment
        """
        logger.info("Starting accessibility analysis", extra={
            'operation': 'ACCESSIBILITY_ANALYSIS_START',
            'screenshot_path': screenshot_path,
            'element_count': len(ui_elements)
        })
        
        try:
            # Load image for color analysis
            image = Image.open(screenshot_path)
            
            # Analyze color contrast
            color_contrast_results = await self._analyze_color_contrast(image, ui_elements)
            
            # Identify accessibility features
            identified_features = await self._identify_accessibility_features(ui_elements, html_context)
            
            # Analyze keyboard navigation potential
            keyboard_nav_score = await self._analyze_keyboard_navigation(ui_elements)
            
            # Analyze screen reader compatibility
            screen_reader_score = await self._analyze_screen_reader_compatibility(ui_elements, html_context)
            
            # Generate AI-powered accessibility assessment
            ai_assessment = await self._generate_ai_accessibility_assessment(
                screenshot_path, ui_elements, html_context
            )
            
            # Calculate overall accessibility score
            overall_score = self._calculate_overall_accessibility_score(
                color_contrast_results,
                identified_features,
                keyboard_nav_score,
                screen_reader_score,
                ai_assessment
            )
            
            # Determine WCAG compliance level
            wcag_level = self._determine_wcag_compliance_level(overall_score, identified_features)
            
            # Generate recommendations
            recommendations = await self._generate_accessibility_recommendations(
                identified_features,
                color_contrast_results,
                overall_score
            )
            
            # Create assessment result
            assessment = AccessibilityAssessment(
                assessment_id=str(uuid.uuid4()),
                overall_score=overall_score,
                wcag_compliance_level=wcag_level,
                identified_features=identified_features,
                missing_features=self._identify_missing_features(identified_features),
                color_contrast_ratios=color_contrast_results,
                keyboard_navigation_score=keyboard_nav_score,
                screen_reader_compatibility=screen_reader_score,
                recommendations=recommendations
            )
            
            logger.info("Accessibility analysis completed", extra={
                'operation': 'ACCESSIBILITY_ANALYSIS_SUCCESS',
                'overall_score': overall_score,
                'wcag_level': wcag_level,
                'feature_count': len(identified_features)
            })
            
            return assessment
            
        except Exception as e:
            logger.error("Accessibility analysis failed", extra={
                'operation': 'ACCESSIBILITY_ANALYSIS_ERROR',
                'screenshot_path': screenshot_path,
                'error': str(e)
            })
            raise
    
    async def _analyze_color_contrast(self, image: Image.Image, ui_elements: List[UIElement]) -> Dict[str, float]:
        """Analyze color contrast ratios for text elements"""
        contrast_results = {}
        
        try:
            image_array = np.array(image)
            
            for element in ui_elements:
                if element.element_type in [UIElementType.TEXT, UIElementType.BUTTON, UIElementType.LINK]:
                    # Extract element region
                    x1, y1, x2, y2 = element.bbox
                    element_region = image_array[y1:y2, x1:x2]
                    
                    # Calculate contrast ratio
                    contrast_ratio = self._calculate_contrast_ratio(element_region)
                    contrast_results[element.element_id] = contrast_ratio
            
            return contrast_results
            
        except Exception as e:
            logger.debug("Color contrast analysis failed", extra={
                'operation': 'COLOR_CONTRAST_ERROR',
                'error': str(e)
            })
            return {}
    
    def _calculate_contrast_ratio(self, image_region: np.ndarray) -> float:
        """Calculate color contrast ratio for an image region"""
        try:
            # Convert to grayscale and find brightest and darkest pixels
            gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
            min_luminance = np.min(gray) / 255.0
            max_luminance = np.max(gray) / 255.0
            
            # Calculate contrast ratio using WCAG formula
            # Add small epsilon to avoid division by zero
            epsilon = 0.05
            lighter = max(min_luminance, max_luminance) + epsilon
            darker = min(min_luminance, max_luminance) + epsilon
            
            contrast_ratio = lighter / darker
            return min(contrast_ratio, 21.0)  # Cap at maximum possible ratio
            
        except Exception:
            return 1.0  # Fallback to lowest ratio
    
    async def _identify_accessibility_features(
        self, 
        ui_elements: List[UIElement], 
        html_context: Optional[str]
    ) -> List[AccessibilityFeature]:
        """Identify accessibility features present in the interface"""
        identified_features = set()
        
        for element in ui_elements:
            # Add features already identified during UI classification
            identified_features.update(element.accessibility_features)
        
        # Analyze HTML context for additional features
        if html_context:
            html_features = self._analyze_html_accessibility_features(html_context)
            identified_features.update(html_features)
        
        # Analyze overall structure
        structure_features = self._analyze_structural_accessibility(ui_elements)
        identified_features.update(structure_features)
        
        return list(identified_features)
    
    def _analyze_html_accessibility_features(self, html_context: str) -> List[AccessibilityFeature]:
        """Analyze HTML for accessibility features"""
        features = []
        html_lower = html_context.lower()
        
        # Check for various accessibility attributes and elements
        if 'alt=' in html_lower:
            features.append(AccessibilityFeature.ALT_TEXT)
        if 'aria-label' in html_lower:
            features.append(AccessibilityFeature.ARIA_LABEL)
        if any(tag in html_lower for tag in ['<label', 'for=']):
            features.append(AccessibilityFeature.FORM_LABELS)
        if 'lang=' in html_lower:
            features.append(AccessibilityFeature.LANGUAGE_ATTRIBUTES)
        if 'skip' in html_lower and 'link' in html_lower:
            features.append(AccessibilityFeature.SKIP_LINKS)
        if 'tabindex' in html_lower:
            features.append(AccessibilityFeature.KEYBOARD_NAVIGATION)
        
        return features
    
    def _analyze_structural_accessibility(self, ui_elements: List[UIElement]) -> List[AccessibilityFeature]:
        """Analyze structural accessibility features"""
        features = []
        
        # Check for heading structure
        headings = [e for e in ui_elements if e.element_type == UIElementType.HEADING]
        if headings:
            features.append(AccessibilityFeature.HEADING_STRUCTURE)
        
        # Check for navigation elements
        nav_elements = [e for e in ui_elements if e.element_type == UIElementType.NAVIGATION]
        if nav_elements:
            features.append(AccessibilityFeature.KEYBOARD_NAVIGATION)
        
        # Check for form structure
        form_elements = [e for e in ui_elements if e.element_type in [UIElementType.FORM, UIElementType.INPUT_FIELD]]
        if form_elements:
            features.append(AccessibilityFeature.FORM_LABELS)
        
        return features
    
    async def _analyze_keyboard_navigation(self, ui_elements: List[UIElement]) -> float:
        """Analyze keyboard navigation potential"""
        interactive_elements = [
            e for e in ui_elements 
            if e.element_type in [UIElementType.BUTTON, UIElementType.LINK, UIElementType.INPUT_FIELD]
        ]
        
        if not interactive_elements:
            return 0.0
        
        # Score based on presence of interactive elements and navigation structure
        nav_score = 0.0
        
        # Base score for having interactive elements
        nav_score += 0.5
        
        # Additional score for navigation elements
        nav_elements = [e for e in ui_elements if e.element_type == UIElementType.NAVIGATION]
        if nav_elements:
            nav_score += 0.3
        
        # Additional score for skip links (would need HTML analysis)
        nav_score += 0.2  # Placeholder
        
        return min(nav_score, 1.0)
    
    async def _analyze_screen_reader_compatibility(
        self, 
        ui_elements: List[UIElement], 
        html_context: Optional[str]
    ) -> float:
        """Analyze screen reader compatibility"""
        score = 0.0
        
        # Check for text alternatives
        text_elements = [e for e in ui_elements if e.text_content]
        total_elements = len(ui_elements)
        
        if total_elements > 0:
            text_ratio = len(text_elements) / total_elements
            score += text_ratio * 0.4
        
        # Check for semantic structure
        semantic_elements = [
            e for e in ui_elements 
            if e.element_type in [UIElementType.HEADING, UIElementType.NAVIGATION, UIElementType.FORM]
        ]
        if semantic_elements:
            score += 0.3
        
        # Check for ARIA features in HTML
        if html_context and 'aria-' in html_context.lower():
            score += 0.3
        
        return min(score, 1.0)
    
    async def _generate_ai_accessibility_assessment(
        self, 
        screenshot_path: str, 
        ui_elements: List[UIElement],
        html_context: Optional[str]
    ) -> Dict[str, Any]:
        """Generate AI-powered accessibility assessment"""
        try:
            # Encode image
            with open(screenshot_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Create assessment prompt
            prompt = self._create_accessibility_assessment_prompt(ui_elements, html_context)
            
            # Call AI model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        }
                    ]
                }
            ]
            
            response = await self.accessibility_llm.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            return self._parse_ai_accessibility_response(response_text)
            
        except Exception as e:
            logger.error("AI accessibility assessment failed", extra={
                'operation': 'AI_ACCESSIBILITY_ERROR',
                'error': str(e)
            })
            return {}
    
    def _create_accessibility_assessment_prompt(
        self, 
        ui_elements: List[UIElement], 
        html_context: Optional[str]
    ) -> str:
        """Create prompt for AI accessibility assessment"""
        prompt = f"""
        Analyze this webpage screenshot for accessibility features and compliance with WCAG {self.config.wcag_compliance_target} guidelines.
        
        Focus on:
        1. Visual accessibility (color contrast, text size, focus indicators)
        2. Structural accessibility (heading hierarchy, navigation structure)
        3. Interactive accessibility (keyboard navigation, form labels)
        4. Content accessibility (alternative text, descriptive links)
        
        Detected UI elements: {len(ui_elements)} elements including buttons, links, forms, and content areas.
        
        Provide assessment in JSON format with:
        - Overall accessibility score (0.0-1.0)
        - Specific accessibility issues identified
        - WCAG compliance violations
        - Recommendations for improvement
        - Priority level for each issue (high, medium, low)
        
        Be thorough and specific in identifying both strengths and areas for improvement.
        """
        
        if html_context:
            prompt += f"\n\nHTML context available for semantic analysis."
        
        return prompt
    
    def _parse_ai_accessibility_response(self, response: str) -> Dict[str, Any]:
        """Parse AI accessibility assessment response"""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return {
                    'score': 0.7,
                    'issues': [],
                    'recommendations': [
                        'Ensure proper color contrast ratios',
                        'Add alternative text for images',
                        'Implement keyboard navigation support'
                    ]
                }
        except Exception:
            return {}
    
    def _calculate_overall_accessibility_score(
        self,
        color_contrast_results: Dict[str, float],
        identified_features: List[AccessibilityFeature],
        keyboard_nav_score: float,
        screen_reader_score: float,
        ai_assessment: Dict[str, Any]
    ) -> float:
        """Calculate overall accessibility score"""
        scores = []
        
        # Color contrast score
        if color_contrast_results:
            avg_contrast = sum(color_contrast_results.values()) / len(color_contrast_results)
            contrast_score = min(avg_contrast / self.config.color_contrast_threshold, 1.0)
            scores.append(contrast_score * 0.25)
        
        # Feature completeness score
        total_possible_features = len(AccessibilityFeature)
        feature_score = len(identified_features) / total_possible_features
        scores.append(feature_score * 0.25)
        
        # Keyboard navigation score
        scores.append(keyboard_nav_score * 0.25)
        
        # Screen reader compatibility score
        scores.append(screen_reader_score * 0.25)
        
        # AI assessment score (if available)
        if ai_assessment.get('score'):
            scores.append(ai_assessment['score'] * 0.1)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _determine_wcag_compliance_level(
        self, 
        overall_score: float, 
        identified_features: List[AccessibilityFeature]
    ) -> str:
        """Determine WCAG compliance level"""
        # Basic compliance level determination
        essential_features = [
            AccessibilityFeature.ALT_TEXT,
            AccessibilityFeature.COLOR_CONTRAST,
            AccessibilityFeature.KEYBOARD_NAVIGATION,
            AccessibilityFeature.FORM_LABELS
        ]
        
        has_essential_features = any(feature in identified_features for feature in essential_features)
        
        if overall_score >= 0.85 and has_essential_features:
            return "AAA"
        elif overall_score >= 0.70 and has_essential_features:
            return "AA"
        elif overall_score >= 0.50:
            return "A"
        else:
            return "Non-compliant"
    
    def _identify_missing_features(self, identified_features: List[AccessibilityFeature]) -> List[AccessibilityFeature]:
        """Identify missing accessibility features"""
        all_features = list(AccessibilityFeature)
        return [feature for feature in all_features if feature not in identified_features]
    
    async def _generate_accessibility_recommendations(
        self,
        identified_features: List[AccessibilityFeature],
        color_contrast_results: Dict[str, float],
        overall_score: float
    ) -> List[str]:
        """Generate accessibility improvement recommendations"""
        recommendations = []
        
        # Color contrast recommendations
        poor_contrast_elements = [
            element_id for element_id, ratio in color_contrast_results.items()
            if ratio < self.config.color_contrast_threshold
        ]
        if poor_contrast_elements:
            recommendations.append(
                f"Improve color contrast for {len(poor_contrast_elements)} elements to meet WCAG {self.config.wcag_compliance_target} standards"
            )
        
        # Missing feature recommendations
        missing_features = self._identify_missing_features(identified_features)
        if AccessibilityFeature.ALT_TEXT in missing_features:
            recommendations.append("Add alternative text for all images and non-text content")
        
        if AccessibilityFeature.KEYBOARD_NAVIGATION in missing_features:
            recommendations.append("Implement keyboard navigation support for all interactive elements")
        
        if AccessibilityFeature.FORM_LABELS in missing_features:
            recommendations.append("Add proper labels for all form inputs")
        
        if AccessibilityFeature.HEADING_STRUCTURE in missing_features:
            recommendations.append("Implement proper heading hierarchy (h1-h6) for content structure")
        
        # Overall score recommendations
        if overall_score < 0.7:
            recommendations.append("Conduct comprehensive accessibility audit and implement foundational accessibility features")
        
        return recommendations


class WebAgentIntegration:
    """
    Integrates with Alita Web Agent for screenshot capture and webpage context
    
    This class provides the bridge between the Alita Web Agent's browser automation
    capabilities and the KGoT screenshot analysis functionality.
    """
    
    def __init__(self, config: ScreenshotAnalysisConfig):
        """
        Initialize web agent integration
        
        Args:
            config (ScreenshotAnalysisConfig): Configuration including web agent URL
        """
        self.config = config
        self.web_agent_url = config.alita_web_agent_url
        
        logger.info("Web Agent Integration initialized", extra={
            'operation': 'WEB_AGENT_INTEGRATION_INIT',
            'web_agent_url': self.web_agent_url
        })
    
    @collect_stats("web_agent_screenshot")
    async def capture_webpage_screenshot(
        self, 
        url: str, 
        capture_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Capture screenshot of webpage using Alita Web Agent
        
        Args:
            url (str): URL to capture
            capture_options (Optional[Dict]): Additional capture options
            
        Returns:
            Dict[str, Any]: Screenshot capture result with metadata
        """
        logger.info("Capturing webpage screenshot", extra={
            'operation': 'WEB_AGENT_SCREENSHOT_START',
            'url': url
        })
        
        try:
            # Prepare capture request
            capture_request = {
                'url': url,
                'action': 'screenshot',
                'options': capture_options or {}
            }
            
            # Call Alita Web Agent
            response = await self._call_web_agent(capture_request)
            
            # Extract screenshot and metadata
            result = self._process_screenshot_response(response, url)
            
            logger.info("Webpage screenshot captured", extra={
                'operation': 'WEB_AGENT_SCREENSHOT_SUCCESS',
                'url': url,
                'screenshot_path': result.get('screenshot_path')
            })
            
            return result
            
        except Exception as e:
            logger.error("Webpage screenshot capture failed", extra={
                'operation': 'WEB_AGENT_SCREENSHOT_ERROR',
                'url': url,
                'error': str(e)
            })
            raise
    
    async def _call_web_agent(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call Alita Web Agent API"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.web_agent_url}/api/navigate",
                    json=request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"Web agent request failed with status {response.status}")
                        
        except Exception as e:
            logger.error("Web agent API call failed", extra={
                'operation': 'WEB_AGENT_API_ERROR',
                'error': str(e)
            })
            raise
    
    def _process_screenshot_response(self, response: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Process screenshot response from web agent"""
        # Extract screenshot data and save to file
        screenshot_data = response.get('screenshot')
        if screenshot_data:
            # Save screenshot to temporary file
            screenshot_path = self._save_screenshot_data(screenshot_data, url)
        else:
            raise Exception("No screenshot data in response")
        
        return {
            'screenshot_path': screenshot_path,
            'url': url,
            'html_content': response.get('html_content'),
            'page_title': response.get('title'),
            'viewport_size': response.get('viewport_size', (1920, 1080)),
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_screenshot_data(self, screenshot_data: str, url: str) -> str:
        """Save base64 screenshot data to file"""
        try:
            # Create screenshots directory
            screenshots_dir = Path(__file__).parent / 'screenshots'
            screenshots_dir.mkdir(exist_ok=True)
            
            # Generate filename
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            timestamp = int(time.time())
            filename = f"screenshot_{url_hash}_{timestamp}.png"
            screenshot_path = screenshots_dir / filename
            
            # Decode and save
            screenshot_bytes = base64.b64decode(screenshot_data)
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot_bytes)
            
            return str(screenshot_path)
            
        except Exception as e:
            logger.error("Failed to save screenshot data", extra={
                'operation': 'SCREENSHOT_SAVE_ERROR',
                'error': str(e)
            })
            raise 


class KnowledgeGraphIntegration:
    """
    Integrates screenshot analysis results with KGoT Knowledge Graph
    
    This class handles the storage of webpage layout analysis, UI elements,
    and accessibility assessments in the KGoT knowledge graph structure.
    """
    
    def __init__(self, config: ScreenshotAnalysisConfig, graph_store: KnowledgeGraphInterface):
        """
        Initialize knowledge graph integration
        
        Args:
            config (ScreenshotAnalysisConfig): Configuration for screenshot analysis
            graph_store (KnowledgeGraphInterface): KGoT graph store instance
        """
        self.config = config
        self.graph_store = graph_store
        
        logger.info("Knowledge Graph Integration initialized", extra={
            'operation': 'KG_INTEGRATION_INIT',
            'graph_backend': config.graph_backend
        })
    
    @collect_stats("knowledge_graph_storage")
    async def store_screenshot_analysis(
        self,
        layout_structure: LayoutStructure,
        accessibility_assessment: AccessibilityAssessment,
        analysis_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store complete screenshot analysis in knowledge graph
        
        Args:
            layout_structure (LayoutStructure): Webpage layout analysis results
            accessibility_assessment (AccessibilityAssessment): Accessibility analysis results
            analysis_metadata (Dict[str, Any]): Additional analysis metadata
            
        Returns:
            Dict[str, Any]: Storage operation results
        """
        logger.info("Storing screenshot analysis in knowledge graph", extra={
            'operation': 'KG_STORAGE_START',
            'layout_id': layout_structure.layout_id,
            'url': layout_structure.url,
            'element_count': len(layout_structure.elements)
        })
        
        try:
            # Create webpage entity
            webpage_entity_id = await self._create_webpage_entity(layout_structure, analysis_metadata)
            
            # Store UI elements as entities and relationships
            element_entities = await self._store_ui_elements(layout_structure.elements, webpage_entity_id)
            
            # Store spatial relationships
            await self._store_spatial_relationships(layout_structure.spatial_relationships, webpage_entity_id)
            
            # Store accessibility assessment
            accessibility_entity_id = await self._store_accessibility_assessment(
                accessibility_assessment, webpage_entity_id
            )
            
            # Create layout pattern entities
            await self._store_layout_patterns(layout_structure.layout_patterns, webpage_entity_id)
            
            # Store analysis metadata
            await self._store_analysis_metadata(analysis_metadata, webpage_entity_id)
            
            result = {
                'webpage_entity_id': webpage_entity_id,
                'accessibility_entity_id': accessibility_entity_id,
                'element_count': len(element_entities),
                'relationship_count': len(layout_structure.spatial_relationships),
                'storage_timestamp': datetime.now().isoformat()
            }
            
            logger.info("Screenshot analysis stored in knowledge graph", extra={
                'operation': 'KG_STORAGE_SUCCESS',
                'webpage_entity_id': webpage_entity_id,
                'element_count': len(element_entities)
            })
            
            return result
            
        except Exception as e:
            logger.error("Failed to store screenshot analysis in knowledge graph", extra={
                'operation': 'KG_STORAGE_ERROR',
                'layout_id': layout_structure.layout_id,
                'error': str(e)
            })
            raise
    
    async def _create_webpage_entity(
        self, 
        layout_structure: LayoutStructure, 
        metadata: Dict[str, Any]
    ) -> str:
        """Create webpage entity in knowledge graph"""
        webpage_entity = {
            'entity_id': f"webpage_{layout_structure.layout_id}",
            'entity_type': 'webpage',
            'properties': {
                'url': layout_structure.url,
                'title': layout_structure.page_title,
                'viewport_width': layout_structure.viewport_size[0],
                'viewport_height': layout_structure.viewport_size[1],
                'layout_id': layout_structure.layout_id,
                'accessibility_score': layout_structure.accessibility_score,
                'analysis_timestamp': metadata.get('timestamp', datetime.now().isoformat()),
                'color_scheme': layout_structure.color_scheme,
                'typography': layout_structure.typography,
                'responsive_breakpoints': layout_structure.responsive_breakpoints
            }
        }
        
        await self.graph_store.addEntity(webpage_entity)
        return webpage_entity['entity_id']
    
    async def _store_ui_elements(self, ui_elements: List[UIElement], webpage_entity_id: str) -> List[str]:
        """Store UI elements as entities in knowledge graph"""
        element_entity_ids = []
        
        for element in ui_elements:
            # Create UI element entity
            element_entity = {
                'entity_id': f"ui_element_{element.element_id}",
                'entity_type': 'ui_element',
                'properties': {
                    'element_type': element.element_type.value,
                    'label': element.label,
                    'confidence': element.confidence,
                    'bbox_x1': element.bbox[0],
                    'bbox_y1': element.bbox[1],
                    'bbox_x2': element.bbox[2],
                    'bbox_y2': element.bbox[3],
                    'text_content': element.text_content,
                    'html_attributes': element.html_attributes,
                    'css_properties': element.css_properties,
                    'accessibility_features': [f.value for f in element.accessibility_features],
                    **element.properties
                }
            }
            
            await self.graph_store.addEntity(element_entity)
            element_entity_ids.append(element_entity['entity_id'])
            
            # Create relationship between webpage and UI element
            containment_triplet = {
                'subject': webpage_entity_id,
                'predicate': 'contains_ui_element',
                'object': element_entity['entity_id'],
                'metadata': {
                    'element_type': element.element_type.value,
                    'position': f"({element.bbox[0]}, {element.bbox[1]})",
                    'confidence': element.confidence
                }
            }
            
            await self.graph_store.addTriplet(containment_triplet)
        
        return element_entity_ids
    
    async def _store_spatial_relationships(
        self, 
        spatial_relationships: List[SpatialRelationship], 
        webpage_entity_id: str
    ):
        """Store spatial relationships between UI elements"""
        for relationship in spatial_relationships:
            # Create spatial relationship triplet
            spatial_triplet = {
                'subject': f"ui_element_{relationship.subject_id}",
                'predicate': f"spatial_{relationship.predicate}",
                'object': f"ui_element_{relationship.object_id}",
                'metadata': {
                    'confidence': relationship.confidence,
                    'relationship_type': 'spatial',
                    'webpage_id': webpage_entity_id,
                    **relationship.metadata
                }
            }
            
            await self.graph_store.addTriplet(spatial_triplet)
    
    async def _store_accessibility_assessment(
        self, 
        assessment: AccessibilityAssessment, 
        webpage_entity_id: str
    ) -> str:
        """Store accessibility assessment as entity and relationships"""
        # Create accessibility assessment entity
        assessment_entity = {
            'entity_id': f"accessibility_assessment_{assessment.assessment_id}",
            'entity_type': 'accessibility_assessment',
            'properties': {
                'overall_score': assessment.overall_score,
                'wcag_compliance_level': assessment.wcag_compliance_level,
                'keyboard_navigation_score': assessment.keyboard_navigation_score,
                'screen_reader_compatibility': assessment.screen_reader_compatibility,
                'identified_features': [f.value for f in assessment.identified_features],
                'missing_features': [f.value for f in assessment.missing_features],
                'color_contrast_ratios': assessment.color_contrast_ratios,
                'recommendations': assessment.recommendations,
                'assessment_timestamp': datetime.now().isoformat()
            }
        }
        
        await self.graph_store.addEntity(assessment_entity)
        
        # Create relationship between webpage and accessibility assessment
        assessment_triplet = {
            'subject': webpage_entity_id,
            'predicate': 'has_accessibility_assessment',
            'object': assessment_entity['entity_id'],
            'metadata': {
                'overall_score': assessment.overall_score,
                'wcag_level': assessment.wcag_compliance_level
            }
        }
        
        await self.graph_store.addTriplet(assessment_triplet)
        
        # Store individual accessibility issues
        for i, issue in enumerate(assessment.issues):
            issue_entity = {
                'entity_id': f"accessibility_issue_{assessment.assessment_id}_{i}",
                'entity_type': 'accessibility_issue',
                'properties': issue
            }
            
            await self.graph_store.addEntity(issue_entity)
            
            # Link issue to assessment
            issue_triplet = {
                'subject': assessment_entity['entity_id'],
                'predicate': 'has_accessibility_issue',
                'object': issue_entity['entity_id'],
                'metadata': {'issue_priority': issue.get('priority', 'medium')}
            }
            
            await self.graph_store.addTriplet(issue_triplet)
        
        return assessment_entity['entity_id']
    
    async def _store_layout_patterns(self, layout_patterns: List[str], webpage_entity_id: str):
        """Store identified layout patterns"""
        for pattern in layout_patterns:
            pattern_entity = {
                'entity_id': f"layout_pattern_{pattern.lower().replace(' ', '_')}",
                'entity_type': 'layout_pattern',
                'properties': {
                    'pattern_name': pattern,
                    'pattern_type': 'layout'
                }
            }
            
            await self.graph_store.addEntity(pattern_entity)
            
            # Link pattern to webpage
            pattern_triplet = {
                'subject': webpage_entity_id,
                'predicate': 'exhibits_layout_pattern',
                'object': pattern_entity['entity_id'],
                'metadata': {'pattern_name': pattern}
            }
            
            await self.graph_store.addTriplet(pattern_triplet)
    
    async def _store_analysis_metadata(self, metadata: Dict[str, Any], webpage_entity_id: str):
        """Store analysis metadata as properties and relationships"""
        # Create analysis session entity
        analysis_entity = {
            'entity_id': f"analysis_session_{int(time.time())}",
            'entity_type': 'screenshot_analysis_session',
            'properties': {
                'analysis_type': 'kgot_alita_screenshot_analysis',
                'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
                'analyzer_version': metadata.get('version', '1.0.0'),
                'configuration': metadata.get('configuration', {}),
                'performance_metrics': metadata.get('performance_metrics', {})
            }
        }
        
        await self.graph_store.addEntity(analysis_entity)
        
        # Link analysis session to webpage
        analysis_triplet = {
            'subject': webpage_entity_id,
            'predicate': 'analyzed_by_session',
            'object': analysis_entity['entity_id'],
            'metadata': {'analysis_type': 'screenshot_analysis'}
        }
        
        await self.graph_store.addTriplet(analysis_triplet)


class KGoTAlitaScreenshotAnalyzer:
    """
    Main KGoT-Alita Screenshot Analyzer Class
    
    Implementation of Task 27: Implement KGoT-Alita Screenshot Analyzer
    - Integrate KGoT Section 2.3 web navigation with Alita Web Agent screenshot capabilities
    - Design webpage layout analysis feeding KGoT Section 2.1 knowledge graph
    - Implement UI element classification stored as KGoT entities and relationships
    - Add accessibility feature identification with knowledge graph annotation
    
    This class orchestrates the complete screenshot analysis pipeline:
    1. Web agent integration for screenshot capture
    2. UI element classification and spatial analysis
    3. Accessibility assessment and WCAG compliance
    4. Knowledge graph storage of analysis results
    """
    
    def __init__(self, config: Optional[ScreenshotAnalysisConfig] = None):
        """
        Initialize the KGoT-Alita Screenshot Analyzer
        
        Args:
            config (Optional[ScreenshotAnalysisConfig]): Configuration for analysis
        """
        self.config = config or ScreenshotAnalysisConfig()
        
        # Initialize components
        self.ui_classifier = UIElementClassifier(self.config)
        self.accessibility_analyzer = AccessibilityAnalyzer(self.config)
        self.web_agent = WebAgentIntegration(self.config)
        
        # Initialize KGoT graph store
        self.graph_store = self._initialize_graph_store()
        self.kg_integration = KnowledgeGraphIntegration(self.config, self.graph_store)
        
        # Performance tracking
        self.analysis_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'avg_processing_time': 0.0
        }
        
        logger.info("KGoT-Alita Screenshot Analyzer initialized", extra={
            'operation': 'SCREENSHOT_ANALYZER_INIT',
            'config': {
                'vision_model': self.config.vision_model,
                'graph_backend': self.config.graph_backend,
                'accessibility_target': self.config.wcag_compliance_target
            }
        })
    
    def _initialize_graph_store(self) -> KnowledgeGraphInterface:
        """Initialize KGoT graph store"""
        try:
            if self.config.graph_backend == "neo4j" and Neo4jKnowledgeGraph:
                graph_store = Neo4jKnowledgeGraph(self.config.graph_store_config)
            else:
                graph_store = NetworkXKnowledgeGraph(self.config.graph_store_config)
            
            logger.info("Graph store initialized", extra={
                'operation': 'GRAPH_STORE_INIT',
                'backend': self.config.graph_backend
            })
            
            return graph_store
            
        except Exception as e:
            logger.error("Failed to initialize graph store", extra={
                'operation': 'GRAPH_STORE_INIT_ERROR',
                'error': str(e)
            })
            raise
    
    @collect_stats("complete_screenshot_analysis")
    async def analyze_webpage_screenshot(
        self,
        url: str,
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete webpage screenshot analysis pipeline
        
        Args:
            url (str): URL of webpage to analyze
            analysis_options (Optional[Dict]): Additional analysis options
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        
        logger.info("Starting complete webpage screenshot analysis", extra={
            'operation': 'COMPLETE_ANALYSIS_START',
            'analysis_id': analysis_id,
            'url': url
        })
        
        try:
            # Step 1: Capture screenshot using Alita Web Agent
            screenshot_data = await self.web_agent.capture_webpage_screenshot(
                url, analysis_options.get('capture_options') if analysis_options else None
            )
            
            # Step 2: Classify UI elements
            ui_elements = await self.ui_classifier.classify_ui_elements(
                screenshot_data['screenshot_path'],
                screenshot_data.get('html_content')
            )
            
            # Step 3: Analyze spatial relationships
            spatial_relationships = await self._analyze_spatial_relationships(ui_elements)
            
            # Step 4: Perform accessibility analysis
            accessibility_assessment = await self.accessibility_analyzer.analyze_accessibility(
                screenshot_data['screenshot_path'],
                ui_elements,
                screenshot_data.get('html_content')
            )
            
            # Step 5: Create layout structure
            layout_structure = LayoutStructure(
                layout_id=analysis_id,
                page_title=screenshot_data.get('page_title', 'Unknown'),
                url=url,
                viewport_size=screenshot_data.get('viewport_size', (1920, 1080)),
                elements=ui_elements,
                spatial_relationships=spatial_relationships,
                accessibility_score=accessibility_assessment.overall_score,
                layout_patterns=await self._identify_layout_patterns(ui_elements),
                color_scheme=await self._extract_color_scheme(screenshot_data['screenshot_path']),
                typography=await self._analyze_typography(ui_elements)
            )
            
            # Step 6: Store in knowledge graph
            analysis_metadata = {
                'analysis_id': analysis_id,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'configuration': self.config.__dict__,
                'screenshot_data': screenshot_data,
                'performance_metrics': {
                    'processing_time': time.time() - start_time,
                    'ui_element_count': len(ui_elements),
                    'spatial_relationship_count': len(spatial_relationships)
                }
            }
            
            kg_storage_result = await self.kg_integration.store_screenshot_analysis(
                layout_structure,
                accessibility_assessment,
                analysis_metadata
            )
            
            # Update statistics
            self._update_analysis_stats(True, time.time() - start_time)
            
            # Prepare comprehensive result
            result = {
                'analysis_id': analysis_id,
                'url': url,
                'layout_structure': {
                    'layout_id': layout_structure.layout_id,
                    'page_title': layout_structure.page_title,
                    'viewport_size': layout_structure.viewport_size,
                    'element_count': len(layout_structure.elements),
                    'accessibility_score': layout_structure.accessibility_score,
                    'layout_patterns': layout_structure.layout_patterns
                },
                'ui_elements': [
                    {
                        'element_id': elem.element_id,
                        'element_type': elem.element_type.value,
                        'label': elem.label,
                        'confidence': elem.confidence,
                        'bbox': elem.bbox,
                        'accessibility_features': [f.value for f in elem.accessibility_features]
                    }
                    for elem in ui_elements
                ],
                'accessibility_assessment': {
                    'overall_score': accessibility_assessment.overall_score,
                    'wcag_compliance_level': accessibility_assessment.wcag_compliance_level,
                    'identified_features': [f.value for f in accessibility_assessment.identified_features],
                    'recommendations': accessibility_assessment.recommendations
                },
                'spatial_relationships': [
                    {
                        'subject_id': rel.subject_id,
                        'predicate': rel.predicate,
                        'object_id': rel.object_id,
                        'confidence': rel.confidence
                    }
                    for rel in spatial_relationships
                ],
                'knowledge_graph_storage': kg_storage_result,
                'analysis_metadata': analysis_metadata,
                'processing_time': time.time() - start_time
            }
            
            logger.info("Complete webpage screenshot analysis finished", extra={
                'operation': 'COMPLETE_ANALYSIS_SUCCESS',
                'analysis_id': analysis_id,
                'processing_time': time.time() - start_time,
                'element_count': len(ui_elements),
                'accessibility_score': accessibility_assessment.overall_score
            })
            
            return result
            
        except Exception as e:
            self._update_analysis_stats(False, time.time() - start_time)
            
            logger.error("Complete webpage screenshot analysis failed", extra={
                'operation': 'COMPLETE_ANALYSIS_ERROR',
                'analysis_id': analysis_id,
                'url': url,
                'error': str(e)
            })
            raise
    
    async def _analyze_spatial_relationships(self, ui_elements: List[UIElement]) -> List[SpatialRelationship]:
        """Analyze spatial relationships between UI elements"""
        relationships = []
        
        for i, element1 in enumerate(ui_elements):
            for j, element2 in enumerate(ui_elements):
                if i >= j:  # Avoid duplicate pairs and self-comparison
                    continue
                
                relationship = self._determine_spatial_relationship(element1, element2)
                if relationship:
                    relationships.append(relationship)
        
        return relationships
    
    def _determine_spatial_relationship(self, elem1: UIElement, elem2: UIElement) -> Optional[SpatialRelationship]:
        """Determine spatial relationship between two UI elements"""
        bbox1 = elem1.bbox
        bbox2 = elem2.bbox
        
        # Calculate centers and overlaps
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        # Determine relationships
        if center1[0] < center2[0]:
            if center1[1] < center2[1]:
                predicate = "above_left_of"
            elif center1[1] > center2[1]:
                predicate = "below_left_of"
            else:
                predicate = "left_of"
        elif center1[0] > center2[0]:
            if center1[1] < center2[1]:
                predicate = "above_right_of"
            elif center1[1] > center2[1]:
                predicate = "below_right_of"
            else:
                predicate = "right_of"
        else:
            if center1[1] < center2[1]:
                predicate = "above"
            elif center1[1] > center2[1]:
                predicate = "below"
            else:
                return None  # Same position
        
        # Calculate confidence based on distance
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        confidence = max(0.1, min(1.0, 100.0 / distance))
        
        return SpatialRelationship(
            subject_id=elem1.element_id,
            predicate=predicate,
            object_id=elem2.element_id,
            confidence=confidence,
            metadata={
                'distance': distance,
                'elem1_type': elem1.element_type.value,
                'elem2_type': elem2.element_type.value
            }
        )
    
    async def _identify_layout_patterns(self, ui_elements: List[UIElement]) -> List[str]:
        """Identify common layout patterns in the webpage"""
        patterns = []
        
        # Check for header/footer pattern
        header_elements = [e for e in ui_elements if e.element_type == UIElementType.HEADER]
        footer_elements = [e for e in ui_elements if e.element_type == UIElementType.FOOTER]
        
        if header_elements and footer_elements:
            patterns.append("header_footer_layout")
        
        # Check for sidebar pattern
        sidebar_elements = [e for e in ui_elements if e.element_type == UIElementType.SIDEBAR]
        if sidebar_elements:
            patterns.append("sidebar_layout")
        
        # Check for navigation pattern
        nav_elements = [e for e in ui_elements if e.element_type == UIElementType.NAVIGATION]
        if nav_elements:
            patterns.append("navigation_pattern")
        
        # Check for form pattern
        form_elements = [e for e in ui_elements if e.element_type == UIElementType.FORM]
        input_elements = [e for e in ui_elements if e.element_type == UIElementType.INPUT_FIELD]
        
        if form_elements or len(input_elements) > 2:
            patterns.append("form_layout")
        
        # Check for grid pattern (multiple similar elements in organized layout)
        if len(ui_elements) > 10:
            patterns.append("content_grid")
        
        return patterns
    
    async def _extract_color_scheme(self, screenshot_path: str) -> Dict[str, str]:
        """Extract dominant color scheme from screenshot"""
        try:
            image = Image.open(screenshot_path)
            image_array = np.array(image)
            
            # Extract dominant colors using K-means clustering
            pixels = image_array.reshape(-1, 3)
            
            # Simple approach: find most common colors
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            
            # Get top 3 colors
            top_indices = np.argsort(counts)[-3:][::-1]
            dominant_colors = unique_colors[top_indices]
            
            color_scheme = {}
            color_names = ['primary', 'secondary', 'accent']
            
            for i, color in enumerate(dominant_colors):
                hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
                color_scheme[color_names[i]] = hex_color
            
            return color_scheme
            
        except Exception as e:
            logger.debug("Color scheme extraction failed", extra={
                'operation': 'COLOR_SCHEME_ERROR',
                'error': str(e)
            })
            return {}
    
    async def _analyze_typography(self, ui_elements: List[UIElement]) -> Dict[str, str]:
        """Analyze typography patterns in UI elements"""
        typography = {}
        
        # Analyze text elements
        text_elements = [e for e in ui_elements if e.element_type in [UIElementType.TEXT, UIElementType.HEADING]]
        
        if text_elements:
            # Basic typography analysis
            typography['has_headings'] = str(any(e.element_type == UIElementType.HEADING for e in text_elements))
            typography['text_element_count'] = str(len(text_elements))
            
            # Analyze text size variations (would need more detailed text analysis)
            typography['estimated_hierarchy'] = "present" if len(text_elements) > 3 else "simple"
        
        return typography
    
    def _update_analysis_stats(self, success: bool, processing_time: float):
        """Update analysis performance statistics"""
        self.analysis_stats['total_analyses'] += 1
        
        if success:
            self.analysis_stats['successful_analyses'] += 1
        else:
            self.analysis_stats['failed_analyses'] += 1
        
        # Update average processing time
        total_successful = self.analysis_stats['successful_analyses']
        if total_successful > 0:
            current_avg = self.analysis_stats['avg_processing_time']
            self.analysis_stats['avg_processing_time'] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
    
    async def get_analysis_capabilities(self) -> Dict[str, Any]:
        """Get current analysis capabilities and configuration"""
        return {
            'analyzer_version': '1.0.0',
            'supported_features': [
                'ui_element_classification',
                'accessibility_analysis',
                'spatial_relationship_extraction',
                'layout_pattern_identification',
                'knowledge_graph_integration',
                'web_agent_integration'
            ],
            'supported_ui_element_types': [e.value for e in UIElementType],
            'supported_accessibility_features': [f.value for f in AccessibilityFeature],
            'configuration': {
                'vision_model': self.config.vision_model,
                'accessibility_model': self.config.accessibility_model,
                'graph_backend': self.config.graph_backend,
                'wcag_compliance_target': self.config.wcag_compliance_target,
                'confidence_threshold': self.config.confidence_threshold
            },
            'performance_stats': self.analysis_stats,
            'integration_status': {
                'web_agent_url': self.config.alita_web_agent_url,
                'graph_store_initialized': self.graph_store is not None,
                'components_ready': all([
                    self.ui_classifier,
                    self.accessibility_analyzer,
                    self.web_agent,
                    self.kg_integration
                ])
            }
        }


# Factory function for creating analyzer instances
def create_kgot_alita_screenshot_analyzer(config: Optional[ScreenshotAnalysisConfig] = None) -> KGoTAlitaScreenshotAnalyzer:
    """
    Factory function to create KGoT-Alita Screenshot Analyzer
    
    Uses modelsrule configuration by default:
    - o3 for vision tasks (screenshot analysis, UI classification, accessibility)  
    - claude-4-sonnet for web agent tasks (browser automation)
    - gemini-2.5-pro for orchestration (complex reasoning)
    - OpenRouter endpoints for all models
    
    Args:
        config (Optional[ScreenshotAnalysisConfig]): Configuration for the analyzer
        
    Returns:
        KGoTAlitaScreenshotAnalyzer: Configured analyzer instance
    """
    return KGoTAlitaScreenshotAnalyzer(config)


# Example usage and testing
if __name__ == "__main__":
    async def test_screenshot_analyzer():
        """Test the KGoT-Alita Screenshot Analyzer"""
        print("Testing KGoT-Alita Screenshot Analyzer...")
        
        # Create configuration (using modelsrule defaults)
        config = ScreenshotAnalysisConfig(
            # Models are auto-configured per modelsrule:
            # vision_model="openai/o3" 
            # web_agent_model="anthropic/claude-3.5-sonnet"
            # orchestration_model="google/gemini-2.5-pro"
            graph_backend="networkx",
            wcag_compliance_target="AA"
        )
        
        # Create analyzer
        analyzer = create_kgot_alita_screenshot_analyzer(config)
        
        # Get capabilities
        capabilities = await analyzer.get_analysis_capabilities()
        print(f"Analyzer capabilities: {json.dumps(capabilities, indent=2)}")
        
        # Test with a sample URL (would need actual web agent running)
        # result = await analyzer.analyze_webpage_screenshot("https://example.com")
        # print(f"Analysis result: {json.dumps(result, indent=2)}")
        
        print("KGoT-Alita Screenshot Analyzer test completed!")
    
    # Run test
    asyncio.run(test_screenshot_analyzer()) 