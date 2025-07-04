# KGoT-Alita Screenshot Analyzer - Task 27 Implementation

## Overview

The KGoT-Alita Screenshot Analyzer is a comprehensive implementation of Task 27 that integrates KGoT Section 2.3 web navigation capabilities with Alita Web Agent screenshot functionality, providing advanced webpage layout analysis that feeds into the KGoT Section 2.1 knowledge graph.

## Implementation Features

### ✅ Task 27 Requirements Fulfilled

1. **✅ Integrate KGoT Section 2.3 web navigation with Alita Web Agent screenshot capabilities**
   - `WebAgentIntegration` class provides seamless integration with the Alita Web Agent
   - Automated screenshot capture with webpage context extraction
   - Full HTML content analysis alongside visual analysis

2. **✅ Design webpage layout analysis feeding KGoT Section 2.1 knowledge graph**
   - `KnowledgeGraphIntegration` class stores all analysis results in KGoT knowledge graph
   - Comprehensive entity and relationship creation for webpage components
   - Support for both NetworkX and Neo4j backends

3. **✅ Implement UI element classification stored as KGoT entities and relationships**
   - `UIElementClassifier` with 30+ UI element types (buttons, forms, navigation, etc.)
   - Computer vision + AI-powered classification pipeline
   - Spatial relationship extraction and analysis

4. **✅ Add accessibility feature identification with knowledge graph annotation**
   - `AccessibilityAnalyzer` with comprehensive WCAG compliance assessment
   - 18 different accessibility features identified and tracked
   - Color contrast analysis, keyboard navigation scoring, screen reader compatibility

## Architecture

### Core Components

```
KGoTAlitaScreenshotAnalyzer (Main Class)
├── UIElementClassifier
│   ├── Computer Vision Detection (OpenCV)
│   ├── AI-Powered Classification (GPT-4 Vision)
│   └── Element Property Extraction
├── AccessibilityAnalyzer
│   ├── WCAG Compliance Assessment
│   ├── Color Contrast Analysis
│   └── Feature Identification
├── WebAgentIntegration
│   ├── Screenshot Capture via Alita Web Agent
│   └── HTML Context Extraction
└── KnowledgeGraphIntegration
    ├── Entity Creation (Webpage, UI Elements, Assessments)
    └── Relationship Storage (Spatial, Containment, Accessibility)
```

### Key Classes and Enums

- **`UIElementType`**: 30+ UI element classifications
- **`AccessibilityFeature`**: 18 accessibility features tracked
- **`UIElement`**: Represents detected UI components with properties
- **`LayoutStructure`**: Complete webpage layout representation
- **`AccessibilityAssessment`**: Comprehensive accessibility analysis results

## Integration Points

### 1. KGoT Section 2.3 Integration
- Leverages existing KGoT visual analysis tools (`ImageQuestionTool`)
- Uses KGoT utility functions (`llm_utils`, `collect_stats`)
- Integrates with KGoT's multimodal input processing

### 2. Alita Web Agent Integration  
- Seamless integration with existing Playwright-based web automation
- Screenshot capture via HTTP API calls to web agent service
- HTML content extraction for semantic analysis

### 3. KGoT Knowledge Graph Integration
- Direct integration with `KnowledgeGraphInterface`
- Support for multiple graph backends (NetworkX, Neo4j)
- Comprehensive triplet storage for all analysis results

## Usage

### Basic Usage

```python
from multimodal.kgot_alita_screenshot_analyzer import (
    create_kgot_alita_screenshot_analyzer,
    ScreenshotAnalysisConfig
)

# Create configuration
config = ScreenshotAnalysisConfig(
    vision_model="openai/gpt-4-vision-preview",
    graph_backend="networkx",
    wcag_compliance_target="AA"
)

# Initialize analyzer
analyzer = create_kgot_alita_screenshot_analyzer(config)

# Analyze webpage
result = await analyzer.analyze_webpage_screenshot("https://example.com")
```

### Configuration Options

```python
config = ScreenshotAnalysisConfig(
    # Vision Models (using OpenRouter endpoints as per modelsrule)
    vision_model="openai/o3",  # o3 for vision tasks
    ui_classification_model="openai/o3",  # o3 for UI element classification
    accessibility_model="openai/o3",  # o3 for accessibility analysis
    web_agent_model="anthropic/claude-3.5-sonnet",  # claude-4-sonnet for web agent tasks
    orchestration_model="google/gemini-2.5-pro",  # gemini-2.5-pro for orchestration
    openrouter_base_url="https://openrouter.ai/api/v1",  # OpenRouter endpoint
    
    # Graph Storage
    graph_backend="networkx",  # or "neo4j"
    
    # Analysis Settings
    wcag_compliance_target="AA",  # or "AAA"
    color_contrast_threshold=4.5,
    confidence_threshold=0.7,
    
    # Web Agent Integration
    alita_web_agent_url="http://localhost:3001"
)
```

## Analysis Pipeline

1. **Screenshot Capture**: Web Agent captures webpage screenshot and HTML
2. **Computer Vision Detection**: OpenCV-based element detection using edge detection and contours
3. **AI Classification**: GPT-4 Vision classifies detected regions into UI element types
4. **Spatial Analysis**: Calculates spatial relationships between elements
5. **Accessibility Assessment**: Comprehensive WCAG compliance analysis
6. **Knowledge Graph Storage**: All results stored as entities and relationships in KGoT graph

## Knowledge Graph Schema

### Entities Created
- **Webpage**: Main page entity with metadata
- **UI Element**: Individual interface components
- **Accessibility Assessment**: Compliance analysis results
- **Layout Pattern**: Identified design patterns
- **Analysis Session**: Metadata about analysis run

### Relationships Created
- **contains_ui_element**: Webpage → UI Element
- **spatial_[relationship]**: UI Element → UI Element (left_of, above, etc.)
- **has_accessibility_assessment**: Webpage → Accessibility Assessment
- **exhibits_layout_pattern**: Webpage → Layout Pattern
- **analyzed_by_session**: Webpage → Analysis Session

## Accessibility Features Tracked

- Alt Text for Images
- ARIA Labels and Attributes  
- Heading Structure (H1-H6)
- Keyboard Navigation Support
- Color Contrast Ratios
- Focus Indicators
- Screen Reader Compatibility
- Skip Links
- Form Labels
- Language Attributes
- Responsive Design
- And more...

## Performance Features

- **Async/Await**: Full asynchronous operation
- **Batch Processing**: Support for multiple concurrent analyses
- **Caching**: Optional result caching with configurable expiry
- **Statistics Tracking**: Performance metrics and success rates
- **Error Handling**: Comprehensive error handling with detailed logging
- **Winston Logging**: Full integration with existing logging infrastructure

## Dependencies

### Core Dependencies
- **OpenCV**: Computer vision processing
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical operations
- **LangChain**: Agent framework (user preference)
- **aiohttp**: Async HTTP client for web agent communication

### KGoT Dependencies
- **KGoT Tools**: ImageQuestionTool, utility functions
- **KGoT Graph Store**: Knowledge graph interface and implementations
- **KGoT Utilities**: Statistics collection, LLM utilities

### Alita Dependencies  
- **Winston Logging**: Comprehensive logging system
- **Existing Visual Analyzer**: Builds on kgot_visual_analyzer.py

## File Structure

```
alita-kgot-enhanced/multimodal/
├── kgot_alita_screenshot_analyzer.py  # Main implementation
├── screenshots/                       # Captured screenshots directory
└── kgot_visual_analyzer.py           # Base visual analysis (existing)
```

## Future Enhancements

1. **OCR Integration**: Add Tesseract OCR for text extraction
2. **Mobile Responsive Analysis**: Multi-viewport analysis
3. **A/B Testing Support**: Compare multiple webpage versions
4. **Real-time Monitoring**: Continuous accessibility monitoring
5. **Custom Element Training**: Fine-tune models for specific domains

## Testing

The implementation includes comprehensive testing capabilities:

```python
# Run built-in test
python alita-kgot-enhanced/multimodal/kgot_alita_screenshot_analyzer.py
```

## Integration with Sequential Thinking MCP

As requested by the user, complex error resolution and multi-step tasks can leverage the Sequential Thinking MCP for enhanced problem-solving capabilities when dealing with complex accessibility issues or layout analysis challenges.

## Conclusion

This implementation successfully fulfills all requirements of Task 27, providing a comprehensive screenshot analysis system that bridges web navigation capabilities with knowledge graph storage, while maintaining full compatibility with existing KGoT and Alita systems. 