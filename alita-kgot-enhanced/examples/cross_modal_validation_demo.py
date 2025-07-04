#!/usr/bin/env python3
"""
KGoT-Alita Cross-Modal Validator Demo with @modelsrule.mdc Models

This demo shows how to use the specialized models:
- o3(vision) for visual analysis and image processing
- claude-4-sonnet(webagent) for reasoning and web agent tasks  
- gemini-2.5-pro(orchestration) for orchestration and coordination

@author: Enhanced Alita KGoT Team
@version: 1.0.0
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from multimodal.kgot_alita_cross_modal_validator import (
    create_kgot_alita_cross_modal_validator,
    CrossModalInput, CrossModalValidationSpec, 
    ModalityType, ValidationLevel
)


async def demo_cross_modal_validation():
    """
    Demonstrate cross-modal validation with @modelsrule.mdc specialized models
    """
    print("🚀 KGoT-Alita Cross-Modal Validator Demo")
    print("=" * 60)
    
    # Configuration using @modelsrule.mdc models
    config = {
        'temperature': 0.3,
        'openrouter_api_key': os.getenv('OPENROUTER_API_KEY', 'your-api-key-here'),
    }
    
    print("📋 Creating validator with specialized models...")
    validator = create_kgot_alita_cross_modal_validator(config)
    
    # Display model assignments
    print("\n🎯 Model Assignments (@modelsrule.mdc):")
    assignments = validator.get_model_assignments()
    for component, model in assignments.items():
        print(f"   {component}: {model}")
    
    # Create diverse cross-modal inputs for testing
    print("\n📝 Creating cross-modal test inputs...")
    inputs = [
        CrossModalInput(
            input_id="text_description",
            modality_type=ModalityType.TEXT,
            content="A red sports car driving on a mountain road with trees on both sides.",
            confidence=0.95,
            metadata={'source': 'user_description', 'language': 'en'}
        ),
        CrossModalInput(
            input_id="image_content", 
            modality_type=ModalityType.IMAGE,
            content="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",  # Base64 image data
            confidence=0.9,
            metadata={'source': 'camera', 'resolution': '1920x1080'}
        ),
        CrossModalInput(
            input_id="structured_data",
            modality_type=ModalityType.STRUCTURED_DATA,
            content={
                "vehicle": {
                    "type": "car",
                    "color": "red", 
                    "category": "sports",
                    "location": "mountain_road"
                },
                "environment": {
                    "terrain": "mountain",
                    "vegetation": "trees",
                    "road_type": "paved"
                }
            },
            confidence=0.88,
            metadata={'source': 'sensor_data', 'format': 'json'}
        ),
        CrossModalInput(
            input_id="analysis_code",
            modality_type=ModalityType.CODE,
            content="""
def analyze_vehicle_scene(image, metadata):
    \"\"\"Analyze vehicle and environment from image data\"\"\"
    vehicle_props = detect_vehicle(image)
    environment = analyze_environment(image)
    
    return {
        'vehicle_type': vehicle_props.get('type', 'unknown'),
        'vehicle_color': vehicle_props.get('color', 'unknown'), 
        'location_type': environment.get('terrain', 'unknown'),
        'road_surface': environment.get('road_type', 'unknown')
    }
            """,
            confidence=0.85,
            metadata={'language': 'python', 'purpose': 'scene_analysis'}
        )
    ]
    
    # Create comprehensive validation specification
    print("\n🔍 Setting up validation specification...")
    validation_spec = CrossModalValidationSpec(
        validation_id="demo_validation_001",
        name="Multi-Modal Vehicle Scene Validation",
        description="Validate consistency between text description, image, structured data, and analysis code",
        inputs=inputs,
        validation_level=ValidationLevel.COMPREHENSIVE,
        expected_consistency=True,
        knowledge_context={
            "domain": "automotive_scene_analysis",
            "task": "vehicle_detection",
            "environment": "outdoor_driving"
        },
        quality_requirements={
            "min_consistency_score": 0.7,
            "max_contradictions": 2,
            "min_confidence": 0.8
        }
    )
    
    print(f"   Validation ID: {validation_spec.validation_id}")
    print(f"   Input Count: {len(validation_spec.inputs)}")
    print(f"   Validation Level: {validation_spec.validation_level.value}")
    print(f"   Expected Consistency: {validation_spec.expected_consistency}")
    
    # Perform comprehensive validation
    print("\n⚡ Running cross-modal validation...")
    print("   This uses all three specialized models:")
    print("   • o3(vision) for image analysis")
    print("   • claude-4-sonnet(webagent) for reasoning")  
    print("   • gemini-2.5-pro(orchestration) for coordination")
    
    try:
        result = await validator.validate_cross_modal_input(validation_spec)
        
        # Display comprehensive results
        print("\n📊 Validation Results:")
        print("=" * 60)
        print(f"🎯 Overall Validation: {'✅ VALID' if result.is_valid else '❌ INVALID'}")
        print(f"📈 Overall Score: {result.overall_score:.3f}")
        print(f"🔍 Contradictions Found: {len(result.contradictions)}")
        print(f"💡 Recommendations: {len(result.recommendations)}")
        
        # Detailed metrics
        print(f"\n📋 Detailed Metrics:")
        metrics = result.metrics
        print(f"   Knowledge Validation Score: {metrics.knowledge_validation_score:.3f}")
        print(f"   Cross-Modal Agreement: {metrics.cross_modal_agreement_score:.3f}")
        print(f"   Overall Confidence: {metrics.overall_confidence:.3f}")
        print(f"   KGoT Quality Score: {metrics.kgot_quality_score:.3f}")
        print(f"   Alita Quality Score: {metrics.alita_quality_score:.3f}")
        print(f"   Unified Quality Score: {metrics.unified_quality_score:.3f}")
        print(f"   Statistical Significance: {'Yes' if metrics.statistical_significance else 'No'}")
        print(f"   Processing Duration: {metrics.validation_duration:.2f}s")
        
        # Consistency scores between modalities
        print(f"\n🔗 Cross-Modal Consistency Scores:")
        for pair_name, consistency in metrics.modality_consistency_scores.items():
            print(f"   {pair_name}:")
            print(f"     Semantic: {consistency.semantic_consistency:.3f}")
            print(f"     Factual: {consistency.factual_consistency:.3f}")
            print(f"     Temporal: {consistency.temporal_consistency:.3f}")
            print(f"     Overall: {consistency.overall_consistency:.3f}")
        
        # Contradiction analysis
        if result.contradictions:
            print(f"\n⚠️  Detected Contradictions:")
            for i, contradiction in enumerate(result.contradictions, 1):
                print(f"   {i}. {contradiction.contradiction_type.value.title()} Contradiction")
                print(f"      Severity: {contradiction.severity.value}")
                print(f"      Description: {contradiction.description}")
                print(f"      Confidence: {contradiction.confidence:.3f}")
                print(f"      Affected Modalities: {[m.value for m in contradiction.affected_modalities]}")
                if contradiction.resolution_suggestions:
                    print(f"      Suggestions: {contradiction.resolution_suggestions[0]}")
                print()
        
        # Recommendations
        if result.recommendations:
            print(f"💡 Validation Recommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Model performance breakdown
        print(f"\n🏃 Model Performance Breakdown:")
        for modality, duration in metrics.processing_time_per_modality.items():
            print(f"   {modality.value}: {duration:.2f}s")
        
        print(f"\n✅ Validation Complete!")
        print(f"🎯 Result: The cross-modal inputs are {'consistent' if result.is_valid else 'inconsistent'}")
        
    except Exception as e:
        print(f"\n❌ Validation Failed: {str(e)}")
        import traceback
        traceback.print_exc()


async def demo_model_specialization():
    """
    Demonstrate model specialization benefits
    """
    print(f"\n🎭 Model Specialization Demo")
    print("=" * 60)
    
    print("The @modelsrule.mdc configuration provides:")
    print()
    print("🔍 o3(vision) - Visual Intelligence:")
    print("   • Advanced image understanding and analysis") 
    print("   • Object detection and spatial relationship recognition")
    print("   • Visual content extraction and description")
    print("   • Optimized for computer vision tasks")
    print()
    print("🤖 claude-4-sonnet(webagent) - Reasoning Excellence:")
    print("   • Advanced reasoning and logical analysis")
    print("   • Consistency checking across modalities")
    print("   • Knowledge validation and factual verification")
    print("   • Contradiction detection and resolution")
    print()
    print("🎯 gemini-2.5-pro(orchestration) - Coordination Master:")
    print("   • Complex workflow orchestration")
    print("   • Statistical analysis and confidence scoring")
    print("   • Multi-component coordination and integration")
    print("   • Comprehensive metrics and reporting")
    print()
    print("This specialization ensures optimal performance for each task type!")


if __name__ == "__main__":
    async def main():
        await demo_model_specialization()
        await demo_cross_modal_validation()
    
    print("🚀 Starting KGoT-Alita Cross-Modal Validator Demo with @modelsrule.mdc")
    asyncio.run(main()) 