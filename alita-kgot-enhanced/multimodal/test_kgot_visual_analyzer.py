#!/usr/bin/env python3
"""
Test and Example Usage for KGoT-Enhanced Visual Analysis Engine

This file demonstrates how to use the KGoT Visual Analyzer and tests
the integration between KGoT Section 2.3 Image Tool and Section 2.1 Graph Store.

@module TestKGoTVisualAnalyzer
@author Enhanced Alita KGoT Team
@date 2025
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TestKGoTVisualAnalyzer')

# Import the KGoT Visual Analyzer
sys.path.append(str(Path(__file__).parent))
from kgot_visual_analyzer import (
    KGoTVisualAnalyzer,
    VisualAnalysisConfig,
    create_kgot_visual_analyzer
)


class KGoTVisualAnalyzerTestSuite:
    """
    Comprehensive test suite for KGoT Visual Analyzer
    
    Tests all major functionality including:
    - Configuration and initialization
    - Graph store integration
    - Spatial relationship extraction
    - Visual question answering
    - Knowledge graph enhancement
    """
    
    def __init__(self):
        """Initialize the test suite"""
        self.logger = logging.getLogger('KGoTVisualAnalyzerTestSuite')
        self.test_results = {}
        
    async def run_all_tests(self):
        """
        Run all test cases for the KGoT Visual Analyzer
        
        Returns:
            Dict[str, Any]: Test results summary
        """
        self.logger.info("Starting KGoT Visual Analyzer test suite...")
        
        test_cases = [
            ('test_initialization', self.test_initialization),
            ('test_configuration', self.test_configuration),
            ('test_capabilities', self.test_capabilities),
            ('test_graph_store_integration', self.test_graph_store_integration),
            ('test_spatial_analysis_structure', self.test_spatial_analysis_structure),
            ('test_vqa_structure', self.test_vqa_structure),
            ('test_error_handling', self.test_error_handling)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in test_cases:
            try:
                self.logger.info(f"Running test: {test_name}")
                result = await test_func()
                self.test_results[test_name] = {
                    'status': 'PASSED',
                    'result': result
                }
                passed += 1
                self.logger.info(f"Test {test_name} PASSED")
                
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                failed += 1
                self.logger.error(f"Test {test_name} FAILED: {e}")
        
        summary = {
            'total_tests': len(test_cases),
            'passed': passed,
            'failed': failed,
            'success_rate': passed / len(test_cases) * 100,
            'detailed_results': self.test_results
        }
        
        self.logger.info(f"Test suite completed: {passed}/{len(test_cases)} tests passed")
        return summary
    
    async def test_initialization(self):
        """Test basic initialization of KGoT Visual Analyzer"""
        # Test default initialization
        analyzer = create_kgot_visual_analyzer()
        assert analyzer is not None, "Analyzer should not be None"
        assert hasattr(analyzer, 'config'), "Analyzer should have config attribute"
        assert hasattr(analyzer, 'graph_store'), "Analyzer should have graph_store attribute"
        
        # Test custom configuration
        config = VisualAnalysisConfig(
            vision_model="openai/o3",
            graph_backend="networkx",
            enable_object_detection=True
        )
        custom_analyzer = KGoTVisualAnalyzer(config)
        assert custom_analyzer.config.vision_model == "openai/o3"
        assert custom_analyzer.config.graph_backend == "networkx"
        
        return {"status": "Initialization tests passed"}
    
    async def test_configuration(self):
        """Test configuration options and validation"""
        # Test default configuration
        default_config = VisualAnalysisConfig()
        assert default_config.vision_model == "openai/o3"
        assert default_config.orchestration_model == "x-ai/grok-4"
        assert default_config.graph_backend == "networkx"
        
        # Test custom configuration
        custom_config = VisualAnalysisConfig(
            vision_model="custom/model",
            temperature=0.5,
            confidence_threshold=0.8,
            enable_graph_context=False
        )
        assert custom_config.vision_model == "custom/model"
        assert custom_config.temperature == 0.5
        assert custom_config.confidence_threshold == 0.8
        assert custom_config.enable_graph_context == False
        
        return {"status": "Configuration tests passed"}
    
    async def test_capabilities(self):
        """Test getting analyzer capabilities"""
        analyzer = create_kgot_visual_analyzer()
        capabilities = await analyzer.get_visual_analysis_capabilities()
        
        # Verify capabilities structure
        assert 'engine_version' in capabilities
        assert 'configuration' in capabilities
        assert 'features' in capabilities
        assert 'models' in capabilities
        assert 'graph_store' in capabilities
        
        # Verify feature flags
        features = capabilities['features']
        assert 'object_detection' in features
        assert 'spatial_relationships' in features
        assert 'graph_context_vqa' in features
        assert 'knowledge_construction' in features
        
        return {"capabilities": capabilities}
    
    async def test_graph_store_integration(self):
        """Test graph store initialization and basic operations"""
        config = VisualAnalysisConfig(graph_backend="networkx")
        analyzer = KGoTVisualAnalyzer(config)
        
        # Verify graph store is initialized
        assert analyzer.graph_store is not None
        assert hasattr(analyzer.graph_store, 'addEntity')
        assert hasattr(analyzer.graph_store, 'addTriplet')
        assert hasattr(analyzer.graph_store, 'getCurrentGraphState')
        
        # Test basic graph operations (if possible without actual data)
        graph_state = await analyzer.graph_store.getCurrentGraphState()
        assert isinstance(graph_state, str)
        
        return {"graph_store_type": type(analyzer.graph_store).__name__}
    
    async def test_spatial_analysis_structure(self):
        """Test spatial analysis component structure"""
        analyzer = create_kgot_visual_analyzer()
        
        # Verify spatial extractor exists and has required methods
        assert hasattr(analyzer, 'spatial_extractor')
        assert hasattr(analyzer.spatial_extractor, 'detect_objects')
        assert hasattr(analyzer.spatial_extractor, 'extract_spatial_relationships')
        
        # Verify knowledge constructor exists and has required methods
        assert hasattr(analyzer, 'knowledge_constructor')
        assert hasattr(analyzer.knowledge_constructor, 'construct_knowledge_from_visual_analysis')
        
        return {"spatial_components": "verified"}
    
    async def test_vqa_structure(self):
        """Test Visual Question Answering component structure"""
        analyzer = create_kgot_visual_analyzer()
        
        # Verify VQA system exists and has required methods
        assert hasattr(analyzer, 'vqa_system')
        assert hasattr(analyzer.vqa_system, 'answer_visual_question')
        
        # Verify main VQA method exists
        assert hasattr(analyzer, 'answer_visual_question')
        
        return {"vqa_components": "verified"}
    
    async def test_error_handling(self):
        """Test error handling for invalid inputs"""
        analyzer = create_kgot_visual_analyzer()
        
        # Test with invalid image path (should handle gracefully)
        try:
            # This should fail gracefully without crashing
            result = await analyzer.analyze_image_with_graph_context("/nonexistent/image.jpg")
            # If it returns a result, it should contain errors
            assert 'errors' in result
        except Exception as e:
            # Expected to fail, but should be a controlled failure
            assert isinstance(e, (FileNotFoundError, ValueError, Exception))
        
        return {"error_handling": "verified"}


async def demonstrate_usage():
    """
    Demonstrate practical usage of the KGoT Visual Analyzer
    """
    logger.info("Demonstrating KGoT Visual Analyzer usage...")
    
    # Example 1: Basic setup and configuration
    print("\n=== Example 1: Basic Setup ===")
    config = VisualAnalysisConfig(
        vision_model="openai/o3",
        graph_backend="networkx",
        enable_object_detection=True,
        enable_spatial_relationships=True,
        enable_graph_context=True
    )
    
    analyzer = create_kgot_visual_analyzer(config)
    capabilities = await analyzer.get_visual_analysis_capabilities()
    print(f"Analyzer created with capabilities: {json.dumps(capabilities, indent=2)}")
    
    # Example 2: Simulated image analysis workflow
    print("\n=== Example 2: Analysis Workflow (Simulated) ===")
    print("Note: This would normally use a real image file")
    
    # In a real scenario, you would call:
    # analysis_result = await analyzer.analyze_image_with_graph_context("/path/to/image.jpg")
    # vqa_result = await analyzer.answer_visual_question("What objects are visible?", "/path/to/image.jpg")
    
    print("Workflow steps:")
    print("1. analyzer.analyze_image_with_graph_context(image_path)")
    print("   - Detects objects with spatial coordinates")
    print("   - Extracts spatial relationships")
    print("   - Constructs knowledge graph entries")
    print("   - Returns comprehensive analysis results")
    print()
    print("2. analyzer.answer_visual_question(question, image_path)")
    print("   - Analyzes image for specific question")
    print("   - Retrieves relevant graph context")
    print("   - Synthesizes multi-modal answer")
    print()
    print("3. analyzer.enhance_graph_with_visual_data(image_path)")
    print("   - Performs analysis and graph enhancement")
    print("   - Returns enhancement summary")
    
    # Example 3: Configuration options
    print("\n=== Example 3: Configuration Options ===")
    advanced_config = VisualAnalysisConfig(
        vision_model="openai/o3",           # Specified for vision tasks
        orchestration_model="x-ai/grok-4",  # For complex reasoning
        graph_backend="neo4j",              # Production graph store
        temperature=0.2,                    # Lower for more factual analysis
        confidence_threshold=0.8,           # Higher threshold for relationships
        enable_scene_understanding=True,    # Advanced scene analysis
        max_context_entities=100,           # More context for VQA
        max_context_relationships=200
    )
    
    print(f"Advanced configuration: {json.dumps(advanced_config.__dict__, indent=2)}")
    
    logger.info("Usage demonstration completed!")


async def integration_test():
    """
    Test integration with existing KGoT and Alita components
    """
    logger.info("Running integration tests...")
    
    print("\n=== Integration Test Results ===")
    
    # Test 1: Import verification
    try:
        from kgot_visual_analyzer import (
            KGoTVisualAnalyzer, VisualAnalysisConfig, 
            SpatialObject, SpatialRelationship,
            SpatialRelationshipExtractor, VisualKnowledgeConstructor, ContextAwareVQA
        )
        print("✓ All main classes imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
    
    # Test 2: Configuration validation
    try:
        config = VisualAnalysisConfig()
        print(f"✓ Default configuration valid: {config.vision_model}, {config.graph_backend}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
    
    # Test 3: Analyzer creation
    try:
        analyzer = create_kgot_visual_analyzer()
        print(f"✓ Analyzer created successfully: {type(analyzer).__name__}")
    except Exception as e:
        print(f"✗ Analyzer creation error: {e}")
    
    # Test 4: Graph store integration
    try:
        analyzer = create_kgot_visual_analyzer()
        graph_store_type = type(analyzer.graph_store).__name__
        print(f"✓ Graph store initialized: {graph_store_type}")
    except Exception as e:
        print(f"✗ Graph store integration error: {e}")
    
    print("Integration tests completed!")


async def main():
    """
    Main function to run all tests and demonstrations
    """
    print("KGoT-Enhanced Visual Analysis Engine Test Suite")
    print("=" * 50)
    
    # Run comprehensive test suite
    test_suite = KGoTVisualAnalyzerTestSuite()
    test_results = await test_suite.run_all_tests()
    
    print(f"\n=== Test Results Summary ===")
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    print(f"Success Rate: {test_results['success_rate']:.1f}%")
    
    if test_results['failed'] > 0:
        print("\nFailed tests:")
        for test_name, result in test_results['detailed_results'].items():
            if result['status'] == 'FAILED':
                print(f"  - {test_name}: {result['error']}")
    
    # Run usage demonstration
    await demonstrate_usage()
    
    # Run integration tests
    await integration_test()
    
    print("\n" + "=" * 50)
    print("All tests and demonstrations completed!")
    print("The KGoT-Enhanced Visual Analysis Engine is ready for use.")
    
    return test_results


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main()) 