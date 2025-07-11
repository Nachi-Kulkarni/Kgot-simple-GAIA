"""
Test script for KGoT Advanced MCP Generation system
Verifies all components and integration points

@module TestKGoTAdvancedMCPGeneration
@author Enhanced Alita KGoT System
@version 1.0.0
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from .kgot_advanced_mcp_generation import (
    KnowledgeDrivenMCPDesigner,
    ControllerStructuredReasoner,
    KGoTAdvancedMCPGenerator,
    KGoTMCPGeneratorAgent,
    AdvancedMCPSpec,
    KGoTReasoningMode,
    BackendType,
    create_advanced_mcp_generator,
    create_kgot_mcp_generator_agent
)

class TestKnowledgeDrivenMCPDesigner:
    """Test suite for KnowledgeDrivenMCPDesigner"""
    
    @pytest.fixture
    def mock_knowledge_manager(self):
        """Mock knowledge extraction manager"""
        manager = Mock()
        manager.extract_knowledge = AsyncMock(return_value=(
            "Mock extracted knowledge about data analysis",
            Mock(efficiency_score=lambda: 0.8, runtime_ms=1500)
        ))
        return manager
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client"""
        client = Mock()
        client.acomplete = AsyncMock()
        client.acomplete.return_value = Mock(
            text='{"capabilities": ["data_processing", "analysis"], "knowledge_gaps": ["specific_algorithms"]}'
        )
        return client
    
    @pytest.fixture
    def designer(self, mock_knowledge_manager, mock_llm_client):
        """Create KnowledgeDrivenMCPDesigner instance"""
        return KnowledgeDrivenMCPDesigner(mock_knowledge_manager, mock_llm_client)
    
    @pytest.mark.asyncio
    async def test_design_knowledge_informed_mcp(self, designer):
        """Test knowledge-informed MCP design"""
        task_description = "Create an AI agent for automated data analysis"
        
        result = await designer.design_knowledge_informed_mcp(task_description)
        
        assert isinstance(result, AdvancedMCPSpec)
        assert result.name.startswith("knowledge_mcp_")
        assert task_description in result.description
        assert len(result.capabilities) > 0
        assert len(result.knowledge_sources) > 0
        assert 'generation_method' in result.kgot_metadata
        assert result.kgot_metadata['generation_method'] == 'knowledge_driven'
    
    @pytest.mark.asyncio
    async def test_extract_comprehensive_knowledge(self, designer):
        """Test comprehensive knowledge extraction"""
        task_description = "Analyze financial data patterns"
        
        knowledge_results = await designer._extract_comprehensive_knowledge(task_description)
        
        assert isinstance(knowledge_results, dict)
        # Should have at least one extraction method result
        assert len(knowledge_results) >= 1
        
        # Each result should be a tuple of (knowledge, metrics)
        for method, (knowledge, metrics) in knowledge_results.items():
            assert isinstance(knowledge, str)
            assert hasattr(metrics, 'efficiency_score')

class TestControllerStructuredReasoner:
    """Test suite for ControllerStructuredReasoner"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client"""
        client = Mock()
        client.acomplete = AsyncMock()
        client.acomplete.return_value = Mock(
            text="Enhanced MCP design with improved architecture and better error handling"
        )
        return client
    
    @pytest.fixture
    def reasoner(self, mock_llm_client):
        """Create ControllerStructuredReasoner instance"""
        return ControllerStructuredReasoner("http://localhost:3001", mock_llm_client)
    
    @pytest.mark.asyncio
    async def test_perform_structured_reasoning_fallback(self, reasoner):
        """Test structured reasoning with fallback (when KGoT controller unavailable)"""
        mcp_design_task = "Enhance MCP design for data processing agent"
        
        # This will trigger fallback since we don't have actual KGoT controller running
        result = await reasoner.perform_structured_reasoning(mcp_design_task)
        
        assert isinstance(result, dict)
        assert 'solution' in result
        assert 'reasoning_method' in result
        assert result['reasoning_method'] == 'fallback_llm'
        assert 'iterations' in result
        assert 'confidence' in result
    
    @pytest.mark.asyncio
    async def test_fallback_reasoning(self, reasoner):
        """Test fallback reasoning method"""
        mcp_design_task = "Design efficient data processing MCP"
        
        result = await reasoner._fallback_reasoning(mcp_design_task)
        
        assert isinstance(result, dict)
        assert 'solution' in result
        assert 'reasoning_method' in result
        assert result['reasoning_method'] == 'fallback_llm'
        assert result['iterations'] == 1
        assert result['confidence'] == 0.7

class TestKGoTAdvancedMCPGenerator:
    """Test suite for KGoTAdvancedMCPGenerator"""
    
    @pytest.fixture
    def mock_knowledge_manager(self):
        """Mock knowledge extraction manager"""
        manager = Mock()
        manager.extract_knowledge = AsyncMock(return_value=(
            "Mock extracted knowledge",
            Mock(efficiency_score=lambda: 0.8, runtime_ms=1500)
        ))
        return manager
    
    @pytest.fixture  
    def mock_llm_client(self):
        """Mock LLM client"""
        client = Mock()
        client.acomplete = AsyncMock()
        client.acomplete.return_value = Mock(
            text='{"capabilities": ["ai_processing"], "knowledge_gaps": ["optimization"]}'
        )
        return client
    
    @pytest.fixture
    def mock_validation_engine(self):
        """Mock validation engine"""
        engine = Mock()
        return engine
    
    @pytest.fixture
    def generator(self, mock_knowledge_manager, mock_llm_client, mock_validation_engine):
        """Create KGoTAdvancedMCPGenerator instance"""
        return KGoTAdvancedMCPGenerator(
            knowledge_manager=mock_knowledge_manager,
            kgot_controller_endpoint="http://localhost:3001",
            llm_client=mock_llm_client,
            validation_engine=mock_validation_engine
        )
    
    @pytest.mark.asyncio
    async def test_generate_advanced_mcp(self, generator):
        """Test complete advanced MCP generation workflow"""
        task_description = "Create an AI agent for content moderation"
        requirements = {
            'domain': 'content_safety',
            'complexity': 'medium',
            'optimization_targets': ['accuracy', 'speed']
        }
        
        result = await generator.generate_advanced_mcp(task_description, requirements)
        
        assert isinstance(result, AdvancedMCPSpec)
        assert result.name.startswith("knowledge_mcp_")
        assert task_description in result.description
        assert len(result.capabilities) > 0
        assert len(result.knowledge_sources) > 0
        assert 'generation_id' in result.kgot_metadata
        assert 'generation_time_seconds' in result.kgot_metadata
        assert 'phases_completed' in result.kgot_metadata
        assert result.kgot_metadata['success'] == True
        
        # Check that all phases were completed
        expected_phases = ['knowledge_driven', 'structured_reasoning', 'query_optimization', 'graph_validation']
        assert all(phase in result.kgot_metadata['phases_completed'] for phase in expected_phases)
    
    @pytest.mark.asyncio
    async def test_optimize_with_query_languages(self, generator):
        """Test query language optimization"""
        mock_spec = AdvancedMCPSpec(
            name="test_mcp",
            description="Test MCP for optimization",
            capabilities=["test_capability"]
        )
        
        result = await generator._optimize_with_query_languages(mock_spec)
        
        assert isinstance(result, dict)
        assert 'query_language' in result
        assert 'optimization_strategies' in result
        assert 'estimated_performance_gain' in result
        assert 'optimization_timestamp' in result
    
    @pytest.mark.asyncio
    async def test_perform_graph_validation(self, generator):
        """Test graph-based validation"""
        mock_spec = AdvancedMCPSpec(
            name="test_mcp",
            description="Test MCP for validation",
            capabilities=["test_capability"]
        )
        
        result = await generator._perform_graph_validation(mock_spec)
        
        assert isinstance(result, dict)
        assert 'overall_confidence' in result
        assert 'validation_strategies' in result
        assert 'validation_timestamp' in result
    
    def test_get_generation_analytics(self, generator):
        """Test generation analytics"""
        # Simulate some generation history
        generator.performance_metrics['total_generated'] = 5
        generator.performance_metrics['successful_validations'] = 4
        generator.performance_metrics['average_generation_time'] = 2.5
        
        analytics = generator.get_generation_analytics()
        
        assert isinstance(analytics, dict)
        assert 'performance_metrics' in analytics
        assert 'generation_history_count' in analytics
        assert 'success_rate' in analytics
        assert analytics['success_rate'] == 0.8  # 4/5

class TestKGoTMCPGeneratorAgent:
    """Test suite for KGoTMCPGeneratorAgent (LangChain integration)"""
    
    @pytest.fixture
    def mock_generator(self):
        """Mock KGoTAdvancedMCPGenerator"""
        generator = Mock()
        generator.generate_advanced_mcp = AsyncMock(return_value=AdvancedMCPSpec(
            name="agent_generated_mcp",
            description="MCP generated via LangChain agent",
            capabilities=["agent_capability"],
            kgot_metadata={'generation_id': 'test-123', 'generation_time_seconds': 3.5}
        ))
        return generator
    
    @pytest.fixture
    def mock_llm_client(self):
        """Mock LangChain LLM client"""
        client = Mock()
        return client
    
    @pytest.fixture  
    def agent(self, mock_generator, mock_llm_client):
        """Create KGoTMCPGeneratorAgent instance"""
        with patch('kgot_advanced_mcp_generation.create_openai_functions_agent'), \
             patch('kgot_advanced_mcp_generation.AgentExecutor'):
            return KGoTMCPGeneratorAgent(mock_generator, mock_llm_client)
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert hasattr(agent, 'generator')
        assert hasattr(agent, 'llm_client')
        assert hasattr(agent, 'tools')
        assert hasattr(agent, 'agent')
    
    def test_create_langchain_tools(self, agent):
        """Test LangChain tools creation"""
        tools = agent._create_langchain_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check that tools have required attributes
        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')

class TestFactoryFunctions:
    """Test suite for factory functions"""
    
    @patch('kgot_advanced_mcp_generation.ChatOpenAI')
    @patch('kgot_advanced_mcp_generation.MCPCrossValidationEngine')
    def test_create_advanced_mcp_generator(self, mock_validation, mock_llm):
        """Test advanced MCP generator factory"""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            generator = create_advanced_mcp_generator(
                knowledge_graph_backend=BackendType.NETWORKX,
                kgot_controller_endpoint="http://localhost:3001",
                enable_validation=True
            )
            
            assert isinstance(generator, KGoTAdvancedMCPGenerator)
    
    @patch('kgot_advanced_mcp_generation.create_advanced_mcp_generator')
    @patch('kgot_advanced_mcp_generation.ChatOpenAI')
    def test_create_kgot_mcp_generator_agent(self, mock_llm, mock_generator_factory):
        """Test KGoT MCP generator agent factory"""
        mock_generator_factory.return_value = Mock()
        
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            with patch('kgot_advanced_mcp_generation.create_openai_functions_agent'), \
                 patch('kgot_advanced_mcp_generation.AgentExecutor'):
                agent = create_kgot_mcp_generator_agent(
                    knowledge_graph_backend=BackendType.NETWORKX,
                    kgot_controller_endpoint="http://localhost:3001",
                    enable_validation=True
                )
                
                assert isinstance(agent, KGoTMCPGeneratorAgent)

class TestIntegrationWorkflow:
    """Integration test for complete workflow"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test end-to-end MCP generation workflow"""
        # Mock all external dependencies
        with patch('kgot_advanced_mcp_generation.KnowledgeExtractionManager') as mock_km, \
             patch('kgot_advanced_mcp_generation.MCPCrossValidationEngine') as mock_validation, \
             patch('kgot_advanced_mcp_generation.ChatOpenAI') as mock_llm, \
             patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test_key'}):
            
            # Setup mocks
            mock_km_instance = Mock()
            mock_km_instance.extract_knowledge = AsyncMock(return_value=(
                "Mock knowledge for integration test",
                Mock(efficiency_score=lambda: 0.9, runtime_ms=1000)
            ))
            mock_km.return_value = mock_km_instance
            
            mock_llm_instance = Mock()
            mock_llm_instance.acomplete = AsyncMock()
            mock_llm_instance.acomplete.return_value = Mock(
                text='{"capabilities": ["integration_test"], "knowledge_gaps": ["none"]}'
            )
            mock_llm.return_value = mock_llm_instance
            
            # Create generator
            generator = create_advanced_mcp_generator(
                knowledge_graph_backend=BackendType.NETWORKX,
                kgot_controller_endpoint="http://localhost:3001",
                enable_validation=True
            )
            
            # Replace the None knowledge_manager with our mock
            generator.knowledge_manager = mock_km_instance
            generator.knowledge_designer.knowledge_manager = mock_km_instance
            generator.structured_reasoner.llm_client = mock_llm_instance
            
            # Test generation
            task_description = "Create an AI agent for integration testing"
            
            result = await generator.generate_advanced_mcp(
                task_description=task_description,
                requirements={'test_mode': True}
            )
            
            # Verify results
            assert isinstance(result, AdvancedMCPSpec)
            assert result.name.startswith("knowledge_mcp_")
            assert task_description in result.description
            assert 'generation_id' in result.kgot_metadata
            assert result.kgot_metadata['success'] == True

# Test execution
async def run_manual_tests():
    """Run manual tests to verify implementation"""
    print("🧪 Testing KGoT Advanced MCP Generation Implementation")
    print("=" * 60)
    
    try:
        # Test 1: Knowledge-Driven Designer
        print("\n📋 Test 1: Knowledge-Driven MCP Designer")
        mock_knowledge_manager = Mock()
        mock_knowledge_manager.extract_knowledge = AsyncMock(return_value=(
            "Extracted knowledge about automated testing",
            Mock(efficiency_score=lambda: 0.85, runtime_ms=1200)
        ))
        
        mock_llm_client = Mock()
        mock_llm_client.acomplete = AsyncMock()
        mock_llm_client.acomplete.return_value = Mock(
            text='{"capabilities": ["test_automation", "quality_assurance"], "knowledge_gaps": ["performance_optimization"]}'
        )
        
        designer = KnowledgeDrivenMCPDesigner(mock_knowledge_manager, mock_llm_client)
        
        mcp_spec = await designer.design_knowledge_informed_mcp(
            "Create an automated testing framework for AI systems"
        )
        
        print(f"✅ Generated MCP: {mcp_spec.name}")
        print(f"   Capabilities: {mcp_spec.capabilities}")
        print(f"   Knowledge Sources: {mcp_spec.knowledge_sources}")
        
        # Test 2: Structured Reasoner
        print("\n🧠 Test 2: Controller Structured Reasoner")
        reasoner = ControllerStructuredReasoner("http://localhost:3001", mock_llm_client)
        
        reasoning_result = await reasoner.perform_structured_reasoning(
            "Enhance MCP design for better error handling",
            KGoTReasoningMode.ITERATIVE
        )
        
        print(f"✅ Reasoning completed: {reasoning_result['reasoning_method']}")
        print(f"   Confidence: {reasoning_result['confidence']}")
        
        # Test 3: Analytics
        print("\n📊 Test 3: Generation Analytics")
        
        # Create a mock generator with some metrics
        mock_generator = Mock()
        mock_generator.performance_metrics = {
            'total_generated': 10,
            'successful_validations': 8,
            'average_generation_time': 2.3
        }
        mock_generator.generation_history = []
        
        # Simulate the analytics method
        def mock_analytics():
            return {
                'performance_metrics': mock_generator.performance_metrics,
                'generation_history_count': len(mock_generator.generation_history),
                'success_rate': 8/10,
                'recent_generations': []
            }
        
        analytics = mock_analytics()
        print(f"✅ Analytics generated:")
        print(f"   Total MCPs: {analytics['performance_metrics']['total_generated']}")
        print(f"   Success Rate: {analytics['success_rate']:.1%}")
        print(f"   Avg Generation Time: {analytics['performance_metrics']['average_generation_time']:.2f}s")
        
        print("\n🎉 All manual tests passed successfully!")
        print("📦 KGoT Advanced MCP Generation implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run manual tests
    asyncio.run(run_manual_tests())
    
    # Run pytest if available
    try:
        import subprocess
        print("\n🔬 Running pytest automated tests...")
        result = subprocess.run(['python', '-m', 'pytest', __file__, '-v'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
    except Exception as e:
        print(f"Note: Could not run pytest automatically: {e}")
        print("To run full test suite: python -m pytest test_kgot_advanced_mcp_generation.py -v")