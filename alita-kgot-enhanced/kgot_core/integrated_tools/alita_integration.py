#!/usr/bin/env python3
"""
Alita Integration Module for KGoT Tools

This module provides seamless integration between KGoT tools and Alita's web agent
navigation capabilities. It implements the bridge layer that allows KGoT tools to
leverage Alita's enhanced web browsing and navigation features.

Key Features:
- Integration with Alita Web Agent's navigation capabilities
- Enhanced web browsing with Alita's intelligent navigation
- Coordinated task execution between KGoT and Alita systems
- Tool result enrichment with Alita's context understanding
- Session management for web-based tool operations

@module AlitaIntegration
@author AI Assistant
@date 2025
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import requests
import websockets

# Setup logging with Winston-compatible structure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AlitaIntegration')

@dataclass
class AlitaWebAgentConfig:
    """Configuration for Alita Web Agent integration"""
    agent_endpoint: str = "http://localhost:8000/api/web-agent"
    websocket_endpoint: str = "ws://localhost:8000/ws/web-agent"
    timeout: int = 30
    max_retries: int = 3
    session_timeout: int = 300  # 5 minutes
    enable_intelligent_navigation: bool = True
    enable_context_enrichment: bool = True

@dataclass
class NavigationContext:
    """Context information for web navigation operations"""
    current_url: str = ""
    page_title: str = ""
    page_content_summary: str = ""
    navigation_history: List[str] = None
    user_intent: str = ""
    search_context: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.navigation_history is None:
            self.navigation_history = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AlitaWebAgentBridge:
    """
    Bridge class for integrating KGoT tools with Alita Web Agent
    
    This class provides methods to enhance tool operations with Alita's
    intelligent web navigation and context understanding capabilities.
    """
    
    def __init__(self, config: Optional[AlitaWebAgentConfig] = None):
        """
        Initialize the Alita Web Agent Bridge
        
        Args:
            config: Configuration for Alita Web Agent integration
        """
        self.config = config or AlitaWebAgentConfig()
        self.session_id: Optional[str] = None
        self.navigation_context = NavigationContext()
        self.active_connections: Dict[str, Any] = {}
        
        logger.info("Initializing Alita Web Agent Bridge", extra={
            'operation': 'ALITA_BRIDGE_INIT',
            'config': self.config.__dict__
        })
    
    async def initialize_session(self, user_intent: str = "") -> str:
        """
        Initialize a new session with Alita Web Agent
        
        Args:
            user_intent: The user's intent or goal for the session
            
        Returns:
            Session ID for the initialized session
        """
        try:
            logger.info("Initializing Alita Web Agent session", extra={
                'operation': 'SESSION_INIT',
                'user_intent': user_intent
            })
            
            session_data = {
                'user_intent': user_intent,
                'timestamp': datetime.now().isoformat(),
                'capabilities': {
                    'intelligent_navigation': self.config.enable_intelligent_navigation,
                    'context_enrichment': self.config.enable_context_enrichment
                }
            }
            
            response = requests.post(
                f"{self.config.agent_endpoint}/session/create",
                json=session_data,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.session_id = result.get('session_id')
                self.navigation_context.user_intent = user_intent
                
                logger.info("Alita Web Agent session initialized", extra={
                    'operation': 'SESSION_INIT_SUCCESS',
                    'session_id': self.session_id
                })
                
                return self.session_id
            else:
                raise Exception(f"Session initialization failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Alita Web Agent session: {str(e)}", extra={
                'operation': 'SESSION_INIT_FAILED',
                'error': str(e)
            })
            # Fallback to local session ID
            self.session_id = f"local_session_{datetime.now().timestamp()}"
            return self.session_id
    
    async def enhance_web_search(self, search_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance web search operations with Alita's intelligent navigation
        
        Args:
            search_query: The search query to enhance
            context: Additional context for the search operation
            
        Returns:
            Enhanced search results with Alita's context understanding
        """
        if not self.session_id:
            await self.initialize_session(f"Web search: {search_query}")
        
        try:
            logger.info("Enhancing web search with Alita", extra={
                'operation': 'WEB_SEARCH_ENHANCE',
                'query': search_query,
                'session_id': self.session_id
            })
            
            enhancement_request = {
                'session_id': self.session_id,
                'search_query': search_query,
                'context': context or {},
                'navigation_context': {
                    'current_url': self.navigation_context.current_url,
                    'user_intent': self.navigation_context.user_intent,
                    'search_context': self.navigation_context.search_context
                },
                'enhancement_options': {
                    'intelligent_navigation': self.config.enable_intelligent_navigation,
                    'context_enrichment': self.config.enable_context_enrichment
                }
            }
            
            response = requests.post(
                f"{self.config.agent_endpoint}/search/enhance",
                json=enhancement_request,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                enhanced_results = response.json()
                
                # Update navigation context
                self.navigation_context.search_context = search_query
                if 'navigation_updates' in enhanced_results:
                    nav_updates = enhanced_results['navigation_updates']
                    self.navigation_context.current_url = nav_updates.get('current_url', self.navigation_context.current_url)
                    self.navigation_context.page_title = nav_updates.get('page_title', self.navigation_context.page_title)
                
                logger.info("Web search enhancement completed", extra={
                    'operation': 'WEB_SEARCH_ENHANCE_SUCCESS',
                    'enhanced_results_count': len(enhanced_results.get('enhanced_results', []))
                })
                
                return enhanced_results
            else:
                logger.warning(f"Search enhancement failed, using fallback: {response.text}")
                return {'enhanced_results': [], 'fallback': True}
                
        except Exception as e:
            logger.error(f"Web search enhancement failed: {str(e)}", extra={
                'operation': 'WEB_SEARCH_ENHANCE_FAILED',
                'error': str(e)
            })
            return {'enhanced_results': [], 'error': str(e)}
    
    async def enhance_page_navigation(self, target_url: str, navigation_intent: str = "") -> Dict[str, Any]:
        """
        Enhance page navigation with Alita's intelligent navigation capabilities
        
        Args:
            target_url: URL to navigate to
            navigation_intent: Intent or purpose of the navigation
            
        Returns:
            Enhanced navigation results with context
        """
        if not self.session_id:
            await self.initialize_session(f"Navigate to: {target_url}")
        
        try:
            logger.info("Enhancing page navigation with Alita", extra={
                'operation': 'PAGE_NAVIGATION_ENHANCE',
                'target_url': target_url,
                'navigation_intent': navigation_intent
            })
            
            navigation_request = {
                'session_id': self.session_id,
                'target_url': target_url,
                'navigation_intent': navigation_intent,
                'current_context': {
                    'current_url': self.navigation_context.current_url,
                    'user_intent': self.navigation_context.user_intent,
                    'navigation_history': self.navigation_context.navigation_history[-10:]  # Last 10 URLs
                },
                'enhancement_options': {
                    'intelligent_navigation': self.config.enable_intelligent_navigation,
                    'context_enrichment': self.config.enable_context_enrichment
                }
            }
            
            response = requests.post(
                f"{self.config.agent_endpoint}/navigation/enhance",
                json=navigation_request,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                navigation_results = response.json()
                
                # Update navigation context
                self.navigation_context.current_url = target_url
                self.navigation_context.navigation_history.append(target_url)
                if 'page_analysis' in navigation_results:
                    page_analysis = navigation_results['page_analysis']
                    self.navigation_context.page_title = page_analysis.get('title', '')
                    self.navigation_context.page_content_summary = page_analysis.get('content_summary', '')
                
                logger.info("Page navigation enhancement completed", extra={
                    'operation': 'PAGE_NAVIGATION_ENHANCE_SUCCESS',
                    'page_title': self.navigation_context.page_title
                })
                
                return navigation_results
            else:
                logger.warning(f"Navigation enhancement failed, using fallback: {response.text}")
                return {'navigation_success': False, 'fallback': True}
                
        except Exception as e:
            logger.error(f"Page navigation enhancement failed: {str(e)}", extra={
                'operation': 'PAGE_NAVIGATION_ENHANCE_FAILED',
                'error': str(e)
            })
            return {'navigation_success': False, 'error': str(e)}
    
    async def enrich_tool_results(self, tool_name: str, tool_results: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enrich tool results with Alita's context understanding
        
        Args:
            tool_name: Name of the tool that generated the results
            tool_results: Original tool results
            context: Additional context for enrichment
            
        Returns:
            Enriched results with Alita's context understanding
        """
        if not self.session_id:
            await self.initialize_session(f"Tool enrichment for: {tool_name}")
        
        try:
            logger.info("Enriching tool results with Alita context", extra={
                'operation': 'TOOL_RESULTS_ENRICH',
                'tool_name': tool_name,
                'session_id': self.session_id
            })
            
            enrichment_request = {
                'session_id': self.session_id,
                'tool_name': tool_name,
                'tool_results': tool_results,
                'context': context or {},
                'navigation_context': {
                    'current_url': self.navigation_context.current_url,
                    'page_title': self.navigation_context.page_title,
                    'page_content_summary': self.navigation_context.page_content_summary,
                    'user_intent': self.navigation_context.user_intent
                },
                'enrichment_options': {
                    'context_enrichment': self.config.enable_context_enrichment
                }
            }
            
            response = requests.post(
                f"{self.config.agent_endpoint}/results/enrich",
                json=enrichment_request,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                enriched_results = response.json()
                
                logger.info("Tool results enrichment completed", extra={
                    'operation': 'TOOL_RESULTS_ENRICH_SUCCESS',
                    'enrichment_type': enriched_results.get('enrichment_type', 'unknown')
                })
                
                return enriched_results
            else:
                logger.warning(f"Results enrichment failed, returning original: {response.text}")
                return {'enriched_results': tool_results, 'fallback': True}
                
        except Exception as e:
            logger.error(f"Tool results enrichment failed: {str(e)}", extra={
                'operation': 'TOOL_RESULTS_ENRICH_FAILED',
                'error': str(e)
            })
            return {'enriched_results': tool_results, 'error': str(e)}
    
    async def coordinate_task_execution(self, task_description: str, tools_involved: List[str]) -> Dict[str, Any]:
        """
        Coordinate task execution between KGoT and Alita systems
        
        Args:
            task_description: Description of the task to coordinate
            tools_involved: List of tools involved in the task
            
        Returns:
            Coordination plan and execution strategy
        """
        if not self.session_id:
            await self.initialize_session(f"Task coordination: {task_description}")
        
        try:
            logger.info("Coordinating task execution with Alita", extra={
                'operation': 'TASK_COORDINATION',
                'task_description': task_description,
                'tools_involved': tools_involved
            })
            
            coordination_request = {
                'session_id': self.session_id,
                'task_description': task_description,
                'tools_involved': tools_involved,
                'current_context': {
                    'navigation_context': self.navigation_context.__dict__,
                    'user_intent': self.navigation_context.user_intent
                },
                'coordination_options': {
                    'intelligent_navigation': self.config.enable_intelligent_navigation,
                    'context_enrichment': self.config.enable_context_enrichment
                }
            }
            
            response = requests.post(
                f"{self.config.agent_endpoint}/coordination/plan",
                json=coordination_request,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                coordination_plan = response.json()
                
                logger.info("Task coordination plan generated", extra={
                    'operation': 'TASK_COORDINATION_SUCCESS',
                    'plan_steps': len(coordination_plan.get('execution_steps', []))
                })
                
                return coordination_plan
            else:
                logger.warning(f"Task coordination failed, using fallback: {response.text}")
                return {'execution_steps': [], 'fallback': True}
                
        except Exception as e:
            logger.error(f"Task coordination failed: {str(e)}", extra={
                'operation': 'TASK_COORDINATION_FAILED',
                'error': str(e)
            })
            return {'execution_steps': [], 'error': str(e)}
    
    async def close_session(self) -> bool:
        """
        Close the current Alita Web Agent session
        
        Returns:
            True if session was closed successfully, False otherwise
        """
        if not self.session_id:
            return True
        
        try:
            logger.info("Closing Alita Web Agent session", extra={
                'operation': 'SESSION_CLOSE',
                'session_id': self.session_id
            })
            
            response = requests.post(
                f"{self.config.agent_endpoint}/session/close",
                json={'session_id': self.session_id},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                logger.info("Alita Web Agent session closed", extra={
                    'operation': 'SESSION_CLOSE_SUCCESS',
                    'session_id': self.session_id
                })
                self.session_id = None
                self.navigation_context = NavigationContext()
                return True
            else:
                logger.warning(f"Session close failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to close Alita Web Agent session: {str(e)}", extra={
                'operation': 'SESSION_CLOSE_FAILED',
                'error': str(e)
            })
            return False
    
    def get_navigation_context(self) -> NavigationContext:
        """
        Get the current navigation context
        
        Returns:
            Current navigation context information
        """
        return self.navigation_context
    
    def update_navigation_context(self, updates: Dict[str, Any]) -> None:
        """
        Update the navigation context with new information
        
        Args:
            updates: Dictionary of context updates
        """
        for key, value in updates.items():
            if hasattr(self.navigation_context, key):
                setattr(self.navigation_context, key, value)
        
        logger.info("Navigation context updated", extra={
            'operation': 'NAVIGATION_CONTEXT_UPDATE',
            'updates': updates
        })

class AlitaToolIntegrator:
    """
    Main integrator class that enhances KGoT tools with Alita capabilities
    """
    
    def __init__(self, web_agent_config: Optional[AlitaWebAgentConfig] = None):
        """
        Initialize the Alita Tool Integrator
        
        Args:
            web_agent_config: Configuration for web agent integration
        """
        self.web_agent_bridge = AlitaWebAgentBridge(web_agent_config)
        self.tool_enhancers: Dict[str, Callable] = {}
        self.active_sessions: Dict[str, str] = {}
        
        logger.info("Alita Tool Integrator initialized", extra={
            'operation': 'TOOL_INTEGRATOR_INIT'
        })
        
        # Register default tool enhancers
        self._register_default_enhancers()
    
    def _register_default_enhancers(self) -> None:
        """Register default tool enhancement functions"""
        self.tool_enhancers.update({
            'ask_search_agent': self._enhance_search_tool,
            'image_inspector': self._enhance_image_tool,
            'llm_query': self._enhance_llm_tool,
            'Python_Code_Executor': self._enhance_python_tool,
        })
    
    async def _enhance_search_tool(self, tool_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance search tool operations with Alita navigation"""
        search_query = tool_input.get('query', '')
        enhanced_results = await self.web_agent_bridge.enhance_web_search(search_query, context)
        return enhanced_results
    
    async def _enhance_image_tool(self, tool_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance image tool operations with Alita context"""
        # For image tools, provide navigation context for better understanding
        navigation_context = self.web_agent_bridge.get_navigation_context()
        enhanced_context = {
            **context,
            'page_context': {
                'current_url': navigation_context.current_url,
                'page_title': navigation_context.page_title,
                'user_intent': navigation_context.user_intent
            }
        }
        return {'enhanced_context': enhanced_context}
    
    async def _enhance_llm_tool(self, tool_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance LLM tool operations with Alita context enrichment"""
        query = tool_input.get('query', '')
        navigation_context = self.web_agent_bridge.get_navigation_context()
        
        # Enrich the query with navigation context
        enriched_query = f"""
        Context: {navigation_context.user_intent}
        Current Page: {navigation_context.page_title}
        Page Summary: {navigation_context.page_content_summary}
        
        Query: {query}
        """
        
        return {'enriched_query': enriched_query, 'original_query': query}
    
    async def _enhance_python_tool(self, tool_input: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance Python tool operations with Alita coordination"""
        # For Python tools, provide coordination for task execution
        code = tool_input.get('code', '')
        coordination_plan = await self.web_agent_bridge.coordinate_task_execution(
            f"Execute Python code: {code[:100]}...",
            ['Python_Code_Executor']
        )
        return {'coordination_plan': coordination_plan}
    
    async def enhance_tool_execution(self, tool_name: str, tool_input: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance tool execution with Alita capabilities
        
        Args:
            tool_name: Name of the tool being executed
            tool_input: Input parameters for the tool
            context: Additional context for enhancement
            
        Returns:
            Enhanced tool execution parameters and context
        """
        context = context or {}
        
        try:
            logger.info("Enhancing tool execution with Alita", extra={
                'operation': 'TOOL_EXECUTION_ENHANCE',
                'tool_name': tool_name
            })
            
            # Apply tool-specific enhancements
            if tool_name in self.tool_enhancers:
                enhancer = self.tool_enhancers[tool_name]
                enhancement_results = await enhancer(tool_input, context)
                
                logger.info("Tool execution enhancement completed", extra={
                    'operation': 'TOOL_EXECUTION_ENHANCE_SUCCESS',
                    'tool_name': tool_name
                })
                
                return {
                    'enhanced_input': tool_input,
                    'enhanced_context': context,
                    'alita_enhancements': enhancement_results
                }
            else:
                # Generic enhancement for unregistered tools
                navigation_context = self.web_agent_bridge.get_navigation_context()
                return {
                    'enhanced_input': tool_input,
                    'enhanced_context': {
                        **context,
                        'navigation_context': navigation_context.__dict__
                    },
                    'alita_enhancements': {'generic_enhancement': True}
                }
                
        except Exception as e:
            logger.error(f"Tool execution enhancement failed: {str(e)}", extra={
                'operation': 'TOOL_EXECUTION_ENHANCE_FAILED',
                'tool_name': tool_name,
                'error': str(e)
            })
            return {
                'enhanced_input': tool_input,
                'enhanced_context': context,
                'alita_enhancements': {'error': str(e)}
            }
    
    async def enhance_tool_results(self, tool_name: str, tool_results: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance tool results with Alita context understanding
        
        Args:
            tool_name: Name of the tool that generated results
            tool_results: Original tool results
            context: Additional context for enhancement
            
        Returns:
            Enhanced tool results
        """
        return await self.web_agent_bridge.enrich_tool_results(tool_name, tool_results, context)
    
    async def initialize_session(self, user_intent: str = "") -> str:
        """Initialize a new Alita integration session"""
        return await self.web_agent_bridge.initialize_session(user_intent)
    
    async def close_session(self) -> bool:
        """Close the current Alita integration session"""
        return await self.web_agent_bridge.close_session()


# Factory function for creating Alita Tool Integrator
def create_alita_integrator(config: Optional[AlitaWebAgentConfig] = None) -> AlitaToolIntegrator:
    """
    Factory function to create an Alita Tool Integrator
    
    Args:
        config: Configuration for Alita web agent integration
        
    Returns:
        Configured AlitaToolIntegrator instance
    """
    logger.info("Creating Alita Tool Integrator", extra={
        'operation': 'INTEGRATOR_FACTORY_CREATE'
    })
    
    return AlitaToolIntegrator(config)

if __name__ == "__main__":
    """Test script for Alita Integration"""
    import asyncio
    
    async def test_alita_integration():
        print("Testing Alita Integration...")
        
        try:
            # Create integrator
            integrator = create_alita_integrator()
            
            # Initialize session
            session_id = await integrator.initialize_session("Testing Alita integration capabilities")
            print(f"Session initialized: {session_id}")
            
            # Test tool enhancement
            search_enhancement = await integrator.enhance_tool_execution(
                'ask_search_agent',
                {'query': 'Test search query'},
                {'test_context': True}
            )
            print(f"Search tool enhancement: {search_enhancement}")
            
            # Close session
            await integrator.close_session()
            print("Session closed successfully")
            
            print("Alita Integration test completed successfully!")
            
        except Exception as e:
            print(f"Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run the test
    asyncio.run(test_alita_integration()) 
