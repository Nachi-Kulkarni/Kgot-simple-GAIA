#!/usr/bin/env python3
"""
KGoT Integrated Tools Module

Implementation of the complete Integrated Tools suite as specified in KGoT research paper Section 2.3
with Alita integration and specific AI model configurations.

This module provides:
- Python Code Tool for dynamic script generation and execution
- LLM Tool with 2.5 Pro (1m context) for auxiliary language model integration  
- Image Tool with OpenAI o3 for multimodal inputs using Vision models
- Surfer Agent with Claude 4 Sonnet for web interaction
- Wikipedia Tool with granular navigation capabilities
- ExtractZip Tool and Text Inspector Tool for content conversion
- Integration with Alita Web Agent's navigation capabilities
- Connection to KGoT Controller's tool orchestration system

@module IntegratedTools
@author AI Assistant  
@date 2025
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

# Module exports
__all__ = [
    'IntegratedToolsManager',
    'AlitaIntegrator', 
    'KGoTToolBridge',
    'create_integrated_tools_manager',
    'create_alita_integrator'
]

import logging

# Setup logging
logger = logging.getLogger('IntegratedTools')
logger.info("KGoT Integrated Tools module initialized")

# Import main classes (with fallback for missing dependencies)
try:
    from .integrated_tools_manager import AlitaIntegratedToolsManager as IntegratedToolsManager
    from .integrated_tools_manager import create_integrated_tools_manager
    logger.info("Integrated Tools Manager imported successfully")
except ImportError as e:
    logger.warning(f"Could not import Integrated Tools Manager: {e}")
    IntegratedToolsManager = None
    create_integrated_tools_manager = None

try:
    from .alita_integration import AlitaToolIntegrator as AlitaIntegrator
    from .alita_integration import create_alita_integrator  
    logger.info("Alita Integration imported successfully")
except ImportError as e:
    logger.warning(f"Could not import Alita Integration: {e}")
    AlitaIntegrator = None
    create_alita_integrator = None

try:
    from .kgot_tool_bridge import KGoTToolBridge
    logger.info("KGoT Tool Bridge imported successfully")
except ImportError as e:
    logger.warning(f"Could not import KGoT Tool Bridge: {e}")
    KGoTToolBridge = None 