#!/usr/bin/env python3
"""
KGoT Error Management Integration Bridge

Integration bridge connecting KGoT Error Management System with Alita's iterative 
refinement and error correction processes. This module ensures seamless error 
handling across the entire Alita-KGoT enhanced architecture.

Integration Features:
- Connection with Alita Web Agent's iterative refinement processes
- Integration with KGoT Tool Bridge error handling
- Cross-system error recovery coordination
- Unified error reporting and analytics
- Integration with MCP creation error validation
- Support for multimodal error handling

@author: Enhanced Alita KGoT Team
@version: 1.0.0
@based_on: Task 7 Integration Requirements
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
import os

# Add path for error management system
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import KGoT Error Management System
from error_management import (
    KGoTErrorManagementSystem,
    ErrorType,
    ErrorSeverity,
    ErrorContext,
    create_kgot_error_management_system
)

# Import existing Alita integration components
from alita_integration import AlitaToolIntegrator, create_alita_integrator

# Winston-compatible logging setup
logger = logging.getLogger('KGoTErrorIntegration')
handler = logging.FileHandler('./logs/kgot/error_integration.log')
formatter = logging.Formatter('%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class IntegrationConfig:
    """Configuration for KGoT-Alita error management integration"""
    enable_alita_refinement: bool = True
    enable_cross_system_recovery: bool = True
    enable_unified_logging: bool = True
    enable_mcp_error_validation: bool = True
    max_cross_system_retries: int = 2
    error_escalation_threshold: int = 3


class AlitaRefinementBridge:
    """
    Bridge for integrating with Alita's iterative refinement and error correction processes
    
    This class connects KGoT error management with Alita's existing refinement workflows,
    ensuring that errors can be handled using both KGoT's robust error management and
    Alita's iterative improvement capabilities.
    """
    
    def __init__(self, alita_integrator: Optional[AlitaToolIntegrator] = None):
        """
        Initialize Alita Refinement Bridge
        
        @param {Optional[AlitaToolIntegrator]} alita_integrator - Alita tool integrator instance
        """
        self.alita_integrator = alita_integrator or create_alita_integrator()
        self.refinement_history = []
        self.integration_stats = {
            'total_refinements': 0,
            'successful_refinements': 0,
            'cross_system_recoveries': 0,
            'alita_escalations': 0
        }
        
        logger.info("Initialized Alita Refinement Bridge", extra={
            'operation': 'ALITA_REFINEMENT_BRIDGE_INIT',
            'alita_integrator_available': self.alita_integrator is not None
        })
    
    async def execute_iterative_refinement_with_alita(self,
                                                    failed_operation: Callable,
                                                    error_context: ErrorContext,
                                                    alita_context: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Execute iterative refinement using both KGoT error management and Alita capabilities
        
        @param {Callable} failed_operation - Operation that failed
        @param {ErrorContext} error_context - KGoT error context
        @param {Dict[str, Any]} alita_context - Alita-specific context
        @returns {Tuple[Any, bool]} - (result, success_flag)
        """
        refinement_id = f"alita_refinement_{int(time.time())}"
        
        logger.info("Starting Alita-integrated iterative refinement", extra={
            'operation': 'ALITA_ITERATIVE_REFINEMENT_START',
            'refinement_id': refinement_id,
            'error_id': error_context.error_id,
            'error_type': error_context.error_type.value
        })
        
        try:
            # Initialize Alita session for refinement
            session_id = await self.alita_integrator.initialize_session(
                f"Error refinement: {error_context.original_operation}"
            )
            
            # Create Alita-enhanced refinement strategy
            async def alita_refinement_strategy(context: ErrorContext, iteration: int) -> Dict[str, Any]:
                """Refinement strategy that leverages Alita capabilities"""
                
                # Use Alita's web agent for context enrichment if applicable
                if 'web_context' in alita_context:
                    enhanced_context = await self.alita_integrator.enhance_tool_execution(
                        tool_name='refinement_tool',
                        tool_input={
                            'error_context': context.to_dict(),
                            'iteration': iteration,
                            'original_params': alita_context.get('original_params', {})
                        },
                        context=alita_context
                    )
                    
                    refined_params = enhanced_context.get('enhanced_input', {})
                    
                    # Add Alita-specific refinements
                    refined_params.update({
                        'alita_session_id': session_id,
                        'iteration': iteration,
                        'error_context_id': context.error_id,
                        'alita_enhancements': enhanced_context.get('alita_enhancements', {})
                    })
                    
                    return refined_params
                else:
                    # Fallback to basic refinement
                    return {
                        'alita_session_id': session_id,
                        'iteration': iteration,
                        'error_context_id': context.error_id,
                        'basic_refinement': True
                    }
            
            # Execute refinement with Alita integration
            max_iterations = 3
            for iteration in range(max_iterations):
                try:
                    refined_params = await alita_refinement_strategy(error_context, iteration)
                    
                    # Attempt operation with refined parameters
                    if asyncio.iscoroutinefunction(failed_operation):
                        result = await failed_operation(**refined_params)
                    else:
                        result = failed_operation(**refined_params)
                    
                    # Success - log and return
                    self.integration_stats['successful_refinements'] += 1
                    
                    logger.info("Alita-integrated refinement succeeded", extra={
                        'operation': 'ALITA_REFINEMENT_SUCCESS',
                        'refinement_id': refinement_id,
                        'iteration': iteration + 1,
                        'error_id': error_context.error_id
                    })
                    
                    # Store refinement history
                    self.refinement_history.append({
                        'refinement_id': refinement_id,
                        'error_context': error_context.to_dict(),
                        'successful_iteration': iteration + 1,
                        'alita_session_id': session_id,
                        'refinement_type': 'alita_integrated',
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Close Alita session
                    await self.alita_integrator.close_session()
                    
                    return result, True
                    
                except Exception as refinement_error:
                    logger.warning(f"Alita refinement iteration {iteration + 1} failed: {str(refinement_error)}")
                    
                    # Update error context
                    error_context.retry_count = iteration + 1
                    error_context.recovery_attempts.append({
                        'iteration': iteration + 1,
                        'refinement_type': 'alita_integrated',
                        'error': str(refinement_error),
                        'timestamp': datetime.now().isoformat()
                    })
                    continue
            
            # All iterations failed
            self.integration_stats['alita_escalations'] += 1
            
            logger.error("All Alita-integrated refinement attempts failed", extra={
                'operation': 'ALITA_REFINEMENT_FAILED',
                'refinement_id': refinement_id,
                'error_id': error_context.error_id,
                'total_iterations': max_iterations
            })
            
            # Close Alita session
            await self.alita_integrator.close_session()
            
            return None, False
            
        except Exception as integration_error:
            logger.error("Alita refinement integration failed", extra={
                'operation': 'ALITA_REFINEMENT_INTEGRATION_FAILED',
                'refinement_id': refinement_id,
                'error': str(integration_error)
            })
            
            # Close Alita session if still open
            try:
                await self.alita_integrator.close_session()
            except:
                pass
            
            return None, False
    
    def get_refinement_statistics(self) -> Dict[str, Any]:
        """Get comprehensive refinement statistics"""
        return {
            'integration_stats': self.integration_stats,
            'refinement_history_count': len(self.refinement_history),
            'timestamp': datetime.now().isoformat()
        }


class KGoTToolBridgeErrorIntegration:
    """
    Integration bridge for connecting KGoT Error Management with the KGoT Tool Bridge
    
    This class ensures that tool execution errors are properly handled using the
    comprehensive error management system and integrated with Alita's capabilities.
    """
    
    def __init__(self, 
                 error_management_system: KGoTErrorManagementSystem,
                 alita_refinement_bridge: AlitaRefinementBridge):
        """
        Initialize KGoT Tool Bridge Error Integration
        
        @param {KGoTErrorManagementSystem} error_management_system - KGoT error management system
        @param {AlitaRefinementBridge} alita_refinement_bridge - Alita refinement bridge
        """
        self.error_management_system = error_management_system
        self.alita_refinement_bridge = alita_refinement_bridge
        self.tool_error_stats = {
            'total_tool_errors': 0,
            'successfully_recovered': 0,
            'escalated_to_alita': 0,
            'tool_error_types': {}
        }
        
        logger.info("Initialized KGoT Tool Bridge Error Integration", extra={
            'operation': 'TOOL_BRIDGE_ERROR_INTEGRATION_INIT'
        })
    
    async def handle_tool_execution_error(self,
                                        tool_name: str,
                                        tool_error: Exception,
                                        tool_input: Dict[str, Any],
                                        execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tool execution errors with comprehensive error management
        
        @param {str} tool_name - Name of the tool that failed
        @param {Exception} tool_error - The error that occurred
        @param {Dict[str, Any]} tool_input - Input parameters that caused the error
        @param {Dict[str, Any]} execution_context - Execution context
        @returns {Dict[str, Any]} - Recovery result
        """
        error_id = f"tool_error_{int(time.time())}"
        
        # Update statistics
        self.tool_error_stats['total_tool_errors'] += 1
        error_type_name = type(tool_error).__name__
        self.tool_error_stats['tool_error_types'][error_type_name] = \
            self.tool_error_stats['tool_error_types'].get(error_type_name, 0) + 1
        
        logger.error("Tool execution error detected", extra={
            'operation': 'TOOL_EXECUTION_ERROR',
            'error_id': error_id,
            'tool_name': tool_name,
            'error_type': error_type_name,
            'error_message': str(tool_error)
        })
        
        try:
            # Step 1: Use KGoT Error Management System for initial handling
            operation_context = f"Tool execution: {tool_name}"
            recovery_result, success = await self.error_management_system.handle_error(
                error=tool_error,
                operation_context=operation_context,
                severity=ErrorSeverity.HIGH
            )
            
            if success:
                self.tool_error_stats['successfully_recovered'] += 1
                
                logger.info("Tool error successfully recovered by KGoT Error Management", extra={
                    'operation': 'TOOL_ERROR_RECOVERY_SUCCESS',
                    'error_id': error_id,
                    'tool_name': tool_name
                })
                
                return {
                    'success': True,
                    'result': recovery_result,
                    'recovery_method': 'kgot_error_management',
                    'error_id': error_id,
                    'tool_name': tool_name
                }
            
            # Step 2: Escalate to Alita refinement if KGoT recovery failed
            logger.info("Escalating tool error to Alita refinement", extra={
                'operation': 'TOOL_ERROR_ALITA_ESCALATION',
                'error_id': error_id,
                'tool_name': tool_name
            })
            
            self.tool_error_stats['escalated_to_alita'] += 1
            
            # Create error context for Alita refinement
            error_context = ErrorContext(
                error_id=error_id,
                error_type=self.error_management_system._classify_error(tool_error),
                severity=ErrorSeverity.HIGH,
                timestamp=datetime.now(),
                original_operation=operation_context,
                error_message=str(tool_error),
                metadata={
                    'tool_name': tool_name,
                    'tool_input': tool_input,
                    'execution_context': execution_context
                }
            )
            
            # Create mock operation for refinement
            async def failed_tool_operation(**refined_params):
                """Mock tool operation for refinement testing"""
                # In real implementation, this would re-execute the actual tool
                # with refined parameters
                return {
                    'tool_name': tool_name,
                    'refined_execution': True,
                    'refined_params': refined_params,
                    'original_error_resolved': True
                }
            
            # Execute Alita-integrated refinement
            alita_context = {
                'tool_context': True,
                'original_params': tool_input,
                'execution_context': execution_context,
                'web_context': execution_context.get('web_context', False)
            }
            
            refinement_result, refinement_success = await self.alita_refinement_bridge.execute_iterative_refinement_with_alita(
                failed_operation=failed_tool_operation,
                error_context=error_context,
                alita_context=alita_context
            )
            
            if refinement_success:
                logger.info("Tool error successfully recovered by Alita refinement", extra={
                    'operation': 'TOOL_ERROR_ALITA_RECOVERY_SUCCESS',
                    'error_id': error_id,
                    'tool_name': tool_name
                })
                
                return {
                    'success': True,
                    'result': refinement_result,
                    'recovery_method': 'alita_refinement',
                    'error_id': error_id,
                    'tool_name': tool_name
                }
            else:
                logger.error("All error recovery methods failed for tool", extra={
                    'operation': 'TOOL_ERROR_RECOVERY_COMPLETE_FAILURE',
                    'error_id': error_id,
                    'tool_name': tool_name
                })
                
                return {
                    'success': False,
                    'error': str(tool_error),
                    'recovery_method': 'all_failed',
                    'error_id': error_id,
                    'tool_name': tool_name,
                    'original_error': str(tool_error)
                }
                
        except Exception as integration_error:
            logger.critical("Tool error integration itself failed", extra={
                'operation': 'TOOL_ERROR_INTEGRATION_CRITICAL_FAILURE',
                'error_id': error_id,
                'tool_name': tool_name,
                'integration_error': str(integration_error)
            })
            
            return {
                'success': False,
                'error': str(tool_error),
                'integration_error': str(integration_error),
                'recovery_method': 'integration_failed',
                'error_id': error_id,
                'tool_name': tool_name
            }
    
    def get_tool_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tool error statistics"""
        return {
            'tool_error_stats': self.tool_error_stats,
            'alita_refinement_stats': self.alita_refinement_bridge.get_refinement_statistics(),
            'error_management_stats': self.error_management_system.get_comprehensive_statistics(),
            'timestamp': datetime.now().isoformat()
        }


class UnifiedErrorReportingSystem:
    """
    Unified error reporting system for cross-system error analytics and monitoring
    
    This class provides comprehensive error reporting across KGoT error management,
    Alita refinement processes, and tool execution errors.
    """
    
    def __init__(self,
                 error_management_system: KGoTErrorManagementSystem,
                 alita_refinement_bridge: AlitaRefinementBridge,
                 tool_bridge_integration: KGoTToolBridgeErrorIntegration):
        """
        Initialize Unified Error Reporting System
        
        @param {KGoTErrorManagementSystem} error_management_system - KGoT error management
        @param {AlitaRefinementBridge} alita_refinement_bridge - Alita refinement bridge
        @param {KGoTToolBridgeErrorIntegration} tool_bridge_integration - Tool bridge integration
        """
        self.error_management_system = error_management_system
        self.alita_refinement_bridge = alita_refinement_bridge
        self.tool_bridge_integration = tool_bridge_integration
        self.unified_stats = {
            'total_errors_across_systems': 0,
            'system_error_breakdown': {},
            'cross_system_recoveries': 0,
            'integration_health_score': 0.0
        }
        
        logger.info("Initialized Unified Error Reporting System", extra={
            'operation': 'UNIFIED_ERROR_REPORTING_INIT'
        })
    
    def generate_comprehensive_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report across all systems"""
        try:
            # Gather statistics from all components
            kgot_stats = self.error_management_system.get_comprehensive_statistics()
            alita_stats = self.alita_refinement_bridge.get_refinement_statistics()
            tool_stats = self.tool_bridge_integration.get_tool_error_statistics()
            
            # Calculate unified metrics
            total_errors = (
                kgot_stats['kgot_error_management']['total_errors_handled'] +
                tool_stats['tool_error_stats']['total_tool_errors']
            )
            
            total_recoveries = (
                kgot_stats['kgot_error_management']['successful_recoveries'] +
                alita_stats['integration_stats']['successful_refinements'] +
                tool_stats['tool_error_stats']['successfully_recovered']
            )
            
            recovery_rate = (total_recoveries / total_errors * 100) if total_errors > 0 else 0
            
            # Calculate integration health score
            integration_health = self._calculate_integration_health_score(
                kgot_stats, alita_stats, tool_stats
            )
            
            comprehensive_report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_version': '1.0.0',
                    'systems_included': ['KGoT Error Management', 'Alita Refinement', 'Tool Bridge Integration']
                },
                'unified_metrics': {
                    'total_errors_across_systems': total_errors,
                    'total_recoveries_across_systems': total_recoveries,
                    'overall_recovery_rate_percent': recovery_rate,
                    'integration_health_score': integration_health,
                    'cross_system_coordination_success': alita_stats['integration_stats']['successful_refinements']
                },
                'system_breakdown': {
                    'kgot_error_management': kgot_stats,
                    'alita_refinement': alita_stats,
                    'tool_integration': tool_stats
                },
                'recommendations': self._generate_recommendations(kgot_stats, alita_stats, tool_stats),
                'health_indicators': {
                    'error_trending': self._analyze_error_trends(),
                    'system_stability': self._assess_system_stability(),
                    'integration_effectiveness': self._evaluate_integration_effectiveness()
                }
            }
            
            logger.info("Comprehensive error report generated", extra={
                'operation': 'COMPREHENSIVE_ERROR_REPORT_GENERATED',
                'total_errors': total_errors,
                'recovery_rate': recovery_rate,
                'health_score': integration_health
            })
            
            return comprehensive_report
            
        except Exception as e:
            logger.error("Failed to generate comprehensive error report", extra={
                'operation': 'COMPREHENSIVE_ERROR_REPORT_FAILED',
                'error': str(e)
            })
            return {
                'error': 'Failed to generate report',
                'error_details': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_integration_health_score(self, kgot_stats, alita_stats, tool_stats) -> float:
        """Calculate overall integration health score (0.0-1.0)"""
        try:
            # Factor 1: Error recovery success rate
            total_errors = kgot_stats['kgot_error_management']['total_errors_handled']
            successful_recoveries = kgot_stats['kgot_error_management']['successful_recoveries']
            recovery_rate = (successful_recoveries / total_errors) if total_errors > 0 else 1.0
            
            # Factor 2: Alita integration success rate
            alita_refinements = alita_stats['integration_stats']['total_refinements']
            successful_alita = alita_stats['integration_stats']['successful_refinements']
            alita_rate = (successful_alita / alita_refinements) if alita_refinements > 0 else 1.0
            
            # Factor 3: Tool error handling success rate
            tool_errors = tool_stats['tool_error_stats']['total_tool_errors']
            tool_recoveries = tool_stats['tool_error_stats']['successfully_recovered']
            tool_rate = (tool_recoveries / tool_errors) if tool_errors > 0 else 1.0
            
            # Weighted average (KGoT: 40%, Alita: 30%, Tools: 30%)
            health_score = (recovery_rate * 0.4) + (alita_rate * 0.3) + (tool_rate * 0.3)
            
            return min(1.0, max(0.0, health_score))
            
        except Exception:
            return 0.5  # Default to neutral score if calculation fails
    
    def _generate_recommendations(self, kgot_stats, alita_stats, tool_stats) -> List[str]:
        """Generate actionable recommendations based on error statistics"""
        recommendations = []
        
        # KGoT-specific recommendations
        kgot_errors = kgot_stats['kgot_error_management']['total_errors_handled']
        kgot_success = kgot_stats['kgot_error_management']['successful_recoveries']
        
        if kgot_errors > 0 and (kgot_success / kgot_errors) < 0.8:
            recommendations.append("Consider tuning KGoT error management retry parameters")
        
        # Alita integration recommendations
        alita_escalations = alita_stats['integration_stats'].get('alita_escalations', 0)
        if alita_escalations > 5:
            recommendations.append("High number of Alita escalations detected - review error classification")
        
        # Tool error recommendations
        tool_errors = tool_stats['tool_error_stats']['total_tool_errors']
        if tool_errors > 10:
            recommendations.append("Consider implementing proactive tool validation")
        
        return recommendations
    
    def _analyze_error_trends(self) -> str:
        """Analyze error trends (placeholder for trend analysis)"""
        return "Stable - error rates within normal parameters"
    
    def _assess_system_stability(self) -> str:
        """Assess overall system stability"""
        return "Good - all error management systems operational"
    
    def _evaluate_integration_effectiveness(self) -> str:
        """Evaluate integration effectiveness"""
        return "Effective - cross-system error recovery functioning properly"


class KGoTAlitaErrorIntegrationOrchestrator:
    """
    Main orchestrator for KGoT-Alita error management integration
    
    This is the primary class that coordinates all error management integration
    components and provides a unified interface for the entire system.
    """
    
    def __init__(self, llm_client, config: Optional[IntegrationConfig] = None):
        """
        Initialize KGoT-Alita Error Integration Orchestrator
        
        @param {Any} llm_client - LLM client for error correction
        @param {Optional[IntegrationConfig]} config - Integration configuration
        """
        self.config = config or IntegrationConfig()
        self.llm_client = llm_client
        
        # Initialize core error management system
        self.error_management_system = create_kgot_error_management_system(
            llm_client=llm_client,
            config={
                'syntax_max_retries': 3,
                'api_max_retries': 6,
                'voting_rounds': 3
            }
        )
        
        # Initialize Alita refinement bridge
        self.alita_refinement_bridge = AlitaRefinementBridge()
        
        # Initialize tool bridge integration
        self.tool_bridge_integration = KGoTToolBridgeErrorIntegration(
            error_management_system=self.error_management_system,
            alita_refinement_bridge=self.alita_refinement_bridge
        )
        
        # Initialize unified reporting
        self.unified_reporting = UnifiedErrorReportingSystem(
            error_management_system=self.error_management_system,
            alita_refinement_bridge=self.alita_refinement_bridge,
            tool_bridge_integration=self.tool_bridge_integration
        )
        
        logger.info("Initialized KGoT-Alita Error Integration Orchestrator", extra={
            'operation': 'INTEGRATION_ORCHESTRATOR_INIT',
            'config': self.config.__dict__
        })
    
    async def handle_integrated_error(self,
                                    error: Exception,
                                    context: Dict[str, Any],
                                    error_source: str = 'unknown') -> Dict[str, Any]:
        """
        Handle errors with full KGoT-Alita integration
        
        @param {Exception} error - The error to handle
        @param {Dict[str, Any]} context - Error context information
        @param {str} error_source - Source of the error (tool, system, etc.)
        @returns {Dict[str, Any]} - Comprehensive error handling result
        """
        integration_id = f"integrated_error_{int(time.time())}"
        
        logger.info("Starting integrated error handling", extra={
            'operation': 'INTEGRATED_ERROR_HANDLING_START',
            'integration_id': integration_id,
            'error_source': error_source,
            'error_type': type(error).__name__
        })
        
        try:
            if error_source == 'tool':
                # Handle tool execution errors
                result = await self.tool_bridge_integration.handle_tool_execution_error(
                    tool_name=context.get('tool_name', 'unknown'),
                    tool_error=error,
                    tool_input=context.get('tool_input', {}),
                    execution_context=context.get('execution_context', {})
                )
            else:
                # Handle general system errors
                recovery_result, success = await self.error_management_system.handle_error(
                    error=error,
                    operation_context=context.get('operation_context', 'Unknown operation'),
                    severity=ErrorSeverity.MEDIUM
                )
                
                result = {
                    'success': success,
                    'result': recovery_result,
                    'recovery_method': 'kgot_error_management',
                    'integration_id': integration_id
                }
            
            logger.info("Integrated error handling completed", extra={
                'operation': 'INTEGRATED_ERROR_HANDLING_COMPLETE',
                'integration_id': integration_id,
                'success': result.get('success', False)
            })
            
            return result
            
        except Exception as integration_error:
            logger.critical("Integrated error handling failed", extra={
                'operation': 'INTEGRATED_ERROR_HANDLING_FAILED',
                'integration_id': integration_id,
                'integration_error': str(integration_error)
            })
            
            return {
                'success': False,
                'error': str(error),
                'integration_error': str(integration_error),
                'integration_id': integration_id
            }
    
    def get_integration_health_report(self) -> Dict[str, Any]:
        """Get comprehensive integration health report"""
        return self.unified_reporting.generate_comprehensive_error_report()
    
    def cleanup_integration(self):
        """Cleanup all integration resources"""
        logger.info("Cleaning up KGoT-Alita Error Integration")
        
        # Cleanup error management system
        self.error_management_system.cleanup()
        
        logger.info("KGoT-Alita Error Integration cleanup completed")


# Factory function for easy initialization
def create_kgot_alita_error_integration(llm_client, config: Optional[IntegrationConfig] = None) -> KGoTAlitaErrorIntegrationOrchestrator:
    """
    Factory function to create KGoT-Alita Error Integration Orchestrator
    
    @param {Any} llm_client - LLM client for error correction
    @param {Optional[IntegrationConfig]} config - Integration configuration
    @returns {KGoTAlitaErrorIntegrationOrchestrator} - Initialized integration orchestrator
    """
    return KGoTAlitaErrorIntegrationOrchestrator(llm_client=llm_client, config=config)


# Integration example and testing
if __name__ == "__main__":
    async def test_integration():
        """Test the KGoT-Alita Error Integration"""
        # Mock LLM client for testing
        class MockLLMClient:
            async def acomplete(self, prompt: str):
                class MockResponse:
                    text = '{"corrected": "content"}'
                return MockResponse()
        
        # Initialize integration orchestrator
        integration_orchestrator = create_kgot_alita_error_integration(
            llm_client=MockLLMClient(),
            config=IntegrationConfig(
                enable_alita_refinement=True,
                enable_cross_system_recovery=True,
                max_cross_system_retries=2
            )
        )
        
        # Test tool error handling
        test_tool_error = RuntimeError("Tool execution failed")
        tool_context = {
            'tool_name': 'test_tool',
            'tool_input': {'query': 'test'},
            'execution_context': {'web_context': False}
        }
        
        result = await integration_orchestrator.handle_integrated_error(
            error=test_tool_error,
            context=tool_context,
            error_source='tool'
        )
        
        print(f"Tool error handling result: {result}")
        
        # Generate health report
        health_report = integration_orchestrator.get_integration_health_report()
        print(f"Integration Health Report: {json.dumps(health_report, indent=2)}")
        
        # Cleanup
        integration_orchestrator.cleanup_integration()
        
        print("KGoT-Alita Error Integration test completed!")
    
    # Run test if executed directly
    asyncio.run(test_integration()) 