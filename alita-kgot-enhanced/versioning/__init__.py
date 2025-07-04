"""
Enhanced MCP Version Management System Package

Task 35 Implementation: Create MCP Version Management System
- Implement MCP versioning with backward compatibility
- Add automated migration tools for MCP updates  
- Create rollback capabilities for failed MCP deployments
- Support A/B testing for MCP improvements

@module MCPVersioning
@author Enhanced Alita KGoT Team
@date 2025
"""

from .mcp_version_management import (
    # Main orchestration classes
    MCPVersionManager,
    MCPVersionManagerConfig,
    
    # Core versioning components
    SemanticVersionEngine,
    BackwardCompatibilityChecker,
    MCPMigrationManager,
    MCPRollbackManager,
    ABTestingFramework,
    
    # Data models and enums
    MCPVersion,
    VersionConstraint,
    MigrationPlan,
    RollbackPoint,
    ABTestConfig,
    ABTestResult,
    CompatibilityReport,
    
    # Enums
    VersionType,
    CompatibilityLevel,
    MigrationStatus,
    RollbackStatus,
    ABTestStatus,
    
    # Utility functions
    create_version_manager,
    parse_version_string,
    compare_versions,
    validate_version_constraint,
    
    # Example usage
    example_version_management_usage
)

# Version management metadata
__version__ = "1.0.0"
__author__ = "Enhanced Alita KGoT Team"

# Feature flags
FEATURE_FLAGS = {
    "semantic_versioning": True,
    "automated_migration": True,
    "rollback_capabilities": True,
    "ab_testing": True,
    "compatibility_checking": True,
    "winston_logging": True,
    "langchain_integration": True,
    "sequential_thinking": True
}

# Export all components
__all__ = [
    # Core classes
    "MCPVersionManager",
    "MCPVersionManagerConfig",
    "SemanticVersionEngine", 
    "BackwardCompatibilityChecker",
    "MCPMigrationManager",
    "MCPRollbackManager",
    "ABTestingFramework",
    
    # Data models
    "MCPVersion",
    "VersionConstraint",
    "MigrationPlan", 
    "RollbackPoint",
    "ABTestConfig",
    "ABTestResult",
    "CompatibilityReport",
    
    # Enums
    "VersionType",
    "CompatibilityLevel",
    "MigrationStatus",
    "RollbackStatus", 
    "ABTestStatus",
    
    # Utilities
    "create_version_manager",
    "parse_version_string",
    "compare_versions",
    "validate_version_constraint",
    "example_version_management_usage"
] 