#!/usr/bin/env python3
"""
Deployment Integration Module for Task 52
Integrates configuration management with deployment pipeline
Consolidates Tasks 45 and 51 functionality
"""

import os
import json
import yaml
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from datetime import datetime

from .advanced_config_management import (
    ConfigurationManager,
    ConfigurationValidator,
    ConfigurationVersionManager
)
from ..deployment.production_deployment import (
    ProductionDeploymentPipeline,
    BlueGreenDeploymentManager
)
from ..monitoring.monitoring_setup import MonitoringManager
from ..validation.deployment_validator import DeploymentValidator


@dataclass
class DeploymentContext:
    """Context for deployment operations"""
    environment: str
    version: str
    config_version: str
    strategy: str = "blue-green"
    enable_monitoring: bool = True
    enable_auto_rollback: bool = True
    validate_before_deploy: bool = True
    comprehensive_validation: bool = False


class UnifiedDeploymentOrchestrator:
    """
    Orchestrates unified deployment pipeline integrating:
    - Configuration management (Task 51)
    - Blue-green deployment (Task 45)
    - Security scanning (Task 38)
    - Monitoring and alerting (Task 43)
    - Validation suite (Task 18)
    """
    
    def __init__(self, config_path: str = "configuration/pipeline_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Initialize managers
        self.config_manager = ConfigurationManager()
        self.config_validator = ConfigurationValidator()
        self.version_manager = ConfigurationVersionManager()
        self.deployment_pipeline = ProductionDeploymentPipeline()
        self.blue_green_manager = BlueGreenDeploymentManager()
        self.monitoring_manager = MonitoringManager()
        self.deployment_validator = DeploymentValidator()
        
        # Load pipeline configuration
        self.pipeline_config = self._load_pipeline_config()
        
    def _load_pipeline_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load pipeline config: {e}")
            return {}
    
    async def deploy(self, context: DeploymentContext) -> Dict[str, Any]:
        """
        Execute unified deployment process
        
        Args:
            context: Deployment context with environment, versions, and options
            
        Returns:
            Deployment result with status and details
        """
        deployment_id = f"{context.environment}-{context.version}-{int(time.time())}"
        
        self.logger.info(f"Starting unified deployment {deployment_id}")
        self.logger.info(f"Context: {context}")
        
        result = {
            "deployment_id": deployment_id,
            "context": context.__dict__,
            "status": "in_progress",
            "stages": {},
            "start_time": datetime.utcnow().isoformat(),
            "errors": []
        }
        
        try:
            # Stage 1: Pre-deployment validation
            if context.validate_before_deploy:
                stage_result = await self._validate_pre_deployment(context)
                result["stages"]["pre_validation"] = stage_result
                if not stage_result["success"]:
                    result["status"] = "failed"
                    result["errors"].extend(stage_result.get("errors", []))
                    return result
            
            # Stage 2: Configuration deployment
            stage_result = await self._deploy_configuration(context)
            result["stages"]["configuration"] = stage_result
            if not stage_result["success"]:
                result["status"] = "failed"
                result["errors"].extend(stage_result.get("errors", []))
                return result
            
            # Stage 3: Application deployment
            stage_result = await self._deploy_application(context)
            result["stages"]["application"] = stage_result
            if not stage_result["success"]:
                result["status"] = "failed"
                result["errors"].extend(stage_result.get("errors", []))
                # Attempt configuration rollback
                await self._rollback_configuration(context)
                return result
            
            # Stage 4: Post-deployment validation
            stage_result = await self._validate_post_deployment(context)
            result["stages"]["post_validation"] = stage_result
            if not stage_result["success"]:
                result["status"] = "failed"
                result["errors"].extend(stage_result.get("errors", []))
                # Attempt full rollback
                await self._rollback_deployment(context)
                return result
            
            # Stage 5: Enable monitoring
            if context.enable_monitoring:
                stage_result = await self._enable_monitoring(context)
                result["stages"]["monitoring"] = stage_result
                if not stage_result["success"]:
                    self.logger.warning("Monitoring setup failed, but deployment continues")
                    result["errors"].extend(stage_result.get("errors", []))
            
            # Stage 6: Setup auto-rollback
            if context.enable_auto_rollback:
                stage_result = await self._setup_auto_rollback(context)
                result["stages"]["auto_rollback"] = stage_result
                if not stage_result["success"]:
                    self.logger.warning("Auto-rollback setup failed, but deployment continues")
                    result["errors"].extend(stage_result.get("errors", []))
            
            result["status"] = "success"
            result["end_time"] = datetime.utcnow().isoformat()
            
            self.logger.info(f"Deployment {deployment_id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            result["status"] = "failed"
            result["errors"].append(str(e))
            result["end_time"] = datetime.utcnow().isoformat()
            
            # Attempt emergency rollback
            try:
                await self._emergency_rollback(context)
            except Exception as rollback_error:
                self.logger.error(f"Emergency rollback failed: {rollback_error}")
                result["errors"].append(f"Rollback failed: {rollback_error}")
            
            return result
    
    async def _validate_pre_deployment(self, context: DeploymentContext) -> Dict[str, Any]:
        """Validate configuration and environment before deployment"""
        self.logger.info("Running pre-deployment validation")
        
        result = {
            "success": True,
            "errors": [],
            "validations": {}
        }
        
        try:
            # Validate configuration version exists and is compatible
            config_validation = await self.config_validator.validate_version(
                context.config_version, context.version
            )
            result["validations"]["config_version"] = config_validation
            
            if not config_validation["valid"]:
                result["success"] = False
                result["errors"].extend(config_validation.get("errors", []))
            
            # Validate environment readiness
            env_validation = await self.deployment_validator.validate_environment(
                context.environment
            )
            result["validations"]["environment"] = env_validation
            
            if not env_validation["ready"]:
                result["success"] = False
                result["errors"].extend(env_validation.get("errors", []))
            
            # Validate deployment strategy
            strategy_validation = await self._validate_deployment_strategy(context)
            result["validations"]["strategy"] = strategy_validation
            
            if not strategy_validation["valid"]:
                result["success"] = False
                result["errors"].extend(strategy_validation.get("errors", []))
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Pre-deployment validation failed: {e}")
        
        return result
    
    async def _deploy_configuration(self, context: DeploymentContext) -> Dict[str, Any]:
        """Deploy configuration with versioning and validation"""
        self.logger.info(f"Deploying configuration version {context.config_version}")
        
        result = {
            "success": True,
            "errors": [],
            "config_deployment": {}
        }
        
        try:
            # Create configuration snapshot for rollback
            snapshot_id = await self.version_manager.create_snapshot(
                context.environment, context.config_version
            )
            result["config_deployment"]["snapshot_id"] = snapshot_id
            
            # Deploy configuration
            deployment_result = await self.config_manager.deploy_configuration(
                environment=context.environment,
                version=context.config_version,
                validate_before_deploy=context.validate_before_deploy
            )
            
            result["config_deployment"].update(deployment_result)
            
            if not deployment_result.get("success", False):
                result["success"] = False
                result["errors"].extend(deployment_result.get("errors", []))
                return result
            
            # Wait for configuration propagation
            await self._wait_for_config_propagation(context.environment)
            
            # Validate configuration is active
            validation_result = await self.config_validator.validate_active_config(
                context.environment, context.config_version
            )
            
            if not validation_result["valid"]:
                result["success"] = False
                result["errors"].extend(validation_result.get("errors", []))
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Configuration deployment failed: {e}")
        
        return result
    
    async def _deploy_application(self, context: DeploymentContext) -> Dict[str, Any]:
        """Deploy application using blue-green strategy"""
        self.logger.info(f"Deploying application version {context.version}")
        
        result = {
            "success": True,
            "errors": [],
            "app_deployment": {}
        }
        
        try:
            # Execute blue-green deployment
            deployment_result = await self.blue_green_manager.deploy(
                environment=context.environment,
                version=context.version,
                config_version=context.config_version,
                strategy=context.strategy
            )
            
            result["app_deployment"].update(deployment_result)
            
            if not deployment_result.get("success", False):
                result["success"] = False
                result["errors"].extend(deployment_result.get("errors", []))
                return result
            
            # Wait for application to stabilize
            await self._wait_for_app_stabilization(context.environment)
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Application deployment failed: {e}")
        
        return result
    
    async def _validate_post_deployment(self, context: DeploymentContext) -> Dict[str, Any]:
        """Validate deployment after completion"""
        self.logger.info("Running post-deployment validation")
        
        result = {
            "success": True,
            "errors": [],
            "validations": {}
        }
        
        try:
            # Health check validation
            health_validation = await self.deployment_validator.validate_health(
                context.environment, comprehensive=context.comprehensive_validation
            )
            result["validations"]["health"] = health_validation
            
            if not health_validation["healthy"]:
                result["success"] = False
                result["errors"].extend(health_validation.get("errors", []))
            
            # Configuration validation
            config_validation = await self.config_validator.validate_deployment(
                context.environment, context.config_version
            )
            result["validations"]["configuration"] = config_validation
            
            if not config_validation["valid"]:
                result["success"] = False
                result["errors"].extend(config_validation.get("errors", []))
            
            # Performance validation
            if context.comprehensive_validation:
                perf_validation = await self.deployment_validator.validate_performance(
                    context.environment
                )
                result["validations"]["performance"] = perf_validation
                
                if not perf_validation["acceptable"]:
                    result["success"] = False
                    result["errors"].extend(perf_validation.get("errors", []))
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Post-deployment validation failed: {e}")
        
        return result
    
    async def _enable_monitoring(self, context: DeploymentContext) -> Dict[str, Any]:
        """Enable monitoring and alerting for deployment"""
        self.logger.info("Enabling monitoring and alerting")
        
        result = {
            "success": True,
            "errors": [],
            "monitoring": {}
        }
        
        try:
            monitoring_result = await self.monitoring_manager.enable_monitoring(
                environment=context.environment,
                version=context.version,
                config_version=context.config_version
            )
            
            result["monitoring"].update(monitoring_result)
            
            if not monitoring_result.get("success", False):
                result["success"] = False
                result["errors"].extend(monitoring_result.get("errors", []))
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Monitoring setup failed: {e}")
        
        return result
    
    async def _setup_auto_rollback(self, context: DeploymentContext) -> Dict[str, Any]:
        """Setup automated rollback triggers"""
        self.logger.info("Setting up automated rollback")
        
        result = {
            "success": True,
            "errors": [],
            "auto_rollback": {}
        }
        
        try:
            rollback_result = await self.deployment_pipeline.setup_auto_rollback(
                environment=context.environment,
                version=context.version,
                config_version=context.config_version
            )
            
            result["auto_rollback"].update(rollback_result)
            
            if not rollback_result.get("success", False):
                result["success"] = False
                result["errors"].extend(rollback_result.get("errors", []))
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(f"Auto-rollback setup failed: {e}")
        
        return result
    
    async def _validate_deployment_strategy(self, context: DeploymentContext) -> Dict[str, Any]:
        """Validate deployment strategy is supported and configured"""
        supported_strategies = ["blue-green", "canary", "rolling"]
        
        if context.strategy not in supported_strategies:
            return {
                "valid": False,
                "errors": [f"Unsupported deployment strategy: {context.strategy}"]
            }
        
        # Check strategy-specific requirements
        if context.strategy == "blue-green":
            # Validate blue-green configuration
            bg_config = self.pipeline_config.get("blue_green", {})
            if not bg_config:
                return {
                    "valid": False,
                    "errors": ["Blue-green deployment not configured"]
                }
        
        return {"valid": True, "errors": []}
    
    async def _wait_for_config_propagation(self, environment: str, timeout: int = 60):
        """Wait for configuration to propagate across all services"""
        self.logger.info(f"Waiting for configuration propagation in {environment}")
        await asyncio.sleep(30)  # Base wait time
        
        # Additional validation can be added here
        
    async def _wait_for_app_stabilization(self, environment: str, timeout: int = 120):
        """Wait for application to stabilize after deployment"""
        self.logger.info(f"Waiting for application stabilization in {environment}")
        await asyncio.sleep(60)  # Base wait time
        
        # Additional health checks can be added here
    
    async def _rollback_configuration(self, context: DeploymentContext):
        """Rollback configuration to previous version"""
        self.logger.warning(f"Rolling back configuration in {context.environment}")
        try:
            await self.config_manager.rollback_configuration(
                environment=context.environment
            )
        except Exception as e:
            self.logger.error(f"Configuration rollback failed: {e}")
    
    async def _rollback_deployment(self, context: DeploymentContext):
        """Rollback both application and configuration"""
        self.logger.warning(f"Rolling back deployment in {context.environment}")
        try:
            # Rollback application first
            await self.blue_green_manager.rollback(
                environment=context.environment
            )
            
            # Then rollback configuration
            await self._rollback_configuration(context)
            
        except Exception as e:
            self.logger.error(f"Deployment rollback failed: {e}")
    
    async def _emergency_rollback(self, context: DeploymentContext):
        """Emergency rollback procedure"""
        self.logger.critical(f"Executing emergency rollback in {context.environment}")
        try:
            await self.deployment_pipeline.emergency_rollback(
                environment=context.environment
            )
        except Exception as e:
            self.logger.critical(f"Emergency rollback failed: {e}")


def main():
    """CLI interface for deployment integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Deployment Orchestrator")
    parser.add_argument("--environment", required=True, choices=["staging", "production"])
    parser.add_argument("--version", required=True, help="Application version")
    parser.add_argument("--config-version", required=True, help="Configuration version")
    parser.add_argument("--strategy", default="blue-green", choices=["blue-green", "canary", "rolling"])
    parser.add_argument("--enable-monitoring", action="store_true", default=True)
    parser.add_argument("--enable-auto-rollback", action="store_true", default=True)
    parser.add_argument("--validate-before-deploy", action="store_true", default=True)
    parser.add_argument("--comprehensive", action="store_true", help="Enable comprehensive validation")
    parser.add_argument("--config-path", default="configuration/pipeline_config.yaml")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create deployment context
    context = DeploymentContext(
        environment=args.environment,
        version=args.version,
        config_version=args.config_version,
        strategy=args.strategy,
        enable_monitoring=args.enable_monitoring,
        enable_auto_rollback=args.enable_auto_rollback,
        validate_before_deploy=args.validate_before_deploy,
        comprehensive_validation=args.comprehensive
    )
    
    # Execute deployment
    orchestrator = UnifiedDeploymentOrchestrator(args.config_path)
    
    async def run_deployment():
        result = await orchestrator.deploy(context)
        print(json.dumps(result, indent=2))
        return result["status"] == "success"
    
    success = asyncio.run(run_deployment())
    exit(0 if success else 1)


if __name__ == "__main__":
    main()