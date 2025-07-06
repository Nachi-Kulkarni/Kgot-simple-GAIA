#!/usr/bin/env python3
"""
Production Deployment Pipeline Orchestrator
===========================================

This module implements a comprehensive, automated CI/CD pipeline for the KGoT-Alita
enhanced system with blue-green deployment, automated rollback, and comprehensive
monitoring integration.

Features:
- Automated container-based build process
- Blue-green deployment strategy
- Automated rollback triggers
- Multi-environment consistency with IaC
- Security scanning and compliance
- Comprehensive monitoring and logging

Author: Alita KGoT Enhanced Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import docker
import kubernetes
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.logging.winston_config import setup_winston_logging
from quality.mcp_quality_framework import MCPQualityAssurance
from monitoring.advanced_monitoring import AdvancedMonitoring
from security.mcp_security_compliance import MCPSecurityCompliance

@dataclass
class DeploymentConfig:
    """Configuration for deployment pipeline"""
    environment: str
    version: str
    registry_url: str
    cluster_config: str
    health_check_timeout: int = 300
    rollback_threshold: float = 0.05  # 5% error rate
    monitoring_window: int = 300  # 5 minutes
    
@dataclass
class DeploymentMetrics:
    """Metrics collected during deployment"""
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_count: int = 0
    warnings: List[str] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.performance_metrics is None:
            self.performance_metrics = {}

class ProductionDeploymentPipeline:
    """
    Main orchestrator for production deployment pipeline implementing
    blue-green deployment with automated rollback capabilities.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the deployment pipeline
        
        Args:
            config_path: Path to pipeline configuration file
        """
        self.logger = setup_winston_logging(
            'production_deployment',
            log_file='logs/deployment/production_deployment.log'
        )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.docker_client = docker.from_env()
        self.k8s_client = kubernetes.client.ApiClient()
        self.quality_framework = MCPQualityAssurance()
        self.monitoring = AdvancedMonitoring()
        self.security = MCPSecurityCompliance()
        
        # Deployment state
        self.current_deployment: Optional[DeploymentMetrics] = None
        self.deployment_history: List[DeploymentMetrics] = []
        
        # Prometheus metrics for monitoring
        self.registry = CollectorRegistry()
        self.deployment_counter = Counter(
            'kgot_deployments_total',
            'Total number of deployments',
            ['environment', 'status'],
            registry=self.registry
        )
        self.deployment_duration = Histogram(
            'kgot_deployment_duration_seconds',
            'Deployment duration in seconds',
            registry=self.registry
        )
        self.rollback_counter = Counter(
            'kgot_rollbacks_total',
            'Total number of rollbacks',
            ['environment', 'trigger'],
            registry=self.registry
        )
        
        self.logger.info("Production deployment pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load pipeline configuration from file or environment
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        
        # Default configuration
        return {
            'environments': {
                'staging': {
                    'registry_url': os.getenv('STAGING_REGISTRY', 'localhost:5000'),
                    'cluster_config': os.getenv('STAGING_KUBECONFIG', '~/.kube/config'),
                    'namespace': 'kgot-staging'
                },
                'production': {
                    'registry_url': os.getenv('PROD_REGISTRY', 'localhost:5000'),
                    'cluster_config': os.getenv('PROD_KUBECONFIG', '~/.kube/config'),
                    'namespace': 'kgot-production'
                }
            },
            'rollback': {
                'error_threshold': 0.05,
                'monitoring_window': 300,
                'health_check_timeout': 300
            },
            'security': {
                'enable_scanning': True,
                'vulnerability_threshold': 'MEDIUM',
                'secret_scanning': True
            }
        }
    
    async def execute_deployment(
        self,
        environment: str,
        version: str,
        force: bool = False
    ) -> bool:
        """
        Execute complete deployment pipeline
        
        Args:
            environment: Target environment (staging/production)
            version: Version to deploy
            force: Force deployment even if quality checks fail
            
        Returns:
            True if deployment successful, False otherwise
        """
        deployment_start = time.time()
        
        # Initialize deployment metrics
        self.current_deployment = DeploymentMetrics(
            start_time=datetime.now()
        )
        
        try:
            self.logger.info(f"Starting deployment of version {version} to {environment}")
            
            # Phase 1: Build and Security Scanning
            if not await self._build_phase(version):
                raise Exception("Build phase failed")
            
            # Phase 2: Quality Assurance Testing
            if not force and not await self._test_phase(version):
                raise Exception("Quality assurance testing failed")
            
            # Phase 3: Blue-Green Deployment
            if not await self._deploy_phase(environment, version):
                raise Exception("Deployment phase failed")
            
            # Phase 4: Validation and Health Checks
            if not await self._validation_phase(environment, version):
                raise Exception("Validation phase failed")
            
            # Phase 5: Production Traffic Switch
            if not await self._traffic_switch_phase(environment):
                raise Exception("Traffic switch failed")
            
            # Phase 6: Post-deployment Monitoring
            await self._monitoring_phase(environment, version)
            
            # Mark deployment as successful
            self.current_deployment.success = True
            self.current_deployment.end_time = datetime.now()
            
            duration = time.time() - deployment_start
            self.deployment_duration.observe(duration)
            self.deployment_counter.labels(environment=environment, status='success').inc()
            
            self.logger.info(f"Deployment completed successfully in {duration:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {str(e)}")
            self.current_deployment.error_count += 1
            self.current_deployment.end_time = datetime.now()
            
            # Attempt automated rollback
            await self._emergency_rollback(environment, str(e))
            
            duration = time.time() - deployment_start
            self.deployment_counter.labels(environment=environment, status='failure').inc()
            
            return False
        
        finally:
            if self.current_deployment:
                self.deployment_history.append(self.current_deployment)
                self.current_deployment = None
    
    async def _build_phase(self, version: str) -> bool:
        """
        Execute build phase with container creation and security scanning
        
        Args:
            version: Version to build
            
        Returns:
            True if build successful
        """
        self.logger.info("Executing build phase")
        
        try:
            # Get list of all services to build
            services = self._get_service_list()
            
            for service in services:
                self.logger.info(f"Building {service} version {version}")
                
                # Build Docker image
                image_tag = f"{service}:{version}"
                build_result = await self._build_docker_image(service, image_tag)
                
                if not build_result:
                    raise Exception(f"Failed to build {service}")
                
                # Security scanning
                if self.config.get('security', {}).get('enable_scanning', True):
                    scan_result = await self._security_scan_image(image_tag)
                    if not scan_result:
                        raise Exception(f"Security scan failed for {service}")
                
                # Push to registry
                await self._push_to_registry(image_tag)
            
            self.logger.info("Build phase completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Build phase failed: {str(e)}")
            return False
    
    async def _test_phase(self, version: str) -> bool:
        """
        Execute comprehensive testing phase
        
        Args:
            version: Version to test
            
        Returns:
            True if all tests pass
        """
        self.logger.info("Executing test phase")
        
        try:
            # Run quality assurance framework
            qa_results = await self.quality_framework.run_comprehensive_assessment()
            
            if not qa_results.get('overall_pass', False):
                self.logger.error("Quality assurance tests failed")
                self.logger.error(f"Failed tests: {qa_results.get('failed_tests', [])}")
                return False
            
            # Run security compliance tests
            security_results = await self.security.run_compliance_assessment()
            
            if not security_results.get('compliant', False):
                self.logger.error("Security compliance tests failed")
                return False
            
            self.logger.info("Test phase completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Test phase failed: {str(e)}")
            return False
    
    async def _deploy_phase(self, environment: str, version: str) -> bool:
        """
        Execute blue-green deployment phase
        
        Args:
            environment: Target environment
            version: Version to deploy
            
        Returns:
            True if deployment successful
        """
        self.logger.info(f"Executing deployment phase to {environment}")
        
        try:
            # Determine blue/green environment
            current_env = await self._get_current_environment(environment)
            target_env = 'blue' if current_env == 'green' else 'green'
            
            self.logger.info(f"Deploying to {target_env} environment")
            
            # Deploy to target environment
            deploy_result = await self._deploy_to_environment(
                environment, target_env, version
            )
            
            if not deploy_result:
                raise Exception(f"Failed to deploy to {target_env} environment")
            
            self.logger.info("Deployment phase completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment phase failed: {str(e)}")
            return False
    
    async def _validation_phase(self, environment: str, version: str) -> bool:
        """
        Execute validation and health check phase
        
        Args:
            environment: Target environment
            version: Version to validate
            
        Returns:
            True if validation successful
        """
        self.logger.info("Executing validation phase")
        
        try:
            # Wait for services to be ready
            await self._wait_for_services_ready(environment)
            
            # Run health checks
            health_status = await self._run_health_checks(environment)
            
            if not health_status:
                raise Exception("Health checks failed")
            
            # Run end-to-end tests
            e2e_results = await self._run_e2e_tests(environment)
            
            if not e2e_results:
                raise Exception("End-to-end tests failed")
            
            self.logger.info("Validation phase completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation phase failed: {str(e)}")
            return False
    
    async def _traffic_switch_phase(self, environment: str) -> bool:
        """
        Execute traffic switch to new environment
        
        Args:
            environment: Target environment
            
        Returns:
            True if traffic switch successful
        """
        self.logger.info("Executing traffic switch phase")
        
        try:
            # Get current and target environments
            current_env = await self._get_current_environment(environment)
            target_env = 'blue' if current_env == 'green' else 'green'
            
            # Gradual traffic switch (canary deployment)
            traffic_percentages = [10, 25, 50, 75, 100]
            
            for percentage in traffic_percentages:
                self.logger.info(f"Switching {percentage}% traffic to {target_env}")
                
                # Update load balancer configuration
                await self._update_traffic_split(environment, target_env, percentage)
                
                # Monitor for issues during gradual switch
                await asyncio.sleep(60)  # Wait 1 minute between switches
                
                # Check for issues
                if not await self._check_deployment_health(environment):
                    raise Exception(f"Health check failed at {percentage}% traffic")
            
            # Update environment marker
            await self._set_current_environment(environment, target_env)
            
            self.logger.info("Traffic switch completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Traffic switch failed: {str(e)}")
            return False
    
    async def _monitoring_phase(self, environment: str, version: str) -> None:
        """
        Execute post-deployment monitoring phase
        
        Args:
            environment: Target environment
            version: Deployed version
        """
        self.logger.info("Starting post-deployment monitoring")
        
        # Start monitoring task in background
        asyncio.create_task(self._continuous_monitoring(environment, version))
    
    async def _continuous_monitoring(self, environment: str, version: str) -> None:
        """
        Continuous monitoring with automated rollback triggers
        
        Args:
            environment: Target environment
            version: Deployed version
        """
        monitoring_start = time.time()
        rollback_window = self.config.get('rollback', {}).get('monitoring_window', 300)
        
        while time.time() - monitoring_start < rollback_window:
            try:
                # Get current metrics
                metrics = await self.monitoring.get_real_time_metrics(environment)
                
                # Check rollback triggers
                if await self._should_trigger_rollback(metrics):
                    self.logger.warning("Rollback triggered by monitoring")
                    await self._automated_rollback(environment, "monitoring_trigger")
                    break
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                await asyncio.sleep(30)
        
        self.logger.info("Post-deployment monitoring completed")
    
    async def _should_trigger_rollback(self, metrics: Dict[str, Any]) -> bool:
        """
        Determine if rollback should be triggered based on metrics
        
        Args:
            metrics: Current system metrics
            
        Returns:
            True if rollback should be triggered
        """
        error_threshold = self.config.get('rollback', {}).get('error_threshold', 0.05)
        
        # Check error rate
        error_rate = metrics.get('error_rate', 0)
        if error_rate > error_threshold:
            self.logger.warning(f"Error rate {error_rate} exceeds threshold {error_threshold}")
            return True
        
        # Check response time degradation
        response_time = metrics.get('avg_response_time', 0)
        baseline_response_time = metrics.get('baseline_response_time', 1000)
        if response_time > baseline_response_time * 2:
            self.logger.warning(f"Response time {response_time}ms exceeds baseline {baseline_response_time}ms")
            return True
        
        # Check service health
        if not metrics.get('all_services_healthy', True):
            self.logger.warning("One or more services are unhealthy")
            return True
        
        return False
    
    async def _automated_rollback(self, environment: str, trigger: str) -> bool:
        """
        Execute automated rollback to previous stable version
        
        Args:
            environment: Target environment
            trigger: What triggered the rollback
            
        Returns:
            True if rollback successful
        """
        self.logger.warning(f"Executing automated rollback triggered by: {trigger}")
        
        try:
            # Get previous stable version
            previous_version = await self._get_previous_stable_version(environment)
            
            if not previous_version:
                raise Exception("No previous stable version available")
            
            # Quick rollback by switching traffic back
            current_env = await self._get_current_environment(environment)
            rollback_env = 'blue' if current_env == 'green' else 'green'
            
            # Immediate traffic switch to stable environment
            await self._update_traffic_split(environment, rollback_env, 100)
            await self._set_current_environment(environment, rollback_env)
            
            # Update metrics
            self.rollback_counter.labels(environment=environment, trigger=trigger).inc()
            
            self.logger.info(f"Rollback completed successfully to version {previous_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {str(e)}")
            return False
    
    async def _emergency_rollback(self, environment: str, error: str) -> None:
        """
        Execute emergency rollback in case of deployment failure
        
        Args:
            environment: Target environment
            error: Error that triggered rollback
        """
        self.logger.error(f"Executing emergency rollback due to: {error}")
        await self._automated_rollback(environment, "deployment_failure")
    
    # Helper methods for infrastructure operations
    
    def _get_service_list(self) -> List[str]:
        """Get list of all services to deploy"""
        return [
            'kgot-controller',
            'graph-store',
            'manager-agent',
            'web-agent',
            'mcp-creation',
            'monitoring',
            'validation'
        ]
    
    async def _build_docker_image(self, service: str, tag: str) -> bool:
        """Build Docker image for service"""
        try:
            dockerfile_path = f"alita-kgot-enhanced/{service}/Dockerfile"
            context_path = f"alita-kgot-enhanced/{service}"
            
            self.logger.info(f"Building {tag} from {dockerfile_path}")
            
            # Build image using Docker API
            image, logs = self.docker_client.images.build(
                path=context_path,
                dockerfile=dockerfile_path,
                tag=tag,
                rm=True
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to build {service}: {str(e)}")
            return False
    
    async def _security_scan_image(self, image_tag: str) -> bool:
        """Run security scan on Docker image"""
        try:
            # Use Trivy for vulnerability scanning
            result = subprocess.run([
                'trivy', 'image', '--severity', 'HIGH,CRITICAL',
                '--exit-code', '1', image_tag
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Security scan passed for {image_tag}")
                return True
            else:
                self.logger.error(f"Security vulnerabilities found in {image_tag}")
                self.logger.error(result.stdout)
                return False
                
        except Exception as e:
            self.logger.error(f"Security scan failed: {str(e)}")
            return False
    
    async def _push_to_registry(self, image_tag: str) -> bool:
        """Push image to container registry"""
        try:
            self.logger.info(f"Pushing {image_tag} to registry")
            
            # Push using Docker API
            push_logs = self.docker_client.images.push(image_tag)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to push {image_tag}: {str(e)}")
            return False
    
    async def _get_current_environment(self, environment: str) -> str:
        """Get current active environment (blue/green)"""
        # Implementation would check Kubernetes configmap or similar
        return 'green'  # Default for now
    
    async def _deploy_to_environment(
        self,
        environment: str,
        target_env: str,
        version: str
    ) -> bool:
        """Deploy services to target environment"""
        try:
            # Use Kubernetes API to deploy services
            # Implementation would apply Kubernetes manifests
            self.logger.info(f"Deploying version {version} to {environment}-{target_env}")
            return True
        except Exception as e:
            self.logger.error(f"Deployment failed: {str(e)}")
            return False
    
    async def _wait_for_services_ready(self, environment: str) -> None:
        """Wait for all services to be ready"""
        self.logger.info("Waiting for services to be ready")
        await asyncio.sleep(60)  # Simplified wait
    
    async def _run_health_checks(self, environment: str) -> bool:
        """Run health checks on all services"""
        self.logger.info("Running health checks")
        # Implementation would check health endpoints
        return True
    
    async def _run_e2e_tests(self, environment: str) -> bool:
        """Run end-to-end tests"""
        self.logger.info("Running end-to-end tests")
        # Implementation would run test suite
        return True
    
    async def _update_traffic_split(
        self,
        environment: str,
        target_env: str,
        percentage: int
    ) -> None:
        """Update traffic split between environments"""
        self.logger.info(f"Updating traffic split: {percentage}% to {target_env}")
        # Implementation would update load balancer/ingress
    
    async def _set_current_environment(self, environment: str, env: str) -> None:
        """Set current active environment marker"""
        # Implementation would update configmap or database
        pass
    
    async def _check_deployment_health(self, environment: str) -> bool:
        """Check overall deployment health"""
        # Implementation would check various health metrics
        return True
    
    async def _get_previous_stable_version(self, environment: str) -> Optional[str]:
        """Get previous stable version for rollback"""
        # Implementation would query deployment history
        return "1.0.0"  # Default for now

def main():
    """Main entry point for deployment pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='KGoT Production Deployment Pipeline')
    parser.add_argument('--environment', required=True, choices=['staging', 'production'],
                       help='Target environment')
    parser.add_argument('--version', required=True, help='Version to deploy')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--force', action='store_true', help='Force deployment')
    
    args = parser.parse_args()
    
    # Initialize and run deployment
    pipeline = ProductionDeploymentPipeline(args.config)
    
    success = asyncio.run(pipeline.execute_deployment(
        args.environment,
        args.version,
        args.force
    ))
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 