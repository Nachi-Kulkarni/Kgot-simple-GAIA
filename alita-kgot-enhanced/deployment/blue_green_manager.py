#!/usr/bin/env python3
"""
Blue-Green Deployment Manager
============================

This module implements blue-green deployment strategy for the KGoT-Alita
enhanced system, providing zero-downtime deployments with automated
rollback capabilities.

Features:
- Blue-green environment management
- Gradual traffic switching (canary deployment)
- Health monitoring and validation
- Automated rollback on failure
- Load balancer integration

Author: Alita KGoT Enhanced Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import kubernetes
from kubernetes import client, config
import aiohttp
import prometheus_client

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.logging.winston_config import setup_winston_logging

@dataclass
class EnvironmentState:
    """State information for a deployment environment"""
    name: str  # 'blue' or 'green'
    version: str
    status: str  # 'active', 'inactive', 'deploying', 'failed'
    health_score: float
    last_deployed: datetime
    traffic_percentage: int = 0
    service_health: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.service_health is None:
            self.service_health = {}

@dataclass
class DeploymentTarget:
    """Configuration for a deployment target"""
    environment: str  # 'staging' or 'production'
    namespace: str
    domain: str
    services: List[str]
    resources: Dict[str, Any]

class BlueGreenDeploymentManager:
    """
    Manages blue-green deployments with zero-downtime switching
    and comprehensive health monitoring.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the blue-green deployment manager
        
        Args:
            config_path: Path to deployment configuration
        """
        self.logger = setup_winston_logging(
            'blue_green_deployment',
            log_file='logs/deployment/blue_green.log'
        )
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()  # Try in-cluster first
        except:
            config.load_kube_config()  # Fall back to local config
        
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_networking_v1 = client.NetworkingV1Api()
        
        # Environment state tracking
        self.environments: Dict[str, Dict[str, EnvironmentState]] = {}
        
        # Service definitions
        self.services = [
            'kgot-controller',
            'graph-store',
            'manager-agent',
            'web-agent',
            'monitoring'
        ]
        
        # Health check configuration
        self.health_check_timeout = 300  # 5 minutes
        self.health_check_interval = 10  # 10 seconds
        
        self.logger.info("Blue-green deployment manager initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            'health_check': {
                'timeout': 300,
                'interval': 10,
                'retries': 3
            },
            'traffic_switching': {
                'gradual': True,
                'steps': [10, 25, 50, 75, 100],
                'step_duration': 60
            },
            'rollback': {
                'auto_trigger': True,
                'health_threshold': 0.8,
                'error_threshold': 0.05
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
            
            # Merge configurations
            default_config.update(user_config)
        
        return default_config
    
    async def deploy_to_environment(
        self,
        target: DeploymentTarget,
        version: str,
        image_registry: str
    ) -> bool:
        """
        Deploy application to blue or green environment
        
        Args:
            target: Deployment target configuration
            version: Version to deploy
            image_registry: Container registry URL
            
        Returns:
            True if deployment successful
        """
        try:
            # Determine target environment (blue/green)
            current_env = await self._get_active_environment(target.environment)
            target_env = 'blue' if current_env == 'green' else 'green'
            
            self.logger.info(f"Deploying version {version} to {target_env} environment")
            
            # Initialize environment state
            env_state = EnvironmentState(
                name=target_env,
                version=version,
                status='deploying',
                health_score=0.0,
                last_deployed=datetime.now()
            )
            
            # Ensure environment state tracking exists
            if target.environment not in self.environments:
                self.environments[target.environment] = {}
            
            self.environments[target.environment][target_env] = env_state
            
            # Deploy services to target environment
            for service in target.services:
                success = await self._deploy_service(
                    service, target, target_env, version, image_registry
                )
                if not success:
                    env_state.status = 'failed'
                    raise Exception(f"Failed to deploy service: {service}")
            
            # Wait for services to be ready
            if not await self._wait_for_services_ready(target, target_env):
                env_state.status = 'failed'
                raise Exception("Services failed to become ready")
            
            # Perform health checks
            health_score = await self._perform_health_checks(target, target_env)
            env_state.health_score = health_score
            
            if health_score < self.config.get('rollback', {}).get('health_threshold', 0.8):
                env_state.status = 'failed'
                raise Exception(f"Health check failed. Score: {health_score}")
            
            env_state.status = 'inactive'  # Ready but not receiving traffic
            self.logger.info(f"Successfully deployed to {target_env} environment")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment to {target_env} failed: {str(e)}")
            if target.environment in self.environments and target_env in self.environments[target.environment]:
                self.environments[target.environment][target_env].status = 'failed'
            return False
    
    async def switch_traffic(
        self,
        target: DeploymentTarget,
        force_immediate: bool = False
    ) -> bool:
        """
        Switch traffic from current to new environment
        
        Args:
            target: Deployment target
            force_immediate: Skip gradual switching
            
        Returns:
            True if traffic switch successful
        """
        try:
            current_env = await self._get_active_environment(target.environment)
            new_env = 'blue' if current_env == 'green' else 'green'
            
            self.logger.info(f"Switching traffic from {current_env} to {new_env}")
            
            # Check if new environment is ready
            if (target.environment not in self.environments or 
                new_env not in self.environments[target.environment]):
                raise Exception(f"Target environment {new_env} not found")
            
            new_env_state = self.environments[target.environment][new_env]
            if new_env_state.status != 'inactive':
                raise Exception(f"Target environment {new_env} is not ready")
            
            if force_immediate:
                # Immediate switch
                await self._update_traffic_routing(target, new_env, 100)
                await self._set_active_environment(target.environment, new_env)
                new_env_state.status = 'active'
                new_env_state.traffic_percentage = 100
                
                if current_env in self.environments[target.environment]:
                    self.environments[target.environment][current_env].status = 'inactive'
                    self.environments[target.environment][current_env].traffic_percentage = 0
            else:
                # Gradual traffic switching
                traffic_steps = self.config.get('traffic_switching', {}).get('steps', [10, 25, 50, 75, 100])
                step_duration = self.config.get('traffic_switching', {}).get('step_duration', 60)
                
                for percentage in traffic_steps:
                    self.logger.info(f"Switching {percentage}% traffic to {new_env}")
                    
                    # Update traffic routing
                    await self._update_traffic_routing(target, new_env, percentage)
                    new_env_state.traffic_percentage = percentage
                    
                    if current_env in self.environments[target.environment]:
                        self.environments[target.environment][current_env].traffic_percentage = 100 - percentage
                    
                    # Monitor health during switch
                    await asyncio.sleep(step_duration)
                    
                    health_score = await self._check_environment_health(target, new_env)
                    if health_score < self.config.get('rollback', {}).get('health_threshold', 0.8):
                        self.logger.warning(f"Health degradation detected at {percentage}% traffic")
                        # Automatic rollback
                        await self._rollback_traffic(target, current_env)
                        return False
                
                # Complete the switch
                await self._set_active_environment(target.environment, new_env)
                new_env_state.status = 'active'
                
                if current_env in self.environments[target.environment]:
                    self.environments[target.environment][current_env].status = 'inactive'
            
            self.logger.info(f"Traffic successfully switched to {new_env}")
            return True
            
        except Exception as e:
            self.logger.error(f"Traffic switch failed: {str(e)}")
            return False
    
    async def rollback_deployment(
        self,
        target: DeploymentTarget,
        reason: str = "manual"
    ) -> bool:
        """
        Rollback to previous stable environment
        
        Args:
            target: Deployment target
            reason: Reason for rollback
            
        Returns:
            True if rollback successful
        """
        try:
            current_env = await self._get_active_environment(target.environment)
            previous_env = 'blue' if current_env == 'green' else 'green'
            
            self.logger.warning(f"Rolling back from {current_env} to {previous_env}. Reason: {reason}")
            
            # Check if previous environment exists and is stable
            if (target.environment not in self.environments or 
                previous_env not in self.environments[target.environment]):
                raise Exception(f"Previous environment {previous_env} not available")
            
            previous_env_state = self.environments[target.environment][previous_env]
            if previous_env_state.status == 'failed':
                raise Exception(f"Previous environment {previous_env} is in failed state")
            
            # Immediate traffic switch to previous environment
            await self._update_traffic_routing(target, previous_env, 100)
            await self._set_active_environment(target.environment, previous_env)
            
            # Update environment states
            previous_env_state.status = 'active'
            previous_env_state.traffic_percentage = 100
            
            if current_env in self.environments[target.environment]:
                self.environments[target.environment][current_env].status = 'failed'
                self.environments[target.environment][current_env].traffic_percentage = 0
            
            self.logger.info(f"Rollback completed successfully to {previous_env}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {str(e)}")
            return False
    
    async def get_environment_status(self, environment: str) -> Dict[str, Any]:
        """
        Get current status of both blue and green environments
        
        Args:
            environment: Environment name (staging/production)
            
        Returns:
            Environment status information
        """
        if environment not in self.environments:
            return {'blue': None, 'green': None, 'active': None}
        
        active_env = await self._get_active_environment(environment)
        
        status = {
            'active': active_env,
            'blue': None,
            'green': None
        }
        
        for env_name, env_state in self.environments[environment].items():
            status[env_name] = {
                'version': env_state.version,
                'status': env_state.status,
                'health_score': env_state.health_score,
                'traffic_percentage': env_state.traffic_percentage,
                'last_deployed': env_state.last_deployed.isoformat(),
                'service_health': env_state.service_health
            }
        
        return status
    
    # Private helper methods
    
    async def _deploy_service(
        self,
        service: str,
        target: DeploymentTarget,
        environment: str,
        version: str,
        registry: str
    ) -> bool:
        """Deploy a single service to target environment"""
        try:
            # Create deployment manifest
            deployment = self._create_deployment_manifest(
                service, target, environment, version, registry
            )
            
            # Apply deployment
            deployment_name = f"{service}-{environment}"
            
            try:
                # Try to update existing deployment
                self.k8s_apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=target.namespace,
                    body=deployment
                )
            except client.ApiException as e:
                if e.status == 404:
                    # Create new deployment
                    self.k8s_apps_v1.create_namespaced_deployment(
                        namespace=target.namespace,
                        body=deployment
                    )
                else:
                    raise
            
            # Create or update service
            service_manifest = self._create_service_manifest(service, target, environment)
            service_name = f"{service}-{environment}"
            
            try:
                self.k8s_core_v1.patch_namespaced_service(
                    name=service_name,
                    namespace=target.namespace,
                    body=service_manifest
                )
            except client.ApiException as e:
                if e.status == 404:
                    self.k8s_core_v1.create_namespaced_service(
                        namespace=target.namespace,
                        body=service_manifest
                    )
                else:
                    raise
            
            self.logger.info(f"Successfully deployed {service} to {environment}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy {service}: {str(e)}")
            return False
    
    def _create_deployment_manifest(
        self,
        service: str,
        target: DeploymentTarget,
        environment: str,
        version: str,
        registry: str
    ) -> client.V1Deployment:
        """Create Kubernetes deployment manifest"""
        
        # Get resource configuration
        resources = target.resources.get(service, target.resources.get('default', {}))
        replicas = target.resources.get('replicas', {}).get(service, 1)
        
        # Container configuration
        container = client.V1Container(
            name=service,
            image=f"{registry}/{service}:{version}",
            ports=[client.V1ContainerPort(container_port=8080)],
            resources=client.V1ResourceRequirements(
                requests=resources.get('requests', {}),
                limits=resources.get('limits', {})
            ),
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path="/health",
                    port=8080
                ),
                initial_delay_seconds=60,
                period_seconds=30,
                timeout_seconds=10,
                failure_threshold=3
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path="/ready",
                    port=8080
                ),
                initial_delay_seconds=30,
                period_seconds=10,
                timeout_seconds=5,
                failure_threshold=3
            ),
            env=[
                client.V1EnvVar(
                    name="ENVIRONMENT",
                    value=environment
                ),
                client.V1EnvVar(
                    name="VERSION",
                    value=version
                )
            ]
        )
        
        # Pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={
                    "app": service,
                    "environment": environment,
                    "version": version
                }
            ),
            spec=client.V1PodSpec(
                containers=[container]
            )
        )
        
        # Deployment spec
        deployment_spec = client.V1DeploymentSpec(
            replicas=replicas,
            selector=client.V1LabelSelector(
                match_labels={
                    "app": service,
                    "environment": environment
                }
            ),
            template=pod_template,
            strategy=client.V1DeploymentStrategy(
                type="RollingUpdate",
                rolling_update=client.V1RollingUpdateDeployment(
                    max_unavailable="25%",
                    max_surge="25%"
                )
            )
        )
        
        # Deployment object
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=f"{service}-{environment}",
                namespace=target.namespace,
                labels={
                    "app": service,
                    "environment": environment,
                    "version": version
                }
            ),
            spec=deployment_spec
        )
        
        return deployment
    
    def _create_service_manifest(
        self,
        service: str,
        target: DeploymentTarget,
        environment: str
    ) -> client.V1Service:
        """Create Kubernetes service manifest"""
        
        service_spec = client.V1ServiceSpec(
            selector={
                "app": service,
                "environment": environment
            },
            ports=[
                client.V1ServicePort(
                    port=80,
                    target_port=8080,
                    protocol="TCP"
                )
            ],
            type="ClusterIP"
        )
        
        service_obj = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=f"{service}-{environment}",
                namespace=target.namespace,
                labels={
                    "app": service,
                    "environment": environment
                }
            ),
            spec=service_spec
        )
        
        return service_obj
    
    async def _wait_for_services_ready(
        self,
        target: DeploymentTarget,
        environment: str
    ) -> bool:
        """Wait for all services to be ready"""
        self.logger.info(f"Waiting for services to be ready in {environment}")
        
        timeout = time.time() + self.health_check_timeout
        
        while time.time() < timeout:
            all_ready = True
            
            for service in target.services:
                deployment_name = f"{service}-{environment}"
                
                try:
                    deployment = self.k8s_apps_v1.read_namespaced_deployment(
                        name=deployment_name,
                        namespace=target.namespace
                    )
                    
                    ready_replicas = deployment.status.ready_replicas or 0
                    desired_replicas = deployment.spec.replicas or 0
                    
                    if ready_replicas < desired_replicas:
                        all_ready = False
                        break
                        
                except client.ApiException:
                    all_ready = False
                    break
            
            if all_ready:
                self.logger.info("All services are ready")
                return True
            
            await asyncio.sleep(self.health_check_interval)
        
        self.logger.error("Timeout waiting for services to be ready")
        return False
    
    async def _perform_health_checks(
        self,
        target: DeploymentTarget,
        environment: str
    ) -> float:
        """Perform comprehensive health checks and return health score"""
        self.logger.info(f"Performing health checks for {environment}")
        
        healthy_services = 0
        total_services = len(target.services)
        
        env_state = self.environments[target.environment][environment]
        
        for service in target.services:
            is_healthy = await self._check_service_health(target, service, environment)
            env_state.service_health[service] = is_healthy
            
            if is_healthy:
                healthy_services += 1
        
        health_score = healthy_services / total_services if total_services > 0 else 0.0
        self.logger.info(f"Health score for {environment}: {health_score:.2f}")
        
        return health_score
    
    async def _check_service_health(
        self,
        target: DeploymentTarget,
        service: str,
        environment: str
    ) -> bool:
        """Check health of a single service"""
        try:
            # Use port-forward to check health endpoint
            service_name = f"{service}-{environment}"
            
            # For now, check if deployment is ready
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace=target.namespace
            )
            
            ready_replicas = deployment.status.ready_replicas or 0
            desired_replicas = deployment.spec.replicas or 0
            
            return ready_replicas == desired_replicas and desired_replicas > 0
            
        except Exception as e:
            self.logger.error(f"Health check failed for {service}: {str(e)}")
            return False
    
    async def _check_environment_health(
        self,
        target: DeploymentTarget,
        environment: str
    ) -> float:
        """Check current health of environment"""
        return await self._perform_health_checks(target, environment)
    
    async def _update_traffic_routing(
        self,
        target: DeploymentTarget,
        environment: str,
        percentage: int
    ) -> None:
        """Update traffic routing to environment"""
        self.logger.info(f"Updating traffic routing: {percentage}% to {environment}")
        
        # This would update ingress/load balancer configuration
        # Implementation depends on specific load balancer (nginx, istio, etc.)
        
        # For demonstration, we'll update ingress annotations
        try:
            ingress_name = f"kgot-ingress"
            
            # Get current ingress
            ingress = self.k8s_networking_v1.read_namespaced_ingress(
                name=ingress_name,
                namespace=target.namespace
            )
            
            # Update annotations for traffic splitting
            if not ingress.metadata.annotations:
                ingress.metadata.annotations = {}
            
            ingress.metadata.annotations[f"nginx.ingress.kubernetes.io/canary"] = "true"
            ingress.metadata.annotations[f"nginx.ingress.kubernetes.io/canary-weight"] = str(percentage)
            
            # Apply update
            self.k8s_networking_v1.patch_namespaced_ingress(
                name=ingress_name,
                namespace=target.namespace,
                body=ingress
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to update ingress routing: {str(e)}")
    
    async def _rollback_traffic(
        self,
        target: DeploymentTarget,
        environment: str
    ) -> None:
        """Rollback traffic to specified environment"""
        await self._update_traffic_routing(target, environment, 100)
    
    async def _get_active_environment(self, environment: str) -> str:
        """Get currently active environment (blue/green)"""
        # This would check configmap or similar state store
        # For now, return default
        return 'green'
    
    async def _set_active_environment(self, environment: str, env_name: str) -> None:
        """Set active environment marker"""
        # This would update configmap or state store
        self.logger.info(f"Setting {env_name} as active environment for {environment}")

def main():
    """Main entry point for blue-green deployment manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Blue-Green Deployment Manager')
    parser.add_argument('action', choices=['deploy', 'switch', 'rollback', 'status'],
                       help='Action to perform')
    parser.add_argument('--environment', required=True, choices=['staging', 'production'],
                       help='Target environment')
    parser.add_argument('--version', help='Version to deploy')
    parser.add_argument('--registry', default='localhost:5000', help='Container registry')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = BlueGreenDeploymentManager(args.config)
    
    # Create deployment target
    target = DeploymentTarget(
        environment=args.environment,
        namespace=f"kgot-{args.environment}",
        domain=f"{args.environment}.kgot.local",
        services=['kgot-controller', 'graph-store', 'manager-agent', 'web-agent', 'monitoring'],
        resources={}
    )
    
    async def run_action():
        if args.action == 'deploy':
            if not args.version:
                print("Version required for deploy action")
                return False
            
            return await manager.deploy_to_environment(target, args.version, args.registry)
        
        elif args.action == 'switch':
            return await manager.switch_traffic(target)
        
        elif args.action == 'rollback':
            return await manager.rollback_deployment(target, "manual")
        
        elif args.action == 'status':
            status = await manager.get_environment_status(args.environment)
            print(json.dumps(status, indent=2, default=str))
            return True
    
    success = asyncio.run(run_action())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 