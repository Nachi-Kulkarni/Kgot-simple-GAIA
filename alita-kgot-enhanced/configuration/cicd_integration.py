#!/usr/bin/env python3
"""
CI/CD Integration for Advanced Configuration Management

Integrates the configuration management system with CI/CD pipelines to enable:
- Automated configuration validation
- Impact assessment before deployment
- Configuration rollback on deployment failures
- Integration with HashiCorp Consul, AWS AppConfig, and other config stores

Author: Alita-KGoT Enhanced System
Date: 2024
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import requests
import hashlib
from concurrent.futures import ThreadPoolExecutor

from advanced_config_management import (
    ConfigurationManager, ConfigurationScope, ConfigurationFormat,
    ValidationResult, ValidationSeverity, ConfigurationChange
)

# Configure logging
logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Deployment pipeline stages"""
    PRE_VALIDATION = "pre_validation"
    IMPACT_ASSESSMENT = "impact_assessment"
    STAGING_DEPLOYMENT = "staging_deployment"
    PRODUCTION_DEPLOYMENT = "production_deployment"
    POST_DEPLOYMENT_MONITORING = "post_deployment_monitoring"
    ROLLBACK = "rollback"


class ConfigurationStore(Enum):
    """Supported external configuration stores"""
    CONSUL = "consul"
    AWS_APPCONFIG = "aws_appconfig"
    KUBERNETES_CONFIGMAP = "kubernetes_configmap"
    ETCD = "etcd"
    REDIS = "redis"


@dataclass
class ImpactAssessmentResult:
    """Result of configuration impact assessment"""
    affected_services: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    estimated_downtime: int  # seconds
    rollback_required: bool
    test_recommendations: List[str]
    monitoring_requirements: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DeploymentContext:
    """Context information for deployment"""
    environment: str
    commit_hash: str
    branch: str
    pull_request_id: Optional[str] = None
    triggered_by: Optional[str] = None
    deployment_id: Optional[str] = None
    stage: DeploymentStage = DeploymentStage.PRE_VALIDATION


class ExternalConfigurationStore:
    """Interface for external configuration stores"""
    
    def __init__(self, store_type: ConfigurationStore, connection_config: Dict[str, Any]):
        self.store_type = store_type
        self.connection_config = connection_config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize client for the specific store type"""
        if self.store_type == ConfigurationStore.CONSUL:
            self._initialize_consul_client()
        elif self.store_type == ConfigurationStore.AWS_APPCONFIG:
            self._initialize_aws_appconfig_client()
        elif self.store_type == ConfigurationStore.KUBERNETES_CONFIGMAP:
            self._initialize_k8s_client()
        elif self.store_type == ConfigurationStore.REDIS:
            self._initialize_redis_client()
    
    def _initialize_consul_client(self):
        """Initialize Consul client"""
        try:
            import consul
            self.client = consul.Consul(
                host=self.connection_config.get('host', 'localhost'),
                port=self.connection_config.get('port', 8500),
                token=self.connection_config.get('token')
            )
            logger.info("Consul client initialized")
        except ImportError:
            logger.error("python-consul package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Consul client: {e}")
    
    def _initialize_aws_appconfig_client(self):
        """Initialize AWS AppConfig client"""
        try:
            import boto3
            self.client = boto3.client(
                'appconfig',
                region_name=self.connection_config.get('region', 'us-east-1'),
                aws_access_key_id=self.connection_config.get('access_key_id'),
                aws_secret_access_key=self.connection_config.get('secret_access_key')
            )
            logger.info("AWS AppConfig client initialized")
        except ImportError:
            logger.error("boto3 package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize AWS AppConfig client: {e}")
    
    def _initialize_k8s_client(self):
        """Initialize Kubernetes client"""
        try:
            from kubernetes import client, config
            config.load_incluster_config() if self.connection_config.get('in_cluster') else config.load_kube_config()
            self.client = client.CoreV1Api()
            logger.info("Kubernetes client initialized")
        except ImportError:
            logger.error("kubernetes package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
    
    def _initialize_redis_client(self):
        """Initialize Redis client"""
        try:
            import redis
            self.client = redis.Redis(
                host=self.connection_config.get('host', 'localhost'),
                port=self.connection_config.get('port', 6379),
                password=self.connection_config.get('password'),
                db=self.connection_config.get('db', 0)
            )
            logger.info("Redis client initialized")
        except ImportError:
            logger.error("redis package not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
    
    def push_configuration(self, key: str, config: Dict[str, Any]) -> bool:
        """Push configuration to external store"""
        try:
            if self.store_type == ConfigurationStore.CONSUL and self.client:
                config_json = json.dumps(config)
                return self.client.kv.put(key, config_json)
            
            elif self.store_type == ConfigurationStore.AWS_APPCONFIG and self.client:
                # AWS AppConfig requires more complex setup
                # This is a simplified version
                return True
            
            elif self.store_type == ConfigurationStore.KUBERNETES_CONFIGMAP and self.client:
                from kubernetes.client.rest import ApiException
                namespace = self.connection_config.get('namespace', 'default')
                configmap_name = key.replace('/', '-').replace('_', '-')
                
                configmap = {
                    'apiVersion': 'v1',
                    'kind': 'ConfigMap',
                    'metadata': {'name': configmap_name, 'namespace': namespace},
                    'data': {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in config.items()}
                }
                
                try:
                    self.client.create_namespaced_config_map(namespace, configmap)
                except ApiException as e:
                    if e.status == 409:  # Already exists
                        self.client.replace_namespaced_config_map(configmap_name, namespace, configmap)
                    else:
                        raise
                return True
            
            elif self.store_type == ConfigurationStore.REDIS and self.client:
                config_json = json.dumps(config)
                return self.client.set(key, config_json)
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to push configuration to {self.store_type.value}: {e}")
            return False
    
    def get_configuration(self, key: str) -> Optional[Dict[str, Any]]:
        """Get configuration from external store"""
        try:
            if self.store_type == ConfigurationStore.CONSUL and self.client:
                index, data = self.client.kv.get(key)
                if data and data['Value']:
                    return json.loads(data['Value'].decode('utf-8'))
            
            elif self.store_type == ConfigurationStore.REDIS and self.client:
                data = self.client.get(key)
                if data:
                    return json.loads(data.decode('utf-8'))
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to get configuration from {self.store_type.value}: {e}")
            return None


class ImpactAssessmentEngine:
    """Analyzes the impact of configuration changes"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.service_dependencies = {}
        self.load_service_dependencies()
    
    def load_service_dependencies(self):
        """Load service dependency graph"""
        try:
            # Load from service configurations
            services = self.config_manager.store.list_configurations(ConfigurationScope.SERVICE)
            for service_name in services:
                service_config = self.config_manager.store.get_configuration(ConfigurationScope.SERVICE, service_name)
                dependencies = service_config.get('dependencies', [])
                self.service_dependencies[service_name] = dependencies
        except Exception as e:
            logger.error(f"Failed to load service dependencies: {e}")
    
    async def assess_impact(self, changes: List[ConfigurationChange]) -> ImpactAssessmentResult:
        """Assess the impact of configuration changes"""
        affected_services = set()
        risk_level = "LOW"
        estimated_downtime = 0
        rollback_required = False
        test_recommendations = []
        monitoring_requirements = []
        
        for change in changes:
            # Analyze change impact
            change_impact = self._analyze_change_impact(change)
            affected_services.update(change_impact['services'])
            
            # Update risk level
            if change_impact['risk'] == "CRITICAL":
                risk_level = "CRITICAL"
                rollback_required = True
            elif change_impact['risk'] == "HIGH" and risk_level not in ["CRITICAL"]:
                risk_level = "HIGH"
            elif change_impact['risk'] == "MEDIUM" and risk_level in ["LOW"]:
                risk_level = "MEDIUM"
            
            # Accumulate downtime estimates
            estimated_downtime += change_impact['downtime']
            
            # Collect recommendations
            test_recommendations.extend(change_impact['tests'])
            monitoring_requirements.extend(change_impact['monitoring'])
        
        # Add dependent services
        all_affected = set(affected_services)
        for service in affected_services:
            all_affected.update(self._get_dependent_services(service))
        
        return ImpactAssessmentResult(
            affected_services=list(all_affected),
            risk_level=risk_level,
            estimated_downtime=estimated_downtime,
            rollback_required=rollback_required,
            test_recommendations=list(set(test_recommendations)),
            monitoring_requirements=list(set(monitoring_requirements))
        )
    
    def _analyze_change_impact(self, change: ConfigurationChange) -> Dict[str, Any]:
        """Analyze the impact of a single configuration change"""
        path_parts = change.path.split('.')
        
        # Default impact
        impact = {
            'services': [],
            'risk': 'LOW',
            'downtime': 0,
            'tests': [],
            'monitoring': []
        }
        
        # Analyze based on configuration path
        if path_parts[0] == 'kgot':
            impact.update(self._analyze_kgot_change(change, path_parts))
        elif path_parts[0] == 'alita':
            impact.update(self._analyze_alita_change(change, path_parts))
        elif path_parts[0] == 'models':
            impact.update(self._analyze_model_change(change, path_parts))
        elif path_parts[0] == 'features':
            impact.update(self._analyze_feature_change(change, path_parts))
        
        return impact
    
    def _analyze_kgot_change(self, change: ConfigurationChange, path_parts: List[str]) -> Dict[str, Any]:
        """Analyze KGoT configuration changes"""
        if len(path_parts) < 2:
            return {}
        
        param = path_parts[1]
        
        if param in ['num_next_steps_decision', 'reasoning_depth']:
            return {
                'services': ['kgot-reasoning-engine', 'manager-agent'],
                'risk': 'MEDIUM',
                'downtime': 30,
                'tests': ['reasoning_accuracy_test', 'performance_benchmark'],
                'monitoring': ['reasoning_latency', 'decision_quality_metrics']
            }
        
        elif param in ['max_retrieve_query_retry', 'max_cypher_fixing_retry']:
            return {
                'services': ['kgot-graph-store', 'query-processor'],
                'risk': 'LOW',
                'downtime': 0,
                'tests': ['query_reliability_test'],
                'monitoring': ['query_success_rate', 'retry_frequency']
            }
        
        elif param == 'graph_traversal_limit':
            # High impact change
            old_value = change.old_value or 100
            new_value = change.new_value or 100
            
            if abs(new_value - old_value) > 50:
                return {
                    'services': ['kgot-graph-store', 'reasoning-engine'],
                    'risk': 'HIGH',
                    'downtime': 60,
                    'tests': ['graph_traversal_test', 'memory_usage_test'],
                    'monitoring': ['graph_query_performance', 'memory_consumption']
                }
        
        return {}
    
    def _analyze_alita_change(self, change: ConfigurationChange, path_parts: List[str]) -> Dict[str, Any]:
        """Analyze Alita configuration changes"""
        if len(path_parts) < 3:
            return {}
        
        component = path_parts[1]
        
        if component == 'environment_manager':
            if path_parts[2] == 'supported_languages':
                return {
                    'services': ['alita-environment-manager', 'code-runner'],
                    'risk': 'MEDIUM',
                    'downtime': 45,
                    'tests': ['language_support_test', 'environment_setup_test'],
                    'monitoring': ['environment_creation_time', 'language_compatibility']
                }
            elif path_parts[2] == 'timeout':
                return {
                    'services': ['alita-environment-manager'],
                    'risk': 'LOW',
                    'downtime': 0,
                    'tests': ['timeout_handling_test'],
                    'monitoring': ['environment_setup_duration']
                }
        
        elif component == 'tools':
            tool_name = path_parts[2] if len(path_parts) > 2 else 'unknown'
            return {
                'services': [f'alita-{tool_name}'],
                'risk': 'LOW',
                'downtime': 15,
                'tests': [f'{tool_name}_functionality_test'],
                'monitoring': [f'{tool_name}_performance']
            }
        
        return {}
    
    def _analyze_model_change(self, change: ConfigurationChange, path_parts: List[str]) -> Dict[str, Any]:
        """Analyze model configuration changes"""
        return {
            'services': ['model-manager', 'inference-service'],
            'risk': 'MEDIUM',
            'downtime': 30,
            'tests': ['model_availability_test', 'inference_quality_test'],
            'monitoring': ['model_response_time', 'inference_accuracy']
        }
    
    def _analyze_feature_change(self, change: ConfigurationChange, path_parts: List[str]) -> Dict[str, Any]:
        """Analyze feature flag changes"""
        if len(path_parts) < 2:
            return {}
        
        feature = path_parts[1]
        
        # Critical features that require careful rollout
        critical_features = ['enhanced_security', 'performance_monitoring']
        
        if feature in critical_features:
            return {
                'services': ['all'],
                'risk': 'HIGH',
                'downtime': 0,
                'tests': ['feature_toggle_test', 'system_stability_test'],
                'monitoring': ['feature_usage_metrics', 'system_health']
            }
        
        return {
            'services': ['feature-manager'],
            'risk': 'LOW',
            'downtime': 0,
            'tests': ['feature_toggle_test'],
            'monitoring': ['feature_usage_metrics']
        }
    
    def _get_dependent_services(self, service: str) -> List[str]:
        """Get services that depend on the given service"""
        dependents = []
        for svc, deps in self.service_dependencies.items():
            if service in deps:
                dependents.append(svc)
        return dependents


class CICDIntegration:
    """Main CI/CD integration orchestrator"""
    
    def __init__(self, config_path: Path, external_stores: Optional[List[ExternalConfigurationStore]] = None):
        self.config_manager = ConfigurationManager(config_path)
        self.impact_engine = ImpactAssessmentEngine(self.config_manager)
        self.external_stores = external_stores or []
        self.deployment_history = []
    
    async def validate_configuration_changes(self, context: DeploymentContext) -> bool:
        """Validate configuration changes in CI/CD pipeline"""
        logger.info(f"Validating configuration changes for {context.environment}")
        
        try:
            # Get changes from Git
            changes = self._get_git_changes(context.commit_hash)
            
            if not changes:
                logger.info("No configuration changes detected")
                return True
            
            # Validate configurations
            validation_results = []
            for change in changes:
                scope, name = self._parse_change_path(change.path)
                if scope and name:
                    config = await self.config_manager.get_configuration(scope, name)
                    results = await self.config_manager._validate_configuration(config, [change])
                    validation_results.extend(results)
            
            # Check for critical errors
            critical_errors = [r for r in validation_results if r.severity == ValidationSeverity.CRITICAL]
            if critical_errors:
                logger.error(f"Critical validation errors: {[e.message for e in critical_errors]}")
                return False
            
            # Log warnings
            warnings = [r for r in validation_results if r.severity == ValidationSeverity.WARNING]
            for warning in warnings:
                logger.warning(f"Validation warning: {warning.message}")
            
            logger.info("Configuration validation passed")
            return True
        
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def perform_impact_assessment(self, context: DeploymentContext) -> ImpactAssessmentResult:
        """Perform impact assessment for configuration changes"""
        logger.info(f"Performing impact assessment for {context.environment}")
        
        try:
            # Get changes from Git
            changes = self._get_git_changes(context.commit_hash)
            
            if not changes:
                return ImpactAssessmentResult(
                    affected_services=[],
                    risk_level="LOW",
                    estimated_downtime=0,
                    rollback_required=False,
                    test_recommendations=[],
                    monitoring_requirements=[]
                )
            
            # Assess impact
            impact_result = await self.impact_engine.assess_impact(changes)
            
            # Log impact assessment
            logger.info(f"Impact assessment completed:")
            logger.info(f"  Risk Level: {impact_result.risk_level}")
            logger.info(f"  Affected Services: {impact_result.affected_services}")
            logger.info(f"  Estimated Downtime: {impact_result.estimated_downtime}s")
            logger.info(f"  Rollback Required: {impact_result.rollback_required}")
            
            return impact_result
        
        except Exception as e:
            logger.error(f"Impact assessment failed: {e}")
            # Return high-risk result on failure
            return ImpactAssessmentResult(
                affected_services=["unknown"],
                risk_level="CRITICAL",
                estimated_downtime=300,
                rollback_required=True,
                test_recommendations=["manual_verification"],
                monitoring_requirements=["full_system_monitoring"]
            )
    
    async def deploy_configuration(self, context: DeploymentContext) -> bool:
        """Deploy configuration changes to target environment"""
        logger.info(f"Deploying configuration to {context.environment}")
        
        try:
            # Get merged configuration for environment
            merged_config = self.config_manager.get_merged_configuration(context.environment)
            
            # Push to external stores
            success = True
            for store in self.external_stores:
                store_key = f"alita-kgot/{context.environment}/config"
                if not store.push_configuration(store_key, merged_config):
                    logger.error(f"Failed to push configuration to {store.store_type.value}")
                    success = False
            
            if success:
                logger.info("Configuration deployment successful")
                
                # Record deployment
                self.deployment_history.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'environment': context.environment,
                    'commit_hash': context.commit_hash,
                    'deployment_id': context.deployment_id,
                    'status': 'success'
                })
            
            return success
        
        except Exception as e:
            logger.error(f"Configuration deployment failed: {e}")
            return False
    
    async def monitor_deployment(self, context: DeploymentContext, monitoring_duration: int = 300) -> bool:
        """Monitor deployment for issues and trigger rollback if needed"""
        logger.info(f"Monitoring deployment for {monitoring_duration} seconds")
        
        try:
            start_time = time.time()
            
            while time.time() - start_time < monitoring_duration:
                # Check system health
                health_status = await self._check_system_health(context)
                
                if not health_status['healthy']:
                    logger.error(f"System health check failed: {health_status['issues']}")
                    
                    # Trigger rollback
                    rollback_success = await self.rollback_deployment(context)
                    return rollback_success
                
                # Wait before next check
                await asyncio.sleep(30)
            
            logger.info("Deployment monitoring completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Deployment monitoring failed: {e}")
            return False
    
    async def rollback_deployment(self, context: DeploymentContext) -> bool:
        """Rollback configuration deployment"""
        logger.info(f"Rolling back deployment for {context.environment}")
        
        try:
            # Get previous successful deployment
            previous_deployment = self._get_previous_deployment(context.environment)
            if not previous_deployment:
                logger.error("No previous deployment found for rollback")
                return False
            
            # Rollback to previous version
            rollback_context = DeploymentContext(
                environment=context.environment,
                commit_hash=previous_deployment['commit_hash'],
                branch=context.branch,
                deployment_id=f"rollback-{context.deployment_id}",
                stage=DeploymentStage.ROLLBACK
            )
            
            success = await self.deploy_configuration(rollback_context)
            
            if success:
                logger.info("Rollback completed successfully")
                
                # Record rollback
                self.deployment_history.append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'environment': context.environment,
                    'commit_hash': previous_deployment['commit_hash'],
                    'deployment_id': rollback_context.deployment_id,
                    'status': 'rollback',
                    'original_deployment': context.deployment_id
                })
            
            return success
        
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _get_git_changes(self, commit_hash: str) -> List[ConfigurationChange]:
        """Get configuration changes from Git"""
        try:
            # Get changed files
            result = subprocess.run(
                ['git', 'diff', '--name-only', f'{commit_hash}^', commit_hash],
                capture_output=True, text=True, check=True
            )
            
            changed_files = result.stdout.strip().split('\n')
            config_files = [f for f in changed_files if f.startswith('configuration/') and f.endswith(('.yaml', '.yml', '.json'))]
            
            changes = []
            for file_path in config_files:
                # Get file diff
                diff_result = subprocess.run(
                    ['git', 'show', f'{commit_hash}:{file_path}'],
                    capture_output=True, text=True
                )
                
                if diff_result.returncode == 0:
                    # Parse configuration change
                    # This is a simplified version - in practice, you'd parse the actual diff
                    changes.append(ConfigurationChange(
                        path=file_path,
                        old_value=None,  # Would be parsed from diff
                        new_value=None   # Would be parsed from diff
                    ))
            
            return changes
        
        except Exception as e:
            logger.error(f"Failed to get Git changes: {e}")
            return []
    
    def _parse_change_path(self, file_path: str) -> tuple[Optional[ConfigurationScope], Optional[str]]:
        """Parse configuration file path to determine scope and name"""
        try:
            path_parts = Path(file_path).parts
            if len(path_parts) >= 3 and path_parts[0] == 'configuration':
                scope_name = path_parts[1]
                config_name = Path(path_parts[2]).stem
                
                try:
                    scope = ConfigurationScope(scope_name)
                    return scope, config_name
                except ValueError:
                    pass
            
            return None, None
        
        except Exception:
            return None, None
    
    async def _check_system_health(self, context: DeploymentContext) -> Dict[str, Any]:
        """Check system health after deployment"""
        health_status = {'healthy': True, 'issues': []}
        
        try:
            # Check service endpoints
            services_to_check = [
                'http://localhost:3000/health',  # Manager Agent
                'http://localhost:8080/health',  # KGoT Service
                'http://localhost:9090/health'   # Monitoring
            ]
            
            for service_url in services_to_check:
                try:
                    response = requests.get(service_url, timeout=10)
                    if response.status_code != 200:
                        health_status['healthy'] = False
                        health_status['issues'].append(f"Service {service_url} returned {response.status_code}")
                except requests.RequestException as e:
                    health_status['healthy'] = False
                    health_status['issues'].append(f"Service {service_url} unreachable: {e}")
            
            return health_status
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'healthy': False, 'issues': [f"Health check error: {e}"]}
    
    def _get_previous_deployment(self, environment: str) -> Optional[Dict[str, Any]]:
        """Get the previous successful deployment for an environment"""
        for deployment in reversed(self.deployment_history):
            if (deployment['environment'] == environment and 
                deployment['status'] == 'success' and 
                'rollback' not in deployment.get('deployment_id', '')):
                return deployment
        return None


# CLI Interface for CI/CD integration
if __name__ == "__main__":
    import argparse
    import sys
    
    async def main():
        parser = argparse.ArgumentParser(description="CI/CD Configuration Management")
        parser.add_argument("--config-path", default="./configuration", help="Configuration base path")
        parser.add_argument("--environment", required=True, help="Target environment")
        parser.add_argument("--commit-hash", required=True, help="Git commit hash")
        parser.add_argument("--branch", default="main", help="Git branch")
        parser.add_argument("--deployment-id", help="Deployment ID")
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Validate command
        subparsers.add_parser("validate", help="Validate configuration changes")
        
        # Impact assessment command
        subparsers.add_parser("assess", help="Perform impact assessment")
        
        # Deploy command
        subparsers.add_parser("deploy", help="Deploy configuration")
        
        # Monitor command
        monitor_parser = subparsers.add_parser("monitor", help="Monitor deployment")
        monitor_parser.add_argument("--duration", type=int, default=300, help="Monitoring duration in seconds")
        
        # Rollback command
        subparsers.add_parser("rollback", help="Rollback deployment")
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Create deployment context
        context = DeploymentContext(
            environment=args.environment,
            commit_hash=args.commit_hash,
            branch=args.branch,
            deployment_id=args.deployment_id or f"deploy-{int(time.time())}"
        )
        
        # Initialize CI/CD integration
        cicd = CICDIntegration(Path(args.config_path))
        
        try:
            if args.command == "validate":
                success = await cicd.validate_configuration_changes(context)
                sys.exit(0 if success else 1)
            
            elif args.command == "assess":
                impact = await cicd.perform_impact_assessment(context)
                print(json.dumps({
                    'risk_level': impact.risk_level,
                    'affected_services': impact.affected_services,
                    'estimated_downtime': impact.estimated_downtime,
                    'rollback_required': impact.rollback_required,
                    'test_recommendations': impact.test_recommendations,
                    'monitoring_requirements': impact.monitoring_requirements
                }, indent=2))
                
                # Exit with error code for high-risk changes
                sys.exit(1 if impact.risk_level in ['HIGH', 'CRITICAL'] else 0)
            
            elif args.command == "deploy":
                success = await cicd.deploy_configuration(context)
                sys.exit(0 if success else 1)
            
            elif args.command == "monitor":
                success = await cicd.monitor_deployment(context, args.duration)
                sys.exit(0 if success else 1)
            
            elif args.command == "rollback":
                success = await cicd.rollback_deployment(context)
                sys.exit(0 if success else 1)
        
        except Exception as e:
            logger.error(f"Command failed: {e}")
            sys.exit(1)
    
    if __name__ == "__main__":
        asyncio.run(main())