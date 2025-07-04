#!/usr/bin/env python3
"""
KGoT Containerization Infrastructure

This module provides comprehensive containerization support for the Knowledge Graph of Thoughts (KGoT)
system integrated with Alita. It supports both Docker for standard deployments and Sarus for HPC
environments, with automatic environment detection, resource management, and integration with existing systems

Features:
- Automatic environment detection (Docker vs Sarus vs Cloud)
- Container lifecycle management (start, stop, restart, scale)
- Resource allocation and cost optimization integration
- Health monitoring and performance metrics
- LangChain agent container support
- Integration with existing Winston logging and error management
- Support for local, cloud, and HPC deployments

Author: KGoT Enhanced Alita System
Created: 2024
License: MIT
"""

import os
import sys
import json
import time
import logging
import asyncio
import subprocess
import platform
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import docker
import psutil
import requests
from docker.client import DockerClient
from docker.models.containers import Container
from docker.errors import DockerException, NotFound, APIError

# Configure logging to integrate with Winston-style logging
import pathlib
log_dir = pathlib.Path('./logs/kgot')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/kgot/containerization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('KGoTContainerization')


class DeploymentEnvironment(Enum):
    """
    Enumeration of supported deployment environments
    
    LOCAL_DOCKER: Standard Docker deployment on local machine
    CLOUD_DOCKER: Docker deployment in cloud environment (AWS, GCP, Azure)
    HPC_SARUS: Sarus deployment for High Performance Computing environments
    KUBERNETES: Kubernetes orchestrated deployment
    HYBRID: Mixed environment deployment
    """
    LOCAL_DOCKER = "local_docker"
    CLOUD_DOCKER = "cloud_docker" 
    HPC_SARUS = "hpc_sarus"
    KUBERNETES = "kubernetes"
    HYBRID = "hybrid"


class ContainerState(Enum):
    """
    Container state enumeration for standardized state management
    """
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ContainerConfig:
    """
    Configuration data class for container specifications
    
    Attributes:
        name: Unique container name
        image: Docker/Sarus image name
        ports: Port mappings (host:container)
        environment: Environment variables
        volumes: Volume mounts
        networks: Network connections
        health_check: Health check configuration
        resource_limits: CPU and memory limits
        dependencies: Container dependencies
        restart_policy: Restart behavior
    """
    name: str
    image: str
    ports: Dict[str, str] = None
    environment: Dict[str, str] = None
    volumes: Dict[str, str] = None
    networks: List[str] = None
    health_check: Dict[str, Any] = None
    resource_limits: Dict[str, str] = None
    dependencies: List[str] = None
    restart_policy: str = "unless-stopped"
    
    def __post_init__(self):
        """Initialize default values for None fields"""
        if self.ports is None:
            self.ports = {}
        if self.environment is None:
            self.environment = {}
        if self.volumes is None:
            self.volumes = {}
        if self.networks is None:
            self.networks = []
        if self.health_check is None:
            self.health_check = {}
        if self.resource_limits is None:
            self.resource_limits = {}
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ResourceMetrics:
    """
    Resource utilization metrics for containers
    
    Attributes:
        cpu_percent: CPU utilization percentage
        memory_usage: Memory usage in bytes
        memory_percent: Memory utilization percentage
        network_io: Network I/O statistics
        disk_io: Disk I/O statistics
        timestamp: Metrics collection timestamp
    """
    cpu_percent: float
    memory_usage: int
    memory_percent: float
    network_io: Dict[str, int]
    disk_io: Dict[str, int]
    timestamp: datetime


class EnvironmentDetector:
    """
    Detects the deployment environment and available containerization technologies
    
    This class automatically determines whether the system should use Docker, Sarus,
    or other containerization technologies based on the current environment.
    """
    
    def __init__(self):
        """Initialize the environment detector"""
        self.logger = logging.getLogger(f'{__name__}.EnvironmentDetector')
        self._detected_environment: Optional[DeploymentEnvironment] = None
        self._capabilities: Dict[str, bool] = {}
    
    def detect_environment(self) -> DeploymentEnvironment:
        """
        Detect the current deployment environment
        
        Returns:
            DeploymentEnvironment: The detected environment type
            
        Raises:
            RuntimeError: If no suitable containerization technology is available
        """
        self.logger.info("Starting environment detection...")
        
        # Check for Docker availability
        docker_available = self._check_docker_availability()
        
        # Check for Sarus availability (HPC environments)
        sarus_available = self._check_sarus_availability()
        
        # Check for Kubernetes environment
        k8s_available = self._check_kubernetes_environment()
        
        # Check for cloud environment indicators
        cloud_environment = self._detect_cloud_environment()
        
        # Store capabilities for later reference
        self._capabilities = {
            'docker': docker_available,
            'sarus': sarus_available,
            'kubernetes': k8s_available,
            'cloud': cloud_environment is not None
        }
        
        # Determine environment based on available technologies and context
        if k8s_available:
            self._detected_environment = DeploymentEnvironment.KUBERNETES
            self.logger.info("Detected Kubernetes environment")
        elif cloud_environment and docker_available:
            self._detected_environment = DeploymentEnvironment.CLOUD_DOCKER
            self.logger.info(f"Detected cloud Docker environment: {cloud_environment}")
        elif sarus_available and not docker_available:
            self._detected_environment = DeploymentEnvironment.HPC_SARUS
            self.logger.info("Detected HPC Sarus environment")
        elif docker_available:
            self._detected_environment = DeploymentEnvironment.LOCAL_DOCKER
            self.logger.info("Detected local Docker environment")
        else:
            self.logger.error("No suitable containerization technology detected")
            raise RuntimeError("No containerization technology available")
            
        return self._detected_environment
    
    def _check_docker_availability(self) -> bool:
        """Check if Docker is available and accessible"""
        try:
            client = docker.from_env()
            client.ping()
            self.logger.debug("Docker daemon is accessible")
            return True
        except Exception as e:
            self.logger.debug(f"Docker not available: {e}")
            return False
    
    def _check_sarus_availability(self) -> bool:
        """Check if Sarus is available (typical in HPC environments)"""
        try:
            result = subprocess.run(['sarus', '--version'], 
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.logger.debug(f"Sarus available: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            self.logger.debug(f"Sarus not available: {e}")
        return False
    
    def _check_kubernetes_environment(self) -> bool:
        """Check if running in Kubernetes environment"""
        # Check for Kubernetes service account token
        k8s_token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        k8s_env_vars = os.environ.get('KUBERNETES_SERVICE_HOST') is not None
        
        return k8s_token_path.exists() or k8s_env_vars
    
    def _detect_cloud_environment(self) -> Optional[str]:
        """Detect cloud provider environment"""
        # AWS detection
        try:
            response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
            if response.status_code == 200:
                return 'aws'
        except requests.RequestException:
            pass
        
        # GCP detection
        try:
            headers = {'Metadata-Flavor': 'Google'}
            response = requests.get('http://metadata.google.internal/computeMetadata/v1/instance/id', 
                                  headers=headers, timeout=2)
            if response.status_code == 200:
                return 'gcp'
        except requests.RequestException:
            pass
        
        # Azure detection
        try:
            headers = {'Metadata': 'true'}
            response = requests.get('http://169.254.169.254/metadata/instance/compute/vmId', 
                                  headers=headers, timeout=2)
            if response.status_code == 200:
                return 'azure'
        except requests.RequestException:
            pass
        
        return None
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get detected containerization capabilities"""
        return self._capabilities.copy()
    
    def get_detected_environment(self) -> Optional[DeploymentEnvironment]:
        """Get the currently detected environment"""
        return self._detected_environment


class DockerManager:
    """
    Docker container management implementation
    
    Handles Docker-specific operations including container lifecycle management,
    health monitoring, and resource optimization.
    """
    
    def __init__(self):
        """Initialize Docker manager"""
        self.logger = logging.getLogger(f'{__name__}.DockerManager')
        self.client: Optional[DockerClient] = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Docker client connection"""
        try:
            self.client = docker.from_env()
            self.client.ping()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(f"Docker initialization failed: {e}")
    
    async def start_container(self, config: ContainerConfig) -> bool:
        """
        Start a container with the specified configuration
        
        Args:
            config: Container configuration
            
        Returns:
            bool: True if container started successfully
        """
        try:
            self.logger.info(f"Starting container: {config.name}")
            
            # Check if container already exists
            existing_container = self._get_container(config.name)
            if existing_container:
                if existing_container.status == 'running':
                    self.logger.info(f"Container {config.name} already running")
                    return True
                else:
                    # Remove stopped container
                    existing_container.remove()
                    self.logger.info(f"Removed existing stopped container: {config.name}")
            
            # Prepare container arguments
            container_args = {
                'image': config.image,
                'name': config.name,
                'environment': config.environment,
                'ports': config.ports,
                'volumes': config.volumes,
                'network_mode': config.networks[0] if config.networks else None,
                'restart_policy': {'Name': config.restart_policy},
                'detach': True
            }
            
            # Add resource limits if specified
            if config.resource_limits:
                container_args['mem_limit'] = config.resource_limits.get('memory')
                container_args['cpu_count'] = config.resource_limits.get('cpu')
            
            # Start container
            container = self.client.containers.run(**container_args)
            
            # Wait for container to be ready
            await self._wait_for_container_ready(container, config)
            
            self.logger.info(f"Container {config.name} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start container {config.name}: {e}")
            return False
    
    async def stop_container(self, name: str, timeout: int = 30) -> bool:
        """
        Stop a running container
        
        Args:
            name: Container name
            timeout: Stop timeout in seconds
            
        Returns:
            bool: True if container stopped successfully
        """
        try:
            container = self._get_container(name)
            if not container:
                self.logger.warning(f"Container {name} not found")
                return True
            
            if container.status != 'running':
                self.logger.info(f"Container {name} already stopped")
                return True
            
            self.logger.info(f"Stopping container: {name}")
            container.stop(timeout=timeout)
            
            # Wait for container to stop
            container.wait(timeout=timeout + 10)
            
            self.logger.info(f"Container {name} stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop container {name}: {e}")
            return False
    
    async def restart_container(self, name: str) -> bool:
        """
        Restart a container
        
        Args:
            name: Container name
            
        Returns:
            bool: True if container restarted successfully
        """
        try:
            container = self._get_container(name)
            if not container:
                self.logger.error(f"Container {name} not found")
                return False
            
            self.logger.info(f"Restarting container: {name}")
            container.restart(timeout=30)
            
            self.logger.info(f"Container {name} restarted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restart container {name}: {e}")
            return False
    
    def get_container_status(self, name: str) -> ContainerState:
        """
        Get container status
        
        Args:
            name: Container name
            
        Returns:
            ContainerState: Current container state
        """
        try:
            container = self._get_container(name)
            if not container:
                return ContainerState.UNKNOWN
            
            status_map = {
                'running': ContainerState.RUNNING,
                'exited': ContainerState.STOPPED,
                'created': ContainerState.STOPPED,
                'restarting': ContainerState.STARTING,
                'removing': ContainerState.STOPPING,
                'paused': ContainerState.STOPPED,
                'dead': ContainerState.ERROR
            }
            
            return status_map.get(container.status, ContainerState.UNKNOWN)
            
        except Exception as e:
            self.logger.error(f"Failed to get status for container {name}: {e}")
            return ContainerState.ERROR
    
    def get_container_metrics(self, name: str) -> Optional[ResourceMetrics]:
        """
        Get container resource metrics
        
        Args:
            name: Container name
            
        Returns:
            ResourceMetrics: Container resource utilization metrics
        """
        try:
            container = self._get_container(name)
            if not container or container.status != 'running':
                return None
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0
            
            # Network I/O
            network_io = {
                'rx_bytes': sum(net['rx_bytes'] for net in stats['networks'].values()),
                'tx_bytes': sum(net['tx_bytes'] for net in stats['networks'].values())
            }
            
            # Disk I/O
            disk_io = {
                'read_bytes': sum(bio['value'] for bio in stats['blkio_stats']['io_service_bytes_recursive'] 
                                 if bio['op'] == 'Read'),
                'write_bytes': sum(bio['value'] for bio in stats['blkio_stats']['io_service_bytes_recursive'] 
                                  if bio['op'] == 'Write')
            }
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_usage=memory_usage,
                memory_percent=memory_percent,
                network_io=network_io,
                disk_io=disk_io,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics for container {name}: {e}")
            return None
    
    def list_containers(self) -> List[Dict[str, Any]]:
        """
        List all containers managed by this system
        
        Returns:
            List of container information dictionaries
        """
        try:
            containers = []
            for container in self.client.containers.list(all=True):
                containers.append({
                    'name': container.name,
                    'id': container.id,
                    'status': container.status,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    'created': container.attrs['Created'],
                    'ports': container.ports
                })
            return containers
        except Exception as e:
            self.logger.error(f"Failed to list containers: {e}")
            return []
    
    def _get_container(self, name: str) -> Optional[Container]:
        """Get container by name"""
        try:
            return self.client.containers.get(name)
        except NotFound:
            return None
        except Exception as e:
            self.logger.error(f"Error getting container {name}: {e}")
            return None
    
    async def _wait_for_container_ready(self, container: Container, config: ContainerConfig) -> None:
        """Wait for container to be ready using health checks"""
        if not config.health_check:
            # Wait a bit for container to start
            await asyncio.sleep(5)
            return
        
        max_retries = config.health_check.get('retries', 3)
        interval = config.health_check.get('interval', 30)
        timeout = config.health_check.get('timeout', 10)
        
        for attempt in range(max_retries):
            try:
                # Refresh container status
                container.reload()
                if container.status == 'running':
                    # Perform custom health check if specified
                    if 'test' in config.health_check:
                        health_command = config.health_check['test']
                        if isinstance(health_command, list) and health_command[0] == 'CMD':
                            result = container.exec_run(health_command[1:], timeout=timeout)
                            if result.exit_code == 0:
                                return
                    else:
                        return
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.warning(f"Health check attempt {attempt + 1} failed for {config.name}: {e}")
                await asyncio.sleep(interval)
        
        raise RuntimeError(f"Container {config.name} failed to become ready after {max_retries} attempts")


class SarusManager:
    """
    Sarus container management implementation for HPC environments
    
    Handles Sarus-specific operations for High Performance Computing environments
    where Docker is not available due to security constraints.
    """
    
    def __init__(self):
        """Initialize Sarus manager"""
        self.logger = logging.getLogger(f'{__name__}.SarusManager')
        self.running_containers: Dict[str, subprocess.Popen] = {}
        self._verify_sarus_availability()
    
    def _verify_sarus_availability(self) -> None:
        """Verify Sarus is available and functional"""
        try:
            result = subprocess.run(['sarus', '--version'], 
                                    capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("Sarus command failed")
            self.logger.info(f"Sarus manager initialized: {result.stdout.strip()}")
        except Exception as e:
            self.logger.error(f"Sarus not available: {e}")
            raise RuntimeError(f"Sarus initialization failed: {e}")
    
    async def start_container(self, config: ContainerConfig) -> bool:
        """
        Start a container using Sarus
        
        Args:
            config: Container configuration
            
        Returns:
            bool: True if container started successfully
        """
        try:
            self.logger.info(f"Starting Sarus container: {config.name}")
            
            # Build Sarus command
            sarus_cmd = self._build_sarus_command(config)
            
            # Start container process
            process = subprocess.Popen(
                sarus_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store process reference
            self.running_containers[config.name] = process
            
            # Wait a bit to check if process started successfully
            await asyncio.sleep(2)
            
            if process.poll() is None:  # Process is still running
                self.logger.info(f"Sarus container {config.name} started successfully")
                return True
            else:
                # Process terminated, get error output
                _, stderr = process.communicate()
                self.logger.error(f"Sarus container {config.name} failed to start: {stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start Sarus container {config.name}: {e}")
            return False
    
    async def stop_container(self, name: str, timeout: int = 30) -> bool:
        """
        Stop a Sarus container
        
        Args:
            name: Container name
            timeout: Stop timeout in seconds
            
        Returns:
            bool: True if container stopped successfully
        """
        try:
            if name not in self.running_containers:
                self.logger.warning(f"Sarus container {name} not found in running containers")
                return True
            
            process = self.running_containers[name]
            
            if process.poll() is not None:  # Process already terminated
                del self.running_containers[name]
                self.logger.info(f"Sarus container {name} already stopped")
                return True
            
            self.logger.info(f"Stopping Sarus container: {name}")
            
            # Send SIGTERM
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if timeout exceeded
                self.logger.warning(f"Force killing Sarus container {name}")
                process.kill()
                process.wait()
            
            del self.running_containers[name]
            self.logger.info(f"Sarus container {name} stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop Sarus container {name}: {e}")
            return False
    
    async def restart_container(self, name: str) -> bool:
        """
        Restart a Sarus container
        
        Args:
            name: Container name
            
        Returns:
            bool: True if container restarted successfully
        """
        try:
            # For Sarus, we need to stop and start again
            if name in self.running_containers:
                await self.stop_container(name)
            
            # We need the original config to restart - this is a limitation
            # In practice, the orchestrator should handle this
            self.logger.warning(f"Sarus restart for {name} requires original configuration")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to restart Sarus container {name}: {e}")
            return False
    
    def get_container_status(self, name: str) -> ContainerState:
        """
        Get Sarus container status
        
        Args:
            name: Container name
            
        Returns:
            ContainerState: Current container state
        """
        try:
            if name not in self.running_containers:
                return ContainerState.STOPPED
            
            process = self.running_containers[name]
            
            if process.poll() is None:
                return ContainerState.RUNNING
            else:
                # Process terminated, clean up
                del self.running_containers[name]
                return ContainerState.STOPPED
                
        except Exception as e:
            self.logger.error(f"Failed to get status for Sarus container {name}: {e}")
            return ContainerState.ERROR
    
    def get_container_metrics(self, name: str) -> Optional[ResourceMetrics]:
        """
        Get Sarus container resource metrics
        
        Note: Sarus doesn't provide built-in metrics like Docker.
        This implementation provides basic process-level metrics.
        
        Args:
            name: Container name
            
        Returns:
            ResourceMetrics: Container resource utilization metrics
        """
        try:
            if name not in self.running_containers:
                return None
            
            process = self.running_containers[name]
            
            if process.poll() is not None:
                return None
            
            # Get basic process metrics using psutil
            try:
                proc = psutil.Process(process.pid)
                cpu_percent = proc.cpu_percent()
                memory_info = proc.memory_info()
                
                return ResourceMetrics(
                    cpu_percent=cpu_percent,
                    memory_usage=memory_info.rss,
                    memory_percent=proc.memory_percent(),
                    network_io={'rx_bytes': 0, 'tx_bytes': 0},  # Not available for Sarus
                    disk_io={'read_bytes': 0, 'write_bytes': 0},  # Not available for Sarus
                    timestamp=datetime.now()
                )
            except psutil.NoSuchProcess:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get metrics for Sarus container {name}: {e}")
            return None
    
    def list_containers(self) -> List[Dict[str, Any]]:
        """
        List all Sarus containers
        
        Returns:
            List of container information dictionaries
        """
        containers = []
        for name, process in self.running_containers.items():
            status = 'running' if process.poll() is None else 'stopped'
            containers.append({
                'name': name,
                'id': str(process.pid),
                'status': status,
                'image': 'unknown',  # Sarus doesn't track image info easily
                'created': 'unknown',
                'ports': {}
            })
        return containers
    
    def _build_sarus_command(self, config: ContainerConfig) -> List[str]:
        """Build Sarus command from container configuration"""
        cmd = ['sarus', 'run']
        
        # Add environment variables
        for key, value in config.environment.items():
            cmd.extend(['--env', f'{key}={value}'])
        
        # Add volume mounts
        for host_path, container_path in config.volumes.items():
            cmd.extend(['--mount', f'type=bind,source={host_path},destination={container_path}'])
        
        # Add resource limits if supported
        if config.resource_limits:
            if 'memory' in config.resource_limits:
                # Sarus may support memory limits depending on configuration
                pass
        
        # Add working directory if specified
        if 'workdir' in config.environment:
            cmd.extend(['--workdir', config.environment['workdir']])
        
        # Add image
        cmd.append(config.image)
        
        # Add command if specified in environment
        if 'CONTAINER_CMD' in config.environment:
            cmd.extend(config.environment['CONTAINER_CMD'].split())
        
        return cmd 


class ResourceManager:
    """
    Resource allocation and cost optimization manager
    
    Integrates with the existing performance optimization system to manage
    container resources based on cost constraints and performance requirements.
    """
    
    def __init__(self, optimization_service_url: str = "http://optimization-service:3000"):
        """Initialize resource manager"""
        self.logger = logging.getLogger(f'{__name__}.ResourceManager')
        self.optimization_service_url = optimization_service_url
        self._cost_thresholds: Dict[str, float] = {}
        self._resource_pools: Dict[str, Dict[str, Any]] = {}
    
    def calculate_resource_allocation(self, container_configs: List[ContainerConfig]) -> Dict[str, Dict[str, str]]:
        """
        Calculate optimal resource allocation for containers based on cost optimization
        
        Args:
            container_configs: List of container configurations
            
        Returns:
            Dict mapping container names to resource allocations
        """
        try:
            self.logger.info("Calculating optimal resource allocation...")
            
            allocations = {}
            
            for config in container_configs:
                # Get performance requirements
                performance_reqs = self._get_performance_requirements(config.name)
                
                # Calculate cost-optimal allocation
                allocation = self._optimize_allocation(config, performance_reqs)
                
                allocations[config.name] = allocation
                
                self.logger.debug(f"Allocation for {config.name}: {allocation}")
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Failed to calculate resource allocation: {e}")
            return {}
    
    def monitor_resource_costs(self, container_metrics: Dict[str, ResourceMetrics]) -> Dict[str, float]:
        """
        Monitor resource costs for running containers
        
        Args:
            container_metrics: Current container metrics
            
        Returns:
            Dict mapping container names to estimated costs
        """
        try:
            costs = {}
            
            for container_name, metrics in container_metrics.items():
                # Calculate cost based on resource usage
                cpu_cost = (metrics.cpu_percent / 100.0) * self._get_cpu_hourly_cost()
                memory_cost = (metrics.memory_usage / (1024**3)) * self._get_memory_hourly_cost()  # GB
                
                total_cost = cpu_cost + memory_cost
                costs[container_name] = total_cost
                
                # Check cost thresholds
                if container_name in self._cost_thresholds:
                    if total_cost > self._cost_thresholds[container_name]:
                        self.logger.warning(f"Container {container_name} exceeding cost threshold: ${total_cost:.4f}/hour")
            
            return costs
            
        except Exception as e:
            self.logger.error(f"Failed to monitor resource costs: {e}")
            return {}
    
    def set_cost_threshold(self, container_name: str, threshold: float) -> None:
        """Set cost threshold for a container"""
        self._cost_thresholds[container_name] = threshold
        self.logger.info(f"Set cost threshold for {container_name}: ${threshold}/hour")
    
    def get_scaling_recommendations(self, container_metrics: Dict[str, ResourceMetrics]) -> Dict[str, str]:
        """
        Get scaling recommendations based on resource utilization and costs
        
        Args:
            container_metrics: Current container metrics
            
        Returns:
            Dict mapping container names to scaling recommendations
        """
        recommendations = {}
        
        for container_name, metrics in container_metrics.items():
            if metrics.cpu_percent > 80 and metrics.memory_percent > 80:
                recommendations[container_name] = "scale_up"
            elif metrics.cpu_percent < 20 and metrics.memory_percent < 20:
                recommendations[container_name] = "scale_down"
            else:
                recommendations[container_name] = "maintain"
        
        return recommendations
    
    def _get_performance_requirements(self, container_name: str) -> Dict[str, Any]:
        """Get performance requirements from optimization service"""
        try:
            response = requests.get(f"{self.optimization_service_url}/performance/{container_name}", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self.logger.debug(f"Could not get performance requirements for {container_name}: {e}")
        
        # Default requirements
        return {
            'cpu_cores': 1,
            'memory_gb': 2,
            'priority': 'medium'
        }
    
    def _optimize_allocation(self, config: ContainerConfig, performance_reqs: Dict[str, Any]) -> Dict[str, str]:
        """Optimize resource allocation based on requirements and costs"""
        # Simple optimization logic - in production this would be more sophisticated
        allocation = {
            'memory': f"{performance_reqs.get('memory_gb', 2)}g",
            'cpu': str(performance_reqs.get('cpu_cores', 1))
        }
        
        # Adjust based on container priority
        priority = performance_reqs.get('priority', 'medium')
        if priority == 'high':
            # Increase allocation for high priority containers
            memory_gb = int(allocation['memory'][:-1]) * 1.5
            allocation['memory'] = f"{memory_gb}g"
        elif priority == 'low':
            # Reduce allocation for low priority containers
            memory_gb = max(1, int(allocation['memory'][:-1]) * 0.75)
            allocation['memory'] = f"{memory_gb}g"
        
        return allocation
    
    def _get_cpu_hourly_cost(self) -> float:
        """Get CPU cost per core per hour (simplified pricing)"""
        return 0.05  # $0.05 per CPU core per hour
    
    def _get_memory_hourly_cost(self) -> float:
        """Get memory cost per GB per hour (simplified pricing)"""
        return 0.01  # $0.01 per GB per hour


class HealthMonitor:
    """
    Container health monitoring and alerting system
    
    Monitors container health, performance metrics, and provides alerting
    for issues that require attention.
    """
    
    def __init__(self, alert_webhook_url: Optional[str] = None):
        """Initialize health monitor"""
        self.logger = logging.getLogger(f'{__name__}.HealthMonitor')
        self.alert_webhook_url = alert_webhook_url
        self._health_history: Dict[str, List[Dict[str, Any]]] = {}
        self._alert_cooldowns: Dict[str, datetime] = {}
    
    async def check_container_health(self, container_manager, container_name: str) -> Dict[str, Any]:
        """
        Check health of a specific container
        
        Args:
            container_manager: Docker or Sarus manager instance
            container_name: Name of container to check
            
        Returns:
            Dict containing health status and details
        """
        try:
            # Get container status
            status = container_manager.get_container_status(container_name)
            
            # Get resource metrics
            metrics = container_manager.get_container_metrics(container_name)
            
            # Determine health status
            health_status = self._evaluate_health(status, metrics)
            
            # Record health check
            health_record = {
                'timestamp': datetime.now(),
                'status': status.value,
                'health': health_status,
                'metrics': asdict(metrics) if metrics else None
            }
            
            # Store in history
            if container_name not in self._health_history:
                self._health_history[container_name] = []
            
            self._health_history[container_name].append(health_record)
            
            # Keep only last 100 records
            if len(self._health_history[container_name]) > 100:
                self._health_history[container_name] = self._health_history[container_name][-100:]
            
            # Check for alerts
            await self._check_alerts(container_name, health_record)
            
            return health_record
            
        except Exception as e:
            self.logger.error(f"Failed to check health for container {container_name}: {e}")
            return {
                'timestamp': datetime.now(),
                'status': 'error',
                'health': 'unhealthy',
                'error': str(e)
            }
    
    async def monitor_all_containers(self, container_manager, interval: int = 30) -> None:
        """
        Continuously monitor all containers
        
        Args:
            container_manager: Docker or Sarus manager instance
            interval: Monitoring interval in seconds
        """
        self.logger.info(f"Starting continuous health monitoring (interval: {interval}s)")
        
        while True:
            try:
                # Get list of containers
                containers = container_manager.list_containers()
                
                # Check health for each container
                for container in containers:
                    await self.check_container_health(container_manager, container['name'])
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary for all monitored containers"""
        summary = {
            'total_containers': len(self._health_history),
            'healthy_containers': 0,
            'unhealthy_containers': 0,
            'unknown_containers': 0,
            'container_details': {}
        }
        
        for container_name, history in self._health_history.items():
            if history:
                latest = history[-1]
                health = latest['health']
                
                if health == 'healthy':
                    summary['healthy_containers'] += 1
                elif health == 'unhealthy':
                    summary['unhealthy_containers'] += 1
                else:
                    summary['unknown_containers'] += 1
                
                summary['container_details'][container_name] = {
                    'health': health,
                    'status': latest['status'],
                    'last_check': latest['timestamp'].isoformat()
                }
        
        return summary
    
    def _evaluate_health(self, status: ContainerState, metrics: Optional[ResourceMetrics]) -> str:
        """Evaluate container health based on status and metrics"""
        if status == ContainerState.ERROR:
            return 'unhealthy'
        elif status != ContainerState.RUNNING:
            return 'unknown'
        
        if metrics is None:
            return 'unknown'
        
        # Check resource thresholds
        if metrics.cpu_percent > 90:
            return 'unhealthy'
        if metrics.memory_percent > 95:
            return 'unhealthy'
        
        return 'healthy'
    
    async def _check_alerts(self, container_name: str, health_record: Dict[str, Any]) -> None:
        """Check if alerts should be sent for container health issues"""
        # Check cooldown period
        cooldown_key = f"{container_name}_{health_record['health']}"
        if cooldown_key in self._alert_cooldowns:
            time_since_alert = datetime.now() - self._alert_cooldowns[cooldown_key]
            if time_since_alert.total_seconds() < 300:  # 5 minute cooldown
                return
        
        # Send alert for unhealthy containers
        if health_record['health'] == 'unhealthy':
            await self._send_alert(container_name, health_record)
            self._alert_cooldowns[cooldown_key] = datetime.now()
    
    async def _send_alert(self, container_name: str, health_record: Dict[str, Any]) -> None:
        """Send health alert"""
        alert_message = {
            'container': container_name,
            'health': health_record['health'],
            'status': health_record['status'],
            'timestamp': health_record['timestamp'].isoformat(),
            'metrics': health_record.get('metrics')
        }
        
        self.logger.warning(f"Health alert for {container_name}: {health_record['health']}")
        
        # Send webhook alert if configured
        if self.alert_webhook_url:
            try:
                response = requests.post(self.alert_webhook_url, json=alert_message, timeout=10)
                if response.status_code == 200:
                    self.logger.info(f"Alert sent successfully for {container_name}")
                else:
                    self.logger.error(f"Failed to send alert for {container_name}: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Failed to send webhook alert for {container_name}: {e}")


class ConfigManager:
    """
    Configuration management for different deployment environments
    
    Manages environment-specific configurations and integrates with
    the existing environment management system.
    """
    
    def __init__(self, config_dir: str = "./config"):
        """Initialize configuration manager"""
        self.logger = logging.getLogger(f'{__name__}.ConfigManager')
        self.config_dir = Path(config_dir)
        self._environment_configs: Dict[str, Dict[str, Any]] = {}
        self._load_configurations()
    
    def get_container_configs(self, environment: DeploymentEnvironment) -> List[ContainerConfig]:
        """
        Get container configurations for specific environment
        
        Args:
            environment: Target deployment environment
            
        Returns:
            List of container configurations
        """
        try:
            env_name = environment.value
            
            if env_name not in self._environment_configs:
                self.logger.warning(f"No configuration found for environment: {env_name}")
                return []
            
            env_config = self._environment_configs[env_name]
            container_configs = []
            
            for service_name, service_config in env_config.get('services', {}).items():
                config = ContainerConfig(
                    name=service_name,
                    image=service_config['image'],
                    ports=service_config.get('ports', {}),
                    environment=service_config.get('environment', {}),
                    volumes=service_config.get('volumes', {}),
                    networks=service_config.get('networks', []),
                    health_check=service_config.get('health_check', {}),
                    resource_limits=service_config.get('resource_limits', {}),
                    dependencies=service_config.get('dependencies', []),
                    restart_policy=service_config.get('restart_policy', 'unless-stopped')
                )
                container_configs.append(config)
            
            self.logger.info(f"Loaded {len(container_configs)} container configs for {env_name}")
            return container_configs
            
        except Exception as e:
            self.logger.error(f"Failed to get container configs for {environment}: {e}")
            return []
    
    def get_environment_config(self, environment: DeploymentEnvironment) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        env_name = environment.value
        return self._environment_configs.get(env_name, {})
    
    def update_container_config(self, environment: DeploymentEnvironment, 
                              container_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update container configuration
        
        Args:
            environment: Target environment
            container_name: Container to update
            updates: Configuration updates
            
        Returns:
            bool: True if update successful
        """
        try:
            env_name = environment.value
            
            if env_name not in self._environment_configs:
                self.logger.error(f"Environment {env_name} not found")
                return False
            
            if container_name not in self._environment_configs[env_name].get('services', {}):
                self.logger.error(f"Container {container_name} not found in {env_name}")
                return False
            
            # Apply updates
            service_config = self._environment_configs[env_name]['services'][container_name]
            for key, value in updates.items():
                service_config[key] = value
            
            # Save configuration
            self._save_configuration(environment)
            
            self.logger.info(f"Updated configuration for {container_name} in {env_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update container config: {e}")
            return False
    
    def _load_configurations(self) -> None:
        """Load all environment configurations"""
        try:
            config_file = self.config_dir / "containers" / "environments.json"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self._environment_configs = json.load(f)
                self.logger.info(f"Loaded configurations for {len(self._environment_configs)} environments")
            else:
                # Create default configurations
                self._create_default_configurations()
                
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            self._create_default_configurations()
    
    def _create_default_configurations(self) -> None:
        """Create default environment configurations"""
        self.logger.info("Creating default environment configurations")
        
        # Default configuration template
        default_services = {
            'alita-manager': {
                'image': 'alita-manager:latest',
                'ports': {'3000': '3000'},
                'environment': {
                    'NODE_ENV': 'production',
                    'LOG_LEVEL': 'info'
                },
                'health_check': {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:3000/health'],
                    'interval': 30,
                    'timeout': 10,
                    'retries': 3
                },
                'resource_limits': {
                    'memory': '1g',
                    'cpu': '1'
                }
            },
            'kgot-controller': {
                'image': 'kgot-controller:latest',
                'environment': {
                    'NODE_ENV': 'production',
                    'LOG_LEVEL': 'info'
                },
                'resource_limits': {
                    'memory': '2g',
                    'cpu': '2'
                }
            },
            'neo4j': {
                'image': 'neo4j:5.15-community',
                'ports': {'7474': '7474', '7687': '7687'},
                'environment': {
                    'NEO4J_AUTH': 'neo4j/password'
                },
                'resource_limits': {
                    'memory': '2g',
                    'cpu': '2'
                }
            }
        }
        
        self._environment_configs = {
            'local_docker': {
                'services': default_services.copy()
            },
            'cloud_docker': {
                'services': default_services.copy()
            },
            'hpc_sarus': {
                'services': default_services.copy()
            }
        }
        
        # Save default configurations
        for env in DeploymentEnvironment:
            self._save_configuration(env)
    
    def _save_configuration(self, environment: DeploymentEnvironment) -> None:
        """Save environment configuration to file"""
        try:
            config_dir = self.config_dir / "containers"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            config_file = config_dir / "environments.json"
            
            with open(config_file, 'w') as f:
                json.dump(self._environment_configs, f, indent=2)
                
            self.logger.debug(f"Saved configuration for {environment.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")


class ContainerOrchestrator:
    """
    Main container orchestration system
    
    This is the central class that coordinates all containerization activities,
    integrating Docker/Sarus managers with resource management, health monitoring,
    and configuration management.
    """
    
    def __init__(self, config_dir: str = "./config", 
                 alert_webhook_url: Optional[str] = None):
        """
        Initialize the container orchestrator
        
        Args:
            config_dir: Configuration directory path
            alert_webhook_url: Optional webhook URL for health alerts
        """
        self.logger = logging.getLogger(f'{__name__}.ContainerOrchestrator')
        
        # Initialize components
        self.env_detector = EnvironmentDetector()
        self.config_manager = ConfigManager(config_dir)
        self.resource_manager = ResourceManager()
        self.health_monitor = HealthMonitor(alert_webhook_url)
        
        # Container managers (initialized based on environment)
        self.docker_manager: Optional[DockerManager] = None
        self.sarus_manager: Optional[SarusManager] = None
        self.current_manager = None
        
        # State tracking
        self.current_environment: Optional[DeploymentEnvironment] = None
        self.managed_containers: Dict[str, ContainerConfig] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info("Container orchestrator initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the orchestrator for the current environment
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing container orchestrator...")
            
            # Detect environment
            self.current_environment = self.env_detector.detect_environment()
            capabilities = self.env_detector.get_capabilities()
            
            self.logger.info(f"Detected environment: {self.current_environment.value}")
            self.logger.info(f"Available capabilities: {capabilities}")
            
            # Initialize appropriate container manager
            if self.current_environment in [DeploymentEnvironment.LOCAL_DOCKER, 
                                          DeploymentEnvironment.CLOUD_DOCKER]:
                if capabilities['docker']:
                    self.docker_manager = DockerManager()
                    self.current_manager = self.docker_manager
                    self.logger.info("Docker manager initialized")
                else:
                    raise RuntimeError("Docker environment detected but Docker not available")
                    
            elif self.current_environment == DeploymentEnvironment.HPC_SARUS:
                if capabilities['sarus']:
                    self.sarus_manager = SarusManager()
                    self.current_manager = self.sarus_manager
                    self.logger.info("Sarus manager initialized")
                else:
                    raise RuntimeError("HPC environment detected but Sarus not available")
                    
            else:
                raise RuntimeError(f"Unsupported environment: {self.current_environment}")
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            self.logger.info("Container orchestrator initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            return False
    
    async def deploy_service_stack(self) -> bool:
        """
        Deploy the complete KGoT-Alita service stack
        
        Returns:
            bool: True if deployment successful
        """
        try:
            self.logger.info("Starting service stack deployment...")
            
            if not self.current_manager:
                raise RuntimeError("Container manager not initialized")
            
            # Get container configurations for current environment
            container_configs = self.config_manager.get_container_configs(self.current_environment)
            
            if not container_configs:
                raise RuntimeError("No container configurations found")
            
            # Calculate optimal resource allocation
            resource_allocations = self.resource_manager.calculate_resource_allocation(container_configs)
            
            # Apply resource allocations
            for config in container_configs:
                if config.name in resource_allocations:
                    config.resource_limits.update(resource_allocations[config.name])
            
            # Sort containers by dependencies
            sorted_configs = self._sort_by_dependencies(container_configs)
            
            # Deploy containers in dependency order
            for config in sorted_configs:
                self.logger.info(f"Deploying container: {config.name}")
                
                success = await self.current_manager.start_container(config)
                if not success:
                    self.logger.error(f"Failed to deploy container: {config.name}")
                    return False
                
                # Store managed container
                self.managed_containers[config.name] = config
                
                # Wait for dependencies to be ready
                await self._wait_for_dependencies(config)
            
            self.logger.info("Service stack deployment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy service stack: {e}")
            return False
    
    async def stop_service_stack(self) -> bool:
        """
        Stop the complete service stack
        
        Returns:
            bool: True if stop successful
        """
        try:
            self.logger.info("Stopping service stack...")
            
            if not self.current_manager:
                self.logger.warning("No container manager available")
                return True
            
            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Stop containers in reverse dependency order
            sorted_configs = list(reversed(self._sort_by_dependencies(list(self.managed_containers.values()))))
            
            for config in sorted_configs:
                self.logger.info(f"Stopping container: {config.name}")
                await self.current_manager.stop_container(config.name)
            
            # Clear managed containers
            self.managed_containers.clear()
            
            self.logger.info("Service stack stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop service stack: {e}")
            return False
    
    async def restart_service(self, service_name: str) -> bool:
        """
        Restart a specific service
        
        Args:
            service_name: Name of service to restart
            
        Returns:
            bool: True if restart successful
        """
        try:
            self.logger.info(f"Restarting service: {service_name}")
            
            if not self.current_manager:
                raise RuntimeError("Container manager not initialized")
            
            if service_name not in self.managed_containers:
                raise RuntimeError(f"Service {service_name} not managed by orchestrator")
            
            success = await self.current_manager.restart_container(service_name)
            
            if success:
                self.logger.info(f"Service {service_name} restarted successfully")
            else:
                self.logger.error(f"Failed to restart service: {service_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to restart service {service_name}: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of all managed services
        
        Returns:
            Dict containing service status information
        """
        try:
            if not self.current_manager:
                return {'error': 'Container manager not initialized'}
            
            service_status = {}
            
            for service_name in self.managed_containers.keys():
                status = self.current_manager.get_container_status(service_name)
                metrics = self.current_manager.get_container_metrics(service_name)
                
                service_status[service_name] = {
                    'status': status.value,
                    'metrics': asdict(metrics) if metrics else None
                }
            
            # Add health summary
            health_summary = self.health_monitor.get_health_summary()
            
            return {
                'environment': self.current_environment.value if self.current_environment else 'unknown',
                'services': service_status,
                'health_summary': health_summary,
                'resource_costs': self._get_current_costs()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get service status: {e}")
            return {'error': str(e)}
    
    async def scale_service(self, service_name: str, action: str) -> bool:
        """
        Scale a service up or down
        
        Args:
            service_name: Name of service to scale
            action: 'scale_up' or 'scale_down'
            
        Returns:
            bool: True if scaling successful
        """
        try:
            self.logger.info(f"Scaling service {service_name}: {action}")
            
            if service_name not in self.managed_containers:
                raise RuntimeError(f"Service {service_name} not managed")
            
            config = self.managed_containers[service_name]
            
            # Adjust resource limits based on scaling action
            if action == 'scale_up':
                # Increase resources by 50%
                current_memory = int(config.resource_limits.get('memory', '1g')[:-1])
                new_memory = int(current_memory * 1.5)
                config.resource_limits['memory'] = f"{new_memory}g"
                
                current_cpu = int(config.resource_limits.get('cpu', '1'))
                new_cpu = min(current_cpu * 2, 8)  # Cap at 8 CPUs
                config.resource_limits['cpu'] = str(new_cpu)
                
            elif action == 'scale_down':
                # Decrease resources by 25%
                current_memory = int(config.resource_limits.get('memory', '2g')[:-1])
                new_memory = max(1, int(current_memory * 0.75))  # Minimum 1GB
                config.resource_limits['memory'] = f"{new_memory}g"
                
                current_cpu = int(config.resource_limits.get('cpu', '2'))
                new_cpu = max(1, int(current_cpu * 0.75))  # Minimum 1 CPU
                config.resource_limits['cpu'] = str(new_cpu)
            
            # Restart container with new resources
            await self.current_manager.stop_container(service_name)
            success = await self.current_manager.start_container(config)
            
            if success:
                self.logger.info(f"Service {service_name} scaled successfully")
            else:
                self.logger.error(f"Failed to scale service: {service_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to scale service {service_name}: {e}")
            return False
    
    def _sort_by_dependencies(self, configs: List[ContainerConfig]) -> List[ContainerConfig]:
        """Sort container configurations by dependency order"""
        sorted_configs = []
        remaining_configs = configs.copy()
        
        while remaining_configs:
            # Find containers with no unresolved dependencies
            ready_configs = []
            for config in remaining_configs:
                deps_resolved = all(
                    dep in [c.name for c in sorted_configs] 
                    for dep in config.dependencies
                )
                if deps_resolved:
                    ready_configs.append(config)
            
            if not ready_configs:
                # Circular dependency or missing dependency
                self.logger.warning("Circular or missing dependencies detected, using original order")
                sorted_configs.extend(remaining_configs)
                break
            
            # Add ready configs and remove from remaining
            sorted_configs.extend(ready_configs)
            for config in ready_configs:
                remaining_configs.remove(config)
        
        return sorted_configs
    
    async def _wait_for_dependencies(self, config: ContainerConfig) -> None:
        """Wait for container dependencies to be ready"""
        if not config.dependencies:
            return
        
        self.logger.info(f"Waiting for dependencies of {config.name}: {config.dependencies}")
        
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            all_ready = True
            
            for dep_name in config.dependencies:
                status = self.current_manager.get_container_status(dep_name)
                if status != ContainerState.RUNNING:
                    all_ready = False
                    break
            
            if all_ready:
                self.logger.info(f"All dependencies ready for {config.name}")
                return
            
            await asyncio.sleep(5)
        
        self.logger.warning(f"Timeout waiting for dependencies of {config.name}")
    
    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring"""
        self.monitoring_task = asyncio.create_task(
            self.health_monitor.monitor_all_containers(self.current_manager, interval=30)
        )
        self.logger.info("Health monitoring started")
    
    def _get_current_costs(self) -> Dict[str, float]:
        """Get current resource costs for all managed containers"""
        if not self.current_manager:
            return {}
        
        container_metrics = {}
        for service_name in self.managed_containers.keys():
            metrics = self.current_manager.get_container_metrics(service_name)
            if metrics:
                container_metrics[service_name] = metrics
        
        return self.resource_manager.monitor_resource_costs(container_metrics)


# CLI Interface
async def main():
    """Main CLI interface for containerization system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='KGoT Containerization CLI')
    parser.add_argument('command', choices=['deploy', 'stop', 'status', 'restart', 'scale'])
    parser.add_argument('--service', help='Service name for service-specific operations')
    parser.add_argument('--action', choices=['scale_up', 'scale_down'], help='Scaling action')
    parser.add_argument('--config-dir', default='/app/config', help='Configuration directory')
    parser.add_argument('--alert-webhook', help='Alert webhook URL')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ContainerOrchestrator(args.config_dir, args.alert_webhook)
    
    if not await orchestrator.initialize():
        print("Failed to initialize orchestrator")
        sys.exit(1)
    
    try:
        if args.command == 'deploy':
            success = await orchestrator.deploy_service_stack()
            print(f"Deployment {'successful' if success else 'failed'}")
            
        elif args.command == 'stop':
            success = await orchestrator.stop_service_stack()
            print(f"Stop {'successful' if success else 'failed'}")
            
        elif args.command == 'status':
            status = orchestrator.get_service_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif args.command == 'restart':
            if not args.service:
                print("Service name required for restart")
                sys.exit(1)
            success = await orchestrator.restart_service(args.service)
            print(f"Restart {'successful' if success else 'failed'}")
            
        elif args.command == 'scale':
            if not args.service or not args.action:
                print("Service name and action required for scaling")
                sys.exit(1)
            success = await orchestrator.scale_service(args.service, args.action)
            print(f"Scaling {'successful' if success else 'failed'}")
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        await orchestrator.stop_service_stack()


if __name__ == '__main__':
    asyncio.run(main()) 