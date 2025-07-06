#!/usr/bin/env python3
"""
Task 51: Advanced Configuration Management System

A unified configuration-as-code system for the Alita-KGoT Enhanced project that provides:
- Centralized configuration management
- Dynamic configuration updates
- Configuration validation and impact assessment
- Version control integration
- Rollback capabilities

Author: Alita-KGoT Enhanced System
Date: 2024
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import toml
except ImportError:
    toml = None

# Configure logging
logger = logging.getLogger(__name__)


class ConfigurationFormat(Enum):
    """Supported configuration file formats"""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    ENV = "env"


class ConfigurationScope(Enum):
    """Configuration scope levels"""
    GLOBAL = "global"
    ENVIRONMENT = "environment"
    SERVICE = "service"
    FEATURE = "feature"


class ValidationSeverity(Enum):
    """Validation result severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    severity: ValidationSeverity
    message: str
    path: str
    suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConfigurationChange:
    """Represents a configuration change"""
    path: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ConfigurationVersion:
    """Configuration version metadata"""
    version: str
    timestamp: datetime
    changes: List[ConfigurationChange]
    commit_hash: Optional[str] = None
    rollback_data: Optional[Dict[str, Any]] = None


class ConfigurationValidator(ABC):
    """Abstract base class for configuration validators"""
    
    @abstractmethod
    async def validate(self, config: Dict[str, Any], changes: List[ConfigurationChange]) -> List[ValidationResult]:
        """Validate configuration changes"""
        pass


class KGoTParameterValidator(ConfigurationValidator):
    """Validator for KGoT-specific parameters"""
    
    KGOT_PARAMETERS = {
        'num_next_steps_decision': {'type': int, 'min': 1, 'max': 10, 'default': 3},
        'max_retrieve_query_retry': {'type': int, 'min': 1, 'max': 5, 'default': 3},
        'max_cypher_fixing_retry': {'type': int, 'min': 1, 'max': 5, 'default': 2},
        'reasoning_depth': {'type': int, 'min': 1, 'max': 20, 'default': 5},
        'graph_traversal_limit': {'type': int, 'min': 10, 'max': 1000, 'default': 100}
    }
    
    async def validate(self, config: Dict[str, Any], changes: List[ConfigurationChange]) -> List[ValidationResult]:
        """Validate KGoT parameters"""
        results = []
        
        kgot_config = config.get('kgot', {})
        
        for param, constraints in self.KGOT_PARAMETERS.items():
            if param in kgot_config:
                value = kgot_config[param]
                
                # Type validation
                if not isinstance(value, constraints['type']):
                    results.append(ValidationResult(
                        severity=ValidationSeverity.ERROR,
                        message=f"Parameter '{param}' must be of type {constraints['type'].__name__}",
                        path=f"kgot.{param}",
                        suggestion=f"Convert value to {constraints['type'].__name__}"
                    ))
                    continue
                
                # Range validation
                if 'min' in constraints and value < constraints['min']:
                    results.append(ValidationResult(
                        severity=ValidationSeverity.ERROR,
                        message=f"Parameter '{param}' value {value} is below minimum {constraints['min']}",
                        path=f"kgot.{param}",
                        suggestion=f"Set value to at least {constraints['min']}"
                    ))
                
                if 'max' in constraints and value > constraints['max']:
                    results.append(ValidationResult(
                        severity=ValidationSeverity.WARNING,
                        message=f"Parameter '{param}' value {value} exceeds recommended maximum {constraints['max']}",
                        path=f"kgot.{param}",
                        suggestion=f"Consider reducing value to {constraints['max']} or below"
                    ))
        
        return results


class AlitaConfigurationValidator(ConfigurationValidator):
    """Validator for Alita-specific configurations"""
    
    async def validate(self, config: Dict[str, Any], changes: List[ConfigurationChange]) -> List[ValidationResult]:
        """Validate Alita configurations"""
        results = []
        
        alita_config = config.get('alita', {})
        
        # Validate environment manager settings
        env_manager = alita_config.get('environment_manager', {})
        if 'supported_languages' in env_manager:
            supported = env_manager['supported_languages']
            if not isinstance(supported, list) or not supported:
                results.append(ValidationResult(
                    severity=ValidationSeverity.ERROR,
                    message="Alita environment manager must support at least one language",
                    path="alita.environment_manager.supported_languages",
                    suggestion="Add at least one supported language (e.g., 'python', 'javascript')"
                ))
        
        # Validate tool configurations
        tools = alita_config.get('tools', {})
        for tool_name, tool_config in tools.items():
            if 'timeout' in tool_config:
                timeout = tool_config['timeout']
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    results.append(ValidationResult(
                        severity=ValidationSeverity.ERROR,
                        message=f"Tool '{tool_name}' timeout must be a positive number",
                        path=f"alita.tools.{tool_name}.timeout",
                        suggestion="Set timeout to a positive number in seconds"
                    ))
        
        return results


class ModelConfigurationValidator(ConfigurationValidator):
    """Validator for model configurations"""
    
    async def validate(self, config: Dict[str, Any], changes: List[ConfigurationChange]) -> List[ValidationResult]:
        """Validate model configurations"""
        results = []
        
        models = config.get('models', {})
        
        for provider_name, provider_config in models.items():
            if 'models' in provider_config:
                for model_name, model_config in provider_config['models'].items():
                    # Validate cost configuration
                    if 'cost_per_token' in model_config:
                        cost_config = model_config['cost_per_token']
                        for cost_type in ['input', 'output']:
                            if cost_type in cost_config:
                                cost = cost_config[cost_type]
                                if not isinstance(cost, (int, float)) or cost < 0:
                                    results.append(ValidationResult(
                                        severity=ValidationSeverity.ERROR,
                                        message=f"Model '{model_name}' {cost_type} cost must be non-negative",
                                        path=f"models.{provider_name}.models.{model_name}.cost_per_token.{cost_type}",
                                        suggestion="Set cost to a non-negative number"
                                    ))
                    
                    # Validate context length
                    if 'context_length' in model_config:
                        context_length = model_config['context_length']
                        if not isinstance(context_length, int) or context_length <= 0:
                            results.append(ValidationResult(
                                severity=ValidationSeverity.ERROR,
                                message=f"Model '{model_name}' context_length must be a positive integer",
                                path=f"models.{provider_name}.models.{model_name}.context_length",
                                suggestion="Set context_length to a positive integer"
                            ))
        
        return results


class ConfigurationStore:
    """Centralized configuration storage and retrieval"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.cache = {}
        self.cache_timestamps = {}
        self.lock = threading.RLock()
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get file content hash for cache invalidation"""
        if not file_path.exists():
            return ""
        
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from file"""
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(content) or {}
            elif file_path.suffix.lower() == '.json':
                return json.loads(content)
            elif file_path.suffix.lower() == '.toml' and toml:
                return toml.loads(content)
            elif file_path.suffix.lower() == '.env':
                return self._parse_env_file(content)
            else:
                logger.warning(f"Unsupported configuration file format: {file_path}")
                return {}
        
        except Exception as e:
            logger.error(f"Failed to load configuration file {file_path}: {e}")
            return {}
    
    def _parse_env_file(self, content: str) -> Dict[str, Any]:
        """Parse .env file content"""
        config = {}
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip().strip('"\'')
        return config
    
    def get_configuration(self, scope: ConfigurationScope, name: str, force_reload: bool = False) -> Dict[str, Any]:
        """Get configuration for a specific scope and name"""
        cache_key = f"{scope.value}:{name}"
        
        with self.lock:
            file_path = self.base_path / scope.value / f"{name}.yaml"
            
            # Check if we need to reload from disk
            current_hash = self._get_file_hash(file_path)
            cached_hash = self.cache_timestamps.get(cache_key, "")
            
            if force_reload or cache_key not in self.cache or current_hash != cached_hash:
                self.cache[cache_key] = self._load_file(file_path)
                self.cache_timestamps[cache_key] = current_hash
            
            return self.cache[cache_key].copy()
    
    def set_configuration(self, scope: ConfigurationScope, name: str, config: Dict[str, Any], format_type: ConfigurationFormat = ConfigurationFormat.YAML) -> bool:
        """Set configuration for a specific scope and name"""
        try:
            file_path = self.base_path / scope.value
            file_path.mkdir(parents=True, exist_ok=True)
            
            if format_type == ConfigurationFormat.YAML:
                file_path = file_path / f"{name}.yaml"
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif format_type == ConfigurationFormat.JSON:
                file_path = file_path / f"{name}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, sort_keys=False)
            elif format_type == ConfigurationFormat.TOML and toml:
                file_path = file_path / f"{name}.toml"
                with open(file_path, 'w', encoding='utf-8') as f:
                    toml.dump(config, f)
            else:
                logger.error(f"Unsupported format: {format_type}")
                return False
            
            # Invalidate cache
            cache_key = f"{scope.value}:{name}"
            with self.lock:
                if cache_key in self.cache:
                    del self.cache[cache_key]
                if cache_key in self.cache_timestamps:
                    del self.cache_timestamps[cache_key]
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def list_configurations(self, scope: ConfigurationScope) -> List[str]:
        """List all configurations in a scope"""
        scope_path = self.base_path / scope.value
        if not scope_path.exists():
            return []
        
        configs = []
        for file_path in scope_path.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.yaml', '.yml', '.json', '.toml', '.env']:
                configs.append(file_path.stem)
        
        return sorted(configs)


class ConfigurationWatcher(FileSystemEventHandler):
    """Watches for configuration file changes and triggers reloads"""
    
    def __init__(self, config_manager: 'ConfigurationManager'):
        self.config_manager = config_manager
        self.observer = Observer()
        self.watched_paths = set()
    
    def start_watching(self, path: Path):
        """Start watching a path for changes"""
        if path not in self.watched_paths:
            self.observer.schedule(self, str(path), recursive=True)
            self.watched_paths.add(path)
            
            if not self.observer.is_alive():
                self.observer.start()
    
    def stop_watching(self):
        """Stop watching for changes"""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json', '.toml', '.env')):
            logger.info(f"Configuration file changed: {event.src_path}")
            asyncio.create_task(self.config_manager.reload_configuration(Path(event.src_path)))


class ConfigurationVersionManager:
    """Manages configuration versions and rollbacks"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / ".versions"
        self.versions_path.mkdir(parents=True, exist_ok=True)
        self.current_version = None
    
    def create_version(self, changes: List[ConfigurationChange], commit_hash: Optional[str] = None) -> str:
        """Create a new configuration version"""
        version_id = f"v{int(time.time())}"
        timestamp = datetime.now(timezone.utc)
        
        # Create rollback data
        rollback_data = {}
        for change in changes:
            rollback_data[change.path] = change.old_value
        
        version = ConfigurationVersion(
            version=version_id,
            timestamp=timestamp,
            changes=changes,
            commit_hash=commit_hash,
            rollback_data=rollback_data
        )
        
        # Save version metadata
        version_file = self.versions_path / f"{version_id}.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump({
                'version': version.version,
                'timestamp': version.timestamp.isoformat(),
                'commit_hash': version.commit_hash,
                'changes': [{
                    'path': change.path,
                    'old_value': change.old_value,
                    'new_value': change.new_value,
                    'timestamp': change.timestamp.isoformat(),
                    'user': change.user,
                    'reason': change.reason
                } for change in version.changes],
                'rollback_data': version.rollback_data
            }, f, indent=2)
        
        self.current_version = version_id
        return version_id
    
    def get_version(self, version_id: str) -> Optional[ConfigurationVersion]:
        """Get version metadata"""
        version_file = self.versions_path / f"{version_id}.json"
        if not version_file.exists():
            return None
        
        try:
            with open(version_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            changes = []
            for change_data in data['changes']:
                changes.append(ConfigurationChange(
                    path=change_data['path'],
                    old_value=change_data['old_value'],
                    new_value=change_data['new_value'],
                    timestamp=datetime.fromisoformat(change_data['timestamp']),
                    user=change_data.get('user'),
                    reason=change_data.get('reason')
                ))
            
            return ConfigurationVersion(
                version=data['version'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                changes=changes,
                commit_hash=data.get('commit_hash'),
                rollback_data=data.get('rollback_data')
            )
        
        except Exception as e:
            logger.error(f"Failed to load version {version_id}: {e}")
            return None
    
    def list_versions(self) -> List[str]:
        """List all available versions"""
        versions = []
        for version_file in self.versions_path.glob("v*.json"):
            versions.append(version_file.stem)
        return sorted(versions, reverse=True)
    
    def rollback_to_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get rollback data for a specific version"""
        version = self.get_version(version_id)
        if version and version.rollback_data:
            return version.rollback_data
        return None


class ConfigurationManager:
    """Main configuration management orchestrator"""
    
    def __init__(self, base_path: Path, enable_watching: bool = True):
        self.base_path = Path(base_path)
        self.store = ConfigurationStore(self.base_path)
        self.version_manager = ConfigurationVersionManager(self.base_path)
        self.validators = []
        self.change_listeners = []
        self.enable_watching = enable_watching
        
        if enable_watching:
            self.watcher = ConfigurationWatcher(self)
            self.watcher.start_watching(self.base_path)
        
        # Register default validators
        self.register_validator(KGoTParameterValidator())
        self.register_validator(AlitaConfigurationValidator())
        self.register_validator(ModelConfigurationValidator())
        
        # Initialize default configurations
        self._initialize_default_configurations()
    
    def _initialize_default_configurations(self):
        """Initialize default configuration files"""
        # KGoT default configuration
        kgot_config = {
            'kgot': {
                'num_next_steps_decision': 3,
                'max_retrieve_query_retry': 3,
                'max_cypher_fixing_retry': 2,
                'reasoning_depth': 5,
                'graph_traversal_limit': 100,
                'enable_caching': True,
                'cache_ttl': 3600,
                'parallel_processing': True,
                'max_concurrent_queries': 10
            }
        }
        
        # Alita default configuration
        alita_config = {
            'alita': {
                'environment_manager': {
                    'supported_languages': ['python', 'javascript', 'typescript', 'bash'],
                    'timeout': 300,
                    'max_memory': '1GB',
                    'enable_sandboxing': True
                },
                'tools': {
                    'code_runner': {
                        'timeout': 60,
                        'max_output_size': '10MB',
                        'enable_networking': False
                    },
                    'file_manager': {
                        'max_file_size': '100MB',
                        'allowed_extensions': ['.py', '.js', '.ts', '.md', '.txt', '.json', '.yaml'],
                        'scan_for_secrets': True
                    }
                }
            }
        }
        
        # Feature flags configuration
        feature_flags = {
            'features': {
                'advanced_reasoning': True,
                'multimodal_processing': True,
                'real_time_collaboration': False,
                'experimental_models': False,
                'enhanced_security': True,
                'performance_monitoring': True
            }
        }
        
        # Save default configurations if they don't exist
        configs_to_create = [
            (ConfigurationScope.GLOBAL, 'kgot', kgot_config),
            (ConfigurationScope.GLOBAL, 'alita', alita_config),
            (ConfigurationScope.GLOBAL, 'features', feature_flags)
        ]
        
        for scope, name, config in configs_to_create:
            if name not in self.store.list_configurations(scope):
                self.store.set_configuration(scope, name, config)
    
    def register_validator(self, validator: ConfigurationValidator):
        """Register a configuration validator"""
        self.validators.append(validator)
    
    def register_change_listener(self, listener: Callable[[List[ConfigurationChange]], None]):
        """Register a change listener"""
        self.change_listeners.append(listener)
    
    async def get_configuration(self, scope: ConfigurationScope, name: str, force_reload: bool = False) -> Dict[str, Any]:
        """Get configuration with optional force reload"""
        return self.store.get_configuration(scope, name, force_reload)
    
    async def update_configuration(self, scope: ConfigurationScope, name: str, updates: Dict[str, Any], user: Optional[str] = None, reason: Optional[str] = None) -> bool:
        """Update configuration with validation and versioning"""
        try:
            # Get current configuration
            current_config = await self.get_configuration(scope, name)
            
            # Apply updates
            new_config = current_config.copy()
            self._deep_update(new_config, updates)
            
            # Track changes
            changes = self._detect_changes(current_config, new_config, f"{scope.value}.{name}")
            for change in changes:
                change.user = user
                change.reason = reason
            
            # Validate changes
            validation_results = await self._validate_configuration(new_config, changes)
            
            # Check for critical errors
            critical_errors = [r for r in validation_results if r.severity == ValidationSeverity.CRITICAL]
            if critical_errors:
                logger.error(f"Critical validation errors prevent configuration update: {critical_errors}")
                return False
            
            # Log warnings
            warnings = [r for r in validation_results if r.severity == ValidationSeverity.WARNING]
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning.message} at {warning.path}")
            
            # Save configuration
            if self.store.set_configuration(scope, name, new_config):
                # Create version
                version_id = self.version_manager.create_version(changes)
                logger.info(f"Configuration updated successfully. Version: {version_id}")
                
                # Notify listeners
                for listener in self.change_listeners:
                    try:
                        listener(changes)
                    except Exception as e:
                        logger.error(f"Error notifying change listener: {e}")
                
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep update dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any], base_path: str = "") -> List[ConfigurationChange]:
        """Detect changes between configurations"""
        changes = []
        
        # Check for modified and new keys
        for key, new_value in new_config.items():
            current_path = f"{base_path}.{key}" if base_path else key
            
            if key not in old_config:
                changes.append(ConfigurationChange(
                    path=current_path,
                    old_value=None,
                    new_value=new_value
                ))
            elif old_config[key] != new_value:
                if isinstance(old_config[key], dict) and isinstance(new_value, dict):
                    changes.extend(self._detect_changes(old_config[key], new_value, current_path))
                else:
                    changes.append(ConfigurationChange(
                        path=current_path,
                        old_value=old_config[key],
                        new_value=new_value
                    ))
        
        # Check for removed keys
        for key, old_value in old_config.items():
            if key not in new_config:
                current_path = f"{base_path}.{key}" if base_path else key
                changes.append(ConfigurationChange(
                    path=current_path,
                    old_value=old_value,
                    new_value=None
                ))
        
        return changes
    
    async def _validate_configuration(self, config: Dict[str, Any], changes: List[ConfigurationChange]) -> List[ValidationResult]:
        """Validate configuration using all registered validators"""
        all_results = []
        
        for validator in self.validators:
            try:
                results = await validator.validate(config, changes)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Validator {validator.__class__.__name__} failed: {e}")
                all_results.append(ValidationResult(
                    severity=ValidationSeverity.ERROR,
                    message=f"Validator {validator.__class__.__name__} failed: {e}",
                    path="validation"
                ))
        
        return all_results
    
    async def rollback_to_version(self, version_id: str) -> bool:
        """Rollback configuration to a specific version"""
        try:
            rollback_data = self.version_manager.rollback_to_version(version_id)
            if not rollback_data:
                logger.error(f"No rollback data found for version {version_id}")
                return False
            
            # Apply rollback changes
            success = True
            for path, old_value in rollback_data.items():
                path_parts = path.split('.')
                if len(path_parts) >= 2:
                    scope_name = path_parts[0]
                    config_name = path_parts[1]
                    
                    try:
                        scope = ConfigurationScope(scope_name)
                        current_config = await self.get_configuration(scope, config_name)
                        
                        # Apply rollback value
                        config_path = '.'.join(path_parts[2:]) if len(path_parts) > 2 else None
                        if config_path:
                            self._set_nested_value(current_config, config_path, old_value)
                        else:
                            current_config = old_value if old_value is not None else {}
                        
                        if not self.store.set_configuration(scope, config_name, current_config):
                            success = False
                            logger.error(f"Failed to rollback configuration {scope_name}.{config_name}")
                    
                    except Exception as e:
                        success = False
                        logger.error(f"Failed to rollback path {path}: {e}")
            
            if success:
                logger.info(f"Successfully rolled back to version {version_id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set nested value in configuration"""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        if value is None:
            current.pop(keys[-1], None)
        else:
            current[keys[-1]] = value
    
    async def reload_configuration(self, file_path: Path):
        """Reload configuration from file change"""
        try:
            # Determine scope and name from file path
            relative_path = file_path.relative_to(self.base_path)
            scope_name = relative_path.parts[0]
            config_name = relative_path.stem
            
            scope = ConfigurationScope(scope_name)
            
            # Force reload from disk
            await self.get_configuration(scope, config_name, force_reload=True)
            
            logger.info(f"Reloaded configuration: {scope_name}.{config_name}")
        
        except Exception as e:
            logger.error(f"Failed to reload configuration from {file_path}: {e}")
    
    def get_merged_configuration(self, environment: str = "production") -> Dict[str, Any]:
        """Get merged configuration for all scopes and environment"""
        merged = {}
        
        # Load global configurations
        for config_name in self.store.list_configurations(ConfigurationScope.GLOBAL):
            config = self.store.get_configuration(ConfigurationScope.GLOBAL, config_name)
            merged.update(config)
        
        # Load environment-specific configurations
        env_configs = self.store.list_configurations(ConfigurationScope.ENVIRONMENT)
        if environment in env_configs:
            env_config = self.store.get_configuration(ConfigurationScope.ENVIRONMENT, environment)
            self._deep_update(merged, env_config)
        
        return merged
    
    def export_configuration(self, output_path: Path, format_type: ConfigurationFormat = ConfigurationFormat.YAML) -> bool:
        """Export all configurations to a single file"""
        try:
            merged_config = self.get_merged_configuration()
            
            if format_type == ConfigurationFormat.YAML:
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
            elif format_type == ConfigurationFormat.JSON:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_config, f, indent=2, sort_keys=False)
            elif format_type == ConfigurationFormat.TOML and toml:
                with open(output_path, 'w', encoding='utf-8') as f:
                    toml.dump(merged_config, f)
            else:
                logger.error(f"Unsupported export format: {format_type}")
                return False
            
            logger.info(f"Configuration exported to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_configuration(self, input_path: Path, scope: ConfigurationScope, name: str) -> bool:
        """Import configuration from file"""
        try:
            if not input_path.exists():
                logger.error(f"Import file does not exist: {input_path}")
                return False
            
            config = self.store._load_file(input_path)
            return self.store.set_configuration(scope, name, config)
        
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get overall configuration system status"""
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'current_version': self.version_manager.current_version,
            'total_versions': len(self.version_manager.list_versions()),
            'watching_enabled': self.enable_watching,
            'validators_count': len(self.validators),
            'listeners_count': len(self.change_listeners),
            'scopes': {}
        }
        
        for scope in ConfigurationScope:
            configs = self.store.list_configurations(scope)
            status['scopes'][scope.value] = {
                'count': len(configs),
                'configurations': configs
            }
        
        return status
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'watcher') and self.enable_watching:
            self.watcher.stop_watching()


# CLI Interface for configuration management
if __name__ == "__main__":
    import argparse
    import sys
    
    def setup_logging(level: str = "INFO"):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def main():
        parser = argparse.ArgumentParser(description="Advanced Configuration Management CLI")
        parser.add_argument("--config-path", default="./config", help="Configuration base path")
        parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Status command
        subparsers.add_parser("status", help="Show configuration status")
        
        # Get command
        get_parser = subparsers.add_parser("get", help="Get configuration")
        get_parser.add_argument("scope", choices=[s.value for s in ConfigurationScope])
        get_parser.add_argument("name")
        
        # Update command
        update_parser = subparsers.add_parser("update", help="Update configuration")
        update_parser.add_argument("scope", choices=[s.value for s in ConfigurationScope])
        update_parser.add_argument("name")
        update_parser.add_argument("--file", help="JSON/YAML file with updates")
        update_parser.add_argument("--key")
        update_parser.add_argument("--value")
        update_parser.add_argument("--user")
        update_parser.add_argument("--reason")
        
        # Rollback command
        rollback_parser = subparsers.add_parser("rollback", help="Rollback to version")
        rollback_parser.add_argument("version")
        
        # Export command
        export_parser = subparsers.add_parser("export", help="Export configuration")
        export_parser.add_argument("output_file")
        export_parser.add_argument("--format", choices=[f.value for f in ConfigurationFormat], default="yaml")
        
        # Import command
        import_parser = subparsers.add_parser("import", help="Import configuration")
        import_parser.add_argument("input_file")
        import_parser.add_argument("scope", choices=[s.value for s in ConfigurationScope])
        import_parser.add_argument("name")
        
        # Versions command
        subparsers.add_parser("versions", help="List configuration versions")
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        setup_logging(args.log_level)
        
        config_manager = ConfigurationManager(Path(args.config_path))
        
        try:
            if args.command == "status":
                status = config_manager.get_configuration_status()
                print(json.dumps(status, indent=2))
            
            elif args.command == "get":
                scope = ConfigurationScope(args.scope)
                config = await config_manager.get_configuration(scope, args.name)
                print(json.dumps(config, indent=2))
            
            elif args.command == "update":
                scope = ConfigurationScope(args.scope)
                
                if args.file:
                    with open(args.file, 'r') as f:
                        if args.file.endswith('.yaml') or args.file.endswith('.yml'):
                            updates = yaml.safe_load(f)
                        else:
                            updates = json.load(f)
                elif args.key and args.value:
                    updates = {args.key: args.value}
                else:
                    print("Either --file or --key/--value must be provided")
                    return
                
                success = await config_manager.update_configuration(
                    scope, args.name, updates, args.user, args.reason
                )
                print(f"Update {'successful' if success else 'failed'}")
            
            elif args.command == "rollback":
                success = await config_manager.rollback_to_version(args.version)
                print(f"Rollback {'successful' if success else 'failed'}")
            
            elif args.command == "export":
                format_type = ConfigurationFormat(args.format)
                success = config_manager.export_configuration(Path(args.output_file), format_type)
                print(f"Export {'successful' if success else 'failed'}")
            
            elif args.command == "import":
                scope = ConfigurationScope(args.scope)
                success = config_manager.import_configuration(Path(args.input_file), scope, args.name)
                print(f"Import {'successful' if success else 'failed'}")
            
            elif args.command == "versions":
                versions = config_manager.version_manager.list_versions()
                for version in versions:
                    version_data = config_manager.version_manager.get_version(version)
                    if version_data:
                        print(f"{version}: {version_data.timestamp} ({len(version_data.changes)} changes)")
        
        finally:
            config_manager.cleanup()
    
    if __name__ == "__main__":
        asyncio.run(main())