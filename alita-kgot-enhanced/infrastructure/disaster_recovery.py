#!/usr/bin/env python3
"""
KGoT-Alita Disaster Recovery and Backup System

Implements comprehensive backup strategies and disaster recovery procedures for:
- KGoT Graph Store (Neo4j) with automated point-in-time backups
- RAG-MCP Vector Index with regular backup schedules
- Cross-system data synchronization and consistency
- Automated backup validation and testing
- Disaster recovery orchestration with RTO/RPO monitoring

Based on KGoT Section 3.1 Graph Store Module and RAG-MCP Section 1.2 contributions.

@module DisasterRecovery
@author Enhanced Alita KGoT System
@version 1.0.0
"""

import os
import sys
import json
import logging
import asyncio
import aiofiles
import subprocess
import shutil
import hashlib
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import tarfile
import gzip
import boto3
from botocore.exceptions import ClientError

# Neo4j backup support
try:
    import neo4j
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available")

# Vector database support
try:
    import faiss
    import numpy as np
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    logging.warning("Vector database libraries not available")

# Setup logging
project_root = Path(__file__).parent.parent
log_dir = project_root / 'logs' / 'disaster_recovery'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(operation)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'disaster_recovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DisasterRecovery')

class BackupType(Enum):
    """Types of backups supported"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATING = "validating"
    VALIDATED = "validated"
    VALIDATION_FAILED = "validation_failed"

class RecoveryStatus(Enum):
    """Recovery operation status"""
    NOT_STARTED = "not_started"
    PREPARING = "preparing"
    RESTORING_GRAPH = "restoring_graph"
    RESTORING_VECTOR = "restoring_vector"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BackupMetadata:
    """Metadata for backup operations"""
    backup_id: str
    backup_type: BackupType
    component: str  # 'neo4j', 'vector_index', 'config'
    timestamp: datetime
    file_path: str
    file_size: int
    checksum: str
    status: BackupStatus
    validation_result: Optional[Dict[str, Any]] = None
    retention_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DisasterRecoveryConfig:
    """Configuration for disaster recovery system"""
    # Backup settings
    backup_frequency_minutes: int = 15  # High frequency as specified
    backup_retention_days: int = 30
    backup_storage_path: str = "/var/backups/kgot"
    remote_backup_enabled: bool = True
    remote_backup_bucket: Optional[str] = None
    
    # Neo4j settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # Vector index settings
    vector_index_path: str = "/var/lib/kgot/vector_index"
    
    # Recovery settings
    recovery_test_frequency_days: int = 90  # Quarterly testing
    rto_target_minutes: int = 60  # Recovery Time Objective
    rpo_target_minutes: int = 15  # Recovery Point Objective
    
    # Validation settings
    validation_enabled: bool = True
    validation_timeout_minutes: int = 30
    
    # Monitoring settings
    monitoring_enabled: bool = True
    alert_webhook_url: Optional[str] = None

class Neo4jBackupService:
    """Service for Neo4j graph database backups"""
    
    def __init__(self, config: DisasterRecoveryConfig):
        self.config = config
        self.driver = None
        
    async def initialize(self):
        """Initialize Neo4j connection"""
        if not NEO4J_AVAILABLE:
            raise RuntimeError("Neo4j driver not available")
            
        try:
            self.driver = neo4j.AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            
            # Test connection
            async with self.driver.session() as session:
                await session.run("RETURN 1")
                
            logger.info("Neo4j backup service initialized", extra={'operation': 'NEO4J_BACKUP_INIT'})
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j backup service: {e}", 
                        extra={'operation': 'NEO4J_BACKUP_INIT_FAILED'})
            raise
    
    async def create_backup(self, backup_type: BackupType = BackupType.FULL) -> BackupMetadata:
        """Create Neo4j backup using native backup tools"""
        backup_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        try:
            # Create backup directory
            backup_dir = Path(self.config.backup_storage_path) / "neo4j" / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Use Neo4j Admin backup command
            backup_file = backup_dir / f"neo4j_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}.dump"
            
            # Execute neo4j-admin backup
            cmd = [
                "neo4j-admin", "database", "dump",
                "--database", self.config.neo4j_database,
                "--to-path", str(backup_file)
            ]
            
            logger.info(f"Starting Neo4j backup: {backup_id}", 
                       extra={'operation': 'NEO4J_BACKUP_START'})
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Neo4j backup failed: {stderr.decode()}")
            
            # Calculate file size and checksum
            file_size = backup_file.stat().st_size
            checksum = await self._calculate_checksum(backup_file)
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                component="neo4j",
                timestamp=timestamp,
                file_path=str(backup_file),
                file_size=file_size,
                checksum=checksum,
                status=BackupStatus.COMPLETED,
                retention_until=timestamp + timedelta(days=self.config.backup_retention_days)
            )
            
            logger.info(f"Neo4j backup completed: {backup_id}", 
                       extra={'operation': 'NEO4J_BACKUP_COMPLETED', 'size': file_size})
            
            return metadata
            
        except Exception as e:
            logger.error(f"Neo4j backup failed: {e}", 
                        extra={'operation': 'NEO4J_BACKUP_FAILED'})
            raise
    
    async def restore_backup(self, backup_metadata: BackupMetadata, target_database: str = None) -> bool:
        """Restore Neo4j backup"""
        try:
            target_db = target_database or self.config.neo4j_database
            backup_file = Path(backup_metadata.file_path)
            
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            # Verify checksum
            current_checksum = await self._calculate_checksum(backup_file)
            if current_checksum != backup_metadata.checksum:
                raise ValueError("Backup file checksum mismatch")
            
            logger.info(f"Starting Neo4j restore: {backup_metadata.backup_id}", 
                       extra={'operation': 'NEO4J_RESTORE_START'})
            
            # Stop database if running
            await self._stop_database(target_db)
            
            # Execute neo4j-admin restore
            cmd = [
                "neo4j-admin", "database", "load",
                "--from-path", str(backup_file),
                "--database", target_db,
                "--overwrite-destination"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Neo4j restore failed: {stderr.decode()}")
            
            # Start database
            await self._start_database(target_db)
            
            logger.info(f"Neo4j restore completed: {backup_metadata.backup_id}", 
                       extra={'operation': 'NEO4J_RESTORE_COMPLETED'})
            
            return True
            
        except Exception as e:
            logger.error(f"Neo4j restore failed: {e}", 
                        extra={'operation': 'NEO4J_RESTORE_FAILED'})
            return False
    
    async def validate_backup(self, backup_metadata: BackupMetadata) -> Dict[str, Any]:
        """Validate Neo4j backup by attempting restore to temporary database"""
        temp_db_name = f"temp_validation_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"Validating Neo4j backup: {backup_metadata.backup_id}", 
                       extra={'operation': 'NEO4J_BACKUP_VALIDATION_START'})
            
            # Restore to temporary database
            success = await self.restore_backup(backup_metadata, temp_db_name)
            
            if not success:
                return {'valid': False, 'error': 'Restore failed'}
            
            # Run health checks
            health_results = await self._run_health_checks(temp_db_name)
            
            # Cleanup temporary database
            await self._cleanup_database(temp_db_name)
            
            validation_result = {
                'valid': health_results['success'],
                'node_count': health_results.get('node_count', 0),
                'relationship_count': health_results.get('relationship_count', 0),
                'health_checks': health_results,
                'validated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Neo4j backup validation completed: {backup_metadata.backup_id}", 
                       extra={'operation': 'NEO4J_BACKUP_VALIDATION_COMPLETED', 
                             'valid': validation_result['valid']})
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Neo4j backup validation failed: {e}", 
                        extra={'operation': 'NEO4J_BACKUP_VALIDATION_FAILED'})
            return {'valid': False, 'error': str(e)}
        finally:
            # Ensure cleanup
            try:
                await self._cleanup_database(temp_db_name)
            except:
                pass
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            async for chunk in f:
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _stop_database(self, database: str):
        """Stop Neo4j database"""
        # Implementation depends on Neo4j deployment method
        pass
    
    async def _start_database(self, database: str):
        """Start Neo4j database"""
        # Implementation depends on Neo4j deployment method
        pass
    
    async def _cleanup_database(self, database: str):
        """Cleanup temporary database"""
        try:
            cmd = ["neo4j-admin", "database", "delete", "--database", database]
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.communicate()
        except Exception as e:
            logger.warning(f"Failed to cleanup database {database}: {e}")
    
    async def _run_health_checks(self, database: str) -> Dict[str, Any]:
        """Run health checks on database"""
        try:
            # Connect to specific database
            driver = neo4j.AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            
            async with driver.session(database=database) as session:
                # Count nodes
                result = await session.run("MATCH (n) RETURN count(n) as node_count")
                record = await result.single()
                node_count = record["node_count"]
                
                # Count relationships
                result = await session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                record = await result.single()
                rel_count = record["rel_count"]
                
                # Test basic query
                result = await session.run("MATCH (n) RETURN n LIMIT 1")
                sample_node = await result.single()
                
            await driver.close()
            
            return {
                'success': True,
                'node_count': node_count,
                'relationship_count': rel_count,
                'sample_query_success': sample_node is not None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()

class VectorIndexBackupService:
    """Service for RAG-MCP Vector Index backups"""
    
    def __init__(self, config: DisasterRecoveryConfig):
        self.config = config
    
    async def create_backup(self, backup_type: BackupType = BackupType.FULL) -> BackupMetadata:
        """Create vector index backup"""
        backup_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        try:
            # Create backup directory
            backup_dir = Path(self.config.backup_storage_path) / "vector_index" / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create compressed archive of vector index
            backup_file = backup_dir / f"vector_index_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}.tar.gz"
            
            logger.info(f"Starting vector index backup: {backup_id}", 
                       extra={'operation': 'VECTOR_BACKUP_START'})
            
            # Create tar.gz archive
            with tarfile.open(backup_file, 'w:gz') as tar:
                vector_path = Path(self.config.vector_index_path)
                if vector_path.exists():
                    tar.add(vector_path, arcname='vector_index')
            
            # Calculate file size and checksum
            file_size = backup_file.stat().st_size
            checksum = await self._calculate_checksum(backup_file)
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                component="vector_index",
                timestamp=timestamp,
                file_path=str(backup_file),
                file_size=file_size,
                checksum=checksum,
                status=BackupStatus.COMPLETED,
                retention_until=timestamp + timedelta(days=self.config.backup_retention_days)
            )
            
            logger.info(f"Vector index backup completed: {backup_id}", 
                       extra={'operation': 'VECTOR_BACKUP_COMPLETED', 'size': file_size})
            
            return metadata
            
        except Exception as e:
            logger.error(f"Vector index backup failed: {e}", 
                        extra={'operation': 'VECTOR_BACKUP_FAILED'})
            raise
    
    async def restore_backup(self, backup_metadata: BackupMetadata, target_path: str = None) -> bool:
        """Restore vector index backup"""
        try:
            target_dir = Path(target_path or self.config.vector_index_path)
            backup_file = Path(backup_metadata.file_path)
            
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            # Verify checksum
            current_checksum = await self._calculate_checksum(backup_file)
            if current_checksum != backup_metadata.checksum:
                raise ValueError("Backup file checksum mismatch")
            
            logger.info(f"Starting vector index restore: {backup_metadata.backup_id}", 
                       extra={'operation': 'VECTOR_RESTORE_START'})
            
            # Remove existing index
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            # Extract backup
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(target_dir.parent)
                
                # Rename extracted directory
                extracted_dir = target_dir.parent / 'vector_index'
                if extracted_dir.exists() and extracted_dir != target_dir:
                    extracted_dir.rename(target_dir)
            
            logger.info(f"Vector index restore completed: {backup_metadata.backup_id}", 
                       extra={'operation': 'VECTOR_RESTORE_COMPLETED'})
            
            return True
            
        except Exception as e:
            logger.error(f"Vector index restore failed: {e}", 
                        extra={'operation': 'VECTOR_RESTORE_FAILED'})
            return False
    
    async def validate_backup(self, backup_metadata: BackupMetadata) -> Dict[str, Any]:
        """Validate vector index backup"""
        temp_dir = None
        
        try:
            logger.info(f"Validating vector index backup: {backup_metadata.backup_id}", 
                       extra={'operation': 'VECTOR_BACKUP_VALIDATION_START'})
            
            # Create temporary directory
            temp_dir = Path(tempfile.mkdtemp(prefix='vector_validation_'))
            
            # Restore to temporary location
            success = await self.restore_backup(backup_metadata, str(temp_dir))
            
            if not success:
                return {'valid': False, 'error': 'Restore failed'}
            
            # Run validation checks
            validation_result = await self._run_vector_health_checks(temp_dir)
            
            logger.info(f"Vector index backup validation completed: {backup_metadata.backup_id}", 
                       extra={'operation': 'VECTOR_BACKUP_VALIDATION_COMPLETED', 
                             'valid': validation_result['valid']})
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Vector index backup validation failed: {e}", 
                        extra={'operation': 'VECTOR_BACKUP_VALIDATION_FAILED'})
            return {'valid': False, 'error': str(e)}
        finally:
            # Cleanup temporary directory
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            async for chunk in f:
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _run_vector_health_checks(self, vector_dir: Path) -> Dict[str, Any]:
        """Run health checks on vector index"""
        try:
            if not VECTOR_DB_AVAILABLE:
                return {'valid': False, 'error': 'Vector database libraries not available'}
            
            # Check if index files exist
            index_files = list(vector_dir.glob('*.index'))
            metadata_files = list(vector_dir.glob('*.json'))
            
            if not index_files:
                return {'valid': False, 'error': 'No index files found'}
            
            # Try to load FAISS index
            try:
                index_file = index_files[0]
                index = faiss.read_index(str(index_file))
                vector_count = index.ntotal
                dimension = index.d
                
                # Test search functionality
                if vector_count > 0:
                    test_vector = np.random.random((1, dimension)).astype('float32')
                    distances, indices = index.search(test_vector, min(5, vector_count))
                    search_success = len(indices[0]) > 0
                else:
                    search_success = True  # Empty index is valid
                
                return {
                    'valid': True,
                    'vector_count': vector_count,
                    'dimension': dimension,
                    'index_files': len(index_files),
                    'metadata_files': len(metadata_files),
                    'search_test_success': search_success,
                    'validated_at': datetime.now().isoformat()
                }
                
            except Exception as e:
                return {'valid': False, 'error': f'Failed to load index: {str(e)}'}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}

class BackupManager:
    """Central backup management coordinator"""
    
    def __init__(self, config: DisasterRecoveryConfig):
        self.config = config
        self.neo4j_service = Neo4jBackupService(config)
        self.vector_service = VectorIndexBackupService(config)
        self.backup_metadata: List[BackupMetadata] = []
        self.is_running = False
        
        # Initialize backup storage
        Path(config.backup_storage_path).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize backup services"""
        await self.neo4j_service.initialize()
        logger.info("Backup manager initialized", extra={'operation': 'BACKUP_MANAGER_INIT'})
    
    async def start_scheduled_backups(self):
        """Start scheduled backup operations"""
        self.is_running = True
        
        logger.info(f"Starting scheduled backups every {self.config.backup_frequency_minutes} minutes", 
                   extra={'operation': 'SCHEDULED_BACKUPS_START'})
        
        while self.is_running:
            try:
                await self.create_coordinated_backup()
                await asyncio.sleep(self.config.backup_frequency_minutes * 60)
            except Exception as e:
                logger.error(f"Scheduled backup failed: {e}", 
                           extra={'operation': 'SCHEDULED_BACKUP_FAILED'})
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def create_coordinated_backup(self) -> List[BackupMetadata]:
        """Create coordinated backup across all systems"""
        backup_session_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        logger.info(f"Starting coordinated backup session: {backup_session_id}", 
                   extra={'operation': 'COORDINATED_BACKUP_START'})
        
        backups = []
        
        try:
            # Create backups in parallel for consistency
            tasks = [
                self.neo4j_service.create_backup(),
                self.vector_service.create_backup()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Backup component failed: {result}", 
                               extra={'operation': 'COORDINATED_BACKUP_COMPONENT_FAILED'})
                else:
                    backups.append(result)
                    self.backup_metadata.append(result)
            
            # Validate backups if enabled
            if self.config.validation_enabled and backups:
                await self._validate_backups(backups)
            
            # Upload to remote storage if enabled
            if self.config.remote_backup_enabled and backups:
                await self._upload_to_remote_storage(backups)
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
            logger.info(f"Coordinated backup session completed: {backup_session_id}", 
                       extra={'operation': 'COORDINATED_BACKUP_COMPLETED', 
                             'backup_count': len(backups)})
            
            return backups
            
        except Exception as e:
            logger.error(f"Coordinated backup session failed: {e}", 
                        extra={'operation': 'COORDINATED_BACKUP_FAILED'})
            raise
    
    async def _validate_backups(self, backups: List[BackupMetadata]):
        """Validate all backups in parallel"""
        logger.info("Starting backup validation", extra={'operation': 'BACKUP_VALIDATION_START'})
        
        validation_tasks = []
        
        for backup in backups:
            if backup.component == "neo4j":
                task = self.neo4j_service.validate_backup(backup)
            elif backup.component == "vector_index":
                task = self.vector_service.validate_backup(backup)
            else:
                continue
            
            validation_tasks.append((backup, task))
        
        # Run validations with timeout
        timeout = self.config.validation_timeout_minutes * 60
        
        for backup, task in validation_tasks:
            try:
                validation_result = await asyncio.wait_for(task, timeout=timeout)
                backup.validation_result = validation_result
                backup.status = BackupStatus.VALIDATED if validation_result['valid'] else BackupStatus.VALIDATION_FAILED
                
                logger.info(f"Backup validation completed: {backup.backup_id}", 
                           extra={'operation': 'BACKUP_VALIDATION_COMPLETED', 
                                 'valid': validation_result['valid']})
                
            except asyncio.TimeoutError:
                backup.status = BackupStatus.VALIDATION_FAILED
                backup.validation_result = {'valid': False, 'error': 'Validation timeout'}
                logger.error(f"Backup validation timeout: {backup.backup_id}", 
                           extra={'operation': 'BACKUP_VALIDATION_TIMEOUT'})
            except Exception as e:
                backup.status = BackupStatus.VALIDATION_FAILED
                backup.validation_result = {'valid': False, 'error': str(e)}
                logger.error(f"Backup validation failed: {backup.backup_id}: {e}", 
                           extra={'operation': 'BACKUP_VALIDATION_FAILED'})
    
    async def _upload_to_remote_storage(self, backups: List[BackupMetadata]):
        """Upload backups to remote storage (S3)"""
        if not self.config.remote_backup_bucket:
            logger.warning("Remote backup enabled but no bucket configured")
            return
        
        try:
            s3_client = boto3.client('s3')
            
            for backup in backups:
                backup_file = Path(backup.file_path)
                s3_key = f"kgot-backups/{backup.component}/{backup.timestamp.strftime('%Y/%m/%d')}/{backup_file.name}"
                
                logger.info(f"Uploading backup to S3: {backup.backup_id}", 
                           extra={'operation': 'S3_UPLOAD_START'})
                
                s3_client.upload_file(str(backup_file), self.config.remote_backup_bucket, s3_key)
                
                # Update metadata with S3 location
                backup.metadata['s3_bucket'] = self.config.remote_backup_bucket
                backup.metadata['s3_key'] = s3_key
                
                logger.info(f"Backup uploaded to S3: {backup.backup_id}", 
                           extra={'operation': 'S3_UPLOAD_COMPLETED'})
                
        except Exception as e:
            logger.error(f"Failed to upload backups to S3: {e}", 
                        extra={'operation': 'S3_UPLOAD_FAILED'})
    
    async def _cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
        
        expired_backups = [
            backup for backup in self.backup_metadata 
            if backup.retention_until and backup.retention_until < datetime.now()
        ]
        
        for backup in expired_backups:
            try:
                backup_file = Path(backup.file_path)
                if backup_file.exists():
                    backup_file.unlink()
                
                self.backup_metadata.remove(backup)
                
                logger.info(f"Removed expired backup: {backup.backup_id}", 
                           extra={'operation': 'BACKUP_CLEANUP'})
                
            except Exception as e:
                logger.error(f"Failed to cleanup backup {backup.backup_id}: {e}", 
                           extra={'operation': 'BACKUP_CLEANUP_FAILED'})
    
    def stop_scheduled_backups(self):
        """Stop scheduled backup operations"""
        self.is_running = False
        logger.info("Scheduled backups stopped", extra={'operation': 'SCHEDULED_BACKUPS_STOP'})
    
    async def close(self):
        """Close backup services"""
        self.stop_scheduled_backups()
        await self.neo4j_service.close()

class DisasterRecoveryOrchestrator:
    """Orchestrates complete disaster recovery procedures"""
    
    def __init__(self, config: DisasterRecoveryConfig):
        self.config = config
        self.backup_manager = BackupManager(config)
        self.recovery_status = RecoveryStatus.NOT_STARTED
        
    async def initialize(self):
        """Initialize disaster recovery system"""
        await self.backup_manager.initialize()
        logger.info("Disaster recovery orchestrator initialized", 
                   extra={'operation': 'DR_ORCHESTRATOR_INIT'})
    
    async def execute_disaster_recovery(self, target_timestamp: datetime = None) -> bool:
        """Execute complete disaster recovery procedure"""
        recovery_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting disaster recovery: {recovery_id}", 
                   extra={'operation': 'DISASTER_RECOVERY_START'})
        
        try:
            self.recovery_status = RecoveryStatus.PREPARING
            
            # Find appropriate backups
            neo4j_backup, vector_backup = await self._find_consistent_backups(target_timestamp)
            
            if not neo4j_backup or not vector_backup:
                raise RuntimeError("Could not find consistent backup set")
            
            logger.info(f"Found consistent backup set: Neo4j={neo4j_backup.backup_id}, Vector={vector_backup.backup_id}", 
                       extra={'operation': 'BACKUP_SET_FOUND'})
            
            # Restore Neo4j
            self.recovery_status = RecoveryStatus.RESTORING_GRAPH
            neo4j_success = await self.backup_manager.neo4j_service.restore_backup(neo4j_backup)
            
            if not neo4j_success:
                raise RuntimeError("Neo4j restore failed")
            
            # Restore Vector Index
            self.recovery_status = RecoveryStatus.RESTORING_VECTOR
            vector_success = await self.backup_manager.vector_service.restore_backup(vector_backup)
            
            if not vector_success:
                raise RuntimeError("Vector index restore failed")
            
            # Validate recovery
            self.recovery_status = RecoveryStatus.VALIDATING
            validation_success = await self._validate_recovery(neo4j_backup, vector_backup)
            
            if not validation_success:
                raise RuntimeError("Recovery validation failed")
            
            # Calculate RTO
            recovery_time_minutes = (time.time() - start_time) / 60
            
            self.recovery_status = RecoveryStatus.COMPLETED
            
            logger.info(f"Disaster recovery completed: {recovery_id}", 
                       extra={'operation': 'DISASTER_RECOVERY_COMPLETED', 
                             'recovery_time_minutes': recovery_time_minutes})
            
            # Check RTO compliance
            if recovery_time_minutes > self.config.rto_target_minutes:
                logger.warning(f"RTO target exceeded: {recovery_time_minutes:.2f} > {self.config.rto_target_minutes}", 
                             extra={'operation': 'RTO_TARGET_EXCEEDED'})
            
            return True
            
        except Exception as e:
            self.recovery_status = RecoveryStatus.FAILED
            logger.error(f"Disaster recovery failed: {e}", 
                        extra={'operation': 'DISASTER_RECOVERY_FAILED'})
            return False
    
    async def _find_consistent_backups(self, target_timestamp: datetime = None) -> Tuple[Optional[BackupMetadata], Optional[BackupMetadata]]:
        """Find consistent backup set based on timestamps"""
        if target_timestamp is None:
            target_timestamp = datetime.now()
        
        # Find backups closest to target timestamp
        neo4j_backups = [b for b in self.backup_manager.backup_metadata if b.component == "neo4j" and b.status == BackupStatus.VALIDATED]
        vector_backups = [b for b in self.backup_manager.backup_metadata if b.component == "vector_index" and b.status == BackupStatus.VALIDATED]
        
        if not neo4j_backups or not vector_backups:
            return None, None
        
        # Sort by timestamp proximity
        neo4j_backups.sort(key=lambda b: abs((b.timestamp - target_timestamp).total_seconds()))
        vector_backups.sort(key=lambda b: abs((b.timestamp - target_timestamp).total_seconds()))
        
        # Find best matching pair within RPO window
        rpo_window = timedelta(minutes=self.config.rpo_target_minutes)
        
        for neo4j_backup in neo4j_backups:
            for vector_backup in vector_backups:
                time_diff = abs(neo4j_backup.timestamp - vector_backup.timestamp)
                if time_diff <= rpo_window:
                    return neo4j_backup, vector_backup
        
        # If no consistent pair found within RPO, return closest
        logger.warning("No backup pair found within RPO window, using closest available", 
                      extra={'operation': 'RPO_TARGET_EXCEEDED'})
        
        return neo4j_backups[0], vector_backups[0]
    
    async def _validate_recovery(self, neo4j_backup: BackupMetadata, vector_backup: BackupMetadata) -> bool:
        """Validate successful recovery"""
        try:
            # Validate Neo4j
            neo4j_validation = await self.backup_manager.neo4j_service._run_health_checks(self.config.neo4j_database)
            
            # Validate Vector Index
            vector_validation = await self.backup_manager.vector_service._run_vector_health_checks(Path(self.config.vector_index_path))
            
            success = neo4j_validation['success'] and vector_validation['valid']
            
            logger.info(f"Recovery validation result: {success}", 
                       extra={'operation': 'RECOVERY_VALIDATION', 
                             'neo4j_valid': neo4j_validation['success'],
                             'vector_valid': vector_validation['valid']})
            
            return success
            
        except Exception as e:
            logger.error(f"Recovery validation failed: {e}", 
                        extra={'operation': 'RECOVERY_VALIDATION_FAILED'})
            return False
    
    async def run_disaster_recovery_test(self) -> Dict[str, Any]:
        """Run disaster recovery test (game day exercise)"""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting disaster recovery test: {test_id}", 
                   extra={'operation': 'DR_TEST_START'})
        
        try:
            # Create test environment
            test_config = DisasterRecoveryConfig(
                neo4j_database=f"test_dr_{uuid.uuid4().hex[:8]}",
                vector_index_path=f"/tmp/test_vector_index_{uuid.uuid4().hex[:8]}"
            )
            
            test_orchestrator = DisasterRecoveryOrchestrator(test_config)
            await test_orchestrator.initialize()
            
            # Execute recovery
            success = await test_orchestrator.execute_disaster_recovery()
            
            # Calculate metrics
            test_time_minutes = (time.time() - start_time) / 60
            
            result = {
                'test_id': test_id,
                'success': success,
                'test_time_minutes': test_time_minutes,
                'rto_compliance': test_time_minutes <= self.config.rto_target_minutes,
                'tested_at': datetime.now().isoformat()
            }
            
            logger.info(f"Disaster recovery test completed: {test_id}", 
                       extra={'operation': 'DR_TEST_COMPLETED', 'success': success})
            
            return result
            
        except Exception as e:
            logger.error(f"Disaster recovery test failed: {e}", 
                        extra={'operation': 'DR_TEST_FAILED'})
            return {
                'test_id': test_id,
                'success': False,
                'error': str(e),
                'tested_at': datetime.now().isoformat()
            }
    
    async def close(self):
        """Close disaster recovery system"""
        await self.backup_manager.close()

# Main disaster recovery system
class DisasterRecoverySystem:
    """Main disaster recovery system interface"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config_data = json.load(f)
            self.config = DisasterRecoveryConfig(**config_data)
        else:
            self.config = DisasterRecoveryConfig()
        
        self.orchestrator = DisasterRecoveryOrchestrator(self.config)
        self.is_running = False
    
    async def start(self):
        """Start disaster recovery system"""
        await self.orchestrator.initialize()
        
        # Start scheduled backups
        asyncio.create_task(self.orchestrator.backup_manager.start_scheduled_backups())
        
        # Schedule periodic DR tests
        if self.config.recovery_test_frequency_days > 0:
            asyncio.create_task(self._schedule_dr_tests())
        
        self.is_running = True
        logger.info("Disaster recovery system started", extra={'operation': 'DR_SYSTEM_START'})
    
    async def _schedule_dr_tests(self):
        """Schedule periodic disaster recovery tests"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.recovery_test_frequency_days * 24 * 60 * 60)
                if self.is_running:
                    await self.orchestrator.run_disaster_recovery_test()
            except Exception as e:
                logger.error(f"Scheduled DR test failed: {e}", 
                           extra={'operation': 'SCHEDULED_DR_TEST_FAILED'})
    
    async def stop(self):
        """Stop disaster recovery system"""
        self.is_running = False
        await self.orchestrator.close()
        logger.info("Disaster recovery system stopped", extra={'operation': 'DR_SYSTEM_STOP'})
    
    async def create_backup(self) -> List[BackupMetadata]:
        """Create manual backup"""
        return await self.orchestrator.backup_manager.create_coordinated_backup()
    
    async def restore_from_backup(self, target_timestamp: datetime = None) -> bool:
        """Restore from backup"""
        return await self.orchestrator.execute_disaster_recovery(target_timestamp)
    
    async def test_disaster_recovery(self) -> Dict[str, Any]:
        """Run disaster recovery test"""
        return await self.orchestrator.run_disaster_recovery_test()
    
    def get_backup_status(self) -> List[Dict[str, Any]]:
        """Get status of all backups"""
        return [
            {
                'backup_id': backup.backup_id,
                'component': backup.component,
                'timestamp': backup.timestamp.isoformat(),
                'status': backup.status.value,
                'file_size': backup.file_size,
                'validated': backup.validation_result is not None and backup.validation_result.get('valid', False)
            }
            for backup in self.orchestrator.backup_manager.backup_metadata
        ]

if __name__ == "__main__":
    async def main():
        # Example usage
        dr_system = DisasterRecoverySystem()
        
        try:
            await dr_system.start()
            
            # Create a backup
            backups = await dr_system.create_backup()
            print(f"Created {len(backups)} backups")
            
            # Test disaster recovery
            test_result = await dr_system.test_disaster_recovery()
            print(f"DR test result: {test_result}")
            
            # Keep running
            await asyncio.sleep(3600)  # Run for 1 hour
            
        finally:
            await dr_system.stop()
    
    asyncio.run(main())