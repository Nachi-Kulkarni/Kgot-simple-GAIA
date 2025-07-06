#!/usr/bin/env python3
"""
KGoT-Alita Disaster Recovery CLI Tool

Command-line interface for managing disaster recovery operations including:
- Manual backup creation
- Disaster recovery execution
- Backup validation
- System status monitoring
- Recovery testing

Usage:
    python dr_cli.py backup create
    python dr_cli.py backup list
    python dr_cli.py backup validate <backup_id>
    python dr_cli.py recovery execute [--timestamp YYYY-MM-DD-HH-MM-SS]
    python dr_cli.py recovery test
    python dr_cli.py status
    python dr_cli.py monitor

@module DisasterRecoveryCLI
@author Enhanced Alita KGoT System
@version 1.0.0
"""

import asyncio
import argparse
import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import disaster recovery system
from disaster_recovery import DisasterRecoverySystem, DisasterRecoveryConfig

# Setup logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DR_CLI')

class DisasterRecoveryCLI:
    """Command-line interface for disaster recovery operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "infrastructure/disaster_recovery_config.json"
        self.dr_system = None
    
    async def initialize(self):
        """Initialize disaster recovery system"""
        try:
            self.dr_system = DisasterRecoverySystem(self.config_path)
            await self.dr_system.start()
            logger.info("Disaster recovery system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize disaster recovery system: {e}")
            sys.exit(1)
    
    async def cleanup(self):
        """Cleanup disaster recovery system"""
        if self.dr_system:
            await self.dr_system.stop()
    
    async def create_backup(self):
        """Create manual backup"""
        try:
            logger.info("Creating manual backup...")
            backups = await self.dr_system.create_backup()
            
            print(f"\n✅ Successfully created {len(backups)} backups:")
            for backup in backups:
                print(f"  - {backup.component}: {backup.backup_id}")
                print(f"    Timestamp: {backup.timestamp}")
                print(f"    Size: {backup.file_size:,} bytes")
                print(f"    Status: {backup.status.value}")
                print()
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            print(f"❌ Backup creation failed: {e}")
            sys.exit(1)
    
    async def list_backups(self):
        """List all available backups"""
        try:
            backups = self.dr_system.get_backup_status()
            
            if not backups:
                print("No backups found.")
                return
            
            print(f"\n📋 Found {len(backups)} backups:\n")
            
            # Group by component
            neo4j_backups = [b for b in backups if b['component'] == 'neo4j']
            vector_backups = [b for b in backups if b['component'] == 'vector_index']
            
            if neo4j_backups:
                print("🗄️  Neo4j Graph Store Backups:")
                for backup in sorted(neo4j_backups, key=lambda x: x['timestamp'], reverse=True):
                    status_icon = "✅" if backup['validated'] else "⚠️"
                    print(f"  {status_icon} {backup['backup_id'][:8]}... - {backup['timestamp']} ({backup['file_size']:,} bytes)")
                print()
            
            if vector_backups:
                print("🔍 Vector Index Backups:")
                for backup in sorted(vector_backups, key=lambda x: x['timestamp'], reverse=True):
                    status_icon = "✅" if backup['validated'] else "⚠️"
                    print(f"  {status_icon} {backup['backup_id'][:8]}... - {backup['timestamp']} ({backup['file_size']:,} bytes)")
                print()
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            print(f"❌ Failed to list backups: {e}")
            sys.exit(1)
    
    async def validate_backup(self, backup_id: str):
        """Validate specific backup"""
        try:
            # Find backup metadata
            backups = self.dr_system.orchestrator.backup_manager.backup_metadata
            backup = next((b for b in backups if b.backup_id.startswith(backup_id)), None)
            
            if not backup:
                print(f"❌ Backup not found: {backup_id}")
                sys.exit(1)
            
            print(f"🔍 Validating backup: {backup.backup_id}")
            print(f"Component: {backup.component}")
            print(f"Created: {backup.timestamp}")
            print()
            
            # Run validation
            if backup.component == "neo4j":
                result = await self.dr_system.orchestrator.backup_manager.neo4j_service.validate_backup(backup)
            elif backup.component == "vector_index":
                result = await self.dr_system.orchestrator.backup_manager.vector_service.validate_backup(backup)
            else:
                print(f"❌ Unknown backup component: {backup.component}")
                sys.exit(1)
            
            if result['valid']:
                print("✅ Backup validation successful!")
                if 'node_count' in result:
                    print(f"   Nodes: {result['node_count']:,}")
                if 'relationship_count' in result:
                    print(f"   Relationships: {result['relationship_count']:,}")
                if 'vector_count' in result:
                    print(f"   Vectors: {result['vector_count']:,}")
                if 'dimension' in result:
                    print(f"   Dimensions: {result['dimension']}")
            else:
                print(f"❌ Backup validation failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
            
        except Exception as e:
            logger.error(f"Backup validation failed: {e}")
            print(f"❌ Backup validation failed: {e}")
            sys.exit(1)
    
    async def execute_recovery(self, timestamp_str: Optional[str] = None):
        """Execute disaster recovery"""
        try:
            target_timestamp = None
            if timestamp_str:
                try:
                    target_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")
                except ValueError:
                    print("❌ Invalid timestamp format. Use: YYYY-MM-DD-HH-MM-SS")
                    sys.exit(1)
            
            print("🚨 DISASTER RECOVERY EXECUTION")
            print("==============================")
            if target_timestamp:
                print(f"Target timestamp: {target_timestamp}")
            else:
                print("Target timestamp: Latest available")
            print()
            
            # Confirm execution
            response = input("⚠️  This will restore the system from backup. Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Recovery cancelled.")
                return
            
            print("\n🔄 Starting disaster recovery...")
            start_time = datetime.now()
            
            success = await self.dr_system.restore_from_backup(target_timestamp)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            if success:
                print(f"\n✅ Disaster recovery completed successfully!")
                print(f"Recovery time: {duration:.2f} minutes")
                
                # Check RTO compliance
                rto_target = self.dr_system.config.rto_target_minutes
                if duration <= rto_target:
                    print(f"✅ RTO target met ({duration:.2f} ≤ {rto_target} minutes)")
                else:
                    print(f"⚠️  RTO target exceeded ({duration:.2f} > {rto_target} minutes)")
            else:
                print(f"\n❌ Disaster recovery failed!")
                print(f"Duration: {duration:.2f} minutes")
                sys.exit(1)
            
        except Exception as e:
            logger.error(f"Disaster recovery failed: {e}")
            print(f"❌ Disaster recovery failed: {e}")
            sys.exit(1)
    
    async def test_recovery(self):
        """Run disaster recovery test"""
        try:
            print("🧪 DISASTER RECOVERY TEST")
            print("==========================")
            print("Running game day exercise...\n")
            
            start_time = datetime.now()
            result = await self.dr_system.test_disaster_recovery()
            end_time = datetime.now()
            
            print(f"Test ID: {result['test_id']}")
            print(f"Duration: {result.get('test_time_minutes', 0):.2f} minutes")
            
            if result['success']:
                print("✅ Disaster recovery test PASSED!")
                
                rto_compliance = result.get('rto_compliance', False)
                if rto_compliance:
                    print("✅ RTO target compliance: PASSED")
                else:
                    print("⚠️  RTO target compliance: FAILED")
            else:
                print("❌ Disaster recovery test FAILED!")
                if 'error' in result:
                    print(f"Error: {result['error']}")
                sys.exit(1)
            
        except Exception as e:
            logger.error(f"Disaster recovery test failed: {e}")
            print(f"❌ Disaster recovery test failed: {e}")
            sys.exit(1)
    
    async def show_status(self):
        """Show system status"""
        try:
            print("📊 DISASTER RECOVERY SYSTEM STATUS")
            print("===================================")
            
            # System configuration
            config = self.dr_system.config
            print(f"Backup frequency: Every {config.backup_frequency_minutes} minutes")
            print(f"Backup retention: {config.backup_retention_days} days")
            print(f"RTO target: {config.rto_target_minutes} minutes")
            print(f"RPO target: {config.rpo_target_minutes} minutes")
            print(f"Remote backup: {'Enabled' if config.remote_backup_enabled else 'Disabled'}")
            print(f"Validation: {'Enabled' if config.validation_enabled else 'Disabled'}")
            print()
            
            # Backup statistics
            backups = self.dr_system.get_backup_status()
            neo4j_backups = [b for b in backups if b['component'] == 'neo4j']
            vector_backups = [b for b in backups if b['component'] == 'vector_index']
            validated_backups = [b for b in backups if b['validated']]
            
            print(f"📈 Backup Statistics:")
            print(f"  Total backups: {len(backups)}")
            print(f"  Neo4j backups: {len(neo4j_backups)}")
            print(f"  Vector backups: {len(vector_backups)}")
            print(f"  Validated backups: {len(validated_backups)}")
            print()
            
            # Latest backups
            if backups:
                latest_neo4j = max([b for b in neo4j_backups], key=lambda x: x['timestamp'], default=None)
                latest_vector = max([b for b in vector_backups], key=lambda x: x['timestamp'], default=None)
                
                print(f"🕐 Latest Backups:")
                if latest_neo4j:
                    print(f"  Neo4j: {latest_neo4j['timestamp']} ({'✅ Validated' if latest_neo4j['validated'] else '⚠️ Not validated'})")
                if latest_vector:
                    print(f"  Vector: {latest_vector['timestamp']} ({'✅ Validated' if latest_vector['validated'] else '⚠️ Not validated'})")
                print()
            
            # Recovery status
            recovery_status = self.dr_system.orchestrator.recovery_status
            print(f"🔄 Recovery Status: {recovery_status.value}")
            
        except Exception as e:
            logger.error(f"Failed to show status: {e}")
            print(f"❌ Failed to show status: {e}")
            sys.exit(1)
    
    async def monitor(self):
        """Monitor system in real-time"""
        try:
            print("📡 DISASTER RECOVERY MONITORING")
            print("===============================")
            print("Press Ctrl+C to stop monitoring\n")
            
            while True:
                # Clear screen (simple version)
                print("\033[2J\033[H", end="")
                
                print(f"📡 Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 50)
                
                # Show current status
                await self.show_status()
                
                # Wait before next update
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            print(f"❌ Monitoring failed: {e}")
            sys.exit(1)

def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="KGoT-Alita Disaster Recovery CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dr_cli.py backup create
  python dr_cli.py backup list
  python dr_cli.py backup validate abc12345
  python dr_cli.py recovery execute --timestamp 2024-01-15-14-30-00
  python dr_cli.py recovery test
  python dr_cli.py status
  python dr_cli.py monitor
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to disaster recovery configuration file',
        default='infrastructure/disaster_recovery_config.json'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backup commands
    backup_parser = subparsers.add_parser('backup', help='Backup operations')
    backup_subparsers = backup_parser.add_subparsers(dest='backup_action')
    
    backup_subparsers.add_parser('create', help='Create manual backup')
    backup_subparsers.add_parser('list', help='List all backups')
    
    validate_parser = backup_subparsers.add_parser('validate', help='Validate backup')
    validate_parser.add_argument('backup_id', help='Backup ID to validate')
    
    # Recovery commands
    recovery_parser = subparsers.add_parser('recovery', help='Recovery operations')
    recovery_subparsers = recovery_parser.add_subparsers(dest='recovery_action')
    
    execute_parser = recovery_subparsers.add_parser('execute', help='Execute disaster recovery')
    execute_parser.add_argument(
        '--timestamp', '-t',
        help='Target timestamp (YYYY-MM-DD-HH-MM-SS)',
        default=None
    )
    
    recovery_subparsers.add_parser('test', help='Run disaster recovery test')
    
    # Status and monitoring
    subparsers.add_parser('status', help='Show system status')
    subparsers.add_parser('monitor', help='Monitor system in real-time')
    
    return parser

async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = DisasterRecoveryCLI(args.config)
    
    try:
        await cli.initialize()
        
        if args.command == 'backup':
            if args.backup_action == 'create':
                await cli.create_backup()
            elif args.backup_action == 'list':
                await cli.list_backups()
            elif args.backup_action == 'validate':
                await cli.validate_backup(args.backup_id)
            else:
                parser.print_help()
        
        elif args.command == 'recovery':
            if args.recovery_action == 'execute':
                await cli.execute_recovery(args.timestamp)
            elif args.recovery_action == 'test':
                await cli.test_recovery()
            else:
                parser.print_help()
        
        elif args.command == 'status':
            await cli.show_status()
        
        elif args.command == 'monitor':
            await cli.monitor()
        
        else:
            parser.print_help()
    
    finally:
        await cli.cleanup()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"CLI error: {e}")
        sys.exit(1)