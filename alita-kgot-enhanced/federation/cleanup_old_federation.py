#!/usr/bin/env python3
"""
Federation System Cleanup Script
===============================
Safely removes or backs up the old authenticated MCP Federation System components.

Usage:
    python cleanup_old_federation.py --backup     # Backup old files
    python cleanup_old_federation.py --remove     # Remove old files
    python cleanup_old_federation.py --status     # Show current status
    python cleanup_old_federation.py --restore    # Restore from backup
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class FederationCleanup:
    """Manages cleanup of old federation system components."""
    
    def __init__(self, federation_dir: Path = None):
        self.federation_dir = federation_dir or Path(__file__).parent
        self.backup_dir = self.federation_dir / "backup_old_federation"
        self.manifest_file = self.backup_dir / "backup_manifest.json"
        
        # Files to backup/remove
        self.old_federation_files = [
            "mcp_federation.py",
            "federated_rag_mcp_engine.py",
            "tests/test_federation_api.py"
        ]
        
        # Documentation files that reference old federation
        self.docs_to_update = [
            "../docs/TASK_44_MCP_FEDERATION_SYSTEM.md",
            "README_SIMPLE_LOCAL_SERVER.md"
        ]
    
    def get_status(self) -> Dict[str, any]:
        """Get current status of federation files."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "old_files_exist": {},
            "new_files_exist": {},
            "backup_exists": self.backup_dir.exists(),
            "backup_files": []
        }
        
        # Check old federation files
        for file_path in self.old_federation_files:
            full_path = self.federation_dir / file_path
            status["old_files_exist"][file_path] = full_path.exists()
        
        # Check new simple federation files
        new_files = [
            "simple_local_mcp_server.py",
            "simple_federated_rag_mcp_engine.py",
            "test_simple_mcp_system.py",
            "start_simple_server.py"
        ]
        
        for file_path in new_files:
            full_path = self.federation_dir / file_path
            status["new_files_exist"][file_path] = full_path.exists()
        
        # Check backup files
        if self.backup_dir.exists():
            status["backup_files"] = [f.name for f in self.backup_dir.rglob("*") if f.is_file()]
        
        return status
    
    def create_backup(self) -> bool:
        """Create backup of old federation files."""
        print("🔄 Creating backup of old federation files...")
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        
        backup_manifest = {
            "timestamp": datetime.now().isoformat(),
            "backed_up_files": [],
            "original_locations": {}
        }
        
        backed_up_count = 0
        
        for file_path in self.old_federation_files:
            source_path = self.federation_dir / file_path
            
            if source_path.exists():
                # Create backup path maintaining directory structure
                backup_path = self.backup_dir / file_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_path, backup_path)
                
                backup_manifest["backed_up_files"].append(file_path)
                backup_manifest["original_locations"][file_path] = str(source_path)
                
                print(f"  ✅ Backed up: {file_path}")
                backed_up_count += 1
            else:
                print(f"  ⚠️  File not found: {file_path}")
        
        # Save manifest
        with open(self.manifest_file, 'w') as f:
            json.dump(backup_manifest, f, indent=2)
        
        print(f"\n✅ Backup complete! {backed_up_count} files backed up to {self.backup_dir}")
        return backed_up_count > 0
    
    def remove_old_files(self, confirm: bool = True) -> bool:
        """Remove old federation files."""
        if confirm:
            print("⚠️  This will permanently remove old federation files!")
            response = input("Are you sure? (yes/no): ").lower().strip()
            if response != "yes":
                print("❌ Operation cancelled")
                return False
        
        print("🗑️  Removing old federation files...")
        
        removed_count = 0
        
        for file_path in self.old_federation_files:
            full_path = self.federation_dir / file_path
            
            if full_path.exists():
                if full_path.is_file():
                    full_path.unlink()
                    print(f"  ✅ Removed: {file_path}")
                    removed_count += 1
                elif full_path.is_dir():
                    shutil.rmtree(full_path)
                    print(f"  ✅ Removed directory: {file_path}")
                    removed_count += 1
            else:
                print(f"  ⚠️  File not found: {file_path}")
        
        print(f"\n✅ Removal complete! {removed_count} files removed")
        return removed_count > 0
    
    def restore_from_backup(self) -> bool:
        """Restore files from backup."""
        if not self.backup_dir.exists():
            print("❌ No backup directory found")
            return False
        
        if not self.manifest_file.exists():
            print("❌ No backup manifest found")
            return False
        
        print("🔄 Restoring files from backup...")
        
        # Load manifest
        with open(self.manifest_file, 'r') as f:
            manifest = json.load(f)
        
        restored_count = 0
        
        for file_path in manifest.get("backed_up_files", []):
            backup_path = self.backup_dir / file_path
            restore_path = self.federation_dir / file_path
            
            if backup_path.exists():
                # Create parent directories if needed
                restore_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file back
                shutil.copy2(backup_path, restore_path)
                print(f"  ✅ Restored: {file_path}")
                restored_count += 1
            else:
                print(f"  ❌ Backup file not found: {file_path}")
        
        print(f"\n✅ Restore complete! {restored_count} files restored")
        return restored_count > 0
    
    def show_status(self):
        """Display current status."""
        status = self.get_status()
        
        print("📊 Federation System Status")
        print("=" * 40)
        
        print("\n🗂️  Old Federation Files:")
        for file_path, exists in status["old_files_exist"].items():
            icon = "✅" if exists else "❌"
            print(f"  {icon} {file_path}")
        
        print("\n🆕 New Simple Federation Files:")
        for file_path, exists in status["new_files_exist"].items():
            icon = "✅" if exists else "❌"
            print(f"  {icon} {file_path}")
        
        print(f"\n💾 Backup Status:")
        if status["backup_exists"]:
            print(f"  ✅ Backup directory exists: {self.backup_dir}")
            print(f"  📁 Backup files: {len(status['backup_files'])}")
            for backup_file in status["backup_files"][:5]:  # Show first 5
                print(f"    - {backup_file}")
            if len(status["backup_files"]) > 5:
                print(f"    ... and {len(status['backup_files']) - 5} more")
        else:
            print("  ❌ No backup directory found")
        
        # Recommendations
        print("\n💡 Recommendations:")
        old_files_count = sum(status["old_files_exist"].values())
        new_files_count = sum(status["new_files_exist"].values())
        
        if old_files_count > 0 and new_files_count > 0:
            print("  🔄 Both old and new federation systems are present")
            print("  📝 Consider running --backup then --remove to clean up")
        elif old_files_count > 0:
            print("  ⚠️  Only old federation system found")
            print("  📝 Run the setup to create new simple federation files")
        elif new_files_count > 0:
            print("  ✅ New simple federation system is ready")
            print("  🚀 You can start using the simple local server")
        else:
            print("  ❌ No federation system files found")
            print("  📝 Run the setup to create federation files")
    
    def update_documentation(self):
        """Update documentation to reflect the migration."""
        print("📝 Updating documentation...")
        
        # Add migration notice to TASK_44 documentation
        task_44_path = self.federation_dir / "../docs/TASK_44_MCP_FEDERATION_SYSTEM.md"
        if task_44_path.exists():
            with open(task_44_path, 'r') as f:
                content = f.read()
            
            migration_notice = """

## 🔄 Migration Notice

**IMPORTANT**: This documentation describes the original authenticated MCP Federation System. 
For local development, consider using the simplified local MCP server instead:

- **Simple Local Server**: `federation/simple_local_mcp_server.py`
- **Migration Guide**: `federation/MIGRATION_GUIDE.md`
- **Quick Start**: `federation/start_simple_server.py --demo`

The simple local server removes authentication complexity and is perfect for development and testing.
"""
            
            if "Migration Notice" not in content:
                # Insert after the first heading
                lines = content.split('\n')
                insert_index = 1
                for i, line in enumerate(lines):
                    if line.startswith('## '):
                        insert_index = i
                        break
                
                lines.insert(insert_index, migration_notice)
                
                with open(task_44_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                print(f"  ✅ Updated: {task_44_path.name}")
            else:
                print(f"  ⚠️  Migration notice already exists in: {task_44_path.name}")
        
        print("📝 Documentation update complete")


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup old MCP Federation System components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_old_federation.py --status     # Check current status
  python cleanup_old_federation.py --backup     # Backup old files
  python cleanup_old_federation.py --remove     # Remove old files (with confirmation)
  python cleanup_old_federation.py --restore    # Restore from backup
        """
    )
    
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--backup", action="store_true", help="Backup old federation files")
    parser.add_argument("--remove", action="store_true", help="Remove old federation files")
    parser.add_argument("--restore", action="store_true", help="Restore files from backup")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--update-docs", action="store_true", help="Update documentation")
    
    args = parser.parse_args()
    
    # Change to federation directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    cleanup = FederationCleanup()
    
    print("🧹 Federation System Cleanup Tool")
    print("=" * 35)
    
    if args.status or not any([args.backup, args.remove, args.restore, args.update_docs]):
        cleanup.show_status()
    
    if args.backup:
        cleanup.create_backup()
    
    if args.remove:
        cleanup.remove_old_files(confirm=not args.force)
    
    if args.restore:
        cleanup.restore_from_backup()
    
    if args.update_docs:
        cleanup.update_documentation()
    
    print("\n🎉 Cleanup operations complete!")
    print("\n📚 Next steps:")
    print("  1. Test the simple federation system: python test_simple_mcp_system.py")
    print("  2. Start the simple server: python start_simple_server.py --demo")
    print("  3. Read the migration guide: MIGRATION_GUIDE.md")


if __name__ == "__main__":
    main()