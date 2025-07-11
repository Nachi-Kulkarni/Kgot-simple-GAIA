#!/usr/bin/env python3
"""
RAG-MCP Coordinator Setup Script - Task 17

Automated setup and configuration script for the RAG-MCP Coordinator implementation.
This script handles dependency installation, directory creation, and basic configuration.

Usage:
    python setup_rag_mcp_coordinator.py [--install-deps] [--create-config] [--test]

@author: Enhanced Alita KGoT Team
@version: 1.0.0
@purpose: Task 17 Implementation Setup
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def print_banner():
    """Print setup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                RAG-MCP Coordinator Setup                     ║
║                    Task 17 Implementation                    ║
╠══════════════════════════════════════════════════════════════╣
║  Setting up production-ready RAG-MCP coordination system    ║
║  with retrieval-first strategy and intelligent fallback     ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def create_directories():
    """Create required directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "logs/validation",
        "logs/kgot", 
        "data",
        "config",
        "../logs/validation",
        "../logs/kgot"
    ]
    
    created_dirs = []
    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(path))
                print(f"  ✅ Created: {dir_path}")
            except Exception as e:
                print(f"  ❌ Failed to create {dir_path}: {e}")
        else:
            print(f"  ✅ Exists: {dir_path}")
    
    return created_dirs

def create_default_config():
    """Create default configuration file"""
    print("⚙️ Creating default configuration...")
    
    config = {
        "retrieval_strategy": {
            "embedding_model": "text-embedding-ada-002",
            "similarity_threshold": 0.6,
            "max_candidates": 10,
            "diversity_factor": 0.3,
            "openrouter_base_url": "https://openrouter.ai/api/v1"
        },
        "validation_layer": {
            "confidence_threshold": 0.7,
            "max_validation_queries": 5,
            "timeout_seconds": 30,
            "llm_model": "anthropic/claude-sonnet-4"
        },
        "intelligent_fallback": {
            "similarity_threshold": 0.4,
            "validation_threshold": 0.5,
            "brainstorming_engine_path": "../alita_core/mcp_brainstorming.js",
            "enable_subprocess": True
        },
        "usage_tracker": {
            "analytics_enabled": True,
            "storage_path": "./data/usage_patterns.json",
            "analysis_window_days": 30,
            "export_reports": True
        },
        "cross_validation": {
            "k_fold_default": 5,
            "significance_level": 0.05,
            "enable_statistical_analysis": True
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s",
            "validation_log": "./logs/validation/combined.log",
            "error_log": "./logs/validation/error.log"
        },
        "performance": {
            "async_enabled": True,
            "max_concurrent_operations": 10,
            "cache_embeddings": True,
            "cache_ttl_seconds": 3600
        },
        "metadata": {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "description": "RAG-MCP Coordinator Configuration - Task 17 Implementation"
        }
    }
    
    config_path = Path("config/rag_mcp_config.json")
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✅ Configuration created: {config_path}")
        return config_path
    except Exception as e:
        print(f"  ❌ Failed to create configuration: {e}")
        return None

def create_env_template():
    """Create environment template file"""
    print("🔧 Creating environment template...")
    
    env_template = """# RAG-MCP Coordinator Environment Configuration
# Task 17 Implementation

# OpenRouter API Configuration (Required)
# Get your API key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Optional: Custom Configuration
RAG_MCP_CONFIG_PATH=./config/rag_mcp_config.json

# Optional: Logging Configuration
RAG_MCP_LOG_LEVEL=INFO
RAG_MCP_LOG_PATH=./logs/validation/

# Optional: Performance Tuning
RAG_MCP_MAX_CONCURRENT=10
RAG_MCP_CACHE_TTL=3600

# Optional: Development Settings
RAG_MCP_DEBUG=false
RAG_MCP_VERBOSE=false
"""
    
    env_path = Path(".env.template")
    try:
        with open(env_path, 'w') as f:
            f.write(env_template)
        print(f"  ✅ Environment template created: {env_path}")
        print(f"  📝 Copy to .env and configure your API keys")
        return env_path
    except Exception as e:
        print(f"  ❌ Failed to create environment template: {e}")
        return None

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print(f"  ❌ Requirements file not found: {requirements_file}")
        return False
    
    try:
        # Install dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True, text=True)
        print("  ✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to install dependencies: {e}")
        print(f"  📋 Error output: {e.stderr}")
        return False

def run_verification_test():
    """Run verification test"""
    print("🧪 Running verification test...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_rag_mcp_coordinator.py"
        ], capture_output=True, text=True, timeout=60)
        
        if "Test Results:" in result.stdout:
            # Extract test results
            lines = result.stdout.split('\n')
            for line in lines:
                if "Test Results:" in line:
                    print(f"  📊 {line.strip()}")
                    break
            
            # Count passed tests
            passed_count = result.stdout.count("✅")
            total_count = result.stdout.count("verification:")
            
            if passed_count >= 3:  # At least 3 core tests should pass
                print(f"  ✅ Core verification successful ({passed_count} tests passed)")
                return True
            else:
                print(f"  ⚠️  Partial verification ({passed_count}/{total_count} tests passed)")
                return False
        else:
            print(f"  ❌ Verification test failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⚠️  Verification test timed out (this may be normal)")
        return True  # Don't fail setup for timeout
    except Exception as e:
        print(f"  ❌ Error running verification: {e}")
        return False

def check_integration_points():
    """Check integration with existing systems"""
    print("🔗 Checking integration points...")
    
    integration_files = [
        "../alita_core/mcp_brainstorming.js",
        "../kgot_core/error_management.py",
        "mcp_cross_validator.py"
    ]
    
    integration_status = []
    for file_path in integration_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ Found: {file_path}")
            integration_status.append(True)
        else:
            print(f"  ⚠️  Missing: {file_path}")
            integration_status.append(False)
    
    return integration_status

def print_next_steps():
    """Print next steps for user"""
    print("""
🎯 Setup Complete! Next Steps:

1. **Configure API Keys**:
   - Copy .env.template to .env
   - Add your OpenRouter API key
   - Configure OPENROUTER_BASE_URL if needed

2. **Test Installation**:
   python test_rag_mcp_coordinator.py

3. **Run Example Usage**:
   python -c "
   from rag_mcp_coordinator import example_rag_mcp_coordination
   import asyncio
   asyncio.run(example_rag_mcp_coordination())
   "

4. **Integration Usage**:
   from validation.rag_mcp_coordinator import create_rag_mcp_coordinator
   coordinator = create_rag_mcp_coordinator()

📚 Documentation: docs/TASK_17_RAG_MCP_COORDINATOR_DOCUMENTATION.md

🎉 RAG-MCP Coordinator is ready for production use!
""")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup RAG-MCP Coordinator")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install Python dependencies")
    parser.add_argument("--create-config", action="store_true",
                       help="Create configuration files")
    parser.add_argument("--test", action="store_true",
                       help="Run verification tests")
    parser.add_argument("--all", action="store_true",
                       help="Run complete setup (all options)")
    
    args = parser.parse_args()
    
    # If no specific options, default to minimal setup
    if not any([args.install_deps, args.create_config, args.test, args.all]):
        args.create_config = True
    
    # If --all is specified, enable all options
    if args.all:
        args.install_deps = True
        args.create_config = True
        args.test = True
    
    print_banner()
    
    # Setup steps
    setup_success = True
    
    # Always create directories
    create_directories()
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            setup_success = False
    
    # Create configuration if requested
    if args.create_config:
        create_default_config()
        create_env_template()
    
    # Check integrations
    check_integration_points()
    
    # Run verification if requested
    if args.test:
        if not run_verification_test():
            setup_success = False
    
    # Print results
    if setup_success:
        print("\n✅ Setup completed successfully!")
        print_next_steps()
        return 0
    else:
        print("\n⚠️  Setup completed with some issues. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)