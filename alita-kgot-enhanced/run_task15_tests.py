#!/usr/bin/env python3
"""
Task 15 Test Runner Script

This script handles dependency installation and runs the Task 15 integration tests
with proper error handling for asyncio event loop issues.

Usage:
    python run_task15_tests.py

@author Alita KGoT Enhanced Team  
@date 2025
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required test dependencies"""
    print("🔧 Installing test dependencies...")
    
    dependencies = [
        'pytest',
        'pytest-asyncio', 
        'nest_asyncio',
        'unittest-xml-reporting'
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                         check=True, capture_output=True)
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not install {dep}: {e}")
            print(f"   You may need to install it manually: pip install {dep}")

def run_tests():
    """Run the Task 15 integration tests"""
    print("\n🧪 Running Task 15 Integration Tests...")
    
    # Set working directory to the script location
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Set environment variables for testing
    os.environ.setdefault('OPENROUTER_API_KEY', 'test_placeholder')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    try:
        # Try to run with pytest first (more robust)
        print("Attempting to run with pytest...")
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'test_integration_task15.py', 
            '-v', '--tb=short'
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 Tests completed successfully with pytest!")
            return True
        else:
            print(f"\n⚠️  Pytest returned code {result.returncode}, trying direct execution...")
            
    except FileNotFoundError:
        print("Pytest not available, running directly...")
    
    # Fallback to direct execution
    try:
        print("Running test file directly...")
        result = subprocess.run([sys.executable, 'test_integration_task15.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 Tests completed successfully!")
            return True
        else:
            print(f"\n❌ Tests failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running tests: {str(e)}")
        return False

def main():
    """Main entry point"""
    print("=" * 60)
    print("🧪 Task 15 Integration Test Runner")
    print("   KGoT Surfer Agent + Alita Web Agent Integration")
    print("=" * 60)
    
    # Install dependencies
    install_dependencies()
    
    # Run tests
    success = run_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ SUCCESS: Task 15 implementation is ready!")
        print("   All integration tests passed.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ ISSUES DETECTED: Please review test output above.")
        print("   Some tests may have failed or encountered errors.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main() 