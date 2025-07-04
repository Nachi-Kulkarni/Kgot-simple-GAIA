#!/usr/bin/env python3
"""
Startup Script for LangChain Sequential Manager Agent

This script initializes and starts the LangChain Sequential Manager with proper
environment configuration, logging setup, and error handling.

Usage:
    python start_langchain_manager.py [--port PORT] [--host HOST] [--config CONFIG_PATH]

Environment Variables:
    OPENROUTER_API_KEY: Required - OpenRouter API key for model access
    PORT: Optional - Server port (default: 8000)
    HOST: Optional - Server host (default: 0.0.0.0)
    LOG_LEVEL: Optional - Logging level (default: INFO)
    NODE_ENV: Optional - Environment mode (development/production)

@author Alita-KGoT Development Team
@version 1.0.0
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from alita_core.manager_agent.langchain_sequential_manager import LangChainSequentialManager
except ImportError as e:
    print(f"Error importing LangChainSequentialManager: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


def setup_environment():
    """
    Setup environment variables and validate configuration
    
    Returns:
        bool: True if environment is properly configured, False otherwise
    """
    # Check for required environment variables
    required_env_vars = ['OPENROUTER_API_KEY']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following environment variables:")
        print("export OPENROUTER_API_KEY='your_openrouter_api_key'")
        print("\nOptional environment variables:")
        print("export PORT=8000")
        print("export HOST=0.0.0.0")
        print("export LOG_LEVEL=INFO")
        print("export NODE_ENV=development")
        return False
    
    # Set default values for optional variables
    os.environ.setdefault('PORT', '8000')
    os.environ.setdefault('HOST', '0.0.0.0')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    os.environ.setdefault('NODE_ENV', 'production')
    
    return True


def setup_logging():
    """Setup logging configuration"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/langchain_manager.log', mode='a')
        ]
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete - Level: {log_level}")
    return logger


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='LangChain Sequential Manager Agent Startup Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python start_langchain_manager.py
    python start_langchain_manager.py --port 8080 --host localhost
    python start_langchain_manager.py --config custom_config.json
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=None,
        help='Server port (overrides PORT environment variable)'
    )
    
    parser.add_argument(
        '--host', '-H',
        type=str,
        default=None,
        help='Server host (overrides HOST environment variable)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file (default: config/models/model_config.json)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode (sets LOG_LEVEL=DEBUG and NODE_ENV=development)'
    )
    
    return parser.parse_args()


async def main():
    """
    Main startup function for the LangChain Sequential Manager
    """
    print("🚀 Starting LangChain Sequential Manager Agent...")
    print("=" * 60)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle debug mode
    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'
        os.environ['NODE_ENV'] = 'development'
        print("🐛 Debug mode enabled")
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging()
    
    # Determine configuration
    config_path = args.config or "config/models/model_config.json"
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        logger.info("Using default configuration")
    
    # Determine host and port
    host = args.host or os.getenv('HOST', '0.0.0.0')
    port = args.port or int(os.getenv('PORT', 8000))
    
    print(f"📝 Configuration: {config_path}")
    print(f"🌐 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔑 API Key: {'✓ Set' if os.getenv('OPENROUTER_API_KEY') else '✗ Missing'}")
    print(f"📊 Log Level: {os.getenv('LOG_LEVEL', 'INFO')}")
    print(f"🔧 Environment: {os.getenv('NODE_ENV', 'production')}")
    print("=" * 60)
    
    try:
        # Initialize the LangChain Sequential Manager
        print("🧠 Initializing LangChain Sequential Manager...")
        manager = LangChainSequentialManager(config_path=config_path)
        
        print("✅ Manager initialized successfully")
        print("🤖 Starting agent services...")
        
        # Start the server
        print(f"🌟 LangChain Sequential Manager is starting on http://{host}:{port}")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        await manager.start_server(host=host, port=port)
        
    except KeyboardInterrupt:
        print("\n🛑 Received shutdown signal...")
        await manager.shutdown()
        print("✅ LangChain Sequential Manager stopped gracefully")
        
    except Exception as e:
        logger.error(f"Failed to start LangChain Sequential Manager: {e}")
        print(f"❌ Error: {e}")
        
        # Print troubleshooting tips
        print("\n🔧 Troubleshooting Tips:")
        print("1. Verify OPENROUTER_API_KEY is set correctly")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. Ensure configuration file exists and is valid JSON")
        print("4. Check that the specified port is not in use")
        print("5. Review logs for detailed error information")
        
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 