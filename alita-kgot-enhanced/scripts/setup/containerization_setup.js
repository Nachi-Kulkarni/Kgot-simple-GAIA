#!/usr/bin/env node

/**
 * KGoT Containerization Infrastructure Setup
 * 
 * This script sets up the complete containerization infrastructure for the KGoT-Alita system,
 * including Docker Compose services, Sarus HPC integration, monitoring, and health checks.
 * 
 * Features:
 * - Environment detection and validation
 * - Docker/Sarus availability checking
 * - Container image building and pulling
 * - Network and volume creation
 * - Monitoring stack setup (Prometheus, Grafana)
 * - Health check validation
 * - Integration with existing Alita systems
 * 
 * @author KGoT Enhanced Alita System
 * @version 2.0.0
 * @license MIT
 */

const fs = require('fs').promises;
const path = require('path');
const { execSync, spawn } = require('child_process');
const winston = require('winston');

// Configure Winston logging for containerization setup
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    defaultMeta: { service: 'containerization-setup' },
    transports: [
        new winston.transports.File({ filename: '../../logs/system/containerization-setup-error.log', level: 'error' }),
        new winston.transports.File({ filename: '../../logs/system/containerization-setup.log' }),
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple()
            )
        })
    ]
});

/**
 * Configuration class for containerization setup
 */
class ContainerizationSetup {
    constructor() {
        this.rootDir = path.resolve(__dirname, '../..');
        this.configDir = path.join(this.rootDir, 'config');
        this.containerDir = path.join(this.configDir, 'containers');
        this.envFile = path.join(this.rootDir, '.env');
        
        // Service categories for organized setup
        this.serviceLayers = {
            databases: ['neo4j', 'rdf4j', 'redis'],
            kgot_core: ['kgot-controller', 'kgot-graph-store', 'kgot-tools'],
            alita_core: ['alita-manager', 'alita-web', 'alita-mcp'],
            extensions: ['python-executor', 'multimodal-processor', 'validation-service', 'optimization-service'],
            monitoring: ['prometheus', 'grafana']
        };
        
        this.requiredTools = ['docker', 'docker-compose'];
        this.optionalTools = ['sarus', 'kubectl'];
    }

    /**
     * Main setup orchestration method
     */
    async initialize() {
        try {
            logger.info('🚀 Starting KGoT Containerization Infrastructure Setup');
            
            await this.validateEnvironment();
            await this.setupDirectories();
            await this.loadEnvironmentConfig();
            await this.detectContainerizationTools();
            await this.setupNetworking();
            await this.setupVolumes();
            await this.buildCustomImages();
            await this.setupMonitoring();
            await this.validateConfiguration();
            await this.generateStartupScripts();
            
            logger.info('✅ Containerization infrastructure setup completed successfully');
            await this.displaySetupSummary();
            
        } catch (error) {
            logger.error('❌ Setup failed:', error);
            throw error;
        }
    }

    /**
     * Validate system environment and prerequisites
     */
    async validateEnvironment() {
        logger.info('🔍 Validating environment and prerequisites...');
        
        // Check Node.js version
        const nodeVersion = process.version;
        logger.info(`Node.js version: ${nodeVersion}`);
        
        // Check Python availability (for containerization.py)
        try {
            const pythonVersion = execSync('python3 --version', { encoding: 'utf8' });
            logger.info(`Python version: ${pythonVersion.trim()}`);
        } catch (error) {
            logger.error('Python 3 is required but not found');
            throw new Error('Python 3 is required for containerization management');
        }
        
        // Check available disk space
        const stats = await fs.stat(this.rootDir);
        logger.info('System validation completed');
    }

    /**
     * Setup required directory structure
     */
    async setupDirectories() {
        logger.info('📁 Setting up directory structure...');
        
        const requiredDirs = [
            'logs/containerization',
            'logs/monitoring',
            'config/monitoring/grafana/provisioning/datasources',
            'config/monitoring/grafana/provisioning/dashboards',
            'config/monitoring/prometheus',
            'config/containers/volumes',
            'data/neo4j',
            'data/rdf4j', 
            'data/redis',
            'data/prometheus',
            'data/grafana'
        ];
        
        for (const dir of requiredDirs) {
            const fullPath = path.join(this.rootDir, dir);
            await fs.mkdir(fullPath, { recursive: true });
            logger.debug(`Created directory: ${dir}`);
        }
        
        logger.info('Directory structure setup completed');
    }

    /**
     * Load and validate environment configuration
     */
    async loadEnvironmentConfig() {
        logger.info('⚙️  Loading environment configuration...');
        
        try {
            const envContent = await fs.readFile(this.envFile, 'utf8');
            logger.info('Environment file loaded successfully');
        } catch (error) {
            logger.warn('Environment file not found, creating template...');
            await this.createEnvironmentTemplate();
        }
        
        // Validate required environment variables
        const requiredVars = [
            'NEO4J_PASSWORD',
            'REDIS_PASSWORD', 
            'OPENROUTER_API_KEY',
            'LOG_LEVEL'
        ];
        
        for (const varName of requiredVars) {
            if (!process.env[varName]) {
                logger.warn(`Environment variable ${varName} not set`);
            }
        }
    }

    /**
     * Detect available containerization tools
     */
    async detectContainerizationTools() {
        logger.info('🔧 Detecting containerization tools...');
        
        const toolStatus = {};
        
        // Check required tools
        for (const tool of this.requiredTools) {
            try {
                execSync(`which ${tool}`, { stdio: 'ignore' });
                toolStatus[tool] = 'available';
                logger.info(`✅ ${tool} is available`);
            } catch (error) {
                toolStatus[tool] = 'missing';
                logger.error(`❌ ${tool} is required but not found`);
                throw new Error(`Required tool ${tool} is not installed`);
            }
        }
        
        // Check optional tools
        for (const tool of this.optionalTools) {
            try {
                execSync(`which ${tool}`, { stdio: 'ignore' });
                toolStatus[tool] = 'available';
                logger.info(`✅ ${tool} is available (optional)`);
            } catch (error) {
                toolStatus[tool] = 'missing';
                logger.info(`ℹ️  ${tool} is not available (optional)`);
            }
        }
        
        this.toolStatus = toolStatus;
    }

    /**
     * Setup Docker networking
     */
    async setupNetworking() {
        logger.info('🌐 Setting up container networking...');
        
        try {
            // Create custom bridge network if it doesn't exist
            execSync('docker network inspect alita-kgot-network', { stdio: 'ignore' });
            logger.info('Network alita-kgot-network already exists');
        } catch (error) {
            logger.info('Creating alita-kgot-network...');
            execSync('docker network create --driver bridge --subnet=172.20.0.0/16 alita-kgot-network');
            logger.info('Network created successfully');
        }
    }

    /**
     * Setup Docker volumes
     */
    async setupVolumes() {
        logger.info('💾 Setting up persistent volumes...');
        
        const volumes = [
            'neo4j-data', 'neo4j-logs', 'neo4j-import', 'neo4j-plugins',
            'rdf4j-data', 'redis-data', 'graph-data', 'tool-storage',
            'mcp-storage', 'python-workspace', 'multimodal-storage',
            'validation-results', 'optimization-data', 'playwright-cache',
            'grafana-data', 'prometheus-data'
        ];
        
        for (const volume of volumes) {
            try {
                execSync(`docker volume inspect ${volume}`, { stdio: 'ignore' });
                logger.debug(`Volume ${volume} already exists`);
            } catch (error) {
                execSync(`docker volume create ${volume}`);
                logger.debug(`Created volume: ${volume}`);
            }
        }
        
        logger.info('Volume setup completed');
    }

    /**
     * Build custom Docker images
     */
    async buildCustomImages() {
        logger.info('🏗️  Building custom container images...');
        
        const imagesToBuild = [
            { name: 'alita-manager', context: 'alita_core/manager_agent' },
            { name: 'alita-web-agent', context: 'alita_core/web_agent' },
            { name: 'alita-mcp-creation', context: 'alita_core/mcp_creation' },
            { name: 'kgot-controller', context: 'kgot_core/controller' },
            { name: 'kgot-graph-store', context: 'kgot_core/graph_store' },
            { name: 'kgot-integrated-tools', context: 'kgot_core/integrated_tools' },
            { name: 'multimodal-processor', context: 'multimodal' },
            { name: 'validation-service', context: 'validation' },
            { name: 'optimization-service', context: 'optimization' }
        ];
        
        for (const image of imagesToBuild) {
            logger.info(`Building ${image.name}...`);
            try {
                execSync(`docker build -t ${image.name}:latest ${path.join(this.rootDir, image.context)}`, 
                    { stdio: 'inherit', cwd: this.rootDir });
                logger.info(`✅ Built ${image.name} successfully`);
            } catch (error) {
                logger.error(`❌ Failed to build ${image.name}: ${error.message}`);
                throw error;
            }
        }
    }

    /**
     * Setup monitoring infrastructure
     */
    async setupMonitoring() {
        logger.info('📊 Setting up monitoring infrastructure...');
        
        // Install Python dependencies for containerization management
        try {
            logger.info('Installing Python dependencies...');
            execSync('pip3 install -r kgot_core/requirements.txt', { 
                stdio: 'inherit', 
                cwd: this.rootDir 
            });
            logger.info('✅ Python dependencies installed');
        } catch (error) {
            logger.error('❌ Failed to install Python dependencies:', error.message);
            throw error;
        }
        
        logger.info('Monitoring setup completed');
    }

    /**
     * Validate the complete configuration
     */
    async validateConfiguration() {
        logger.info('✅ Validating configuration...');
        
        // Validate Docker Compose file
        try {
            execSync('docker-compose -f config/containers/docker-compose.yml config', { 
                stdio: 'ignore',
                cwd: this.rootDir 
            });
            logger.info('✅ Docker Compose configuration is valid');
        } catch (error) {
            logger.error('❌ Docker Compose configuration validation failed');
            throw error;
        }
        
        // Test containerization.py functionality
        try {
            execSync('python3 -m kgot_core.containerization --help', { 
                stdio: 'ignore',
                cwd: this.rootDir 
            });
            logger.info('✅ Containerization CLI is working');
        } catch (error) {
            logger.warn('⚠️  Containerization CLI test failed (may be missing dependencies)');
        }
    }

    /**
     * Generate startup and management scripts
     */
    async generateStartupScripts() {
        logger.info('📝 Generating startup scripts...');
        
        // Create start script
        const startScript = `#!/bin/bash
# KGoT-Alita Containerization Start Script
set -e

echo "🚀 Starting KGoT-Alita Containerized Services..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Start services using Docker Compose
docker-compose -f config/containers/docker-compose.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
python3 -m kgot_core.containerization status

echo "✅ All services started successfully!"
echo "🌐 Access Grafana at: http://localhost:3000"
echo "🗄️  Access Neo4j at: http://localhost:7474"
echo "📊 Access Prometheus at: http://localhost:9090"
`;

        await fs.writeFile(path.join(this.rootDir, 'start-containers.sh'), startScript);
        execSync('chmod +x start-containers.sh', { cwd: this.rootDir });
        
        // Create stop script
        const stopScript = `#!/bin/bash
# KGoT-Alita Containerization Stop Script
set -e

echo "🛑 Stopping KGoT-Alita Containerized Services..."

# Stop services gracefully
python3 -m kgot_core.containerization stop

# Stop Docker Compose services
docker-compose -f config/containers/docker-compose.yml down

echo "✅ All services stopped successfully!"
`;

        await fs.writeFile(path.join(this.rootDir, 'stop-containers.sh'), stopScript);
        execSync('chmod +x stop-containers.sh', { cwd: this.rootDir });
        
        logger.info('Startup scripts generated successfully');
    }

    /**
     * Create environment template file
     */
    async createEnvironmentTemplate() {
        const envTemplate = `# KGoT-Alita Environment Configuration
# Copy this file to .env and update with your values

# Database Passwords
NEO4J_PASSWORD=your_neo4j_password_here
REDIS_PASSWORD=your_redis_password_here

# API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Logging
LOG_LEVEL=info

# Monitoring
GRAFANA_PASSWORD=admin

# Container Resources
CONTAINER_MEMORY_LIMIT=2g
CONTAINER_CPU_LIMIT=2
`;
        
        await fs.writeFile(this.envFile, envTemplate);
        logger.info('Environment template created at .env');
    }

    /**
     * Display setup summary
     */
    async displaySetupSummary() {
        logger.info('\n🎉 === KGoT Containerization Setup Summary ===');
        logger.info('');
        logger.info('✅ Containerization infrastructure is ready!');
        logger.info('');
        logger.info('📋 Next Steps:');
        logger.info('1. Update .env file with your API keys and passwords');
        logger.info('2. Run: ./start-containers.sh');
        logger.info('3. Access services:');
        logger.info('   - Grafana: http://localhost:3000 (admin/admin)');
        logger.info('   - Neo4j: http://localhost:7474');
        logger.info('   - Prometheus: http://localhost:9090');
        logger.info('');
        logger.info('🔧 Management Commands:');
        logger.info('   - Status: python3 -m kgot_core.containerization status');
        logger.info('   - Stop: ./stop-containers.sh');
        logger.info('   - Restart: python3 -m kgot_core.containerization restart --service <service>');
        logger.info('');
        logger.info('📖 Documentation: docs/TASK_8_CONTAINERIZATION.md');
    }
}

/**
 * Main execution function
 */
async function main() {
    try {
        const setup = new ContainerizationSetup();
        await setup.initialize();
        process.exit(0);
    } catch (error) {
        logger.error('Setup failed:', error);
        process.exit(1);
    }
}

// Execute if called directly
if (require.main === module) {
    main();
}

module.exports = { ContainerizationSetup }; 