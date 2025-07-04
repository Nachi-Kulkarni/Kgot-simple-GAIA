/**
 * Initial Setup Script for Alita-KGoT Enhanced System
 * 
 * This script helps users set up the system for the first time by:
 * - Creating necessary directories
 * - Setting up default configurations
 * - Validating environment variables
 * - Initializing database schemas
 * - Setting up monitoring dashboards
 * 
 * @module InitialSetup
 */

const fs = require('fs').promises;
const path = require('path');
const { loggers } = require('../../config/logging/winston_config');

const logger = loggers.system;

/**
 * Main setup orchestrator
 */
class AlitaKGoTSetup {
  constructor() {
    this.rootDir = path.join(__dirname, '../..');
    this.requiredDirectories = [
      'logs/alita',
      'logs/kgot', 
      'logs/system',
      'logs/errors',
      'logs/multimodal',
      'logs/validation',
      'logs/optimization',
      'uploads',
      'temp',
      'data/neo4j',
      'data/rdf4j',
      'data/redis'
    ];
  }

  /**
   * Run the complete setup process
   */
  async setup() {
    try {
      logger.logOperation('info', 'SETUP_START', 'Starting Alita-KGoT Enhanced System setup');
      
      console.log('🚀 Alita-KGoT Enhanced System Setup');
      console.log('====================================');
      
      await this.createDirectories();
      await this.validateEnvironment();
      await this.setupDefaultConfigs();
      await this.displayNextSteps();
      
      logger.logOperation('info', 'SETUP_COMPLETE', 'Setup completed successfully');
      console.log('✅ Setup completed successfully!');
      
    } catch (error) {
      logger.logError('SETUP_ERROR', error);
      console.error('❌ Setup failed:', error.message);
      process.exit(1);
    }
  }

  /**
   * Create all required directories
   */
  async createDirectories() {
    console.log('\n📁 Creating required directories...');
    
    for (const dir of this.requiredDirectories) {
      const fullPath = path.join(this.rootDir, dir);
      try {
        await fs.mkdir(fullPath, { recursive: true });
        console.log(`  ✓ Created: ${dir}`);
      } catch (error) {
        if (error.code !== 'EEXIST') {
          throw new Error(`Failed to create directory ${dir}: ${error.message}`);
        }
        console.log(`  ✓ Exists: ${dir}`);
      }
    }
  }

  /**
   * Validate environment configuration
   */
  async validateEnvironment() {
    console.log('\n🔍 Validating environment configuration...');
    
    const envPath = path.join(this.rootDir, '.env');
    const envTemplatePath = path.join(this.rootDir, 'env.template');
    
    try {
      await fs.access(envPath);
      console.log('  ✓ .env file exists');
    } catch (error) {
      console.log('  ⚠️  .env file not found, creating from template...');
      try {
        const template = await fs.readFile(envTemplatePath, 'utf8');
        await fs.writeFile(envPath, template);
        console.log('  ✓ Created .env from template');
        console.log('  ⚠️  Please edit .env file with your actual values');
      } catch (templateError) {
        throw new Error('Failed to create .env file from template');
      }
    }

    // Load and validate environment variables
    require('dotenv').config({ path: envPath });
    
    const requiredVars = [
      'OPENROUTER_API_KEY',
      'NEO4J_PASSWORD', 
      'REDIS_PASSWORD'
    ];
    
    const missingVars = [];
    for (const varName of requiredVars) {
      if (!process.env[varName] || process.env[varName].includes('your_')) {
        missingVars.push(varName);
      }
    }
    
    if (missingVars.length > 0) {
      console.log(`  ⚠️  Please configure these environment variables in .env:`);
      missingVars.forEach(varName => console.log(`     - ${varName}`));
    } else {
      console.log('  ✓ All required environment variables are set');
    }
  }

  /**
   * Setup default configurations
   */
  async setupDefaultConfigs() {
    console.log('\n⚙️  Setting up default configurations...');
    
    // Create default monitoring configuration
    const monitoringDir = path.join(this.rootDir, 'config/monitoring');
    await fs.mkdir(monitoringDir, { recursive: true });
    
    // Create Prometheus configuration
    const prometheusConfig = `
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'alita-manager'
    static_configs:
      - targets: ['alita-manager:3000']
  
  - job_name: 'alita-web'
    static_configs:
      - targets: ['alita-web:3001']
      
  - job_name: 'kgot-controller'
    static_configs:
      - targets: ['kgot-controller:3003']
`;

    await fs.writeFile(
      path.join(monitoringDir, 'prometheus.yml'), 
      prometheusConfig.trim()
    );
    console.log('  ✓ Created Prometheus configuration');

    // Create Grafana provisioning directory
    const grafanaDir = path.join(monitoringDir, 'grafana');
    await fs.mkdir(grafanaDir, { recursive: true });
    console.log('  ✓ Created Grafana configuration directory');
  }

  /**
   * Display next steps for the user
   */
  async displayNextSteps() {
    console.log('\n🎯 Next Steps:');
    console.log('==============');
    console.log('1. Edit .env file with your API keys and passwords');
    console.log('2. Install dependencies: npm install');
    console.log('3. Start with Docker: npm run docker:up');
    console.log('4. Or start individual services:');
    console.log('   - Manager Agent: npm run start:manager');
    console.log('   - Web Agent: npm run start:web');
    console.log('   - KGoT Controller: npm run start:kgot');
    console.log('');
    console.log('🔗 Access Points:');
    console.log('   - Manager Agent: http://localhost:3000');
    console.log('   - Neo4j Browser: http://localhost:7474');
    console.log('   - Grafana: http://localhost:3000 (admin/your_grafana_password)');
    console.log('   - Prometheus: http://localhost:9090');
    console.log('');
    console.log('📚 Documentation: README.md');
    console.log('🔧 Troubleshooting: Check logs/ directory for detailed logs');
  }
}

// Run setup if this file is executed directly
if (require.main === module) {
  const setup = new AlitaKGoTSetup();
  setup.setup().catch(error => {
    console.error('Setup failed:', error);
    process.exit(1);
  });
}

module.exports = AlitaKGoTSetup; 