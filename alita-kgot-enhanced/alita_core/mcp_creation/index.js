/**
 * Alita MCP Creation Service
 * 
 * Node.js service for creating and managing Model Context Protocols (MCPs)
 * Integrates with Python-based KGoT Advanced MCP Generation
 * 
 * @module AlitaMCPCreation
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3002;

// Middleware
console.log('🔧 MCP Creation Service: Configuring middleware...');
app.use(helmet());
console.log('🛡️ MCP Creation Service: Helmet security middleware enabled');
app.use(cors());
console.log('🌐 MCP Creation Service: CORS middleware enabled');
app.use(express.json({ limit: '10mb' }));
console.log('📝 MCP Creation Service: JSON body parser enabled (limit: 10mb)');
app.use(express.urlencoded({ extended: true }));
console.log('📋 MCP Creation Service: URL-encoded body parser enabled');

// Rate limiting
console.log('⏱️ MCP Creation Service: Configuring rate limiting...');
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use(limiter);
console.log('🚦 MCP Creation Service: Rate limiting enabled (100 requests per 15 minutes)');

// Health check endpoint
app.get('/health', (req, res) => {
  console.log('🏥 MCP Creation Service: Health check requested');
  res.status(200).json({
    status: 'healthy',
    service: 'alita-mcp-creation',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
  console.log('✅ MCP Creation Service: Health check response sent');
});

// Status endpoint
app.get('/status', (req, res) => {
  console.log('📊 MCP Creation Service: Status check requested');
  res.status(200).json({
    service: 'Alita MCP Creation Service',
    status: 'running',
    port: PORT,
    endpoints: {
      health: '/health',
      status: '/status',
      createMCP: '/api/mcp/create',
      listMCPs: '/api/mcp/list'
    },
    timestamp: new Date().toISOString()
  });
  console.log('✅ MCP Creation Service: Status response sent');
});

// Create MCP endpoint
app.post('/api/mcp/create', async (req, res) => {
  console.log('🔧 MCP Creation Service: MCP creation request received');
  try {
    const { name, description, requirements } = req.body;
    console.log(`📝 MCP Creation Service: Creating MCP with name: ${name}`);
    console.log(`📋 MCP Creation Service: Description: ${description}`);
    console.log(`⚙️ MCP Creation Service: Requirements:`, requirements);
    
    if (!name || !description) {
      console.log('❌ MCP Creation Service: Missing required fields');
      return res.status(400).json({
        error: 'Missing required fields: name and description'
      });
    }

    // Call Python MCP generation script
    const result = await createMCP(name, description, requirements);
    console.log('✅ MCP Creation Service: MCP created successfully');
    
    res.status(200).json({
      success: true,
      message: 'MCP created successfully',
      data: result
    });
  } catch (error) {
    console.error('❌ MCP Creation Service: Error creating MCP:', error);
    res.status(500).json({
      error: 'Failed to create MCP',
      message: error.message
    });
  }
});

// List MCPs endpoint
app.get('/api/mcp/list', async (req, res) => {
  console.log('📋 MCP Creation Service: MCP list request received');
  try {
    const mcps = await listMCPs();
    console.log(`✅ MCP Creation Service: Found ${mcps.length} MCPs`);
    res.status(200).json({
      success: true,
      data: mcps
    });
  } catch (error) {
    console.error('❌ MCP Creation Service: Error listing MCPs:', error);
    res.status(500).json({
      error: 'Failed to list MCPs',
      message: error.message
    });
  }
});

/**
 * Create MCP using Python script
 * @param {string} name - MCP name
 * @param {string} description - MCP description
 * @param {object} requirements - MCP requirements
 * @returns {Promise<object>} Creation result
 */
function createMCP(name, description, requirements = {}) {
  console.log(`🐍 MCP Creation Service: Executing Python MCP generation script`);
  console.log(`📝 MCP Creation Service: Script parameters - Name: ${name}, Description: ${description}`);
  
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, 'kgot_advanced_mcp_generation.py');
    const args = [
      pythonScript,
      '--name', name,
      '--description', description,
      '--requirements', JSON.stringify(requirements)
    ];
    console.log(`🔧 MCP Creation Service: Python script path: ${pythonScript}`);

    const python = spawn('python3', args);
    let output = '';
    let error = '';

    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.stderr.on('data', (data) => {
      error += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        console.log(`✅ MCP Creation Service: Python script completed successfully (exit code: ${code})`);
        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (parseError) {
          console.log('⚠️ MCP Creation Service: Script output is not valid JSON, returning raw output');
          resolve({ output, raw: true });
        }
      } else {
        console.error(`❌ MCP Creation Service: Python script failed with exit code ${code}`);
        console.error(`❌ MCP Creation Service: Script error output: ${error}`);
        reject(new Error(`Python script failed with code ${code}: ${error}`));
      }
    });
  });
}

/**
 * List available MCPs
 * @returns {Promise<Array>} List of MCPs
 */
async function listMCPs() {
  console.log('📂 MCP Creation Service: Listing available MCPs...');
  try {
    const mcpDir = path.join(__dirname, 'generated_mcps');
    console.log(`📁 MCP Creation Service: Scanning directory: ${mcpDir}`);
    const files = await fs.readdir(mcpDir);
    const mcpFiles = files.filter(file => file.endsWith('.json'));
    console.log(`📋 MCP Creation Service: Found ${mcpFiles.length} MCP files`);
    return mcpFiles.map(file => ({
      name: file.replace('.json', ''),
      path: path.join(mcpDir, file)
    }));
  } catch (error) {
    console.error('❌ MCP Creation Service: Error reading MCP directory:', error);
    return [];
  }
}

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('❌ MCP Creation Service: Unhandled error occurred:', error);
  console.error(`❌ MCP Creation Service: Error stack: ${error.stack}`);
  res.status(500).json({
    error: 'Internal server error',
    message: error.message
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: `Route ${req.method} ${req.path} not found`
  });
});

// Start server
console.log('🚀 MCP Creation Service: Starting Alita MCP Creation Service...');
app.listen(PORT, () => {
  console.log('✅ MCP Creation Service: Server started successfully!');
  console.log(`🌐 MCP Creation Service: Running on port ${PORT}`);
  console.log(`🏥 MCP Creation Service: Health check: http://localhost:${PORT}/health`);
  console.log(`📊 MCP Creation Service: Status: http://localhost:${PORT}/status`);
  console.log('🎯 MCP Creation Service: Available endpoints:');
  console.log('   - POST /api/mcp/create - Create new MCP');
  console.log('   - GET /api/mcp/list - List available MCPs');
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('🛑 MCP Creation Service: Received SIGTERM signal');
  console.log('🔄 MCP Creation Service: Shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('🛑 MCP Creation Service: Received SIGINT signal (Ctrl+C)');
  console.log('🔄 MCP Creation Service: Shutting down gracefully...');
  process.exit(0);
});