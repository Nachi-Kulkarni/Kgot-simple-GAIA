#!/usr/bin/env node

/**
 * Validation Service Entry Point
 * Provides testing, benchmarking, and quality assurance capabilities
 */

const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3007;

// Middleware
console.log('🔧 Validation Service: Configuring middleware...');
app.use(cors());
console.log('✅ Validation Service: CORS middleware configured');
app.use(express.json({ limit: '50mb' }));
console.log('✅ Validation Service: JSON parser middleware configured (limit: 50MB)');
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
console.log('✅ Validation Service: URL-encoded parser middleware configured (limit: 50MB)');

// Logging setup
console.log('📁 Validation Service: Setting up logging directory...');
const logDir = path.join(__dirname, 'logs', 'validation');
if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
    console.log(`✅ Validation Service: Created logging directory at ${logDir}`);
} else {
    console.log(`✅ Validation Service: Using existing logging directory at ${logDir}`);
}

// Health check endpoint
app.get('/health', (req, res) => {
    console.log('🏥 Validation Service: Health check requested');
    res.status(200).json({
        status: 'healthy',
        service: 'validation-service',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
    console.log('✅ Validation Service: Health check response sent');
});

// Status endpoint
app.get('/status', (req, res) => {
    console.log('📊 Validation Service: Status information requested');
    const statusInfo = {
        service: 'Validation Service',
        version: '1.0.0',
        status: 'running',
        capabilities: [
            'performance-validation',
            'cross-validation',
            'rag-coordination',
            'quality-assurance',
            'benchmarking'
        ],
        endpoints: {
            health: '/health',
            status: '/status',
            validate: '/api/validate',
            benchmark: '/api/benchmark',
            qa: '/api/qa'
        }
    };
    res.json(statusInfo);
    console.log(`✅ Validation Service: Status response sent with ${statusInfo.capabilities.length} capabilities`);
});

// Validation endpoint
app.post('/api/validate', async (req, res) => {
    console.log('🔍 Validation Service: Validation request received');
    try {
        const { type, data, config } = req.body;
        console.log(`📋 Validation Service: Validation type: ${type}`);
        console.log(`📊 Validation Service: Data size: ${JSON.stringify(data).length} characters`);
        console.log(`⚙️ Validation Service: Config:`, config);
        
        // Run Python validation script based on type
        let scriptPath;
        switch (type) {
            case 'performance':
                scriptPath = path.join(__dirname, 'kgot_alita_performance_validator.py');
                console.log('🚀 Validation Service: Using performance validator');
                break;
            case 'cross':
                scriptPath = path.join(__dirname, 'mcp_cross_validator.py');
                console.log('🔄 Validation Service: Using cross validator');
                break;
            case 'rag':
                scriptPath = path.join(__dirname, 'rag_mcp_coordinator.py');
                console.log('🧠 Validation Service: Using RAG coordinator');
                break;
            default:
                console.log(`❌ Validation Service: Invalid validation type: ${type}`);
                return res.status(400).json({ error: 'Invalid validation type' });
        }
        
        const result = await runPythonScript(scriptPath, { data, config });
        console.log('✅ Validation Service: Validation completed successfully');
        res.json({ success: true, result });
    } catch (error) {
        console.error('❌ Validation Service: Validation error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Benchmark endpoint
app.post('/api/benchmark', async (req, res) => {
    console.log('📈 Validation Service: Benchmark request received');
    try {
        const { config } = req.body;
        console.log('⚙️ Validation Service: Benchmark config:', config);
        
        const scriptPath = path.join(__dirname, 'kgot_alita_performance_validator.py');
        console.log('🚀 Validation Service: Running performance benchmark');
        
        const result = await runPythonScript(scriptPath, { mode: 'benchmark', config });
        console.log('✅ Validation Service: Benchmark completed successfully');
        res.json({ success: true, result });
    } catch (error) {
        console.error('❌ Validation Service: Benchmark error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Quality Assurance endpoint
app.post('/api/qa', async (req, res) => {
    console.log('🔬 Validation Service: Quality Assurance request received');
    try {
        const { data, tests } = req.body;
        console.log(`📊 Validation Service: QA data size: ${JSON.stringify(data).length} characters`);
        console.log(`🧪 Validation Service: Number of tests: ${Array.isArray(tests) ? tests.length : 'N/A'}`);
        
        const scriptPath = path.join(__dirname, 'unified_pipeline_validator.py');
        console.log('🔍 Validation Service: Running unified pipeline validation');
        
        const result = await runPythonScript(scriptPath, { data, tests });
        console.log('✅ Validation Service: Quality Assurance completed successfully');
        res.json({ success: true, result });
    } catch (error) {
        console.error('❌ Validation Service: QA error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Helper function to run Python scripts
function runPythonScript(scriptPath, args = {}) {
    console.log(`🐍 Validation Service: Executing Python script: ${path.basename(scriptPath)}`);
    console.log(`📋 Validation Service: Script arguments:`, args);
    
    return new Promise((resolve, reject) => {
        const python = spawn('python3', [scriptPath, JSON.stringify(args)]);
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
                console.log(`✅ Validation Service: Python script completed successfully (exit code: ${code})`);
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (e) {
                    console.log('⚠️ Validation Service: Script output is not valid JSON, returning raw output');
                    resolve({ output: output.trim() });
                }
            } else {
                console.error(`❌ Validation Service: Python script failed with exit code ${code}`);
                console.error(`❌ Validation Service: Script error output: ${error}`);
                reject(new Error(`Python script failed with code ${code}: ${error}`));
            }
        });
    });
}

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('❌ Validation Service: Unhandled error occurred:', error);
    console.error(`❌ Validation Service: Error stack: ${error.stack}`);
    res.status(500).json({
        error: 'Internal server error',
        message: error.message
    });
});

// Start server
console.log('🚀 Validation Service: Starting Validation Service...');
app.listen(PORT, '0.0.0.0', () => {
    console.log('✅ Validation Service: Server started successfully!');
    console.log(`🌐 Validation Service: Running on port ${PORT}`);
    console.log(`🏥 Validation Service: Health check: http://localhost:${PORT}/health`);
    console.log(`📊 Validation Service: Status: http://localhost:${PORT}/status`);
    console.log('🎯 Validation Service: Available endpoints:');
    console.log('   - POST /api/validate - Validation operations');
    console.log('   - POST /api/benchmark - Performance benchmarking');
    console.log('   - POST /api/qa - Quality assurance testing');
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('🛑 Validation Service: Received SIGTERM signal');
    console.log('🔄 Validation Service: Shutting down gracefully...');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('🛑 Validation Service: Received SIGINT signal (Ctrl+C)');
    console.log('🔄 Validation Service: Shutting down gracefully...');
    process.exit(0);
});