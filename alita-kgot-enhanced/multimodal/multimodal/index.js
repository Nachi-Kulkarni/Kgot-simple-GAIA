#!/usr/bin/env node

/**
 * Multimodal Processing Service Entry Point
 * Supports audio, vision, and text processing capabilities
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3006;

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, '..', 'storage', getFileCategory(file.mimetype));
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ 
    storage: storage,
    limits: {
        fileSize: 100 * 1024 * 1024 // 100MB limit
    }
});

// Middleware
console.log('🔧 Multimodal Service: Configuring middleware...');
app.use(cors());
console.log('✅ Multimodal Service: CORS middleware configured');
app.use(express.json({ limit: '50mb' }));
console.log('✅ Multimodal Service: JSON parser middleware configured (limit: 50MB)');
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
console.log('✅ Multimodal Service: URL-encoded parser middleware configured (limit: 50MB)');

// Logging setup
console.log('📁 Multimodal Service: Setting up logging directory...');
const logDir = path.join(__dirname, '..', 'logs', 'multimodal');
if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
    console.log(`✅ Multimodal Service: Created logging directory at ${logDir}`);
} else {
    console.log(`✅ Multimodal Service: Using existing logging directory at ${logDir}`);
}

// Helper function to categorize files
function getFileCategory(mimetype) {
    if (mimetype.startsWith('image/')) return 'images';
    if (mimetype.startsWith('audio/')) return 'audio';
    if (mimetype.startsWith('video/')) return 'video';
    if (mimetype.startsWith('text/')) return 'text';
    return 'other';
}

// Health check endpoint
app.get('/health', (req, res) => {
    console.log('🏥 Multimodal Service: Health check requested');
    res.status(200).json({
        status: 'healthy',
        service: 'multimodal-processor',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
    console.log('✅ Multimodal Service: Health check response sent');
});

// Status endpoint
app.get('/status', (req, res) => {
    console.log('📊 Multimodal Service: Status information requested');
    const statusInfo = {
        service: 'Multimodal Processing Service',
        version: '1.0.0',
        status: 'running',
        capabilities: [
            'image-analysis',
            'audio-processing',
            'video-processing',
            'text-processing',
            'cross-modal-validation',
            'visual-analysis',
            'screenshot-analysis'
        ],
        endpoints: {
            health: '/health',
            status: '/status',
            'process-image': '/api/process/image',
            'process-audio': '/api/process/audio',
            'process-video': '/api/process/video',
            'analyze-visual': '/api/analyze/visual',
            'analyze-screenshot': '/api/analyze/screenshot'
        }
    };
    res.json(statusInfo);
    console.log(`✅ Multimodal Service: Status response sent with ${statusInfo.capabilities.length} capabilities`);
});

// Image processing endpoint
app.post('/api/process/image', upload.single('image'), async (req, res) => {
    console.log('🖼️ Multimodal Service: Image processing request received');
    try {
        if (!req.file) {
            console.log('❌ Multimodal Service: No image file provided in request');
            return res.status(400).json({ error: 'No image file provided' });
        }
        
        const { analysis_type = 'basic' } = req.body;
        console.log(`🔍 Multimodal Service: Processing image with analysis type: ${analysis_type}`);
        console.log(`📁 Multimodal Service: Image file: ${req.file.originalname} (${req.file.size} bytes)`);
        
        const scriptPath = path.join(__dirname, '..', 'kgot_visual_analyzer.py');
        
        const result = await runPythonScript(scriptPath, {
            image_path: req.file.path,
            analysis_type,
            mode: 'image_processing'
        });
        
        console.log('✅ Multimodal Service: Image processing completed successfully');
        res.json({ success: true, result, file_info: req.file });
    } catch (error) {
        console.error('❌ Multimodal Service: Image processing error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Screenshot analysis endpoint
app.post('/api/analyze/screenshot', upload.single('screenshot'), async (req, res) => {
    console.log('📸 Multimodal Service: Screenshot analysis request received');
    try {
        if (!req.file) {
            console.log('❌ Multimodal Service: No screenshot file provided in request');
            return res.status(400).json({ error: 'No screenshot file provided' });
        }
        
        const { analysis_config = {} } = req.body;
        console.log(`🔍 Multimodal Service: Analyzing screenshot with config:`, analysis_config);
        console.log(`📁 Multimodal Service: Screenshot file: ${req.file.originalname} (${req.file.size} bytes)`);
        
        const scriptPath = path.join(__dirname, '..', 'kgot_alita_screenshot_analyzer.py');
        
        const result = await runPythonScript(scriptPath, {
            screenshot_path: req.file.path,
            config: analysis_config
        });
        
        console.log('✅ Multimodal Service: Screenshot analysis completed successfully');
        res.json({ success: true, result, file_info: req.file });
    } catch (error) {
        console.error('❌ Multimodal Service: Screenshot analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Visual analysis endpoint
app.post('/api/analyze/visual', upload.single('visual'), async (req, res) => {
    console.log('👁️ Multimodal Service: Visual analysis request received');
    try {
        if (!req.file) {
            console.log('❌ Multimodal Service: No visual file provided in request');
            return res.status(400).json({ error: 'No visual file provided' });
        }
        
        const { analysis_params = {} } = req.body;
        console.log(`🔍 Multimodal Service: Analyzing visual content with params:`, analysis_params);
        console.log(`📁 Multimodal Service: Visual file: ${req.file.originalname} (${req.file.size} bytes)`);
        
        const scriptPath = path.join(__dirname, '..', 'kgot_visual_analyzer.py');
        
        const result = await runPythonScript(scriptPath, {
            visual_path: req.file.path,
            params: analysis_params,
            mode: 'visual_analysis'
        });
        
        console.log('✅ Multimodal Service: Visual analysis completed successfully');
        res.json({ success: true, result, file_info: req.file });
    } catch (error) {
        console.error('❌ Multimodal Service: Visual analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Cross-modal validation endpoint
app.post('/api/validate/cross-modal', upload.array('files', 10), async (req, res) => {
    console.log('🔄 Multimodal Service: Cross-modal validation request received');
    try {
        if (!req.files || req.files.length === 0) {
            console.log('❌ Multimodal Service: No files provided for cross-modal validation');
            return res.status(400).json({ error: 'No files provided for cross-modal validation' });
        }
        
        const { validation_config = {} } = req.body;
        console.log(`🔍 Multimodal Service: Validating ${req.files.length} files with config:`, validation_config);
        req.files.forEach((file, index) => {
            console.log(`📁 Multimodal Service: File ${index + 1}: ${file.originalname} (${file.size} bytes, ${file.mimetype})`);
        });
        
        const scriptPath = path.join(__dirname, '..', 'kgot_alita_cross_modal_validator.py');
        
        const filePaths = req.files.map(file => file.path);
        const result = await runPythonScript(scriptPath, {
            file_paths: filePaths,
            config: validation_config
        });
        
        console.log(`✅ Multimodal Service: Cross-modal validation completed for ${req.files.length} files`);
        res.json({ success: true, result, files_processed: req.files.length });
    } catch (error) {
        console.error('❌ Multimodal Service: Cross-modal validation error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Audio processing endpoint
app.post('/api/process/audio', upload.single('audio'), async (req, res) => {
    console.log('🎵 Multimodal Service: Audio processing request received');
    try {
        if (!req.file) {
            console.log('❌ Multimodal Service: No audio file provided in request');
            return res.status(400).json({ error: 'No audio file provided' });
        }
        
        const { processing_type = 'basic' } = req.body;
        console.log(`🔍 Multimodal Service: Processing audio with type: ${processing_type}`);
        console.log(`📁 Multimodal Service: Audio file: ${req.file.originalname} (${req.file.size} bytes, ${req.file.mimetype})`);
        
        // For now, return basic file info - can be extended with actual audio processing
        const result = {
            message: 'Audio processing endpoint ready',
            file_info: req.file,
            processing_type
        };
        
        console.log('✅ Multimodal Service: Audio processing completed (basic mode)');
        res.json({ success: true, result });
    } catch (error) {
        console.error('❌ Multimodal Service: Audio processing error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Video processing endpoint
app.post('/api/process/video', upload.single('video'), async (req, res) => {
    console.log('🎬 Multimodal Service: Video processing request received');
    try {
        if (!req.file) {
            console.log('❌ Multimodal Service: No video file provided in request');
            return res.status(400).json({ error: 'No video file provided' });
        }
        
        const { processing_type = 'basic' } = req.body;
        console.log(`🔍 Multimodal Service: Processing video with type: ${processing_type}`);
        console.log(`📁 Multimodal Service: Video file: ${req.file.originalname} (${req.file.size} bytes, ${req.file.mimetype})`);
        
        // For now, return basic file info - can be extended with actual video processing
        const result = {
            message: 'Video processing endpoint ready',
            file_info: req.file,
            processing_type
        };
        
        console.log('✅ Multimodal Service: Video processing completed (basic mode)');
        res.json({ success: true, result });
    } catch (error) {
        console.error('❌ Multimodal Service: Video processing error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Helper function to run Python scripts
function runPythonScript(scriptPath, args = {}) {
    console.log(`🐍 Multimodal Service: Executing Python script: ${path.basename(scriptPath)}`);
    console.log(`📋 Multimodal Service: Script arguments:`, args);
    
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
                console.log(`✅ Multimodal Service: Python script completed successfully (exit code: ${code})`);
                try {
                    const result = JSON.parse(output);
                    resolve(result);
                } catch (e) {
                    console.log('⚠️ Multimodal Service: Script output is not valid JSON, returning raw output');
                    resolve({ output: output.trim() });
                }
            } else {
                console.error(`❌ Multimodal Service: Python script failed with exit code ${code}`);
                console.error(`❌ Multimodal Service: Script error output: ${error}`);
                reject(new Error(`Python script failed with code ${code}: ${error}`));
            }
        });
    });
}

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('❌ Multimodal Service: Unhandled error occurred:', error);
    console.error(`❌ Multimodal Service: Error stack: ${error.stack}`);
    res.status(500).json({
        error: 'Internal server error',
        message: error.message
    });
});

// Start server
console.log('🚀 Multimodal Service: Starting Multimodal Processing Service...');
app.listen(PORT, '0.0.0.0', () => {
    console.log('✅ Multimodal Service: Server started successfully!');
    console.log(`🌐 Multimodal Service: Running on port ${PORT}`);
    console.log(`🏥 Multimodal Service: Health check: http://localhost:${PORT}/health`);
    console.log(`📊 Multimodal Service: Status: http://localhost:${PORT}/status`);
    console.log('🎯 Multimodal Service: Available endpoints:');
    console.log('   - POST /api/process/image - Image processing');
    console.log('   - POST /api/process/audio - Audio processing');
    console.log('   - POST /api/process/video - Video processing');
    console.log('   - POST /api/analyze/visual - Visual analysis');
    console.log('   - POST /api/analyze/screenshot - Screenshot analysis');
    console.log('   - POST /api/validate/cross-modal - Cross-modal validation');
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('🛑 Multimodal Service: Received SIGTERM signal');
    console.log('🔄 Multimodal Service: Shutting down gracefully...');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('🛑 Multimodal Service: Received SIGINT signal (Ctrl+C)');
    console.log('🔄 Multimodal Service: Shutting down gracefully...');
    process.exit(0);
});