// Simple test to check basic functionality
console.log('=== Starting Simple Test ===');

try {
  console.log('1. Testing basic Node.js functionality...');
  console.log('   Node.js version:', process.version);
  console.log('   Current working directory:', process.cwd());
  
  console.log('2. Testing environment variables...');
  console.log('   OPENROUTER_API_KEY:', process.env.OPENROUTER_API_KEY ? 'SET' : 'NOT SET');
  console.log('   PORT:', process.env.PORT);
  
  console.log('3. Testing dotenv loading...');
  require('dotenv').config();
  console.log('   dotenv loaded successfully');
  
  console.log('4. Testing basic imports...');
  const express = require('express');
  console.log('   express imported successfully');
  
  const fs = require('fs');
  console.log('   fs imported successfully');
  
  console.log('5. Testing file system access...');
  const configExists = fs.existsSync('./config/models/model_config.json');
  console.log('   model_config.json exists:', configExists);
  
  if (configExists) {
    const modelConfig = require('./config/models/model_config.json');
    console.log('   model_config.json loaded successfully');
    console.log('   Primary model:', modelConfig.alita_config?.manager_agent?.primary_model);
  }
  
  console.log('6. Testing LangChain imports...');
  const { ChatOpenAI } = require('@langchain/openai');
  console.log('   @langchain/openai imported successfully');
  
  console.log('7. Creating basic Express app...');
  const app = express();
  app.use(express.json());
  
  app.get('/test', (req, res) => {
    res.json({ status: 'working', timestamp: new Date().toISOString() });
  });
  
  const server = app.listen(8001, () => {
    console.log('   Test server started on port 8001');
    
    // Test the endpoint
    setTimeout(() => {
      const http = require('http');
      const options = {
        hostname: 'localhost',
        port: 8001,
        path: '/test',
        method: 'GET'
      };
      
      const req = http.request(options, (res) => {
        let data = '';
        res.on('data', (chunk) => {
          data += chunk;
        });
        res.on('end', () => {
          console.log('   Test endpoint response:', data);
          server.close(() => {
            console.log('   Test server closed');
            console.log('\n=== Simple Test Completed Successfully ===');
            process.exit(0);
          });
        });
      });
      
      req.on('error', (err) => {
        console.log('   Test endpoint error:', err.message);
        server.close(() => {
          console.log('   Test server closed due to error');
          process.exit(1);
        });
      });
      
      req.end();
    }, 1000);
  });
  
} catch (error) {
  console.error('=== Test Failed ===');
  console.error('Error:', error.message);
  console.error('Stack:', error.stack);
  process.exit(1);
}