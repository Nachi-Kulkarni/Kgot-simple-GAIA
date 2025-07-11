#!/usr/bin/env node

// Test environment variable loading
require('dotenv').config();

console.log('=== Environment Variable Test ===');
console.log('OPENROUTER_API_KEY:', process.env.OPENROUTER_API_KEY ? 'SET (length: ' + process.env.OPENROUTER_API_KEY.length + ')' : 'NOT SET');
console.log('NODE_ENV:', process.env.NODE_ENV);
console.log('PORT:', process.env.PORT);
console.log('LOG_LEVEL:', process.env.LOG_LEVEL);

// Test from the manager agent directory
process.chdir('./alita_core/manager_agent');
console.log('\n=== Testing from manager_agent directory ===');
console.log('Current working directory:', process.cwd());
console.log('OPENROUTER_API_KEY:', process.env.OPENROUTER_API_KEY ? 'SET (length: ' + process.env.OPENROUTER_API_KEY.length + ')' : 'NOT SET');

// Test from the kgot_core directory
process.chdir('../../kgot_core/controller');
console.log('\n=== Testing from kgot_core/controller directory ===');
console.log('Current working directory:', process.cwd());
console.log('OPENROUTER_API_KEY:', process.env.OPENROUTER_API_KEY ? 'SET (length: ' + process.env.OPENROUTER_API_KEY.length + ')' : 'NOT SET');