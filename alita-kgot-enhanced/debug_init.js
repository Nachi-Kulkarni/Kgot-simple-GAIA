// Debug initialization script
require('dotenv').config();

console.log('=== Environment Check ===');
console.log('OPENROUTER_API_KEY:', process.env.OPENROUTER_API_KEY ? 'SET' : 'NOT SET');
console.log('PORT:', process.env.PORT);
console.log('NODE_ENV:', process.env.NODE_ENV);

console.log('\n=== Module Loading Test ===');
try {
  const { ChatOpenAI } = require('@langchain/openai');
  console.log('✓ @langchain/openai loaded successfully');
} catch (error) {
  console.log('✗ @langchain/openai failed:', error.message);
}

try {
  const modelConfig = require('./config/models/model_config.json');
  console.log('✓ model_config.json loaded successfully');
  console.log('Primary model:', modelConfig.alita_config.manager_agent.primary_model);
} catch (error) {
  console.log('✗ model_config.json failed:', error.message);
}

try {
  const { KGoTController } = require('./kgot_core/controller/kgot_controller');
  console.log('✓ KGoTController loaded successfully');
} catch (error) {
  console.log('✗ KGoTController failed:', error.message);
}

try {
  const { MCPCrossValidationCoordinator } = require('./kgot_core/controller/mcp_cross_validation');
  console.log('✓ MCPCrossValidationCoordinator loaded successfully');
} catch (error) {
  console.log('✗ MCPCrossValidationCoordinator failed:', error.message);
}

try {
  const { SequentialThinkingIntegration } = require('./alita_core/manager_agent/sequential_thinking_integration');
  console.log('✓ SequentialThinkingIntegration loaded successfully');
} catch (error) {
  console.log('✗ SequentialThinkingIntegration failed:', error.message);
}

console.log('\n=== Basic OpenRouter Test ===');
try {
  const { ChatOpenAI } = require('@langchain/openai');
  const modelConfig = require('./config/models/model_config.json');
  
  const llm = new ChatOpenAI({
    openAIApiKey: process.env.OPENROUTER_API_KEY,
    configuration: {
      baseURL: modelConfig.model_providers.openrouter.base_url,
    },
    modelName: modelConfig.model_providers.openrouter.models['o3'].model_id,
    temperature: 0.1,
    maxTokens: 10000,
    timeout: 10000,
  });
  
  console.log('✓ OpenRouter LLM instance created successfully');
  
  // Test a simple call
  llm.invoke([{ role: 'user', content: 'Hello' }])
    .then(response => {
      console.log('✓ OpenRouter API call successful');
      console.log('Response preview:', response.content.substring(0, 50) + '...');
    })
    .catch(error => {
      console.log('✗ OpenRouter API call failed:', error.message);
    });
    
} catch (error) {
  console.log('✗ OpenRouter LLM creation failed:', error.message);
}

console.log('\n=== Initialization Test Complete ===');