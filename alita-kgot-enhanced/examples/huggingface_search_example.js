#!/usr/bin/env node
/**
 * Hugging Face Search Example
 * 
 * Demonstrates how to use the new HuggingFaceSearchTool in the Alita Web Agent
 * instead of the previous Google search implementation.
 * 
 * This example shows:
 * - Basic search with Hugging Face Agents framework
 * - Integration with KGoT Surfer Agent capabilities
 * - Different search types (informational, navigational, research)
 * - Wikipedia integration and granular navigation
 * 
 * @example
 * node examples/huggingface_search_example.js
 */

const path = require('path');
const { HuggingFaceSearchTool } = require('../alita_core/web_agent/index.js');

/**
 * Example configuration for Hugging Face search
 */
const searchConfig = {
  model_name: 'webagent',
  temperature: 0.1,
  kgot_path: path.join(__dirname, '../knowledge-graph-of-thoughts'),
  max_iterations: 12
};

/**
 * Demonstrate basic informational search
 */
async function demonstrateInformationalSearch() {
  console.log('\n🔍 Demonstrating Informational Search with Hugging Face Agents...\n');
  
  const searchTool = new HuggingFaceSearchTool(null, searchConfig);
  
  const searchQuery = {
    query: "What are the latest developments in transformer neural networks for 2024?",
    searchType: "informational",
    includeWikipedia: true,
    detailed: true
  };
  
  try {
    const result = await searchTool.searchWithHuggingFace(JSON.stringify(searchQuery));
    const parsedResult = JSON.parse(result);
    
    console.log('Search Query:', searchQuery.query);
    console.log('Search Type:', searchQuery.searchType);
    console.log('Framework:', parsedResult.metadata.framework);
    console.log('\nResults Summary:');
    console.log(parsedResult.results.summary);
    console.log('\nDetailed Results:');
    console.log(parsedResult.results.detailed_results.substring(0, 500) + '...');
    
  } catch (error) {
    console.error('Search failed:', error.message);
  }
}

/**
 * Demonstrate navigational search
 */
async function demonstrateNavigationalSearch() {
  console.log('\n🧭 Demonstrating Navigational Search with KGoT Integration...\n');
  
  const searchTool = new HuggingFaceSearchTool(null, searchConfig);
  
  const searchQuery = {
    query: "OpenAI GPT-4 official documentation and API reference",
    searchType: "navigational",
    includeWikipedia: false,
    detailed: false
  };
  
  try {
    const result = await searchTool.searchWithHuggingFace(JSON.stringify(searchQuery));
    const parsedResult = JSON.parse(result);
    
    console.log('Navigation Query:', searchQuery.query);
    console.log('Search Type:', searchQuery.searchType);
    console.log('Framework:', parsedResult.metadata.framework);
    console.log('\nNavigation Results:');
    console.log(parsedResult.results.summary);
    
  } catch (error) {
    console.error('Navigation search failed:', error.message);
  }
}

/**
 * Demonstrate research search with comprehensive analysis
 */
async function demonstrateResearchSearch() {
  console.log('\n📚 Demonstrating Research Search with Comprehensive Analysis...\n');
  
  const searchTool = new HuggingFaceSearchTool(null, searchConfig);
  
  const searchQuery = {
    query: "Compare the effectiveness of RAG vs fine-tuning for domain-specific AI applications",
    searchType: "research",
    includeWikipedia: true,
    detailed: true
  };
  
  try {
    const result = await searchTool.searchWithHuggingFace(JSON.stringify(searchQuery));
    const parsedResult = JSON.parse(result);
    
    console.log('Research Query:', searchQuery.query);
    console.log('Search Type:', searchQuery.searchType);
    console.log('Framework:', parsedResult.metadata.framework);
    console.log('Status:', parsedResult.results.status);
    console.log('\nResearch Summary:');
    console.log(parsedResult.results.summary);
    console.log('\nAdditional Context:');
    console.log(parsedResult.results.additional_context);
    
  } catch (error) {
    console.error('Research search failed:', error.message);
  }
}

/**
 * Main demonstration function
 */
async function main() {
  console.log('='.repeat(80));
  console.log('🤖 Alita Web Agent - Hugging Face Search Integration Demo');
  console.log('='.repeat(80));
  console.log('\nThis demo showcases the new Hugging Face Agents-based search');
  console.log('functionality that replaces the previous Google search implementation.');
  console.log('\nFeatures demonstrated:');
  console.log('• Hugging Face Transformers Agents framework');
  console.log('• KGoT Surfer Agent integration');
  console.log('• Intelligent browsing with context awareness');
  console.log('• Wikipedia knowledge integration');
  console.log('• Granular navigation (PageUp, PageDown, Find)');
  console.log('• Multi-step reasoning and content analysis');
  
  // Check if KGoT path exists
  const fs = require('fs');
  if (!fs.existsSync(searchConfig.kgot_path)) {
    console.error('\n❌ Error: KGoT path not found:', searchConfig.kgot_path);
    console.log('Please ensure the knowledge-graph-of-thoughts directory exists.');
    return;
  }
  
  console.log('\n✅ KGoT integration path verified:', searchConfig.kgot_path);
  
  try {
    // Run different types of searches
    await demonstrateInformationalSearch();
    await demonstrateNavigationalSearch();
    await demonstrateResearchSearch();
    
    console.log('\n' + '='.repeat(80));
    console.log('🎉 Demo completed successfully!');
    console.log('='.repeat(80));
    console.log('\nNext steps:');
    console.log('• Configure environment variables for KGoT models');
    console.log('• Integrate with your specific LLM provider (OpenRouter recommended)');
    console.log('• Customize search parameters for your use case');
    console.log('• Explore granular navigation features');
    
  } catch (error) {
    console.error('\n❌ Demo failed:', error.message);
    console.log('\nTroubleshooting tips:');
    console.log('• Ensure Python 3 is installed and accessible');
    console.log('• Verify KGoT dependencies are installed');
    console.log('• Check environment variables for LLM configuration');
    console.log('• Ensure network connectivity for web searches');
  }
}

// Run the demo if this file is executed directly
if (require.main === module) {
  main().catch(console.error);
}

module.exports = {
  demonstrateInformationalSearch,
  demonstrateNavigationalSearch,
  demonstrateResearchSearch,
  searchConfig
}; 