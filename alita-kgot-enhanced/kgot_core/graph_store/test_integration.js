/**
 * KGoT Graph Store Module - Integration Test
 * 
 * Simple test to verify the implementation works correctly
 * Tests basic functionality of the Knowledge Graph Store
 * 
 * This test demonstrates:
 * - Factory creation and backend selection
 * - Knowledge graph initialization
 * - Triplet addition and querying
 * - Entity management
 * - Snapshot functionality
 * - MCP metrics integration
 * 
 * @module GraphStoreIntegrationTest
 */

const path = require('path');

// Import the Graph Store Module
const {
  createKnowledgeGraph,
  createDevelopmentGraph,
  getAvailableBackends,
  getRecommendedBackend,
  NetworkXKnowledgeGraph,
  logger
} = require('./index');

/**
 * Run integration tests for the Graph Store Module
 */
async function runIntegrationTests() {
  logger.info('Starting KGoT Graph Store Integration Tests', {
    operation: 'INTEGRATION_TEST_START',
    timestamp: new Date().toISOString()
  });

  try {
    // Test 1: Backend availability and recommendations
    console.log('\n=== Test 1: Backend Availability ===');
    const backends = getAvailableBackends();
    console.log('Available backends:', JSON.stringify(backends, null, 2));
    
    const devBackend = getRecommendedBackend('development');
    const prodBackend = getRecommendedBackend('production');
    console.log(`Recommended for development: ${devBackend}`);
    console.log(`Recommended for production: ${prodBackend}`);

    // Test 2: Create development knowledge graph
    console.log('\n=== Test 2: Create Development Graph ===');
    const devGraph = await createDevelopmentGraph({
      logLevel: 'debug',
      developmentMode: true
    });
    
    console.log(`Created graph with backend: ${devGraph.backend}`);
    console.log(`Graph initialized: ${devGraph.isInitialized}`);

    // Test 3: Initialize the database
    console.log('\n=== Test 3: Initialize Database ===');
    await devGraph.initDatabase(1, 'test_run');
    console.log('Database initialized successfully');

    // Test 4: Add entities
    console.log('\n=== Test 4: Add Entities ===');
    
    // Add some test entities
    await devGraph.addEntity({
      id: 'user_001',
      type: 'User',
      properties: {
        name: 'Alice',
        role: 'researcher',
        department: 'AI Lab'
      }
    });

    await devGraph.addEntity({
      id: 'project_001',
      type: 'Project',
      properties: {
        name: 'Knowledge Graph Research',
        status: 'active',
        budget: 100000
      }
    });

    await devGraph.addEntity({
      id: 'task_001',
      type: 'Task',
      properties: {
        title: 'Implement KGoT Graph Store',
        priority: 'high',
        deadline: '2024-12-31'
      }
    });

    console.log('Added 3 test entities');

    // Test 5: Add triplets (relationships)
    console.log('\n=== Test 5: Add Triplets ===');
    
    await devGraph.addTriplet({
      subject: 'user_001',
      predicate: 'worksOn',
      object: 'project_001',
      metadata: {
        role: 'lead researcher',
        startDate: '2024-01-01'
      }
    });

    await devGraph.addTriplet({
      subject: 'project_001',
      predicate: 'contains',
      object: 'task_001',
      metadata: {
        priority: 'critical',
        assignedDate: '2024-01-15'
      }
    });

    await devGraph.addTriplet({
      subject: 'user_001',
      predicate: 'assignedTo',
      object: 'task_001',
      metadata: {
        assignmentDate: '2024-01-15',
        estimatedHours: 40
      }
    });

    console.log('Added 3 test triplets');

    // Test 6: Query entities
    console.log('\n=== Test 6: Query Entities ===');
    
    const allUsers = await devGraph.queryEntities({ type: 'User' });
    console.log(`Found ${allUsers.result.length} users:`, allUsers.result);

    const allProjects = await devGraph.queryEntities({ type: 'Project' });
    console.log(`Found ${allProjects.result.length} projects:`, allProjects.result);

    // Test 7: Query relationships
    console.log('\n=== Test 7: Query Relationships ===');
    
    const userRelationships = await devGraph.queryRelationships({ subject: 'user_001' });
    console.log(`User relationships:`, userRelationships.result);

    const projectTasks = await devGraph.queryRelationships({ predicate: 'contains' });
    console.log(`Project contains relationships:`, projectTasks.result);

    // Test 8: Execute custom queries
    console.log('\n=== Test 8: Execute Custom Queries ===');
    
    // Find all entities connected to user_001
    const customQuery = `
      const connectedEntities = [];
      const userId = 'user_001';
      
      // Find direct connections
      for (const [key, edge] of edges) {
        if (edge.source === userId || edge.target === userId) {
          const connectedId = edge.source === userId ? edge.target : edge.source;
          const connectedEntity = getNode(connectedId);
          if (connectedEntity) {
            connectedEntities.push({
              id: connectedId,
              entity: connectedEntity,
              relationship: edge.relationship,
              direction: edge.source === userId ? 'outgoing' : 'incoming'
            });
          }
        }
      }
      
      result = connectedEntities;
    `;

    const queryResult = await devGraph.executeQuery(customQuery);
    console.log('Connected entities to user_001:', queryResult.result);

    // Test 9: Get graph state
    console.log('\n=== Test 9: Graph State ===');
    const graphState = await devGraph.getCurrentGraphState();
    console.log('Current graph state:');
    console.log(graphState);

    // Test 10: MCP Metrics
    console.log('\n=== Test 10: MCP Metrics ===');
    const metrics = devGraph.getMCPMetrics();
    console.log('MCP Metrics:', JSON.stringify(metrics, null, 2));

    // Test 11: Export snapshot
    console.log('\n=== Test 11: Export Snapshot ===');
    const snapshotPath = await devGraph.exportSnapshot();
    console.log(`Snapshot exported to: ${snapshotPath}`);

    // Test 12: Test connection
    console.log('\n=== Test 12: Test Connection ===');
    const connectionTest = await devGraph.testConnection();
    console.log(`Connection status: ${connectionTest.connected ? 'Connected' : 'Failed'}`);

    // Test 13: Close the graph
    console.log('\n=== Test 13: Close Graph ===');
    await devGraph.close();
    console.log('Graph closed successfully');

    console.log('\n=== All Tests Completed Successfully! ===');
    
    logger.info('KGoT Graph Store Integration Tests completed successfully', {
      operation: 'INTEGRATION_TEST_SUCCESS',
      timestamp: new Date().toISOString(),
      metricsAtEnd: metrics
    });

    return true;

  } catch (error) {
    console.error('\n=== Test Failed ===');
    console.error('Error:', error.message);
    console.error('Stack:', error.stack);
    
    logger.error('KGoT Graph Store Integration Tests failed', {
      operation: 'INTEGRATION_TEST_FAILED',
      error: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString()
    });

    return false;
  }
}

/**
 * Run the tests if this file is executed directly
 */
if (require.main === module) {
  console.log('🚀 Running KGoT Graph Store Integration Tests...\n');
  
  runIntegrationTests()
    .then(success => {
      if (success) {
        console.log('\n✅ Integration tests passed!');
        process.exit(0);
      } else {
        console.log('\n❌ Integration tests failed!');
        process.exit(1);
      }
    })
    .catch(error => {
      console.error('\n💥 Test runner error:', error);
      process.exit(1);
    });
}

module.exports = {
  runIntegrationTests
}; 