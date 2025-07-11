# Sequential Thinking MCP Integration

## Overview

The Sequential Thinking MCP Integration is a sophisticated reasoning engine that serves as the core intelligence layer for the Alita Manager Agent. It provides advanced complexity detection, systematic routing logic, and intelligent decision trees for coordinating between Alita MCP creation and KGoT knowledge processing systems.

**Status**: ✅ **FIXED** - Schema validation issues resolved, now using DynamicTool for better LangChain compatibility.

## Key Features

### 🧠 Complexity Detection Triggers
- **Task Complexity Score > 7**: Automatically triggers when tasks exceed complexity thresholds
- **Multiple Errors (> 3)**: Activates for error-heavy scenarios requiring systematic resolution
- **Cross-System Coordination**: Detects when both Alita and KGoT systems are needed
- **Multi-Domain Tasks**: Identifies tasks spanning multiple knowledge domains

### 🔄 Thought Process Templates
- **Task Decomposition**: Breaks complex tasks into manageable subtasks
- **Error Resolution**: Systematically resolves multiple or cascading errors
- **System Coordination**: Coordinates operations across Alita and KGoT systems
- **Knowledge Processing**: Integrates knowledge from multiple sources

### 🎯 System Routing Intelligence
- **Decision Trees**: Intelligent routing between Alita and KGoT systems
- **Coordination Strategies**: Parallel, sequential, and integrated execution
- **Fallback Mechanisms**: Graceful degradation and error recovery

### 🔧 Recent Fixes (January 2025)
- **Tool Schema Resolution**: Fixed StructuredTool validation issues by migrating to DynamicTool
- **Context Binding**: Resolved `this` context binding issues in tool execution
- **Input Validation**: Enhanced JSON parsing and required field validation
- **Error Handling**: Improved error messages and recovery mechanisms
- **OpenAI Compatibility**: Eliminated Zod schema warnings for OpenAI API compliance

## Architecture

```mermaid
graph TD
    A[Manager Agent] --> B[Sequential Thinking Integration]
    B --> C[Complexity Detector]
    B --> D[Thought Templates]
    B --> E[Decision Trees]
    B --> F[System Router]
    
    C --> G[Task Analysis]
    C --> H[Error Detection]
    C --> I[System Requirements]
    
    D --> J[Task Decomposition]
    D --> K[Error Resolution]
    D --> L[System Coordination]
    D --> M[Knowledge Processing]
    
    E --> N[System Selection]
    E --> O[Coordination Strategy]
    
    F --> P[Alita System]
    F --> Q[KGoT System]
    F --> R[Parallel Execution]
```

## Configuration

### Initialization Options

```javascript
const sequentialThinking = new SequentialThinkingIntegration({
  // Complexity detection thresholds
  complexityThreshold: 7,        // Task complexity score trigger
  errorThreshold: 3,             // Error count trigger
  maxThoughts: 15,               // Maximum reasoning steps

  // Reasoning engine configuration
  enableCrossSystemCoordination: true,  // Enable Alita-KGoT coordination
  enableAdaptiveThinking: true,         // Dynamic thought adjustment
  thoughtTimeout: 30000,               // 30 second timeout per step

  // Integration settings
  enableMCPCreationRouting: true,      // Route to MCP creation
  enableKGoTProcessing: true           // Route to KGoT processing
});
```

## Usage

### 1. LangChain Tool Integration

The Sequential Thinking integration automatically creates a DynamicTool that can be used by the Manager Agent:

```javascript
// The tool is automatically added to the agent's tools
const tools = this.createAgentTools(); // Includes sequential_thinking tool

// The tool now uses DynamicTool for better compatibility
// with OpenAI function calling and avoids Zod schema issues
```

### 2. Tool Input Format

The Sequential Thinking tool accepts JSON input with the following structure:

```javascript
{
  "taskId": "task_001",                    // Required: Unique task identifier
  "description": "Complex task description", // Required: Detailed task description
  "requirements": [                        // Optional: Array of requirements
    {
      "description": "Web scraping capability",
      "priority": "high",
      "complexity": "medium"
    }
  ],
  "errors": [                             // Optional: Array of errors
    {
      "type": "integration",
      "message": "System coordination failure",
      "severity": "high"
    }
  ],
  "systemsInvolved": ["Alita", "KGoT"],   // Optional: Systems array
  "dataTypes": ["text", "structured"],    // Optional: Data types array
  "interactions": [                       // Optional: Interaction definitions
    {
      "type": "bidirectional",
      "complexity": "high",
      "systems": ["Alita", "KGoT"]
    }
  ],
  "timeline": {                           // Optional: Timeline object
    "urgency": "high",
    "deadline": "2024-01-15T10:00:00Z"
  },
  "dependencies": ["web_agent", "mcp_creation"] // Optional: Dependencies array
}
```

### 3. Direct API Access

#### Trigger Sequential Thinking

```bash
POST /sequential-thinking
Content-Type: application/json

{
  "taskId": "task_001",
  "description": "Complex multi-system integration task",
  "requirements": [
    {
      "description": "Web scraping capability",
      "priority": "high",
      "complexity": "medium"
    },
    {
      "description": "Knowledge graph integration",
      "priority": "critical",
      "complexity": "high"
    }
  ],
  "errors": [
    {
      "type": "integration",
      "message": "System coordination failure",
      "severity": "high"
    }
  ],
  "systemsInvolved": ["Alita", "KGoT"],
  "dataTypes": ["text", "structured", "graph"],
  "interactions": [
    {
      "type": "bidirectional",
      "complexity": "high",
      "systems": ["Alita", "KGoT"]
    }
  ],
  "timeline": {
    "urgency": "high",
    "deadline": "2024-01-15T10:00:00Z"
  },
  "dependencies": ["web_agent", "mcp_creation", "knowledge_graph"]
}
```

#### Response Format

```json
{
  "status": "completed",
  "sessionId": "thinking_task_001_1703123456789",
  "complexityScore": 9,
  "complexityFactors": [
    "Multi-system coordination (2 systems)",
    "High error count (1 errors)",
    "Complex interactions (1)"
  ],
  "template": "System Coordination",
  "conclusions": {
    "keyInsights": {
      "complexityAssessment": "High complexity multi-system task",
      "systemCoordination": "Multi-system coordination required",
      "recommendedActions": ["Implement proper inter-system communication"]
    },
    "overallApproach": {
      "strategy": "coordinated_multi_system",
      "primary": "coordination",
      "description": "Coordinate between Alita and KGoT systems with proper synchronization"
    },
    "actionPlan": [
      {
        "step": 1,
        "action": "Initialize system coordination",
        "description": "Set up communication between Alita and KGoT systems",
        "system": "both",
        "priority": "high"
      }
    ],
    "riskAssessment": {
      "overallRisk": "medium",
      "identifiedRisks": [
        {
          "type": "coordination_failure",
          "severity": "medium",
          "description": "Systems may fail to coordinate properly",
          "mitigation": "Implement robust error handling and fallback mechanisms"
        }
      ]
    }
  },
  "routingRecommendations": {
    "systemSelection": {
      "primary": "both",
      "secondary": null,
      "coordination": "integrated",
      "reasoning": "Complex multi-domain task requires integrated approach"
    },
    "coordinationStrategy": {
      "strategy": "integrated_workflow",
      "execution": "parallel_with_synchronization",
      "monitoring": "real_time",
      "fallback": "graceful_degradation"
    },
    "executionSequence": [
      {
        "step": 1,
        "system": "coordinator",
        "action": "initialize_both_systems",
        "description": "Initialize both Alita and KGoT systems"
      },
      {
        "step": 2,
        "system": "both",
        "action": "parallel_execution",
        "description": "Execute tasks in parallel with synchronization"
      }
    ]
  },
  "duration": 1234,
  "thoughtCount": 6,
  "timestamp": "2025-01-10T22:48:56.125Z"
}
```

#### Error Response Format

```json
{
  "status": "error",
  "error": "taskId and description are required fields",
  "recommendation": "Provide both taskId and description",
  "timestamp": "2025-01-10T22:48:56.125Z"
}
```

#### Not Required Response Format

```json
{
  "status": "not_required",
  "message": "Task complexity does not require sequential thinking",
  "complexityScore": 5,
  "complexityFactors": [
    "Single system operation (1 system)",
    "Low error count (0 errors)"
  ],
  "recommendation": "Proceed with standard processing"
}
```

### 4. Monitoring Active Sessions

```bash
GET /sequential-thinking/sessions
```

Response:
```json
{
  "activeSessions": 2,
  "sessions": [
    {
      "sessionId": "thinking_task_001_1703123456789",
      "taskId": "task_001",
      "template": "System Coordination",
      "progress": 0.8,
      "duration": 15234,
      "status": "active"
    },
    {
      "sessionId": "thinking_task_002_1703123456790",
      "taskId": "task_002", 
      "template": "Error Resolution",
      "progress": 1.0,
      "duration": 8567,
      "status": "completed"
    }
  ],
  "timestamp": "2025-01-10T22:48:56.125Z"
}
```

## Complexity Detection Algorithm

### Scoring Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Multi-system involvement | +3 | Tasks requiring both Alita and KGoT |
| Error count > 3 | +4 | High error scenarios |
| Data type diversity | +2 | Multiple data formats |
| Complex interactions | +3 | Bidirectional or high-complexity interactions |
| Complex requirements | +2 | Critical or high-complexity requirements |
| High dependencies | +2 | More than 5 dependencies |
| Time constraints | +1 | High urgency scenarios |

### Trigger Conditions

```javascript
const triggerConditions = {
  complexityThreshold: complexityScore > 7,
  errorThreshold: errors.length > 3,
  crossSystemRequired: systemsInvolved.length > 1,
  multiDomainTask: dataTypes.length > 3,
  cascadingErrors: errors.some(e => e.type === 'cascading'),
  integrationRequired: interactions.some(i => i.type === 'integration')
};
```

## Decision Trees

### System Selection Logic

```mermaid
graph TD
    A[Task Analysis] --> B{Requires Knowledge Graph?}
    B -->|Yes| C{Complex Reasoning?}
    C -->|Yes| D[Primary: KGoT<br/>Secondary: Alita if MCP needed]
    C -->|No| E{Requires MCP Creation?}
    
    B -->|No| E
    E -->|Yes| F{Web/Code Generation?}
    F -->|Yes| G[Primary: Alita<br/>Secondary: KGoT if needed]
    F -->|No| H{High Data Complexity?}
    
    H -->|Yes| I[Both Systems<br/>Integrated Approach]
    H -->|No| J[Primary: Alita<br/>Single System]
```

### Coordination Strategy

```mermaid
graph TD
    A[System Selection] --> B{Both Systems?}
    B -->|Yes| C[Integrated Workflow<br/>Parallel + Sync]
    B -->|No| D{Secondary System?}
    D -->|Yes| E{High Complexity?}
    E -->|Yes| F[Sequential Coordination<br/>Primary → Secondary]
    E -->|No| G[Single System<br/>Direct Execution]
    D -->|No| G
```

## Thought Process Templates

### 1. Task Decomposition

**Triggers**: Complexity score > 7, multiple domains, unclear requirements

**Steps**:
1. Analyze overall task complexity and scope
2. Identify key domains and technologies involved
3. Break down into logical subtasks with dependencies
4. Prioritize subtasks based on criticality
5. Validate decomposition against requirements
6. Create execution plan with resource allocation

### 2. Error Resolution

**Triggers**: Error count > 3, cascading failures, system integration errors

**Steps**:
1. Categorize and prioritize all detected errors
2. Identify root causes and error propagation patterns
3. Determine symptoms vs. causes
4. Plan resolution sequence to minimize cascading effects
5. Identify required tools and system components
6. Validate resolution plan against constraints

### 3. System Coordination

**Triggers**: Cross-system required, data synchronization, workflow orchestration

**Steps**:
1. Map capabilities to appropriate systems (Alita vs KGoT)
2. Identify data flow and synchronization points
3. Plan inter-system communication and coordination
4. Design fallback strategies for system failures
5. Optimize workflow for parallel vs. sequential execution
6. Validate coordination plan against limitations

### 4. Knowledge Processing

**Triggers**: Knowledge integration, complex reasoning, multi-source analysis

**Steps**:
1. Identify and categorize available knowledge sources
2. Analyze knowledge quality and reliability
3. Design integration strategy for disparate knowledge types
4. Plan reasoning workflow for knowledge synthesis
5. Validate integrated knowledge for consistency
6. Generate actionable insights from processed knowledge

## Event System

### Events Emitted

#### `thinkingProgress`
```javascript
{
  sessionId: "thinking_task_001_1703123456789",
  progress: 0.6,
  currentThought: {
    conclusion: "Task complexity analysis: Score 9/10...",
    recommendations: ["Consider phased approach"]
  }
}
```

#### `thinkingComplete`
```javascript
{
  sessionId: "thinking_task_001_1703123456789",
  duration: 15234,
  thoughtCount: 6,
  template: {
    name: "System Coordination"
  },
  conclusions: { /* ... */ },
  systemRecommendations: { /* ... */ }
}
```

#### `systemCoordinationPlan`
```javascript
{
  sessionId: "thinking_task_001_1703123456789",
  routingRecommendations: {
    systemSelection: { primary: "both" },
    coordinationStrategy: { strategy: "integrated_workflow" }
  },
  conclusions: { /* ... */ },
  taskId: "task_001"
}
```

## Integration with Manager Agent

### Automatic Tool Creation

```javascript
// DynamicTool automatically added to LangChain agent tools
const sequentialThinkingTool = this.sequentialThinking.createSequentialThinkingTool();

// Fixed version resolves schema validation issues:
// - Uses DynamicTool instead of StructuredTool
// - Proper context binding with sequentialThinkingInstance
// - Enhanced input validation and error handling
```

### Event Coordination

```javascript
// Listen for thinking completion
this.sequentialThinking.on('thinkingComplete', (thinkingSession) => {
  // Emit system coordination event
  this.emit('systemCoordinationPlan', {
    sessionId: thinkingSession.sessionId,
    routingRecommendations: thinkingSession.systemRecommendations,
    conclusions: thinkingSession.conclusions,
    taskId: thinkingSession.taskId
  });
});

// Execute coordination based on recommendations
this.on('systemCoordinationPlan', async (coordinationPlan) => {
  await this.executeSystemCoordination(coordinationPlan);
});
```

## Testing

### Basic Complexity Detection Test

```javascript
const taskContext = {
  taskId: "test_001",
  description: "Simple web scraping task",
  systemsInvolved: ["Alita"],
  errors: [],
  requirements: [{ priority: "medium" }],
  dataTypes: ["text"]
};

const analysis = sequentialThinking.detectComplexity(taskContext);
console.log(analysis.shouldTriggerSequentialThinking); // false
```

### Complex Multi-System Test

```javascript
const complexTaskContext = {
  taskId: "test_002",
  description: "Multi-system knowledge integration with error handling",
  systemsInvolved: ["Alita", "KGoT"],
  errors: [
    { type: "cascading", severity: "high" },
    { type: "integration", severity: "medium" },
    { type: "timeout", severity: "low" },
    { type: "validation", severity: "high" }
  ],
  requirements: [
    { priority: "critical", complexity: "high" },
    { priority: "high", complexity: "high" },
    { priority: "medium", complexity: "medium" }
  ],
  dataTypes: ["text", "structured", "graph", "multimedia"],
  interactions: [
    { type: "bidirectional", complexity: "high" },
    { type: "integration", complexity: "high" }
  ],
  dependencies: ["web_agent", "mcp_creation", "knowledge_graph", "validation", "optimization", "multimodal"]
};

const analysis = sequentialThinking.detectComplexity(complexTaskContext);
console.log(analysis.complexityScore); // Should be > 7
console.log(analysis.shouldTriggerSequentialThinking); // true
console.log(analysis.recommendedTemplate.name); // "Error Resolution" or "System Coordination"
```

## Fixed Implementation Details

### Tool Schema Validation Resolution

**Problem**: StructuredTool schema properties were not accessible, causing validation failures:
```javascript
// Previous failing validation:
hasName: false,
hasDescription: false,
hasSchema: false
```

**Solution**: Migrated to DynamicTool with proper property exposure:
```javascript
// Fixed implementation:
return new DynamicTool({
  name: "sequential_thinking",
  description: "Triggers sequential thinking...",
  func: async (input) => { /* ... */ }
});
```

### Context Binding Issues

**Problem**: `this` context lost when tool function executed by LangChain.

**Solution**: Store instance reference for proper context binding:
```javascript
const sequentialThinkingInstance = this;

// Use instance reference in tool function:
const complexityAnalysis = sequentialThinkingInstance.detectComplexity(taskContext);
```

### Input Validation Improvements

**Previous**: Relied on Zod schema validation (which was failing).

**Current**: Manual validation with clear error messages:
```javascript
// Validate required fields
if (!taskId || !description) {
  return JSON.stringify({
    status: 'error',
    error: 'taskId and description are required fields',
    recommendation: 'Provide both taskId and description'
  });
}
```

## Performance Considerations

### Memory Management
- Automatic session cleanup after 1 hour
- Complexity score caching for frequent tasks
- Event listener cleanup on shutdown

### Timeouts
- Individual thought step timeout: 30 seconds
- Overall thinking process: Based on step count × timeout
- API endpoint timeout: 60 seconds

### Monitoring
- Active session tracking
- Performance metrics logging
- Error rate monitoring

## Best Practices

### 1. Task Context Structure
- Always provide `taskId` and `description` (required fields)
- Include relevant `systemsInvolved` for accurate routing
- Specify `errors` array for error-heavy scenarios
- Define `requirements` with priority and complexity
- List `dataTypes` for multi-modal tasks

### 2. Error Handling
- Monitor for `SEQUENTIAL_THINKING_ERROR` log events
- Implement fallback to standard processing
- Use timeout configurations appropriate for task complexity
- Handle JSON parsing errors gracefully

### 3. Performance Optimization
- Clean up sessions periodically using `cleanupSessions()`
- Cache complexity analysis for repeated tasks
- Use appropriate complexity thresholds for your use case
- Monitor active session count

### 4. Monitoring
- Track active sessions via `/sequential-thinking/sessions`
- Monitor system status via `/status` endpoint
- Set up alerts for failed thinking processes
- Monitor memory usage and cleanup cycles

## Troubleshooting

### Common Issues

#### Sequential Thinking Tool Not Loading
**Symptoms**: Tool validation failures, undefined properties
**Solution**: Ensure DynamicTool is properly imported and instance context is bound
```javascript
// Check logs for:
// ❌ Sequential Thinking tool has invalid schema
// ✅ Sequential Thinking tool added successfully
```

#### Schema Validation Errors
**Symptoms**: OpenAI API warnings about optional fields
**Solution**: Fixed in current implementation - no longer uses Zod schemas
```javascript
// Old problem: .optional() without .nullable()
// New solution: Manual validation in DynamicTool func
```

#### Context Binding Failures
**Symptoms**: `this.detectComplexity is not a function`
**Solution**: Use instance reference approach:
```javascript
const sequentialThinkingInstance = this;
// Use sequentialThinkingInstance.detectComplexity() in tool function
```

#### Input Parsing Issues
**Symptoms**: JSON parse errors, missing required fields
**Solution**: Enhanced input validation:
```javascript
const params = typeof input === 'string' ? JSON.parse(input) : input;
if (!taskId || !description) {
  return JSON.stringify({ status: 'error', ... });
}
```

### Debug Logging

Enable debug logging for detailed insights:
```javascript
// Set LOG_LEVEL=debug in environment
process.env.LOG_LEVEL = 'debug';
```

Key log operations to monitor:
- `SEQUENTIAL_THINKING_INIT`
- `TOOL_CREATION`
- `SEQUENTIAL_THINKING_TOOL_INVOKED`
- `COMPLEXITY_DETECTION`
- `SEQUENTIAL_THINKING_START`
- `THOUGHT_STEP_EXECUTION`
- `SYSTEM_COORDINATION_EXECUTION`

### Testing the Fixed Implementation

```bash
# Test the fixed sequential thinking tool
curl -X POST http://localhost:8888/sequential-thinking \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": "test_001",
    "description": "Test complex multi-system task",
    "systemsInvolved": ["Alita", "KGoT"],
    "errors": [
      {"type": "integration", "message": "Connection failed", "severity": "high"},
      {"type": "timeout", "message": "Request timeout", "severity": "medium"},
      {"type": "validation", "message": "Schema error", "severity": "high"},
      {"type": "coordination", "message": "Sync failure", "severity": "high"}
    ],
    "dataTypes": ["text", "structured", "graph", "multimedia"]
  }'

# Expected response should now show:
# "status": "completed" with full thinking results
# OR "status": "not_required" with complexity analysis
# No more undefined property errors
```

## Future Enhancements

### Planned Features
1. **Machine Learning Integration**: Learn from past thinking sessions
2. **Custom Template Creation**: User-defined thought process templates
3. **Real-time Collaboration**: Multi-agent thinking sessions
4. **Performance Analytics**: Detailed metrics and optimization suggestions
5. **External System Integration**: Connect to additional reasoning systems

### Extensibility Points
- Custom complexity scoring algorithms
- Additional thought process templates
- New coordination strategies
- Enhanced fallback mechanisms
- Integration with external MCP tools

## Contributing

When extending the Sequential Thinking Integration:

1. Follow the existing event-driven architecture
2. Maintain comprehensive logging for all operations
3. Include proper error handling and fallbacks
4. Add tests for new complexity detection factors
5. Update documentation for new features
6. Use DynamicTool for new LangChain tool integrations
7. Ensure proper context binding for class methods

## License

This Sequential Thinking MCP Integration is part of the Alita-KGoT Enhanced system and follows the same licensing terms.