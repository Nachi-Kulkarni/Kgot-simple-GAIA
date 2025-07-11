/**
 * Tool Validation Tests
 * Tests for LangChain tool schema validation and tool creation
 */

const { z } = require('zod');
const { StructuredTool } = require('langchain/tools');
const { AlitaManagerAgent } = require('../alita_core/manager_agent/index');

describe('Tool Validation Tests', () => {
  let managerAgent;

  beforeAll(async () => {
    // Mock environment variables for testing
    process.env.NODE_ENV = 'test';
    process.env.PORT = '0'; // Use random available port
    
    // Create manager agent instance
    managerAgent = new AlitaManagerAgent();
    
    // Wait for initialization
    await managerAgent.waitForInitialization();
  });

  afterAll(async () => {
    if (managerAgent) {
      await managerAgent.stop();
    }
  });

  describe('Tool Schema Validation', () => {
    test('should validate tool with correct argsSchema', () => {
      const validTool = new StructuredTool({
        name: 'test_tool',
        description: 'A test tool for validation',
        schema: z.object({
          input: z.string().describe('Test input')
        }),
        func: async ({ input }) => `Processed: ${input}`
      });

      // Test the validation logic used in createAgentTools
      const isValidTool = validTool && 
        validTool.name && 
        validTool.description && 
        validTool.argsSchema && 
        (validTool.argsSchema._def || 
         validTool.argsSchema.shape || 
         typeof validTool.argsSchema.parse === 'function' ||
         typeof validTool.argsSchema.safeParse === 'function');

      expect(isValidTool).toBe(true);
      expect(validTool.name).toBe('test_tool');
      expect(validTool.description).toBe('A test tool for validation');
      expect(validTool.argsSchema).toBeDefined();
    });

    test('should reject tool with missing argsSchema', () => {
      const invalidTool = {
        name: 'invalid_tool',
        description: 'A tool without proper schema',
        // Missing argsSchema
        func: async () => 'test'
      };

      const isValidTool = invalidTool && 
        invalidTool.name && 
        invalidTool.description && 
        invalidTool.argsSchema && 
        (invalidTool.argsSchema._def || 
         invalidTool.argsSchema.shape || 
         typeof invalidTool.argsSchema.parse === 'function' ||
         typeof invalidTool.argsSchema.safeParse === 'function');

      expect(isValidTool).toBe(false);
    });

    test('should validate Zod schema properties', () => {
      const zodSchema = z.object({
        input: z.string(),
        optional: z.number().optional()
      });

      // Test different Zod schema validation approaches
      const hasZodProperties = 
        zodSchema._def || 
        zodSchema.shape || 
        typeof zodSchema.parse === 'function' ||
        typeof zodSchema.safeParse === 'function';

      expect(hasZodProperties).toBe(true);
      expect(typeof zodSchema.parse).toBe('function');
      expect(typeof zodSchema.safeParse).toBe('function');
    });

    test('should handle schema validation errors gracefully', () => {
      const schema = z.object({
        requiredField: z.string()
      });

      // Test valid input
      const validResult = schema.safeParse({ requiredField: 'test' });
      expect(validResult.success).toBe(true);

      // Test invalid input
      const invalidResult = schema.safeParse({ wrongField: 'test' });
      expect(invalidResult.success).toBe(false);
      expect(invalidResult.error).toBeDefined();
    });
  });

  describe('Tool Creation and Validation', () => {
    test('should create valid agent tools', async () => {
      // Wait for components to be ready
      await managerAgent.waitForComponentsReady();
      
      // Create agent tools
      const tools = await managerAgent.createAgentTools();
      
      expect(Array.isArray(tools)).toBe(true);
      expect(tools.length).toBeGreaterThan(0);
      
      // Validate each tool
      tools.forEach((tool, index) => {
        expect(tool).toBeDefined();
        expect(tool.name).toBeDefined();
        expect(tool.description).toBeDefined();
        expect(tool.argsSchema).toBeDefined();
        
        // Test that argsSchema has required Zod properties
        const hasValidSchema = 
          tool.argsSchema._def || 
          tool.argsSchema.shape || 
          typeof tool.argsSchema.parse === 'function' ||
          typeof tool.argsSchema.safeParse === 'function';
        
        expect(hasValidSchema).toBe(true);
      });
    });

    test('should filter out invalid tools', async () => {
      // This test verifies that the validation logic properly filters invalid tools
      const mockInvalidTool = {
        name: 'invalid_tool',
        description: 'Invalid tool for testing',
        // Missing argsSchema intentionally
        func: async () => 'test'
      };

      // Test the validation logic directly
      const isValidTool = mockInvalidTool && 
        mockInvalidTool.name && 
        mockInvalidTool.description && 
        mockInvalidTool.argsSchema && 
        (mockInvalidTool.argsSchema._def || 
         mockInvalidTool.argsSchema.shape || 
         typeof mockInvalidTool.argsSchema.parse === 'function' ||
         typeof mockInvalidTool.argsSchema.safeParse === 'function');

      expect(isValidTool).toBe(false);
    });
  });

  describe('Sequential Thinking Tool Validation', () => {
    test('should validate sequential thinking tool if available', async () => {
      if (managerAgent.sequentialThinking) {
        const sequentialThinkingTool = managerAgent.sequentialThinking.createSequentialThinkingTool();
        
        expect(sequentialThinkingTool).toBeDefined();
        expect(sequentialThinkingTool.name).toBeDefined();
        expect(sequentialThinkingTool.description).toBeDefined();
        expect(sequentialThinkingTool.argsSchema).toBeDefined();
        
        // Validate argsSchema properties
        const isValidSequentialTool = sequentialThinkingTool && 
          sequentialThinkingTool.name && 
          sequentialThinkingTool.description && 
          sequentialThinkingTool.argsSchema && 
          (sequentialThinkingTool.argsSchema._def || 
           sequentialThinkingTool.argsSchema.shape || 
           typeof sequentialThinkingTool.argsSchema.parse === 'function' ||
           typeof sequentialThinkingTool.argsSchema.safeParse === 'function');
        
        expect(isValidSequentialTool).toBe(true);
      } else {
        console.log('Sequential Thinking not available for testing');
      }
    });
  });

  describe('Error Handling', () => {
    test('should handle tool execution errors gracefully', async () => {
      const errorTool = new StructuredTool({
        name: 'error_tool',
        description: 'A tool that throws errors for testing',
        schema: z.object({
          shouldError: z.boolean().describe('Whether to throw an error')
        }),
        func: async ({ shouldError }) => {
          if (shouldError) {
            throw new Error('Test error');
          }
          return 'Success';
        }
      });

      // Test successful execution
      const successResult = await errorTool.func({ shouldError: false });
      expect(successResult).toBe('Success');

      // Test error handling
      await expect(errorTool.func({ shouldError: true }))
        .rejects
        .toThrow('Test error');
    });
  });

  describe('Monitoring Integration', () => {
    test('should have monitoring service available', () => {
      const { monitoring } = require('../alita_core/manager_agent/monitoring');
      
      expect(monitoring).toBeDefined();
      expect(typeof monitoring.recordToolUsage).toBe('function');
      expect(typeof monitoring.recordSystemError).toBe('function');
      expect(typeof monitoring.getMetrics).toBe('function');
    });

    test('should record tool usage metrics', () => {
      const { monitoring } = require('../alita_core/manager_agent/monitoring');
      
      // Test metric recording (should not throw)
      expect(() => {
        monitoring.recordToolUsage('test_tool', 'success', 0.5);
        monitoring.recordToolUsage('test_tool', 'error', 1.0);
      }).not.toThrow();
    });
  });
});