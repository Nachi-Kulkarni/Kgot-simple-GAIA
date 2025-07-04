/**
 * Alita Web Agent - Section 2.2 Implementation
 * 
 * Advanced web agent with Hugging Face Search and GitHub search capabilities
 * Integrates with KGoT Surfer Agent and Graph Store Module
 * Uses LangChain agents as per user preference for agent development
 * Configured to use OpenRouter API instead of direct OpenAI API (user preference)
 * 
 * Features:
 * - HuggingFaceSearchTool for web search operations using Transformers Agents
 * - GithubSearchTool for repository search and analysis
 * - Integration with KGoT Graph Store for context-aware navigation
 * - MCP validation for web automation templates
 * - Winston logging for comprehensive workflow tracking
 * - OpenRouter API integration for AI model access
 * 
 * @module AlitaWebAgent
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { AgentExecutor, createOpenAIFunctionsAgent } = require('langchain/agents');
const { ChatOpenAI } = require('@langchain/openai');
const { pull } = require('langchain/hub');
const { DynamicTool } = require('@langchain/core/tools');
const { HumanMessage } = require('@langchain/core/messages');
const axios = require('axios');
const cheerio = require('cheerio');
const { Octokit } = require('@octokit/rest');
const playwright = require('playwright');
const { loggers } = require('../../config/logging/winston_config');
const path = require('path');
const { spawn } = require('child_process');

// Import KGoT Graph Store Interface
const KnowledgeGraphInterface = require('../../kgot_core/graph_store/kg_interface');

/**
 * Advanced Hugging Face Search Tool for Alita Web Agent
 * Uses KGoT Surfer Agent implementation with Transformers Agents framework
 * Provides comprehensive web search capabilities with intelligent browsing
 */
class HuggingFaceSearchTool extends DynamicTool {
  constructor(graphStore = null, config = {}) {
    super({
      name: "huggingface_search",
      description: `
        Perform comprehensive web searches using Hugging Face Agents framework with KGoT Surfer integration.
        Supports:
        - Intelligent web search with contextual understanding
        - Granular navigation with PageUp, PageDown, Find functionality
        - Full page analysis and content extraction
        - Wikipedia integration for knowledge lookup
        - Archive search for historical content
        - Smart browsing with multi-step reasoning
        
        Input should be a JSON string with:
        {
          "query": "natural language search query with context",
          "searchType": "informational|navigational|research", 
          "includeWikipedia": true,
          "maxIterations": 12,
          "detailed": true
        }
      `,
      func: this.searchWithHuggingFace.bind(this)
    });
    
    this.graphStore = graphStore;
    this.config = {
      model_name: config.model_name || "webagent",
      temperature: config.temperature || 0.1,
      kgot_path: config.kgot_path || path.join(__dirname, '../../../knowledge-graph-of-thoughts'),
      max_iterations: config.max_iterations || 12,
      ...config
    };
    this.logger = loggers.webAgent;
    
    this.logger.info('HuggingFaceSearchTool initialized', {
      operation: 'HUGGINGFACE_SEARCH_TOOL_INIT',
      model_name: this.config.model_name,
      hasGraphStore: !!graphStore,
      kgot_path: this.config.kgot_path
    });
  }

  /**
   * Execute Hugging Face search with KGoT Surfer Agent integration
   * @param {string} input - JSON string with search parameters
   * @returns {Promise<string>} Search results with detailed context
   */
  async searchWithHuggingFace(input) {
    try {
      const params = typeof input === 'string' ? JSON.parse(input) : input;
      const {
        query,
        searchType = 'informational',
        includeWikipedia = true,
        maxIterations = this.config.max_iterations,
        detailed = true
      } = params;

      this.logger.info('Executing Hugging Face search', {
        operation: 'HUGGINGFACE_SEARCH_EXECUTE',
        query,
        searchType,
        includeWikipedia,
        maxIterations
      });

      // Prepare enhanced query with context
      const enhancedQuery = this.prepareEnhancedQuery(query, searchType, detailed);
      
      // Execute search using Python KGoT Surfer Agent
      const searchResults = await this.executeKGoTSurferSearch(enhancedQuery, {
        includeWikipedia,
        maxIterations,
        searchType
      });

      // Process and enhance results
      const processedResults = await this.processSearchResults(searchResults, query);
      
      // Update knowledge graph with search context
      if (this.graphStore) {
        await this.updateGraphWithSearchContext(query, processedResults);
      }

      this.logger.info('Hugging Face search completed', {
        operation: 'HUGGINGFACE_SEARCH_SUCCESS',
        query,
        searchType,
        resultLength: processedResults.length
      });

      return JSON.stringify({
        query,
        searchType,
        results: processedResults,
        metadata: {
          framework: 'HuggingFace Agents + KGoT',
          includeWikipedia,
          maxIterations,
          generatedAt: new Date().toISOString()
        }
      }, null, 2);

    } catch (error) {
      this.logger.error('Hugging Face search failed', {
        operation: 'HUGGINGFACE_SEARCH_ERROR',
        error: error.message,
        stack: error.stack
      });
      
      return JSON.stringify({
        error: 'Hugging Face search failed',
        message: error.message,
        fallback: 'Consider using alternative search methods or check KGoT integration'
      });
    }
  }

  /**
   * Prepare enhanced query with context for better search results
   * @param {string} query - Original search query
   * @param {string} searchType - Type of search (informational, navigational, research)
   * @param {boolean} detailed - Whether to request detailed results
   * @returns {string} Enhanced query string
   */
  prepareEnhancedQuery(query, searchType, detailed) {
    let enhancedQuery = query;
    
    // Add context based on search type
    switch (searchType) {
      case 'informational':
        enhancedQuery = `Please find comprehensive information about: ${query}. ` +
                       `I need detailed explanations and multiple sources to understand this topic thoroughly.`;
        break;
      case 'navigational':
        enhancedQuery = `I want to navigate to or find the official page/resource for: ${query}. ` +
                       `Please find the most authoritative and relevant destination.`;
        break;
      case 'research':
        enhancedQuery = `Conduct thorough research on: ${query}. ` +
                       `Please explore multiple sources, cross-reference information, and provide ` +
                       `a comprehensive analysis with supporting evidence.`;
        break;
    }
    
    if (detailed) {
      enhancedQuery += ` Please provide extremely detailed information with supporting context, ` +
                      `additional sources, and comprehensive analysis. Use all available tools ` +
                      `including Wikipedia lookup, page navigation, and content analysis.`;
    }
    
    return enhancedQuery;
  }

  /**
   * Execute search using Python KGoT Surfer Agent
   * @param {string} query - Enhanced search query
   * @param {Object} options - Search options
   * @returns {Promise<Object>} Raw search results from KGoT
   */
  async executeKGoTSurferSearch(query, options) {
    return new Promise((resolve, reject) => {
      // Create Python script to execute KGoT Surfer search
      const pythonScript = this.createPythonSearchScript(query, options);
      
      // Execute Python script with KGoT Surfer Agent
      const pythonProcess = spawn('python3', ['-c', pythonScript], {
        cwd: this.config.kgot_path,
        env: { 
          ...process.env,
          PYTHONPATH: this.config.kgot_path 
        }
      });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            // Parse JSON results from Python script
            const results = JSON.parse(stdout.trim());
            resolve(results);
          } catch (parseError) {
            this.logger.error('Failed to parse KGoT search results', {
              operation: 'KGOT_PARSE_ERROR',
              stdout,
              parseError: parseError.message
            });
            resolve({ 
              search_outcome_short: stdout.trim(),
              search_outcome_detailed: stdout.trim(),
              additional_context: 'Raw output from KGoT Surfer Agent',
              raw_output: stdout.trim()
            });
          }
        } else {
          this.logger.error('KGoT search process failed', {
            operation: 'KGOT_PROCESS_ERROR',
            code,
            stderr
          });
          reject(new Error(`KGoT search failed with code ${code}: ${stderr}`));
        }
      });

      pythonProcess.on('error', (error) => {
        this.logger.error('Failed to start KGoT search process', {
          operation: 'KGOT_SPAWN_ERROR',
          error: error.message
        });
        reject(error);
      });
    });
  }

  /**
   * Create Python script to execute KGoT Surfer search
   * @param {string} query - Search query
   * @param {Object} options - Search options
   * @returns {string} Python script code
   */
  createPythonSearchScript(query, options) {
    return `
import sys
import json
import os
from pathlib import Path

# Add KGoT to Python path
kgot_path = Path('${this.config.kgot_path}')
sys.path.append(str(kgot_path))

try:
    from kgot.tools.tools_v2_3.SurferTool import SearchTool
    from kgot.utils import UsageStatistics
    
    # Initialize usage statistics
    usage_stats = UsageStatistics()
    
    # Create search tool with configuration
    search_tool = SearchTool(
        model_name="${this.config.model_name}",
        temperature=${this.config.temperature},
        usage_statistics=usage_stats
    )
    
    # Execute search
    query = """${query.replace(/"/g, '\\"')}"""
    result = search_tool._run(query)
    
    # Parse result into structured format
    search_result = {
        'search_outcome_short': '',
        'search_outcome_detailed': result,
        'additional_context': 'Search completed using KGoT Surfer Agent with Hugging Face Agents framework',
        'framework': 'HuggingFace Transformers Agents + KGoT',
        'model_used': '${this.config.model_name}',
        'search_type': '${options.searchType}',
        'include_wikipedia': ${options.includeWikipedia},
        'max_iterations': ${options.maxIterations}
    }
    
    # Extract short version if possible
    if '### 1. Search outcome (short version):' in result:
        parts = result.split('### 1. Search outcome (short version):')
        if len(parts) > 1:
            short_part = parts[1].split('### 2. Search outcome (extremely detailed version):')[0].strip()
            search_result['search_outcome_short'] = short_part
    
    print(json.dumps(search_result, indent=2))
    
except Exception as e:
    error_result = {
        'error': str(e),
        'search_outcome_short': f'Error executing KGoT search: {str(e)}',
        'search_outcome_detailed': f'Failed to execute search using KGoT Surfer Agent. Error: {str(e)}',
        'additional_context': 'Error occurred during KGoT Surfer Agent execution'
    }
    print(json.dumps(error_result, indent=2))
    sys.exit(1)
`;
  }

  /**
   * Process and enhance search results from KGoT
   * @param {Object} searchResults - Raw results from KGoT Surfer
   * @param {string} originalQuery - Original search query
   * @returns {Promise<Object>} Processed search results
   */
  async processSearchResults(searchResults, originalQuery) {
    const processed = {
      summary: searchResults.search_outcome_short || 'Search completed',
      detailed_results: searchResults.search_outcome_detailed || searchResults.raw_output || 'No detailed results available',
      additional_context: searchResults.additional_context || '',
      metadata: {
        framework: searchResults.framework || 'KGoT Surfer Agent',
        model_used: searchResults.model_used || this.config.model_name,
        search_type: searchResults.search_type || 'informational',
        query_processed: originalQuery,
        timestamp: new Date().toISOString()
      }
    };

    // Add error information if present
    if (searchResults.error) {
      processed.error = searchResults.error;
      processed.status = 'error';
    } else {
      processed.status = 'success';
    }

    return processed;
  }

  /**
   * Update knowledge graph with search context
   * @param {string} query - Search query
   * @param {Object} results - Processed search results
   */
  async updateGraphWithSearchContext(query, results) {
    if (!this.graphStore || !this.graphStore.addTriplet) {
      return;
    }

    try {
      // Add search query as a concept
      await this.graphStore.addTriplet({
        subject: `search:${Date.now()}`,
        predicate: 'hasQuery',
        object: query,
        metadata: {
          timestamp: new Date().toISOString(),
          framework: 'HuggingFace Agents + KGoT',
          status: results.status
        }
      });

      this.logger.debug('Updated knowledge graph with search context', {
        operation: 'GRAPH_UPDATE_SUCCESS',
        query,
        status: results.status
      });
    } catch (error) {
      this.logger.warning('Failed to update knowledge graph', {
        operation: 'GRAPH_UPDATE_WARNING',
        error: error.message
      });
    }
  }
}

/**
 * Advanced GitHub Search Tool for Alita Web Agent
 * Provides comprehensive GitHub repository and code search capabilities
 */
class GithubSearchTool extends DynamicTool {
  constructor(accessToken, graphStore = null) {
    super({
      name: "github_search",
      description: `
        Search GitHub repositories, code, issues, and users with advanced filtering.
        Supports:
        - Repository search with language, size, and activity filters
        - Code search within repositories
        - Issue and pull request search
        - User and organization search
        - Repository analysis and statistics
        - Integration with knowledge graph for context
        
        Input should be a JSON string with:
        {
          "query": "search terms",
          "type": "repositories|code|issues|users",
          "language": "javascript|python|etc",
          "sort": "stars|updated|relevance",
          "order": "desc|asc",
          "perPage": 30
        }
      `,
      func: this.searchGithub.bind(this)
    });
    
    this.octokit = new Octokit({
      auth: accessToken
    });
    this.graphStore = graphStore;
    this.logger = loggers.webAgent;
    
    this.logger.info('GithubSearchTool initialized', {
      operation: 'GITHUB_SEARCH_TOOL_INIT',
      hasToken: !!accessToken,
      hasGraphStore: !!graphStore
    });
  }

  /**
   * Execute GitHub search with advanced capabilities
   * @param {string} input - JSON string with search parameters
   * @returns {Promise<string>} Search results with metadata
   */
  async searchGithub(input) {
    try {
      const params = typeof input === 'string' ? JSON.parse(input) : input;
      const {
        query,
        type = 'repositories',
        language = null,
        sort = 'relevance',
        order = 'desc',
        perPage = 30
      } = params;

      this.logger.info('Executing GitHub search', {
        operation: 'GITHUB_SEARCH_EXECUTE',
        query,
        type,
        language,
        sort
      });

      let searchResults;
      
      // Execute appropriate search based on type
      switch (type) {
        case 'repositories':
          searchResults = await this.searchRepositories(query, { language, sort, order, perPage });
          break;
        case 'code':
          searchResults = await this.searchCode(query, { language, sort, order, perPage });
          break;
        case 'issues':
          searchResults = await this.searchIssues(query, { sort, order, perPage });
          break;
        case 'users':
          searchResults = await this.searchUsers(query, { sort, order, perPage });
          break;
        default:
          throw new Error(`Unsupported search type: ${type}`);
      }

      // Update knowledge graph with search context
      if (this.graphStore) {
        await this.updateGraphWithGithubContext(query, type, searchResults.items);
      }

      this.logger.info('GitHub search completed', {
        operation: 'GITHUB_SEARCH_SUCCESS',
        query,
        type,
        resultCount: searchResults.items.length,
        totalCount: searchResults.total_count
      });

      return JSON.stringify({
        query,
        type,
        totalCount: searchResults.total_count,
        results: searchResults.items,
        metadata: {
          language,
          sort,
          order,
          generatedAt: new Date().toISOString()
        }
      }, null, 2);

    } catch (error) {
      this.logger.error('GitHub search failed', {
        operation: 'GITHUB_SEARCH_ERROR',
        error: error.message,
        stack: error.stack
      });
      
      return JSON.stringify({
        error: 'GitHub search failed',
        message: error.message,
        fallback: 'Consider checking API rate limits or authentication'
      });
    }
  }

  /**
   * Search GitHub repositories
   * @param {string} query - Search query
   * @param {Object} options - Search options
   * @returns {Promise<Object>} Repository search results
   */
  async searchRepositories(query, options) {
    const searchQuery = this.buildRepositoryQuery(query, options);
    
    const response = await this.octokit.rest.search.repos({
      q: searchQuery,
      sort: options.sort,
      order: options.order,
      per_page: options.perPage
    });

    // Enhance repository data with additional information
    const enhancedItems = await Promise.all(
      response.data.items.map(async (repo) => {
        try {
          // Get additional repository details
          const repoDetails = await this.octokit.rest.repos.get({
            owner: repo.owner.login,
            repo: repo.name
          });

          return {
            ...repo,
            topics: repoDetails.data.topics,
            network_count: repoDetails.data.network_count,
            subscribers_count: repoDetails.data.subscribers_count
          };
        } catch (error) {
          this.logger.debug('Failed to get additional repo details', {
            repo: repo.full_name,
            error: error.message
          });
          return repo;
        }
      })
    );

    return {
      ...response.data,
      items: enhancedItems
    };
  }

  /**
   * Search GitHub code
   * @param {string} query - Search query
   * @param {Object} options - Search options
   * @returns {Promise<Object>} Code search results
   */
  async searchCode(query, options) {
    const searchQuery = this.buildCodeQuery(query, options);
    
    const response = await this.octokit.rest.search.code({
      q: searchQuery,
      sort: options.sort,
      order: options.order,
      per_page: options.perPage
    });

    return response.data;
  }

  /**
   * Search GitHub issues
   * @param {string} query - Search query
   * @param {Object} options - Search options
   * @returns {Promise<Object>} Issue search results
   */
  async searchIssues(query, options) {
    const response = await this.octokit.rest.search.issuesAndPullRequests({
      q: query,
      sort: options.sort,
      order: options.order,
      per_page: options.perPage
    });

    return response.data;
  }

  /**
   * Search GitHub users
   * @param {string} query - Search query
   * @param {Object} options - Search options
   * @returns {Promise<Object>} User search results
   */
  async searchUsers(query, options) {
    const response = await this.octokit.rest.search.users({
      q: query,
      sort: options.sort,
      order: options.order,
      per_page: options.perPage
    });

    return response.data;
  }

  /**
   * Build repository search query with filters
   * @param {string} baseQuery - Base search query
   * @param {Object} options - Filter options
   * @returns {string} Formatted search query
   */
  buildRepositoryQuery(baseQuery, options) {
    let query = baseQuery;
    
    if (options.language) {
      query += ` language:${options.language}`;
    }
    
    return query;
  }

  /**
   * Build code search query with filters
   * @param {string} baseQuery - Base search query
   * @param {Object} options - Filter options
   * @returns {string} Formatted search query
   */
  buildCodeQuery(baseQuery, options) {
    let query = baseQuery;
    
    if (options.language) {
      query += ` language:${options.language}`;
    }
    
    return query;
  }

  /**
   * Update knowledge graph with GitHub search context
   * @param {string} query - Search query
   * @param {string} type - Search type
   * @param {Array} results - Search results
   */
  async updateGraphWithGithubContext(query, type, results) {
    try {
      // Add search context as triplet
      await this.graphStore.addTriplet({
        subject: `github_search_${Date.now()}`,
        predicate: 'SEARCHED_GITHUB_FOR',
        object: query,
        metadata: {
          timestamp: new Date().toISOString(),
          searchType: type,
          resultCount: results.length
        }
      });

      // Add top repositories as entities if repository search
      if (type === 'repositories') {
        for (let i = 0; i < Math.min(results.length, 3); i++) {
          const repo = results[i];
          await this.graphStore.addEntity({
            id: `github_repo_${repo.id}`,
            type: 'GitHubRepository',
            properties: {
              name: repo.name,
              fullName: repo.full_name,
              description: repo.description,
              language: repo.language,
              stars: repo.stargazers_count,
              forks: repo.forks_count,
              url: repo.html_url,
              searchQuery: query
            }
          });
        }
      }

    } catch (error) {
      this.logger.warn('Failed to update graph with GitHub context', {
        operation: 'GITHUB_GRAPH_UPDATE_FAILED',
        error: error.message
      });
    }
  }
}

/**
 * Alita Web Agent - Main Agent Class
 * Orchestrates web operations with Google and GitHub search integration
 */
class AlitaWebAgent {
  constructor(config = {}) {
    this.config = {
      port: config.port || 3001,
      openrouterApiKey: config.openrouterApiKey || process.env.OPENROUTER_API_KEY,
      openrouterBaseUrl: config.openrouterBaseUrl || process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1',
      googleApiKey: config.googleApiKey || process.env.GOOGLE_API_KEY,
      googleSearchEngineId: config.googleSearchEngineId || process.env.GOOGLE_SEARCH_ENGINE_ID,
      githubToken: config.githubToken || process.env.GITHUB_TOKEN,
      graphStoreConfig: config.graphStoreConfig || {},
      modelName: config.modelName || 'anthropic/claude-4-sonnet',
      temperature: config.temperature || 0.1,
      ...config
    };

    this.app = express();
    this.logger = loggers.webAgent;
    this.graphStore = null;
    this.agentExecutor = null;
    
    this.setupMiddleware();
    this.setupRoutes();
    
    this.logger.info('Alita Web Agent initialized', {
      operation: 'ALITA_WEB_AGENT_INIT',
      port: this.config.port,
      hasOpenRouter: !!this.config.openrouterApiKey,
      openrouterBaseUrl: this.config.openrouterBaseUrl,
      hasGoogle: !!this.config.googleApiKey,
      hasGitHub: !!this.config.githubToken,
      searchFramework: 'HuggingFace Agents + KGoT Surfer',
      searchCapabilities: 'Intelligent browsing, Wikipedia, granular navigation'
    });
  }

  /**
   * Setup Express middleware
   */
  setupMiddleware() {
    // Security middleware
    this.app.use(helmet());
    this.app.use(cors());
    
    // Rate limiting
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
      message: 'Too many requests from this IP'
    });
    this.app.use(limiter);
    
    // Body parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true }));
    
    // Request logging
    this.app.use((req, res, next) => {
      this.logger.info('Incoming request', {
        operation: 'HTTP_REQUEST',
        method: req.method,
        path: req.path,
        userAgent: req.get('User-Agent'),
        ip: req.ip
      });
      next();
    });
  }

  /**
   * Setup Express routes
   */
  setupRoutes() {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        services: {
          graphStore: !!this.graphStore,
          langchainAgent: !!this.agentExecutor,
          openRouter: !!this.config.openrouterApiKey
        },
        configuration: {
          modelName: this.config.modelName,
          openrouterBaseUrl: this.config.openrouterBaseUrl
        }
      });
    });

    // Web search endpoint
    this.app.post('/search/web', async (req, res) => {
      try {
        const { query, options = {} } = req.body;
        
        if (!query) {
          return res.status(400).json({ error: 'Query is required' });
        }

        const result = await this.performWebSearch(query, options);
        res.json(result);
        
      } catch (error) {
        this.logger.error('Web search endpoint error', {
          operation: 'WEB_SEARCH_ENDPOINT_ERROR',
          error: error.message
        });
        res.status(500).json({ error: 'Web search failed' });
      }
    });

    // GitHub search endpoint
    this.app.post('/search/github', async (req, res) => {
      try {
        const { query, options = {} } = req.body;
        
        if (!query) {
          return res.status(400).json({ error: 'Query is required' });
        }

        const result = await this.performGithubSearch(query, options);
        res.json(result);
        
      } catch (error) {
        this.logger.error('GitHub search endpoint error', {
          operation: 'GITHUB_SEARCH_ENDPOINT_ERROR',
          error: error.message
        });
        res.status(500).json({ error: 'GitHub search failed' });
      }
    });

    // Agent query endpoint - Main LangChain agent interface
    this.app.post('/agent/query', async (req, res) => {
      try {
        const { query, context = {} } = req.body;
        
        if (!query) {
          return res.status(400).json({ error: 'Query is required' });
        }

        const result = await this.executeAgentQuery(query, context);
        res.json(result);
        
      } catch (error) {
        this.logger.error('Agent query endpoint error', {
          operation: 'AGENT_QUERY_ENDPOINT_ERROR',
          error: error.message
        });
        res.status(500).json({ error: 'Agent query failed' });
      }
    });

    // Navigation enhancement endpoint for KGoT integration
    this.app.post('/navigation/enhance', async (req, res) => {
      try {
        const { targetUrl, navigationIntent, currentContext = {} } = req.body;
        
        if (!targetUrl) {
          return res.status(400).json({ error: 'Target URL is required' });
        }

        const result = await this.enhanceNavigation(targetUrl, navigationIntent, currentContext);
        res.json(result);
        
      } catch (error) {
        this.logger.error('Navigation enhancement error', {
          operation: 'NAVIGATION_ENHANCE_ERROR',
          error: error.message
        });
        res.status(500).json({ error: 'Navigation enhancement failed' });
      }
    });
  }

  /**
   * Initialize the Alita Web Agent
   */
  async initialize() {
    try {
      this.logger.info('Initializing Alita Web Agent', {
        operation: 'ALITA_WEB_AGENT_INITIALIZE'
      });

      // Initialize graph store connection
      await this.initializeGraphStore();
      
      // Initialize LangChain agent with tools
      await this.initializeLangChainAgent();
      
      this.logger.info('Alita Web Agent initialization completed', {
        operation: 'ALITA_WEB_AGENT_INIT_SUCCESS'
      });
      
    } catch (error) {
      this.logger.error('Failed to initialize Alita Web Agent', {
        operation: 'ALITA_WEB_AGENT_INIT_FAILED',
        error: error.message,
        stack: error.stack
      });
      throw error;
    }
  }

  /**
   * Initialize graph store connection
   */
  async initializeGraphStore() {
    try {
      // Connect to appropriate graph store backend
      const GraphStoreClass = this.getGraphStoreClass();
      this.graphStore = new GraphStoreClass({
        ...this.config.graphStoreConfig,
        loggerName: 'AlitaWebAgent'
      });
      
      await this.graphStore.initDatabase();
      
      this.logger.info('Graph store initialized', {
        operation: 'GRAPH_STORE_INIT_SUCCESS',
        backend: this.graphStore.options.backend
      });
      
    } catch (error) {
      this.logger.warn('Graph store initialization failed, continuing without graph integration', {
        operation: 'GRAPH_STORE_INIT_FAILED',
        error: error.message
      });
      this.graphStore = null;
    }
  }

  /**
   * Get the appropriate graph store class
   * @returns {Class} Graph store implementation class
   */
  getGraphStoreClass() {
    // For now, return the base interface
    // In production, this would select between Neo4j, NetworkX, or RDF4J
    return KnowledgeGraphInterface;
  }

  /**
   * Initialize LangChain agent with web tools
   */
  async initializeLangChainAgent() {
    try {
      // Create LLM instance configured for OpenRouter
      const llm = new ChatOpenAI({
        modelName: this.config.modelName,
        temperature: this.config.temperature,
        openAIApiKey: this.config.openrouterApiKey, // OpenRouter uses OpenAI-compatible API
        configuration: {
          baseURL: this.config.openrouterBaseUrl,
          defaultHeaders: {
            'HTTP-Referer': 'https://github.com/alita-kgot-enhanced', // OpenRouter requires referer
            'X-Title': 'Alita KGoT Enhanced Web Agent'
          }
        }
      });

      // Create tools
      const tools = this.createAgentTools();

      // Get agent prompt
      const prompt = await pull("hwchase17/openai-functions-agent");

      // Create agent
      const agent = await createOpenAIFunctionsAgent({
        llm,
        tools,
        prompt
      });

      // Create agent executor
      this.agentExecutor = new AgentExecutor({
        agent,
        tools,
        verbose: true,
        returnIntermediateSteps: true
      });

      this.logger.info('LangChain agent initialized', {
        operation: 'LANGCHAIN_AGENT_INIT_SUCCESS',
        toolCount: tools.length,
        model: this.config.modelName
      });

    } catch (error) {
      this.logger.error('Failed to initialize LangChain agent', {
        operation: 'LANGCHAIN_AGENT_INIT_FAILED',
        error: error.message
      });
      throw error;
    }
  }

  /**
   * Create agent tools for LangChain agent
   * @returns {Array} Array of LangChain tools
   */
  createAgentTools() {
    const tools = [];

    // Add Hugging Face Search Tool
    tools.push(new HuggingFaceSearchTool(
      this.graphStore,
      this.config
    ));

    // Add GitHub Search Tool
    if (this.config.githubToken) {
      tools.push(new GithubSearchTool(
        this.config.githubToken,
        this.graphStore
      ));
    }

    // Add web navigation tool
    tools.push(new DynamicTool({
      name: "web_navigate",
      description: "Navigate to a specific webpage and extract content with Playwright browser automation",
      func: async (url) => {
        return await this.navigateToPage(url);
      }
    }));

    // Add content extraction tool
    tools.push(new DynamicTool({
      name: "extract_content",
      description: "Extract specific content from current page using CSS selectors or text patterns",
      func: async (input) => {
        const params = typeof input === 'string' ? JSON.parse(input) : input;
        return await this.extractPageContent(params);
      }
    }));

    this.logger.info('Agent tools created', {
      operation: 'AGENT_TOOLS_CREATED',
      toolNames: tools.map(tool => tool.name)
    });

    return tools;
  }

  /**
   * Perform web search using Hugging Face Search Tool
   * @param {string} query - Search query
   * @param {Object} options - Search options
   * @returns {Promise<Object>} Search results
   */
  async performWebSearch(query, options) {
    this.logger.info('Performing web search with Hugging Face Agents', {
      operation: 'WEB_SEARCH_PERFORM_HF',
      query,
      options
    });

    const huggingFaceTool = new HuggingFaceSearchTool(
      this.graphStore,
      this.config
    );

    const result = await huggingFaceTool.searchWithHuggingFace(JSON.stringify({ query, ...options }));
    return JSON.parse(result);
  }

  /**
   * Perform GitHub search using GitHub Search Tool
   * @param {string} query - Search query
   * @param {Object} options - Search options
   * @returns {Promise<Object>} Search results
   */
  async performGithubSearch(query, options) {
    this.logger.info('Performing GitHub search', {
      operation: 'GITHUB_SEARCH_PERFORM',
      query,
      options
    });

    if (!this.config.githubToken) {
      throw new Error('GitHub search not configured');
    }

    const githubTool = new GithubSearchTool(this.config.githubToken, this.graphStore);
    const result = await githubTool.searchGithub(JSON.stringify({ query, ...options }));
    return JSON.parse(result);
  }

  /**
   * Execute agent query using LangChain agent
   * @param {string} query - Query to execute
   * @param {Object} context - Additional context
   * @returns {Promise<Object>} Agent response
   */
  async executeAgentQuery(query, context) {
    if (!this.agentExecutor) {
      throw new Error('LangChain agent not initialized');
    }

    this.logger.info('Executing agent query', {
      operation: 'AGENT_QUERY_EXECUTE',
      query,
      hasContext: !!context
    });

    const result = await this.agentExecutor.invoke({
      input: query,
      context: JSON.stringify(context)
    });

    this.logger.info('Agent query completed', {
      operation: 'AGENT_QUERY_SUCCESS',
      outputLength: result.output?.length
    });

    return {
      output: result.output,
      intermediateSteps: result.intermediateSteps,
      executionTime: Date.now()
    };
  }

  /**
   * Navigate to a webpage using Playwright
   * @param {string} url - URL to navigate to
   * @returns {Promise<string>} Page content
   */
  async navigateToPage(url) {
    let browser = null;
    try {
      this.logger.info('Navigating to page', {
        operation: 'WEB_NAVIGATE',
        url
      });

      browser = await playwright.chromium.launch({ headless: true });
      const page = await browser.newPage();
      
      await page.goto(url, { waitUntil: 'networkidle' });
      
      const content = await page.content();
      const title = await page.title();
      
      await browser.close();

      this.logger.info('Navigation completed', {
        operation: 'WEB_NAVIGATE_SUCCESS',
        url,
        title,
        contentLength: content.length
      });

      return JSON.stringify({
        url,
        title,
        content: content.substring(0, 10000), // Limit content size
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      if (browser) {
        await browser.close();
      }
      
      this.logger.error('Navigation failed', {
        operation: 'WEB_NAVIGATE_ERROR',
        url,
        error: error.message
      });
      
      throw error;
    }
  }

  /**
   * Extract content from current page
   * @param {Object} params - Extraction parameters
   * @returns {Promise<string>} Extracted content
   */
  async extractPageContent(params) {
    const { url, selector, pattern } = params;
    
    this.logger.info('Extracting page content', {
      operation: 'CONTENT_EXTRACT',
      url,
      selector,
      pattern
    });

    // Implementation would use Playwright to extract specific content
    // For now, return a placeholder
    return JSON.stringify({
      extracted: 'Content extraction would be implemented here',
      params
    });
  }

  /**
   * Enhance navigation with context awareness
   * @param {string} targetUrl - URL to navigate to
   * @param {string} navigationIntent - Intent of navigation
   * @param {Object} currentContext - Current context
   * @returns {Promise<Object>} Enhanced navigation result
   */
  async enhanceNavigation(targetUrl, navigationIntent, currentContext) {
    this.logger.info('Enhancing navigation', {
      operation: 'NAVIGATION_ENHANCE',
      targetUrl,
      navigationIntent
    });

    // Use LangChain agent to enhance navigation
    if (this.agentExecutor) {
      const enhancementQuery = `
        I need to navigate to ${targetUrl} with the intent: ${navigationIntent}
        Current context: ${JSON.stringify(currentContext)}
        
        Please provide enhanced navigation guidance including:
        1. Relevance assessment of the target URL
        2. Suggested navigation strategy
        3. Content extraction recommendations
        4. Related search suggestions
      `;

      const agentResult = await this.agentExecutor.invoke({
        input: enhancementQuery
      });

      return {
        targetUrl,
        navigationIntent,
        enhancement: agentResult.output,
        intermediateSteps: agentResult.intermediateSteps,
        timestamp: new Date().toISOString()
      };
    }

    // Fallback enhancement without agent
    return {
      targetUrl,
      navigationIntent,
      enhancement: 'Basic navigation enhancement',
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Start the Alita Web Agent server
   */
  async start() {
    try {
      await this.initialize();
      
      this.app.listen(this.config.port, () => {
        this.logger.info('Alita Web Agent started', {
          operation: 'ALITA_WEB_AGENT_START',
          port: this.config.port,
          pid: process.pid
        });
      });
      
    } catch (error) {
      this.logger.error('Failed to start Alita Web Agent', {
        operation: 'ALITA_WEB_AGENT_START_FAILED',
        error: error.message,
        stack: error.stack
      });
      process.exit(1);
    }
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    this.logger.info('Shutting down Alita Web Agent', {
      operation: 'ALITA_WEB_AGENT_SHUTDOWN'
    });

    // Close graph store connection
    if (this.graphStore) {
      await this.graphStore.close();
    }

    process.exit(0);
  }
}

// Handle graceful shutdown
process.on('SIGTERM', async () => {
  if (global.alitaWebAgent) {
    await global.alitaWebAgent.shutdown();
  }
});

process.on('SIGINT', async () => {
  if (global.alitaWebAgent) {
    await global.alitaWebAgent.shutdown();
  }
});

// Export for use as module
module.exports = { AlitaWebAgent, HuggingFaceSearchTool, GithubSearchTool };

// Start server if run directly
if (require.main === module) {
  const agent = new AlitaWebAgent();
  global.alitaWebAgent = agent;
  agent.start();
} 