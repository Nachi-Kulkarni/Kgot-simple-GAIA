/**
 * Alita Web Agent - Simplified Implementation
 * 
 * Basic web agent service with health checks and API endpoints
 * 
 * @module AlitaWebAgent
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const axios = require('axios');
const { JSDOM } = require('jsdom');
const { Octokit } = require('@octokit/rest');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
console.log('[WEB-AGENT] Configuring security middleware (Helmet)...');
app.use(helmet());
console.log('[WEB-AGENT] Configuring CORS middleware...');
app.use(cors());
console.log('[WEB-AGENT] Configuring JSON body parser (limit: 10mb)...');
app.use(express.json({ limit: '10mb' }));
console.log('[WEB-AGENT] Configuring URL-encoded body parser...');
app.use(express.urlencoded({ extended: true }));

// Rate limiting
console.log('[WEB-AGENT] Configuring rate limiting (100 requests per 15 minutes)...');
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use(limiter);

// Health check endpoint
app.get('/health', (req, res) => {
  console.log('[WEB-AGENT] Health check requested from:', req.ip);
  const response = {
    status: 'healthy',
    service: 'alita-web-agent',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  };
  console.log('[WEB-AGENT] Health check response:', response);
  res.status(200).json(response);
});

// Status endpoint
app.get('/status', (req, res) => {
  console.log('[WEB-AGENT] Status check requested from:', req.ip);
  const memUsage = process.memoryUsage();
  const response = {
    service: 'Alita Web Agent',
    status: 'running',
    port: PORT,
    endpoints: {
      health: '/health',
      status: '/status',
      search: '/api/search',
      github: '/api/github/search'
    },
    memory: {
      rss: `${Math.round(memUsage.rss / 1024 / 1024)}MB`,
      heapUsed: `${Math.round(memUsage.heapUsed / 1024 / 1024)}MB`,
      heapTotal: `${Math.round(memUsage.heapTotal / 1024 / 1024)}MB`
    },
    timestamp: new Date().toISOString()
  };
  console.log('[WEB-AGENT] Status response:', response);
  res.status(200).json(response);
});

// Basic web search endpoint
app.post('/api/search', async (req, res) => {
  try {
    console.log('[WEB-AGENT] Web search request received from:', req.ip);
    const { query, type = 'web' } = req.body;
    console.log('[WEB-AGENT] Search parameters - Query:', query, 'Type:', type);
    
    if (!query) {
      console.log('[WEB-AGENT] Search request rejected - missing query parameter');
      return res.status(400).json({ error: 'Query parameter is required' });
    }

    console.log('[WEB-AGENT] Performing web search for query:', query);
    // Simple web search implementation
    const searchResults = await performWebSearch(query);
    console.log('[WEB-AGENT] Web search completed - Found', searchResults.length, 'results');
    
    const response = {
      query,
      type,
      results: searchResults,
      timestamp: new Date().toISOString()
    };
    console.log('[WEB-AGENT] Web search response sent successfully');
    res.json(response);
  } catch (error) {
    console.error('[WEB-AGENT] Web search error:', error.message);
    res.status(500).json({ error: 'Search failed', message: error.message });
  }
});

// GitHub search endpoint
app.post('/api/github/search', async (req, res) => {
  try {
    console.log('[WEB-AGENT] GitHub search request received from:', req.ip);
    const { query, type = 'repositories' } = req.body;
    console.log('[WEB-AGENT] GitHub search parameters - Query:', query, 'Type:', type);
    
    if (!query) {
      console.log('[WEB-AGENT] GitHub search request rejected - missing query parameter');
      return res.status(400).json({ error: 'Query parameter is required' });
    }

    console.log('[WEB-AGENT] Initializing GitHub API client...');
    const octokit = new Octokit();
    console.log('[WEB-AGENT] Searching GitHub repositories for:', query);
    const searchResults = await octokit.rest.search.repos({
      q: query,
      sort: 'stars',
      order: 'desc',
      per_page: 10
    });
    console.log('[WEB-AGENT] GitHub search completed - Found', searchResults.data.items.length, 'repositories');
    
    const response = {
      query,
      type,
      results: searchResults.data.items.map(repo => ({
        name: repo.name,
        full_name: repo.full_name,
        description: repo.description,
        stars: repo.stargazers_count,
        url: repo.html_url,
        language: repo.language
      })),
      timestamp: new Date().toISOString()
    };
    console.log('[WEB-AGENT] GitHub search response sent successfully');
    res.json(response);
  } catch (error) {
    console.error('[WEB-AGENT] GitHub search error:', error.message);
    res.status(500).json({ error: 'GitHub search failed', message: error.message });
  }
});

// Simple web search function
async function performWebSearch(query) {
  try {
    console.log('[WEB-AGENT] Starting web search execution for query:', query);
    // This is a simplified implementation
    // In a real scenario, you would integrate with a proper search API
    const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
    console.log('[WEB-AGENT] Search URL constructed:', searchUrl);
    
    console.log('[WEB-AGENT] Making HTTP request to search engine...');
    const response = await axios.get(searchUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      },
      timeout: 10000
    });
    console.log('[WEB-AGENT] HTTP response received, status:', response.status);
    
    // Parse HTML content with jsdom
    console.log('[WEB-AGENT] Parsing HTML content with JSDOM...');
    const dom = new JSDOM(response.data);
    const document = dom.window.document;
    const results = [];
    
    // Extract search results (simplified example)
    console.log('[WEB-AGENT] Extracting search results from HTML...');
    const searchResults = document.querySelectorAll('.g');
    console.log('[WEB-AGENT] Found', searchResults.length, 'potential search result elements');
    
    searchResults.forEach((element, i) => {
      if (i < 5) { // Limit to 5 results
        const titleElement = element.querySelector('h3');
        const linkElement = element.querySelector('a');
        const snippetElement = element.querySelector('.VwiC3b');
        
        const title = titleElement ? titleElement.textContent : '';
        const link = linkElement ? linkElement.href : '';
        const snippet = snippetElement ? snippetElement.textContent : '';
        
        if (title && link) {
          results.push({
            title,
            link: link.startsWith('/url?q=') ? link.substring(7).split('&')[0] : link,
            snippet
          });
          console.log('[WEB-AGENT] Extracted result', i + 1, ':', title.substring(0, 50) + '...');
        }
      }
    });
    
    console.log('[WEB-AGENT] Web search execution completed successfully, returning', results.length, 'results');
    return results;
  } catch (error) {
    console.error('[WEB-AGENT] Web search execution error:', error.message);
    console.log('[WEB-AGENT] Returning fallback search result due to error');
    return [{
      title: 'Search temporarily unavailable',
      link: '#',
      snippet: 'Please try again later'
    }];
  }
}

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('[WEB-AGENT] Unhandled error occurred:', error.message);
  console.error('[WEB-AGENT] Error stack:', error.stack);
  console.log('[WEB-AGENT] Sending 500 error response to client');
  res.status(500).json({
    error: 'Internal server error',
    message: error.message
  });
});

// 404 handler
app.use((req, res) => {
  console.log('[WEB-AGENT] 404 - Endpoint not found:', req.method, req.path, 'from IP:', req.ip);
  res.status(404).json({
    error: 'Not found',
    message: 'The requested endpoint does not exist'
  });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log('='.repeat(60));
  console.log('[WEB-AGENT] 🚀 Alita Web Agent Service Started Successfully');
  console.log('[WEB-AGENT] Server Details:');
  console.log(`[WEB-AGENT]   - Port: ${PORT}`);
  console.log(`[WEB-AGENT]   - Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`[WEB-AGENT]   - Process ID: ${process.pid}`);
  console.log('[WEB-AGENT] Available Endpoints:');
  console.log(`[WEB-AGENT]   - Health Check: http://localhost:${PORT}/health`);
  console.log(`[WEB-AGENT]   - Status: http://localhost:${PORT}/status`);
  console.log(`[WEB-AGENT]   - Web Search: POST http://localhost:${PORT}/api/search`);
  console.log(`[WEB-AGENT]   - GitHub Search: POST http://localhost:${PORT}/api/github/search`);
  console.log('[WEB-AGENT] Service ready to accept requests');
  console.log('='.repeat(60));
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('[WEB-AGENT] 🛑 SIGTERM received, initiating graceful shutdown...');
  console.log('[WEB-AGENT] Cleaning up resources and closing connections...');
  console.log('[WEB-AGENT] Alita Web Agent Service stopped');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('[WEB-AGENT] 🛑 SIGINT received, initiating graceful shutdown...');
  console.log('[WEB-AGENT] Cleaning up resources and closing connections...');
  console.log('[WEB-AGENT] Alita Web Agent Service stopped');
  process.exit(0);
});

process.on('uncaughtException', (error) => {
  console.error('[WEB-AGENT] 💥 Uncaught Exception:', error.message);
  console.error('[WEB-AGENT] Stack trace:', error.stack);
  console.log('[WEB-AGENT] Shutting down due to uncaught exception...');
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('[WEB-AGENT] 💥 Unhandled Promise Rejection at:', promise);
  console.error('[WEB-AGENT] Reason:', reason);
  console.log('[WEB-AGENT] Shutting down due to unhandled promise rejection...');
  process.exit(1);
});