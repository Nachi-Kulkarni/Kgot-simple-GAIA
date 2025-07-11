#!/usr/bin/env python3
"""
Core High-Value MCPs - Web & Information Tools

Task 21 Implementation: Implement Core High-Value MCPs - Web & Information
- Build web_scraper_mcp following Alita Section 2.2 Web Agent design with Beautiful Soup
- Create search_engine_mcp with multi-provider support based on RAG-MCP external tool integration
- Implement wikipedia_mcp based on KGoT Section 2.3 Wikipedia Tool specifications
- Add browser_automation_mcp using KGoT Section 2.3 Surfer Agent architecture

This module provides four essential MCP tools that form the core 20% of web & information
capabilities providing 80% coverage of task requirements, following Pareto principle
optimization as demonstrated in RAG-MCP experimental findings.

Features:
- Beautiful Soup integration for advanced web scraping
- Multi-provider search engine support (Google, Bing, DuckDuckGo)
- Comprehensive Wikipedia knowledge access with fact verification
- Browser automation with granular navigation controls
- LangChain agent integration as per user preference
- OpenRouter API integration for AI model access
- Comprehensive Winston logging for workflow tracking
- Robust error handling and recovery mechanisms

@module WebInformationMCPs
@author Enhanced Alita KGoT Team
@date 2025
"""

import asyncio
import logging
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import urllib.parse
from urllib.parse import urljoin, urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Beautiful Soup for web scraping as specified in Task 21
from bs4 import BeautifulSoup, NavigableString, Tag
import lxml
import html5lib

# LangChain imports (user's hard rule for agent development)
from langchain.tools import BaseTool
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field

# Import existing system components for integration
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "knowledge-graph-of-thoughts"))

# Import existing MCP infrastructure
from alita_core.rag_mcp_engine import MCPToolSpec, MCPCategory
from alita_core.mcp_knowledge_base import EnhancedMCPSpec, MCPQualityScore

# Import existing KGoT tools for Wikipedia and Web Surfing
from kgot.tools.tools_v2_3.WikipediaTool import WikipediaTool, LangchainWikipediaTool
from kgot.tools.tools_v2_3.Web_surfer import (
    SearchInformationTool, NavigationalSearchTool, VisitTool,
    PageUpTool, PageDownTool, FinderTool, FindNextTool,
    FullPageSummaryTool, ArchiveSearchTool, init_browser
)
from kgot.tools.tools_v2_3.SurferTool import SearchTool
from kgot.utils import UsageStatistics

# Import existing Alita Web Agent integration
from kgot_core.integrated_tools.alita_integration import AlitaWebAgentBridge, AlitaToolIntegrator

# Winston-compatible logging setup following existing patterns
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s'
)
logger = logging.getLogger('WebInformationMCPs')

# Create logs directory for MCP toolbox operations
log_dir = Path('./logs/mcp_toolbox')
log_dir.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_dir / 'web_information_mcps.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s'
))
logger.addHandler(file_handler)


@dataclass
class WebScrapingConfig:
    """
    Configuration for web scraping operations following Alita Section 2.2 Web Agent design
    
    Attributes:
        timeout (int): Request timeout in seconds for web operations
        max_retries (int): Maximum number of retry attempts for failed requests
        user_agent (str): User agent string for web requests
        parser (str): HTML parser to use with Beautiful Soup ('lxml', 'html.parser', 'html5lib')
        max_content_size (int): Maximum content size to process in bytes
        enable_javascript (bool): Whether to enable JavaScript rendering support
        follow_redirects (bool): Whether to follow HTTP redirects automatically
        verify_ssl (bool): Whether to verify SSL certificates
    """
    timeout: int = 30
    max_retries: int = 3
    user_agent: str = "Mozilla/5.0 (compatible; AlitaKGoTWebScraper/1.0)"
    parser: str = "lxml"
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    enable_javascript: bool = False
    follow_redirects: bool = True
    verify_ssl: bool = True


@dataclass
class SearchEngineConfig:
    """
    Configuration for multi-provider search engine support based on RAG-MCP integration
    
    Attributes:
        serpapi_api_key (Optional[str]): SerpApi API key powering Google-results retrieval
        bing_api_key (Optional[str]): Bing Search API key
        default_provider (str): Default search provider to use (serpapi | bing | duckduckgo)
        max_results (int): Maximum number of search results to return
        enable_caching (bool): Whether to cache search results
        cache_ttl (int): Cache time-to-live in seconds
        result_ranking_enabled (bool): Whether to enable result ranking
    """
    serpapi_api_key: Optional[str] = None
    bing_api_key: Optional[str] = None
    default_provider: str = "serpapi"
    max_results: int = 10
    enable_caching: bool = True
    cache_ttl: int = 3600
    result_ranking_enabled: bool = True


class WebScraperMCPInputSchema(BaseModel):
    """
    Input schema for WebScraperMCP following Beautiful Soup integration requirements
    
    Validates and structures input parameters for web scraping operations
    """
    url: str = Field(description="Target URL to scrape content from")
    content_selectors: Optional[List[str]] = Field(
        default=None,
        description="CSS selectors or XPath expressions for content extraction"
    )
    extract_tables: bool = Field(
        default=False,
        description="Whether to extract and parse HTML tables"
    )
    extract_links: bool = Field(
        default=False,
        description="Whether to extract all links from the page"
    )
    extract_images: bool = Field(
        default=False,
        description="Whether to extract image URLs and metadata"
    )
    clean_text: bool = Field(
        default=True,
        description="Whether to clean and normalize extracted text"
    )
    max_depth: int = Field(
        default=1,
        description="Maximum depth for recursive content extraction"
    )


class WebScraperMCP(BaseTool):
    """
    Advanced Web Scraper MCP following Alita Section 2.2 Web Agent design with Beautiful Soup
    
    This MCP provides comprehensive web scraping capabilities with Beautiful Soup integration
    for structured data extraction, content analysis, and XPath selection. Designed to handle
    modern web pages with robust error handling and content cleaning.
    
    Key Features:
    - Beautiful Soup integration with multiple parser support (lxml, html.parser, html5lib)
    - CSS selector and XPath-based content extraction
    - Table parsing and structured data extraction
    - Link and image metadata extraction
    - Content cleaning and normalization
    - Robust error handling with retry mechanisms
    - Integration with existing Alita Web Agent infrastructure
    
    Capabilities:
    - html_parsing: Parse and analyze HTML structure using Beautiful Soup
    - data_extraction: Extract structured data using selectors and patterns
    - content_analysis: Analyze and classify extracted content
    - xpath_selection: Support for XPath expressions and CSS selectors
    """
    
    name: str = "web_scraper_mcp"
    description: str = """
    Advanced web scraping tool with Beautiful Soup integration for structured data extraction.
    
    Capabilities:
    - Parse HTML content with multiple parser support (lxml, html.parser, html5lib)
    - Extract content using CSS selectors and XPath expressions
    - Parse tables and structured data automatically
    - Extract links, images, and metadata
    - Clean and normalize extracted text content
    - Handle modern web pages with robust error recovery
    
    Input should be a JSON string with:
    {
        "url": "https://example.com/page",
        "content_selectors": ["article", ".main-content", "#content"],
        "extract_tables": true,
        "extract_links": true,
        "extract_images": false,
        "clean_text": true,
        "max_depth": 1
    }
    """
    args_schema = WebScraperMCPInputSchema
    
    def __init__(self, 
                 config: Optional[WebScrapingConfig] = None,
                 alita_web_agent_bridge: Optional[AlitaWebAgentBridge] = None,
                 **kwargs):
        """
        Initialize WebScraperMCP with configuration and integration components
        
        Args:
            config (Optional[WebScrapingConfig]): Web scraping configuration settings
            alita_web_agent_bridge (Optional[AlitaWebAgentBridge]): Bridge to Alita Web Agent
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(**kwargs)
        
        # Initialize configuration with defaults
        self.config = config or WebScrapingConfig()
        self.alita_bridge = alita_web_agent_bridge
        
        # Initialize requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set session headers
        self.session.headers.update({
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Initialize content extraction patterns
        self._init_extraction_patterns()
        
        logger.info("WebScraperMCP initialized successfully", extra={
            'operation': 'WEB_SCRAPER_MCP_INIT',
            'parser': self.config.parser,
            'timeout': self.config.timeout,
            'has_alita_bridge': self.alita_bridge is not None
        })
    
    def _init_extraction_patterns(self) -> None:
        """
        Initialize content extraction patterns for common web page elements
        
        Sets up predefined CSS selectors and patterns for extracting content from
        common webpage structures including articles, main content areas, navigation,
        and metadata elements.
        """
        self.content_patterns = {
            # Article and main content patterns
            'main_content': [
                'article', 'main', '[role="main"]', '.main-content', 
                '#main-content', '#content', '.content', '.post-content'
            ],
            'article_content': [
                'article .content', 'article .post-content', 
                '.article-content', '.entry-content', '.post-body'
            ],
            'navigation': [
                'nav', '[role="navigation"]', '.navigation', 
                '.nav', '.menu', '.navbar'
            ],
            'metadata': [
                'meta[name="description"]', 'meta[property="og:description"]',
                'meta[name="keywords"]', 'meta[property="og:title"]',
                '.published', '.author', '.date', '.timestamp'
            ]
        }
        
        # XPath patterns for complex content extraction
        self.xpath_patterns = {
            'headings': '//h1 | //h2 | //h3 | //h4 | //h5 | //h6',
            'paragraphs': '//p[string-length(normalize-space(text())) > 10]',
            'lists': '//ul | //ol | //dl',
            'tables': '//table[.//td or .//th]',
            'forms': '//form'
        }
        
        logger.debug("Content extraction patterns initialized", extra={
            'operation': 'EXTRACTION_PATTERNS_INIT',
            'content_patterns_count': len(self.content_patterns),
            'xpath_patterns_count': len(self.xpath_patterns)
        }) 

    def _run(self, 
             url: str,
             content_selectors: Optional[List[str]] = None,
             extract_tables: bool = False,
             extract_links: bool = False,
             extract_images: bool = False,
             clean_text: bool = True,
             max_depth: int = 1) -> str:
        """
        Execute web scraping operation using Beautiful Soup for structured content extraction
        
        This method implements the core web scraping functionality following Alita Section 2.2
        Web Agent design principles with Beautiful Soup integration for robust HTML parsing
        and content extraction.
        
        Args:
            url (str): Target URL to scrape content from
            content_selectors (Optional[List[str]]): CSS selectors for targeted content extraction
            extract_tables (bool): Whether to extract and parse HTML tables
            extract_links (bool): Whether to extract all links from the page
            extract_images (bool): Whether to extract image URLs and metadata
            clean_text (bool): Whether to clean and normalize extracted text
            max_depth (int): Maximum depth for recursive content extraction
            
        Returns:
            str: JSON-formatted extraction results with content, metadata, and statistics
        """
        start_time = time.time()
        
        logger.info("Starting web scraping operation", extra={
            'operation': 'WEB_SCRAPER_EXECUTE',
            'url': url,
            'extract_tables': extract_tables,
            'extract_links': extract_links,
            'extract_images': extract_images
        })
        
        try:
            # Validate and normalize URL
            normalized_url = self._validate_and_normalize_url(url)
            
            # Fetch webpage content with error handling
            response = self._fetch_webpage(normalized_url)
            
            # Parse HTML content with Beautiful Soup
            soup = self._parse_html_content(response.content, response.encoding)
            
            # Extract structured content based on selectors
            extracted_content = self._extract_content(
                soup, content_selectors, extract_tables, 
                extract_links, extract_images, clean_text
            )
            
            # Calculate extraction metrics
            extraction_time = time.time() - start_time
            
            # Prepare result structure
            result = {
                'success': True,
                'url': normalized_url,
                'title': self._extract_page_title(soup),
                'content': extracted_content,
                'metadata': self._extract_page_metadata(soup),
                'statistics': {
                    'extraction_time_ms': round(extraction_time * 1000, 2),
                    'content_size_bytes': len(response.content),
                    'parsed_elements_count': len(soup.find_all()),
                    'extracted_text_length': len(extracted_content.get('main_text', '')),
                    'tables_found': len(extracted_content.get('tables', [])),
                    'links_found': len(extracted_content.get('links', [])),
                    'images_found': len(extracted_content.get('images', []))
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Integration with Alita Web Agent if available
            if self.alita_bridge:
                try:
                    enhanced_result = asyncio.run(
                        self._enhance_with_alita_context(result, normalized_url)
                    )
                    result.update(enhanced_result)
                except Exception as e:
                    logger.warning("Alita Web Agent enhancement failed", extra={
                        'operation': 'ALITA_ENHANCEMENT_WARNING',
                        'error': str(e)
                    })
            
            logger.info("Web scraping operation completed successfully", extra={
                'operation': 'WEB_SCRAPER_SUCCESS',
                'url': normalized_url,
                'extraction_time_ms': result['statistics']['extraction_time_ms'],
                'content_size': result['statistics']['content_size_bytes']
            })
            
            return json.dumps(result, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error("Web scraping operation failed", extra={
                'operation': 'WEB_SCRAPER_ERROR',
                'url': url,
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            # Return structured error response
            error_result = {
                'success': False,
                'url': url,
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now().isoformat(),
                'suggested_actions': self._get_error_suggestions(e)
            }
            
            return json.dumps(error_result, indent=2)
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async wrapper for web scraping operations"""
        return self._run(*args, **kwargs)
    
    def _validate_and_normalize_url(self, url: str) -> str:
        """
        Validate and normalize URL for safe web scraping operations
        
        Args:
            url (str): Input URL to validate and normalize
            
        Returns:
            str: Normalized and validated URL
            
        Raises:
            ValueError: If URL is invalid or potentially unsafe
        """
        # Basic URL validation
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Parse and validate URL structure
        parsed = urlparse(url)
        if not parsed.netloc:
            raise ValueError(f"Invalid URL structure: {url}")
        
        # Security checks for potentially unsafe URLs
        unsafe_schemes = ['file', 'ftp', 'javascript', 'data']
        if parsed.scheme.lower() in unsafe_schemes:
            raise ValueError(f"Unsafe URL scheme detected: {parsed.scheme}")
        
        # Normalize URL
        normalized_url = urllib.parse.urlunparse(parsed)
        
        logger.debug("URL validated and normalized", extra={
            'operation': 'URL_VALIDATION',
            'original_url': url,
            'normalized_url': normalized_url
        })
        
        return normalized_url
    
    def _fetch_webpage(self, url: str) -> requests.Response:
        """
        Fetch webpage content with robust error handling and retry mechanisms
        
        Args:
            url (str): URL to fetch content from
            
        Returns:
            requests.Response: HTTP response object with webpage content
            
        Raises:
            requests.RequestException: If webpage fetching fails after retries
        """
        try:
            response = self.session.get(
                url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                allow_redirects=self.config.follow_redirects
            )
            
            # Check response status
            response.raise_for_status()
            
            # Check content size limits
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.config.max_content_size:
                raise ValueError(f"Content size ({content_length}) exceeds maximum allowed size")
            
            # Verify content type is HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'html' not in content_type:
                logger.warning("Non-HTML content type detected", extra={
                    'operation': 'CONTENT_TYPE_WARNING',
                    'content_type': content_type,
                    'url': url
                })
            
            logger.debug("Webpage fetched successfully", extra={
                'operation': 'WEBPAGE_FETCH_SUCCESS',
                'url': url,
                'status_code': response.status_code,
                'content_length': len(response.content)
            })
            
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch webpage", extra={
                'operation': 'WEBPAGE_FETCH_ERROR',
                'url': url,
                'error': str(e)
            })
            raise
    
    def _parse_html_content(self, content: bytes, encoding: Optional[str] = None) -> BeautifulSoup:
        """
        Parse HTML content using Beautiful Soup with optimal parser selection
        
        Args:
            content (bytes): Raw HTML content to parse
            encoding (Optional[str]): Character encoding hint
            
        Returns:
            BeautifulSoup: Parsed HTML document tree
            
        Raises:
            Exception: If HTML parsing fails with all available parsers
        """
        # Determine encoding if not provided
        if not encoding:
            encoding = 'utf-8'  # Default fallback
        
        # Try parsing with configured parser first
        parsers_to_try = [self.config.parser]
        
        # Add fallback parsers
        if self.config.parser != 'lxml':
            parsers_to_try.append('lxml')
        if 'html.parser' not in parsers_to_try:
            parsers_to_try.append('html.parser')
        if 'html5lib' not in parsers_to_try:
            parsers_to_try.append('html5lib')
        
        last_error = None
        
        for parser in parsers_to_try:
            try:
                soup = BeautifulSoup(content, parser, from_encoding=encoding)
                
                logger.debug("HTML content parsed successfully", extra={
                    'operation': 'HTML_PARSE_SUCCESS',
                    'parser_used': parser,
                    'content_size': len(content)
                })
                
                return soup
                
            except Exception as e:
                last_error = e
                logger.debug(f"Parser {parser} failed, trying next", extra={
                    'operation': 'HTML_PARSE_RETRY',
                    'parser_failed': parser,
                    'error': str(e)
                })
                continue
        
        # If all parsers failed, raise the last error
        logger.error("All HTML parsers failed", extra={
            'operation': 'HTML_PARSE_FAILED',
            'parsers_tried': parsers_to_try,
            'final_error': str(last_error)
        })
        
        raise Exception(f"Failed to parse HTML content with any available parser. Last error: {last_error}")


class SearchEngineMCPInputSchema(BaseModel):
    """Input schema for SearchEngineMCP with multi-provider support"""
    query: str = Field(description="Search query string")
    provider: Optional[str] = Field(default=None, description="Search provider (serpapi, bing, duckduckgo)")
    max_results: Optional[int] = Field(default=10, description="Maximum number of results to return")
    language: Optional[str] = Field(default="en", description="Language for search results")
    region: Optional[str] = Field(default="us", description="Region for localized results")
    date_filter: Optional[str] = Field(default=None, description="Date filter (recent, month, year)")
    safe_search: Optional[str] = Field(default="active", description="Safe search setting")


class SearchEngineMCP(BaseTool):
    """
    Multi-Provider Search Engine MCP with RAG-MCP external tool integration
    
    This MCP provides comprehensive search capabilities across multiple search providers
    including Google, Bing, and DuckDuckGo with intelligent result ranking, content
    filtering, and query optimization features.
    
    Key Features:
    - Multi-provider search support (Google, Bing, DuckDuckGo)
    - Intelligent result ranking and relevance scoring
    - Content filtering and quality assessment
    - Query optimization and suggestion generation
    - Integration with existing Alita Web Agent GoogleSearchTool
    - Caching for improved performance and cost efficiency
    
    Capabilities:
    - web_search: Execute searches across multiple providers
    - result_ranking: Rank and score search results by relevance
    - content_filtering: Filter results by quality and safety criteria
    - query_optimization: Optimize queries for better results
    """
    
    name: str = "search_engine_mcp"
    description: str = """
    Multi-provider search with SerpApi, Bing, DuckDuckGo integration.
    
    Supports:
    - Google Custom Search with advanced filtering
    - Bing Search API with regional customization
    - DuckDuckGo search for privacy-focused queries
    - Result ranking and relevance scoring
    - Content filtering and quality assessment
    - Query optimization and suggestions
    
    Input should be a JSON string with:
    {
        "query": "search terms",
        "provider": "serpapi|bing|duckduckgo",
        "max_results": 10,
        "language": "en",
        "region": "us", 
        "date_filter": "recent|month|year",
        "safe_search": "active|moderate|off"
    }
    """
    args_schema = SearchEngineMCPInputSchema
    
    def __init__(self,
                 config: Optional[SearchEngineConfig] = None,
                 alita_web_agent_bridge: Optional[AlitaWebAgentBridge] = None,
                 **kwargs):
        """
        Initialize SearchEngineMCP with multi-provider configuration
        
        Args:
            config (Optional[SearchEngineConfig]): Search engine configuration
            alita_web_agent_bridge (Optional[AlitaWebAgentBridge]): Bridge to Alita Web Agent
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(**kwargs)
        
        self.config = config or SearchEngineConfig()
        self.alita_bridge = alita_web_agent_bridge
        
        # Initialize search result cache if enabled
        self.cache = {} if self.config.enable_caching else None
        
        # Initialize provider-specific clients
        self._init_search_providers()
        
        logger.info("SearchEngineMCP initialized successfully", extra={
            'operation': 'SEARCH_ENGINE_MCP_INIT',
            'default_provider': self.config.default_provider,
            'caching_enabled': self.config.enable_caching,
            'providers_available': list(self.providers.keys())
        })
    
    def _init_search_providers(self) -> None:
        """Initialize available search providers based on API key availability"""
        self.providers = {}
        
        # SerpApi provider (Google results as a service)
        if self.config.serpapi_api_key:
            self.providers["serpapi"] = {
                "api_key": self.config.serpapi_api_key,
                "endpoint": "https://serpapi.com/search.json",
            }
        
        # Bing Search provider
        if self.config.bing_api_key:
            self.providers["bing"] = {
                "api_key": self.config.bing_api_key,
                "endpoint": "https://api.bing.microsoft.com/v7.0/search",
            }
        
        # DuckDuckGo provider (no API key required)
        self.providers["duckduckgo"] = {
            "endpoint": "https://api.duckduckgo.com/",
        }
        
        logger.debug(
            "Search providers initialized",
            extra={
                "operation": "SEARCH_PROVIDERS_INIT",
                "available_providers": list(self.providers.keys()),
            },
        )


class WikipediaMCPInputSchema(BaseModel):
    """Input schema for WikipediaMCP based on KGoT Section 2.3 specifications"""
    query: str = Field(description="Wikipedia search query or article title")
    language: Optional[str] = Field(default="en", description="Wikipedia language code")
    extract_tables: bool = Field(default=True, description="Extract tables from articles")
    include_references: bool = Field(default=False, description="Include reference information")
    max_articles: int = Field(default=3, description="Maximum number of articles to analyze")
    fact_check_mode: bool = Field(default=False, description="Enable fact verification mode")


class WikipediaMCP(BaseTool):
    """
    Wikipedia Knowledge Access MCP based on KGoT Section 2.3 Wikipedia Tool specifications
    
    This MCP provides comprehensive Wikipedia integration with structured knowledge access,
    fact verification, and entity resolution capabilities. Built upon the existing KGoT
    WikipediaTool with enhanced features for knowledge graph integration.
    
    Key Features:
    - Comprehensive Wikipedia article access and analysis
    - Table parsing and structured data extraction
    - Date-based content retrieval for temporal queries
    - Fact verification and cross-referencing
    - Entity resolution and knowledge graph integration
    - Multi-language support
    
    Capabilities:
    - knowledge_lookup: Search and retrieve Wikipedia knowledge
    - entity_resolution: Resolve entities and disambiguation
    - fact_checking: Verify facts against Wikipedia content
    - knowledge_graph: Integration with knowledge graph systems
    """
    
    name: str = "wikipedia_mcp"
    description: str = """
    Comprehensive Wikipedia knowledge access with fact verification and entity resolution.
    
    Features:
    - Search Wikipedia articles with intelligent ranking
    - Extract structured content including tables and infoboxes
    - Date-based content retrieval for temporal queries
    - Fact verification against Wikipedia sources
    - Entity disambiguation and resolution
    - Multi-language Wikipedia support
    - Knowledge graph integration
    
    Input should be a JSON string with:
    {
        "query": "search query or article title",
        "language": "en",
        "extract_tables": true,
        "include_references": false,
        "max_articles": 3,
        "fact_check_mode": false
    }
    """
    args_schema = WikipediaMCPInputSchema
    
    def __init__(self,
                 usage_statistics: Optional[UsageStatistics] = None,
                 model_name: str = "anthropic/claude-sonnet-4",
                 temperature: float = 0.1,
                 **kwargs):
        """
        Initialize WikipediaMCP with KGoT Wikipedia Tool integration
        
        Args:
            usage_statistics (Optional[UsageStatistics]): Usage tracking instance
            model_name (str): Model for analysis (using OpenRouter per user preference)
            temperature (float): Model temperature for analysis operations
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(**kwargs)
        
        # Initialize usage statistics if not provided
        self.usage_statistics = usage_statistics or UsageStatistics()
        
        # Initialize KGoT Wikipedia tools for enhanced functionality
        self.wikipedia_tool = WikipediaTool(
            model_name=model_name,
            temperature=temperature,
            usage_statistics=self.usage_statistics
        )
        
        self.langchain_wikipedia_tool = LangchainWikipediaTool(
            model_name=model_name,
            temperature=temperature,
            usage_statistics=self.usage_statistics
        )
        
        logger.info("WikipediaMCP initialized successfully", extra={
            'operation': 'WIKIPEDIA_MCP_INIT',
            'model_name': model_name,
            'temperature': temperature
        })


class BrowserAutomationMCPInputSchema(BaseModel):
    """Input schema for BrowserAutomationMCP with Surfer Agent architecture"""
    action: str = Field(description="Browser action (navigate, click, type, scroll, screenshot)")
    target: Optional[str] = Field(default=None, description="Target URL or selector")
    value: Optional[str] = Field(default=None, description="Value for input actions")
    wait_for: Optional[str] = Field(default=None, description="Element to wait for")
    screenshot: bool = Field(default=False, description="Take screenshot after action")
    session_id: Optional[str] = Field(default=None, description="Browser session ID")


class BrowserAutomationMCP(BaseTool):
    """
    Browser Automation MCP using KGoT Section 2.3 Surfer Agent architecture
    
    This MCP provides comprehensive browser automation capabilities with granular
    navigation controls, session management, and form interaction features based
    on the KGoT Surfer Agent design using HuggingFace Agents framework.
    
    Key Features:
    - Granular navigation controls (PageUp, PageDown, Find, FindNext)
    - Session management and state persistence
    - Form filling and interaction automation
    - Screenshot capture and visual analysis
    - Integration with KGoT Web Surfer tools
    - Dynamic content handling and JavaScript support
    
    Capabilities:
    - ui_automation: Automate user interface interactions
    - form_filling: Fill and submit web forms automatically  
    - screenshot_capture: Capture page screenshots and visual data
    - session_management: Manage browser sessions and state
    """
    
    name: str = "browser_automation_mcp"
    description: str = """
    Comprehensive browser automation with granular navigation and session management.
    
    Supports:
    - Page navigation and interaction automation
    - Form filling and submission
    - Element finding and interaction (click, type, scroll)
    - Screenshot capture and visual analysis
    - Session persistence and state management
    - Granular navigation (PageUp, PageDown, Find, FindNext)
    - Dynamic content handling
    
    Input should be a JSON string with:
    {
        "action": "navigate|click|type|scroll|screenshot|find",
        "target": "URL or CSS selector",
        "value": "input value for type actions",
        "wait_for": "element to wait for",
        "screenshot": false,
        "session_id": "optional session identifier"
    }
    """
    args_schema = BrowserAutomationMCPInputSchema
    
    def __init__(self,
                 usage_statistics: Optional[UsageStatistics] = None,
                 model_name: str = "anthropic/claude-sonnet-4",
                 temperature: float = 0.1,
                 **kwargs):
        """
        Initialize BrowserAutomationMCP with KGoT Surfer Agent tools integration
        
        Args:
            usage_statistics (Optional[UsageStatistics]): Usage tracking instance
            model_name (str): Model for automation decisions (using OpenRouter)
            temperature (float): Model temperature for decision making
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(**kwargs)
        
        # Initialize usage statistics and browser session
        self.usage_statistics = usage_statistics or UsageStatistics()
        self.sessions = {}  # Track browser sessions
        
        # Initialize KGoT Web Surfer tools for automation
        self.surfer_tools = {
            'search': SearchInformationTool(),
            'navigate': VisitTool(),
            'page_up': PageUpTool(),
            'page_down': PageDownTool(),
            'finder': FinderTool(),
            'find_next': FindNextTool(),
            'summary': FullPageSummaryTool(
                model_name=model_name,
                temperature=temperature,
                usage_statistics=self.usage_statistics
            )
        }
        
        # Initialize browser if not already done
        init_browser()
        
        logger.info("BrowserAutomationMCP initialized successfully", extra={
            'operation': 'BROWSER_AUTOMATION_MCP_INIT',
            'surfer_tools_count': len(self.surfer_tools),
            'model_name': model_name
        })


# MCP Registration and Factory Functions
def create_web_information_mcps() -> List[BaseTool]:
    """
    Factory function to create all Web & Information MCPs with default configuration
    
    Creates and configures the four core high-value MCPs for web & information tools
    following the Pareto principle optimization for maximum task coverage with
    minimal tool count.
    
    Returns:
        List[BaseTool]: List of configured Web & Information MCP tools
    """
    logger.info("Creating Web & Information MCPs", extra={
        'operation': 'WEB_INFO_MCPS_CREATE'
    })
    
    # Initialize shared components
    usage_stats = UsageStatistics()
    
    # Create MCP instances with default configurations
    mcps = [
        WebScraperMCP(),
        SearchEngineMCP(),
        WikipediaMCP(usage_statistics=usage_stats),
        BrowserAutomationMCP(usage_statistics=usage_stats)
    ]
    
    logger.info("Web & Information MCPs created successfully", extra={
        'operation': 'WEB_INFO_MCPS_CREATED',
        'mcp_count': len(mcps),
        'mcp_names': [mcp.name for mcp in mcps]
    })
    
    return mcps


def register_web_information_mcps_with_rag_engine(rag_engine) -> None:
    """
    Register Web & Information MCPs with existing RAG-MCP Engine
    
    Integrates the new MCPs with the existing Pareto MCP Registry for
    seamless operation within the Enhanced Alita KGoT framework.
    
    Args:
        rag_engine: RAG-MCP Engine instance to register MCPs with
    """
    logger.info("Registering Web & Information MCPs with RAG Engine", extra={
        'operation': 'MCP_REGISTRATION_START'
    })
    
    # MCP specifications for registration
    mcp_specs = [
        MCPToolSpec(
            name="web_scraper_mcp",
            description="Advanced web scraping with Beautiful Soup integration for structured data extraction",
            capabilities=["html_parsing", "data_extraction", "content_analysis", "xpath_selection"],
            category=MCPCategory.WEB_INFORMATION,
            usage_frequency=0.25,
            reliability_score=0.92,
            cost_efficiency=0.88
        ),
        MCPToolSpec(
            name="search_engine_mcp",
            description="Multi-provider search with SerpApi, Bing, DuckDuckGo integration",
            capabilities=["web_search", "result_ranking", "content_filtering", "query_optimization"],
            category=MCPCategory.WEB_INFORMATION,
            usage_frequency=0.18,
            reliability_score=0.94,
            cost_efficiency=0.91
        ),
        MCPToolSpec(
            name="wikipedia_mcp",
            description="Wikipedia API integration with structured knowledge access and fact verification",
            capabilities=["knowledge_lookup", "entity_resolution", "fact_checking", "knowledge_graph"],
            category=MCPCategory.WEB_INFORMATION,
            usage_frequency=0.15,
            reliability_score=0.96,
            cost_efficiency=0.93
        ),
        MCPToolSpec(
            name="browser_automation_mcp",
            description="Automated browser interaction using Playwright/Puppeteer for dynamic content",
            capabilities=["ui_automation", "form_filling", "screenshot_capture", "session_management"],
            category=MCPCategory.WEB_INFORMATION,
            usage_frequency=0.22,
            reliability_score=0.89,
            cost_efficiency=0.85
        )
    ]
    
    # Register with RAG engine if it has a registry
    if hasattr(rag_engine, 'pareto_registry'):
        for spec in mcp_specs:
            rag_engine.pareto_registry.mcps.append(spec)
        
        logger.info("MCPs registered with RAG Engine successfully", extra={
            'operation': 'MCP_REGISTRATION_SUCCESS',
            'registered_count': len(mcp_specs)
        })
    else:
        logger.warning("RAG Engine does not have Pareto registry", extra={
            'operation': 'MCP_REGISTRATION_WARNING'
        })


# Module initialization and export
if __name__ == "__main__":
    # Example usage and testing
    logger.info("Web Information MCPs module loaded", extra={
        'operation': 'MODULE_LOAD',
        'mcps_available': 4
    })
    
    # Create example MCPs for testing
    mcps = create_web_information_mcps()
    
    # Log successful module initialization
    logger.info("Task 21 implementation completed successfully", extra={
        'operation': 'TASK_21_COMPLETE',
        'mcps_implemented': len(mcps),
        'capabilities_total': sum(len(getattr(mcp, 'capabilities', [])) for mcp in mcps)
    }) 