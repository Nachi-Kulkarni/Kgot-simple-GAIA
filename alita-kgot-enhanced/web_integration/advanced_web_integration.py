"""
Advanced Web Integration (Task 41)
=================================
Unifies capabilities of Alita Web Agent and KGoT Surfer Agent into a single, extensible **AdvancedWebAgent**.

Key design goals (extracted from Alita & KGoT papers):
1. Comprehensive tool-set: Google/SerpApi search, GitHub search, Wikipedia lookup, granular navigation tools (Visit, PageUp/Down, Find), and support for JS-heavy sites.
2. MCP-driven automation: Ability to trigger pre-validated MCP workflows for multi-step web tasks.
3. Context-aware navigation: Extract page elements into the session Knowledge Graph and query it when deciding the next action.
4. Complex-UI handling: Delegate screenshot analysis to KGoT-Alita screenshot analyzer when HTML interaction is insufficient.

This file provides the *initial* skeleton for the agent. Subsequent tasks (T41-2 .. T41-5) will iteratively extend each subsection.
"""

from __future__ import annotations

# Standard Library
import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import json
import requests
import re
from datetime import datetime
from bs4 import BeautifulSoup
import networkx as nx

# Third-Party & Project-Specific Imports
# NOTE: Some of these may be implemented in later tasks. They are kept as type-only imports or inside
#       try/except blocks for now to avoid ImportErrors during the skeleton phase.

try:
    # User prefers Winston-style logging across the codebase → reuse existing helper if available.
    from config.logging.winston_config import setup_winston_logger  # type: ignore
except (ModuleNotFoundError, ImportError):
    # Fallback to Python logging so the module remains functional even if Winston wrapper is absent.
    def setup_winston_logger(component: str = "ADVANCED_WEB_AGENT") -> logging.Logger:  # noqa: D401
        """Simple fallback returning a standard Python logger when Winston wrapper is unavailable."""

        logger = logging.getLogger(component)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# LangChain / tooling
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from pydantic import BaseModel, Field

# KGoT Surfer & utilities
from kgot.tools.tools_v2_3.SurferTool import SearchTool  # type: ignore
from kgot.tools.tools_v2_3.Web_surfer import (
    VisitTool,
    PageUpTool,
    PageDownTool,
    FinderTool,
    FindNextTool,
    init_browser,
)
from kgot.tools.tools_v2_3.WikipediaTool import LangchainWikipediaTool  # type: ignore
from kgot.utils import UsageStatistics  # type: ignore

# Ensure project root is in PYTHONPATH for dynamic imports (linter & runtime)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from multimodal.kgot_alita_screenshot_analyzer import create_kgot_alita_screenshot_analyzer  # type: ignore
except ImportError:  # pragma: no cover – optional heavy dependency
    create_kgot_alita_screenshot_analyzer = None  # type: ignore

# === Configuration Dataclass ==========================================================

@dataclass
class AdvancedWebIntegrationConfig:
    """Configuration for :class:`AdvancedWebAgent`.

    Attributes
    ----------
    search_timeout : int
        Global timeout (in seconds) for web search queries.
    kgot_model_name : str
        LLM model used for LangChain agent orchestration (defaults to o3 – user preference).
    temperature : float
        Sampling temperature for the LLM.
    graph_store_backend : str
        Backend used for the Knowledge Graph (networkx | neo4j | rdf4j).
    graph_store_config : Dict[str, Any]
        Custom connection/config parameters for the chosen graph store.
    serpapi_api_key : Optional[str]
        API key enabling SerpApi-powered search. If *None*, system falls back to GoogleSearchTool.
    enable_mcp_validation : bool
        Whether to run MCP-level validation before executing an MCP workflow.
    mcp_confidence_threshold : float
        Confidence threshold (0-1) above which the agent auto-executes a matching MCP.
    """

    # --- General settings ----------------------------------------------------
    search_timeout: int = 30

    # --- LLM / LangChain settings -------------------------------------------
    kgot_model_name: str = "o3"
    temperature: float = 0.1

    # --- Knowledge-Graph settings -------------------------------------------
    graph_store_backend: str = "networkx"
    graph_store_config: Dict[str, Any] = field(default_factory=dict)

    # --- Search / Tool settings ---------------------------------------------
    serpapi_api_key: Optional[str] = None

    # --- MCP settings --------------------------------------------------------
    enable_mcp_validation: bool = True
    mcp_confidence_threshold: float = 0.7

    # --- Web Automation Templates -------------------------------------------
    web_automation_templates: Dict[str, Any] = field(
        default_factory=lambda: {
            "login_and_download": {
                "steps": ["login", "navigate", "download"],
                "validation_rules": ["auth_success", "file_integrity"],
            },
            "code_research": {
                "steps": ["search", "cross_reference", "synthesize"],
                "validation_rules": ["source_diversity", "fact_consistency"],
            },
        }
    )


# === Main Agent Skeleton ==============================================================

class AdvancedWebAgent:
    """Unified web agent combining Alita & KGoT Surfer capabilities.

    The following *placeholder* methods will be fully implemented in subsequent tasks:

    - :py:meth:`initialize` sets up KG backend, search/navigation tools, and LangChain agent.
    - :py:meth:`execute_task` runs an arbitrary user query or workflow through the agent.
    - :py:meth:`execute_mcp_template` triggers a pre-validated MCP workflow if applicable.
    - :py:meth:`navigate_and_extract` performs context-aware navigation, updating the KG.
    - :py:meth:`handle_complex_ui` delegates interaction with complex UI elements to the screenshot analyzer.

    The skeleton ensures a clean, testable structure and Winston-compatible logging from day one.
    """

    # ----------------------------------------------------------------------
    # INITIALISATION
    # ----------------------------------------------------------------------
    def __init__(self, config: Optional[AdvancedWebIntegrationConfig] = None):
        """Instantiate **AdvancedWebAgent**.

        Parameters
        ----------
        config : Optional[AdvancedWebIntegrationConfig]
            Custom configuration. If *None*, a default config is instantiated.
        """

        self.config = config or AdvancedWebIntegrationConfig()
        self.logger = setup_winston_logger("ADVANCED_WEB_AGENT")

        # Placeholders for core components – will be set in *initialize*.
        self.graph_store = None  # type: Optional["KnowledgeGraphInterface"]
        self.langchain_agent = None  # type: Optional["AgentExecutor"]
        self.kgot_tools: List[Any] = []
        self.alita_tools: List[Any] = []
        self.integrated_tools: List[Any] = []
        
        # Browser and WebSurfer components
        self.browser = None  # type: Optional[Any]
        self.web_surfer = None  # type: Optional[Any]

        # Session tracking helpers ------------------------------------------------
        self.active_session_id: Optional[str] = None
        self.navigation_context: Dict[str, Any] = {}

        self.logger.info(
            "AdvancedWebAgent instantiated – awaiting initialization",
            extra={"operation": "AGENT_INIT", "config": self.config.__dict__},
        )

        # Internal helper for usage statistics (for SearchTool)
        self._usage_statistics = UsageStatistics()  # noqa: WPS437 – external util provides stats

        # MCP tools will be populated after _initialize_mcp_tools executes
        self.mcp_tools: List[Any] = []

        self.screenshot_analyzer = None  # type: Optional[Any]

    # ----------------------------------------------------------------------
    # PRIVATE – GRAPH STORE INITIALISATION
    # ----------------------------------------------------------------------

    async def _initialize_graph_store(self) -> None:  # noqa: D401
        """Set up a lightweight NetworkX knowledge graph for the current session."""

        self.logger.info("Initializing in-memory NetworkX graph", extra={"operation": "KG_INIT_START"})

        try:
            self.graph_store = nx.DiGraph()
            # Add session root node
            self.graph_store.add_node("session", type="session")
            self.logger.info("Knowledge graph ready", extra={"operation": "KG_INIT_SUCCESS"})
        except Exception as exc:
            self.logger.logError("KG_INIT", exc)
            # Graceful degradation – still allow rest of agent to run
            self.graph_store = None

    # ----------------------------------------------------------------------
    # PRIVATE – TOOL INITIALISATION
    # ----------------------------------------------------------------------

    async def _initialize_toolset(self) -> None:
        """Build the full list of tools exposed to the LangChain agent."""

        self.logger.info("Loading web tools", extra={"operation": "TOOLS_INIT_START"})

        # --- Generic search tools ----------------------------------------------------
        try:
            search_tool = SearchTool(
                model_name=self.config.kgot_model_name,
                temperature=self.config.temperature,
                usage_statistics=self._usage_statistics,
            )
            wikipedia_tool = LangchainWikipediaTool()
            self.kgot_tools = [search_tool, wikipedia_tool]
        except Exception as exc:  # pragma: no cover – defensive
            self.logger.logError("TOOLS_INIT", exc)
            self.kgot_tools = []

        # --- Custom Google/SerpApi and GitHub search tools ---------------------------

        class GoogleSearchTool(BaseTool):  # noqa: WPS431 – inner class intentional
            """Lightweight Google (or SerpApi) search wrapper."""

            name = "google_search"
            description = (
                "Perform a Google web search and return the raw JSON results (top ~10). "
                "Falls back to basic HTML scraping if an official API is not configured."
            )

            def __init__(self, api_key: Optional[str], *args: Any, **kwargs: Any):  # noqa: D401
                super().__init__(*args, **kwargs)
                self._api_key = api_key

            def _run(self, query: str) -> str:  # noqa: D401
                try:
                    if self._api_key:
                        endpoint = "https://serpapi.com/search.json"
                        payload = {"api_key": self._api_key, "q": query, "num": 10}
                        resp = requests.get(endpoint, params=payload, timeout=15)
                        resp.raise_for_status()
                        return json.dumps(resp.json())
                    # Fallback simple search (HTML) – return URL string so downstream tools can visit
                    encoded = requests.utils.quote(query)
                    url = f"https://www.google.com/search?q={encoded}"
                    return json.dumps({"query": query, "url": url})
                except Exception as exc:
                    return f"GoogleSearchTool error: {exc}"

            async def _arun(self, query: str) -> str:  # noqa: D401
                return self._run(query)

        class GitHubSearchTool(BaseTool):  # noqa: WPS431
            """Search GitHub repositories via public REST API."""

            name = "github_search"
            description = (
                "Search GitHub repositories, code, or issues. "
                "Default search type is 'repositories'."
            )

            class _ArgsSchema(BaseModel):  # noqa: WPS431 – nested pydantic for clarity
                query: str = Field(..., description="GitHub search query")
                search_type: str = Field(
                    "repositories",
                    description="GitHub search type: repositories | code | issues | users",
                )

            args_schema = _ArgsSchema

            def _run(self, query: str, search_type: str = "repositories") -> str:  # noqa: D401
                try:
                    endpoint = f"https://api.github.com/search/{search_type}"
                    resp = requests.get(endpoint, params={"q": query, "per_page": 10}, timeout=15)
                    resp.raise_for_status()
                    return json.dumps(resp.json())
                except Exception as exc:
                    return f"GitHubSearchTool error: {exc}"

            async def _arun(self, query: str, search_type: str = "repositories") -> str:  # noqa: D401
                return self._run(query, search_type)

        # --- Navigation tool wrappers -----------------------------------------------

        # Minimal schemas for no-arg tools
        class _EmptySchema(BaseModel):
            """Empty args schema marker."""

            pass

        def _wrap_web_surfer_tool(tool_cls, name: str, description: str, arg_fields: Dict[str, Any]):  # noqa: ANN001
            """Factory helper returning a LangChain BaseTool wrapper around a WebSurfer tool."""

            if not arg_fields:
                args_schema = _EmptySchema
            else:
                schema_dict = {}
                for field_name, (annotation, field_def) in arg_fields.items():
                    schema_dict[field_name] = field_def
                    schema_dict['__annotations__'] = schema_dict.get('__annotations__', {})
                    schema_dict['__annotations__'][field_name] = annotation
                args_schema = type("DynamicSchema", (BaseModel,), schema_dict)

            class _Wrapper(BaseTool):  # noqa: WPS430, WPS431 – intentional dynamic class
                name = name
                description = description
                args_schema = args_schema

                def _run(self, **kwargs: Any) -> str:  # noqa: D401
                    try:
                        init_browser()
                        tool_instance = tool_cls()
                        return tool_instance.forward(**kwargs)  # type: ignore[arg-type]
                    except Exception as exc:
                        return f"{name} error: {exc}"

                async def _arun(self, **kwargs: Any) -> str:  # noqa: D401
                    return self._run(**kwargs)

            return _Wrapper()

        # Build navigation tool wrappers
        visit_page_tool = _wrap_web_surfer_tool(
            VisitTool,
            "visit_page",
            "Visit a webpage given its URL and return the visible text.",
            {"url": (str, Field(..., description="Absolute or relative URL to visit"))},
        )
        page_up_tool = _wrap_web_surfer_tool(
            PageUpTool,
            "page_up",
            "Scroll the viewport UP one page length.",
            {},
        )
        page_down_tool = _wrap_web_surfer_tool(
            PageDownTool,
            "page_down",
            "Scroll the viewport DOWN one page length.",
            {},
        )
        find_tool = _wrap_web_surfer_tool(
            FinderTool,
            "find_on_page",
            "Find the first occurrence of a string on the current page (Ctrl+F).",
            {"search_string": (str, Field(..., description="String to search for"))},
        )
        find_next_tool = _wrap_web_surfer_tool(
            FindNextTool,
            "find_next",
            "Find the next occurrence of the previously searched string.",
            {},
        )

        navigation_tools = [
            visit_page_tool,
            page_up_tool,
            page_down_tool,
            find_tool,
            find_next_tool,
        ]

        # --- Aggregate all tools ------------------------------------------------------
        self.alita_tools = [
            GoogleSearchTool(self.config.serpapi_api_key),
            GitHubSearchTool(),
        ]

        self.integrated_tools = self.kgot_tools + self.alita_tools + navigation_tools

        # Load MCPs (they may rely on previously imported langchain); keep separate call for clarity
        await self._initialize_mcp_tools()

        self.logger.info(
            "Toolset loaded",
            extra={
                "operation": "TOOLS_INIT_SUCCESS",
                "kgot_tools": len(self.kgot_tools),
                "alita_tools": len(self.alita_tools),
                "nav_tools": len(navigation_tools),
                "mcp_tools": len(self.mcp_tools),
            },
        )

    async def _initialize_mcp_tools(self) -> None:
        """Dynamically load MCP toolsets and append to *integrated_tools*."""

        self.logger.info("Loading MCP toolsets", extra={"operation": "MCP_INIT_START"})

        try:
            from mcp_toolbox.web_information_mcps import create_web_information_mcps  # type: ignore
            self.mcp_tools.extend(create_web_information_mcps())
        except Exception as exc:  # pragma: no cover – may fail if deps missing
            self.logger.warning("Failed to load web_information_mcps", extra={"error": str(exc)})

        try:
            from mcp_toolbox.communication_mcps import create_communication_mcps  # type: ignore
            self.mcp_tools.extend(create_communication_mcps())
        except Exception as exc:
            self.logger.warning("Failed to load communication_mcps", extra={"error": str(exc)})

        # Deduplicate by tool name
        unique = {}
        for tool in self.mcp_tools:
            unique[tool.name] = tool
        self.mcp_tools = list(unique.values())

        self.logger.info(
            "MCP toolsets loaded",
            extra={"operation": "MCP_INIT_SUCCESS", "count": len(self.mcp_tools)}
        )

        # Expose to integrated tools list if already initialized
        if self.integrated_tools is not None:
            self.integrated_tools.extend(self.mcp_tools)

    # ----------------------------------------------------------------------
    # PRIVATE – AGENT CREATION
    # ----------------------------------------------------------------------

    async def _create_langchain_agent(self) -> None:  # noqa: D401
        """Instantiate the LangChain agent that orchestrates tool calls."""

        if not self.integrated_tools:
            raise RuntimeError("Toolset must be initialized before creating agent")

        llm = ChatOpenAI(
            model_name=self.config.kgot_model_name,
            temperature=self.config.temperature,
        )

        self.langchain_agent = AgentExecutor.from_agent_and_tools(
            agent=create_openai_functions_agent(llm, self.integrated_tools),
            tools=self.integrated_tools,
            verbose=True,
            max_iterations=12,
            return_intermediate_steps=True,
        )

        self.logger.info(
            "LangChain agent ready",
            extra={"operation": "LANGCHAIN_INIT_SUCCESS", "total_tools": len(self.integrated_tools)},
        )

    # ----------------------------------------------------------------------
    # PUBLIC API METHODS (placeholders)
    # ----------------------------------------------------------------------
    async def initialize(self) -> bool:  # noqa: D401
        """Initialize all sub-systems (toolset + agent).

        Returns
        -------
        bool
            *True* on success, *False* if an unrecoverable error occurred.
        """

        self.logger.info("Initialization start", extra={"operation": "AGENT_INITIALIZE_START"})

        try:
            # 1) Create / init knowledge graph
            await self._initialize_graph_store()

            # 2) Initialize browser with WebSurfer integration
            await self._initialize_browser()

            # 3) Build toolset
            await self._initialize_toolset()

            # 4) Create LangChain agent
            await self._create_langchain_agent()

            self.logger.info("Initialization completed", extra={"operation": "AGENT_INITIALIZE_SUCCESS"})
            return True

        except Exception as exc:
            self.logger.logError("AGENT_INITIALIZE", exc)
            return False

    async def _initialize_browser(self) -> None:
        """Initialize browser session with WebSurfer integration."""
        try:
            # Import WebSurfer components
            from web_surfer import WebSurferAgent
            from web_surfer.browser import Browser
            
            # Initialize browser with configuration
            browser_config = {
                "headless": getattr(self.config, 'browser_headless', True),
                "viewport": {"width": 1280, "height": 720},
                "timeout": 30000,
                "user_agent": "Mozilla/5.0 (compatible; AdvancedWebAgent/1.0)"
            }
            
            self.browser = Browser(**browser_config)
            await self.browser.start()
            
            # Initialize WebSurfer agent for advanced navigation
            self.web_surfer = WebSurferAgent(
                browser=self.browser,
                llm_config={
                    "model": self.config.kgot_model_name,
                    "temperature": 0.1
                }
            )
            
            self.logger.info("Browser and WebSurfer initialized successfully")
            
        except ImportError:
            self.logger.warning("WebSurfer not available, using fallback browser")
            # Fallback to basic browser simulation
            self.browser = {
                "url": None,
                "title": None,
                "content": None,
                "session_id": f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            self.web_surfer = None
            
        except Exception as exc:
            self.logger.logError("BROWSER_INIT", exc)
            raise RuntimeError(f"Failed to initialize browser: {exc}") from exc

    async def execute_task(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a web-based task using the unified agent.

        Parameters
        ----------
        query : str
            Natural-language description of the task (e.g., "Find latest NumPy release notes").
        context : Optional[Dict[str, Any]]
            Additional context or constraints provided by the caller.

        Returns
        -------
        Dict[str, Any]
            Structured results (schema will be refined in later tasks).
        """

        if not self.langchain_agent:
            raise RuntimeError("AdvancedWebAgent not initialized. Call initialize() first.")

        self.logger.info(
            "Executing task", extra={"operation": "TASK_EXECUTE", "query": query}
        )
        
        try:
            # Prepare context-aware prompt
            context_str = ""
            if context:
                context_str = f"\nAdditional context: {json.dumps(context, indent=2)}"
            
            # Enhanced prompt with tool usage guidance
            enhanced_prompt = f"""
            Task: {query}{context_str}
            
            You have access to comprehensive web tools including:
            - Search tools (Google/SerpApi, GitHub, Wikipedia)
            - Navigation tools (Visit, PageUp, PageDown, Find)
            - MCP automation tools for complex workflows
            
            Please execute this task step by step:
            1. If this is a search task, use appropriate search tools first
            2. If you need to navigate web pages, use the navigation tools
            3. Extract and structure the relevant information
            4. Provide a comprehensive summary of findings
            
            Return your results in a structured format.
            """
            
            # Execute through LangChain agent
            result = await self.langchain_agent.ainvoke({"input": enhanced_prompt})
            
            # Process and structure the result
            structured_result = {
                "status": "success",
                "query": query,
                "context": context or {},
                "result": result.get("output", result),
                "agent_steps": result.get("intermediate_steps", []),
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(
                "Task execution completed", 
                extra={
                    "operation": "TASK_EXECUTE_SUCCESS", 
                    "query": query,
                    "steps_count": len(structured_result["agent_steps"])
                }
            )
            
            return structured_result
            
        except Exception as exc:
            self.logger.logError("TASK_EXECUTE", exc)
            return {
                "status": "error",
                "query": query,
                "context": context or {},
                "error": str(exc),
                "timestamp": datetime.now().isoformat()
            }

    async def execute_mcp_template(self, template_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to run a pre-validated MCP workflow matching *template_name*.

        This method will consult the confidence threshold set in the config before executing.
        """

        self.logger.info(
            "MCP execution requested",
            extra={"operation": "MCP_EXECUTION_REQUEST", "template": template_name},
        )

        template = self.config.web_automation_templates.get(template_name)
        if not template:
            return {
                "status": "template_not_found", 
                "template": template_name,
                "available_templates": list(self.config.web_automation_templates.keys())
            }

        # Enhanced validation: check both required steps and parameter completeness
        required_steps = template.get("steps", [])
        validation_rules = template.get("validation_rules", [])
        
        # Calculate confidence based on parameter coverage and validation rules
        provided_params = set(params.keys())
        step_coverage = len([step for step in required_steps if any(param in step for param in provided_params)]) / (len(required_steps) or 1)
        rule_coverage = len([rule for rule in validation_rules if any(param in rule for param in provided_params)]) / (len(validation_rules) or 1)
        
        confidence = (step_coverage + rule_coverage) / 2 if validation_rules else step_coverage

        self.logger.debug(
            "MCP validation result",
            extra={
                "operation": "MCP_VALIDATION",
                "confidence": confidence,
                "threshold": self.config.mcp_confidence_threshold,
                "step_coverage": step_coverage,
                "rule_coverage": rule_coverage
            },
        )

        if self.config.enable_mcp_validation and confidence < self.config.mcp_confidence_threshold:
            return {
                "status": "validation_failed", 
                "confidence": confidence,
                "threshold": self.config.mcp_confidence_threshold,
                "required_steps": required_steps,
                "provided_params": list(provided_params)
            }

        # Ensure tools available in agent
        if not self.langchain_agent:
            raise RuntimeError("Agent not initialized")

        # Construct enhanced prompt with step-by-step guidance
        agent_prompt = f"""
        Execute the '{template_name}' MCP workflow with the following configuration:
        
        Template Steps: {json.dumps(required_steps, indent=2)}
        Validation Rules: {json.dumps(validation_rules, indent=2)}
        Parameters: {json.dumps(params, indent=2)}
        
        Please execute each step in order:
        1. Validate all required parameters are available
        2. Execute each step using the appropriate MCP tools
        3. Apply validation rules to ensure quality
        4. Return a structured JSON result with:
           - status: success/partial/failed
           - executed_steps: list of completed steps
           - results: detailed results for each step
           - validation_results: outcomes of validation rules
           - confidence_score: final confidence in results
        
        Use the available MCP tools and web navigation capabilities as needed.
        """

        try:
            start_time = datetime.now()
            result = await self.langchain_agent.ainvoke({"input": agent_prompt})
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "template": template_name,
                "confidence": confidence,
                "execution_time_seconds": execution_time,
                "result": result.get("output", result),
                "agent_steps": result.get("intermediate_steps", []),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as exc:
            self.logger.logError("MCP_EXECUTION", exc)
            return {
                "status": "execution_error", 
                "template": template_name,
                "error": str(exc),
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }

    async def navigate_and_extract(self, url: str, goal: str) -> Dict[str, Any]:
        """Context-aware navigation routine.

        Will iteratively visit *url*, update the KG with extracted elements, and take the next action
        towards *goal*.
        """

        if not self.graph_store:
            self.logger.warning("navigate_and_extract called without KG", extra={"operation": "NAV_NO_KG"})
            return {"status": "kg_not_initialized"}

        steps_taken: List[Dict[str, Any]] = []

        current_url = url
        for depth in range(10):  # hard-stop depth
            self.logger.info(
                "Navigation step", extra={"operation": "NAV_STEP", "depth": depth, "url": current_url}
            )

            try:
                # Use the proper VisitTool from integrated tools
                visit_tool = next((t for t in self.integrated_tools if t.name == "visit_page"), None)
                if not visit_tool:
                    self.logger.error("VisitTool not found in integrated tools")
                    break
                
                # Execute visit tool
                page_result = await visit_tool._arun(url=current_url)
                page_text = page_result if isinstance(page_result, str) else str(page_result)
                
            except Exception as exc:
                self.logger.error(f"Failed to visit {current_url}: {exc}")
                break

            # Parse page content
            soup = BeautifulSoup(page_text, "html.parser")

            # Extract headers and links
            headers = [h.get_text(strip=True) for h in soup.find_all(re.compile("^h[1-6]$"))]
            links = [(a.get_text(strip=True), a.get("href")) for a in soup.find_all("a", href=True) if a.get("href")]

            # Update KG nodes
            page_node = f"page_{depth}_{hash(current_url) % 10000}"
            self.graph_store.add_node(page_node, type="page", url=current_url, depth=depth)
            
            for i, header in enumerate(headers[:10]):  # Limit to avoid overwhelming KG
                header_node = f"header_{depth}_{i}_{hash(header) % 1000}"
                self.graph_store.add_node(header_node, type="header", text=header)
                self.graph_store.add_edge(page_node, header_node, relation="contains")
                
            for i, (text, href) in enumerate(links[:20]):  # Limit links
                if href and len(href) > 0:
                    # Resolve relative URLs
                    if not href.startswith(('http://', 'https://')):
                        try:
                            href = requests.compat.urljoin(current_url, href)
                        except Exception:
                            continue
                    
                    link_node = f"link_{depth}_{i}_{hash(href) % 1000}"
                    self.graph_store.add_node(link_node, type="link", text=text, href=href)
                    self.graph_store.add_edge(page_node, link_node, relation="links")

            steps_taken.append({
                "url": current_url, 
                "headers": headers[:5], 
                "links": [(text, href) for text, href in links[:5]],
                "depth": depth
            })

            # Goal check – search in page text
            page_content = soup.get_text().lower()
            if goal.lower() in page_content:
                self.logger.info("Goal content found", extra={"operation": "NAV_GOAL_FOUND", "url": current_url})
                
                # Extract relevant content around goal
                goal_context = self._extract_goal_context(soup, goal)
                
                return {
                    "status": "success", 
                    "url": current_url, 
                    "steps": steps_taken,
                    "goal_content": goal_context,
                    "depth_reached": depth
                }

            # Choose next link using KG-informed decision
            next_url = self._choose_next_link(links, goal, current_url)
            
            if not next_url:
                self.logger.warning("No further links to follow", extra={"operation": "NAV_NO_LINKS"})
                break

            current_url = next_url

        return {"status": "incomplete", "steps": steps_taken, "depth_reached": depth}
    
    def _extract_goal_context(self, soup: BeautifulSoup, goal: str) -> str:
        """Extract relevant content around the goal text."""
        try:
            text = soup.get_text()
            goal_lower = goal.lower()
            text_lower = text.lower()
            
            # Find goal position and extract context
            goal_pos = text_lower.find(goal_lower)
            if goal_pos != -1:
                start = max(0, goal_pos - 200)
                end = min(len(text), goal_pos + len(goal) + 200)
                return text[start:end].strip()
            
            return ""
        except Exception:
            return ""
    
    def _choose_next_link(self, links: List[tuple], goal: str, current_url: str) -> Optional[str]:
        """Choose the most relevant next link based on goal and KG context."""
        goal_lower = goal.lower()
        scored_links = []
        
        for text, href in links:
            if not href or href.startswith('#') or href.startswith('mailto:'):
                continue
                
            # Resolve relative URLs
            if not href.startswith(('http://', 'https://')):
                try:
                    href = requests.compat.urljoin(current_url, href)
                except Exception:
                    continue
            
            # Score link relevance
            score = 0
            text_lower = text.lower()
            href_lower = href.lower()
            
            # Goal keyword in link text or URL
            if goal_lower in text_lower:
                score += 10
            if goal_lower in href_lower:
                score += 8
            
            # Common relevant terms
            relevant_terms = ['doc', 'guide', 'tutorial', 'api', 'reference', 'manual']
            for term in relevant_terms:
                if term in text_lower or term in href_lower:
                    score += 2
            
            if score > 0:
                scored_links.append((score, href))
        
        # Return highest scored link
        if scored_links:
            scored_links.sort(reverse=True)
            return scored_links[0][1]
        
        # Fallback to first valid link
        for text, href in links:
            if href and not href.startswith('#') and not href.startswith('mailto:'):
                if not href.startswith(('http://', 'https://')):
                    try:
                        return requests.compat.urljoin(current_url, href)
                    except Exception:
                        continue
                return href
        
        return None

    async def handle_complex_ui(self, element_description: str, screenshot_path: Optional[str] = None) -> Dict[str, Any]:
        """Handle complex UI elements using screenshot analysis.

        When the agent encounters a complex web element that is difficult to interact with via HTML
        (e.g., a canvas or a custom JavaScript widget), it will use the KGoT-Alita Screenshot Analyzer.
        """
        try:
            self.logger.info(
                "Complex UI handling requested",
                extra={"operation": "COMPLEX_UI_ANALYSIS", "element": element_description}
            )
            
            # Take screenshot if not provided
            if not screenshot_path and hasattr(self, 'browser') and hasattr(self.browser, 'screenshot'):
                screenshot_path = f"/tmp/ui_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                await self.browser.screenshot(path=screenshot_path)
                self.logger.debug(f"Screenshot captured: {screenshot_path}")
            
            # Prepare analysis request for screenshot analyzer
            analysis_request = {
                "image_path": screenshot_path,
                "element_description": element_description,
                "analysis_type": "ui_interaction",
                "context": {
                    "current_url": getattr(self, 'browser', {}).get('url', 'unknown'),
                    "page_title": getattr(self, 'browser', {}).get('title', 'unknown'),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Check if screenshot analyzer is available in MCP tools
            screenshot_analyzer = None
            for tool in self.mcp_tools:
                if hasattr(tool, 'name') and 'screenshot' in tool.name.lower():
                    screenshot_analyzer = tool
                    break
            
            if screenshot_analyzer:
                # Use MCP screenshot analyzer
                analysis_result = await screenshot_analyzer.run(analysis_request)
                
                # Extract actionable information
                ui_elements = analysis_result.get('ui_elements', [])
                recommended_actions = analysis_result.get('recommended_actions', [])
                
                # Update knowledge graph with UI analysis
                if self.graph_store:
                    ui_node_id = f"ui_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.graph_store.add_node(
                        ui_node_id,
                        type="ui_analysis",
                        element_description=element_description,
                        screenshot_path=screenshot_path,
                        ui_elements=ui_elements,
                        recommended_actions=recommended_actions,
                        timestamp=datetime.now().isoformat()
                    )
                
                return {
                    "status": "success",
                    "element_description": element_description,
                    "screenshot_path": screenshot_path,
                    "ui_elements": ui_elements,
                    "recommended_actions": recommended_actions,
                    "analysis_confidence": analysis_result.get('confidence', 0.0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Fallback: Use LangChain agent with enhanced prompt
                if not self.langchain_agent:
                    raise RuntimeError("Neither screenshot analyzer nor LangChain agent available")
                
                fallback_prompt = f"""
                Analyze the complex UI element described as: "{element_description}"
                
                Based on the current page context and available browser tools, provide:
                1. Possible interaction strategies
                2. Alternative approaches if direct interaction fails
                3. Recommended next steps
                
                Available tools include: VisitTool, PageUpTool, PageDownTool, FinderTool, FindNextTool
                
                Return a structured response with actionable recommendations.
                """
                
                result = await self.langchain_agent.ainvoke({"input": fallback_prompt})
                
                return {
                    "status": "fallback_analysis",
                    "element_description": element_description,
                    "screenshot_path": screenshot_path,
                    "analysis_method": "langchain_fallback",
                    "recommendations": result.get("output", str(result)),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as exc:
            self.logger.logError("COMPLEX_UI_ANALYSIS", exc)
            return {
                "status": "analysis_error",
                "element_description": element_description,
                "error": str(exc),
                "timestamp": datetime.now().isoformat()
            }

    # ----------------------------------------------------------------------
    # CLEAN-UP
    # ----------------------------------------------------------------------
    async def close(self) -> None:
        """Clean up any persistent resources (browsers, KG connections, etc.)."""

        self.logger.info("Shutting down AdvancedWebAgent", extra={"operation": "AGENT_CLOSE"})
        
        try:
            # Close browser and WebSurfer
            if self.web_surfer:
                try:
                    if hasattr(self.web_surfer, 'close'):
                        await self.web_surfer.close()
                    self.logger.debug("WebSurfer closed")
                except Exception as exc:
                    self.logger.logError("WEBSURFER_CLOSE", exc)
                    
            if self.browser:
                try:
                    if hasattr(self.browser, 'close'):
                        await self.browser.close()
                    elif hasattr(self.browser, 'stop'):
                        await self.browser.stop()
                    self.logger.debug("Browser closed")
                except Exception as exc:
                    self.logger.logError("BROWSER_CLOSE", exc)
            
            # Close graph store connection
            if self.graph_store and hasattr(self.graph_store, 'close'):
                await self.graph_store.close()
                self.logger.info("Graph store closed", extra={"operation": "GRAPH_STORE_CLOSE"})
            
            # Clean up browser sessions if any tools have them
            for tool in self.integrated_tools:
                if hasattr(tool, 'close') and callable(getattr(tool, 'close')):
                    try:
                        await tool.close()
                    except Exception as exc:
                        self.logger.warning(f"Failed to close tool {tool.name}: {exc}")
            
            # Clear tool references
            self.integrated_tools.clear()
            self.langchain_agent = None
            self.browser = None
            self.web_surfer = None
            
            self.logger.info("AdvancedWebAgent shutdown completed", extra={"operation": "AGENT_CLOSE_SUCCESS"})
            
        except Exception as exc:
            self.logger.logError("AGENT_CLOSE", exc)


# ------------------------------------------------------------------------------
# Module Convenience Function
# ------------------------------------------------------------------------------

async def create_advanced_web_agent(
    config: Optional[AdvancedWebIntegrationConfig] = None,
) -> AdvancedWebAgent:  # noqa: D401
    """Factory helper mirroring the pattern used across the codebase."""

    agent = AdvancedWebAgent(config)
    await agent.initialize()
    return agent