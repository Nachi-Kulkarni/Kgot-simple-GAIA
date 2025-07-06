import asyncio
from typing import Any, Dict, List, Tuple, Optional
import logging

# Initialize a module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configure basic logging; in production, load centralized config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# --- Cost Optimization Integration ---
try:
    from optimization.advanced_cost_optimization import AdvancedCostOptimizer  # type: ignore
except (ImportError, ModuleNotFoundError):
    # The optimizer module may not be available in some testing environments.
    AdvancedCostOptimizer = None  # type: ignore

# NEW IMPORT – Learning Engine predictor
try:
    from learning.mcp_learning_engine import MCPLearningEngine, PerformancePrediction
except (ImportError, ModuleNotFoundError):
    MCPLearningEngine = None  # type: ignore
    PerformancePrediction = None  # type: ignore

class IntelligentMCCOrchestrator:
    """
    Orchestrates complex, multi-MCP workflows by dynamically composing,
    executing, and optimizing chains of MCPs.
    """

    def __init__(
        self,
        sequential_thinking_mcp: Any,
        mcp_performance_analytics: Any,
        mcp_registry: Dict[str, Any],
        *,
        learning_engine: Optional["MCPLearningEngine"] = None,
        cost_optimizer: Optional["AdvancedCostOptimizer"] = None,
        default_budget: Optional[float] = None,
    ):
        """Create an orchestrator instance.

        Args:
            sequential_thinking_mcp: Component responsible for decomposing a task prompt into subtasks.
            mcp_performance_analytics: Component that can estimate cost/latency for MCP execution plans.
            mcp_registry: A mapping of MCP identifiers to fully-functional MCP instances. **Must** contain
                the keys expected by `_select_mcp_for_subtask` or override that method.
            learning_engine: Optional MCPLearningEngine instance for adaptive candidate selection.
            cost_optimizer: Optional AdvancedCostOptimizer instance for cost optimization.
            default_budget: Optional default budget for cost optimization.
        """

        if not mcp_registry:
            raise ValueError("mcp_registry cannot be empty. Supply real MCP instances.")

        self.sequential_thinking_mcp = sequential_thinking_mcp
        self.mcp_performance_analytics = mcp_performance_analytics
        self.mcp_registry = mcp_registry
        self.mcp_chain = MCPChain()

        # Optional AdvancedCostOptimizer
        self.cost_optimizer = cost_optimizer
        self.default_budget = default_budget

        # NEW param
        self.learning_engine = learning_engine  # May be None

    async def execute_task(self, task_prompt: str, *, budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes a task by orchestrating a dynamic workflow of MCPs.

        Args:
            task_prompt: The user's task prompt.
            budget: Optional budget for cost optimization.

        Returns:
            A dictionary containing the results of the task execution.
        """
        # 1. Decompose task into subtasks using Sequential Thinking MCP
        subtasks = self.sequential_thinking_mcp.decompose(task_prompt)

        # 2. Generate dynamic execution graph
        execution_graph = self._generate_execution_graph(subtasks)

        # 3a. (Optional) Pre-flight cost estimation using AdvancedCostOptimizer
        if self.cost_optimizer is not None:
            workflow_steps = self._graph_to_workflow(execution_graph)
            effective_budget = budget if budget is not None else self.default_budget

            try:
                cost_result = await self.cost_optimizer.optimize_and_execute(
                    workflow_steps,
                    budget=effective_budget,
                )

                if cost_result.get("status") == "error":
                    # Abort early and surface cost error
                    logger.warning(
                        "Task aborted due to budget constraint", extra=cost_result
                    )
                    return {
                        "status": "aborted",
                        "reason": "budget_exceeded",
                        "details": cost_result,
                    }

                # Attach cost metadata for later reference
                execution_graph["estimated_cost"] = cost_result.get("estimated_cost")
                execution_graph["cost_recommendations"] = cost_result.get("recommendations", [])

            except Exception as exc:  # pragma: no cover
                # Optimizer failed – continue without blocking task execution
                logger.exception("Cost optimization failed – proceeding without it: %s", exc)

        # 3b. Perform cost-benefit analysis using legacy analytics
        optimized_graph = self._optimize_workflow(execution_graph)

        # 4. Execute the workflow
        results = await self._execute_workflow(optimized_graph)

        # 5. Append cost metadata (if available)
        if "estimated_cost" in optimized_graph:
            results["estimated_cost"] = optimized_graph["estimated_cost"]
            results["cost_recommendations"] = optimized_graph.get("cost_recommendations", [])

        return results

    def _generate_execution_graph(self, subtasks: List[Any]) -> Dict[str, Any]:
        """
        Generates a dynamic execution graph from a list of subtasks.
        Handles parallel execution paths specified in the subtask list.
        """
        execution_graph = {'nodes': [], 'edges': []}
        node_map = {}
        
        # First pass: create nodes
        for i, subtask in enumerate(subtasks):
            if isinstance(subtask, str):
                mcp_name = f"mcp_{i}"
                node = {'name': mcp_name, 'subtask': subtask, 'dependencies': []}
                execution_graph['nodes'].append(node)
                node_map[subtask] = mcp_name
            elif isinstance(subtask, dict) and 'parallel' in subtask:
                for parallel_subtask in subtask['parallel']:
                    mcp_name = f"mcp_{i}_{parallel_subtask.replace(' ', '_')}"
                    node = {'name': mcp_name, 'subtask': parallel_subtask, 'dependencies': []}
                    execution_graph['nodes'].append(node)
                    node_map[parallel_subtask] = mcp_name

        # Second pass: create edges (dependencies)
        for i, subtask in enumerate(subtasks):
            if i > 0:
                prev_subtask = subtasks[i-1]
                if isinstance(subtask, str):
                    current_mcp_name = node_map[subtask]
                    if isinstance(prev_subtask, str):
                        prev_mcp_name = node_map[prev_subtask]
                        execution_graph['edges'].append({'from': prev_mcp_name, 'to': current_mcp_name})
                    elif isinstance(prev_subtask, dict) and 'parallel' in prev_subtask:
                        for parallel_subtask in prev_subtask['parallel']:
                            prev_mcp_name = node_map[parallel_subtask]
                            execution_graph['edges'].append({'from': prev_mcp_name, 'to': current_mcp_name})
                elif isinstance(subtask, dict) and 'parallel' in subtask:
                     if isinstance(prev_subtask, str):
                        prev_mcp_name = node_map[prev_subtask]
                        for parallel_subtask in subtask['parallel']:
                            current_mcp_name = node_map[parallel_subtask]
                            execution_graph['edges'].append({'from': prev_mcp_name, 'to': current_mcp_name})

        # Update dependencies in nodes
        for edge in execution_graph['edges']:
            for node in execution_graph['nodes']:
                if node['name'] == edge['to']:
                    node['dependencies'].append(edge['from'])

        return execution_graph

    def _optimize_workflow(self, execution_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimizes the MCP workflow using cost-benefit analysis.
        """
        logger.info("--- Optimizing Workflow ---")
        total_estimated_cost = 0
        total_estimated_latency = 0

        for node in execution_graph['nodes']:
            mcp_name = node['name']
            # In a real implementation, we would have more sophisticated cost/latency models
            cost, latency = self.mcp_performance_analytics.get_estimate(mcp_name)
            total_estimated_cost += cost
            total_estimated_latency += latency
            logger.debug(
                "Cost / latency estimate", extra={"mcp": mcp_name, "cost": cost, "latency": latency}
            )

        logger.info(
            "Optimization summary", extra={
                "total_cost": total_estimated_cost,
                "total_latency": total_estimated_latency,
            }
        )
        logger.info("--- Optimization Complete ---")

        # Placeholder for actual optimization logic, like suggesting cheaper MCPs
        # For now, we just return the original graph
        return execution_graph

    async def _execute_workflow(self, execution_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the MCP workflow, handling parallel execution and dependencies.
        """
        completed_mcps: Dict[str, Any] = {}
        execution_results: Dict[str, Any] = {}

        # Keep looping until every node has completed execution
        while len(completed_mcps) < len(execution_graph["nodes"]):
            # Identify all nodes whose dependencies are satisfied and are not yet scheduled or completed
            ready_nodes = [
                node for node in execution_graph["nodes"]
                if node["name"] not in completed_mcps
                and all(dep in completed_mcps for dep in node["dependencies"])
            ]

            if not ready_nodes:
                logger.error("Deadlock detected – no tasks ready but workflow not complete")
                raise RuntimeError("Workflow execution error: Cycle detected or dependencies not met.")

            # Create tasks for all ready nodes concurrently
            async def _run_node(node):
                mcp_name = node["name"]
                subtask = node["subtask"]
                dependencies = [completed_mcps[dep] for dep in node["dependencies"]]

                mcp = self._select_mcp_for_subtask(subtask)
                logger.info(
                    "Executing MCP", extra={"mcp": mcp_name, "subtask": subtask, "deps": node["dependencies"]}
                )
                result = await mcp.execute(subtask, dependencies)
                return mcp_name, result

            # Run the batch of ready nodes concurrently
            batch_results = await asyncio.gather(*[_run_node(node) for node in ready_nodes])

            # Persist results and mark tasks complete
            for mcp_name, result in batch_results:
                self.mcp_chain.add_result(mcp_name, result)
                completed_mcps[mcp_name] = result
                execution_results[mcp_name] = result

        return execution_results

    # ---------------------------------------------------------------------
    # Helper Utilities
    # ---------------------------------------------------------------------

    def _select_mcp_for_subtask(self, subtask: str):
        """Selects the appropriate MCP instance for a given subtask description.

        This is currently a heuristic keyword matcher. In production, this could
        be replaced with a more sophisticated semantic matcher or a lookup
        table populated by the Sequential Thinking MCP.
        """

        # Begin by identifying candidate MCP keys via heuristic
        candidates: List[str] = []
        subtask_lower = subtask.lower()
        if "analyz" in subtask_lower:
            candidates = [k for k in self.mcp_registry.keys() if "analyze" in k]
        elif "historical" in subtask_lower:
            candidates = [k for k in self.mcp_registry.keys() if "search_historical" in k]
        elif "visitor" in subtask_lower:
            candidates = [k for k in self.mcp_registry.keys() if "search_visitor" in k]
        elif "synthes" in subtask_lower:
            candidates = [k for k in self.mcp_registry.keys() if "synthesize" in k]
        # fallback single mapping
        if not candidates:
            raise ValueError(f"Unable to map subtask to MCP candidates: '{subtask}'")

        # If only one candidate, return directly
        if len(candidates) == 1 or self.learning_engine is None:
            selected_key = candidates[0]
        else:
            # Use learning engine predictor to choose best among candidates
            predictions = []
            for mcp_key in candidates:
                try:
                    pred: PerformancePrediction = self.learning_engine.predict_performance(
                        mcp_key, task_complexity=0.5  # Heuristic; could compute complexity per subtask
                    )
                    score = pred.success_rate - 0.3 * (pred.cost)  # simple multi-objective
                    predictions.append((mcp_key, score, pred))
                except Exception as exc:
                    logger.warning("Prediction failed for %s: %s", mcp_key, exc)
            if predictions:
                selected_key = max(predictions, key=lambda t: t[1])[0]
            else:
                selected_key = candidates[0]

        if selected_key not in self.mcp_registry:
            raise KeyError(f"MCP '{selected_key}' not found in supplied registry")
        return self.mcp_registry[selected_key]

    def _graph_to_workflow(self, execution_graph: Dict[str, Any]):
        """Converts an execution graph to a linear list of workflow steps.

        For cost estimation we only need a rough representation of the work
        to be performed. Each MCP node is treated as an ``mcp_call`` step with
        a symbolic cost, preserving their original order of appearance.
        """

        workflow = []
        for node in execution_graph.get("nodes", []):
            workflow.append({
                "type": "mcp_call",
                "name": node["name"],
            })
        return workflow

class MCPChain:
    """
    Manages the flow of data between chained MCPs.
    """

    def __init__(self):
        self.results = {}

    def add_result(self, mcp_name: str, result: Any):
        """
        Adds the result of an MCP to the chain.
        """
        self.results[mcp_name] = result

    def get_result(self, mcp_name: str) -> Any:
        """
        Retrieves the result of a previously executed MCP.
        """
        return self.results.get(mcp_name)

# The module intentionally contains **no** demonstration code or dummy objects.
# Supply actual implementations via dependency injection when instantiating
# `IntelligentMCCOrchestrator` in your application entry-point.
