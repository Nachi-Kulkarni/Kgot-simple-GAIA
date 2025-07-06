"""
This script implements the MCP Learning and Adaptation Engine, a core component
for enabling "maximal self-evolution" as described in the Alita paper. It creates
a continuous feedback loop where the system learns from its actions, predicts
performance, and autonomously tunes its own components (MCPs) to improve over time.

Key Components:
- MCPPerformancePredictor: A simulated ML model to predict MCP success, latency, and cost.
- AdaptiveMCPSelector: Intelligently selects the best MCP for a task based on predictions.
- MCPParameterTuner: Automatically refines MCP parameters based on execution history.
- MCPLearningEngine: Orchestrates the feedback loop, recording results and triggering learning.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Coroutine
from .mcp_performance_predictor import MCPPerformancePredictorML, PredictorConfig  # NEW IMPORT

# NEW import for performance tracker
try:
    from alita_core.mcp_knowledge_base import MCPPerformanceTracker  # type: ignore
except (ImportError, ModuleNotFoundError):
    MCPPerformanceTracker = None  # type: ignore

# ==============================================================================
# 1. SETUP & DATA STRUCTURES
# ==============================================================================

# Configure basic logging to observe the engine's behavior.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class ExecutionRecord:
    """
    Stores the result of a single MCP execution. This data forms the basis
    for all learning and adaptation.

    Attributes:
        mcp_id: The unique identifier for the MCP.
        task_complexity: A normalized value (0.0 to 1.0) representing task difficulty.
        success: Boolean indicating if the execution was successful.
        latency_ms: The time taken for the execution in milliseconds.
        cost: The computed cost of the execution (e.g., in USD).
        timestamp: The UNIX timestamp of the execution.
        parameters: The parameters used for this specific execution.
    """

    mcp_id: str
    task_complexity: float
    success: bool
    latency_ms: int
    cost: float
    timestamp: float
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformancePrediction:
    """
    Holds the predicted performance metrics for an MCP for a future task.

    Attributes:
        success_rate: The predicted probability of success (0.0 to 1.0).
        latency_ms: The predicted execution time in milliseconds.
        cost: The predicted cost.
    """

    success_rate: float
    latency_ms: int
    cost: float


# ==============================================================================
# 2. CORE LEARNING COMPONENTS
# ==============================================================================


class MCPPerformancePredictor:
    """
    Predicts MCP performance based on historical data.

    This class simulates a machine learning model that learns from past
    MCP executions. In a real-world scenario, this would be replaced by a more
    sophisticated model using libraries like scikit-learn, TensorFlow, or PyTorch
    to learn a function of task complexity and other features.
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}  # Stores a "model" for each MCP

    def train(self, records: List[ExecutionRecord]):
        """
        Trains the predictor on historical execution data.

        This method aggregates historical data to build a predictive model for each MCP.

        Args:
            records: A list of ExecutionRecord objects.
        """
        logging.info(f"Training performance predictor with {len(records)} records.")
        mcp_data: Dict[str, List[ExecutionRecord]] = {}
        for r in records:
            mcp_data.setdefault(r.mcp_id, []).append(r)

        for mcp_id, mcp_records in mcp_data.items():
            if not mcp_records:
                continue

            # Calculate simple averages for the simulation
            avg_success = sum(r.success for r in mcp_records) / len(mcp_records)
            avg_latency = sum(r.latency_ms for r in mcp_records) / len(mcp_records)
            avg_cost = sum(r.cost for r in mcp_records) / len(mcp_records)

            self._models[mcp_id] = {
                "avg_success": avg_success,
                "avg_latency": avg_latency,
                "avg_cost": avg_cost,
            }

    def predict(self, mcp_id: str, task_complexity: float) -> PerformancePrediction:
        """
        Predicts the performance of an MCP for a given task.

        Args:
            mcp_id: The ID of the MCP.
            task_complexity: The complexity of the task (0.0 to 1.0).

        Returns:
            A PerformancePrediction object with estimated metrics.
        """
        model = self._models.get(mcp_id)
        if not model:
            # Return default "unknown" values if no model exists for the MCP
            return PerformancePrediction(success_rate=0.5, latency_ms=5000, cost=0.01)

        # Simulate prediction: adjust baseline averages based on task complexity.
        # Higher complexity lowers success rate and increases latency/cost.
        predicted_success = model["avg_success"] * (1 - 0.5 * task_complexity)
        predicted_latency = int(model["avg_latency"] * (1 + task_complexity))
        predicted_cost = model["avg_cost"] * (1 + task_complexity)

        return PerformancePrediction(
            success_rate=max(0.0, min(1.0, predicted_success)),
            latency_ms=predicted_latency,
            cost=predicted_cost,
        )


class AdaptiveMCPSelector:
    """
    Selects the best MCP for a task using performance predictions.

    This component embodies the "maximal self-evolution" principle by
    learning which tools are best suited for a task over time, moving
    beyond static, predefined decision trees. It uses a multi-objective
    optimization approach.
    """

    def __init__(self, predictor: MCPPerformancePredictor):
        self._predictor = predictor
        # Weights for the scoring function, tunable for different strategies
        self.success_weight = 0.7
        self.cost_weight = 0.3

    def select_mcp(
        self, mcp_candidates: List[str], task_complexity: float
    ) -> Optional[str]:
        """
        Selects the optimal MCP from a list of candidates based on a scoring model.

        The score is calculated as: score = w_success * success_rate - w_cost * normalized_cost.

        Args:
            mcp_candidates: A list of MCP IDs that can perform the task.
            task_complexity: The complexity of the task.

        Returns:
            The ID of the selected MCP, or None if no candidates are provided.
        """
        if not mcp_candidates:
            return None

        predictions = [
            (mcp_id, self._predictor.predict(mcp_id, task_complexity))
            for mcp_id in mcp_candidates
        ]

        # Normalize cost for fair comparison in the scoring function
        max_cost = max(p.cost for _, p in predictions) if predictions else 1.0
        max_cost = max(max_cost, 0.0001)  # Avoid division by zero

        scored_mcps = []
        for mcp_id, prediction in predictions:
            normalized_cost = prediction.cost / max_cost
            score = (self.success_weight * prediction.success_rate) - (
                self.cost_weight * normalized_cost
            )
            scored_mcps.append((mcp_id, score))
            logging.info(
                f"  - Candidate: {mcp_id:<20} | Score: {score:6.2f} "
                f"(Success: {prediction.success_rate:.2%}, Cost: ${prediction.cost:.4f})"
            )

        if not scored_mcps:
            return None

        best_mcp_id, best_score = max(scored_mcps, key=lambda item: item[1])
        return best_mcp_id


class MCPParameterTuner:
    """
    Autonomously tunes MCP parameters to optimize performance.

    This component analyzes execution history to find better parameter
    configurations. In a production system, this could leverage libraries
    like Optuna or Hyperopt for more advanced Bayesian optimization instead
    of the simple rule-based approach demonstrated here.
    """

    def __init__(self):
        self._mcp_configs: Dict[str, Dict[str, Any]] = {
            "web_scraper_mcp": {"timeout": 30.0},
            "api_client_mcp": {"retries": 3},
        }

    def get_parameters(self, mcp_id: str) -> Dict[str, Any]:
        """Gets the current parameters for a given MCP."""
        return self._mcp_configs.get(mcp_id, {})

    def tune(self, mcp_id: str, history: List[ExecutionRecord]):
        """
        Analyzes history and suggests new, potentially better, parameters.

        Args:
            mcp_id: The ID of the MCP to tune.
            history: The recent execution history for this MCP.
        """
        if not history:
            return

        # Example of simple, rule-based tuning for a web scraper
        if mcp_id == "web_scraper_mcp":
            failures = sum(not r.success for r in history)
            if (failures / len(history)) > 0.5:
                current_timeout = self._mcp_configs[mcp_id].get("timeout", 30.0)
                new_timeout = round(current_timeout * 1.5, 1)
                logging.info(
                    f"Tuning {mcp_id}: High failure rate detected. "
                    f"Increasing timeout to {new_timeout}s"
                )
                self._mcp_configs[mcp_id]["timeout"] = new_timeout

        logging.info(f"Updated parameters for {mcp_id}: {self._mcp_configs[mcp_id]}")


# ==============================================================================
# 3. MAIN ENGINE ORCHESTRATOR
# ==============================================================================


class MCPLearningEngine:
    """
    Main orchestrator for the continuous learning and adaptation feedback loop.

    This class integrates all other components to implement the "Self-Evolving"
    cycle shown in Alita's architecture. It records execution results, triggers
    retraining of the performance predictor, and initiates parameter tuning.
    """

    def __init__(self, performance_tracker: Optional["MCPPerformanceTracker"] = None, retrain_interval_sec: int = 86400):
        self._predictor: MCPPerformancePredictor = MCPPerformancePredictor()
        self._ml_predictor: MCPPerformancePredictorML = MCPPerformancePredictorML()
        self.selector = AdaptiveMCPSelector(self._predictor)
        self.tuner = MCPParameterTuner()
        self.execution_history: List[ExecutionRecord] = []
        self.performance_tracker = performance_tracker
        self._retrain_interval_sec = retrain_interval_sec
        # start periodic retrain loop if running inside event loop
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._periodic_retrain_loop())
        except RuntimeError:
            # No running loop (e.g., during import)
            pass

    async def _periodic_retrain_loop(self):
        """Periodically retrain predictors and tuner."""
        while True:
            await asyncio.sleep(self._retrain_interval_sec)
            try:
                logging.info("[LearningEngine] Nightly retraining started – records=%d", len(self.execution_history))
                self._predictor.train(self.execution_history)
                self.train_advanced_predictor(self.execution_history)
            except Exception as exc:
                logging.error("Nightly retraining failed: %s", exc)

    def record_execution(self, record: ExecutionRecord):
        """
        Records a new MCP execution result to the history. In a real system,
        this would write to a persistent database (e.g., Prometheus, a SQL DB).
        """
        logging.info(f"Recording execution for {record.mcp_id}")
        self.execution_history.append(record)
        # async record to performance tracker if available
        if self.performance_tracker is not None:
            if hasattr(self.performance_tracker, "record_usage"):
                # schedule without blocking
                asyncio.create_task(
                    self.performance_tracker.record_usage(
                        mcp_name=record.mcp_id,
                        operation_type="task_execution",
                        success=record.success,
                        response_time_ms=record.latency_ms,
                        cost=record.cost,
                        user_rating=0.0,
                        context={"task_complexity": record.task_complexity}
                    )
                )

    async def run_feedback_loop(self):
        """
        Periodically runs the learning and adaptation cycle.
        This would typically be a background process (e.g., a nightly job).
        """
        await asyncio.sleep(0.01) # Simulate async operation
        logging.info("--- Continuous Improvement: Retraining Models ---")
        self._predictor.train(self.execution_history)

        await asyncio.sleep(0.01)
        logging.info("--- Continuous Improvement: Tuning Parameters ---")
        mcps_with_history = {r.mcp_id for r in self.execution_history}
        for mcp_id in mcps_with_history:
            mcp_history = [r for r in self.execution_history if r.mcp_id == mcp_id]
            self.tuner.tune(mcp_id, mcp_history)

    def train_advanced_predictor(self, records: List[ExecutionRecord]):
        """Train the enhanced ML-based predictor and fall back to legacy if insufficient data."""
        if len(records) < 50:
            logging.warning("Not enough records (%d) for ML predictor – skipping advanced training.", len(records))
            return
        metrics = self._ml_predictor.train(records)
        logging.info("Advanced ML predictor trained — metrics: %s", metrics)

    def _select_best_mcp(self, mcp_candidates: List[str], task_complexity: float) -> Optional[str]:
        """Select best MCP using ML predictor if available, else fallback."""
        if self._ml_predictor.is_ready():
            selector = AdaptiveMCPSelector(self._ml_predictor)  # type: ignore[arg-type]
        else:
            selector = self.selector
        return selector.select_mcp(mcp_candidates, task_complexity)

    # NEW PUBLIC HELPER --------------------------------------------------
    def predict_performance(self, mcp_id: str, task_complexity: float, parameters: Optional[Dict[str, Any]] = None) -> PerformancePrediction:
        """Return performance prediction using the best available model."""
        if self._ml_predictor.is_ready():
            feature_dict = MCPPerformancePredictorML.build_feature_dict(mcp_id, task_complexity, parameters)
            return self._ml_predictor.predict(feature_dict)
        # fallback legacy
        return self._predictor.predict(mcp_id, task_complexity)


# ==============================================================================
# 4. SIMULATION
# ==============================================================================


async def main():
    """Main function to run a demonstration of the learning engine."""
    engine = MCPLearningEngine()
    mcp_ids = ["web_scraper_mcp", "api_client_mcp"]

    # --- Phase 1: Initial Data Generation ---
    # Simulate some initial MCP executions to populate the history.
    logging.info("\n--- Simulating Initial MCP Executions ---")
    for _ in range(20):
        mcp_id = random.choice(mcp_ids)
        params = engine.tuner.get_parameters(mcp_id)
        complexity = random.random()
        # Web scraper is less reliable by default in this simulation
        success_chance = 0.9 if mcp_id == "api_client_mcp" else 0.4
        record = ExecutionRecord(
            mcp_id=mcp_id,
            task_complexity=complexity,
            success=random.random() < success_chance,
            latency_ms=random.randint(200, 5000) + int(complexity * 3000),
            cost=random.uniform(0.001, 0.01) + (complexity * 0.005),
            parameters=params,
            timestamp=time.time(),
        )
        engine.record_execution(record)

    # --- Phase 2: Run the Learning Loop ---
    # Trigger the learning process based on the generated data.
    await engine.run_feedback_loop()

    # --- Phase 3: Demonstrate Adaptive Selection ---
    logging.info("\n--- Adaptive MCP Selection Example ---")
    complex_task_complexity = 0.8
    selected_mcp = engine._select_best_mcp(mcp_ids, complex_task_complexity)
    logging.info(
        f"For a complex task, the adaptive engine selected: {selected_mcp}"
    )

    # --- Phase 4: Demonstrate Automated Tuning ---
    logging.info("\n--- Automated Parameter Tuning Example ---")
    # Simulate a failure for the web scraper
    failed_record = ExecutionRecord(
        mcp_id="web_scraper_mcp",
        task_complexity=0.9,
        success=False,
        latency_ms=30000,
        cost=0.01,
        parameters=engine.tuner.get_parameters("web_scraper_mcp"),
        timestamp=time.time(),
    )
    engine.record_execution(failed_record)

    # Rerun the feedback loop to see the tuner react
    await engine.run_feedback_loop()

    # --- Phase 5: Demonstrate Performance Prediction ---
    logging.info("\n--- Performance Prediction Example ---")
    prediction_task_complexity = 0.5
    predicted_perf = engine._predictor.predict(
        "api_client_mcp", prediction_task_complexity
    )
    logging.info(f"Predicted performance for 'api_client_mcp' on a medium task:")
    logging.info(f"  - Success Rate: {predicted_perf.success_rate:.2%}")
    logging.info(f"  - Latency: {predicted_perf.latency_ms}ms")
    logging.info(f"  - Cost: ${predicted_perf.cost:.4f}")


# ==============================================================================
# 5. OPTIONAL FASTAPI SERVICE (Real-Time Prediction API)
# ==============================================================================
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    _app: Optional[FastAPI] = None

    class _PredictionRequest(BaseModel):
        mcp_id: str
        task_complexity: float
        parameters: Optional[Dict[str, Any]] = None

    def create_prediction_app(engine: MCPLearningEngine) -> FastAPI:
        """Create a FastAPI app exposing /predict endpoint."""
        app = FastAPI(title="MCP Learning Engine – Prediction API")

        @app.post("/predict")
        async def predict(req: _PredictionRequest):
            try:
                pred = engine.predict_performance(req.mcp_id, req.task_complexity, req.parameters)
                return {
                    "mcp_id": req.mcp_id,
                    "success_rate": pred.success_rate,
                    "latency_ms": pred.latency_ms,
                    "cost": pred.cost,
                }
            except Exception as e:
                logging.error("Prediction API error: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

        return app

except ImportError:  # pragma: no cover – fastapi optional
    logging.warning("FastAPI not available – Real-time prediction API disabled.")
    _app = None


if __name__ == "__main__":
    asyncio.run(main())
