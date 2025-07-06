import random
import unittest

from learning.mcp_performance_predictor import MCPPerformancePredictorML
from learning.mcp_learning_engine import ExecutionRecord, MCPLearningEngine


class TestMCPPerformancePredictor(unittest.TestCase):
    """Verify that the enhanced ML predictor can train on synthetic data and make predictions."""

    def setUp(self):
        # Generate a balanced synthetic dataset for a dummy MCP
        self.records = []
        for i in range(100):
            self.records.append(
                ExecutionRecord(
                    mcp_id="dummy_mcp",
                    task_complexity=random.random(),
                    success=(i % 2 == 0),  # 50% success rate
                    latency_ms=random.randint(100, 3000),
                    cost=random.uniform(0.001, 0.02),
                    parameters={"p": i},
                    timestamp=0.0,
                )
            )

    def test_train_and_predict(self):
        predictor = MCPPerformancePredictorML()
        predictor.train(self.records)
        self.assertTrue(predictor.is_ready(), "Predictor should be ready after training")

        feature_dict = MCPPerformancePredictorML.build_feature_dict(
            mcp_id="dummy_mcp",
            task_complexity=0.5,
            parameters={"p": 42},
        )
        prediction = predictor.predict(feature_dict)
        # Basic sanity checks on prediction output range
        self.assertGreaterEqual(prediction.success_rate, 0.0)
        self.assertLessEqual(prediction.success_rate, 1.0)
        self.assertGreater(prediction.latency_ms, 0)
        self.assertGreaterEqual(prediction.cost, 0.0)


class TestRuleBasedParameterTuner(unittest.TestCase):
    """Ensure the legacy rule-based tuner increases timeout for failing web_scraper_mcp."""

    def test_timeout_increases_on_failures(self):
        engine = MCPLearningEngine()

        # Inject failing history for the web_scraper_mcp (all failures)
        for _ in range(10):
            engine.execution_history.append(
                ExecutionRecord(
                    mcp_id="web_scraper_mcp",
                    task_complexity=random.random(),
                    success=False,
                    latency_ms=20000,
                    cost=0.01,
                    parameters={},
                    timestamp=0.0,
                )
            )

        # Trigger tuning logic
        engine.tuner.tune("web_scraper_mcp", engine.execution_history)
        tuned_params = engine.tuner.get_parameters("web_scraper_mcp")

        self.assertIn("timeout", tuned_params)
        self.assertGreater(tuned_params["timeout"], 30.0) 