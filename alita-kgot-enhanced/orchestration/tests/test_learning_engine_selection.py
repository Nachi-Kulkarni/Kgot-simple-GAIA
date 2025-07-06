import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from orchestration.intelligent_mcp_orchestration import IntelligentMCCOrchestrator
from learning.mcp_learning_engine import PerformancePrediction


class DummySequentialThinking:
    """Minimal stub for sequential-thinking decomposition."""

    def decompose(self, prompt):  # pylint: disable=unused-argument
        return [prompt]


class DummyPerformanceAnalytics:
    """Returns constant estimates so cost does not influence selection."""

    def get_estimate(self, _mcp_name):  # pylint: disable=unused-argument
        return 0.01, 0.1


class DummyLearningEngine:
    """Provides deterministic performance predictions favouring *fast* MCPs."""

    def predict_performance(self, mcp_id, task_complexity, parameters=None):  # pylint: disable=unused-argument
        if "fast" in mcp_id:
            return PerformancePrediction(success_rate=0.9, latency_ms=1000, cost=0.02)
        return PerformancePrediction(success_rate=0.6, latency_ms=2000, cost=0.03)


class TestAdaptiveMCPSelection(unittest.TestCase):
    """Verify orchestrator selects the candidate favoured by the learning engine."""

    def test_learning_engine_guides_selection(self):
        # MCP instances are simple stubs with async execute methods
        registry = {
            "analyze_fast_mcp": SimpleNamespace(execute=AsyncMock(return_value="ok_fast")),
            "analyze_slow_mcp": SimpleNamespace(execute=AsyncMock(return_value="ok_slow")),
        }

        orchestrator = IntelligentMCCOrchestrator(
            sequential_thinking_mcp=DummySequentialThinking(),
            mcp_performance_analytics=DummyPerformanceAnalytics(),
            mcp_registry=registry,
            learning_engine=DummyLearningEngine(),
        )

        # Access protected helper directly to test candidate selection
        selected_instance = orchestrator._select_mcp_for_subtask("Analyze customer data")
        self.assertIs(selected_instance, registry["analyze_fast_mcp"]) 