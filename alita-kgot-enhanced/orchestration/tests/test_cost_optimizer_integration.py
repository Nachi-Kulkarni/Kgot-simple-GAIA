import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from orchestration.intelligent_mcp_orchestration import IntelligentMCCOrchestrator


class DummySequentialThinking:
    """Minimal stub for sequential thinking MCP."""

    def decompose(self, prompt):  # pylint: disable=unused-argument
        return ["simple subtask"]


class DummyPerformanceAnalytics:
    """Stub that returns constant cost/latency for any MCP."""

    def get_estimate(self, _mcp_name):  # pylint: disable=unused-argument
        return 0.01, 0.1


class TestCostOptimizerIntegration(IsolatedAsyncioTestCase):
    async def test_budget_enforcement_aborts_execution(self):
        """Orchestrator should abort task when optimizer reports budget exceeded."""

        # Prepare dummy cost optimizer with desired behaviour
        mock_optimizer = SimpleNamespace()
        mock_optimizer.optimize_and_execute = AsyncMock(
            return_value={
                "status": "error",
                "message": "Estimated cost $0.10 exceeds budget $0.05",
                "recommendations": [],
            }
        )

        orchestrator = IntelligentMCCOrchestrator(
            sequential_thinking_mcp=DummySequentialThinking(),
            mcp_performance_analytics=DummyPerformanceAnalytics(),
            mcp_registry={
                "mcp_analyze": SimpleNamespace(execute=AsyncMock(return_value="ok"))
            },
            cost_optimizer=mock_optimizer,
            default_budget=0.05,
        )

        result = await orchestrator.execute_task("Analyze data", budget=0.05)
        self.assertEqual(result.get("status"), "aborted")
        self.assertEqual(result.get("reason"), "budget_exceeded")
        mock_optimizer.optimize_and_execute.assert_awaited() 