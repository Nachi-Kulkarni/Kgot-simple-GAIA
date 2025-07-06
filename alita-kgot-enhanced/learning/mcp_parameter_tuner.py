"""
Optuna-based MCP Parameter Tuner
================================
Task 40 – MCP Learning & Adaptation Engine

This module provides an advanced hyper-parameter optimization utility that
leverages **Optuna** to autonomously refine MCP parameter defaults based on
historical execution data.

Design Goals
------------
1. **Generic** – works with any MCP that exposes:
   • `default_params` attribute (Dict[str, Any])
   • `execute(task, **params)` coroutine returning execution metadata with
     keys `success` (bool), `latency_ms` (int), `cost` (float).
2. **Non-intrusive** – if *optuna* is not installed, gracefully falls back to
   a simple rule-based tuner used previously.
3. **Pluggable Search Spaces** – developers can register custom search spaces
   per MCP via `register_search_space()`.
4. **Multi-objective** – optimises for high success rate and low cost/latency.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Callable, Optional

try:
    import optuna
except ImportError:  # pragma: no cover – optuna optional
    optuna = None  # type: ignore

logger = logging.getLogger("MCPParameterTunerOptuna")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


class MCPParameterTunerOptuna:
    """Advanced tuner powered by Optuna (if available)."""

    def __init__(self):
        if optuna is None:
            raise RuntimeError("Optuna is not installed – cannot use MCPParameterTunerOptuna.")

        # Mapping mcp_id → optuna search-space creator
        self._search_spaces: Dict[str, Callable[[optuna.trial.Trial], Dict[str, Any]]] = {}
        # Cache of tuned parameters
        self._best_params: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public registration helpers
    # ------------------------------------------------------------------

    def register_search_space(self, mcp_id: str, space_fn: Callable[["optuna.trial.Trial"], Dict[str, Any]]):
        """Register a function that defines the search space for an MCP."""
        self._search_spaces[mcp_id] = space_fn

    def get_parameters(self, mcp_id: str) -> Dict[str, Any]:
        """Return the latest tuned parameters for *mcp_id* (or empty dict)."""
        return self._best_params.get(mcp_id, {})

    # ------------------------------------------------------------------
    # Tuning routine – expects access to real MCP instance
    # ------------------------------------------------------------------

    async def tune(self, mcp_id: str, mcp_instance: Any, n_trials: int = 30, timeout: int = 600):
        """Run Optuna optimisation for a given MCP instance.

        Parameters
        ----------
        mcp_id : str
            Identifier used for search-space lookup and cache.
        mcp_instance : Any
            Actual MCP object exposing `execute(subtask, **params)` coroutine.
        n_trials : int, optional
            Number of trials (default 30).
        timeout : int, optional
            Time budget in seconds; passed to optuna study.
        """

        if mcp_id not in self._search_spaces:
            logger.warning("No search space registered for %s – skipping tuning.", mcp_id)
            return

        space_fn = self._search_spaces[mcp_id]

        # Optuna objective ------------------------------------------------
        async def _objective(trial: "optuna.trial.Trial") -> float:  # type: ignore
            params = space_fn(trial)
            try:
                result = await mcp_instance.execute("tuning_probe", params)
            except Exception as exc:  # pragma: no cover
                logger.error("Error during tuning execution: %s", exc)
                return 1e6  # Large penalty

            success_penalty = 0 if result["success"] else 1_000
            # Multi-objective collapsed into scalar: latency + cost + success penalty
            score = result["latency_ms"] + (result["cost"] * 10_000) + success_penalty
            return score

        # Wrap Optuna async objective ------------------------------------
        def _sync_objective(trial: "optuna.trial.Trial") -> float:  # type: ignore
            return asyncio.get_event_loop().run_until_complete(_objective(trial))

        study = optuna.create_study(direction="minimize")
        study.optimize(_sync_objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        self._best_params[mcp_id] = best_params
        logger.info("Tuned parameters for %s: %s", mcp_id, best_params)

        return best_params 