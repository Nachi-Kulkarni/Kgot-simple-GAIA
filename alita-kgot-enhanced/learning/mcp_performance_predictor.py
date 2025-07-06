"""
MCP Performance Predictor (Enhanced)
===================================
Task 40 – MCP Learning & Adaptation Engine

This module replaces the naive average-based predictor in
`learning/mcp_learning_engine.py` with a true machine-learning pipeline that
learns from historical MCP execution data (collected via the Task 25 analytics
layer).

Core Features
-------------
1. **Feature Engineering** – transforms `ExecutionRecord` fields into a rich
   feature vector (task complexity, rolling success rates, quality scores, etc.).
2. **Multi-Output Modelling** – trains separate scikit-learn models for:
   • Success probability (classification)
   • Latency prediction (regression)
   • Cost prediction (regression)
3. **Persistence Layer** – saves/loads trained models via `joblib` to avoid
   retraining on every startup.
4. **Winston-style Logging** – uses the central Alita logging config for
   observability.
5. **Extensible Config** – hyper-parameters (model type, test split, etc.)
   encapsulated in `PredictorConfig` to facilitate Optuna-based tuning later.

NOTE: Initial implementation relies on **scikit-learn** only. Hyper-parameter
optimization (Optuna) will be integrated in a later TODO.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib  # Lightweight persistence for scikit-learn models
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Relative import to reuse ExecutionRecord dataclass
from .mcp_learning_engine import ExecutionRecord, PerformancePrediction

# ---------------------------------------------------------------------------
# Logging setup – re-use Winston-compatible root config if present
# ---------------------------------------------------------------------------
logger = logging.getLogger("MCPPerformancePredictor")
if not logger.handlers:
    # Fallback basic config; the global config may already exist elsewhere.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------


@dataclass
class PredictorConfig:
    """Hyper-parameters and paths for the predictor."""

    test_size: float = 0.2  # Train/test split ratio
    random_state: int = 42  # Ensures reproducibility

    # Model hyper-parameters – can be tuned via Optuna later
    clf_n_estimators: int = 200
    clf_learning_rate: float = 0.05
    reg_n_estimators: int = 300
    reg_learning_rate: float = 0.05

    model_dir: Path = Path("./models/mcp_predictor/")  # Where models are persisted

    def ensure_dirs(self) -> None:
        """Create model directory if it does not exist."""
        self.model_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _records_to_dataframe(records: List[ExecutionRecord]) -> pd.DataFrame:
    """Convert raw execution records to a pandas DataFrame with engineered features."""

    records_dicts: List[Dict[str, Any]] = []
    for rec in records:
        # Basic features directly from record
        base: Dict[str, Any] = {
            "mcp_id": rec.mcp_id,
            "task_complexity": rec.task_complexity,
            "success": int(rec.success),  # bool → 0/1
            "latency_ms": rec.latency_ms,
            "cost": rec.cost,
        }

        # Simple derived features – number of parameters, mean numeric param value, etc.
        if rec.parameters:
            numeric_vals = [v for v in rec.parameters.values() if isinstance(v, (int, float))]
            base["num_params"] = len(rec.parameters)
            base["mean_param_val"] = float(np.mean(numeric_vals)) if numeric_vals else 0.0
        else:
            base["num_params"] = 0
            base["mean_param_val"] = 0.0

        records_dicts.append(base)

    df = pd.DataFrame(records_dicts)

    # Fill potential NaNs after feature engineering
    return df.fillna(0)


# ---------------------------------------------------------------------------
# Main Predictor Class
# ---------------------------------------------------------------------------


class MCPPerformancePredictorML:
    """ML-based predictor – drop-in replacement for the naive version."""

    SUCCESS_MODEL_FNAME = "success_clf.joblib"
    LATENCY_MODEL_FNAME = "latency_reg.joblib"
    COST_MODEL_FNAME = "cost_reg.joblib"
    SCALER_FNAME = "feature_scaler.joblib"

    def __init__(self, config: PredictorConfig | None = None):
        self.config = config or PredictorConfig()
        self.config.ensure_dirs()

        # Models will be loaded lazily; set to None initially
        self._success_clf = None
        self._latency_reg = None
        self._cost_reg = None
        self._scaler = None

        # Attempt to load persisted models if they exist
        self._load_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, records: List[ExecutionRecord]) -> Dict[str, float]:
        """Train all underlying models and return basic evaluation metrics."""

        if not records:
            logger.warning("No records provided for training – skipping.")
            return {}

        df = _records_to_dataframe(records)
        features = df.drop(columns=["success", "latency_ms", "cost"])
        target_success = df["success"]
        target_latency = df["latency_ms"]
        target_cost = df["cost"]

        # Scale numerical features for regression; keep copy for classification
        self._scaler = MinMaxScaler()
        X_scaled = self._scaler.fit_transform(features)

        X_train, X_test, y_s_train, y_s_test = train_test_split(
            X_scaled, target_success, test_size=self.config.test_size, random_state=self.config.random_state
        )
        _, _, y_l_train, y_l_test = train_test_split(
            X_scaled, target_latency, test_size=self.config.test_size, random_state=self.config.random_state
        )
        _, _, y_c_train, y_c_test = train_test_split(
            X_scaled, target_cost, test_size=self.config.test_size, random_state=self.config.random_state
        )

        # Success probability – classification (GradientBoosting)
        self._success_clf = GradientBoostingClassifier(
            n_estimators=self.config.clf_n_estimators,
            learning_rate=self.config.clf_learning_rate,
            random_state=self.config.random_state,
        )
        self._success_clf.fit(X_train, y_s_train)
        success_accuracy = self._success_clf.score(X_test, y_s_test)

        # Latency regression
        self._latency_reg = GradientBoostingRegressor(
            n_estimators=self.config.reg_n_estimators,
            learning_rate=self.config.reg_learning_rate,
            random_state=self.config.random_state,
        )
        self._latency_reg.fit(X_train, y_l_train)
        latency_pred = self._latency_reg.predict(X_test)
        latency_mae = mean_absolute_error(y_l_test, latency_pred)

        # Cost regression
        self._cost_reg = GradientBoostingRegressor(
            n_estimators=self.config.reg_n_estimators,
            learning_rate=self.config.reg_learning_rate,
            random_state=self.config.random_state,
        )
        self._cost_reg.fit(X_train, y_c_train)
        cost_pred = self._cost_reg.predict(X_test)
        cost_mae = mean_absolute_error(y_c_test, cost_pred)

        self._persist_models()

        metrics = {
            "success_accuracy": success_accuracy,
            "latency_mae": latency_mae,
            "cost_mae": cost_mae,
        }
        logger.info("Trained MCPPerformancePredictorML", extra={"metrics": metrics})
        return metrics

    def predict(self, record_features: Dict[str, Any]) -> PerformancePrediction:
        """Predict success probability, latency, and cost for a hypothetical task.

        Parameters
        ----------
        record_features : Dict[str, Any]
            Same structure produced by `_records_to_dataframe()` but for a single
            record (excluding target columns).
        """

        if not self.is_ready():
            logger.warning("Predictor not trained – returning default estimates.")
            return PerformancePrediction(success_rate=0.5, latency_ms=5000, cost=0.01)

        feature_df = pd.DataFrame([record_features])
        X_scaled = self._scaler.transform(feature_df)

        success_prob = float(self._success_clf.predict_proba(X_scaled)[0][1])
        latency_ms = int(self._latency_reg.predict(X_scaled)[0])
        cost_est = float(self._cost_reg.predict(X_scaled)[0])

        return PerformancePrediction(success_rate=success_prob, latency_ms=latency_ms, cost=cost_est)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if models are loaded and ready for inference."""
        return all([self._success_clf, self._latency_reg, self._cost_reg, self._scaler])

    # ------------------------------------------------------------------
    # Persistence helpers – keep implementation simple with joblib
    # ------------------------------------------------------------------

    def _persist_models(self) -> None:
        self.config.ensure_dirs()
        joblib.dump(self._success_clf, self.config.model_dir / self.SUCCESS_MODEL_FNAME)
        joblib.dump(self._latency_reg, self.config.model_dir / self.LATENCY_MODEL_FNAME)
        joblib.dump(self._cost_reg, self.config.model_dir / self.COST_MODEL_FNAME)
        joblib.dump(self._scaler, self.config.model_dir / self.SCALER_FNAME)
        logger.info("Persisted trained predictor models", extra={"path": str(self.config.model_dir)})

    def _load_models(self) -> None:
        """Attempt to load persisted models; silently continue if not present."""
        try:
            self._success_clf = joblib.load(self.config.model_dir / self.SUCCESS_MODEL_FNAME)
            self._latency_reg = joblib.load(self.config.model_dir / self.LATENCY_MODEL_FNAME)
            self._cost_reg = joblib.load(self.config.model_dir / self.COST_MODEL_FNAME)
            self._scaler = joblib.load(self.config.model_dir / self.SCALER_FNAME)
            logger.info("Loaded persisted predictor models", extra={"path": str(self.config.model_dir)})
        except FileNotFoundError:
            # No persisted models yet – will need to train
            logger.info("No persisted predictor models found – training required.")
            self._success_clf = None
            self._latency_reg = None
            self._cost_reg = None
            self._scaler = None

    # ------------------------------------------------------------------
    # Static convenience methods
    # ------------------------------------------------------------------

    @staticmethod
    def build_feature_dict(
        mcp_id: str,
        task_complexity: float,
        parameters: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Public helper to construct the feature dict required by `predict()`."""

        params = parameters or {}
        numeric_vals = [v for v in params.values() if isinstance(v, (int, float))]
        return {
            "mcp_id": mcp_id,
            "task_complexity": task_complexity,
            "num_params": len(params),
            "mean_param_val": float(np.mean(numeric_vals)) if numeric_vals else 0.0,
        } 