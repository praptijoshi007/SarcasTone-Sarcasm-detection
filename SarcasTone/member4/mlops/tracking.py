"""MLflow Experiment Tracking and Model Versioning for SarcasTone.

Tracks:
- Multimodal training and evaluation metrics (Macro F1, ECE, PSID score)
- Hyperparameters and ablation experiment flags
- Model versioning and registry transitions
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

log = logging.getLogger(__name__)


class ExperimentTracker:
    """Wrapper managing experiment tracking and model registration."""

    def __init__(
        self,
        experiment_name: str = "SarcasTone_Multimodal",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "file:./outputs/mlruns")
        self.enabled = HAS_MLFLOW

        if self.enabled:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            log.info("MLflow tracking initialized: %s (%s)", self.experiment_name, self.tracking_uri)
        else:
            log.warning("MLflow not installed in environment; logging to local telemetry.")

    def log_evaluation_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: Optional[list[Path]] = None,
    ) -> Optional[str]:
        """Record evaluation parameters, performance metrics, and artifacts."""
        if not self.enabled:
            log.info("Local Telemetry | Run: %s | Metrics: %s", run_name, metrics)
            return "local_telemetry_run"

        with mlflow.start_run(run_name=run_name) as run:
            # Log params
            for k, v in params.items():
                mlflow.log_param(k, str(v))

            # Log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            # Log artifacts
            if artifacts:
                for art in artifacts:
                    if art.is_file():
                        mlflow.log_artifact(str(art))

            log.info("Logged MLflow run %s with ID %s", run_name, run.info.run_id)
            return run.info.run_id
