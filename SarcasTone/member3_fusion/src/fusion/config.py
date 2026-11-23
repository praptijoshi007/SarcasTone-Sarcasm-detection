"""Configuration loader for member3_fusion.

Reads configs/fusion_config.yaml and merges experiment-specific overrides.
All consumers import from this module to get a single source of truth.

Fix: experiment blocks now contain a nested ``model:`` dict, so the deep-merge
correctly propagates ablation flags (fusion_strategy, psid_enabled, etc.)
into the model config block rather than dumping them at the top level.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    import sys
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    import yaml_fallback as yaml

# member3_fusion/src/fusion/config.py
# parents[0] = fusion/, parents[1] = src/, parents[2] = member3_fusion/, parents[3] = SarcasTone/
_REPO_ROOT     = Path(__file__).resolve().parents[3]           # SarcasTone/
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fusion_config.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    config_path: Path | str | None = None,
    experiment: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict:
    """Load fusion config, optionally merging an experiment block.

    Args:
        config_path: Path to a YAML config file. Defaults to
            ``member3_fusion/configs/fusion_config.yaml``.
        experiment: Name of a key under ``experiments:`` to merge on top of
            the base config (e.g. ``"full_model"``).
            The experiment block is expected to contain a nested ``model:`` dict
            whose keys override the base ``model:`` section.
        overrides: Flat or nested dict of key=value pairs to apply last.

    Returns:
        Merged config dict. Relative ``data:`` and ``output:`` paths are
        resolved against the repository root.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    cfg  = _load_yaml(path)

    if experiment:
        exp_block = cfg.get("experiments", {}).get(experiment)
        if exp_block is None:
            raise KeyError(
                f"Experiment '{experiment}' not found in {path}. "
                f"Available: {list(cfg.get('experiments', {}).keys())}"
            )
        cfg = _deep_merge(cfg, exp_block)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    # Resolve relative data/output paths against repo root
    cfg = _resolve_paths(cfg, _REPO_ROOT)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a deep copy of base.

    Both dicts may contain nested dicts; they are merged recursively.
    Non-dict values in override replace those in base.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_paths(cfg: dict, repo_root: Path) -> dict:
    """Resolve relative paths under ``data:`` and ``output:`` sections."""
    for section in ("data", "output"):
        block = cfg.get(section, {})
        for key, value in block.items():
            is_path_key = key.endswith(
                ("_path", "_csv", "_json", "_npz", "_dir")
            )
            if (
                is_path_key
                and isinstance(value, str)
                and value
                and not os.path.isabs(value)
            ):
                cfg[section][key] = (repo_root / value).as_posix()
    return cfg


def get_repo_root() -> Path:
    return _REPO_ROOT


def get_checkpoint_dir(cfg: dict) -> Path:
    p = Path(cfg["output"]["checkpoint_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_experiment_dir(cfg: dict, experiment: str) -> Path:
    p = Path(cfg["output"]["experiments_dir"]) / experiment
    p.mkdir(parents=True, exist_ok=True)
    return p
