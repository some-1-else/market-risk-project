from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the transparent baseline risk model."""

    pca_components: int = 3
    final_scenarios: int = 10_000
    backtest_scenarios: int = 3_000
    random_seed: int = 42
    distribution: str = "multivariate_normal"


BASELINE_CONFIG = ModelConfig()

