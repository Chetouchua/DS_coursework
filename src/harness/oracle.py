"""
Correctness oracle: hidden target vector generation and analytical validation.
"""
import os
import numpy as np
from src.harness.config import ModelConfig


class CorrectnessOracle:
    def __init__(self, model_cfg: ModelConfig, seed: Optional[int] = None, data_dir: str = "data"):
        self.num_keys = model_cfg.num_keys
        self.value_width = model_cfg.value_width
        self.data_dir = data_dir
        self.rng = np.random.default_rng(seed)

        # Generate target w* vector
        self.w_star = self.rng.normal(
            loc=model_cfg.target.mean,
            scale=model_cfg.target.std,
            size=(self.num_keys, self.value_width),
        ).astype(np.float64)

    def save(self, experiment_name: str) -> str:
        """Saves target weights to disk for workers to reference."""
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, f"{experiment_name}.npy")
        np.save(path, self.w_star)
        return path

    def expected_distance(self, initial_error: float, eta: float, num_workers: int, iteration: int) -> float:
        """Analytic expectation under sequential consistency:
        ||w(t) - w*|| = (|1 - eta * m|)^t * ||w(0) - w*||
        """
        contraction = abs(1.0 - eta * num_workers)
        return (contraction**iteration) * initial_error