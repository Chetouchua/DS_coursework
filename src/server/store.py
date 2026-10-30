import asyncio
from typing import Callable, List, Optional
import numpy as np


def default_sgd(weights: np.ndarray, gradients: np.ndarray, lr: float) -> None:
    """Default SGD update rule: w <- w - eta * g"""
    weights -= lr * gradients  # In-place C-level vectorized update


class ParameterStore:
    def __init__(
        self,
        num_keys: int,
        value_width: int,
        learning_rate: float,
        update_rule: Optional[Callable[[np.ndarray, np.ndarray, float], None]] = None,
    ):
        self.num_keys = num_keys
        self.value_width = value_width
        self.learning_rate = learning_rate
        self.weights = np.zeros((num_keys, value_width), dtype=np.float64)
        self.update_rule = update_rule or default_sgd  # Requirement A1
        self._lock = asyncio.Lock()  # Single async mutex protecting weight aggregation

    async def pull_dense(self, begin: int, end: int) -> np.ndarray:
        async with self._lock:
            # Return flattened copy of the slice
            return self.weights[begin:end, :].ravel().copy()

    async def pull_sparse(self, keys: List[int]) -> np.ndarray:
        async with self._lock:
            return self.weights[keys, :].ravel().copy()

    async def push_dense(self, begin: int, end: int, gradient_flat: List[float]) -> None:
        grad_matrix = np.array(gradient_flat, dtype=np.float64).reshape(-1, self.value_width)
        async with self._lock:
            # Atomic vectorized in-place update without looping in Python
            self.update_rule(self.weights[begin:end, :], grad_matrix, self.learning_rate)

    async def push_sparse(self, keys: List[int], gradient_flat: List[float]) -> None:
        grad_matrix = np.array(gradient_flat, dtype=np.float64).reshape(-1, self.value_width)
        async with self._lock:
            self.update_rule(self.weights[keys, :], grad_matrix, self.learning_rate)