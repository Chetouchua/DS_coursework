import asyncio
from typing import Callable, List, Optional, Sequence, Union
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
        # Store weights in C-contiguous memory
        self.weights = np.zeros((num_keys, value_width), dtype=np.float64, order="C")
        self.update_rule = update_rule or default_sgd
        self._lock = asyncio.Lock()

    async def pull_dense(self, begin: int, end: int) -> np.ndarray:
        async with self._lock:
            # Flattened copy prevents race conditions if caller mutates it
            return self.weights[begin:end, :].flatten()

    async def pull_sparse(self, keys: Sequence[int]) -> np.ndarray:
        async with self._lock:
            return self.weights[keys, :].flatten()

    async def push_dense(
        self, begin: int, end: int, gradient_flat: Union[Sequence[float], np.ndarray]
    ) -> None:
        grad_matrix = np.asarray(gradient_flat, dtype=np.float64).reshape(-1, self.value_width)
        async with self._lock:
            # begin:end is a slice/view: in-place modification mutates self.weights directly
            self.update_rule(self.weights[begin:end, :], grad_matrix, self.learning_rate)

    async def push_sparse(
        self, keys: Sequence[int], gradient_flat: Union[Sequence[float], np.ndarray]
    ) -> None:
        grad_matrix = np.asarray(gradient_flat, dtype=np.float64).reshape(-1, self.value_width)
        key_indices = np.asarray(keys, dtype=np.intp)

        async with self._lock:
            if self.update_rule is default_sgd:
                # Fast path: handles potential duplicate keys properly via atomic accumulation
                np.subtract.at(self.weights, key_indices, self.learning_rate * grad_matrix)
            else:
                # Custom update rule path: extract copy, apply rule, write back
                updated_slice = self.weights[key_indices, :].copy()
                self.update_rule(updated_slice, grad_matrix, self.learning_rate)
                self.weights[key_indices, :] = updated_slice