"""
Data types, exceptions, and Task handles for the Parameter Server client library.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import numpy as np

# Keys can be a Python range, slice, list of ints, or numpy array
KeysLike = Union[range, slice, List[int], np.ndarray]


# --- Coursework Error Hierarchy (Section 5.2) ---
class PSError(Exception):
    """Base class for all errors raised by the parameter server client library."""


class InvalidArgumentError(PSError):
    """Raised on invalid key ranges, empty keys, or dimension/width mismatches."""


class DependencyFailedError(PSError):
    """Raised when an antecedent task declared in deps fails."""


class PSTimeoutError(PSError, TimeoutError):
    """Raised when wait() expires before the task completes."""


class ServerUnavailableError(PSError):
    """Raised when connection to the parameter server cannot be established."""


# --- Core Data Structures Matching Section 5.2 ---
@dataclass(frozen=True)
class TaskId:
    """Unique task identifier (worker_id, seq) for idempotence."""
    worker_id: int
    seq: int


@dataclass(frozen=True)
class Range:
    """Half-open interval [begin, end) of parameter keys."""
    begin: int
    end: int


@dataclass
class Values:
    """Numeric parameter vectors returned by a pull."""
    keys: List[int]
    data: np.ndarray
    width: int


class Task:
    """
    Handle returned immediately by Client.push() and Client.pull().
    A task may be awaited or queried for completion without blocking.
    """

    def __init__(
        self,
        client: Any,
        task_id: TaskId,
        kind: str,
        n_keys: int,
    ):
        self._client = client
        self.task_id = task_id
        self.kind = kind  # 'push' or 'pull'
        self._n_keys = n_keys
        self._cached_result: Optional[Union[np.ndarray, bool]] = None

    def wait(self, timeout_ms: int = 0) -> Optional[np.ndarray]:
        """
        Blocks until task completion or timeout.
        For pull: returns weights as an (n_keys, width) numpy array.
        For push: returns None.
        A timed-out task is still running; calling wait again is legal.
        """
        if self._cached_result is not None:
            return None if self._cached_result is True else self._cached_result

        result = self._client._wait(self, timeout_ms=timeout_ms)

        if self.kind == "pull":
            # Reshape raw 1D float array into (n_keys, width) matrix
            data = np.asarray(result.values.data, dtype=np.float64)
            width = result.values.width if result.values.width > 0 else 1
            reshaped = data.reshape((self._n_keys, width)) if width > 1 else data
            self._cached_result = reshaped
            return self._cached_result
        else:
            self._cached_result = True
            return None

    def done(self) -> bool:
        """Non-blocking completion query."""
        if self._cached_result is not None:
            return True
        try:
            self.wait(timeout_ms=1)
            return True
        except PSTimeoutError:
            return False