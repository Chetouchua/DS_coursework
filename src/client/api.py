"""
Ergonomic client interface for the Parameter Server (Part A & B).
"""
import hashlib
import logging
import threading
from typing import Iterable, List, Optional, Tuple, Union

import grpc
import numpy as np

from src.client.types import (
    DependencyFailedError,
    InvalidArgumentError,
    KeysLike,
    PSError,
    PSTimeoutError,
    Range,
    ServerUnavailableError,
    Task,
    TaskId,
)
from src.generated.ps_pb2 import (
    CallOptions as ProtoCallOptions,
    KeyList as ProtoKeyList,
    PullRequest,
    PushRequest,
    Range as ProtoRange,
    TaskId as ProtoTaskId,
    WaitRequest,
)
from src.generated.ps_pb2_grpc import ServerStub

log = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 128 * 1024 * 1024  # 128 MB

channel_options = [
    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
]


class Client:
    """
    Worker client connection to a parameter server.
    Exposes non-blocking push and pull operations returning Task handles.
    """

    def __init__(self, worker_id: int, server_addr: str = "localhost:50051"):
        self.worker_id = worker_id
        self._server_addr = server_addr
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[ServerStub] = None

        self._seq = 0
        self._seq_lock = threading.Lock()
        self._active_handles: List[Task] = []
        self._registered_key_cache = set()

    def connect(self) -> "Client":
        """Establishes gRPC channel and client stub."""
        self._channel = grpc.insecure_channel(self._server_addr, options=channel_options)
        self._stub = ServerStub(self._channel)
        return self

    def close(self) -> None:
        """Closes channel and releases network resources."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self) -> "Client":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --- Public Parameter Server API ---

    def pull(
        self,
        target: Union[KeysLike, Range],
        iteration: int = 0,
        deps: Iterable[Union[Task, ProtoTaskId]] = (),
    ) -> Task:
        """Submits an asynchronous pull request and returns a Task handle immediately."""
        proto_target, n_keys = self._parse_target(target)
        task_id = self._next_task_id()

        proto_task_id = ProtoTaskId(worker_id=task_id.worker_id, task_id=task_id.seq)
        req = PullRequest(
            task_id=proto_task_id,
            opts=self._build_call_options(deps, iteration),
            **proto_target,
        )

        try:
            self._stub.Pull(req)
        except grpc.RpcError as e:
            raise self._translate_rpc_error(e)

        handle = Task(self, task_id, kind="pull", n_keys=n_keys)
        self._active_handles.append(handle)
        return handle

    def push(
        self,
        target: Union[KeysLike, Range],
        values: Union[List[float], np.ndarray],
        iteration: int = 0,
        deps: Iterable[Union[Task, ProtoTaskId]] = (),
    ) -> Task:
        """Submits an asynchronous push request and returns a Task handle immediately."""
        proto_target, n_keys = self._parse_target(target)
        task_id = self._next_task_id()

        grad_array = np.asarray(values, dtype=np.float64)
        proto_task_id = ProtoTaskId(worker_id=task_id.worker_id, task_id=task_id.seq)
        req = PushRequest(
            task_id=proto_task_id,
            values=grad_array.ravel().tolist(),
            opts=self._build_call_options(deps, iteration),
            **proto_target,
        )

        try:
            self._stub.Push(req)
        except grpc.RpcError as e:
            raise self._translate_rpc_error(e)

        handle = Task(self, task_id, kind="push", n_keys=n_keys)
        self._active_handles.append(handle)
        return handle

    def wait_all(self) -> None:
        """Blocks until all outstanding submitted tasks complete."""
        pending = self._active_handles
        self._active_handles = []
        for task in pending:
            task.wait()

    # --- Internal Helpers ---

    def _wait(self, handle: Task, timeout_ms: int = 0):
        """Internal synchronous call to Server.Wait RPC."""
        proto_task = ProtoTaskId(worker_id=handle.task_id.worker_id, task_id=handle.task_id.seq)
        req = WaitRequest(task_id=proto_task, timeout_ms=timeout_ms)

        try:
            timeout_sec = (timeout_ms / 1000.0) if timeout_ms > 0 else None
            return self._stub.Wait(req, timeout=timeout_sec)
        except grpc.RpcError as e:
            raise self._translate_rpc_error(e)

    def _next_task_id(self) -> TaskId:
        """Thread-safely generate monotonic sequence numbers for idempotency."""
        with self._seq_lock:
            self._seq += 1
            return TaskId(worker_id=self.worker_id, seq=self._seq)

    def _parse_target(self, target: Union[KeysLike, Range]) -> Tuple[dict, int]:
        """Parses target into protobuf Range, KeyList, or cached hash."""
        if isinstance(target, Range):
            if target.end <= target.begin:
                raise InvalidArgumentError(f"Invalid range: [{target.begin}, {target.end})")
            return {"r": ProtoRange(begin=target.begin, end=target.end)}, target.end - target.begin

        if isinstance(target, range) and target.step == 1:
            if len(target) == 0:
                raise InvalidArgumentError("Key range cannot be empty")
            return {"r": ProtoRange(begin=target.start, end=target.stop)}, len(target)

        if isinstance(target, slice):
            if target.start is None or target.stop is None or target.step not in (None, 1):
                raise InvalidArgumentError("Slice target must specify start and stop with step 1")
            return {"r": ProtoRange(begin=target.start, end=target.stop)}, target.stop - target.start

        arr = np.asarray(target, dtype=np.uint64)
        if arr.size == 0:
            raise InvalidArgumentError("Key list cannot be empty")

        if arr.size > 1 and not np.all(arr[1:] > arr[:-1]):
            bad_idx = int(np.argmax(~(arr[1:] > arr[:-1]))) + 1
            raise InvalidArgumentError(
                f"Keys must be strictly increasing, but keys[{bad_idx}]={arr[bad_idx]} <= keys[{bad_idx - 1}]"
            )

        key_bytes = arr.tobytes()
        key_hash = hashlib.blake2b(key_bytes, digest_size=8).digest()

        if key_hash in self._registered_key_cache:
            return {"cached_key_hash": key_hash}, int(arr.size)
        else:
            self._registered_key_cache.add(key_hash)
            return {"k": ProtoKeyList(keys=arr.tolist()), "key_hash": key_hash}, int(arr.size)

    @staticmethod
    def _build_call_options(deps: Iterable[Union[Task, ProtoTaskId]], iteration: int) -> ProtoCallOptions:
        proto_deps = []
        for d in deps:
            if isinstance(d, Task):
                proto_deps.append(ProtoTaskId(worker_id=d.task_id.worker_id, task_id=d.task_id.seq))
            elif isinstance(d, ProtoTaskId):
                proto_deps.append(d)
            else:
                raise InvalidArgumentError(f"deps must contain Task handles, got {type(d).__name__}")
        return ProtoCallOptions(deps=proto_deps, iteration=iteration)

    @staticmethod
    def _translate_rpc_error(e: grpc.RpcError) -> PSError:
        code = e.code()
        msg = e.details() or ""
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            if "dependency" in msg.lower():
                return DependencyFailedError(msg)
            return InvalidArgumentError(msg)
        if code == grpc.StatusCode.ABORTED and "dependency" in msg.lower():
            return DependencyFailedError(msg)
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return PSTimeoutError(msg)
        if code == grpc.StatusCode.UNAVAILABLE:
            return ServerUnavailableError(f"Parameter Server unavailable: {msg}")
        return PSError(f"{code.name}: {msg}")