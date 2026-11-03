#Maybe it would work better on A2 task?
import logging
from typing import Iterable, List, Optional, Union
import grpc 
import numpy as np
import hashlib 
from grpc_generated.server_pb2 import (
    CallOptions, KeyList, PullRequest, PushRequest, Range, Result, Task, Values, WaitRequest)
from grpc_generated.server_pb2_grpc import ServerStub 

log = logging.getLogger(__name__)

KeysLike = Union[Range, List[int]]

class PSError(Exception):
    """Base class for all errors raised by this library"""

class InvalidRequestError(PSError):...
class DependencyError(PSError):...
class TaskFailedError(PSError):...
class PSTimeoutError(PSError, TimeoutError):...
class ServerUnavailableError(PSError):...

class Handle:
    #Returned by push/pull. Pass it in deps=[...] or await handle.result().
    def __init__(self, client: "WorkerInterface", task: Task, kind: str):
        self._client, self.task, self.kind, self._n = client, task, kind
        self._cache = None 

    @property
    def ts(self) -> int:
        return self.task.taskId

    async def wait(self, timeout_ms: int = 0) -> Result:
        """Wait for completion and return the full result, including its ack."""
        if self._cache is None:
            self._cache = await self._client.wait(self, timeout_ms)
        return self._cache

    async def result(self, timeout_ms: int = 0) -> Optional[Values]:
        """For pulls, return protobuf Values (keys.keys, data, width); for pushes, None."""
        res = await self.wait(timeout_ms)
        return res.v if self.kind == "pull" else None


class WorkerInterface:
    def __init__(self, worker_id: int, server:str = "localhost:50051"):
        self.worker_id = worker_id
        self._addr = server
        self._channel = None 
        self._stub = None 
        self._handles: List[Handle] = []
        self._agree_key_cache = set()

    def _target(self, keys: KeysLike):
            if isinstance(keys, Range):
                if keys.end <= keys.begin:
                    raise InvalidRequestError("empty or reversed key range")
                return {"r": keys}
            arr = np.asarray(keys, dtype=np.uint64)
            if arr.size == 0:
                raise InvalidRequestError("empty key list")
            if arr.size > 1 and not np.all(arr[1:] > arr[:-1]):
                bad = int(np.argmax(~(arr[1:] > arr[:-1]))) + 1
                raise InvalidRequestError(
                    f"keys must be strictly increasing, but keys[{bad}]={int(arr[bad])} "
                    f"<= keys[{bad-1}]={int(arr[bad-1])}")
    
            key_bytes = arr.tobytes()
            key_hash = hashlib.blake2b(key_bytes, digest_size=8).digest()
            if key_hash in self._agree_key_cache:
                return {"k": KeyList(key_hash=key_hash)}
    
            else:
                return {"k": KeyList(keys=arr.tolist(), key_hash=key_hash)}

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = self._stub = None 

    async def __aenter__(self): return await self.connect()
    async def __aexit__(self, *exc): await self.close()

    #Pulic API
    async def pull(self, keys: KeysLike, deps:Iterable = ()) -> Handle:
        target = self._target(keys)
        req = PullRequest(workerId=self.worker_id, opts=self._opts(deps), **target)
        
        return await self._submit(self._stub.Pull, req, "pull")

    async def push(self, keys: KeysLike, grads, deps: Iterable = ()) -> Handle:
        target = self._target(keys)
        g = np.asarray(grads, dtype=np.float64)
        
        req = PushRequest(workerId=self.worker_id, values=g.ravel().tolist(),
                          opts=self._opts(deps), **target)
        
        return await self._submit(self._stub.Push, req, "push")

    async def connect(self):
        self._channel = grpc.aio.insecure_channel(self._addr)
        self._stub = ServerStub(self._channel)
        return self 

    async def wait(self, handle: "Handle", timeout_ms: int=0) -> Result:
        req = WaitRequest(task=handle.task, timeout_ms=timeout_ms)
        try:
            timeout = timeout_ms / 1000.0 if timeout_ms > 0 else None
            res = await self._stub.Wait(req, timeout = timeout)
            return res 
        except grpc.aio.AioRpcError as e:
            raise self._translate(e)
    
    async def wait_all(self) -> None:
        pending, self._handles = self._handles, []
        for h in pending:
            await h.wait()

    #internals
    async def _submit(self, rpc, req, kind) -> Handle:
        try:
            task = await rpc(req)
        except grpc.aio.AioRpcError as e:
            raise self._translate(e) from e
        if req.WhichOneof("target") == "k" and req.k.key_hash:
            self._agree_key_cache.add(req.k.key_hash)
        h = Handle(self, task, kind)
        self._handles.append(h)
        return h 

    @staticmethod
    def _opts(deps) -> CallOptions:
        tasks = []
        for d in deps:
            if isinstance(d, Handle):
                tasks.append(d.task)
            elif isinstance(d, Task):
                tasks.append(d)
            else:
                raise InvalidRequestError(
                    f"deps must be Handles or Tasks, got {type(d).__name__}")
        return CallOptions(deps=tasks)

    @staticmethod
    def _translate(e: "grpc.aio.AioRpcError") -> PSError:
        code, msg = e.code(), e.details() or ""
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            return DependencyError(msg) if "dependency" in msg else InvalidRequestError(msg)
        if code == grpc.StatusCode.ABORTED:
            return TaskFailedError(msg)
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return PSTimeoutError(msg)
        if code == grpc.StatusCode.UNAVAILABLE:
            return ServerUnavailableError(f"cannot reach server: {msg}")
        return PSError(f"{code.name}: {msg}")





    





