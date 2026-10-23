import argparse
import asyncio
from collections import defaultdict
from dataclasses import dataclass
import random
import time
from typing import List, Literal, Optional
import numpy as np
import grpc
from grpc_generated.server_pb2 import PullRequest, PushRequest, WaitRequest, Result, Task, Range, KeyList, Values
from grpc_generated.server_pb2_grpc import ServerServicer, add_ServerServicer_to_server

## For maintaining task state
@dataclass
class ServerTask:
    task: Task
    operation: Literal['push', 'pull']
    request: PushRequest | PullRequest
    status: Literal['failed', 'success', 'unprocessed', 'processing']
    async_task: Optional[asyncio.Task] = None
    failure_message: Optional[str] = ""

## todo the deps logics.
class ServerServicer(ServerServicer):
    def __init__(self, num_keys, value_width, learning_rate):
        ## The weights are stored in a 2D array, rows/i are the num of keys (parameters) and columns/j are the actual weights for each key
        ## we declare an locks array acting as the lock key by key to avoid concurrent overwrite, can be managed better in later modules
        self.num_keys = num_keys
        self.value_width = value_width
        self.learning_rate = learning_rate
        self.weights = np.zeros((num_keys, value_width), dtype=np.float64)
        self.locks = [asyncio.Lock() for _ in range(num_keys)]
        ## maintained task list shared state
        self.tasks = defaultdict(dict)

    ## GRPC Definitions
    async def Pull(self, request: PullRequest, context) -> Task:
        target_type = request.WhichOneof("target")

        if target_type == 'r' and not self._validate_range(request.r):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid range"
            )
        if target_type == 'k' and not self._validate_keys(request.k):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid key list"
            )
        if target_type is None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid Range and/or Keylist"
            )

        taskId = self._generate_task_id()
        self.tasks[request.workerId][taskId] = ServerTask(
            task=Task(taskId=taskId, workerId=request.workerId),
            operation="pull",
            request=request,
            status="unprocessed"
        )
        task = asyncio.create_task(self.task_processor(self.tasks[request.workerId][taskId])) ## this creates an async pull on wait it will return
        self.tasks[request.workerId][taskId].async_task = task
        return self.tasks[request.workerId][taskId].task

    async def Push(self, request: PushRequest, context) -> Task:
        target_type = request.WhichOneof("target")

        if target_type == 'r':
            if not self._validate_range(request.r):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Invalid range"
                )
            if len(request.values) != (request.r.end-request.r.begin)*self.value_width:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Invalid gradient array length"
                )
        if target_type == 'k':
            if not self._validate_keys(request.k):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Invalid key list"
                )
            if len(request.values) != len(request.k.keys)*self.value_width:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Invalid gradient array length"
                )
        if target_type is None:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid Range and/or Keylist"
            )

        taskId = self._generate_task_id()
        self.tasks[request.workerId][taskId] = ServerTask(
            task=Task(taskId=taskId, workerId=request.workerId),
            operation="push",
            request=request,
            status="unprocessed"
        )
        task = asyncio.create_task(self.task_processor(self.tasks[request.workerId][taskId])) ## this creates an async push on wait it will ack
        self.tasks[request.workerId][taskId].async_task = task
        return self.tasks[request.workerId][taskId].task

    async def Wait(self, request: WaitRequest, context) -> Result:
        task: ServerTask = self._locate_task(request.task)
        if not task:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid task"
            )
        if request.timeout_ms:
            try:
                result = await asyncio.wait_for(asyncio.shield(task.async_task), timeout=request.timeout_ms / 1000)
            except asyncio.TimeoutError:
                await context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "Task wait timed out"
                )
        else:
            result = await asyncio.shield(task.async_task)
        if result is None:
            if task.status == 'failed':
                ## there technically isn't an else
                await context.abort(
                    grpc.StatusCode.ABORTED,
                    task.failure_message
                )
        return result
    ## END GRPC Overrides

    ## GRPC Helper functions
    async def task_processor(self, task: ServerTask) -> Result | None:
        try:
            if task.status != "unprocessed":
                if task.status == "success":
                    return task.result
                else:
                    task.status = "failed"
                    task.failure_message = "Task execution failed"
                    return None

            task.status = "processing"
            ## we should actually check the deps
            if task.operation == "pull":
                target_type = task.request.WhichOneof("target")
                if target_type == 'r':
                    result = await self.pull_dense(task.request.r)
                elif target_type == 'k':
                    result = await self.pull_sparse(task.request.k)
                else:
                    task.status = "failed"
                    task.failure_message = "Invalid Range and/or Keylist"
                    return None
                task.status = "success"
                task.result = result
                return result
            elif task.operation == "push":
                target_type = task.request.WhichOneof("target")
                if target_type == 'r':
                    result = await self.push_dense(task.request.r, task.request.values)
                elif target_type == 'k':
                    result = await self.push_sparse(task.request.k, task.request.values)
                else:
                    task.status = "failed"
                    task.failure_message = "Invalid Range and/or Keylist"
                    return None
                task.status = "success"
                task.result = result
                return result
        except:
            task.status = "failed"
            task.failure_message = "Task execution failed"
            return None

    async def pull_dense(self, target: Range) -> Result:
        if not self._validate_range(target):
            raise ValueError("Invalid range")
        keys = KeyList(keys=list(range(target.begin, target.end)))
        vals = Values(keys=keys, data=self.weights[target.begin:target.end, :].ravel(), width=self.value_width)
        return Result(v=vals, ack=True)
        
    async def pull_sparse(self, target: KeyList) -> Result:
        if not self._validate_keys(target):
            raise ValueError("Invalid key list")
        vals = Values(keys=target, data=self.weights[target.keys].ravel(), width=self.value_width)
        return Result(v=vals, ack=True)

    async def push_dense(self, target: Range, gradient: List[float]):
        if not self._validate_range(target):
            raise ValueError("Invalid range")
        if len(gradient) != (target.end-target.begin)*self.value_width:
            raise ValueError("Invalid gradient array length")
        idx = 0
        for i in range(target.begin, target.end):
            v_start = idx*self.value_width
            await self._update_weights(i, gradient[v_start:v_start+self.value_width])
            idx += 1
        print(self.weights)
        return Result(ack=True)
    async def push_sparse(self, target: KeyList, gradient: List[float]):
        if not self._validate_keys(target):
            raise ValueError("Invalid key list")
        if len(gradient) != len(target.keys)*self.value_width:
            raise ValueError("Invalid gradient array length")
        idx = 0
        for i in target.keys:
            v_start = idx*self.value_width
            await self._update_weights(i, gradient[v_start:v_start+self.value_width])
            idx += 1
        print(self.weights)
        return Result(ack=True)
    ## END GRPC Helper functions

    ## Only used within class
    def _generate_task_id(self) -> int:
        return random.randint(100_000_000_000, 999_999_999_999)
    def _validate_range(self, range) -> bool:
        if range.end <= range.begin or range.begin >= self.num_keys or range.end > self.num_keys:
            return False
        return True
    def _validate_keys(self, keys) -> bool:
        for key in keys.keys:
            if key >= self.num_keys:
                return False
        return True
    def _locate_task(self, task: Task) -> ServerTask | None:
        if task.workerId in self.tasks and self.tasks[task.workerId] and task.taskId in self.tasks[task.workerId]:
            return self.tasks[task.workerId][task.taskId]
        return None
    async def _update_weights(self, key, gradient) -> None:
        ## first obtain the lock [[actually locks should have a timeout and whatnot this is not crash proof!]]
        async with self.locks[key]:
            gradient = np.array(gradient)
            self.weights[key] -= self.learning_rate * gradient

async def serve(args) -> None:
    ## had bug where server ran the same port 2x not reflecting updates, so added this option
    server = grpc.aio.server(options=[("grpc.so_reuseport", 0)])
    add_ServerServicer_to_server(
        ServerServicer(args.num_keys, args.value_width, args.learning_rate), server
    )
    server.add_insecure_port(f"[::]:{args.port}")
    await server.start()
    print(f"Server started on {args.port}")
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-keys", required=True, type=int)
    parser.add_argument("--value-width", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("-p", "--port", required=False, type=int, default=50051)
    parser.add_argument("-s", "--seed", required=False, type=int)
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    asyncio.run(serve(args))