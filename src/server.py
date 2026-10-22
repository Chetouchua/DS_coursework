import argparse
import asyncio
from collections import defaultdict
import random
import numpy as np
import grpc
from grpc_generated.server_pb2 import PullRequest, PushRequest, WaitRequest, Result, Task, Range, KeyList, Values
from grpc_generated.server_pb2_grpc import ServerServicer, add_ServerServicer_to_server


## todo the deps logics.
class ServerServicer(ServerServicer):
    def __init__(self, num_keys, value_width):
        ## we declare an integer array acting as the lock key by key to avoid concurrent overwrite, can be managed better in later modules
        self.num_keys = num_keys
        self.value_width = value_width
        self.weights = np.zeros((num_keys, value_width), dtype=np.float64)
        self.locks = [0] * num_keys
        ## maintained task list shared state
        self.tasks = defaultdict(list)

    ## GRPC definitions
    async def Pull(self, request: PullRequest, context) -> Task:
        target_type = request.WhichOneof("target")

        if target_type == 'r' and not self._validate_range(request.r):
            print("call error", request.r, type(request.r))
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid range"
            )
        if target_type == 'k' and not self._validate_keys(request.k):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid key list"
            )
        task = (Task(taskId=self._generate_task_id(), workerId=request.workerId), "pull", request)
        self.tasks[request.workerId].append(task)
        return task[0]

    async def Push(self, request: PushRequest, context) -> Task:
        target_type = request.WhichOneof("target")

        if target_type == 'r' and not self._validate_range(request.r):
            print("call error", request.r, type(request.r))
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid range"
            )
        if target_type == 'k' and not self._validate_keys(request.k):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid key list"
            )
        task = (Task(taskId=self._generate_task_id(), workerId=request.workerId), "push", request)
        self.tasks[request.workerId].append(task)
        return task[0]

    async def Wait(self, request: WaitRequest, context) -> Result:
        task = self._locate_task(request.task)
        if not task:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid task"
            )
        ## we should actually check the deps
        if task[1] == "pull":
            target_type = task[2].WhichOneof("target")
            if target_type == 'r':
                return await self.pull_dense(task[2].r, context)
            elif target_type == 'k':
                return await self.pull_sparse(task[2].k, context)
        elif task[1] == "push":
            pass
        ## should never go here actually
        await context.abort(
            grpc.StatusCode.UNKNOWN,
            "Server routing error"
        )
    ## END GRPC Overrides

    ## GRPC Helper functions
    async def pull_dense(self, target: Range, context) -> Result:
        if not self._validate_range(target):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid range"
            )
        keys = KeyList(keys=list(range(target.begin, target.end)))
        vals = Values(keys=keys, data=self.weights[target.begin:target.end, :].ravel(), width=self.value_width)
        return Result(v=vals, ack=True)
        
    async def pull_sparse(self, target: KeyList, context) -> Result:
        ## validate keys first
        if not self._validate_keys(target):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid key list"
            )
        vals = Values(keys=target, data=self.weights[target.keys].ravel(), width=self.value_width)
        return Result(v=vals, ack=True)

    def push_range(self, target: Range):
        pass
    def push_keys(self, target: KeyList):
        pass
    ## END GRPC Helper functions

    ## Only used within class
    def _generate_task_id(self) -> int:
        return random.randint(1_000_000, 9_999_999)
    def _validate_range(self, range) -> bool:
        if range.end <= range.begin or range.begin >= self.num_keys or range.end > self.num_keys:
            return False
        return True
    def _validate_keys(self, keys) -> bool:
        for key in keys.keys:
            if key >= self.num_keys:
                return False
        return True
    def _locate_task(self, task: Task) -> tuple | None:
        for item in self.tasks[task.workerId]:
            if item[0].taskId == task.taskId:
                return item
        return None

async def serve(args) -> None:
    ## had bug where server ran the same port 2x not reflecting updates, so added this option
    server = grpc.aio.server(options=[("grpc.so_reuseport", 0)])
    add_ServerServicer_to_server(
        ServerServicer(args.num_keys, args.value_width), server
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
    parser.add_argument("-p", "--port", required=False, type=int, default=50051)
    parser.add_argument("-s", "--seed", required=False, type=int)
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    asyncio.run(serve(args))