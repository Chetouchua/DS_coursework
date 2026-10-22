import argparse
import asyncio
from collections import defaultdict
import random
import numpy as np
import grpc
from grpc_generated.server_pb2 import PullRequest, PushRequest, Task
import grpc_generated.server_pb2_grpc as sp2grpc

class ServerServicer(sp2grpc.ServerServicer):
    def __init__(self, num_keys, value_width):
        ## +1 so the zeroth float can be considered as a lock only taking values of 0 and 1, in the future this can be done better
        self.weights = np.zeros((num_keys, value_width+1), dtype=np.float64)
        ## maintained task list shared state
        self.tasks = defaultdict(list)

    ## private
    def generate_task_id(self):
        ## code to generate random task id, n digits
        n = 6
        return random.randint(10**(n-1), (10**n)-1)

    async def Pull(self, request: PullRequest, context):
        task = (Task(taskId=self.generate_task_id(), workerId=request.workerId), "pull", request)
        self.tasks[request.workerId].append(task)
        return task[0]

    async def Push(self, request: PushRequest, context):
        task = (Task(taskId=self.generate_task_id(), workerId=request.workerId), "push", request)
        self.tasks[request.workerId].append(task)
        return task[0]

async def serve(args) -> None:
    server = grpc.aio.server()
    sp2grpc.add_ServerServicer_to_server(
        ServerServicer(args.num_keys, args.value_width), server
    )
    server.add_insecure_port(f"[::]:{args.port}")
    await server.start()
    print(f"Server started on {args.port}")
    await server.wait_for_termination()

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