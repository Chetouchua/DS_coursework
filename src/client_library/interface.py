import random
from typing import List
import grpc
from grpc_generated.server_pb2 import PullRequest, PushRequest, WaitRequest, Task, Result, KeyList, Range
from grpc_generated.server_pb2_grpc import ServerStub

class WorkerInterface:
    def __init__(self, seed):
        random.seed(seed)
        self.worker_id = random.randint(100_000_000_000, 999_999_999_999)
        ## currently only one server very straightforward!
        self.server = "localhost:50051"
        print(f"Initiated worker ID: {self.worker_id}")

    async def apull_dense(self, begin: int, end: int) -> Task:
        async with grpc.aio.insecure_channel(self.server) as channel:
            stub = ServerStub(channel)
            request = PullRequest(
                workerId=self.worker_id,
                r=Range(
                    begin=begin,
                    end=end
                )
            )
            response = await stub.Pull(request)
            return response

    async def apull_sparse(self, key_arr: List[int]) -> Task:
        async with grpc.aio.insecure_channel(self.server) as channel:
            stub = ServerStub(channel)
            request = PullRequest(
                workerId=self.worker_id,
                k=KeyList(keys=key_arr)
            )
            response = await stub.Pull(request)
            return response

    async def apush_dense(self, begin: int, end: int, gradient: List[float]) -> Task:
        async with grpc.aio.insecure_channel(self.server) as channel:
            stub = ServerStub(channel)
            request = PushRequest(
                workerId=self.worker_id,
                r=Range(
                    begin=begin,
                    end=end
                ),
                values=gradient
            )
            response = await stub.Push(request)
            return response

    async def apush_sparse(self, key_arr: List[int], gradient: List[float]) -> Task:
        async with grpc.aio.insecure_channel(self.server) as channel:
            stub = ServerStub(channel)
            request = PushRequest(
                workerId=self.worker_id,
                k=KeyList(keys=key_arr),
                values=gradient
            )
            response = await stub.Push(request)
            return response

    async def wait(self, task: Task) -> Result:
        async with grpc.aio.insecure_channel(self.server) as channel:
            stub = ServerStub(channel)
            request = WaitRequest(
                task=task,
                timeout_ms=8000 ## arbitrary
            )
            response = await stub.Wait(request)
            return response