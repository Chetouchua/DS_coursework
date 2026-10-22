import asyncio
import grpc
import grpc_generated.server_pb2 as spb2
import grpc_generated.server_pb2_grpc as spb2grpc

async def main():
    async with grpc.aio.insecure_channel(
        "localhost:50051"
    ) as channel:

        # Generated client stub
        stub = spb2grpc.ServerStub(channel)

        # Construct the request
        request = spb2.PullRequest()

        # Invoke the remote function
        response = await stub.Pull(request)

        print(response)


if __name__ == "__main__":
    asyncio.run(main())