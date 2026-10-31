import argparse
import asyncio
from typing import Optional
import grpc
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.generated.ps_pb2 import (
    KeyList,
    PullRequest,
    PullResponse,
    PushRequest,
    PushResponse,
    Range,
    StatusCode,
    Values,
    WaitRequest,
    WaitResponse,
)
from src.generated.ps_pb2_grpc import ServerServicer, add_ServerServicer_to_server
from src.server.dep_tracker import DependencyTracker, ServerTask
from src.server.store import ParameterStore

MAX_MESSAGE_LENGTH = 128 * 1024 * 1024  # 128 MB

server_options = [
    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
    ("grpc.so_reuseport", 0)
]


class ParameterServerServicer(ServerServicer):
    def __init__(self, num_keys: int, value_width: int, learning_rate: float):
        self.num_keys = num_keys
        self.value_width = value_width
        self.store = ParameterStore(num_keys, value_width, learning_rate)
        self.tracker = DependencyTracker()
        self.key_cache = {}

    # --- gRPC Method Overrides ---

    async def Push(self, request: PushRequest, context) -> PushResponse:
        w_id = request.task_id.worker_id
        t_id = request.task_id.task_id

        logging.info(f"Received PUSH from worker {request.task_id.worker_id} for task {request.task_id.task_id}")

        # Idempotence: Return handle if update already scheduled/applied
        existing = self.tracker.get_task(w_id, t_id)
        if existing:
            return PushResponse(task_id=request.task_id)

        target_type = request.WhichOneof("target")
        await self._validate_request(target_type, request, context, is_push=True)

        task_record = ServerTask(
            task_id=request.task_id,
            operation="push",
            status="unprocessed"
        )
        self.tracker.register_task(task_record)
        task_record.async_task = asyncio.create_task(self._process_push(task_record, request))

        return PushResponse(task_id=request.task_id)

    async def Pull(self, request: PullRequest, context) -> PullResponse:
        w_id = request.task_id.worker_id
        t_id = request.task_id.task_id

        logging.info(f"Received PULL from worker {request.task_id.worker_id} for task {request.task_id.task_id}")

        existing = self.tracker.get_task(w_id, t_id)
        if existing:
            return PullResponse(task_id=request.task_id)

        target_type = request.WhichOneof("target")
        await self._validate_request(target_type, request, context, is_push=False)

        task_record = ServerTask(
            task_id=request.task_id,
            operation="pull",
            status="unprocessed"
        )
        self.tracker.register_task(task_record)
        task_record.async_task = asyncio.create_task(self._process_pull(task_record, request))

        return PullResponse(task_id=request.task_id)

    async def Wait(self, request: WaitRequest, context) -> WaitResponse:
        server_task = self.tracker.get_task(request.task_id.worker_id, request.task_id.task_id)
        if not server_task or not server_task.async_task:
            return WaitResponse(
                status=StatusCode.STATUS_CODE_INVALID_ARGUMENT,
                error_message="Task not found."
            )

        # Handle non-blocking timeout
        if request.timeout_ms > 0:
            try:
                await asyncio.wait_for(
                    asyncio.shield(server_task.async_task),
                    timeout=request.timeout_ms / 1000.0
                )
            except asyncio.TimeoutError:
                # Timed out task is still running; calling wait again is legal
                return WaitResponse(status=StatusCode.STATUS_CODE_TIMEOUT, error_message="Task still running.")
        else:
            await asyncio.shield(server_task.async_task)

        if server_task.status == "failed":
            code = (
                StatusCode.STATUS_CODE_DEPENDENCY_FAILED
                if "DEPENDENCY_FAILED" in server_task.failure_message
                else StatusCode.STATUS_CODE_INTERNAL_ERROR
            )
            return WaitResponse(status=code, error_message=server_task.failure_message)

        return WaitResponse(
            status=StatusCode.STATUS_CODE_OK,
            values=server_task.result_values,
            ack=True
        )

    # --- Background Processing with Dependency Checking ---

    async def _process_push(self, task: ServerTask, request: PushRequest):
        w_id = request.task_id.worker_id
        t_id = request.task_id.task_id
        start_time = time.perf_counter()
        # logging.info(f"[_process_push] ENTERED: Worker {request.task_id.worker_id} Task {request.task_id.task_id}")
        
        try:
            task.status = "processing"
            # 1. Enforce Dependencies (Part A3)
            if request.opts and request.opts.deps:
                ok, err = await self.tracker.wait_for_deps(list(request.opts.deps))
                if not ok:
                    self.tracker.mark_failed(w_id, t_id, err)
                    logging.warning(
                        f"[_process_push] Worker {w_id} Task {t_id} FAILED dependencies: {err}"
                    )
                    return

                # 2. Execute Aggregation
                target_type = request.WhichOneof("target")
                grad_values = list(request.values) if hasattr(request, "values") else []
                grad_count = len(grad_values)

                if target_type == "r":
                    begin , end = request.r.begin , request.r.end
                    key_summary = f"dense range [{begin}:{end}] ({end - begin} keys)"
                    await self.store.push_dense(request.r.begin, request.r.end, grad_values)
                elif target_type == "k":
                    raw_keys = list(request.k.keys)
                    sample = raw_keys[:5]
                    key_summary = (
                    f"{len(raw_keys)} sparse keys (sample: {sample}{'...' if len(raw_keys) > 5 else ''})"
                    )
                    await self.store.push_sparse(list(request.k.keys), grad_values)
                elif target_type == "cached_key_hash":
                    keys = self.key_cache[request.cached_key_hash].keys
                    raw_keys = list(keys)
                    sample = raw_keys[:5]
                    key_summary = (
                        f"cached hash {request.cached_key_hash} "
                        f"({len(raw_keys)} keys, sample: {sample}{'...' if len(raw_keys) > 5 else ''})"
                    )
                    await self.store.push_sparse(raw_keys, grad_values)

            self.tracker.mark_success(w_id, t_id)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logging.info(
                f"[_process_push] SUCCESS: Worker {w_id} pushed gradients to {key_summary} "
                f"(task_id={t_id}, gradients={grad_count}, elapsed={elapsed_ms:.2f}ms)"
            )
        except Exception as e:
            self.tracker.mark_failed(w_id,t_id,str(e))
            logging.error(f"[_process_push] ERROR: Worker {w_id} Task {t_id}: {e}", exc_info=True)

    async def _process_pull(self, task: ServerTask, request: PullRequest):
        w_id = request.task_id.worker_id
        t_id = request.task_id.task_id
        start_time = time.perf_counter()
        
        try:
            task.status = "processing"
            # 1. Enforce Dependencies
            if request.opts and request.opts.deps:
                ok, err = await self.tracker.wait_for_deps(list(request.opts.deps))
                if not ok:
                    self.tracker.mark_failed(w_id, t_id, err)
                    logging.warning(
                    f"[_process_pull] Worker {w_id} Task {t_id} FAILED dependencies: {err}"
                    )
                    return

            # 2. Extract Data
            target_type = request.WhichOneof("target")
            if target_type == "r":
                begin , end = request.r.begin , request.r.end
                key_summary = f"dense range [{begin}:{end}] ({end - begin} keys)"
                flat_data = await self.store.pull_dense(request.r.begin, request.r.end)
                keys = KeyList(keys=list(range(request.r.begin, request.r.end)))
            elif target_type == "k":
                raw_keys = list(request.k.keys)
                sample = raw_keys[:5]
                key_summary = (
                f"{len(raw_keys)} sparse keys (sample: {sample}{'...' if len(raw_keys) > 5 else ''})"
                )
                flat_data = await self.store.pull_sparse(list(request.k.keys))
                keys = request.k
            elif target_type == "cached_key_hash":
                keys = self.key_cache[request.cached_key_hash]
                raw_keys = list(keys.keys)
                sample = raw_keys[:5]
                key_summary = (
                    f"cached hash {request.cached_key_hash} "
                    f"({len(raw_keys)} keys, sample: {sample}{'...' if len(raw_keys) > 5 else ''})"
                )
                flat_data = await self.store.pull_sparse(list(keys.keys))

            vals = Values(keys=keys, data=flat_data, width=self.value_width)
            self.tracker.mark_success(w_id, t_id, values=vals)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logging.info(
                f"[_process_pull] SUCCESS: Worker {w_id} pulled {key_summary} "
                f"(task_id={t_id}, values={len(flat_data)}, elapsed={elapsed_ms:.2f}ms)"
            )
        except Exception as e:
            self.tracker.mark_failed(w_id, t_id, str(e))
            logging.error(
            f"[_process_pull] ERROR: Worker {w_id} Task {t_id} encountered exception: {e}",
            exc_info=True,
            )

    # --- Validation ---

    async def _validate_request(self, target_type: Optional[str], request, context, is_push: bool):
        if target_type == "r":
            if request.r.end <= request.r.begin or request.r.end > self.num_keys:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid range")
            if is_push and len(request.values) != (request.r.end - request.r.begin) * self.value_width:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid gradient length")
        elif target_type == "k":
            if any(k >= self.num_keys for k in request.k.keys):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Key index out of bounds")
            if is_push and len(request.values) != len(request.k.keys) * self.value_width:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid gradient length")
            if request.key_hash:
                self.key_cache[request.key_hash] = request.k
        elif target_type == "cached_key_hash":
            if request.cached_key_hash not in self.key_cache:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unknown cached key hash")
        else:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing target range or keys")


async def serve(port: int, num_keys: int, value_width: int, learning_rate: float):
    server = grpc.aio.server(options=server_options)
    servicer = ParameterServerServicer(num_keys, value_width, learning_rate)
    add_ServerServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"Parameter Server online on port {port}")
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-keys", required=True, type=int)
    parser.add_argument("--value-width", required=True, type=int)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("-p", "--port", type=int, default=50051)
    args = parser.parse_args()

    asyncio.run(serve(args.port, args.num_keys, args.value_width, args.learning_rate))