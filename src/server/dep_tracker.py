import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
from src.generated.ps_pb2 import TaskId, Values


@dataclass
class ServerTask:
    task_id: TaskId
    operation: Literal["push", "pull"]
    status: Literal["unprocessed", "processing", "success", "failed"] = "unprocessed"
    async_task: Optional[asyncio.Task] = None
    failure_message: str = ""
    result_values: Optional[Values] = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


class DependencyTracker:
    def __init__(self):
        # Maps (worker_id, task_id) -> ServerTask
        self.tasks: Dict[Tuple[int, int], ServerTask] = {}
        self._lock = asyncio.Lock()

    def get_task(self, worker_id: int, task_id: int) -> Optional[ServerTask]:
        return self.tasks.get((worker_id, task_id))

    def register_task(self, server_task: ServerTask) -> None:
        key = (server_task.task_id.worker_id, server_task.task_id.task_id)
        self.tasks[key] = server_task

    def mark_success(self, worker_id: int, task_id: int, values: Optional[Values] = None) -> None:
        key = (worker_id, task_id)
        task = self.tasks.get(key)
        if task:
            task.status = "success"
            if values is not None:
                task.result_values = values
            task.done_event.set()

    def mark_failed(self, worker_id: int, task_id: int, err: str) -> None:
        key = (worker_id, task_id)
        task = self.tasks.get(key)
        if task:
            task.status = "failed"
            task.failure_message = err
            task.done_event.set()

    async def wait_for_deps(self, deps: List[TaskId], timeout_sec: float = 10.0) -> Tuple[bool, str]:
        """Part A3: Blocks until all antecedent tasks complete.
        Returns (success: bool, error_message: str)
        """
        if not deps:
            return True, ""

        for dep in deps:
            dep_key = (dep.worker_id, dep.task_id)
            dep_task = self.tasks.get(dep_key)

            # Wait briefly if registration is slightly delayed by network interleaving
            retries = 50
            while dep_task is None and retries > 0:
                await asyncio.sleep(0.01)
                dep_task = self.tasks.get(dep_key)
                retries -= 1

            if not dep_task:
                return False, f"Dependency worker={dep.worker_id} task={dep.task_id} not found."

            # Wait for task completion via Event
            try:
                await asyncio.wait_for(dep_task.done_event.wait(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                return False, f"Dependency worker={dep.worker_id} task={dep.task_id} timed out."

            if dep_task.status == "failed":
                return False, f"DEPENDENCY_FAILED: {dep_task.failure_message}"

        return True, ""