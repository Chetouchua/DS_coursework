import asyncio
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
from src.generated.ps_pb2 import StatusCode, TaskId, Values


@dataclass
class ServerTask:
    task_id: TaskId
    operation: Literal["push", "pull"]
    status: Literal["unprocessed", "processing", "success", "failed"]
    async_task: Optional[asyncio.Task] = None
    failure_message: str = ""
    result_values: Optional[Values] = None


class DependencyTracker:
    def __init__(self):
        # Maps (worker_id, task_id) -> ServerTask
        self.tasks: Dict[Tuple[int, int], ServerTask] = {}
        self._cond = asyncio.Condition()

    def get_task(self, worker_id: int, task_id: int) -> Optional[ServerTask]:
        return self.tasks.get((worker_id, task_id))

    def register_task(self, server_task: ServerTask) -> None:
        key = (server_task.task_id.worker_id, server_task.task_id.task_id)
        self.tasks[key] = server_task

    async def wait_for_deps(self, deps: List[TaskId]) -> Tuple[bool, str]:
        """Part A3: Blocks until all antecedent tasks complete.
        Returns (success: bool, error_message: str)
        """
        if not deps:
            return True, ""

        for dep in deps:
            dep_key = (dep.worker_id, dep.task_id)
            dep_task = self.tasks.get(dep_key)

            if not dep_task or not dep_task.async_task:
                return False, f"Dependency ({dep.worker_id}, {dep.task_id}) does not exist."

            # Await completion of the antecedent asyncio task
            await dep_task.async_task

            if dep_task.status == "failed":
                # Section 5.2 error requirement
                return False, "DEPENDENCY_FAILED"

        return True, ""

    async def mark_done(self, worker_id: int, task_id: int, success: bool, msg: str = "") -> None:
        async with self._cond:
            key = (worker_id, task_id)
            if key in self.tasks:
                self.tasks[key].status = "success" if success else "failed"
                self.tasks[key].failure_message = msg
            self._cond.notify_all()