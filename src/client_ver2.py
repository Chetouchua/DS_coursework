#a worker loops meet up the requirements in A2
import asyncio
from client_library.interface_ver2 import WorkerInterface

async def runtraining(data, compute_grad):
    async with WorkerInterface(worker_id=1) as ps:
        for batch in data:
            pulled = await ps.pull(batch.keys)
            w = await pulled.result()
            g = compute_grad(w, batch)
            await ps.push(batch.keys, g, deps=[pulled])
        await ps.wait_all()