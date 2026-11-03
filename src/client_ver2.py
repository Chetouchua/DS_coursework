# a worker loops meet up the requirements in A2
# adjusted to work with loop
import argparse
import asyncio
import random
from client_library.interface_ver2 import WorkerInterface
from client_library.machinelearning import MachineLearning

async def runtraining(ml: MachineLearning):
    async with WorkerInterface(worker_id=args.worker_id) as ps:
        for _ in range(args.iterations):
            keys = ml.get_keys()
            pulled = await ps.pull(keys)
            w = await pulled.result()
            g = ml.compute_grad(w.keys.keys, w.data, w.width)
            pushed = await ps.push(keys, g, deps=[pulled])
            await pushed.wait()
        await ps.wait_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True, type=int)
    parser.add_argument("--num-keys", required=True, type=int)
    parser.add_argument("--value-width", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--keys-per-iter", required=True, type=int)
    parser.add_argument("--target-weight-file", required=True, type=str)
    parser.add_argument("--gradient-type", required=True, type=str)
    parser.add_argument("--gradient-noise", required=False, type=float, default=0.0)
    parser.add_argument("--mode", required=True, type=str, choices=['sparse','dense'])
    parser.add_argument("-s", "--seed", required=False, type=int)
    args = parser.parse_args()
    if args.seed:
        random.seed(args.seed)
    ml = MachineLearning(args.target_weight_file, args.gradient_type, args.gradient_noise, args.mode, args.num_keys, args.keys_per_iter, args.seed)
    asyncio.run(runtraining(ml))
