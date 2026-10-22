import asyncio
import argparse
import random
from client_library.interface import WorkerInterface

async def main(args):
    ## just a single run to do POC of this working
    w = WorkerInterface(args.seed)

    if args.mode == 'sparse':
        sparse_choices = random.sample(list(range(args.num_keys)), args.keys_per_iter)
        task = await w.apull_sparse(sparse_choices)
    elif args.mode == 'dense':
        begin = random.choice(list(range(args.num_keys-args.keys_per_iter)))
        end = begin + args.keys_per_iter
        task = await w.apull_dense(begin, end)
    weights = await w.wait(task)
    print(weights.ack, "Got weights", len(weights.v.data))
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-keys", required=True, type=int)
    parser.add_argument("--value-width", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--keys-per-iter", required=True, type=int)
    parser.add_argument("--mode", required=True, type=str, choices=['sparse','dense'])
    parser.add_argument("-s", "--seed", required=False, type=int)
    args = parser.parse_args()
    if args.seed:
        random.seed(args.seed)
    asyncio.run(main(args))