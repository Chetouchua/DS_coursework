import asyncio
import argparse
import random
from client_library.interface import WorkerInterface
import numpy as np

class MachineLearning:
    def __init__(self, file_path, gradient_type, noise, seed):
        self.target = np.load(file_path, mmap_mode="r")
        self.is_target_diff = gradient_type == 'target_diff'
        self.noise = noise
        self.seed = seed

    def compute_gradients(self, arr, weights, width):
        ## Gradient is simply W - W* + Sampled Noise
        if self.seed:
            rng = np.random.default_rng(self.seed)
        else:
            rng = np.random.default_rng(None)
        noise_arr = rng.normal(0, self.noise, size=len(weights))
        idx = 0
        for i in arr:
            for j in range(width):
                if self.is_target_diff:
                    weights[idx] = weights[idx] - self.target[i][j] + noise_arr[idx]
                else:
                    weights[idx] = 0.01 ## just direct deterministic number should ideally be f(k , t, i)
                idx += 1
        return weights

async def main(args):
    mode = args.mode
    w = WorkerInterface(args.seed)
    ml = MachineLearning(args.target_weight_file, args.gradient_type, args.gradient_noise, args.seed)    

    for i in range(args.iterations):
        if mode == 'sparse':
            sparse_choices = random.sample(list(range(args.num_keys)), args.keys_per_iter)
            task = await w.apull_sparse(sparse_choices)
            weight_result = await w.wait(task)
        elif mode == 'dense':
            begin = random.choice(list(range(args.num_keys-args.keys_per_iter)))
            end = begin + args.keys_per_iter
            task = await w.apull_dense(begin, end)
        weight_result = await w.wait(task)
        gradients = ml.compute_gradients(weight_result.v.keys.keys, weight_result.v.data, weight_result.v.width)
        if mode == 'sparse':
            task = await w.apush_sparse(sparse_choices, gradients)
        elif mode == 'dense':
            task = await w.apush_dense(begin, end, gradients)
        await w.wait(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    asyncio.run(main(args))