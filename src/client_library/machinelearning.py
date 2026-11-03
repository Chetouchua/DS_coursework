import random
from typing import List
import numpy as np
from grpc_generated.server_pb2 import Range

## ML work in separate class
class MachineLearning:
    def __init__(self, file_path, gradient_type, noise, keys_request_type, num_keys, keys_per_iter, seed):
        self.target = np.load(file_path, mmap_mode="r")
        self.is_target_diff = gradient_type == 'target_diff'
        self.noise = noise
        self.seed = seed
        self.keys_request_type = keys_request_type
        self.num_keys = num_keys
        self.keys_per_iter = keys_per_iter
        if self.seed:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng(None)

    def compute_grad(self, arr, weights, width):
        ## Gradient is simply W - W* + Sampled Noise
        noise_arr = self.rng.normal(0, self.noise, size=len(weights))
        idx = 0
        for i in arr:
            for j in range(width):
                if self.is_target_diff:
                    weights[idx] = weights[idx] - self.target[i][j] + noise_arr[idx]
                else:
                    weights[idx] = 0.01 ## just direct deterministic number should properly be f(k , t, i)
                idx += 1
        return weights

    def get_keys(self) -> List | Range:
        if self.keys_request_type == "dense":
            return self.get_random_dense(self.num_keys, self.keys_per_iter)
        if self.keys_request_type == "sparse":
            return self.get_random_sparse(self.num_keys, self.keys_per_iter)
        raise ValueError() ## wont go here

    def get_random_sparse(self, num_keys, keys_per_iter):
        return sorted(random.sample(list(range(num_keys)), keys_per_iter)) ## uses random sampling but we need zipf and other things refer document

    def get_random_dense(self, num_keys, keys_per_iter):
        begin = random.randrange(num_keys - keys_per_iter + 1)
        end = begin + keys_per_iter
        keys = Range(begin=begin, end=end)
        return keys