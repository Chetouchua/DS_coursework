"""
Worker process entrypoint invoked by src/harness/driver.py.
"""
import argparse
import numpy as np
from src.client import Client, Range


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter Server Worker")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-keys", type=int, required=True)
    parser.add_argument("--value-width", type=int, default=1)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--mode", type=str, default="dense")
    parser.add_argument("--keys-per-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-file", type=str, required=True)
    parser.add_argument("--gradient-type", type=str, default="target_diff")
    parser.add_argument("--gradient-noise", type=float, default=0.0)
    parser.add_argument("--server-addr", type=str, default="localhost:50051")
    return parser.parse_args()


def run_worker():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    w_star = np.load(args.target_file).reshape(args.num_keys, args.value_width)

    with Client(worker_id=args.worker_id, server_addr=args.server_addr) as client:
        for t in range(1, args.iterations + 1):
            if args.mode == "dense":
                target_range = Range(0, args.num_keys)
                pull_task = client.pull(target_range, iteration=t)
                current_w = pull_task.wait()

                diff = current_w.reshape(-1) - w_star.reshape(-1)
                if args.gradient_noise > 0.0:
                    diff += rng.normal(0.0, args.gradient_noise, size=diff.shape)

                grads = diff.tolist()
                push_task = client.push(
                    target_range,
                    grads,
                    iteration=t,
                    deps=[pull_task],
                )
                push_task.wait()

            elif args.mode == "sparse":
                keys = np.sort(
                    rng.choice(args.num_keys, size=min(args.keys_per_iter, args.num_keys), replace=False)
                )
                pull_task = client.pull(keys, iteration=t)
                current_w = pull_task.wait()

                selected_target = w_star[keys].reshape(-1)
                diff = current_w.reshape(-1) - selected_target
                if args.gradient_noise > 0.0:
                    diff += rng.normal(0.0, args.gradient_noise, size=diff.shape)

                grads = diff.tolist()
                push_task = client.push(
                    keys,
                    grads,
                    iteration=t,
                    deps=[pull_task],
                )
                push_task.wait()

        client.wait_all()


if __name__ == "__main__":
    run_worker()