"""
End-to-end integration test for the Parameter Server:
Server Lifecycle -> Target Oracle -> Background Observer -> Worker Threads -> Convergence Check
"""
import os
import subprocess
import sys
import threading
import time
import numpy as np
import pytest

from src.client import Client, Range
from src.harness.observer import ModelObserver


def test_full_parameter_server_workflow():
    port = 50055
    server_addr = f"localhost:{port}"
    num_keys = 100
    value_width = 1
    learning_rate = 0.1
    num_workers = 4
    iterations_per_worker = 25
    output_dir = "results/test_workflow"

    os.makedirs("data", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate known ground-truth target weights w*
    rng = np.random.default_rng(42)
    w_star = rng.normal(loc=5.0, scale=1.0, size=(num_keys, value_width)).astype(np.float64)
    target_path = "data/test_workflow_target.npy"
    np.save(target_path, w_star)

    # 2. Start the Parameter Server as a subprocess
    server_cmd = [
        sys.executable, "-m", "src.server.service",
        "--num-keys", str(num_keys),
        "--value-width", str(value_width),
        "--learning-rate", str(learning_rate),
        "--port", str(port),
    ]
    server_proc = subprocess.Popen(server_cmd)
    time.sleep(1.0)  # Wait for socket initialization

    observer = None
    try:
        # 3. Start the Observer logging to CSV
        observer = ModelObserver(
            server_addr=server_addr,
            num_keys=num_keys,
            target_weights_path=target_path,
            output_dir=output_dir,
            interval_sec=0.25,
            worker_id=999,
        )
        observer.start()

        # 4. Worker function executing training loop: gradient = (w - w*)
        def worker_loop(worker_id: int):
            with Client(worker_id=worker_id, server_addr=server_addr) as client:
                for t in range(1, iterations_per_worker + 1):
                    # Pull keys
                    pull_task = client.pull(Range(0, num_keys), iteration=t)
                    current_w = pull_task.wait()

                    # Ensure both current_w and w_star are 1D arrays of size 100
                    diff = current_w.reshape(-1) - w_star.reshape(-1)
                    grads = diff.tolist()  # 100 elements

                    # Push gradients with dependency on the pull
                    push_task = client.push(
                        Range(0, num_keys),
                        grads,
                        iteration=t,
                        deps=[pull_task]
                    )
                    push_task.wait()
                    time.sleep(0.01)

                client.wait_all()

        # 5. Launch concurrent worker threads
        threads = [
            threading.Thread(target=worker_loop, args=(wid,))
            for wid in range(1, num_workers + 1)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        # Allow final update to settle
        time.sleep(0.5)

        # 6. Verify convergence
        with Client(worker_id=1000, server_addr=server_addr) as client:
            final_w = client.pull(Range(0, num_keys)).wait()

        final_w = final_w.reshape(w_star.shape)
        initial_l2 = float(np.linalg.norm(np.zeros_like(w_star) - w_star))
        final_l2 = float(np.linalg.norm(final_w - w_star))

        print(f"\n" + "=" * 60)
        print(f"Initial L2 Distance: {initial_l2:.4f}")
        print(f"Final L2 Distance:   {final_l2:.4f}")
        print(f"Sample w* (keys 0-4):  {w_star[:5].ravel()}")
        print(f"Sample w  (keys 0-4):  {final_w[:5].ravel()}")
        print(f"CSV Logs saved to:   {output_dir}/observer_metrics.csv")
        print("=" * 60 + "\n")

        # The weights must converge closer to w* than the zero initialization
        assert final_l2 < initial_l2, f"Model failed to converge: {final_l2} >= {initial_l2}"

    finally:
        # 7. Clean teardown
        if observer is not None:
            observer.stop()

        if server_proc.poll() is None:
            server_proc.kill()
            server_proc.wait()


if __name__ == "__main__":
    test_full_parameter_server_workflow()