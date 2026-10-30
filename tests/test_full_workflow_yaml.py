"""
Integration test driving the full workflow directly from default_part_a.yaml,
with evaluation summary logging (initial vs final L2, weights comparison, and JSON/txt output).
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
import pytest

from src.client import Client, Range
from src.harness.config import load_config
from src.harness.observer import ModelObserver
from src.harness.oracle import CorrectnessOracle


def get_config_path(filename: str = "default_part_a.yaml") -> str:
    candidates = [
        Path(filename),
        Path("configs") / filename,
        Path(__file__).resolve().parent.parent / "configs" / filename,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"Could not locate {filename} across standard paths")


def test_workflow_from_yaml():
    cfg_path = get_config_path("default_part_a.yaml")
    cfg = load_config(cfg_path)
    assert cfg is not None, f"Failed to load config at {cfg_path}"

    port = 50055
    server_addr = f"localhost:{port}"
    num_keys = cfg.model.num_keys
    value_width = cfg.model.value_width
    learning_rate = cfg.model.learning_rate
    num_workers = cfg.workers.count
    iterations = cfg.workers.iterations
    output_dir = cfg.output.dir if cfg.output and cfg.output.dir else f"results/{cfg.experiment}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate target ground-truth w*
    oracle = CorrectnessOracle(cfg.model, seed=cfg.seed)
    target_path = oracle.save(cfg.experiment)
    w_star = oracle.w_star

    # 2. Launch Parameter Server
    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    server_cmd = [
        sys.executable, "-m", "src.server.service",
        "--num-keys", str(num_keys),
        "--value-width", str(value_width),
        "--learning-rate", str(learning_rate),
        "--port", str(port),
    ]
    server_proc = subprocess.Popen(server_cmd, env=env)
    time.sleep(1.0)  # Wait for socket initialization

    observer = None
    try:
        # 3. Start Observer thread
        observer_interval = getattr(cfg.output, "observer_every", 1) or 1
        observer = ModelObserver(
            server_addr=server_addr,
            num_keys=num_keys,
            target_weights_path=target_path,
            output_dir=output_dir,
            interval_sec=float(observer_interval),
            worker_id=999,
        )
        observer.start()

        # 4. Worker thread loop
        def worker_loop(worker_id: int):
            with Client(worker_id=worker_id, server_addr=server_addr) as client:
                for t in range(1, iterations + 1):
                    pull_task = client.pull(Range(0, num_keys), iteration=t)
                    current_w = pull_task.wait()

                    # Gradient = (w - w*)
                    diff = current_w.reshape(-1) - w_star.reshape(-1)
                    grads = diff.tolist()

                    push_task = client.push(
                        Range(0, num_keys),
                        grads,
                        iteration=t,
                        deps=[pull_task]
                    )
                    push_task.wait()

                client.wait_all()

        threads = [
            threading.Thread(target=worker_loop, args=(wid,))
            for wid in range(1, num_workers + 1)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        time.sleep(0.5)

        # 5. Pull final weights and record evaluation logs
        with Client(worker_id=1000, server_addr=server_addr) as client:
            final_w_raw = client.pull(Range(0, num_keys)).wait(timeout_ms=5000)

        final_w = np.asarray(final_w_raw, dtype=np.float64).reshape(w_star.shape)
        initial_w = np.zeros_like(w_star)

        initial_l2 = float(np.linalg.norm(initial_w - w_star))
        final_l2 = float(np.linalg.norm(final_w - w_star))
        l2_diff = initial_l2 - final_l2
        pct_improvement = (l2_diff / initial_l2) * 100.0 if initial_l2 > 0 else 0.0

        # Save numpy array
        np.save(os.path.join(output_dir, "final_weights.npy"), final_w)

        # Write summary.txt
        summary_txt_path = os.path.join(output_dir, "summary.txt")
        with open(summary_txt_path, "w") as f:
            f.write(f"Experiment: {cfg.experiment}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Initial L2 Distance (w_0 vs w*): {initial_l2:.6f}\n")
            f.write(f"Final L2 Distance (w_final vs w*): {final_l2:.6f}\n")
            f.write(f"L2 Improvement: {l2_diff:.6f} ({pct_improvement:.2f}% reduction)\n")
            f.write("-" * 60 + "\n")
            f.write("Weight Samples (First 5 keys):\n")
            for k in range(min(5, num_keys)):
                f.write(f"  Key {k:03d} -> Target (w*): {w_star[k].tolist()}, Final (w): {final_w[k].tolist()}\n")
            f.write("=" * 60 + "\n")

        # Write summary.json
        summary_json_path = os.path.join(output_dir, "summary.json")
        with open(summary_json_path, "w") as f:
            json.dump({
                "experiment": cfg.experiment,
                "num_keys": num_keys,
                "value_width": value_width,
                "learning_rate": learning_rate,
                "workers": num_workers,
                "iterations_per_worker": iterations,
                "initial_l2_distance": initial_l2,
                "final_l2_distance": final_l2,
                "target_weights_sample": w_star[:10].tolist(),
                "final_weights_sample": final_w[:10].tolist(),
            }, f, indent=2)

        # Console feedback
        print(f"\n" + "=" * 60)
        print(f"[{cfg.experiment}] Evaluation Results:")
        print(f"  Initial L2 Distance: {initial_l2:.4f}")
        print(f"  Final L2 Distance:   {final_l2:.4f}")
        print(f"  Reduction:           {pct_improvement:.2f}%")
        print(f"  Saved artifacts to:  {output_dir}")
        print(f"    - {output_dir}/summary.txt")
        print(f"    - {output_dir}/summary.json")
        print(f"    - {output_dir}/final_weights.npy")
        print(f"    - {output_dir}/observer_metrics.csv")
        print("=" * 60 + "\n")

        # Convergence assertion
        assert final_l2 < initial_l2, f"Failed to converge: {final_l2} >= {initial_l2}"

    finally:
        if observer:
            observer.stop()
        if server_proc and server_proc.poll() is None:
            server_proc.kill()
            server_proc.wait()