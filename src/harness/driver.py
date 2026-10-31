import json
import os
import subprocess
import sys
import time
from typing import List, Optional

import numpy as np

from src.client import Client, Range
from src.harness.config import ExperimentConfig
from src.harness.observer import ModelObserver
from src.harness.oracle import CorrectnessOracle


class Driver:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.server_proc: Optional[subprocess.Popen] = None
        self.worker_procs: List[subprocess.Popen] = []
        self.observer: Optional[ModelObserver] = None

    def run(self) -> None:
        port = 50055
        server_addr = f"localhost:{port}"
        out_dir = getattr(self.config, "output", None)
        output_dir = out_dir.dir if out_dir and out_dir.dir else f"results/{self.config.experiment}"
        os.makedirs(output_dir, exist_ok=True)

        env = dict(os.environ)
        env["PYTHONPATH"] = "."

        # 1. Generate and save ground-truth w* vector
        oracle = CorrectnessOracle(self.config.model, self.config.seed)
        target_file = oracle.save(self.config.experiment)
        w_star = oracle.w_star

        # 2. Start Parameter Server
        server_cmd = [
            sys.executable, "-m", "src.server.service",
            "--num-keys", str(self.config.model.num_keys),
            "--value-width", str(self.config.model.value_width),
            "--learning-rate", str(self.config.model.learning_rate),
            "--port", str(port),
        ]
        print("[*] Starting Parameter Server...")
        self.server_proc = subprocess.Popen(server_cmd, env=env)
        time.sleep(1.0)  # Allow socket initialization

        # 3. Start Background Observer
        observer_interval = getattr(out_dir, "observer_every", 1) or 1
        print(f"[*] Starting background observer (logging to {output_dir}/observer_metrics.csv)...")
        self.observer = ModelObserver(
            server_addr="localhost:50051",
            num_keys=self.config.model.num_keys,
            target_weights_path=target_file,
            output_dir=output_dir,
            interval_sec=float(observer_interval),
        )
        self.observer.start()

        # 4. Spawn Workers
        workers_cfg = self.config.workers
        access_cfg = getattr(workers_cfg, "access", None) or getattr(self.config, "access", None)
        gradient_cfg = getattr(workers_cfg, "gradient", None) or getattr(self.config, "gradient", None)

        mode = access_cfg.mode if access_cfg else "dense"
        if mode == "sparse":
            keys_per_iter = (
                getattr(access_cfg, "keys_per_iter", None)
                or getattr(access_cfg, "subset_size", None)
                or 100  # Reasonable sparse default
            )
        else:
            # Dense mode uses range_size
            keys_per_iter = (
                getattr(access_cfg, "range_size", None)
                or getattr(access_cfg, "keys_per_iter", None)
                or self.config.model.num_keys
            )
        print(keys_per_iter)
        gradient_type = gradient_cfg.type if gradient_cfg else "target_diff"
        gradient_noise = gradient_cfg.noise_std if gradient_cfg else 0.0

        print(f"[*] Launching {workers_cfg.count} worker(s)...")
        for i in range(workers_cfg.count):
            worker_seed = self.config.seed + i
            worker_cmd = [
                sys.executable, "-m", "src.harness.worker",
                "--worker-id", str(i),
                "--num-keys", str(self.config.model.num_keys),
                "--value-width", str(self.config.model.value_width),
                "--iterations", str(workers_cfg.iterations),
                "--mode", str(mode),
                "--keys-per-iter", str(keys_per_iter),
                "--seed", str(worker_seed),
                "--target-file", target_file,
                "--gradient-type", str(gradient_type),
                "--gradient-noise", str(gradient_noise),
                "--server-addr", server_addr,
            ]
            self.worker_procs.append(subprocess.Popen(worker_cmd, env=env))

        # 5. Wait for workers and save evaluation summary
        try:
            for w in self.worker_procs:
                w.wait()
            print("[+] All workers finished.")

            # Pull final weights and record final evaluation summary
            self._save_summary_and_weights(output_dir, w_star)

        except KeyboardInterrupt:
            print("\n[!] Execution interrupted.")
        finally:
            self.teardown()

    def _save_summary_and_weights(self, output_dir: str, w_star: np.ndarray, server_addr: str = "localhost:50055") -> None:
        """Pulls final model weights from server, computes metrics, and writes summary files."""
        print("[*] Pulling final state and generating summary...")
        try:
            with Client(worker_id=10000, server_addr=server_addr) as client:
                task = client.pull(Range(0, self.config.model.num_keys))
                final_w = task.wait(timeout_ms=5000)

            final_w = np.asarray(final_w, dtype=np.float64).reshape(w_star.shape)
            initial_w = np.zeros_like(w_star)

            initial_l2 = float(np.linalg.norm(initial_w - w_star))
            final_l2 = float(np.linalg.norm(final_w - w_star))

            print("Server max absolute weight:", np.max(np.abs(final_w)))

            # 1. Save raw arrays
            np.save(os.path.join(output_dir, "final_weights.npy"), final_w)

            # 2. Write readable summary log
            summary_txt_path = os.path.join(output_dir, "summary.txt")
            with open(summary_txt_path, "w") as f:
                f.write(f"Experiment: {self.config.experiment}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Initial L2 Distance (w_0 vs w*): {initial_l2:.6f}\n")
                f.write(f"Final L2 Distance (w_final vs w*): {final_l2:.6f}\n")
                f.write(f"L2 Improvement: {initial_l2 - final_l2:.6f} ({((initial_l2 - final_l2) / initial_l2) * 100:.2f}% reduction)\n")
                f.write("-" * 60 + "\n")
                f.write("Weight Samples (First 5 keys):\n")
                for k in range(min(5, self.config.model.num_keys)):
                    f.write(f"  Key {k:03d} -> Target (w*): {w_star[k].tolist()}, Final (w): {final_w[k].tolist()}\n")
                f.write("=" * 60 + "\n")

            # 3. Write structured JSON log (convenient for plotting and scripts)
            summary_json_path = os.path.join(output_dir, "summary.json")
            with open(summary_json_path, "w") as f:
                json.dump({
                    "experiment": self.config.experiment,
                    "num_keys": self.config.model.num_keys,
                    "value_width": self.config.model.value_width,
                    "learning_rate": self.config.model.learning_rate,
                    "workers": self.config.workers.count,
                    "iterations_per_worker": self.config.workers.iterations,
                    "initial_l2_distance": initial_l2,
                    "final_l2_distance": final_l2,
                    "target_weights_sample": w_star[:10].tolist(),
                    "final_weights_sample": final_w[:10].tolist(),
                }, f, indent=2)

            print(f"[+] Summary written to: {summary_txt_path}")
            print(f"[+] Final weights saved to: {output_dir}/final_weights.npy")

        except Exception as e:
            print(f"[!] Warning: Failed to pull final state for summary: {e}")

    def teardown(self) -> None:
        if self.observer:
            print("[*] Stopping observer...")
            self.observer.stop()

        for w in self.worker_procs:
            if w.poll() is None:
                w.kill()
                w.wait()

        if self.server_proc and self.server_proc.poll() is None:
            self.server_proc.kill()
            self.server_proc.wait()

        print("[+] Cluster teardown complete.")