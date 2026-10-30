"""
Dedicated out-of-band observer thread to periodically snapshot model state.
"""
import csv
import os
import threading
import time
from typing import Optional
import numpy as np

from src.client import Client, Range


class ModelObserver(threading.Thread):
    def __init__(
        self,
        server_addr: str,
        num_keys: int,
        target_weights_path: str,
        output_dir: str,
        interval_sec: float = 1.0,
        worker_id: int = 9999,  # Dedicated observer ID
    ):
        super().__init__(daemon=True)
        self.server_addr = server_addr
        self.num_keys = num_keys
        self.interval_sec = interval_sec
        self.worker_id = worker_id
        self.output_dir = output_dir

        self.w_star = np.load(target_weights_path)
        self._stop_event = threading.Event()

        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.output_dir, "observer_metrics.csv")

    def run(self):
        # Open CSV file and write header
        with open(self.log_file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "wall_time_sec",
                "l2_distance_to_target",
                "mean_weight",
                "std_weight",
                "min_weight",
                "max_weight",
            ])

            start_time = time.time()

            # Connect client
            with Client(worker_id=self.worker_id, server_addr=self.server_addr) as client:
                while not self._stop_event.is_set():
                    try:
                        self._record_snapshot(client, writer, start_time)
                        f.flush()
                    except Exception as e:
                        # Server might still be starting or in teardown
                        pass

                    # Wait for interval or until stopped
                    self._stop_event.wait(self.interval_sec)

                # Final snapshot upon termination
                try:
                    self._record_snapshot(client, writer, start_time)
                    f.flush()
                except Exception:
                    pass

    def _record_snapshot(self, client: Client, writer, start_time: float):
        elapsed = time.time() - start_time

        # Pull entire weight vector
        task = client.pull(Range(0, self.num_keys), iteration=0)
        current_w = task.wait(timeout_ms=5000)

        if current_w is None:
            return

        current_w = np.asarray(current_w, dtype=np.float64).reshape(self.w_star.shape)

        # Compute L2 error: ||w(t) - w*||_2
        l2_dist = float(np.linalg.norm(current_w - self.w_star))
        mean_val = float(np.mean(current_w))
        std_val = float(np.std(current_w))
        min_val = float(np.min(current_w))
        max_val = float(np.max(current_w))

        writer.writerow([
            f"{elapsed:.3f}",
            f"{l2_dist:.6f}",
            f"{mean_val:.6f}",
            f"{std_val:.6f}",
            f"{min_val:.6f}",
            f"{max_val:.6f}",
        ])

    def stop(self):
        """Signal thread to take final snapshot and stop."""
        self._stop_event.set()
        self.join(timeout=2.0)