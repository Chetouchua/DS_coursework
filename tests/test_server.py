import threading
import numpy as np
import pytest
from src.client import Client, Range


def test_concurrent_pushes():
    """Verify that concurrent pushes to the same keys are not lost or corrupted."""
    num_threads = 5
    pushes_per_thread = 20
    learning_rate = 0.05

    def worker_job(worker_id):
        client = Client(worker_id=worker_id, server_addr="localhost:50051").connect()
        for t in range(pushes_per_thread):
            # Each push sends gradient = 1.0 for keys [10, 15)
            client.push(Range(10, 15), np.ones(5), iteration=t).wait()
        client.close()

    threads = [
        threading.Thread(target=worker_job, args=(w,))
        for w in range(1, num_threads + 1)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # Verify total update
    client = Client(worker_id=99, server_addr="localhost:50051").connect()
    weights = client.pull(Range(10, 15)).wait().ravel()
    client.close()

    # Expected: 0.0 - (lr * total_pushes * 1.0)
    total_pushes = num_threads * pushes_per_thread
    expected_val = -(learning_rate * total_pushes)
    assert np.allclose(weights, expected_val)