"""Generate the Burgers and Darcy datasets into data/ (about 10 minutes on a laptop CPU).

    python scripts/make_data.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuralops.pdes import make_burgers, make_darcy  # noqa: E402

N_SAMPLES = 1200  # 1000 train + 200 test

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    t = time.time()
    u0, u1 = make_burgers(N_SAMPLES, n=1024, seed=0)
    np.savez_compressed("data/burgers_1024.npz", u0=u0, u1=u1)
    print(f"burgers: {u0.shape} in {time.time() - t:.0f}s", flush=True)
    t = time.time()
    a, u = make_darcy(N_SAMPLES, n=129, seed=1)
    np.savez_compressed("data/darcy_129.npz", a=a, u=u)
    print(f"darcy: {a.shape} in {time.time() - t:.0f}s", flush=True)
