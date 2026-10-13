"""Interactive demo: trained Fourier Neural Operators against the numerical solvers, at any resolution.

    streamlit run app.py

NumPy only (the models were exported from PyTorch), so it also runs in the
browser through Pyodide/stlite.
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from neuralops import numpy_fno
from neuralops.pdes import burgers, darcy, darcy_coefficient, grf_1d, grf_2d_neumann
from neuralops.metrics import rel_l2

st.set_page_config(page_title="Neural Operators", page_icon="〰️", layout="wide")


@st.cache_resource
def load(name):
    return numpy_fno.load(os.path.join("models", name))


@st.cache_data
def test_set(name):
    with np.load(os.path.join("app_data", name)) as z:
        return {k: z[k] for k in z.files}


st.title("Neural operators: what they learn, and where they break")
st.caption("A Fourier Neural Operator trained on one grid, run against the solver that generated its training "
           "data. Pick a test case or draw a new random input; the model and the solver both run live.")
tab_b, tab_d, tab_r = st.tabs(["Burgers equation", "Darcy flow", "Benchmark results"])

with tab_b:
    st.markdown("**u_t + u u_x = 0.1 u_xx** on a periodic interval. The FNO maps the initial condition to the "
                "solution at t = 1. It was trained at **256 points**; finer grids are zero-shot. Note the scales: "
                "viscosity leaves under 2% of the initial amplitude by t = 1, which is why this classic benchmark is easy.")
    c1, c2, c3 = st.columns(3)
    src = c1.radio("Input", ["Held-out test sample", "New random initial condition"], key="b_src")
    n = c2.select_slider("Grid points", [128, 256, 512, 1024], value=1024)
    if src.startswith("Held"):
        i = c3.slider("Test sample", 0, 23, 0)
        d = test_set("burgers_test.npz")
        u0_full, u1_full = d["u0"][i].astype(float), d["u1"][i].astype(float)
        u0 = u0_full[:: 1024 // n]
        t0 = time.perf_counter()
        truth = burgers(u0)[0]  # re-solve at this resolution so the timing is comparable
        t_solver = time.perf_counter() - t0
    else:
        seed = c3.number_input("Seed", 0, 10_000, 7)
        u0 = grf_1d(1, n, np.random.default_rng(int(seed)))[0]
        t0 = time.perf_counter()
        truth = burgers(u0)[0]
        t_solver = time.perf_counter() - t0
    w = load("burgers_fno.npz")
    t0 = time.perf_counter()
    pred = numpy_fno.fno1d(w, u0)[0]
    t_fno = time.perf_counter() - t0
    err = rel_l2(pred[None], truth[None])[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Relative L2 error", f"{100 * err:.2f}%")
    m2.metric("Solver time", f"{1000 * t_solver:.0f} ms")
    m3.metric("FNO time", f"{1000 * t_fno:.1f} ms")
    x = np.arange(n) / n
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].plot(x, u0, color="0.4")
    ax[0].set_title("initial condition u(x, 0)")
    ax[1].plot(x, truth, label="solver", lw=2)
    ax[1].plot(x, pred, "--", label="FNO")
    ax[1].set_title("solution u(x, 1)")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("x")
    fig.tight_layout()
    st.pyplot(fig)

with tab_d:
    st.markdown("**−∇·(a∇u) = 1** on the unit square, u = 0 on the boundary. The FNO maps the permeability a "
                "(two materials, 3 and 12) to the pressure u. It was trained at **65×65**.")
    c1, c2, c3 = st.columns(3)
    src = c1.radio("Input", ["Held-out test sample", "New random permeability"], key="d_src")
    n = c2.select_slider("Grid", [33, 65, 129], value=129)
    if src.startswith("Held"):
        i = c3.slider("Test sample", 0, 11, 0, key="d_i")
        d = test_set("darcy_test.npz")
        a = d["a"][i].astype(float)[:: 128 // (n - 1), :: 128 // (n - 1)]
    else:
        seed = c3.number_input("Seed", 0, 10_000, 3, key="d_seed")
        a = darcy_coefficient(grf_2d_neumann(1, n, np.random.default_rng(int(seed))))[0]
    t0 = time.perf_counter()
    truth = darcy(a)
    t_solver = time.perf_counter() - t0
    w = load("darcy_fno.npz")
    t0 = time.perf_counter()
    pred = numpy_fno.fno2d(w, (a - float(w["a_mu"])) / float(w["a_sd"]))[0] * float(w["u_sd"])
    t_fno = time.perf_counter() - t0
    err = rel_l2(pred[None], truth[None])[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Relative L2 error", f"{100 * err:.2f}%")
    m2.metric("Solver time", f"{1000 * t_solver:.0f} ms")
    m3.metric("FNO time", f"{1000 * t_fno:.0f} ms")
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
    vmax = truth.max()
    for axi, img, title, kw in ((ax[0], a, "permeability a", {"cmap": "gray"}),
                                (ax[1], truth, "pressure u (solver)", {"vmin": 0, "vmax": vmax}),
                                (ax[2], pred, "pressure u (FNO)", {"vmin": 0, "vmax": vmax}),
                                (ax[3], np.abs(pred - truth), "|error|", {"cmap": "magma"})):
        im = axi.imshow(img, origin="lower", **kw)
        axi.set_title(title)
        axi.set_xticks([])
        axi.set_yticks([])
        fig.colorbar(im, ax=axi, fraction=0.046)
    fig.tight_layout()
    st.pyplot(fig)

with tab_r:
    if os.path.exists("reports/benchmark.md"):
        st.markdown(open("reports/benchmark.md", encoding="utf-8").read())
    else:
        st.info("Run `python scripts/run_benchmark.py` to produce the benchmark.")
