"""Train and evaluate every model on Burgers and Darcy; write reports/ and the app's exported models.

    python scripts/make_data.py && python scripts/run_benchmark.py

Hyperparameters follow Li et al. (2021) and are not tuned, so there is no
validation split: the 200 test samples are only used for the numbers below.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuralops import numpy_fno  # noqa: E402
from neuralops.linear import PCARidge  # noqa: E402
from neuralops.models import CNN1d, DeepONet1d, FNO1d, FNO2d, UNet2d, n_params  # noqa: E402
from neuralops.pdes import burgers, darcy  # noqa: E402
from neuralops.train import fit, fourier_resample, grid_resample, predict, rel_l2  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN, N_TEST = 1000, 200
QUICK = os.environ.get("QUICK") == "1"  # tiny run for smoke tests
EPOCHS_1D, EPOCHS_2D = (5, 3) if QUICK else (500, 300)


def summary(err: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    boots = err[rng.integers(0, len(err), (2000, len(err)))].mean(1)
    return {"mean": float(err.mean()), "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "median": float(np.median(err))}


def log(msg):
    print(msg, flush=True)


def burgers_benchmark(rep):
    d = np.load("data/burgers_1024.npz")
    u0, u1 = d["u0"], d["u1"]
    tr, te = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_TEST)
    res_train, res_test = 256, (256, 512, 1024)
    sub = lambda a, n: a[:, ::1024 // n]  # noqa: E731  (the 1024 grid nests the coarser ones)
    xtr, ytr = sub(u0[tr], res_train), sub(u1[tr], res_train)

    models = {"FNO": FNO1d(), "DeepONet": DeepONet1d(), "CNN": CNN1d()}
    out = {"train_resolution": res_train, "params": {}, "native": {}, "interp": {}}
    preds_256 = {}
    for name, m in models.items():
        log(f"  Burgers {name} ({n_params(m):,} params)")
        fit(m, xtr, ytr, epochs=EPOCHS_1D, device=DEV, log=log)
        out["params"][name] = n_params(m)
        for n in res_test:
            p = predict(m, sub(u0[te], n), device=DEV)
            out["native"].setdefault(name, {})[n] = summary(rel_l2(p, sub(u1[te], n)))
        preds_256[name] = predict(m, sub(u0[te], 256), device=DEV)
    pca = PCARidge().fit(xtr, ytr)
    out["params"]["PCA + ridge"] = int(pca.Vx.size + pca.Vy.size + pca.W.size)
    preds_256["PCA + ridge"] = pca.predict(sub(u0[te], 256))
    # Every model's 256-point prediction, Fourier-interpolated to the finer grids. (The finer grids
    # nest the 256-point one, so this uses no information the native run does not have.)
    for name, p in preds_256.items():
        for n in res_test:
            out["interp"].setdefault(name, {})[n] = summary(rel_l2(fourier_resample(p, n), sub(u1[te], n)))

    # Data efficiency at the training resolution.
    out["data_efficiency"] = {}
    for n_tr in (100, 250, 500, 1000):
        for name, make in (("FNO", FNO1d), ("CNN", CNN1d)):
            if n_tr == N_TRAIN:
                r = out["native"][name][256]
            else:
                m = fit(make(), xtr[:n_tr], ytr[:n_tr], epochs=EPOCHS_1D, device=DEV, log=lambda *_: None)
                r = summary(rel_l2(predict(m, sub(u0[te], 256), device=DEV), sub(u1[te], 256)))
            out["data_efficiency"].setdefault(name, {})[n_tr] = r
        p = PCARidge().fit(xtr[:n_tr], ytr[:n_tr]).predict(sub(u0[te], 256))
        out["data_efficiency"].setdefault("PCA + ridge", {})[n_tr] = summary(rel_l2(p, sub(u1[te], 256)))
        log(f"  data efficiency n={n_tr} done")

    # Speed: the trained FNO against the solver that generated the data, per sample.
    fno = models["FNO"]
    w = numpy_fno.export(fno.cpu())
    x1024 = sub(u0[te], 1024)[:20].astype(np.float64)
    t = time.perf_counter(); burgers(x1024[:5]); t_solver = (time.perf_counter() - t) / 5
    t = time.perf_counter(); numpy_fno.fno1d(w, x1024); t_np = (time.perf_counter() - t) / 20
    out["seconds_per_sample_1024"] = {"spectral solver (NumPy, CPU)": t_solver, "FNO (NumPy, CPU)": t_np}
    out["numpy_vs_torch_max_abs_diff"] = float(np.abs(numpy_fno.fno1d(w, x1024[:4]) -
                                                       predict(fno, x1024[:4].astype(np.float32), device="cpu")).max())
    os.makedirs("models", exist_ok=True)
    np.savez_compressed("models/burgers_fno.npz", **{k: (v.astype(np.complex64) if np.iscomplexobj(v) else v)
                                                      for k, v in w.items()})
    os.makedirs("app_data", exist_ok=True)
    np.savez_compressed("app_data/burgers_test.npz", u0=u0[te][:24], u1=u1[te][:24])
    rep["burgers"] = out


def darcy_benchmark(rep):
    d = np.load("data/darcy_129.npz")
    a, u = d["a"], d["u"]
    tr, te = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_TEST)
    sub = lambda f, n: f[:, ::128 // (n - 1), ::128 // (n - 1)]  # noqa: E731  (nested vertex grids)
    res_train, res_test = 65, (33, 65, 129)
    # Scalar normalisation (a per-point one would be tied to the training grid).
    a_mu, a_sd, u_sd = float(a[tr].mean()), float(a[tr].std()), float(u[tr].std())
    na = lambda x: (x - a_mu) / a_sd  # noqa: E731
    xtr, ytr = na(sub(a[tr], res_train)).astype(np.float32), (sub(u[tr], res_train) / u_sd).astype(np.float32)

    models = {"FNO": FNO2d(), "U-Net": UNet2d()}
    out = {"train_resolution": res_train, "params": {}, "native": {}, "interp": {}}
    for name, m in models.items():
        log(f"  Darcy {name} ({n_params(m):,} params)")
        fit(m, xtr, ytr, epochs=EPOCHS_2D, device=DEV, log=log)
        out["params"][name] = n_params(m)
        for n in res_test:
            p = predict(m, na(sub(a[te], n)).astype(np.float32), batch=20, device=DEV) * u_sd
            out["native"].setdefault(name, {})[n] = summary(rel_l2(p, sub(u[te], n)))
    pca = PCARidge().fit(xtr, ytr)
    out["params"]["PCA + ridge"] = int(pca.Vx.size + pca.Vy.size + pca.W.size)
    # Fixed-resolution route, using only the n x n input: resample the input to the training grid,
    # predict there, resample the prediction back to n x n.
    for name, m in models.items():
        for n in res_test:
            x = na(grid_resample(sub(a[te], n), res_train)).astype(np.float32)
            p = predict(m, x, batch=20, device=DEV) * u_sd
            out["interp"].setdefault(name, {})[n] = summary(rel_l2(grid_resample(p, n), sub(u[te], n)))
    for n in res_test:
        p = pca.predict(na(grid_resample(sub(a[te], n), res_train))) * u_sd
        out["interp"].setdefault("PCA + ridge", {})[n] = summary(rel_l2(grid_resample(p, n), sub(u[te], n)))

    fno = models["FNO"]
    w = numpy_fno.export(fno.cpu())
    w.update(a_mu=np.array(a_mu), a_sd=np.array(a_sd), u_sd=np.array(u_sd))
    x = sub(a[te], 129)[:4]
    t = time.perf_counter()
    for ai in x:
        darcy(ai.astype(float))
    t_solver = (time.perf_counter() - t) / len(x)
    t = time.perf_counter(); numpy_fno.fno2d(w, na(x)); t_np = (time.perf_counter() - t) / len(x)
    out["seconds_per_sample_129"] = {"finite-volume solver (SciPy sparse, CPU)": t_solver, "FNO (NumPy, CPU)": t_np}
    out["numpy_vs_torch_max_abs_diff"] = float(np.abs(numpy_fno.fno2d(w, na(x[:2])) -
                                                       predict(fno, na(x[:2]).astype(np.float32), device="cpu")).max())
    np.savez_compressed("models/darcy_fno.npz", **{k: (v.astype(np.complex64) if np.iscomplexobj(v) else v)
                                                    for k, v in w.items()})
    np.savez_compressed("app_data/darcy_test.npz", a=a[te][:12].astype(np.int8), u=u[te][:12])
    rep["darcy"] = out


def _int_keys(d):
    """JSON turns grid sizes into string keys; turn them back for the report."""
    return {int(k) if k.isdigit() else k: v for k, v in d.items()}


def write_markdown(rep):
    b, dd = rep["burgers"], rep["darcy"]
    f = lambda s: f"{100 * s['mean']:.2f}% ({100 * s['ci'][0]:.2f}–{100 * s['ci'][1]:.2f})"  # noqa: E731
    L = ["# Benchmark results", "",
         "Mean relative L2 error on 200 test samples (95% bootstrap interval). Lower is better.", "",
         f"## Burgers (ν = 0.1), trained at {b['train_resolution']} points", "",
         "Native: the model is run directly on the finer grid. Interpolated: the model predicts at 256 points "
         "and the prediction is Fourier-interpolated to the finer grid.", "",
         "| model | params | 256 | 512 native | 1024 native | 512 interpolated | 1024 interpolated |",
         "|---|---|---|---|---|---|---|"]
    for name in b["interp"]:
        nat = b["native"].get(name)
        L.append(f"| {name} | {b['params'][name]:,} | {f(b['interp'][name][256])} | "
                 f"{f(nat[512]) if nat else '—'} | {f(nat[1024]) if nat else '—'} | "
                 f"{f(b['interp'][name][512])} | {f(b['interp'][name][1024])} |")
    L += ["", "Training-set size (error at 256 points):", "", "| model | " +
          " | ".join(str(k) for k in b["data_efficiency"]["FNO"]) + " |", "|---|" + "---|" * len(b["data_efficiency"]["FNO"])]
    for name, row in b["data_efficiency"].items():
        L.append(f"| {name} | " + " | ".join(f(v) for v in row.values()) + " |")
    s = b["seconds_per_sample_1024"]
    L += ["", "Time per sample at 1024 points on a laptop CPU: " +
          ", ".join(f"{k} {1000 * v:.1f} ms" for k, v in s.items()) +
          f" (NumPy FNO matches PyTorch to {b['numpy_vs_torch_max_abs_diff']:.1e}).", "",
          f"## Darcy flow, trained at {dd['train_resolution']}×{dd['train_resolution']}", "",
          "| model | params | 33×33 native | 65×65 | 129×129 native | 33×33 interpolated | 129×129 interpolated |",
          "|---|---|---|---|---|---|---|"]
    for name in dd["interp"]:
        nat = dd["native"].get(name)
        L.append(f"| {name} | {dd['params'][name]:,} | {f(nat[33]) if nat else '—'} | {f(dd['interp'][name][65])} | "
                 f"{f(nat[129]) if nat else '—'} | {f(dd['interp'][name][33])} | {f(dd['interp'][name][129])} |")
    s = dd["seconds_per_sample_129"]
    L += ["", "Time per sample at 129×129 on a laptop CPU: " +
          ", ".join(f"{k} {1000 * v:.1f} ms" for k, v in s.items()) +
          f" (NumPy FNO matches PyTorch to {dd['numpy_vs_torch_max_abs_diff']:.1e}).", ""]
    open("reports/benchmark.md", "w", encoding="utf-8", newline="\n").write("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs("reports", exist_ok=True)
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    rep = {"device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "torch": torch.__version__}
    if only:  # rerun one PDE, keep the other's stored results
        rep = {**json.load(open("reports/results.json")), **rep}
    t = time.time()
    if only in (None, "burgers"):
        burgers_benchmark(rep)
    if only in (None, "darcy"):
        darcy_benchmark(rep)
    rep["minutes_last_run"] = (time.time() - t) / 60
    json.dump(rep, open("reports/results.json", "w"), indent=2)
    write_markdown(json.loads(json.dumps(rep), object_hook=_int_keys))
    log(open("reports/benchmark.md", encoding="utf-8").read())
