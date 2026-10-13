# Neural operators: what they learn, and where they break

A controlled benchmark of **Fourier Neural Operators** (Li et al., ICLR 2021) against **DeepONet**, a CNN / U-Net and a linear PCA baseline on two classic operator-learning problems, with numerical solvers that are tested against exact solutions and an interactive demo that runs the trained FNO in NumPy, even in the browser.

The question is not "can an FNO fit Burgers" (it can) but which of the usual claims hold up when checked: **resolution-invariant "zero-shot super-resolution"**, **data efficiency**, and **speed against a numerical solver**.

## Findings

Mean relative L2 error on 200 held-out samples, with 95% bootstrap intervals; full tables in [`reports/benchmark.md`](reports/benchmark.md).

**Burgers (u_t + u u_x = 0.1 u_xx, map u(·,0) → u(·,1)), trained at 256 points**

| model | params | 256 points | 1024 points, run natively | 1024 points, 256-point prediction interpolated |
|---|---|---|---|---|
| FNO | 550k | **0.39%** (0.37–0.42) | 0.39% | 0.39% |
| DeepONet | 235k | **0.32%** (0.29–0.36) | 0.32% | 0.32% |
| CNN (circular, 8 residual blocks) | 329k | 16.8% | 72.1% | 16.8% |
| PCA + ridge (linear) | 37k | 11.6% | n/a | 11.6% |

**Darcy flow (−∇·(a∇u) = 1, map permeability a → pressure u), trained at 65×65**

| model | params | 65×65 | 129×129, run natively | 129×129 via the 65×65 grid | 33×33, run natively | 33×33 via the 65×65 grid |
|---|---|---|---|---|---|---|
| FNO | 2.37M | **1.09%** (1.03–1.16) | 2.05% | **1.10%** | 3.35% | **1.42%** |
| U-Net | 466k | 6.17% | 52.8% | 6.12% | 61.4% | 8.82% |
| PCA + ridge | 545k | 6.92% | n/a | 6.92% | n/a | 6.94% |

"Via the 65×65 grid" uses only the input at the target resolution: resample it to 65×65, predict, resample the prediction back.

- **The standard Burgers setting is nearly trivial.** At the FNO paper's ν = 0.1, by t = 1 the solution keeps 1.6% of the initial condition's norm (median over the test set). It is essentially one smooth wave, with no energy above what a 256-point grid can represent (at most 7e-16 of it). Any model that gets the lowest Fourier modes right scores well here, which is worth knowing before quoting Burgers numbers.
- **"Zero-shot super-resolution" never beat plain interpolation.**
  - **Burgers:** running the FNO on a 4× finer grid gives exactly the error of Fourier-interpolating its 256-point prediction. With no content above the coarse grid's Nyquist frequency, interpolation is exact, so this setting cannot tell the two apart.
  - **Darcy:** the FNO is *not* resolution-invariant in practice. Run natively on a finer or coarser grid, its error roughly doubles or triples (2.05% at 129×129, 3.35% at 33×33). Simply resampling to its training grid and back does better in both directions (1.10% and 1.42%). Likely causes, not yet isolated: the permeability is a discontinuous two-material field, and the non-periodic domain needs padding.
- **Grid-tied models fail natively, and interpolation rescues them.** A CNN run on a 4× finer grid goes from 16.8% to 72% error, and the U-Net on Darcy from 6% to 53–61%, because their kernels are tied to the training grid spacing. Resampling through the training grid keeps their accuracy. This is the failure that discretisation invariance protects against, and plain resampling avoids it too.
- **DeepONet matches the FNO on Burgers.** It is slightly better here with fewer parameters (its trunk uses periodic Fourier features of x), so the FNO's advantage is architectural convenience rather than accuracy on this problem.
- **The FNO is data-efficient on Burgers.** 100 training samples already give 0.38%. The 250-sample run came out worse (0.89%) than both 100 and 500, which shows the run-to-run variance of a single training seed; more seeds are needed before reading a trend.
- **Speed depends on the solver you compare with.**
  - **Burgers:** at 1024 points on a laptop CPU, the NumPy FNO takes 31 ms per sample against 372 ms for the pseudo-spectral solver (12× faster).
  - **Darcy:** at 129×129 a sparse finite-volume solve takes 66 ms, while the NumPy FNO takes 413 ms (single-run CPU timings that vary with load). For a 2D elliptic problem with a good direct solver, a neural surrogate is not a speed win.

## Data and solvers

- **Burgers**: periodic Gaussian random field initial conditions, solved pseudo-spectrally (integrating factor + RK4, 2/3 dealiasing) at 1024 points. Tests check the heat-equation limit at small amplitude (error 7.7e-9), conservation of the mean, time-step convergence, and that coarser grids reproduce the fine solution (9e-5).
- **Darcy**: −∇·(a∇u) = 1 on the unit square with a two-material permeability (3 / 12) from a thresholded Gaussian random field, solved with a 5-point finite-volume scheme on nested 33/65/129 vertex grids. A manufactured-solution test confirms second-order convergence.
- 1,000 training and 200 test samples each. Hyperparameters follow Li et al. (2021) and were not tuned, so there is no validation split and the test set only produces the reported numbers. Darcy was run twice: the second run fixed the interpolation baseline to use only the target-resolution input.

## Try it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs the trained FNOs (exported to NumPy) next to the solvers on held-out samples or fresh random inputs, at any grid size, and reports the error and the time of both. It needs no PyTorch, so it also runs in the browser.

## Reproduce

```bash
pip install -r requirements-train.txt
python scripts/make_data.py        # ~20 min on a laptop CPU: 1,200 Burgers and 1,200 Darcy samples
python scripts/run_benchmark.py    # ~1 h on an RTX 3050 laptop GPU: every model, table and export
pytest                             # solvers, models, and NumPy-vs-PyTorch FNO equivalence
```

## Layout

```
neuralops/
  pdes.py          Burgers and Darcy data generators and solvers (NumPy/SciPy)
  models.py        FNO1d/2d, DeepONet, CNN, U-Net (PyTorch)
  numpy_fno.py     FNO inference in NumPy, exported from the PyTorch models
  linear.py        PCA + ridge baseline
  metrics.py       relative L2, Fourier and bilinear resampling
  train.py         training loop (relative-L2 loss, Adam, step decay)
scripts/           data generation and the benchmark
models/            exported FNO weights used by the app
app_data/          a few held-out test samples for the app
reports/           benchmark tables (markdown + JSON)
tests/             14 tests
```

## References

- Li, Z. et al. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. *ICLR*.
- Lu, L. et al. (2021). Learning nonlinear operators via DeepONet. *Nature Machine Intelligence* 3, 218–229.
- Kovachki, N. et al. (2023). Neural Operator: Learning Maps Between Function Spaces. *JMLR* 24.
