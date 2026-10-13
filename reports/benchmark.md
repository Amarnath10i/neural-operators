# Benchmark results

Mean relative L2 error on 200 test samples (95% bootstrap interval). Lower is better.

## Burgers (ν = 0.1), trained at 256 points

Native: the model is run directly on the finer grid. Interpolated: the model predicts at 256 points and the prediction is Fourier-interpolated to the finer grid.

| model | params | 256 | 512 native | 1024 native | 512 interpolated | 1024 interpolated |
|---|---|---|---|---|---|---|
| FNO | 549,569 | 0.39% (0.37–0.42) | 0.39% (0.37–0.42) | 0.39% (0.37–0.42) | 0.39% (0.37–0.42) | 0.39% (0.37–0.42) |
| DeepONet | 234,753 | 0.32% (0.29–0.36) | 0.32% (0.29–0.36) | 0.32% (0.29–0.36) | 0.32% (0.29–0.36) | 0.32% (0.29–0.36) |
| CNN | 329,153 | 16.82% (15.18–18.73) | 54.04% (48.91–60.04) | 72.07% (64.98–80.06) | 16.82% (15.18–18.73) | 16.82% (15.18–18.73) |
| PCA + ridge | 36,864 | 11.57% (10.71–12.45) | — | — | 11.57% (10.71–12.45) | 11.57% (10.71–12.45) |

Training-set size (error at 256 points):

| model | 100 | 250 | 500 | 1000 |
|---|---|---|---|---|
| FNO | 0.38% (0.35–0.42) | 0.89% (0.82–0.99) | 0.39% (0.36–0.43) | 0.39% (0.37–0.42) |
| CNN | 23.48% (21.19–26.08) | 18.37% (16.55–20.47) | 16.96% (15.35–18.87) | 16.82% (15.18–18.73) |
| PCA + ridge | 14.13% (12.80–15.54) | 12.77% (11.81–13.79) | 12.10% (11.21–13.05) | 11.57% (10.71–12.45) |

Time per sample at 1024 points on a laptop CPU: spectral solver (NumPy, CPU) 371.9 ms, FNO (NumPy, CPU) 31.1 ms (NumPy FNO matches PyTorch to 1.3e-08).

## Darcy flow, trained at 65×65

| model | params | 33×33 native | 65×65 | 129×129 native | 33×33 interpolated | 129×129 interpolated |
|---|---|---|---|---|---|---|
| FNO | 2,368,001 | 3.35% (3.29–3.42) | 1.09% (1.03–1.16) | 2.05% (1.99–2.13) | 1.42% (1.35–1.49) | 1.10% (1.03–1.17) |
| U-Net | 465,953 | 61.36% (61.04–61.68) | 6.17% (5.85–6.52) | 52.76% (51.99–53.60) | 8.82% (8.16–9.50) | 6.12% (5.79–6.46) |
| PCA + ridge | 544,896 | — | 6.92% (6.76–7.09) | — | 6.94% (6.78–7.11) | 6.92% (6.76–7.09) |

Time per sample at 129×129 on a laptop CPU: finite-volume solver (SciPy sparse, CPU) 65.6 ms, FNO (NumPy, CPU) 413.1 ms (NumPy FNO matches PyTorch to 8.1e-07).
