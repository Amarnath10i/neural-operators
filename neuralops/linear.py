"""PCA + ridge regression: the linear baseline every operator learner should beat.

Inputs and outputs are projected on their leading principal components
(fitted on the training set) and a ridge map is learned between the
coefficients. Fixed to the training grid.
"""
from __future__ import annotations

import numpy as np


class PCARidge:
    def __init__(self, k_in: int = 64, k_out: int = 64, alpha: float = 1e-3):
        self.k_in, self.k_out, self.alpha = k_in, k_out, alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PCARidge":
        X, Y = x.reshape(len(x), -1).astype(float), y.reshape(len(y), -1).astype(float)
        self.shape = y.shape[1:]
        self.mx, self.my = X.mean(0), Y.mean(0)
        self.Vx = np.linalg.svd(X - self.mx, full_matrices=False)[2][:self.k_in]
        self.Vy = np.linalg.svd(Y - self.my, full_matrices=False)[2][:self.k_out]
        A, B = (X - self.mx) @ self.Vx.T, (Y - self.my) @ self.Vy.T
        self.W = np.linalg.solve(A.T @ A + self.alpha * len(A) * np.eye(A.shape[1]), A.T @ B)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        X = x.reshape(len(x), -1).astype(float)
        return (((X - self.mx) @ self.Vx.T) @ self.W @ self.Vy + self.my).reshape((len(x),) + self.shape)
