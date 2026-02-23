#!/usr/bin/env python3
"""Module for the maximization step of the EM algorithm for a GMM."""
import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm for a GMM."""

    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(g, np.ndarray) or g.ndim != 2):
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n_g != n:
        return None, None, None

    if np.any(g < 0):
        return None, None, None

    # 🔥 REQUIRED CHECK
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    n_k = np.sum(g, axis=1)

    if np.any(n_k == 0):
        return None, None, None

    pi = n_k / n
    m = (g @ X) / n_k[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = (g[i][:, np.newaxis] * diff).T @ diff / n_k[i]

    return pi, m, S

