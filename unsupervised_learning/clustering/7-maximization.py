#!/usr/bin/env python3
"""Module for the maximization step of the EM algorithm for a GMM."""
import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm for a GMM."""

    # Check X
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    # Check g
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    # Check dimensions match
    if n_g != n:
        return None, None, None

    # Check probabilities valid
    if np.any(g < 0):
        return None, None, None

    # Columns must sum to 1
    if not np.allclose(g.sum(axis=0), np.ones(n)):
        return None, None, None

    # Compute n_k
    n_k = np.sum(g, axis=1)

    # Avoid division by zero
    if np.any(n_k == 0):
        return None, None, None

    # Update priors
    pi = n_k / n

    # Update means
    m = (g @ X) / n_k[:, np.newaxis]

    # Update covariance matrices
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        weighted = g[i][:, np.newaxis] * diff
        S[i] = weighted.T @ diff / n_k[i]

    return pi, m, S