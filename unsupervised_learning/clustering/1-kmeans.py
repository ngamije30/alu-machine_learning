#!/usr/bin/env python3
"""Module for performing K-means clustering."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer, number of clusters.
        iterations: positive integer, max number of iterations.

    Returns:
        Tuple (C, clss) or (None, None) on failure.
        C: numpy.ndarray of shape (k, d) with centroid means.
        clss: numpy.ndarray of shape (n,) with cluster indices.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, size=(k, d))
    for _ in range(iterations):
        diffs = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        clss = np.argmin(dists, axis=1)
        C_new = np.array([
            X[clss == j].mean(axis=0) if np.any(clss == j)
            else np.random.uniform(low, high)
            for j in range(k)
        ])
        if np.allclose(C, C_new):
            return C_new, clss
        C = C_new
    diffs = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    clss = np.argmin(dists, axis=1)
    return C, clss
