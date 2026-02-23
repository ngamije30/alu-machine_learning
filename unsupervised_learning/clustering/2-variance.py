#!/usr/bin/env python3
"""Module for calculating total intra-cluster variance."""
import numpy as np


def variance(X, C):
    """Calculate total intra-cluster variance for a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        C: numpy.ndarray of shape (k, d) containing centroid means.

    Returns:
        var (float), the total variance, or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    diffs = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    min_dists = np.min(dists, axis=1)
    return np.sum(min_dists ** 2)
