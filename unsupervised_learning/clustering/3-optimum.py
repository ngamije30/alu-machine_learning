#!/usr/bin/env python3
"""Module for finding the optimum number of clusters by variance."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Test for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: positive integer, minimum number of clusters (inclusive).
        kmax: positive integer, maximum number of clusters (inclusive).
        iterations: positive integer, max iterations for K-means.

    Returns:
        Tuple (results, d_vars) or (None, None) on failure.
        results: list of K-means outputs for each cluster size.
        d_vars: list of variance differences from smallest cluster size.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmax - kmin < 1:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    results = []
    d_vars = []
    var_min = None
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        var_k = variance(X, C)
        if var_min is None:
            var_min = var_k
        d_vars.append(var_min - var_k)
    return results, d_vars
