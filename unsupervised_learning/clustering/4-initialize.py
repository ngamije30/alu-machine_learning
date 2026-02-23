#!/usr/bin/env python3
"""Module for initializing variables for a Gaussian Mixture Model."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize variables for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        k: positive integer, number of clusters.

    Returns:
        Tuple (pi, m, S) or (None, None, None) on failure.
        pi: numpy.ndarray of shape (k,) with priors (evenly initialized).
        m: numpy.ndarray of shape (k, d) with centroid means from K-means.
        S: numpy.ndarray of shape (k, d, d) with identity covariance matrices.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    d = X.shape[1]
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    pi = np.full((k,), 1 / k)
    S = np.tile(np.eye(d), (k, 1, 1))
    return pi, m, S
