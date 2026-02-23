#!/usr/bin/env python3
"""Module for initializing K-means cluster centroids."""
import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means using uniform distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer, number of clusters.

    Returns:
        numpy.ndarray of shape (k, d) with initialized centroids,
        or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)
    return np.random.uniform(low, high, size=(k, d))
