#!/usr/bin/env python3
"""Module for PCA with a fixed number of dimensions."""
import numpy as np


def pca(X, ndim):
    """Perform PCA on a dataset to a fixed number of dimensions.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        ndim: new dimensionality of the transformed X.

    Returns:
        T: numpy.ndarray of shape (n, ndim) - the transformed version of X.
    """
    X_m = X - np.mean(X, axis=0)
    U, s, Vh = np.linalg.svd(X_m)
    W = Vh[:ndim].T
    return np.matmul(X_m, W)
