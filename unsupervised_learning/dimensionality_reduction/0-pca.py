#!/usr/bin/env python3
"""Module for PCA using fraction of variance."""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset maintaining a fraction of variance.

    Args:
        X: numpy.ndarray of shape (n, d) with zero mean across all points.
        var: fraction of variance the PCA transformation should maintain.

    Returns:
        W: numpy.ndarray of shape (d, nd) - the weights matrix,
        where nd is the new dimensionality.
    """
    U, s, Vh = np.linalg.svd(X)
    cumvar = np.cumsum(s ** 2) / np.sum(s ** 2)
    nd = np.argmax(cumvar >= var) + 1
    return Vh[:nd].T
