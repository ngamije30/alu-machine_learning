#!/usr/bin/env python3
"""Module for the expectation step of the EM algorithm for a GMM."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculate the expectation step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        pi: numpy.ndarray of shape (k,) with priors for each cluster.
        m: numpy.ndarray of shape (k, d) with centroid means.
        S: numpy.ndarray of shape (k, d, d) with covariance matrices.

    Returns:
        Tuple (g, l) or (None, None) on failure.
        g: numpy.ndarray of shape (k, n) with posterior probabilities.
        l: total log likelihood (float).
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None
    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])
    total = g.sum(axis=0)
    g = g / total
    log_likelihood = np.sum(np.log(total))
    return g, log_likelihood
