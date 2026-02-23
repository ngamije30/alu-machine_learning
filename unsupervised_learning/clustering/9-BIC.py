#!/usr/bin/env python3
"""Module for finding best number of GMM clusters using BIC."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: positive integer, minimum clusters to check (inclusive).
        kmax: positive integer, maximum clusters to check (inclusive).
        iterations: positive integer, max iterations for EM algorithm.
        tol: non-negative float, tolerance for EM algorithm.
        verbose: boolean, whether EM prints information.

    Returns:
        Tuple (best_k, best_result, l, b) or (None, None, None, None)
        on failure.
        best_k: best value for k based on BIC.
        best_result: tuple (pi, m, S) for the best number of clusters.
        l: numpy.ndarray of shape (kmax - kmin + 1) with log likelihoods.
        b: numpy.ndarray of shape (kmax - kmin + 1) with BIC values.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmax < kmin + 1:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    log_likelihoods = []
    bic_values = []
    results = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None
        p = (k - 1) + k * d + k * d * (d + 1) // 2
        bic = p * np.log(n) - 2 * log_l
        log_likelihoods.append(log_l)
        bic_values.append(bic)
        results.append((pi, m, S))
    l = np.array(log_likelihoods)
    b = np.array(bic_values)
    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]
    return best_k, best_result, l, b
