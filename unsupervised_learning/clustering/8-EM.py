#!/usr/bin/env python3
"""Module for the full Expectation-Maximization algorithm for a GMM."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Perform expectation maximization for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        k: positive integer, number of clusters.
        iterations: positive integer, max number of iterations.
        tol: non-negative float, tolerance for early stopping.
        verbose: boolean, whether to print log likelihood info.

    Returns:
        Tuple (pi, m, S, g, l) or (None, None, None, None, None) on failure.
        pi: numpy.ndarray of shape (k,) with cluster priors.
        m: numpy.ndarray of shape (k, d) with centroid means.
        S: numpy.ndarray of shape (k, d, d) with covariance matrices.
        g: numpy.ndarray of shape (k, n) with posterior probabilities.
        l: log likelihood of the model.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None
    l_prev = 0
    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(l, 5)))
        if i > 0 and abs(l - l_prev) <= tol:
            break
        l_prev = l
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None
    g, l = expectation(X, pi, m, S)
    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i, round(l, 5)))
    return pi, m, S, g, l
