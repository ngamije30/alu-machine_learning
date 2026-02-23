#!/usr/bin/env python3
"""Module for calculating the PDF of a Gaussian distribution."""
import numpy as np


def pdf(X, m, S):
    """Calculate the probability density function of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) with data points.
        m: numpy.ndarray of shape (d,) with mean of the distribution.
        S: numpy.ndarray of shape (d, d) with covariance of the distribution.

    Returns:
        P: numpy.ndarray of shape (n,) with PDF values (min 1e-300),
        or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    n, d = X.shape
    if m.shape[0] != d or S.shape != (d, d):
        return None
    det = np.linalg.det(S)
    if det == 0:
        return None
    S_inv = np.linalg.inv(S)
    diff = X - m
    exponent = -0.5 * np.sum(diff @ S_inv * diff, axis=1)
    coeff = 1.0 / (np.sqrt(((2 * np.pi) ** d) * det))
    P = coeff * np.exp(exponent)
    return np.maximum(P, 1e-300)
