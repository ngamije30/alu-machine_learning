#!/usr/bin/env python3
"""Module for Gaussian Mixture Model using sklearn."""
import sklearn.mixture


def gmm(X, k):
    """Calculate a Gaussian Mixture Model from a dataset using sklearn.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: number of clusters.

    Returns:
        Tuple (pi, m, S, clss, bic).
        pi: numpy.ndarray of shape (k,) with cluster priors.
        m: numpy.ndarray of shape (k, d) with centroid means.
        S: numpy.ndarray of shape (k, d, d) with covariance matrices.
        clss: numpy.ndarray of shape (n,) with cluster indices.
        bic: BIC value for the model.
    """
    g = sklearn.mixture.GaussianMixture(n_components=k)
    g.fit(X)
    pi = g.weights_
    m = g.means_
    S = g.covariances_
    clss = g.predict(X)
    bic = g.bic(X)
    return pi, m, S, clss, bic
