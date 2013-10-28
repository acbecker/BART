"""
Miscellaneous routines needed by yamcmc.
"""
__author__ = 'Brandon C. Kelly'

import numpy as np


def CholUpdateR1(L, v, downdate=False):
    """
    Perform the rank-1 Cholesky update (or downdate). Suppose we have the Cholesky decomposition for a matrix, A.
    The rank-1 update computes the Cholesky decomposition of a new matrix B, where B = A + v * v.transpose(). The
    downdate corresponds to B = A - v * v.transpose(). The input array will be overwritten.

    :param L (array-like): A lower-triangular matrix obtained via a Cholesky decomposition. On output, this will
                           be the Cholesky decomposition of the matrix L * L.T +/- v * v.T.
    :param v (array-like): A vector describing the update or downdate.
    :param downdate: A boolean variable describing whether to perform the downdate (downdate=True).
    """

    assert L.shape[0] == L.shape[1]

    sign = 1.0
    if downdate:
        sign = -1.0

    for k in xrange(L.shape[0]):
        r = np.sqrt(L[k, k] * L[k, k] + sign * v[k] * v[k])
        c = r / L[k, k]
        s = v[k] / L[k, k]
        L[k, k] = r
        if k < L.shape[0] - 1:
            L[k, k + 1:] = (L[k, k + 1:] + sign * s * v[k + 1:]) / c
            v[k + 1:] = c * v[k + 1:] - s * L[k, k + 1:]




