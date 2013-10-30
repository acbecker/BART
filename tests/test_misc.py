"""
Test the routines in misc.py
"""

__author__ = 'Brandon C. Kelly'

import numpy as np
import misc
from scipy.linalg import cholesky  # Returns an upper-triangular matrix, which is what we want


def test_CholUpdateR1():
    """
    Compare the rank-1 Cholesky update from misc.CholUpdateR1 with the value from
    numpy.linalg.cholesky
    """

    # Make a positive-definite symmetric array
    corr = np.array([[1.0, 0.3, -0.5], [0.3, 1.0, 0.54], [-0.5, 0.54, 1.0]])
    sigma = np.array([[2.3, 0.0, 0.0], [0.0, 0.45, 0.0], [0.0, 0.0, 13.4]])
    covar = sigma.dot(corr.dot(sigma))

    # Compute the cholesky decomposition: covar = L * L.transpose()
    L = cholesky(covar)

    # Construct the update vector
    z = np.random.standard_normal(3)
    z /= np.linalg.norm(z)
    v = np.sqrt(0.5) * L.transpose().dot(z)

    covar_update = covar + np.outer(v, v)  # Updated covariance matrix
    covar_downdate = covar - np.outer(v, v)  # Downdated covariance matrix

    # Get the cholesky decompositions of the updated and downdated matrices. Do the slow way first.
    Lup0 = cholesky(covar_update)
    Ldown0 = cholesky(covar_downdate)

    # Now get the rank-1 updated and downdated cholesky factors, but do the fast way.
    Lup = L.copy()
    vup = v.copy()
    misc.cholupdate_r1(Lup, vup, False)  # The update/downdate algorithm assumes an upper triangular matrix

    Ldown = L.copy()
    vdown = v.copy()
    misc.cholupdate_r1(Ldown, v, True)

    low_triangle = Lup0 != 0
    frac_diff = np.abs(Lup0[low_triangle] - Lup[low_triangle]) / np.abs(Lup0[low_triangle])
    assert frac_diff.max() < 1e-8

    frac_diff = np.abs(Ldown0[low_triangle] - Ldown[low_triangle]) / np.abs(Ldown0[low_triangle])
    assert frac_diff.max() < 1e-8
    print "Testing of CholUpdateR1 passed."

if __name__ == "__main__":
    test_CholUpdateR1()