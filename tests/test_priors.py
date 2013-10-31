"""
Test the classes in priors.py.
"""

__author__ = 'Brandon C. Kelly'

import numpy as np
import math
from scipy import stats
import priors


def test_Normal():
    """
    Test the normal prior object.
    """
    variance = np.random.chisquare(10) / 10.0
    mu = np.random.standard_cauchy()

    NormalPrior = priors.Normal(mu, variance, 1.0)

    ndraws = 100000

    x = np.empty(ndraws)
    for i in xrange(ndraws):
        x[i] = NormalPrior.draw()

    x0 = np.random.normal(mu, np.sqrt(variance), ndraws)

    # Do K-S test to verify that the distribution of proposals from the NormalProposal object
    # have the same distribution as values drawn from numpy.random.normal.
    ks_statistic, significance = stats.ks_2samp(x, x0)
    assert significance > 1e-3

    # Verify that the value of the log(PDF) returned by the prior object is correct
    x = np.random.standard_cauchy()
    logprior = -0.5 * math.log(2.0 * math.pi * variance) - 0.5 * (x - mu) ** 2 / variance
    pdf_diff = abs(logprior - NormalPrior.logdensity(x)) / abs(logprior)
    assert pdf_diff < 1e-8

    print 'Test for Normal Prior object passed.'


def test_ScaledInvChiSqr():
    """
    Test the scaled-inverse-chi-square prior object.
    """
    dof = np.random.random_integers(2, 20)
    ssqr = np.random.chisquare(10) / 10.0

    InvChiSqr = priors.ScaledInvChiSqr(dof, ssqr, 1.0)

    ndraws = 100000

    x = np.empty(ndraws)
    for i in xrange(ndraws):
        x[i] = InvChiSqr.draw()

    x0 = ssqr * dof / np.random.chisquare(dof, ndraws)

    # Do K-S test to verify the distribution generated by the scaled inverse-chi-square
    # prior object.

    ks_statistic, significance = stats.ks_2samp(x, x0)
    assert significance > 1e-3

    x = np.random.chisquare(3)
    logprior = dof / 2.0 * math.log(ssqr * dof / 2.0) - math.lgamma(dof / 2.0) - \
        (1.0 + dof / 2.0) * math.log(x) - dof * ssqr / (2.0 * x)

    pdf_diff = abs(logprior - InvChiSqr.logdensity(x)) / abs(logprior)
    assert pdf_diff < 1e-8

    print 'Test for Scaled inverse chi-square Prior object passed.'


if __name__ == "__main__":
    test_Normal()
    test_ScaledInvChiSqr()