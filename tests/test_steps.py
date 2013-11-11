"""
Test the classes in steps.py
"""

__author__ = 'Brandon C. Kelly'

import numpy as np
import math
from scipy import stats, linalg
import steps, priors, proposals
import matplotlib.pyplot as plt
from matplotlib import mlab


class NormalMean(steps.Parameter):
    """
    Normal mean parameter class with fixed variance.
    """

    def __init__(self, data, prior, name="mu", track=True, temperature=1.0):
        self.data_mean = np.median(data)
        self.data_var = np.var(data)
        self.ndata = len(data)
        self.prior = prior
        super(NormalMean, self).__init__(name, track, temperature)

    def set_starting_value(self):
        self.value = \
            stats.cauchy.rvs(loc=self.data_mean, scale=np.sqrt(self.data_var / self.ndata))

    def logdensity(self, value):
        chisqr = self.ndata * (self.data_mean - value) ** 2 / self.variance.value
        logpost = -0.5 * chisqr + self.prior.logdensity(value)
        return logpost  # The log-posterior, up to an additive constant

    def random_posterior(self):
        # Only works for a normal prior object.
        post_var = 1.0 / (1.0 / self.prior.variance + self.ndata / self.variance.value)
        post_mean = post_var * (self.prior.mu / self.prior.variance +
                                self.ndata * self.data_mean / self.variance.value)

        return np.random.normal(post_mean, np.sqrt(post_var))

    def SetVariance(self, variance):
        self.variance = variance


class NormalVariance(steps.Parameter):
    """
    Normal variance parameter with fixed mean.
    """

    def __init__(self, data, prior, name="sigsqr", track=True, temperature=1.0):
        self.data_var = np.var(data)
        self.data_mean = np.mean(data)
        self.data = data
        self.ndata = len(data)
        self.prior = prior
        steps.Parameter.__init__(self, name, track, temperature)

    def set_starting_value(self):
        self.value = self.data_var * (self.ndata - 1) / np.random.chisquare(self.ndata - 1)

    def logdensity(self, value):
        ssqr = self.ndata * self.data_var + self.ndata * (self.data_mean - self.mean.value) ** 2
        loglik = -self.ndata / 2.0 * np.log(value) - 0.5 * ssqr / value
        return loglik + self.prior.logdensity(value)  # log-posterior, up to an additive constant

    def random_posterior(self):
        # Only works for a scaled inverse-chi-square prior object
        resids = self.data - self.mean.value
        ssqr = np.var(resids)
        post_dof = self.ndata + self.prior.dof
        post_ssqr = (self.ndata * ssqr + self.prior.dof * self.prior.ssqr) / post_dof
        return post_ssqr * post_dof / np.random.chisquare(post_dof)

    def SetMean(self, mean):
        self.mean = mean


def test_Parameter():
    """
    Test the Parameter object, using a normal model with unknown mean but known variance.
    """
    # First create some data
    mu0 = np.random.standard_cauchy()
    var0 = np.random.chisquare(10) / 10.0

    ndata = 1000
    data = np.random.normal(mu0, var0, ndata)

    # Now instantiate the prior and parameter objects
    MuPrior = priors.Normal(0.0, 100.0, 1.0)
    Mu = NormalMean(data, MuPrior, "mu", True, 1.0)
    VarPrior = priors.ScaledInvChiSqr(2, 1.0, 1.0)
    SigSqr = NormalVariance(data, VarPrior, "sigsqr", True, 1.0)

    # Add parameter objects to the other parameter object
    Mu.SetVariance(SigSqr)
    SigSqr.SetMean(Mu)

    # First test the mean parameter object
    Mu.set_starting_value()
    SigSqr.set_starting_value()

    # Test LogDensity method
    logpost1 = Mu.logdensity(Mu.value)
    logpost2 = Mu.logdensity(Mu.value + 2.0)
    lograt = logpost1 - logpost2

    var = SigSqr.value

    logpost1 = stats.norm.logpdf(Mu.value, Mu.data_mean, np.sqrt(var / ndata)) + \
               stats.norm.logpdf(Mu.value, Mu.prior.mu, Mu.prior.sigma)
    logpost2 = stats.norm.logpdf(Mu.value + 2.0, Mu.data_mean, np.sqrt(var / ndata)) + \
               stats.norm.logpdf(Mu.value + 2.0, Mu.prior.mu, Mu.prior.sigma)
    lograt0 = logpost1 - logpost2

    assert np.abs(lograt - lograt0) / np.abs(lograt0) < 1e-8

    # Now test RandomPosterior method
    ndraws = 100000
    x = np.empty(ndraws)
    for i in xrange(ndraws):
        x[i] = Mu.random_posterior()

    xgrid = np.linspace(data.mean() - 5.0 * np.std(data) / np.sqrt(ndata),
                        data.mean() + 5.0 * np.std(data) / np.sqrt(ndata))
    post_var = 1.0 / (1.0 / Mu.prior.variance + Mu.ndata / Mu.variance.value)
    post_mean = post_var * (Mu.prior.mu / Mu.prior.variance +
                            Mu.ndata * Mu.data_mean / Mu.variance.value)

    post_pdf = 1.0 / np.sqrt(2.0 * np.pi * post_var) * np.exp(-0.5 * (xgrid - post_mean) ** 2 / post_var)

    plt.subplot(211)
    plt.hist(x, bins=25, normed=True)
    plt.plot(xgrid, post_pdf, 'r', lw=2)
    plt.title("Normal Model: Test of random_posterior method")
    plt.xlabel("Mean")
    plt.ylabel("PDF")

    # Now test the variance parameter object

    logpost1 = SigSqr.logdensity(SigSqr.value)
    logpost2 = SigSqr.logdensity(SigSqr.value + 2.0)
    lograt = logpost1 - logpost2

    mean = Mu.value
    ssqr = SigSqr.data_var + (SigSqr.data_mean - mean) ** 2
    post_dof = SigSqr.ndata + SigSqr.prior.dof
    post_ssqr = (SigSqr.ndata * ssqr + SigSqr.prior.dof * SigSqr.prior.ssqr) / post_dof
    logpost1 = post_dof / 2.0 * math.log(post_ssqr * post_dof / 2.0) - \
               math.lgamma(post_dof / 2.0) - (1.0 + post_dof / 2.0) * \
               math.log(SigSqr.value) - post_dof * post_ssqr / (2.0 * SigSqr.value)
    logpost2 = post_dof / 2.0 * math.log(post_ssqr * post_dof / 2.0) - \
               math.lgamma(post_dof / 2.0) - (1.0 + post_dof / 2.0) * \
               math.log(SigSqr.value + 2.0) - post_dof * post_ssqr / (2.0 * (SigSqr.value + 2.0))

    lograt0 = logpost1 - logpost2

    assert np.abs(lograt - lograt0) / np.abs(lograt0) < 1e-8

    # Now test RandomPosterior method
    for i in xrange(ndraws):
        x[i] = SigSqr.random_posterior()

    var_mean = np.var(data)
    var_sig = np.sqrt(2.0 / ndata) * np.var(data)

    xgrid = np.linspace(var_mean - 5.0 * var_sig, var_mean + 5.0 * var_sig)
    invgam_shape = post_dof / 2.0
    invgam_scale = post_dof / 2.0 * post_ssqr

    post_pdf = stats.invgamma.pdf(xgrid, invgam_shape, scale=invgam_scale)

    plt.subplot(212)
    plt.hist(x, bins=25, normed=True)
    plt.plot(xgrid, post_pdf, 'r', lw=2)
    plt.xlabel("Variance")
    plt.ylabel("PDF")
    plt.show()


def test_MetropStep():
    # First create some data
    mu0 = np.random.standard_cauchy()
    var0 = np.random.chisquare(10) / 10.0

    ndata = 1000
    data = np.random.normal(mu0, var0, ndata)

    # Now instantiate the prior and parameter object
    MuPrior = priors.Normal(0.0, 100.0)
    Mu = NormalMean(data, MuPrior, "mu", True, 1.0)
    VarPrior = priors.ScaledInvChiSqr(2, 1.0)
    SigSqr = NormalVariance(data, VarPrior, "sigsqr", True, 1.0)

    # Add parameter objects to the other parameter object
    Mu.SetVariance(SigSqr)
    SigSqr.SetMean(Mu)

    SigSqr.value = var0  # treat the variance as known for this test

    MuProp = proposals.NormalProposal(np.sqrt(np.var(data) / ndata))
    MuMetroStep = steps.MetroStep(Mu, MuProp, report_iter=10000)

    # Make sure we accept a proposal that is the same as the current value
    assert MuMetroStep.accept(Mu.value, Mu.value)

    ndraws = 20000
    mu_draws = np.empty(ndraws)
    for i in xrange(ndraws):
        MuMetroStep.do_step()
        mu_draws[i] = Mu.value

    # Discard first half as burnin
    mu_draws = mu_draws[ndraws / 2:]

    # Compare the histogram of the draws obtained from the Metropolis algorithm
    # with the expected distribution
    xgrid = np.linspace(data.mean() - 5.0 * np.std(data) / np.sqrt(ndata),
                        data.mean() + 5.0 * np.std(data) / np.sqrt(ndata))
    post_var = 1.0 / (1.0 / Mu.prior.variance + Mu.ndata / Mu.variance.value)
    post_mean = post_var * (Mu.prior.mu / Mu.prior.variance +
                            Mu.ndata * Mu.data_mean / Mu.variance.value)

    post_pdf = 1.0 / np.sqrt(2.0 * np.pi * post_var) * np.exp(-0.5 * (xgrid - post_mean) ** 2 / post_var)

    plt.subplot(111)
    plt.hist(mu_draws, bins=25, normed=True)
    plt.plot(xgrid, post_pdf, 'r', lw=2)
    plt.title("Normal Model: Test of Metropolis step method")
    plt.xlabel("Mean")
    plt.ylabel("PDF")
    plt.show()


class BivariateNormalMean(steps.Parameter):
    """
    Mean of a bivariate Normal distribution.
    """

    def __init__(self, data, covar, prior, name="mu", track=True, temperature=1.0):
        self.data_mean = np.mean(data, axis=0)
        self.data = data
        self.ndata = len(data)
        self.covar = covar
        self.prior = prior
        steps.Parameter.__init__(self, name, track, temperature)

    def set_starting_value(self):
        self.value = stats.cauchy.rvs(loc=self.data_mean, scale=np.sqrt(self.data.var(axis=0) / self.ndata))

    def logdensity(self, value):
        corr = self.covar[0, 1] / np.sqrt(self.covar[0, 0] * self.covar[1, 1])

        zsqr = (self.data[:, 0] - value[0]) ** 2 / self.covar[0, 0] + \
               (self.data[:, 1] - value[1]) ** 2 / self.covar[1, 1] - \
               2.0 * corr * (self.data[:, 0] - value[0]) * (self.data[:, 1] - value[1]) / \
               np.sqrt(self.covar[0, 0] * self.covar[1, 1])

        zsqr /= 2.0 * (1.0 - corr ** 2)
        loglik = -0.5 * np.log(self.covar[0, 0] * self.covar[1, 1] * (1.0 - corr ** 2)) - zsqr
        logpost = loglik.sum() + self.prior.logdensity(value)
        return logpost


def test_AdaptiveMetro():
    # First create some data
    mu0 = np.random.standard_cauchy(size=2)

    zz = np.random.multivariate_normal(np.zeros(2), np.identity(2), 5)
    covar = zz.T.dot(zz) / 5.0

    ndata = 1000
    data = np.random.multivariate_normal(mu0, covar, ndata)

    # Now instantiate the prior and parameter object
    prior = priors.Uninformative()
    NormPar = BivariateNormalMean(data, covar, prior, "mu", True, 1.0)

    # Create proposal object for MHA proposals
    prop_covar = np.identity(2)
    UnitProp = proposals.MultiNormalProposal(prop_covar)

    # Create Robust Adaptive Metropolis step object
    target_rate = 0.4
    maxiter = 100000
    prop_covar = np.cov(data, rowvar=0)
    NormStep = steps.AdaptiveMetro(NormPar, UnitProp, prop_covar, target_rate, maxiter)

    # Make sure we accept a proposal that is the same as the current value
    assert NormStep.accept(NormPar.value, NormPar.value)

    # Generate random draws
    ndraws = 20000
    x = np.empty((ndraws, 2))
    for i in xrange(ndraws):
        NormStep.do_step()
        x[i, :] = NormPar.value

    # Make sure acceptance rate is within 2% of the target rate
    acceptance_rate = NormStep.naccept / float(NormStep.niter)
    print 'Acceptance rate:', acceptance_rate
    assert abs(acceptance_rate - target_rate) / abs(target_rate) < 0.02

    # Discard first half as burnin
    x = x[ndraws / 2:, :]
    post_covar = covar / ndata

    # Compare covariance matrix of proposals with posterior covariance

    covar_n = NormStep._cholesky_factor.T.dot(NormStep._cholesky_factor)
    eigenval_n, eigenvect_n = linalg.eig(covar_n)
    eigenval_n = np.diagflat(eigenval_n)
    covroot_n = np.dot(eigenvect_n.T, np.sqrt(eigenval_n).dot(eigenvect_n))  # Matrix square-root of proposal covariance
    covroot_n = covroot_n.real

    eigenval, eigenvect = linalg.eig(post_covar)
    eigenval_inv = np.diagflat(1.0 / eigenval)
    covroot_inv = np.dot(eigenvect.T, np.sqrt(eigenval_inv).dot(eigenvect))
    covroot_inv = covroot_inv.real

    evals, evects = linalg.eig(covroot_n.dot(covroot_inv))
    # Compute the 'suboptimality factor'. This should be unity if the two matrices are proportional.
    subopt_factor = evals.size * np.sum(1.0 / evals ** 2) / (np.sum(1.0 / evals)) ** 2

    assert subopt_factor < 1.01  # Test for proportionality of proposal covariance and posterior covariance

    # Now make nice plots comparing sampled values with the true posterior

    data_mean = data.mean(axis=0)

    xgrid = np.linspace(data_mean[0] - 4.0 * np.sqrt(post_covar[0, 0]),
                        data_mean[0] + 4.0 * np.sqrt(post_covar[0, 0]), 100)
    ygrid = np.linspace(data_mean[1] - 4.0 * np.sqrt(post_covar[1, 1]),
                        data_mean[1] + 4.0 * np.sqrt(post_covar[1, 1]), 100)

    X, Y = np.meshgrid(xgrid, ygrid)

    true_pdf = mlab.bivariate_normal(X, Y, math.sqrt(post_covar[0, 0]), math.sqrt(post_covar[1, 1]),
                                     data_mean[0], data_mean[1], post_covar[0, 1])

    plt.subplot(211)
    post_pdf = 1.0 / np.sqrt(2.0 * np.pi * post_covar[0, 0]) * \
        np.exp(-0.5 * (xgrid - data_mean[0]) ** 2 / post_covar[0, 0])

    plt.hist(x[:, 0], bins=25, normed=True)
    plt.plot(xgrid, post_pdf, 'r', lw=2)
    plt.title("Bivariate Normal Model: Test of Robust Adaptive Metropolis step method")
    plt.xlabel("Mean 1")
    plt.ylabel("PDF")

    plt.subplot(212)
    post_pdf = 1.0 / np.sqrt(2.0 * np.pi * post_covar[1, 1]) * \
        np.exp(-0.5 * (ygrid - data_mean[1]) ** 2 / post_covar[1, 1])

    plt.hist(x[:, 1], bins=25, normed=True)
    plt.plot(ygrid, post_pdf, 'r', lw=2)
    plt.xlabel("Mean 2")
    plt.ylabel("PDF")
    plt.show()

    plt.figure()
    plt.plot(x[:, 0], x[:, 1], '.', ms=2)
    plt.contour(X, Y, true_pdf, linewidths=5)
    plt.title('Test of Robust Adaptive Metropolis Algorithms: Bivariate Normal Model')
    plt.ylabel('Mean 2')
    plt.xlabel('Mean 1')
    plt.show()


if __name__ == "__main__":
    test_Parameter()
    test_MetropStep()
    test_AdaptiveMetro()
    print "All tests passed"
