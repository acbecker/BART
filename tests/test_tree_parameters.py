__author__ = 'brandonkelly'

import unittest
import numpy as np
from scipy import stats, integrate
from tree import *
import matplotlib.pyplot as plt


# generate test data from an ensemble of trees
def build_test_data(X, sigsqr, ngrow=5, mtrees=1):
    if np.isscalar(ngrow):
        ngrow = [ngrow] * mtrees

    ytemp = np.random.standard_normal(X.shape[0])
    forest = []
    mu_list = []
    mu_map = np.zeros(X.shape[0])

    for m in xrange(mtrees):
        tree = BaseTree(X, ytemp, min_samples_leaf=1)
        for i in xrange(ngrow[m]):
            tree.grow()

        mu = np.random.normal(2.0, 1.3, len(tree.terminalNodes))

        forest.append(tree)
        mu_list.append(mu)

        n_idx = 0
        this_mu_map = np.zeros(mu_map.size)
        for leaf in tree.terminalNodes:
            y_in_node = tree.filter(leaf)[1]
            this_mu_map[y_in_node] = mu[n_idx]
            n_idx += 1
        mu_map += this_mu_map

        n_idx += 1

    y = mu_map + np.sqrt(sigsqr) * np.random.standard_normal(X.shape[0])

    for tree in forest:
        tree.y = y
        # rerun filter to update the y-means and variances in each terminal node
        for leaf in tree.terminalNodes:
            x_in_node, y_in_node = tree.filter(leaf)

    if mtrees == 1:
        forest = forest[0]
        mu_list = mu_list[0]

    return forest, mu_list


class SimpleBartStep(object):
    def __init__(self):
        self.nsamples = 500
        self.resids = np.random.standard_normal(self.nsamples)


class VarianceTestCase(unittest.TestCase):
    def setUp(self):
        nsamples = 2000
        nfeatures = 2
        self.X = np.random.standard_cauchy((nsamples, nfeatures))
        self.true_sigsqr = 0.7 ** 2
        tree, mu = build_test_data(self.X, self.true_sigsqr)
        self.y = tree.y
        self.sigsqr = BartVariance(self.X, self.y)
        self.sigsqr.bart_step = SimpleBartStep()
        # get residuals
        mu_map = np.zeros(nsamples)
        n_idx = 0
        for node in tree.terminalNodes:
            y_in_node = tree.filter(node)[1]
            mu_map[y_in_node] = mu[n_idx]
            n_idx += 1

        self.sigsqr.bart_step.resids = self.y - mu_map

    def tearDown(self):
        del self.X
        del self.y
        del self.true_sigsqr
        del self.sigsqr

    def test_prior(self):
        nsamples = self.X.shape[0]
        y = 2.0 + self.X[:, 0] + np.sqrt(self.true_sigsqr) * np.random.standard_normal(nsamples)
        SigSqr = BartVariance(self.X, y)
        SigSqr.bart_step = SimpleBartStep()
        nu = 3.0  # Degrees of freedom for error variance prior; should always be > 3
        q = 0.90  # The quantile of the prior that the sigma2 estimate is placed at
        qchi = stats.chi2.interval(q, nu)[1]
        # scale parameter for error variance scaled inverse-chi-square prior
        lamb = self.true_sigsqr * qchi / nu

        # is the prior scale parameter within 5% of the expected value?
        frac_diff = np.abs(SigSqr.lamb - lamb) / lamb

        prior_msg = "Fractional difference in prior scale parameter for variance parameter is greater than 10%"
        self.assertLess(frac_diff, 0.10, msg=prior_msg)

    def test_random_posterior(self):
        ndraws = 100000
        ssqr_draws = np.empty(ndraws)
        for i in xrange(ndraws):
            ssqr_draws[i] = self.sigsqr.random_posterior()

        nu = self.sigsqr.nu
        prior_ssqr = self.sigsqr.lamb

        post_dof = nu + len(self.y)
        post_ssqr = (nu * prior_ssqr + self.y.size * np.var(self.sigsqr.bart_step.resids)) / post_dof

        igam_shape = post_dof / 2.0
        igam_scale = post_dof * post_ssqr / 2.0
        igamma = stats.distributions.invgamma(igam_shape, scale=igam_scale)

        # test draws from conditional posterior by comparing 1st and 2nd moments to true values
        true_mean = igamma.moment(1)
        frac_diff = np.abs(true_mean - ssqr_draws.mean()) / true_mean
        rpmsg = "Fractional difference in mean from BartVariance.random_posterior() is greater than 2%"
        self.assertLess(frac_diff, 0.02, msg=rpmsg)

        true_ssqr = igamma.moment(2)
        frac_diff = np.abs(true_ssqr - (ssqr_draws.var() + ssqr_draws.mean() ** 2)) / true_ssqr
        rpmsg = "Fractional difference in 2nd moment from BartVariance.random_posterior() is greater than 2%"
        self.assertLess(frac_diff, 0.02, msg=rpmsg)

        # make sure gibbs sampler constrains the correct value
        ssqr_low = np.percentile(ssqr_draws, 1.0)
        ssqr_high = np.percentile(ssqr_draws, 99.0)
        rpmsg = "Value of Variance parameter returned by Gibbs sampler is outside of 99% credibility interval."
        self.assertGreater(self.true_sigsqr, ssqr_low, msg=rpmsg)
        self.assertLess(self.true_sigsqr, ssqr_high, msg=rpmsg)


class MuTestCase(unittest.TestCase):
    def setUp(self):
        nsamples = 2000
        nfeatures = 2
        self.alpha = 0.95
        self.beta = 2.0
        self.X = np.random.standard_cauchy((nsamples, nfeatures))
        self.true_sigsqr = 0.7 ** 2
        tree, mu = build_test_data(self.X, self.true_sigsqr)
        self.true_mu = mu
        self.tree = tree
        self.y = tree.y
        self.mtrees = 1  # single tree model
        self.mu = BartMeanParameter("mu", 1)
        # Rescale y to lie between -0.5 and 0.5
        self.true_mu -= self.y.min()
        self.y -= self.y.min()  # minimum = 0
        self.true_mu /= self.y.max()
        self.true_sigsqr /= self.y.max() ** 2
        self.y /= self.y.max()  # maximum = 1
        self.true_mu -= 0.5
        self.y -= 0.5  # range is -0.5 to 0.5
        tree.y = self.y

        # update moments of y-values in each terminal node since we transformed the data
        for leaf in self.tree.terminalNodes:
            self.tree.filter(leaf)

        self.mu.sigsqr = BartVariance(self.X, self.y)
        self.mu.sigsqr.bart_step = SimpleBartStep()
        self.mu.sigsqr.value = self.true_sigsqr

        self.tree_param = BartTreeParameter('tree', self.X, self.y, self.mtrees, prior_mu=self.mu.mubar,
                                            prior_var=self.mu.prior_var)
        self.tree_param.value = tree
        self.mu.tree = self.tree_param
        self.mu.set_starting_value(self.tree)

    def tearDown(self):
        del self.X
        del self.y
        del self.mu
        del self.tree

    def test_random_posterior(self):
        # first get values of mu drawn from their conditional posterior
        ndraws = 100000
        nleaves = len(self.mu.value)
        mu_draws = np.empty((ndraws, nleaves))
        for i in xrange(ndraws):
            mu_draws[i, :] = self.mu.random_posterior()

        l_idx = 0
        for leaf in self.mu.tree.value.terminalNodes:
            ny = leaf.npts
            ybar = leaf.ybar
            post_var = 1.0 / (1.0 / self.mu.prior_var + ny / self.mu.sigsqr.value)
            post_mean = post_var * (self.mu.mubar / self.mu.prior_var + ny * ybar / self.mu.sigsqr.value)

            # test draws from conditional posterior by comparing 1st and 2nd moments to true values
            zscore = np.abs((post_mean - mu_draws[:, l_idx].mean())) / np.sqrt(post_var / ndraws)
            rpmsg = "Sample mean from BartMeanParameter.random_posterior() differs by more than 3-sigma."
            self.assertLess(zscore, 3.0, msg=rpmsg)

            frac_diff = np.abs(np.sqrt(post_var) - mu_draws[:, l_idx].std()) / np.sqrt(post_var)
            rpmsg = "Fractional difference in standard deviation from BartMeanParameter.random_posterior() is greater" \
                + " than 2%"
            self.assertLess(frac_diff, 0.02, msg=rpmsg)

            # make sure gibbs sampler constrains the correct value
            mu_low = np.percentile(mu_draws[:, l_idx], 1.0)
            mu_high = np.percentile(mu_draws[:, l_idx], 99.0)
            rpmsg = "Value of Terminal Node output parameter returned by Gibbs sampler is outside of 99% credibility" \
                + " interval.\n Violated: " + str(mu_low) + ' < ' + str(self.true_mu[l_idx]) + ' < ' + str(mu_high)
            self.assertGreater(self.true_mu[l_idx], mu_low, msg=rpmsg)
            self.assertLess(self.true_mu[l_idx], mu_high, msg=rpmsg)

            l_idx += 1


class TreeTestCase(unittest.TestCase):
    def setUp(self):
        nsamples = 20  # ensure a small number of data points in each node for we don't get underflows below
        nfeatures = 2
        self.alpha = 0.95
        self.beta = 2.0
        self.X = np.random.standard_cauchy((nsamples, nfeatures))
        self.true_sigsqr = 0.7 ** 2
        tree, mu = build_test_data(self.X, self.true_sigsqr)
        self.true_mu = mu
        self.y = tree.y
        self.mtrees = 1  # single tree model
        self.mu = BartMeanParameter("mu", 1)
        # Rescale y to lie between -0.5 and 0.5
        self.true_mu -= self.y.min()
        self.y -= self.y.min()  # minimum = 0
        self.true_mu /= self.y.max()
        self.true_sigsqr /= self.y.max() ** 2
        self.y /= self.y.max()  # maximum = 1
        self.true_mu -= 0.5
        self.y -= 0.5  # range is -0.5 to 0.5
        tree.y = self.y
        # update moments of y-values in each terminal node since we transformed the data
        for leaf in tree.terminalNodes:
            tree.filter(leaf)

        self.mu.sigsqr = BartVariance(self.X, self.y)
        self.mu.sigsqr.bart_step = SimpleBartStep()
        self.mu.sigsqr.value = self.true_sigsqr

        self.tree = BartTreeParameter('tree', self.X, self.y, self.mtrees, prior_mu=self.mu.mubar,
            prior_var=self.mu.prior_var)
        self.tree.value = tree
        self.tree.sigsqr = self.mu.sigsqr
        self.mu.tree = self.tree

    def tearDown(self):
        del self.X
        del self.y
        del self.mu
        del self.tree

    def test_logdens(self):
        loglik_direct = 0.0
        mugrid = np.linspace(-5.0, 5.0, 1001)
        for leaf in self.tree.value.terminalNodes:
            # compute log-likelihood numerically
            in_node = self.tree.value.filter(leaf)[1]  # find the data that end up in this node
            lik = 1.0
            for i in xrange(leaf.npts):
                lik *= stats.distributions.norm(self.y[in_node][i], np.sqrt(self.true_sigsqr)).pdf(mugrid)
            # add in prior contribution
            lik *= stats.distributions.norm(self.mu.mubar, np.sqrt(self.mu.prior_var)).pdf(mugrid)
            loglik_direct += np.log(integrate.simps(lik, mugrid))

        # make sure numerical and analytical calculation agree
        loglik = self.tree.logdensity(self.tree.value)
        frac_diff = np.abs(loglik_direct - loglik) / np.abs(loglik)
        tree_msg = "Fractional difference between numerical calculation of tree loglik and analytical calculation is" \
            + " greater than 1%"
        self.assertLess(frac_diff, 0.01, msg=tree_msg)

if __name__ == "__main__":
    unittest.main()
