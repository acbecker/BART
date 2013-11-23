__author__ = 'brandonkelly'

import unittest
import numpy as np
from scipy import stats, integrate
from tree import *
import matplotlib.pyplot as plt
from test_tree_parameters import build_test_data, SimpleBartStep


class StepTestCase(unittest.TestCase):
    def setUp(self):
        nsamples = 1000
        nfeatures = 4
        self.alpha = 0.95
        self.beta = 2.0
        self.X = np.random.standard_cauchy((nsamples, nfeatures))
        self.sigsqr0 = 0.7 ** 2
        self.true_sigsqr = self.sigsqr0

        ngrow_list = [4, 7]
        self.mtrees = 2
        forest, mu_list = build_test_data(self.X, self.sigsqr0, ngrow_list, self.mtrees)

        # forest = [forest]
        # mu_list = [mu_list]

        self.y = forest[0].y
        self.y0 = forest[0].y.copy()
        # Rescale y to lie between -0.5 and 0.5
        self.ymin = self.y.min()
        self.ymax = self.y.max()
        self.y = (self.y - self.ymin) / (self.ymax - self.ymin) - 0.5
        self.true_sigsqr /= (self.ymax - self.ymin) ** 2
        for tree in forest:
            tree.y = self.y  # make sure tree objects have the transformed data

        self.sigsqr = BartVariance(self.X, self.y)
        self.sigsqr.value = self.true_sigsqr

        self.mu_list = []
        self.forest = []
        self.mu_map = np.zeros(len(self.y))
        self.nleaves = np.zeros(self.mtrees)
        self.nbranches = np.zeros(self.mtrees)
        id = 1
        for tree, mu in zip(forest, mu_list):
            self.nleaves[id-1] = len(tree.terminalNodes)
            self.nbranches[id-1] = len(tree.internalNodes)

            # rescale mu values since we rescaled the y values
            mu = mu / (self.ymax - self.ymin) - 1.0 / self.mtrees * (self.ymin / (self.ymax - self.ymin) + 0.5)
            mean_param = BartMeanParameter("mu " + str(id), self.mtrees)
            mean_param.value = mu
            mean_param.sigsqr = self.sigsqr

            # Tree parameter object, note that this is different from a BaseTree object
            tree_param = BartTreeParameter('tree ' + str(id), self.X, self.y, self.mtrees, self.alpha, self.beta,
                                          mean_param.mubar, mean_param.prior_var)
            tree_param.value = tree
            mean_param.treeparam = tree_param  # this tree parameter, mu needs to know about it for the Gibbs sampler
            tree_param.sigsqr = self.sigsqr

            # update moments of y-values in each terminal node since we transformed the data
            for leaf in tree_param.value.terminalNodes:
                tree_param.value.filter(leaf)

            self.mu_list.append(mean_param)
            self.forest.append(tree_param)

            self.mu_map += BartStep.node_mu(tree, mean_param)

            id += 1

        self.bart_step = BartStep(self.y, self.forest, self.mu_list, report_iter=5000)
        self.sigsqr.bart_step = self.bart_step

    def tearDown(self):
        del self.X
        del self.y
        del self.mu_list
        del self.forest
        del self.bart_step

    def test_node_mu(self):
        for tree, mu in zip(self.forest, self.mu_list):
            mu_map = BartStep.node_mu(tree.value, mu)
            n_idx = 0
            for leaf in tree.value.terminalNodes:
                in_node = tree.value.filter(leaf)[1]
                for i in xrange(sum(in_node)):
                    self.assertAlmostEquals(mu_map[in_node][i], mu.value[n_idx])
                n_idx += 1

    def test_do_step(self):
        # first make sure data is constructed correctly as a sanity check
        resids = self.mu_map - self.y
        zscore = np.abs(np.mean(resids)) / (np.std(resids) / np.sqrt(resids.size))
        self.assertLess(zscore, 3.0)
        frac_diff = np.abs(resids.std() - np.sqrt(self.true_sigsqr)) / np.sqrt(self.true_sigsqr)
        self.assertLess(frac_diff, 0.05)

        # make sure that when BartStep does y -> resids, that BartMeanParameter knows about the updated node values
        self.bart_step.trees[0].value.y = resids
        n_idx = 0
        for leaf in self.bart_step.trees[0].value.terminalNodes:
            ybar_old = leaf.ybar
            in_node = self.bart_step.trees[0].value.filter(leaf)
            mu_leaf = self.bart_step.mus[0].treeparam.value.terminalNodes[n_idx]
            self.assertAlmostEqual(leaf.ybar, mu_leaf.ybar)
            self.assertNotAlmostEqual(leaf.ybar, ybar_old)
            n_idx += 1

    def test_step_mcmc(self):
        # Tests:
        # 1) Make sure that the y-values are updated, i.e., tree.y != resids
        # 2) Make sure that the true mu(x) values are contained within the 95% credibility interval 95% of the time
        # 3) Make sure that the number of internal and external nodes agree with the true values at the 95% level.
        #
        # The tests are carried out using an MCMC sampler that keeps the Variance parameter fixed.
        burnin = 2000
        niter = 10000

        msg = "Stored y-values in each tree not equal original y-values, BartStep may have changed these internally."
        for i in xrange(burnin):
            self.bart_step.do_step()
            for tree in self.forest:
                self.assertTrue(np.all(tree.y == self.y), msg=msg)

        mu_map = np.zeros((self.y.size, niter))
        nleaves = np.zeros((niter, self.mtrees))
        nbranches = np.zeros((niter, self.mtrees))
        rsigma = np.zeros(niter)

        print 'Running MCMC sampler...'
        for i in xrange(niter):
            self.bart_step.do_step()

            # save MCMC draws
            m = 0
            ypredict = 0.0
            for tree, mu in zip(self.forest, self.mu_list):
                mu_map[:, i] += self.bart_step.node_mu(tree.value, mu)
                ypredict += mu_map[:, i]
                nleaves[i, m] = len(tree.value.terminalNodes)
                nbranches[i, m] = len(tree.value.internalNodes)
                m += 1

            # transform predicted y back to original scale
            ypredict = self.ymin + (self.ymax - self.ymin) * (ypredict + 0.5)
            rsigma[i] = np.std(ypredict - self.y0)

        # make sure we recover the true tree configuration
        for m in xrange(self.mtrees):
            ntrue = np.sum(nbranches[nleaves[:, m] == self.nleaves[m], m] == self.nbranches[m])
            ntrue_fraction = ntrue / float(niter)
            self.assertGreater(ntrue_fraction, 0.05)

        # make sure we recover the correct values of mu(x)
        mu_map_hi = np.percentile(mu_map, 97.5, axis=1)
        mu_map_low = np.percentile(mu_map, 2.5, axis=1)
        out = np.logical_or(self.mu_map > mu_map_hi, self.mu_map < mu_map_low)
        nout = np.sum(out)  # number outside of 95% probability region
        # compare number that fell outside of 95% probability region with expectation from binomial distribution
        signif = 1.0 - stats.distributions.binom(self.y.size, 0.05).cdf(nout)
        print nout
        msg = "Probability of number of mu(x) values outside of 95% probability range is < 1%."
        self.assertGreater(signif, 0.01, msg=msg)


if __name__ == "__main__":
    unittest.main()
