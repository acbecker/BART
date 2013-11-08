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
        self.true_sigsqr = 0.7 ** 2

        ngrow_list = [4, 7]
        self.mtrees = 2
        forest, mu_list = build_test_data(self.X, self.true_sigsqr, ngrow_list, self.mtrees)

        self.y = forest[0].tree.y
        # Rescale y to lie between -0.5 and 0.5
        self.ymin = self.y.min()
        self.y -= self.ymin  # minimum = 0
        self.ymax = self.y.max()
        self.true_sigsqr /= self.ymax ** 2
        self.y /= self.ymax  # maximum = 1
        self.y -= 0.5  # range is -0.5 to 0.5

        self.mu_list = []
        self.forest = []
        idx = 1
        for tree, mu in zip(forest, mu_list):
            # rescale mu values since we rescaled the y values
            mu -= self.ymin
            mu /= self.ymax
            mu -= 0.5
            mean_param = BartMeanParameter("mu " + str(idx), self.mtrees)
            mean_param.tree = tree  # this tree configuration
            mean_param.value = mu
            self.mu_list.append(mean_param)

            # Tree parameter object, note that this is different from a BaseTree object
            tree_param = BartTreeParameter('tree ' + str(idx), self.X, self.y, self.mtrees, self.alpha, self.beta,
                                          mean_param.mubar, mean_param.prior_var)
            tree_param.value = tree

            # update moments of y-values in each terminal node since we transformed the data
            for leaf in tree_param.value.terminalNodes:
                tree_param.value.filter(leaf)

            self.forest.append(tree_param)

        self.mu.sigsqr = BartVariance(self.X, self.y)
        self.mu.sigsqr.value = self.true_sigsqr
        self.tree.sigsqr = self.mu.sigsqr

        self.bart_step = BartStep(self.y, self.forest, self.mu_list, report_iter=5000)
        self.mu.sigsqr.bart_step = self.bart_step

    def tearDown(self):
        del self.X
        del self.y
        del self.mu_list
        del self.forest
        del self.bart_step

    def test_node_mu(self):
        mu_map = BartStep.node_mu(self.tree.value, self.mu)
        n_idx = 0
        for leaf in self.tree.value.terminalNodes:
            in_node = self.tree.value.filter(leaf)[1]
            for i in xrange(sum(in_node)):
                self.assertAlmostEquals(mu_map[in_node][i], self.mu.value[n_idx])
            n_idx += 1

    def test_do_step(self):
        # Tests:
        # 1) Make sure that the y-values are updated, i.e., tree.y != resids
        # 2) Make sure that the true mu(x) values are contained within the 95% credibility interval 95% of the time
        # 3) Make sure that the standard deviation in the residuals agrees with the true value of the variance parameter
        # 4) Make sure that the number of internal and external nodes agree with the true values at the 95% level.
        #
        # Tests 2-4 are carried out using an MCMC sampler that keeps the Variance parameter fixed.
        pass