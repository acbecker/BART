__author__ = 'brandonkelly'

import unittest
import numpy as np
from scipy import stats, integrate
from tree import *
import matplotlib.pyplot as plt
from test_tree_parameters import build_test_data, SimpleBartStep


class ProposalTestCase(unittest.TestCase):
    def setUp(self):
        nsamples = 1000
        nfeatures = 4
        self.alpha = 0.95
        self.beta = 2.0
        self.X = np.random.standard_cauchy((nsamples, nfeatures))
        self.true_sigsqr = 0.7 ** 2
        tree, mu = build_test_data(self.X, self.true_sigsqr)
        self.true_mu = mu
        self.y = tree.y
        self.mtrees = 1  # single tree model
        self.mu = BartMeanParameter("mu", 1)
        self.mu.tree = tree
        # Rescale y to lie between -0.5 and 0.5
        self.true_mu -= self.y.min()
        self.y -= self.y.min()  # minimum = 0
        self.true_mu /= self.y.max()
        self.true_sigsqr /= self.y.max() ** 2
        self.y /= self.y.max()  # maximum = 1
        self.true_mu -= 0.5
        self.y -= 0.5  # range is -0.5 to 0.5

        # Tree parameter object, note that this is different from a BaseTree object
        self.tree = BartTreeParameter('tree', self.X, self.y, self.mtrees, self.alpha, self.beta,
                                      self.mu.mubar, self.mu.prior_var)
        self.tree.value = tree

        # update moments of y-values in each terminal node since we transformed the data
        for leaf in self.tree.value.terminalNodes:
            self.tree.value.filter(leaf)

        self.mu.sigsqr = BartVariance(self.X, self.y)
        self.mu.sigsqr.bart_step = SimpleBartStep()
        self.mu.sigsqr.value = self.true_sigsqr
        self.tree.sigsqr = self.mu.sigsqr

        self.tree_proposal = BartProposal()

    def tearDown(self):
        del self.X
        del self.y
        del self.mu
        del self.tree

    def test_node_mu(self):
        pass

    def test_do_step(self):
        pass