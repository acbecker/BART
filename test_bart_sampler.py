__author__ = 'brandonkelly'

import unittest
import numpy as np
from scipy import stats, integrate
from tree import *
import matplotlib.pyplot as plt
from test_tree_parameters import build_test_data


def build_friedman_data(nsamples, nfeatures):
    # build Friedman's five dimensional test function
    X = np.random.uniform(0.0, 1.0, (nsamples, nfeatures))
    ymean = 10.0 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20.0 * (X[:, 2] - 0.5) ** 2 + 10.0 * X[:, 3] + 4.0 * X[:, 4]
    return X, ymean


class SamplerTestCase(unittest.TestCase):

    def setUp(self):
        # first generate a data set
        nsamples = 1000
        nfeatures = 10
        self.alpha = 0.95
        self.beta = 2.0
        self.sigsqr0 = 0.7 ** 2
        self.true_sigsqr = self.sigsqr0

        X, ymean = build_friedman_data(nsamples, nfeatures)
        self.X = X
        self.true_ymean = ymean
        self.y = ymean + np.sqrt(self.sigsqr0) * np.random.standard_normal(nsamples)
        self.mtrees = 200

        # build MCMC sampler
        self.model = BartModel(X, self.y.copy(), m=self.mtrees, alpha=0.95, beta=2.0)

    def tearDown(self):
        pass

    def test_buildsampler(self):
        pass

    def test_startsampler(self):
        pass

    def test_savevalues(self):
        pass

    def test_predict(self):
        pass

    def test_sampler(self):
        pass


if __name__ == "__main__":
    unittest.main()
