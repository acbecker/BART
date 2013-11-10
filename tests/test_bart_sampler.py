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
        self.true_sigsqr = 0.7 ** 2

        X, ymean = build_friedman_data(nsamples, nfeatures)
        self.X = X
        self.true_ymean = ymean
        self.y = ymean + np.sqrt(self.true_sigsqr) * np.random.standard_normal(nsamples)
        self.mtrees = 200

        # build MCMC sampler
        self.model = BartModel(X, self.y.copy(), m=self.mtrees, alpha=self.alpha, beta=self.beta)

    def tearDown(self):
        del self.X
        del self.y
        del self.model

    def test_buildsampler(self):
        self.assertAlmostEqual(self.model.y.min(), -0.5)
        self.assertAlmostEqual(self.model.y.max(), 0.5)
        self.assertEqual(len(self.model.mus), self.mtrees)
        self.assertEqual(len(self.model.trees), self.mtrees)
        self.assertEqual(len(self.model._steps), 2)

        # make sure parameters are connected
        self.assertTrue(self.model.sigsqr.bart_step == self.model._steps[1])
        for tree, mu in zip(self.model.trees, self.model.mus):
            self.assertTrue(tree.sigsqr == self.model.sigsqr)
            self.assertTrue(mu.sigsqr == self.model.sigsqr)
            self.assertTrue(mu.tree == tree)

    def test_startsampler(self):
        self.model.start()
        # first make sure parameters values are initialized
        self.assertGreater(self.model.sigsqr.value, 0.0)
        for tree in self.model.trees:
            self.assertTrue(tree.value != 0)
        for mu in self.model.mus:
            self.assertTrue(np.any(mu.value != 0))

        # now make sure that the dictionary of mcmc samples is initialized
        self.assertTrue(self.model.sigsqr.name in self.model.mcmc_samples.samples)
        self.assertEqual(len(self.model.mcmc_samples.samples[self.model.sigsqr.name]), 0)
        for tree in self.model.trees:
            self.assertTrue(tree.name in self.model.mcmc_samples.samples)
            self.assertEqual(len(self.model.mcmc_samples.samples[tree.name]), 0)
        for mu in self.model.mus:
            self.assertTrue(mu.name in self.model.mcmc_samples.samples)
            self.assertEqual(len(self.model.mcmc_samples.samples[mu.name]), 0)

    def test_savevalues(self):
        self.model.start()
        self.model.save_values()
        mcmc_samples = self.model.mcmc_samples

        # make sure saved the values in the dictionary of MCMC samples
        self.assertEqual(len(mcmc_samples.samples[self.model.sigsqr.name]), 1)
        self.assertTrue(mcmc_samples.samples[self.model.sigsqr.name][0] == self.model.sigsqr.value)
        for tree in self.model.trees:
            self.assertEqual(len(mcmc_samples.samples[tree.name]), 1)
            self.assertTrue(mcmc_samples.samples[tree.name][0] == tree.value)
        for mu in self.model.mus:
            self.assertEqual(len(mcmc_samples.samples[mu.name]), 1)
            self.assertTrue(np.all(mcmc_samples.samples[mu.name][0] == mu.value))

    def test_predict(self):
        # generate a single tree
        tree, mu = build_test_data(self.X, self.true_sigsqr)
        mu_param = BartMeanParameter('Mu 1', 1)
        mu_param.value = mu
        mu_map = BartStep.node_mu(tree, mu_param)  # true model values for each x

        model = BartModel(self.X, tree.y, m=1)
        bart_sample = BartSample(tree.y, 1, {}, Xtrain=self.X)
        bart_sample.samples['sigsqr'] = [0.0]  # make sure we have one sample
        bart_sample.samples['BART 1'] = [tree]
        mu_transformed = (mu - bart_sample.ymin) / (bart_sample.ymax - bart_sample.ymin) - 0.5
        bart_sample.samples['Mu 1'] = [mu_transformed]
        for i in xrange(self.X.shape[0]):
            ypredict = bart_sample.predict(self.X[i, :])
            self.assertAlmostEqual(ypredict[0], mu_map[i])

    def test_sampler(self):
        pass


if __name__ == "__main__":
    unittest.main()
