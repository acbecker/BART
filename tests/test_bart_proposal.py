__author__ = 'brandonkelly'

import unittest
import numpy as np
from scipy import stats, integrate
from tree import *
import matplotlib.pyplot as plt
from test_tree_parameters import build_test_data, SimpleBartStep


class ProposalTestCase(unittest.TestCase):
    def setUp(self):
        nsamples = 500
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

    def test_draw(self):
        # make sure grow/prune operations return a tree with correct # of terminal nodes
        current_tree = self.tree.value
        ntrials = 1000
        for i in xrange(ntrials):
            new_tree = self.tree_proposal.draw(current_tree)
            nleafs_new = len(new_tree.terminalNodes)
            nleafs_old = len(current_tree.terminalNodes)
            if self.tree_proposal._node is None or self.tree_proposal._node.feature is None:
                # make sure tree configuration is not updated
                self.assertEqual(nleafs_new, nleafs_old)
            elif self.tree_proposal._operation == 'grow':
                # make sure there is one more terminal node
                self.assertEqual(nleafs_new, nleafs_old + 1)
            else:
                # make sure there is one less terminal node
                self.assertEqual(nleafs_new, nleafs_old - 1)

            current_tree = new_tree

    def test_logdensity(self):
        # make sure ratio of transition kernels matches the values computed directly
        current_tree = self.tree.value
        ntrials = 1000
        for i in xrange(ntrials):
            new_tree = self.tree_proposal.draw(current_tree)
            logratio = self.tree_proposal.logdensity(new_tree, current_tree, True)
            logratio = -logratio  # sign of output agrees with MetroStep.accept, reverse for convention of this test
            nleafs_new = len(new_tree.terminalNodes)
            nleafs_old = len(current_tree.terminalNodes)

            if self.tree_proposal._node is None or self.tree_proposal._node.feature is None:
                # tree configuration is not updated
                self.assertAlmostEqual(logratio, 0.0)
                continue

            elif self.tree_proposal._operation == 'grow':
                log_forward = -np.log(nleafs_old) - np.log(current_tree.n_features) - \
                              np.log(self.tree_proposal._node.npts)
                # reverse is the prune update
                log_backward = -np.log(len(new_tree.get_terminal_parents()))

            else:
                log_forward = -np.log(len(current_tree.get_terminal_parents()))
                # reverse mode is grow update
                log_backward = -np.log(nleafs_new) - np.log(new_tree.n_features) - np.log(self.tree_proposal._node.npts)

            logratio_direct = self.tree.logprior(new_tree) - log_forward - \
                (self.tree.logprior(current_tree) - log_backward)
            self.assertAlmostEqual(logratio, logratio_direct)

            current_tree = new_tree

    def test_mcmc(self):
        # run a simple MCMC sampler for the tree configuration to make sure that we correctly constrain the number of
        # internal and terminal nodes
        burnin = 1000
        niter = 5000
        true_nleaves = len(self.tree.value.terminalNodes)
        true_ninodes = len(self.tree.value.internalNodes)
        metro_step = steps.MetroStep(self.tree, self.tree_proposal, niter)
        nleaves = np.zeros(niter)
        ninodes = np.zeros(niter)

        naccepted = 0
        naccept_grow = 0
        naccept_prune = 0
        print 'Doing burnin...'
        for i in xrange(burnin):
            # burnin phase
            if i % 100 == 0:
                print i, '...'
            old_nleaves = len(self.tree.value.terminalNodes)
            old_nbranches = len(self.tree.value.internalNodes)
            metro_step.do_step()
            new_nleaves = len(self.tree.value.terminalNodes)
            new_nbranches = len(self.tree.value.internalNodes)
            accepted = (metro_step.naccept - naccepted) == 1
            if accepted:
                if self.tree_proposal._operation == 'grow':
                    naccept_grow += 1
                elif self.tree_proposal._operation == 'prune':
                    naccept_prune += 1
                if not self.tree_proposal._prohibited_proposal:
                    # make sure the tree configuration has changed
                    self.assertNotEqual(old_nleaves, new_nleaves)
                    self.assertNotEqual(old_nbranches, new_nbranches)
                else:
                    # proposal results in prohibited tree structure, so structure is unchanged
                    self.assertEqual(old_nbranches, new_nbranches)
                    self.assertEqual(old_nleaves, new_nleaves)
                naccepted += 1
            else:
                # proposal rejected, make sure tree is not updated
                self.assertEqual(old_nleaves, new_nleaves)
                self.assertEqual(old_nbranches, new_nbranches)

        print 'Sampling tree structures...'
        for i in xrange(niter):
            if i % 100 == 0:
                print i, '...'
            # now save the number of nodes sampled from their posterior
            metro_step.do_step()
            nleaves[i] = len(self.tree.value.terminalNodes)
            ninodes[i] = len(self.tree.value.internalNodes)

        print 'Number of accepted grow proposals:', naccept_grow
        print 'Number of accepted prune proposals:', naccept_prune

        plt.plot(nleaves, '.')
        plt.plot(ninodes, 'r.')
        plt.ylabel('Number of nodes')
        plt.show()

        ntrue = np.sum(ninodes[nleaves == true_nleaves] == true_ninodes)
        ntrue_fraction = ntrue / float(niter)

        # posterior probability of correct number of internal and terminal nodes should be at least 5%
        print ntrue_fraction
        self.assertGreater(ntrue_fraction, 0.05)


if __name__ == "__main__":
    unittest.main()
