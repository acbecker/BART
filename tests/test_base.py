import unittest
import numpy as np
from tree import BaseTree

class BaseTreeTestCases(unittest.TestCase):
    def setUp(self):
        self.nsamples  = 100
        self.nfeatures = 10
        self.X    = np.random.random((self.nsamples, self.nfeatures)) - 0.5
        self.y    = np.random.random((self.nsamples)) - 0.5
        self.tree = BaseTree(self.X, self.y, min_samples_leaf=1) # to avoid random failures

    def tearDown(self):
        del self.X
        del self.y
        del self.tree

    #

    def testBuildUniform(self):
        headId = self.tree.head.Id
        # No probability of growing a new node
        self.tree.buildUniform(self.tree.head, alpha=0.0, beta=1.0)
        ids = [(x.Id-headId) for x in self.tree.internalNodes]
        self.assertTrue(len(ids) == 0)
        ids = [(x.Id-headId) for x in self.tree.terminalNodes]
        self.assertTrue(len(ids) == 1)
        self.assertTrue(0 in ids)

        # alpha < 1.0
        try:
            self.tree.buildUniform(self.tree.head, alpha=1.0, beta=1.0)
        except AssertionError: 
            pass

        # Ensure at least 1 grow
        self.tree.buildUniform(self.tree.head, alpha=0.999999, beta=1.0)
        ids = [(x.Id-headId) for x in self.tree.internalNodes]
        self.assertTrue(len(ids) > 0)
        ids = [(x.Id-headId) for x in self.tree.terminalNodes]
        self.assertTrue(len(ids) > 1)


    def testPrule(self):
        headId = self.tree.head.Id
        # Split on the head of the tree
        feature, threshold = self.tree.prule(self.tree.head)
        self.assertTrue(feature < self.nfeatures)
        self.assertTrue(threshold in self.X[:,feature])

        # Split another node
        l1, r1 = self.tree.split(self.tree.head, 1, 0.0)
        feature, threshold = self.tree.prule(l1)
        self.assertTrue(feature < self.nfeatures)
        self.assertTrue(threshold in self.X[:,feature])


    def testGrow(self):
        headId = self.tree.head.Id
        # Grow a tree from its head; return the head that was grown
        grown  = self.tree.grow()
        self.assertTrue(grown is self.tree.head)

        ids = [(x.Id-headId) for x in self.tree.internalNodes]
        self.assertTrue(0 in ids)
        ids = [(x.Id-headId) for x in self.tree.terminalNodes]
        self.assertTrue(1 in ids)
        self.assertTrue(2 in ids)


    def testSplit(self):
        headId = self.tree.head.Id
        # Set up a tree that looks like
        #         0 
        #        / \
        #       1   2
        #      / \
        #     3   4
        #        / \
        #       5   6
        # 1: Do so in a way that should work (i.e. the splits leave data in each node)
        l1, r1 = self.tree.split(self.tree.head, 1, 0.0) # nodes 1=L, 2=R
        l2, r2 = self.tree.split(self.tree.head.Left, 2, 0.0) # nodes 3=L, 4=R
        l3, r3 = self.tree.split(self.tree.head.Left.Right, 3, 0.0) # nodes 5=L, 6=R

        self.assertTrue((l1.Id-headId) == 1)
        self.assertTrue((r1.Id-headId) == 2)
        self.assertTrue((l2.Id-headId) == 3)
        self.assertTrue((r2.Id-headId) == 4)
        self.assertTrue((l3.Id-headId) == 5)
        self.assertTrue((r3.Id-headId) == 6)

        self.assertTrue(r3.Parent == l3.Parent)
        self.assertTrue((r2.Parent.Id-headId) == 1)
        self.assertTrue(r1.Left is None)
        self.assertTrue(r1.Right is None)
        self.assertTrue(r1.Parent is self.tree.head)

        # 2: Try to split on a node that already has children
        left, right = self.tree.split(self.tree.head.Left.Right, 3, 0.0) 
        self.assertTrue(left is None)
        self.assertTrue(right is None)

        # 3: Try an additional split that should *not* work because it
        # leaves an empty leaf.  Verify that is is a no-op
        terminal = self.tree.terminalNodes
        internal = self.tree.internalNodes
        left, right = self.tree.split(self.tree.head.Left.Right.Left, 4, -1.0) # nodes 5=L, 6=R
        self.assertTrue(left is None)
        self.assertTrue(right is None)
        self.assertTrue(terminal == self.tree.terminalNodes)
        self.assertTrue(internal == self.tree.internalNodes)

    def testPrune(self):
        headId = self.tree.head.Id
        # Set up a tree that looks like
        #         0 
        #        / \
        #       1   2
        #      / \
        #     3   4
        #    / \
        #   5   6
        # 1: Do so in a way that should work (i.e. the splits leave data in each node)
        l1, r1 = self.tree.split(self.tree.head, 1, 0.0) # nodes 1=L, 2=R
        l2, r2 = self.tree.split(self.tree.head.Left, 2, 0.0) # nodes 3=L, 4=R
        l3, r3 = self.tree.split(self.tree.head.Left.Left, 3, 0.0) # nodes 5=L, 6=R

        # The only way to prune this tree is to get rid of 5 and 6
        parent = self.tree.prune()
        self.assertTrue((parent.Id-headId) == 3)
        ids = [(x.Id-headId) for x in self.tree.internalNodes]
        self.assertTrue(0 in ids)
        self.assertTrue(1 in ids)

        # Then the only way to prune it again is to get rid of 3 and 4
        parent = self.tree.prune()
        self.assertTrue((parent.Id-headId) == 1)
        ids = [(x.Id-headId) for x in self.tree.internalNodes]
        self.assertTrue(0 in ids)

        # Finally, get rid of 1 and 2
        parent = self.tree.prune()
        self.assertTrue(parent == self.tree.head)

        # Can't prune no more!
        parent = self.tree.prune()
        self.assertTrue(parent is None)


    def testTerminalNodes(self):
        headId = self.tree.head.Id
        # Set up a tree that looks like
        #         0 
        #        / \
        #       1   2
        #      / \
        #     3   4
        #        / \
        #       5   6
        # 1: Do so in a way that should work (i.e. the splits leave data in each node)
        l1, r1 = self.tree.split(self.tree.head, 1, 0.0) # nodes 1=L, 2=R
        l2, r2 = self.tree.split(self.tree.head.Left, 2, 0.0) # nodes 3=L, 4=R
        l3, r3 = self.tree.split(self.tree.head.Left.Right, 3, 0.0) # nodes 5=L, 6=R
        ids = [(x.Id-headId) for x in self.tree.terminalNodes]
        self.assertTrue(2 in ids)
        self.assertTrue(3 in ids)
        self.assertTrue(5 in ids)
        self.assertTrue(6 in ids)

        # 2: If you have 2 terminal nodes with a common parent, return those parent nodes
        ids = [(x.Id-headId) for x in self.tree.get_terminal_parents()]
        self.assertTrue(4 in ids)


    def testInternalNodes(self):
        headId = self.tree.head.Id
        # Set up a tree that looks like
        #         0 
        #        / \
        #       1   2
        #          / \
        #         3   4
        #            / \
        #           5   6
        # 1: Do so in a way that should work (i.e. the splits leave data in each node)
        self.tree.split(self.tree.head, 1, 0.0) # nodes 1=L, 2=R
        self.tree.split(self.tree.head.Right, 2, 0.0) # nodes 3=L, 4=R
        self.tree.split(self.tree.head.Right.Right, 3, 0.0) # nodes 5=L, 6=R
        ids = [(x.Id-headId) for x in self.tree.internalNodes]
        self.assertTrue(0 in ids)
        self.assertTrue(2 in ids)
        self.assertTrue(4 in ids)

    def testPlinko(self):
        headId = self.tree.head.Id
        feat   = 1
        # Set up a tree that looks like
        #         0 
        #        / \
        #       1   2
        #          / \
        #         3   4
        n1, n2 = self.tree.split(self.tree.head, feat, 0.0) # nodes 1=L, 2=R
        n3, n4 = self.tree.split(self.tree.head.Right, feat, 0.25) # nodes 3=L, 4=R
        idx1 = np.where(self.X[:,feat] <= 0.0)[0]
        idx2 = np.where(self.X[:,feat]  > 0.0)[0]
        idx3 = np.where((self.X[:,feat]  > 0.0) & (self.X[:,feat]  <= 0.25))[0]
        idx4 = np.where((self.X[:,feat]  > 0.0) & (self.X[:,feat]  > 0.25))[0]
        self.assertTrue(len(idx2) == (len(idx3)+len(idx4)))
        p1 = self.tree.plinko(n1, self.X)
        p2 = self.tree.plinko(n2, self.X)
        p3 = self.tree.plinko(n3, self.X)
        p4 = self.tree.plinko(n4, self.X)
        # Only cutting on feature 1 in this test
        for idxA in range(self.nfeatures):
            for plinko, idxB in zip((p1, p2, p3, p4), (idx1, idx2, idx3, idx4)):
                if idxA == feat:
                    self.assertTrue(np.sum(plinko[:,idxA]) == len(idxB))
                else:
                    self.assertTrue(False not in plinko[:,idxA])

        # Two feature selection
        #            ...
        #             4
        #            / \
        #           5   6
        feat2  = 7
        n5, n6 = self.tree.split(self.tree.head.Right.Right, feat2, 0.0) # nodes 3=L, 4=R
        idx5 = np.where((self.X[:,feat]  > 0.0) & (self.X[:,feat]  > 0.25) & (self.X[:,feat2]  <= 0.0))[0]
        idx6 = np.where((self.X[:,feat]  > 0.0) & (self.X[:,feat]  > 0.25) & (self.X[:,feat2]  >  0.0))[0]
        p5   = self.tree.plinko(n5, self.X)
        p6   = self.tree.plinko(n6, self.X)
        for idxA in range(self.nfeatures):
            for plinko, idxB in zip((p5, p6), (idx5, idx6)):
                if idxA == feat:
                    idxFeat  = plinko[:,idxA] & plinko[:,feat2]
                    self.assertTrue(np.sum(idxFeat) == len(idxB))
                elif idxA == feat2:
                    # Tested above
                    continue
                else:
                    self.assertTrue(False not in plinko[:,idxA])

    def testFilter(self):
        headId = self.tree.head.Id
        # Set up a tree that looks like
        #         0 
        #        / \
        #       1   2
        #      / \
        #     3   4
        feat1  = 1
        feat2  = 6
        n1, n2 = self.tree.split(self.tree.head, feat1, 0.0) # nodes 1=L, 2=R
        n3, n4 = self.tree.split(self.tree.head.Left, feat2, 0.25) # nodes 3=L, 4=R

        idx1 = np.where(self.X[:,feat1] <= 0.0)[0]
        idx2 = np.where(self.X[:,feat1]  > 0.0)[0]
        idx3 = np.where((self.X[:,feat1] <= 0.0) & (self.X[:,feat2]  <= 0.25))[0]
        idx4 = np.where((self.X[:,feat1] <= 0.0) & (self.X[:,feat2]  > 0.25))[0]

        self.tree.filter(n1)
        self.tree.filter(n2)
        self.tree.filter(n3)
        self.tree.filter(n4)

        for node, idx in zip((n1, n2, n3, n4), (idx1, idx2, idx3, idx4)):
            self.assertTrue(len(idx) == node.npts)
            self.assertAlmostEqual(node.ybar, np.mean(self.y[idx]))
            self.assertAlmostEqual(node.yvar, np.std(self.y[idx])**2)

        # For good measure, trim and recheck
        self.tree.prune()
        self.assertTrue(n1.Left is None)
        self.assertTrue(n1.Right is None)
        for node, idx in zip((n1, n2), (idx1, idx2)):
            self.assertTrue(len(idx) == node.npts)
            self.assertAlmostEqual(node.ybar, np.mean(self.y[idx]))
            self.assertAlmostEqual(node.yvar, np.std(self.y[idx])**2)

        # For good measure, grow and recheck
        self.tree.grow()
        self.assertTrue((n1.Left is not None) or (n2.Left is not None))
        self.assertTrue((n1.Right is not None) or (n2.Right is not None))
        for node, idx in zip((n1, n2), (idx1, idx2)):
            self.assertTrue(len(idx) == node.npts)
            self.assertAlmostEqual(node.ybar, np.mean(self.y[idx]))
            self.assertAlmostEqual(node.yvar, np.std(self.y[idx])**2)


if __name__ == "__main__":
    unittest.main()
