import unittest
import numpy as np
from tree import CartTree, Node

class TreeTestCases(unittest.TestCase):
    def setUp(self):
        nsamples  = 100
        nfeatures = 10
        self.X    = np.random.random((nsamples, nfeatures)) - 0.5
        self.y    = np.random.random((nfeatures)) - 0.5
        self.tree = CartTree(self.X, self.y)

    def tearDown(self):
        del self.X
        del self.y
        del self.tree

    def testGrow(self):
        headId = self.tree.head.Id
        self.tree.grow(1, 0.0)
        self.assertTrue([x.Id-headId for x in self.tree.terminalNodes] == [2, 1])
        self.assertTrue([x.Id-headId for x in self.tree.internalNodes] == [0])

    def testSplit(self):
        headId = self.tree.head.Id
        self.tree.split(self.tree.head, 1, 0.0)
        self.tree.split(self.tree.head.Left, 2, 0.0)
        self.tree.split(self.tree.head.Left.Right, 3, 0.0)
        self.assertTrue([x.Id-headId for x in self.tree.terminalNodes] == [2, 6, 5, 3])
        self.assertTrue([x.Id-headId for x in self.tree.internalNodes] == [0, 1, 4])

    def testPrune(self):
        headId = self.tree.head.Id
        self.tree.split(self.tree.head, 1, 0.0)
        self.tree.prune()
        self.assertTrue([x.Id-headId for x in self.tree.terminalNodes] == [0])
        self.assertTrue([x.Id-headId for x in self.tree.internalNodes] == [])

    def testChange(self):
        headId = self.tree.head.Id
        self.tree.grow(1, 0.0)
        self.tree.grow(2, 0.0)
        self.tree.grow(3, 0.0)
        tNodes = self.tree.terminalNodes
        iNodes = self.tree.internalNodes
        node = self.tree.change(4, 1.0)
        self.assertTrue(node.feature == 4)
        self.assertTrue(node.threshold == 1.0)
        self.assertTrue(tNodes == self.tree.terminalNodes) # tree itself is not changed
        self.assertTrue(iNodes == self.tree.internalNodes)

    def testSwap(self):
        headId = self.tree.head.Id
        self.tree.split(self.tree.head, 1, 0.0)
        self.tree.split(self.tree.head.Left, 2, 0.0)
        self.tree.swap()

if __name__ == "__main__":
    unittest.main()
