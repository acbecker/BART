import unittest
import numpy as np
from tree import Node

class NodeTestCases(unittest.TestCase):
    def testParent(self):
        head = Node(None, None)
        self.assertTrue(head.Parent is None)
        self.assertTrue(head.Left is None)
        self.assertTrue(head.Right is None)
        self.assertTrue(head.depth == 0)
        self.assertTrue(head.is_left is None)

        head.feature = 0
        head.threshold = 1.0
        self.assertTrue(head.feature == 0)
        self.assertTrue(head.threshold == 1.0)
        self.assertTrue(head._feature == 0)
        self.assertTrue(head._threshold == 1.0)

    def testChildren(self):
        head  = Node(None, None)
        left  = Node(head, True)
        right = Node(head, False)
        self.assertTrue(head.Left == left)
        self.assertTrue(head.Right == right)
        self.assertTrue(left.Parent == head)
        self.assertTrue(right.Parent == head)
        self.assertTrue(left.is_left is True)
        self.assertTrue(right.is_left is False)

        self.assertTrue(head.depth == 0)
        self.assertTrue(left.depth == 1)
        self.assertTrue(right.depth == 1)

        left.ybar = 2.0
        left.yvar = 1.0
        left.npts = 100
        assert(left.ybar == 2.0)
        assert(left.yvar == 1.0)
        assert(left.npts == 100)
        assert(left._ybar == 2.0)
        assert(left._yvar == 1.0)
        assert(left._npts == 100)

if __name__ == "__main__":
    unittest.main()
