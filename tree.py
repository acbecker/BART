import numpy as np
import collections

class CartProposal(object):
    def __init__(self, tree, alpha=0.95, beta=1.0):
        self.tree = tree
        self.alpha = alpha
        self.beta = beta

    def __call__(self):
        prop = np.random.random()
        if prop < 0.25:
            self.tree.grow()
        elif prop < 0.50:
            self.tree.prune()
        elif prop < 0.75:
            self.tree.change()
        else:
            self.tree.swap()
        

class CartTree(object):
    # Describes the conditional distribution of y given X.  X is a
    # vector of predictors.  Each terminal node has parameter Theta.
    # 
    # y|X has distribution f(y|Theta).

    def __init__(self, X, y, min_samples_leaf=5):
        self.X = X
        self.y = y
        self.n_features = X.shape[0]
        self.n_samples  = X.shape[1]
        self.min_samples_leaf = min_samples_leaf
        self.head = Node(None, None)

        self.terminalNodes = [self.head,]
        self.internalNodes = []

    # GROW step: randomly pick a terminal node and split into 2 new
    # ones by randomly assigning a splitting rule.  
    #
    # Note: this updates the internal/terminal nodes.
    def grow(self, feature, threshold):
        nodes = self.terminalNodes
        rnode = nodes[np.random.randint(len(nodes))]
        return rnode, self.split(rnode, feature, threshold)

    def split(self, parent, feature, threshold):
        # Threshold is of length self.n_features
        nleft  = Node(parent, True)  # Add left node; it registers with parent
        nright = Node(parent, False) # Add right node; it registers with parent
        parent.setThreshold(feature, threshold)
        self.calcTerminalNodes()
        self.calcInternalNodes()
        return nleft, nright


    # PRUNE step: randomly pick a parent of 2 terminal nodes and turn
    # it into a terminal node
    #
    # Note: this updates the internal/terminal nodes.
    def prune(self):
        nodes    = self.terminalNodes
        parents  = [x.Parent for x in nodes]
        dparents = [x for x, y in collections.Counter(parents).items() if y == 2] # make sure there are 2 terminal children
        parent   = dparents[np.random.randint(len(dparents))]
        parent.Left = None
        parent.Right = None
        self.calcTerminalNodes()
        self.calcInternalNodes()
        return parent

    # CHANGE step: randomly pick an internal node and randomly assign
    # it a splitting rule.  
    def change(self, feature, threshold):
        nodes = self.internalNodes
        rnode = nodes[np.random.randint(len(nodes))]
        rnode.setThreshold(feature, threshold)
        return rnode

    # SWAP step: randomly pick a parent-child pair that are both
    # internal nodes.  Swap their splitting rules unless the other
    # child has the identical rule, in which case swap the splitting
    # rule of the parent with both children
    def swap(self):
        nodes  = self.internalNodes
        pnodes = list(set([n.Parent for n in nodes if n.Parent in nodes])) # Find an internal parent node with internal children
        if len(pnodes) == 0:
            return None
        pnode  = pnodes[np.random.randint(len(pnodes))]
        lnode  = pnode.Left
        rnode  = pnode.Right
        # Both children have the same selection; modify both and return
        if lnode.feature == rnode.feature and lnode.threshold == rnode.threshold:
            pfeat   = pnode.feature
            pthresh = pnode.threshold
            cfeat   = lnode.feature
            cthresh = lnode.threshold
            pnode.setThreshold(cfeat, cthresh)
            lnode.setThreshold(pfeat, pthresh)
            rnode.setThreshold(pfeat, pthresh)
            return pnode, lnode, rnode

        # Choose one of them that is also an internal node; modify that one only
        cnodes = []
        if lnode in nodes: cnodes.append(lnode)
        if rnode in nodes: cnodes.append(rnode)
        cnode   = cnodes[np.random.randint(len(cnodes))]
        pfeat   = pnode.feature
        pthresh = pnode.threshold
        cfeat   = cnode.feature
        cthresh = cnode.threshold
        cnode.setThreshold(pfeat, pthresh)
        pnode.setThreshold(cfeat, cthresh)
        return pnode, cnode
        

    def printTree(self, node):
        if node is None:
            return
        self.printTree(node.Left)
        self.printTree(node.Right)
        print node.Id


    # Calculate the terminal nodes of the tree
    def calcTerminalNodes(self):
        self.terminalNodes = []
        self.calcTerminalNodes_(self.head)

    def calcTerminalNodes_(self, node):
        if node.Right is None or node.Left is None:
            self.terminalNodes.append(node)
        if node.Right is not None:
            self.calcTerminalNodes_(node.Right)
        if node.Left is not None:
            self.calcTerminalNodes_(node.Left)


    # Calculate the internal nodes of the tree
    def calcInternalNodes(self):
        self.internalNodes = []
        self.calcInternalNodes_(self.head)

    def calcInternalNodes_(self, node):
        if node.Right is not None and node.Left is not None:
            self.internalNodes.append(node)
        if node.Right is not None:
            self.calcInternalNodes_(node.Right)
        if node.Left is not None:
            self.calcInternalNodes_(node.Left)
            
    def filter(self, node):
        include = np.ones(self.y.shape, dtype=np.bool)
        n = node
        while n.Parent is not None:
            if n.is_left:
                include &= self.X[:,n.Parent.feature] <=  n.Parent.threshold
            else:
                include &= self.X[:,n.Parent.feature] >   n.Parent.threshold
            n = n.Parent
        return np.mean(self.y[include])
            
class Node(object):
    NodeId = 0

    def __init__(self, parent, is_left):
        self.Id        = Node.NodeId
        Node.NodeId   += 1

        self.Parent    = parent # feture and threshold reside in the parent
        self.Left      = None   # data[:, feature] <= threshold
        self.Right     = None   # data[:, feature] > threshold
        self.setThreshold(None, None)

        if self.Parent is not None:
            self.is_left = is_left
            if self.is_left:
                parent.Left = self
            else:
                parent.Right = self

    # NOTE: the parent effetively carries the threshold
    def setThreshold(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

if __name__ == "__main__":
    nsamples  = 100
    nfeatures = 10
    X    = np.random.random((nsamples, nfeatures)) - 0.5
    y    = np.random.random((nsamples)) - 0.5
    tree = CartTree(X, y)
    tree.split(tree.head, 1, 0.0)
    tree.split(tree.head.Left, 2, 0.1)
    tree.split(tree.head.Left.Right, 3, -0.1)
    tree.printTree(tree.head)

    print "Terminal", [x.Id for x in tree.terminalNodes]
    print "Internal", [x.Id for x in tree.internalNodes]

    for node in tree.terminalNodes:
        print node.Id, tree.filter(node)
