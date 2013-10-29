import numpy as np

seed = 666
def randInt(maxInt):
    seed ^= seed << 13
    seed ^= seed >> 17
    seed ^= seed << 5
    return (seed % (0x7FFFFFFF + 1)) % (maxInt)

class CartTree(object):
    def __init__(self, X, y, min_samples_leaf=5):
        self.X = X
        self.y = y
        self.n_features = X.shape[0]
        self.n_samples  = X.shape[1]
        self.min_samples_leaf = min_samples_leaf
        self.head = Node(None, 0, 0, None)

    # GROW step: randomly pick a terminal node and split into 2 new
    # ones by randomly assigning a splitting rule
    def grow(self, feature, threshold):
        nodes = self.terminalNodes()
        rnode = nodes[randInt(len(nodes))]
        self.split(rnode, feature, threshold)

    def split(self, parent, feature, threshold):
        # Threshold is of length self.n_features
        nleft  = Node(parent, feature, threshold, True)  # Add left node; it registers with parent
        nright = Node(parent, feature, threshold, False) # Add right node; it registers with parent
        return nleft, nright


    # PRUNE step: randomly pick a parent of 2 terminal nodes and turn
    # it into a terminal node
    def prune(self):
        nodes = self.terminalNodes()
        parents = list(set([x.parent() for x in nodes]))
        parent = parents[randInt(len(parents))]
        parent.Left = None
        parent.Right = None


    # CHANGE step: randomly pick an internal node and randomly assign
    # it a splitting rule.  
    def change(self, feature, threshold):
        # Threshold is of length self.n_features
        nodes = self.internalNodes()
        rnode = nodes[randInt(len(nodes))]
        rnode.setThreshold(feature, threshold)


    # SWAP step: randomly pick a parent-child pair that are both
    # internal nodes.  Swap their splitting rules unless the other
    # child has the identical rule, in which case swap the splitting
    # rule of the parent with both children
    def swap(self):
        nodes  = self.internalNodes()
        nids   = set([n.Id for n in nodes])
        pids   = set([n.Parent.Id for n in nodes])
        pairs  = list(nids.intersection(pids))
        pairId = pairs[randInt(len(pairs))]
        

    def printTree(self, node):
        if node is None:
            return
        self.printTree(node.Left)
        self.printTree(node.Right)
        print node.Id


    # Calculate the terminal nodes of the tree
    def getTerminalNodes(self):
        self.terminalNodes = []
        self.calcTerminalNodes(self.head)
        return self.terminalNodes 

    def calcTerminalNodes(self, node):
        if node.Right is None or node.Left is None:
            self.terminalNodes.append(node)
        if node.Right is not None:
            self.calcTerminalNodes(node.Right)
        if node.Left is not None:
            self.calcTerminalNodes(node.Left)


    # Calculate the internal nodes of the tree
    def getInternalNodes(self):
        self.internalNodes = []
        self.calcInternalNodes(self.head)
        return self.internalNodes 

    def calcInternalNodes(self, node):
        if node.Right is not None and node.Left is not None:
            self.internalNodes.append(node)
        if node.Right is not None:
            self.calcInternalNodes(node.Right)
        if node.Left is not None:
            self.calcInternalNodes(node.Left)
            
class Node(object):
    NodeId = 0

    def __init__(self, parent, feature, threshold, is_left):
        self.Id        = Node.NodeId
        Node.NodeId   += 1

        self.Parent    = parent
        self.Left      = None # data[:, feature] <= threshold
        self.Right     = None # data[:, feature] > threshold
        self.setThreshold(feature, threshold)

        if self.Parent is not None:
            self.is_left   = is_left
            if self.is_left:
                parent.Left = self
            else:
                parent.Right = self

    def setThreshold(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

if __name__ == "__main__":
    nsamples  = 100
    nfeatures = 10
    X    = np.zeros((nsamples, nfeatures))
    y    = np.zeros((nfeatures))
    tree = CartTree(X, y)
    tree.split(tree.head, 10, 7)
    tree.split(tree.head.Left, 10, 7)
    tree.printTree(tree.head)

    print "Terminal", [x.Id for x in tree.getTerminalNodes()]
    print "Internal", [x.Id for x in tree.getInternalNodes()]
