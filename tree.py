import numpy as np
import collections
import scipy.stats as stats
from scipy.special import gammaln

####
#################
####

class BaseTree(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.n_samples  = X.shape[0]

        # Initialize the tree
        self.head = Node(None, None)
        self.terminalNodes = [self.head,]
        self.internalNodes = []

    def prule(self, node):
        """Implement a uniform draw from the features to split on, and
        then choose the split value uniformly from the set of
        available observed values"""
        feature = np.random.randint(self.n_features)
        idxX, idxY = self.filter(node)
        data = self.X[:, feature][idxX[:, feature]]
        if len(data) == 0: return None, None
        idxD = np.random.randint(len(data))
        threshold = data[idxD]
        return feature, threshold

    # GROW step: randomly pick a terminal node and split into 2 new
    # ones by randomly assigning a splitting rule.  
    def grow(self):
        nodes = self.terminalNodes
        rnode = nodes[np.random.randint(len(nodes))]
        feature, threshold = self.prule(rnode)
        if feature is None or threshold is None:
            return
        self.split(rnode, feature, threshold)

    def split(self, parent, feature, threshold):
        # Threshold is of length self.n_features
        nleft  = Node(parent, True)  # Add left node; it registers with parent
        nright = Node(parent, False) # Add right node; it registers with parent
        parent.setThreshold(feature, threshold)

        fxl, fyl = self.filter(nleft)
        fxr, fyr = self.filter(nright)
        
        # only split if it yields at least nmin points per child
        if np.sum(fyl) >= self.nmin and np.sum(fyr) >= self.nmin:
            self.calcTerminalNodes()
            self.calcInternalNodes()
            return nleft, nright
        else:
            del nleft
            del nright
            parent.setThreshold(None, None)
            parent.Left = None
            parent.Right = None
            return None, None


    # PRUNE step: randomly pick a parent of 2 terminal nodes and turn
    # it into a terminal node
    #
    # Note: this updates the internal/terminal nodes.
    def prune(self):
        nodes    = self.terminalNodes
        if len(nodes) == 0: return
        parents  = [x.Parent for x in nodes]
        if len(parents) == 0: return
        dparents = [x for x, y in collections.Counter(parents).items() if y == 2] # make sure there are 2 terminal children
        if len(dparents) == 0: return
        parent   = dparents[np.random.randint(len(dparents))]
        parent.Left = None
        parent.Right = None
        self.calcTerminalNodes()
        self.calcInternalNodes()


    # CHANGE step: randomly pick an internal node and randomly assign
    # it a splitting rule.  
    def change(self):
        nodes = self.internalNodes
        if len(nodes) == 0: return
        rnode = nodes[np.random.randint(len(nodes))]
        feature0, threshold0 = rnode.feature, rnode.threshold

        # See if we get an acceptable new split
        featureN, thresholdN = self.prule(rnode)
        if featureN is None or thresholdN is None:
            return
        rnode.setThreshold(featureN, thresholdN)

        # Hmm, do I want to descend down the whole tree to see the consequences of this?
        fxl, fyl = self.filter(rnode.Left)
        fxr, fyr = self.filter(rnode.Right)
        if np.sum(fyl) < self.nmin or np.sum(fyr) < self.nmin:
            rnode.setThreshold(feature0, threshold0) # Undo, unacceptable split


    # SWAP step: randomly pick a parent-child pair that are both
    # internal nodes.  Swap their splitting rules unless the other
    # child has the identical rule, in which case swap the splitting
    # rule of the parent with both children
    def swap(self):
        nodes  = self.internalNodes
        if len(nodes) == 0: return
        pnodes = list(set([n.Parent for n in nodes if n.Parent in nodes])) # Find an internal parent node with internal children
        if len(pnodes) == 0: return 
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
            
    # Filter the data that end up in each (terminal) node; return
    # their locations
    def filter(self, node):
        includeX = np.ones(self.X.shape, dtype=np.bool)
        n = node
        while n.Parent is not None:
            if n.is_left:
                includeX[:,n.Parent.feature] &= self.X[:,n.Parent.feature] <=  n.Parent.threshold
            else:
                includeX[:,n.Parent.feature] &= self.X[:,n.Parent.feature] >   n.Parent.threshold
            n = n.Parent
        includeY = np.all(includeX, axis=1)
        return includeX, includeY


        
    

####
#################
####


class BartProposal(object):
    def __init__(self):
        pass

    def __call__(self, tree):
        prop = np.random.uniform()
        if prop < 0.25:
            print "# GROW",
            tree.grow()
        elif prop < 0.50:
            print "# PRUNE",
            tree.prune()
        elif prop < 0.90:
            print "# CHANGE",
            tree.change()
        else:
            print "# SWAP",
            tree.swap()

class BartTree(BaseTree):
    def __init__(self, X, y):
        BaseTree.__init__(self, X, y)

        # rescale y to lie between -0.5 and 0.5
        self.y += np.min(self.y) # minimum = 0
        self.y /= np.max(self.y) # maximum = 1
        self.y -= 0.5            # range is -0.5 to 0.5

        self.k     = 2
        self.nu    = 3.0
        self.q     = 0.90
        self.m     = 200    # number of trees

####
#################
####

class CartProposal(object):
    def __init__(self):
        pass

    def __call__(self, tree):
        prop = np.random.uniform()
        if prop < 0.25:
            print "# GROW",
            tree.grow()
        elif prop < 0.50:
            print "# PRUNE",
            tree.prune()
        elif prop < 0.75:
            print "# CHANGE",
            tree.change()
        else:
            print "# SWAP",
            tree.swap()
        

class CartTree(BaseTree):
    # Describes the conditional distribution of y given X.  X is a
    # vector of predictors.  Each terminal node has parameter Theta.
    # 
    # y|X has distribution f(y|Theta).

    def __init__(self, X, y, 
                 nu, lamb, mubar, a, 
                 alpha=0.95, beta=1.0,
                 min_samples_leaf=5):
        BaseTree.__init__(self, X, y)

        # How big of a tree do we want to make
        self.nmin = min_samples_leaf

        # Tuning parameters of the model
        self.nu    = nu
        self.lamb  = lamb
        self.mubar = mubar
        self.a     = a
        self.alpha = alpha
        self.beta = beta

        # Build the tree
        self.buildUniform(self.head, alpha, beta)

    def buildUniform(self, node, alpha, beta, depth=0):
        psplit = alpha * (1 + depth)**-beta
        rand   = np.random.uniform()
        if rand < psplit:
            feature, threshold = self.prule(node)
            if feature is None or threshold is None:
                print "NO DATA LEFT, rejecting split"
                return
            nleft, nright = self.split(node, feature, threshold)
            if nleft is not None and nright is not None:
                print "EXTENDING node", node.Id, "to depth", depth+1
                self.buildUniform(nleft, alpha, beta, depth=depth+1)
                self.buildUniform(nright, alpha, beta, depth=depth+1)
            else:
                print "NOT EXTENDING node", node.Id, ": too few points"
        else:
            print "NOT SPLITTING node", node.Id, ": did not pass random draw"

    def logprior(self, tree):

        logprior = 0.0
        # first get prior for terminal nodes
        for node in tree.terminalNodes:
            logprior += np.log(1.0 - self.alpha / (1.0 + depth) ** self.beta)

            fxl, fyl = tree.filter(node)
            npts = np.sum(fyl)
            nfeatures = np.sum(fxl, axis=0)

    # NOTE: This part would likely benefit from numba or cython
    def loglik(self, tree):
        """
        Compute the log-likelihood for a proposed tree model. This assumes that the only difference between the input
        tree and self is in the structure of the tree nodes. The prior and data are assumed to be the same.

        @param tree: The proposed tree.
        @return: The log-likelihood of tree.
        """
        lnlike = 0.0

        # Precalculate terms
        t2  = np.log((self.nu * self.lamb)**(0.5 * self.nu))
        t4b = gammaln(0.5 * self.nu)
        
        for node in tree.terminalNodes:
            fxl, fyl = tree.filter(node)
            npts = np.sum(fyl)
            if npts == 0:
                # Damn, this should not happen.
                # DEBUG ME
                continue
            ymean = np.mean(self.y[fyl])
            yvar = np.var(self.y[fyl], ddof=1)

            # Terms that depend on the data moments
            si = (npts - 1) * yvar
            ti = (npts * self.a) / (npts + self.a) * (ymean - self.mubar)**2

            # Calculation of the log likelihood (Chipman Eq 14)
            t1 = -0.5 * npts * np.log(np.pi)
            t3 = +0.5 * np.log(self.a / (npts + self.a))
            t4 = gammaln(0.5 * (npts + self.nu)) - t4b
            t5 = -0.5 * (npts + self.nu) * np.log(si + ti + self.nu * self.lamb)
            lnlike += t1 + t2 + t3 + t4 + t5
            #print npts, ymean, yvar, lnlike

        return lnlike

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

    # NOTE: the parent carries the threshold
    def setThreshold(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

    def find_depth(self):
        """
        Find the depth of this node by moving back through the parents until we reach the base of the tree.
        """
        depth = 0
        parent = self.Parent
        while parent is not None:
            parent = parent.Parent
            depth += 1

        return depth

if __name__ == "__main__":
    nsamples  = 1000
    nfeatures = 20
    X    = np.random.random((nsamples, nfeatures)) - 0.5
    y    = np.random.random((nsamples)) - 0.5
    tree = CartTree(X, y, nu=0.1, lamb=2/0.1, mubar=np.mean(y), a=1.0, alpha=0.99, beta=1.0/np.log(nsamples))
    prop = CartProposal()
    tree.printTree(tree.head)
    for i in range(10000):
        prop(tree)
        print tree.regressionLnlike()

    print "Terminal", [x.Id for x in tree.terminalNodes]
    print "Internal", [x.Id for x in tree.internalNodes]
