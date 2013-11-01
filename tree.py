import numpy as np
import collections
import scipy.stats as stats
from scipy.special import gammaln
import steps
import proposals
import copy
from sklearn import linear_model

####
#################
####

class BaseTree(object):
    def __init__(self, X, y, min_samples_leaf=5):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.n_samples  = X.shape[0]
        self.nmin       = min_samples_leaf

        # Initialize the tree
        self.head = Node(None, None)
        self.terminalNodes = [self.head,]
        self.internalNodes = []

    def buildUniform(self, node, alpha, beta, depth=0):
        psplit = alpha * (1 + depth)**-beta
        rand   = np.random.uniform()
        if rand < psplit:
            feature, threshold, ndata = self.prule(node)
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
            
    def prule(self, node):
        """Implement a uniform draw from the features to split on, and
        then choose the split value uniformly from the set of
        available observed values.  NOTE: do not change the node
        attributes here since this may be rejected"""
        feature = np.random.randint(self.n_features)
        idxX, idxY = self.filter(node)
        data = self.X[:, feature][idxX[:, feature]]
        if len(data) == 0:
            return None, None, None
        idxD = np.random.randint(len(data))
        threshold = data[idxD]
        return feature, threshold, len(data)

    # GROW step: randomly pick a terminal node and split into 2 new
    # ones by randomly assigning a splitting rule.  
    def grow(self):
        nodes = self.terminalNodes
        rnode = nodes[np.random.randint(len(nodes))]
        feature, threshold, ndata_in_node = self.prule(rnode)
        if feature is None or threshold is None:
            return
        self.split(rnode, feature, threshold)

        return rnode

    def split(self, parent, feature, threshold):
        # Threshold is of length self.n_features
        nleft  = Node(parent, True)  # Add left node; it registers with parent
        nright = Node(parent, False) # Add right node; it registers with parent
        parent.setThreshold(feature, threshold)

        fxl, fyl = self.filter(nleft)
        fxr, fyr = self.filter(nright)
        
        # only split if it yields at least nmin points per child
        if nleft.npts >= self.nmin and nright.npts >= self.nmin:
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
        dparents = self.get_terminal_parents()
        if len(dparents) == 0:
            return
        parent   = dparents[np.random.randint(len(dparents))]
        # collapse node
        parent.Left = None
        parent.Right = None
        self.calcTerminalNodes()
        self.calcInternalNodes()

        return parent

    # Find the parents of each pair of terminal nodes.
    def get_terminal_parents(self):
        nodes = self.terminalNodes
        if len(nodes) == 0:
            return
        parents = [x.Parent for x in nodes]
        if len(parents) == 0:
            return
        # make sure there are 2 terminal children
        dparents = [x for x, y in collections.Counter(parents).items() if y == 2]

        return dparents

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

    def plinko(self, node, data):

        includeX = np.ones(data.shape, dtype=np.bool)
        n = node
        while n.Parent is not None:
            if n.is_left:
                includeX[:,n.Parent.feature] &= data[:,n.Parent.feature] <=  n.Parent.threshold
            else:
                includeX[:,n.Parent.feature] &= data[:,n.Parent.feature] >   n.Parent.threshold
            n = n.Parent
        return includeX
        
    
    # Filter the data that end up in each (terminal) node; return
    # their locations
    def filter(self, node):
        includeX = self.plinko(node, self.X)
        includeY = np.all(includeX, axis=1)

        # Set the node values
        node.ybar = np.mean(self.y[includeY])
        node.yvar = np.std(self.y[includeY])**2
        node.npts = np.sum(includeY)

        return includeX, includeY


class BartProposal(proposals.Proposal):
    def __init__(self):
        self.pgrow = 0.5
        self._operation = None  # Last tree operations performed (Grow/Prune)
        self._node = None  # Last node operated on

    def draw(self, current_tree):
        # make a copy since the grow/prune operations operate on the tree object in place
        new_tree = copy.deepcopy(current_tree)

        prop = np.random.uniform()
        if prop < self.pgrow:
            self._node = new_tree.grow()
            self._operation = 'grow'
        else:
            self._node = new_tree.prune()
            self._operation = 'prune'

        return new_tree

    def logdensity(self, proposed_tree, current_tree, forward):
        if not forward:
            # only do calculation for forward transition, since factors cancel in MH ratio
            return 0.0
        # Compute ratio of prior distributions for the two trees. Do this here instead of in the Tree parameter object
        # because many of the factors cancel in the Metropolis-Hastings ratio for this type of proposal.
        alpha = current_tree.alpha
        beta = current_tree.beta
        depth = self._node.depth
        log_prior_ratio = np.log(alpha) - beta * np.log(1.0 + depth) - np.log(1.0 + alpha / (1.0 + depth) ** beta) + \
            2.0 * np.log(1.0 - alpha / (2.0 + depth) ** beta)

        # get log ratio of transition kernels
        if self._operation == 'grow':
            ntnodes = len(current_tree.terminalNodes)
            ntparents = len(proposed_tree.get_terminal_parents())
            logdensity = np.log(ntnodes / ntparents) + log_prior_ratio
        elif self._operation == 'prune':
            ntnodes = len(proposed_tree.terminalNodes)
            ntparents = len(current_tree.get_terminal_parents())
            logdensity = np.log(ntparents) - log_prior_ratio
        else:
            print 'Unknown proposal move.'
            return 0.0

        return logdensity

    def __call__(self, tree):
        prop = np.random.uniform()
        if prop < 0.50:
            #print "# GROW",
            tree.grow()
        else:
            #print "# PRUNE",
            tree.prune()


class BartTree(BaseTree):
    def __init__(self, X, y, alpha, beta, nu, q, a):
        BaseTree.__init__(self, X, y)

        self.buildUniform(self.head, alpha, beta)

class BartTrees(object):
    def __init__(self, X, y, m=200, alpha=0.95, beta=2.0):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.n_samples  = X.shape[0]

        if True:
            sigma = np.std(self.y)
        else:
            regressor = linear_model.Lasso(normalize=True, fit_intercept=True)
            fit       = regressor.fit(X, y)
            sigma     = np.mean(fit.predict(X) - y)
        self.sigsqr = sigma**2

        # Hyperparameters for growing the trees.  Keep them more
        # compact than CART since there are more of them
        self.alpha = alpha
        self.beta = beta

        # Rescale y to lie between -0.5 and 0.5
        self.y -= self.y.min() # minimum = 0
        self.y /= self.y.max() # maximum = 1
        self.y -= 0.5          # range is -0.5 to 0.5

        self.nu    = 3.0  # Degrees of freedom for error variance prior; should always be > 3
        self.q     = 0.90 # The quantile of the prior that the sigma2 estimate is placed at
        qchi       = stats.chi2.interval(self.q, self.nu)[1]
        self.lamb  = self.sigsqr * qchi / self.nu
 
        self.k = 2       # Hyperparameter that yields 95% probability that E(Y|x) is in interval ymin, ymax
        self.m = m       # Number of trees

        self.mumu  = 0.0
        self.sigmu = 0.5 / self.k / np.sqrt(self.m)  
        self.a     = 1.0 / (self.sigmu**2)

        self.trees = []
        for m in range(self.m):
            self.trees.append(BartTree(self.X, self.y, self.alpha, self.beta, self.nu, self.q, self.a))

    @staticmethod
    def node_mu(tree):
        ybarmap = np.zeros(tree.n_samples)
        for node in tree.terminalNodes:
            # NOTE: this should grab the model parameters, not the empirical means
            y_in_node = tree.filter(node)[1]
            assert(np.all(ybarmap[y_in_node] == 0.0))
            ybarmap[y_in_node] = node.ybar
        return ybarmap

    def calcResids(self):
        node_mus = np.zeros((self.n_samples, self.m))
        for m in range(self.m):
            node_mus[:,m] = self.node_mu(self.trees[m])

        treesum = np.sum(node_mus, axis=1)
        for m in range(self.m):
            resids = self.y - treesum + node_mus[:,m]

            # Update
            self.trees[m].y = resids

            # Proposal updates the tree
            # TBD

            # Updated tree sum
            pred = self.node_mu(self.trees[m])
            treesum += (pred - node_mus[:,m])

            # Make sure the node_mus matrix is updated along with the tree
            node_mus[:,m] = pred

        return 

    def predict(self, data):
        # data needs to be shape (self.npredict, self.nfeatures)
        assert (data.shape[1] == self.n_features)
        n_predict = data.shape[0]

        node_mus = np.zeros((n_predict, self.m))
        for m in range(self.m):
            tree     = self.trees[m]
            ybarmap  = np.zeros(n_predict)
            for node in tree.terminalNodes:
                y_in_node = np.all(tree.plinko(node, data), axis=1)
                assert(np.all(ybarmap[y_in_node] == 0.0))
                ybarmap[y_in_node] = node.ybar  # note, this should really use the model parameter
            node_mus[:,m] = ybarmap
        return node_mus

####
#################
####

class CartProposal(object):
    def __init__(self):
        pass

    def __call__(self, tree):
        prop = np.random.uniform()
        if prop < 0.50:
            #print "# GROW",
            tree.grow()
        else:
            #print "# PRUNE",
            tree.prune()
        #elif prop < 0.75:
        #    print "# CHANGE",
        #    tree.change()
        #else:
        #    print "# SWAP",
        #    tree.swap()


class BartTreeParameter(steps.Parameter):

    def __init__(self, name, X, y, mtrees, alpha=0.95, beta=2.0, track=True):
        """
        Constructor for Bart tree configuration parameter class. The tree configuration is treated as a Parameter object
        to be sampled using a MCMC sampler. The 'value' of this parameter is an instance of BaseTree, which is updated
        throughout the MCMC sampler.

        @param name: A string containing the name of the Tree. Used as a key to identify this particular tree.
        @param X: The predictors, an array of shape (n,p).
        @param y: The array of response values, of size n.
        @param mtrees: The number of trees used in the BART model
        @param alpha: Prior parameter on the tree shape, same the notation of Chipman et al. (2010).
        @param beta: Prior parameter controlling the tree depth, same notation of Chipman et al. (2010).
        @param track: When this parameter is tracked (i.e., whether the values are saved) in the MCMC sampler.
        """
        super(BartTreeParameter, self).__init__(name, track)

        self.X = X
        self.y = y

        self.value = BaseTree(X, y)  # parameter 'value' is the tree configuration

        # Setup up the prior distribution
        self.k = 2  # Hyperparameter that yields 95% probability that E(Y|x) is in interval ymin, ymax
        self.mubar = 0.0  # shrink values of mu for each terminal node toward zero
        self.mtrees = mtrees  # the number of trees in the BART model
        self.k = 2.0  # parameter controlling prior variance, i.e., shrinkage amplitude
        self.prior_mu_var = 1.0 / (2.0 * self.k * self.k * self.mtrees)

        self.alpha = alpha
        self.beta = beta

        # Must set this manually before running the MCMC sampler. Necessary because log-likelihood depends on the value
        # of sigma^2.
        self.sigsqr = None  # the instance of BartVariance class for this model.

    def set_starting_value(self):
        try:
            self.sigsqr is not None
        except ValueError:
            "Value of error variance is not set."
        # draw initial tree configuration from the prior
        self.value.buildUniform(self.value.head, self.alpha, self.beta)

    def logprior(self, tree):
        """
        Compute the log-prior for a input tree configuration. This assumes that the only difference between the input
        tree and self is in the structure of the tree nodes. The prior distribution is assumed to be the same.

        @param tree: The proposed tree configuration object.
        @return: The log-prior density of tree.
        """
        logprior = 0.0
        # first get prior for terminal nodes
        for node in tree.terminalNodes:
            # probability of not splitting
            logprior += np.log(1.0 - self.alpha / (1.0 + node.depth) ** self.beta)

        # now get contribution from interior nodes
        for node in tree.internalNodes:
            # probability of splitting this node
            logprior += np.log(self.alpha) - self.beta * np.log(1.0 + node.depth)

            # get number of features and data points that are available for the splitting rule
            fxl, fyl = tree.filter(node)
            nfeatures = np.sum(np.sum(fxl, axis=0) > 1)  # need at least one data point for a split on a feature
            npts = np.sum(fyl)
            # probability of split is discrete uniform over set of available features and data points
            logprior += -np.log(nfeatures) - np.log(npts)

        return logprior

    # NOTE: This part would likely benefit from numba or cython
    def loglik(self, tree):
        """
        Compute the marginal log-likelihood for an input tree configuration. This assumes that the only difference
        between the input tree and self is in the structure of the tree nodes. The prior and data are assumed to be the
        same. Note that the return value is the log-likelihood after marginalizing over mean value parameters in each
        terminal node.

        @param tree: The input tree configuration object.
        @return: The marginal log-likelihood of the tree configuration.
        """
        lnlike = 0.0

        # Precalculate terms
        t2  = np.log((self.nu * self.lamb)**(0.5 * self.nu))
        t4b = gammaln(0.5 * self.nu)

        for node in tree.terminalNodes:

            npts = node.npts
            if npts == 0:
                # Damn, this should not happen.
                # DEBUG ME
                continue

            ymean = node.ybar

            # log-likelihood component after marginalizing over the mean value in each node, a gaussian distribution
            post_var = self.prior_mu_var + self.sigsqr.value / npts
            zsqr = (ymean - self.mubar) ** 2 / post_var

            lnlike += -0.5 * np.log(self.sigsqr.value / npts) - 0.5 * zsqr

        return lnlike

    def logdensity(self, tree):
        loglik = self.loglik(tree)
        return loglik  # ignore prior contribution since factors cancel and we account for this in BartProposal class


class BartMeanParameter(steps.Parameter):

    def __init__(self, name, mtrees, track=True):
        """
        Constructor for Parameter class corresponding to the mean values of the response in each terminal node of a
        BART model.

        @param name: A string specifying the name of this parameters, used as a key in distinguishing parameters.
        @param mtrees: The number of trees in the BART model. The prior depends on this.
        @param track: Whether the parameter values should be tracked and saved by the MCMC sampler, a boolean.
        """
        super(BartMeanParameter, self).__init__(name, track)
        # Set prior parameters
        self.mubar = 0.0  # prior mean
        self.mtrees = mtrees  # the number of trees in the BART model
        self.k = 2.0  # parameter controlling prior variance, i.e., shrinkage amplitude
        self.prior_var = 1.0 / (2.0 * self.k * self.k * self.mtrees)

        # Must set these manually before running the MCMC sampler. Necessary because Gibbs updates need to know the
        # values of the other parameters.
        self.tree = None  # the instance of BartTreeParameter class corresponding to this mean parameter instance
        self.sigsqr = None  # the instance of BartVariance class for this model

    def set_starting_value(self, tree):
        try:
            self.tree is not None
        except ValueError:
            "Tree configuration is not set."
        try:
            self.sigsqr is not None
        except ValueError:
            "Value of error variance is not set."

        self.value = self.random_posterior()

    def random_posterior(self):
        """
        Update the mean y parameter value for each terminal node by drawing from its distribution, conditional on the
        current tree configuration, variance (sigma ** 2), and data.
        """
        mu = np.empty(len(self.tree.terminalNodes))
        n_idx = 0
        for node in self.tree.terminalNodes:
            if node.npts == 0:
                # Damn, this should not happen.
                # DEBUG ME
                continue
            ny_in_node = node.npts
            ymean_in_node = node.ybar

            post_var = 1.0 / (1.0 / self.prior_var + ny_in_node / self.sigsqr.value)
            post_mean = post_var * ny_in_node * ymean_in_node / self.sigsqr.value

            mu[n_idx] = np.random.normal(post_mean, np.sqrt(post_var))
            n_idx += 1

        return mu


class BartVariance(steps.Parameter):

    def __init__(self, X, y, name='sigsqr', track=True):
        super(BartVariance, self).__init__(name, track)
        self.y = y

        # set prior parameter values
        use_naive_prior = True
        if use_naive_prior:
            sigma_hat = np.std(self.y)
        else:
            regressor = linear_model.LassoCV(normalize=True, fit_intercept=True)
            fit = regressor.fit(X, y)
            sigma_hat = np.std(fit.predict(X) - y)
        # These value of sigma_hat should be used to estimate nu and q.
        self.nu = 3.0  # Degrees of freedom for error variance prior; should always be > 3
        self.q = 0.90  # The quantile of the prior that the sigma2 estimate is placed at

        qchi = stats.chi2.interval(self.nu, self.q)[1]
        # scale parameter for error variance scaled inverse-chi-square prior
        self.lamb = sigma_hat ** 2 * qchi / self.nu

        # Must set these manually before running the MCMC sampler. Necessary because Gibbs updates need to know the
        # values of the other parameters.
        self.trees = None  # the ensemble of BartTreeParameter objects for this model
        self.mus = None  # the ensemble of BartMeanParameter objects for this model

    def set_starting_value(self):
        try:
            self.tree is not None
        except ValueError:
            "Parameter for tree configuration is not set."
        try:
            self.mu is not None
        except ValueError:
            "Parameter for mean response in terminal nodes is not set."

        return self.random_posterior()

    def random_posterior(self):

        # resid = self.y - self.trees.ypredict()
        ypredict = np.zeros(len(self.y))
        for tree, mu in self.trees, self.mus:
            n_idx = 0
            for node in tree.terminalNodes:
                x_filtered = tree.filter(node)
                y_filtered = np.all(x_filtered, axis=1)
                ypredict[y_filtered] += mu.value[n_idx]
                n_idx += 1

        resid = self.y - ypredict
        ssqr = np.var(resid)

        post_dof = self.nu + len(self.y)
        post_ssqr = (len(self.y) * ssqr + self.nu * self.lamb) / post_dof

        # new error variance is drawn from scaled inverse-chi-square distribution
        new_sigsqr = post_dof * post_ssqr / np.random.chisquare(post_dof)

        return new_sigsqr


class CartTree(BaseTree, steps.Parameter):
    # Describes the conditional distribution of y given X.  X is a
    # vector of predictors.  Each terminal node has parameter Theta.
    # 
    # y|X has distribution f(y|Theta).

    def __init__(self, X, y, nu, lamb, mubar, a, name, track=True, alpha=0.95, beta=1.0, min_samples_leaf=5):
        BaseTree.__init__(self, X, y, min_samples_leaf)

        # Tuning parameters of the model
        self.nu     = nu
        self.lamb   = lamb
        self.mubar  = mubar
        self.a      = a
        self.alpha  = alpha
        self.beta   = beta
        self.mu     = np.empty(1)
        self.sigsqr = 1.0

        # Calls set_starting_value, which requires we have member variables defined
        steps.Parameter.__init__(self, name, track)

    def set_starting_value(self):
        """
        Set the initial configuration of the tree, just draw from its prior distribution.
        """
        self.buildUniform(self.head, self.alpha, self.beta)

    def logprior(self, tree):
        """
        Compute the log-prior for a proposed tree model. This assumes that the only difference between the input
        tree and self is in the structure of the tree nodes. The prior distribution is assumed to be the same.

        @param tree: The proposed tree.
        @return: The log-prior density of tree.
        """
        logprior = 0.0
        # first get prior for terminal nodes
        for node in tree.terminalNodes:
            # probability of not splitting
            logprior += np.log(1.0 - self.alpha / (1.0 + node.depth) ** self.beta)

        # now get contribution from interior nodes
        for node in tree.internalNodes:
            # probability of splitting this node
            logprior += np.log(self.alpha) - self.beta * np.log(1.0 + node.depth)

            # get number of features and data points that are available for the splitting rule
            fxl, fyl = tree.filter(node)
            nfeatures = np.sum(np.sum(fxl, axis=0) > 1)  # need at least one data point for a split on a feature
            npts = node.npts
            # probability of split is discrete uniform over set of available features and data points
            logprior += -np.log(nfeatures) - np.log(npts)

        return logprior

    # NOTE: This part would likely benefit from numba or cython
    def loglik(self, tree=None):
        """
        Compute the marginal log-likelihood for a proposed tree model. This assumes that the only difference between the
        input tree and self is in the structure of the tree nodes. The prior and data are assumed to be the same. Note
        that the return value is the log-likelihood after marginalizing over mean value parameters in each terminal
        node.

        @param tree: The proposed tree.
        @return: The log-likelihood of tree.
        """
        if tree is None:
            tree = self

        lnlike = 0.0

        # Precalculate terms
        t2  = np.log((self.nu * self.lamb)**(0.5 * self.nu))
        t4b = gammaln(0.5 * self.nu)

        # Random draws for mean-variance shift model.  NOTE: these are
        # unncessary, these distributions are marginalized over.
        #sigsq   = stats.invgamma.rvs(0.5 * self.nu, scale = 0.5 * self.nu * self.lamb)
        #mui     = stats.norm.rvs(self.mubar, scale = sigsq / self.a)
        
        for node in tree.terminalNodes:
            fxl, fyl = tree.filter(node)
            if node.npts == 0:
                # Damn, this should not happen.
                # DEBUG ME
                continue

            ymean = node.ybar
            yvar = node.yvar
            npts = node.npts

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

    def logdensity(self, tree):
        loglik = self.loglik(tree)
        return loglik  # ignore prior contribution since factors cancel and we account for this in BartProposal class

    def update_mu(self):
        """
        Update the mean y parameter value for each terminal node by drawing from its distribution, conditional on the
        current tree configuration, variance (sigma ** 2), and data.
        """
        self.mu = np.empty(len(self.terminalNodes))
        n_idx = 0
        for node in self.terminalNodes:
            x_in_node, y_in_node = self.filter(node)  # boolean values
            if node.npts == 0:
                # Damn, this should not happen.
                # DEBUG ME
                continue
            ny_in_node = node.npts
            ymean_in_node = node.ybar

            post_var = 1.0 / (1.0 / self.prior_mu_var + ny_in_node / self.sigsqr)
            post_mean = post_var * ny_in_node * ymean_in_node / self.sigsqr

            self.mu[n_idx] = np.random.normal(post_mean, np.sqrt(post_var))
            n_idx += 1


class Node(object):
    NodeId = 0

    def __init__(self, parent, is_left):
        self.Id        = Node.NodeId
        Node.NodeId   += 1

        self.Parent    = parent # feature and threshold reside in the parent
        self.Left      = None   # data[:, feature] <= threshold
        self.Right     = None   # data[:, feature] > threshold
        self.setThreshold(None, None)

        if self.Parent is not None:
            self.is_left = is_left
            if self.is_left:
                parent.Left = self
            else:
                parent.Right = self
            self.depth = self.Parent.depth + 1
        else:
            self.depth = 0

        # Moments of the data that end up in this bin
        self.ybar = 0.0
        self.yvar = 0.0
        self.npts = 0

    # NOTE: the parent carries the threshold
    def setThreshold(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold


        

if __name__ == "__main__":
    nsamples  = 1000
    nfeatures = 20
    X    = np.random.random((nsamples, nfeatures)) - 0.5
    y    = np.random.random((nsamples)) - 0.5
    tree = CartTree(X, y, nu=0.1, lamb=2/0.1, mubar=np.mean(y), a=1.0, name=None, alpha=0.99, beta=1.0/np.log(nsamples))
    prop = CartProposal()
    tree.printTree(tree.head)
    #for i in range(10000):
    #    prop(tree)
    #    print tree.loglik()

    print "Terminal", [x.Id for x in tree.terminalNodes]
    print "Internal", [x.Id for x in tree.internalNodes]

    tree = BartTrees(X, y)
    #tree.calcResids()
    npredict = 70
    Xp   = np.random.random((npredict, nfeatures)) - 0.5
    tree.predict(Xp)

    #for i in range(10):
    #    tree.regressionLnlike()

    
