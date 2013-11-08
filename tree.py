import numpy as np
import collections
import scipy.stats as stats
from scipy.special import gammaln
import steps
import samplers
import proposals
import copy
from sklearn import linear_model


class Node(object):
    NodeId = 0

    def __init__(self, parent, is_left):
        """
        Constructor for a node in a binary tree.

        @param parent: The parent node.
        @param is_left: If true, then this node is the left node in the binary split from the parent node.
        """
        self.Id        = Node.NodeId
        Node.NodeId   += 1

        self.Parent    = parent # feature and threshold reside in the parent
        self.Left      = None   # data[:, feature] <= threshold
        self.Right     = None   # data[:, feature] > threshold

        if self.Parent is not None:
            self.is_left = is_left
            if self.is_left:
                parent.Left = self
            else:
                parent.Right = self
            self.depth = self.Parent.depth + 1
        else:
            self.is_left = None
            self.depth = 0

        self._ybar = 0.0
        self._yvar = 0.0
        self._npts = 0

        # NOTE: the parent carries the feature and threshold to split upon
        self._feature = -1
        self._threshold = 0.0

    def getybar(self):
        return self._ybar
    def setybar(self, value):
        self._ybar = value
    def delybar(self):
        del self._ybar
    ybar = property(getybar, setybar, delybar, "0th moment of data that end up in this node")

    def getyvar(self):
        return self._yvar
    def setyvar(self, value):
        self._yvar = value
    def delyvar(self):
        del self._yvar
    yvar = property(getyvar, setyvar, delyvar, "Squared 1st moment of data that end up in this node")

    def getnpts(self):
        return self._npts
    def setnpts(self, value):
        self._npts = value
    def delnpts(self):
        del self._npts
    npts = property(getnpts, setnpts, delnpts, "Number of data points that end up in this node")

    def getfeat(self):
        return self._feature
    def setfeat(self, value):
        self._feature = value
    def delfeat(self):
        del self._feature
    feature = property(getfeat, setfeat, delfeat, "The binary split will be on this feature")

    def getthresh(self):
        return self._threshold
    def setthresh(self, value):
        self._threshold = value
    def delthresh(self):
        del self._threshold
    threshold = property(getthresh, setthresh, delthresh, "The value of the feature seperating the left and right nodes")


class BaseTree(object):

    def __init__(self, X, y, min_samples_leaf=5):
        """
        Constructor for the Base class for binary trees. This class contains methods that provide the functionality
        needed for building and describing a binary tree configuration.

        @param X: The array of feature values, shape (n_samples,n_features).
        @param y: The array of response values, size n_samples.
        @param min_samples_leaf: The minimum number of data points within a leaf (terminal node).
        """
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
        """
        Randomly split the input node into two new nodes. This method is useful for generating a tree configuration
        using node as the trunk by drawing the split probabilities from the distribution defined by alpha and beta.

        @param node: The trunk (head) of the tree. All generated nodes will fall below this node.
        @param alpha: Parameter defining probability of a split, same notation as Chipman et al. (2010).
        @param beta: Parameter defining probability of a split as a funtion of node depth, same notation as
            Chipman et al. (2010).
        @param depth: The depth of the current node.
        @return:
        """
        assert(beta > 0.0)
        assert(alpha < 1.0)
        assert(alpha >= 0.0)
        psplit = alpha * (1 + depth)**-beta
        rand   = np.random.uniform()
        if rand < psplit:
            feature, threshold = self.prule(node)
            if feature is None or threshold is None:
                print "NO DATA LEFT, rejecting split"
                return
            nleft, nright = self.split(node, feature, threshold)
            if (nleft is not None) and (nright is not None):
                print "EXTENDING node", node.Id, "to depth", depth+1, "(%.2f < %.2f)" % (rand, psplit)
                self.buildUniform(nleft, alpha, beta, depth=depth+1)
                self.buildUniform(nright, alpha, beta, depth=depth+1)
            else:
                print "NOT EXTENDING node", node.Id, ": too few points (%d=%.2f)" % (feature, threshold)
        else:
            print "NOT SPLITTING node", node.Id, ": did not pass random draw (%.2f > %.2f at depth %d)" % (rand, psplit, depth)
            
    def prule(self, node):
        """
        Split the node by implementing a uniform draw from the features to split on, and then choose the split value
        uniformly from the set of available observed values.  NOTE: do not change the node attributes here since this
        may be rejected.

        @param node: The node object on which to perform the split.
        """
        feature = np.random.randint(self.n_features)
        idxX, idxY = self.filter(node)
        data = self.X[:, feature][idxX[:, feature]]
        if len(data) == 0:
            return None, None, None
        idxD = np.random.randint(len(data))
        threshold = data[idxD]
        return feature, threshold

    def grow(self):
        """
        Grow a pair of terminal nodes by randomly picking a terminal node and splitting it into two new ones by randomly
        assigning a splitting rule.

        @return: The node that was chosen to be the parent of the new terminal nodes.
        """
        nodes = self.terminalNodes
        rnode = nodes[np.random.randint(len(nodes))]
        feature, threshold = self.prule(rnode)
        if feature is None or threshold is None:
            return None
        self.split(rnode, feature, threshold)

        return rnode

    def split(self, parent, feature, threshold):
        """
        Split a node on the input feature and using the input threshold.

        @param parent: The node to split.
        @param feature: The feature to split the node on.
        @param threshold: The threshold value of feature for the split.
        @return: The new left and right node objects resulting from the split.
        """
        if parent.Left is not None or parent.Right is not None:
            return None, None

        nleft  = Node(parent, True)  # Add left node; it registers with parent
        nright = Node(parent, False) # Add right node; it registers with parent
        parent.feature = feature
        parent.threshold = threshold

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
            parent.feature = None
            parent.threshold = None
            parent.Left = None
            parent.Right = None
            return None, None

    def prune(self):
        """
        Prune the tree by randomly picking a parent of two terminal nodes and then collapsing the terminal nodes into
        the parent.

        @return: The parent node, i.e., the new terminal node.
        """
        dparents = self.get_terminal_parents()
        if len(dparents) == 0:
            return None
        parent   = dparents[np.random.randint(len(dparents))]
        # collapse node
        parent.Left = None
        parent.Right = None
        self.calcTerminalNodes()
        self.calcInternalNodes()

        return parent

    def get_terminal_parents(self):
        """
        Find the parents of each pair of terminal nodes.

        @return: The list of parent nodes of each pair of terminal nodes.
        """
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

    def calcTerminalNodes(self):
        """
        Calculate the terminal nodes of the tree.
        """
        self.terminalNodes = []
        self.calcTerminalNodes_(self.head)

    def calcTerminalNodes_(self, node):
        if node.Right is None or node.Left is None:
            self.terminalNodes.append(node)
        if node.Right is not None:
            self.calcTerminalNodes_(node.Right)
        if node.Left is not None:
            self.calcTerminalNodes_(node.Left)

    def calcInternalNodes(self):
        """
        Calculate the internal nodes of the tree.
        """
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
        """
        Return the indices of the X-values that end up in the input node.

        @param node: The node for which the data point indices are desired.
        @param data: The array of predictors.
        @return: The indices of the predictors in this node.
        """
        includeX = np.ones(data.shape, dtype=np.bool)
        n = node
        while n.Parent is not None:
            if n.is_left:
                includeX[:, n.Parent.feature] &= data[:, n.Parent.feature] <= n.Parent.threshold
            else:
                includeX[:, n.Parent.feature] &= data[:, n.Parent.feature] > n.Parent.threshold
            n = n.Parent

        return includeX


    def filter(self, node):
        """
        Find the data points that end up in the input node by dropping them down the tree, and save the first and
        second moments of the y-values in this node.

        @param node: The node for which the data points are desired.
        @return: A tuple of two boolean arrays indicating whether the a data point ends up in the input node.
        """
        includeX = self.plinko(node, self.X)
        includeY = np.all(includeX, axis=1)
        if not True in includeY:
            return includeX, includeY

        # Set the node values
        node.ybar = np.mean(self.y[includeY])
        node.yvar = np.var(self.y[includeY])
        node.npts = np.sum(includeY)

        return includeX, includeY  # TODO: do we really need to return includeX?



class BartTreeParameter(steps.Parameter):

    def __init__(self, name, X, y, mtrees, alpha=0.95, beta=2.0, prior_mu=0.0, prior_var=2.0, track=True):
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
        @param prior_mu: The prior mean for the terminal node mean parameters.
        @param prior_var: The prior variance for the terminal node mean parameters.
        @param track: When this parameter is tracked (i.e., whether the values are saved) in the MCMC sampler.
        """
        super(BartTreeParameter, self).__init__(name, track)

        self.X = X
        self.y = y

        self.value = BaseTree(X, y)  # parameter 'value' is the tree configuration

        # Setup up the prior distribution
        self.mtrees = mtrees  # the number of trees in the BART model
        self.mubar = prior_mu
        self.prior_mu_var = prior_var

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
            logprior += -np.log(self.value.n_features) - np.log(node.npts)
            if node.npts < 2:
                # should never happen
                logprior = -1e600

        return logprior

    # NOTE: This part would likely benefit from numba or cython
    def loglik(self, tree):
        """
        Compute the marginal log-likelihood for an input tree configuration. This assumes that the only difference
        between the input tree and self is in the structure of the tree nodes. The prior and data are assumed to be the
        same. Note that the return value is the log-likelihood after marginalizing over the mean value parameters in
        each terminal node.

        @param tree: The input tree configuration object.
        @return: The marginal log-likelihood of the tree configuration.
        """
        lnlike = 0.0

        for node in tree.terminalNodes:

            npts = node.npts
            if npts == 0:
                # empty nodes do not contribute to the log-likelihood
                continue

            ymean = node.ybar
            yvar = node.yvar

            # log-likelihood component after marginalizing over the mean value in each node, a gaussian distribution
            post_var = self.prior_mu_var + self.sigsqr.value / npts
            zsqr = (ymean - self.mubar) ** 2 / post_var

            lnlike += -(npts - 1.0) / 2.0 * np.log(2.0 * np.pi * self.sigsqr.value) - 0.5 * np.log(npts) - \
                0.5 * np.log(2.0 * np.pi * post_var) - 0.5 * zsqr - 0.5 * npts * yvar / self.sigsqr.value

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
        self.prior_var = 1.0 / (4.0 * self.k * self.k * self.mtrees)

        # Must set these manually before running the MCMC sampler. Necessary because Gibbs updates need to know the
        # values of the other parameters.
        self.tree = None  # the instance of BaseTree class corresponding to this mean parameter instance
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
            post_mean = post_var * (self.mubar / self.prior_var + ny_in_node * ymean_in_node / self.sigsqr.value)

            mu[n_idx] = np.random.normal(post_mean, np.sqrt(post_var))
            n_idx += 1

        return mu


class BartVariance(steps.Parameter):

    def __init__(self, X, y, name='sigsqr', track=True):
        """
        Constructor for the error variance parameter in the BART model.

        @param X: The array of features, shape (n_samples, n_features).
        @param y: The array of response values, size n_samples.
        @param name: The name of the parameter object, for bookkeeping purposes.
        @param track: A boolean, if true then we save the values in the MCMC sampler.
        """
        super(BartVariance, self).__init__(name, track)
        self.y = y

        # set prior parameter values
        use_naive_prior = False
        if use_naive_prior:
            sigma_hat = np.std(self.y)
        else:
            regressor = linear_model.LassoCV(normalize=True, fit_intercept=True)
            fit = regressor.fit(X, y)
            sigma_hat = np.std(fit.predict(X) - y)
        # These value of sigma_hat should be used to estimate nu and q.
        self.nu = 3.0  # Degrees of freedom for error variance prior; should always be > 3
        self.q = 0.90  # The quantile of the prior that the sigma2 estimate is placed at

        qchi = stats.chi2.interval(self.q, self.nu)[1]
        # scale parameter for error variance scaled inverse-chi-square prior
        self.lamb = sigma_hat ** 2 * qchi / self.nu

        # Must set these manually before running the MCMC sampler. Necessary because Gibbs updates need to know the
        # values of the other parameters.
        self.bart_step = None  # the BartStep object for this model

    def set_starting_value(self):
        try:
            self.bart_step is not None
        except ValueError:
            "Parameter for tree ensemble update step is not set."

        return self.random_posterior()

    def random_posterior(self):
        """
        Obtain a random draw from the posterior distribution of the error variance, conditional on the BART tree
        ensemble.

        @return: A random draw from the conditional posterior of the error variance.
        """
        resid = self.bart_step.resids
        ssqr = np.var(resid)

        post_dof = self.nu + len(self.y)
        post_ssqr = (len(self.y) * ssqr + self.nu * self.lamb) / post_dof

        # new error variance is drawn from scaled inverse-chi-square distribution
        new_sigsqr = post_dof * post_ssqr / np.random.chisquare(post_dof)

        return new_sigsqr


class BartProposal(proposals.Proposal):
    def __init__(self, alpha=0.95, beta=2.0):
        """
        Constructor for object that generates proposed tree configurations, given the current one.
        """
        self.alpha = alpha
        self.beta = beta
        self.pgrow = 0.5  # probability of growing the tree instead of pruning the tree.
        self._operation = None  # Last tree operations performed (Grow/Prune)
        self._node = None  # Last node operated on
        self.log_prior_ratio = 0.0
        self._prohibited_proposal = False

    def draw(self, current_tree):
        """
        Generate a random proposed tree configuration from the input one.
        @param current_tree: The current tree configuration, an instance of the BaseTree class.
        @return: The proposed tree configuration, and instance of BaseTree.
        """
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
        """
        The log-probability of the tree configuration proposal kernel. This includes the contribution from the prior
        distribution in the marginal log-posterior of the tree configuration. In actuality, this return the logarithm of
        the ratio of a forward transition to a backward transition multiplied by the ratio of the priors.

        @param proposed_tree: The proposed tree configuration, an instance of the BaseTree class.
        @param current_tree: The current tree configuration, an instance of the BaseTree class.
        @param forward: A boolean indicating whether to calculate the forward transition or not. This is just for
            convenience, since only the forward transition is calculated for computational efficiency.
        @return: The logarithm of the ratio of the transition kernels and the priors.
        """
        if not forward:
            # only do calculation for forward transition, since factors cancel in MH ratio
            return 0.0
        # Compute ratio of prior distributions for the two trees. Do this here instead of in the Tree parameter object
        # because many of the factors cancel in the Metropolis-Hastings ratio for this type of proposal.

        if self._node is None or self._node.feature is None:
            # tree configuration is unchanged since we could not perform the chosen move
            self._prohibited_proposal = True
            return 0.0

        self._prohibited_proposal = False
        alpha = self.alpha
        beta = self.beta
        depth = self._node.depth
        log_prior_ratio = np.log(alpha) - beta * np.log(1.0 + depth) - np.log(1.0 - alpha / (1.0 + depth) ** beta) + \
            2.0 * np.log(1.0 - alpha / (2.0 + depth) ** beta)
        self.log_prior_ratio = log_prior_ratio

        # get log ratio of transition kernels
        if self._operation == 'grow':
            ntnodes = float(len(current_tree.terminalNodes))
            ntparents = len(proposed_tree.get_terminal_parents())
            logdensity = np.log(ntnodes / ntparents) + log_prior_ratio
        elif self._operation == 'prune':
            ntnodes = float(len(proposed_tree.terminalNodes))
            ntparents = len(current_tree.get_terminal_parents())
            logdensity = np.log(ntparents / ntnodes) - log_prior_ratio
        else:
            self._prohibited_proposal = True
            print 'Unknown proposal move.'
            return 0.0

        return -logdensity  # make sure sign agrees with expectation from MetroStep.accept()


class BartStep(object):

    def __init__(self, y, trees, mus, report_iter=-1):
        """
        Constructor for the MCMC step that updates the BART tree ensemble. Each tree in the ensemble is updated one-at-
        a-time by performing a scan through the individual trees, whereby each tree configuration is first updated using
        a Metropolis-Hastings step, and then the mean values in each terminal node are updated using a Gibbs step.

        @param y: The array of response values, an n_samples size array.
        @param trees: The list of tree configurations, instances of the BaseTree class.
        @param mus: The list of mean values for the terminal nodes of each tree, instances of the BartMeanParameter
            class.
        @param report_iter: Print out a report on the Metropolis-Hastings acceptance rates after this many iterations.
        """
        self.y = y
        self.resids = y
        self.trees = trees
        self.mus = mus
        self._report_iter = report_iter
        self.tree_proposal = BartProposal()  # object to generate a new tree configuration from the current one
        # Objects to perform a Metropolis-Hasting update of the tree configuration for each tree
        self.tree_steps = [steps.MetroStep(tree, self.tree_proposal, self._report_iter) for tree in self.trees]

    @staticmethod
    def node_mu(tree, mu):
        """
        Generating an array mapping the index of the response (y) values to the values predicted by each terminal node
        in the input tree.

        @param tree: The tree configuration used to predict the y values, an instance of the BaseTree class.
        @param mu: The mean value of the terminal nodes that predict the y values, an instance of the BartMeanParameter
            class.
        @return: A array of size n_samples, with each element containing the predicted y value from the input tree.
        """
        try:
            len(tree.terminalNodes) == len(mu.value)
        except ValueError:
            "Number of terminal nodes does not equal number of mu values."

        mu_map = np.zeros(tree.n_samples)
        n_idx = 0
        for node in tree.terminalNodes:
            y_in_node = tree.filter(node)[1]
            assert(np.all(mu_map[y_in_node] == 0.0))
            mu_map[y_in_node] = mu.value[n_idx]
            n_idx += 1

        return mu_map

    def predict(self, data):
        # TODO: Should move this to the BartSample class.
        """
        Predict the value of the response given the input data.

        @param data: The array of predictors. Must have the same number of features (columns) as the training data.
        @return: The predicted value(s) at the input data.
        """
        # data needs to be shape (self.npredict, self.nfeatures)
        assert (data.shape[1] == self.n_features)
        n_predict = data.shape[0]
        node_mus = np.zeros((n_predict, self.m))

        for m in range(self.m):
            tree = self.trees[m]
            mu = self.mus[m]
            n_idx = 0
            for node in tree.terminalNodes:
                y_in_node = np.all(tree.plinko(node, data), axis=1)
                assert(np.all(node_mus[y_in_node, m] == 0.0))
                node_mus[y_in_node, m] = mu.value[n_idx]
                n_idx += 1
        return node_mus

    def do_step(self):
        """
        Update of the configurations and mean parameters of the terminal nodes of each tree in the ensemble. Note that
        this is done in place.
        """
        node_mus = np.zeros((self.n_samples, self.m))
        for m in range(self.m):
            node_mus[:, m] = self.node_mu(self.trees[m], self.mus[m])

        # predicted y is a sum of trees
        treesum = np.sum(node_mus, axis=1)
        for m in range(self.m):
            # leave-one-out residuals
            resids = self.y - (treesum - node_mus[:, m])

            # make leave-one-out resids the new response for the left-out tree
            self.trees[m].y = resids

            # First update the tree configuration using a Metropolis-Hastings step
            self.tree_steps[m].do_step()

            # Now update the mu values in the terminal nodes for this tree, do a Gibbs update
            self.mus[m].value = self.mus[m].random_posterior()

            # Updated tree sum
            pred = self.node_mu(self.trees[m], self.mus[m])
            treesum += (pred - node_mus[:, m])

            # Make sure the node_mus matrix is updated along with the tree
            node_mus[:, m] = pred

            self.trees[m].y = self.y  # restore original y-values

        self.resids = self.y - treesum  # save residuals for use by variance parameter object


class BartModel(samplers.Sampler):
    def __init__(self, X, y, m=200, alpha=0.95, beta=2.0):
        """
        Constructor for BART model class. This class will build the BART model and run the MCMC sampler based on this
        model, enabling Bayesian inference.

        @param X: The array of measured features, of shape (n_samples, n_features).
        @param y: The array of measured response values, size n_samples.
        @param m: The number of trees in the model.
        @param alpha: A tree configuration prior parameter, controlling the probability of a terminal node splitting. In
            the notation of Chipman et al. (2010).
        @param beta: A tree configuration prior parameter, controlling the probability of a terminal node splitting
            given its depth. In the notationof Chipman et al. (2010).
        """
        super(BartModel, self).__init__()
        delattr(self, 'mcmc_samples')  # can't store values in instance of MCMCSample class for BART, so remove it

        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        # Hyperparameters for growing the trees.  Keep them more
        # compact than CART since there are more of them
        self.m = m
        self.alpha = alpha
        self.beta = beta

        # store original y range so we can transform back when making predictions
        self.ymin = self.y.min()
        self.ymax = self.y.max()

        # Rescale y to lie between -0.5 and 0.5
        self.y -= self.ymin  # minimum = 0
        self.y /= self.ymax  # maximum = 1
        self.y -= 0.5        # range is -0.5 to 0.5

        # Build the ensemble of tree configurations and mu values for the terminal nodes
        self.trees = []
        self.mus = []
        for m in range(self.m):
            bname = 'BART ' + str(m + 1)
            mname = 'Mu ' + str(m + 1)
            self.mus.append(BartMeanParameter(mname, self.m))
            self.trees.append(BartTreeParameter(bname, self.X, self.y, self.m, alpha=self.alpha, beta=self.beta,
                                                prior_mu=self.mus[m].mubar, prior_var=self.mus[m].prior_var))

        # Create the variance parameter object
        self.sigsqr = BartVariance(self.X, self.y)

        # now construct the MCMC sampler: a sequence of steps
        self._build_sampler()

        prior_info = {'alpha': self.alpha, 'beta': self.beta, 'prior_mean': self.mus[0].mubar,
                      'prior_var': self.mus[0].prior_var, 'lamb': self.sigsqr.lamb, 'nu': self.sigsqr.nu}
        self.mcmc_samples = BartSample(X, y, m, prior_info)  # store MCMC samples in instance of BartSample class

        self._logliks = []

    def _build_sampler(self):
        # First do a Gibbs update for the variance parameter
        """
        Internal method for building the MCMC sampler. The MCMC sampler consists of a Gibbs update on the variance
        parameter, followed by a series of updates to the tree configuration and terminal node mean parameters for each
        tree in the ensemble.
        """
        self.add_step(steps.GibbStep(self.sigsqr))

        # Alternate between updating the tree configuration and then the terminal node means for each tree
        self.add_step(BartStep(self.y, self.trees, self.mus))

    def _allocate_arrays(self):
        """
        Build dictionary of saved values from MCMC sampler. This dictionary is stored in an instance of BartSample
        class.
        """
        for step in self._steps:
            if step._parameter.track:
                # We are saving this parameter's values as a list, so add to dictionary of samples.
                self.mcmc_samples.samples[step._parameter.name] = []

    def save_values(self):
        """
        Add the current parameter values to the list of MCMC samples.
        """
        self.mcmc_samples.samples[self.sigsqr.name].append(self.sigsqr.value)
        marginal_loglik = 0.0
        for tree, mu in zip(self.trees, self.mus):
            self.mcmc_samples.samples[tree.name].append(tree.value)
            self.mcmc_samples.samples[mu.name].append(mu.value)
            marginal_loglik += tree._log_posterior

        self._logliks.append(marginal_loglik)  # save marginal log-posteriors for tree configurations


class BartSample(object):
    def __init__(self, ytrain, m, prior_info, Xtrain=None):
        """
        Constructor class used to access and use the MCMC samples for a BART model. This class can be used to directly
        access the values of the BART variance, tree configurations, and means of the terminal nodes. In addition,
        the member functions enable one to use the MCMC samples to predict the value of the response at an input set
        of predictors, as well as make feature importance or partial dependency plots.

        @param ytrain: The values of the response used to train the model.
        @param m: The number of tree used in the BART ensemble.
        @param prior_info: A dictionary containing the values of the prior hyperparameters.
        @param Xtrain: The array of predictors used to train the model.
        """
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.m = m
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        self.ymin = self.ytrain.min()  # needed for translating the BART output to the original data scale
        self.ymax = self.ytrain.max()

        # dictionary containing the values of the prior hyperparameters
        self.prior_info = prior_info

        self.samples = {}  # the MCMC samples are stored here

    def predict(self, x):
        pass

    def feature_importance(self):
        pass

    def plot_feature_importance(self):
        pass

    def partial_dependence(self):
        pass

    def plot_partial_dependence(self):
        pass

#### CartTree class is untested, not finished, and probably will not work. It is only left here just in case we
#### we want to use it later.

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

    
