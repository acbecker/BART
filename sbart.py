import numpy as np
from scipy.spatial.distance import cdist
from .tree import BaseTree
from .tree import BartTrees

class AdjacencyMatrix(object):
    __slots__ = ["distances"]
    def __init__(self, distances):
        self.distances = distances
    def makeMatrix(self, i, nneighbors):
        pass

class InversePowerAdjacencyMatrix(AdjacencyMatrix):
    __slots__ = ["power"]
    def __init__(self, distances, power):
        AdjacencyMatrix.__init__(distances)

        self.power = power
        assert(self.power in [-1, -2])

    def makeMatrix(self, i, nneighbors):
        ddist  = self.distances[:,i]
        idx    = np.argsort(ddist)[1:] # Ignore self!
        matrix = ddist**self.power
        return idx[:nneighbors], matrix[idx[:nneighbors]]


class ExponentialAdjacencyMatrix(AdjacencyMatrix):
    __slots__ = ["rdist", "power"]
    def __init__(self, distances, rdist, power):
        AdjacencyMatrix.__init__(distances)

        self.rdist = rdist
        assert(self.rdist < 0.0)
        self.power = power
        assert(self.power in [1,2])

    def makeMatrix(self, i, nneighbors):
        ddist  = self.distances[:,i]
        idx    = np.argsort(ddist)[1:] # Ignore self!
        matrix = np.exp(rdist * ddist**self.power)
        return idx[:nneighbors], matrix[idx[:nneighbors]]


class SBart(object):
    """Variable names are in the formalism of Zhang, 2007

    Assume that W and V are measured for the 2 populations
    V  = n_features1 x n_samples_y
    W  = n_features1 x n_samples_x
    x  = n_samples_x
    c1 = 2 x n_samples_x

    So we find a regression of x given W through a standard BART.  We
    may then predict y from V given that model.  To accout for spatial
    effects we (optionally) use coordinates c1 for the locations of
    each data point.  Finally, we then model Z as a function of y, at
    coordinates c2.

    y  = n_samples_y
    Z  = n_features2 x n_samples_y
    c2 = 2 x n_samples_y

    """
    __slots__ = ["Z", "V", "W", "x", "nmin", 
                 "n_features1", "n_samples_x", "n_samples_y", "n_features2",
                 "rho", "dist1", "dist2"]
    def __init__(self, Z, V, W, x, rho, 
                 c1=None, c2=None, 
                 min_samples_leaf=5):
        self.Z = Z
        self.V = V
        self.W = W
        self.x = x
        self.nmin = min_samples_leaf

        self.n_features1 = self.W.shape[0]
        self.n_samples_x = self.W.shape[1]
        self.n_samples_y = self.V.shape[1]
        self.n_features2 = self.Z.shape[0]

        assert(self.Z.shape == (self.n_features2, self.n_samples_y))
        assert(self.V.shape == (self.n_features1, self.n_samples_y))
        assert(self.W.shape == (self.n_features1, self.n_samples_x))
        assert(self.s.shape == (self.n_samples_x))

        # Smoothing spatial correlation parameter for Conditionally Autoregressive Model
        # rho = -1: not sure what this signifies; anticorrelated?
        # rho =  0: diagonal variance matrix, samples are independent 
        # rho = +1: conditional mean is average of its neighbors
        self.rho = rho
        assert(np.abs(self.rho) < 1)

        if c1 is not None:
            assert(c1.shape == (2, self.n_features1))
            self.dist1 = cdist(c1.T, c1.T)
        else:
            self.dist1 = None

        if c2 is not None:
            assert(c2.shape == (2, self.n_features2))
            self.dist2 = cdist(c2.T, c2.T)
        else:
            self.dist2 = None


    def pseudo(self):
        bart1 = BartTrees()
        bart1.fit(self.W, self.x)
        y = bart.predict(self.V)
        bart2 = BartTrees()
        bart2.fit(self.Z, y)
        return bart1, bart2



