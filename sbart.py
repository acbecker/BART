from .tree import BaseTree
from .tree import BartTrees
class SBart(object):
    """Variable names are in the formalism of Zhang, 2007

    Assume that W and V are measured for the 2 populations
    V = n_features1 x n_samples_y
    W = n_features1 x n_samples_x
    x = n_samples_x

    So we find a regression of x given W through a standard BART.  We
    may then predict y from V given that model.  Finally, we then
    model Z as a function of y.

    y = n_samples_y
    Z = n_features2 x n_samples_y

    """
    def __init__(self, Z, V, W, x, min_samples_leaf=5):
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
        
    def pseudo(self):
        bart1 = BartTrees()
        bart1.fit(self.W, self.x)
        y = bart.predict(self.V)
        bart2 = BartTrees()
        bart2.fit(self.Z, y)
