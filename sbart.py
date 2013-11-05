from .tree import BaseTree

class SBart(object):
    """Variable names are in the formalism of Zhang, 2007

    Assume that W and V are measured for the 2 populations
    V = n_features x n_samples_y
    W = n_features x n_samples_x
    x = n_samples_x

    So we find a regression of x given W through a standard BART.

    """
    def __init__(self, V, W, x, min_samples_leaf=5):
        self.V = V
        self.W = W
        self.x = x
        self.nmin = min_samples_leaf

        
