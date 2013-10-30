"""
This file contains the class definitions for various priors.
"""

__author__ = 'Brandon C. Kelly'

import numpy as np
from scipy import stats


class Prior(object):
    """
    Base class for prior distribution objects. The two main methods of the prior class are Draw() and
    LogDensity(), both of which must be overriden in order to be used. However, it is only required that
    LogDensity be overriden, as it is the only method required by an MCMC sampler.

    This class is merely a base class and should never be instantiated.
    """

    def __init__(self, temperature=1.0):
        """
        Constructor for prior object.

        :param temperature: The prior 'temperature', used for tempered distributions. If you don't
                            know what this means than just use the default of 1.0, since this corresponds
                            to "normal" MCMC.
        """
        self.temperature = temperature

    def logdensity(self, value):
        """
        The logarithm of the prior distribution evaluated at the input value.

        :param value: Compute the prior distribution at this value.
        """
        return 0.0

    def draw(self):
        """
        Return a random draw from the prior distribution.
        """
        return 0.0


class Uninformative(Prior):
    """
    An uninformative prior. This just return 0.0 for the logarithm of the prior distribution for all
    values of the parameter. The Draw() method is undefined for this class and should not be used.
    """

    def logdensity(self, value):
        return 0.0


class Normal(Prior):
    """
    A normal prior.
    """

    def __init__(self, mu, variance, temperature=1.0):
        """
        Constructor for normal prior object.

        :param mu: The prior mean.
        :param variance: The prior variance.
        :param temperature: Temperature of prior distribution, see Prior class documentation.
        """
        Prior.__init__(self, temperature=temperature)
        self.mu = mu
        self.variance = variance
        self.sigma = np.sqrt(variance)

    def logdensity(self, value):
        return stats.norm.logpdf(value, loc=self.mu, scale=self.sigma)

    def draw(self):
        return stats.norm.rvs(loc=self.mu, scale=self.sigma)


class ScaledInvChiSqr(Prior):
    """
    A scaled inverse-chi-square prior object.
    """

    def __init__(self, dof, ssqr, temperature=1.0):
        """
        Constructor for scaled inverse-chi-square prior object.

        :param dof: The prior degrees of freedom.
        :param ssqr: The prior variance.
        :param temperature: Temperature of prior distribution, see Prior class documentation.
        """
        Prior.__init__(self, temperature=temperature)
        # assert dof > 0  # Make sure parameter values are positive
        # assert ssqr > 0
        self.dof = dof
        self.ssqr = ssqr

    def logdensity(self, value):
        return stats.invgamma.logpdf(value, self.dof / 2.0, scale=self.dof * self.ssqr / 2.0)

    def draw(self):
        return stats.invgamma.rvs(self.dof / 2.0, scale=self.dof * self.ssqr / 2.0)
