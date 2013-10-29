"""
This file contains the class definitions for proposal objects. In general, these are needed by any Metropolis-Hastings
steps (the class definitions for M-H steps are given in steps.py). In order to use the M-H step and its subclasses one
must supply a proposal object.
"""

import numpy as np
import math

__author__ = "Brandon C. Kelly"


class Proposal(object):
    """
    Base proposal class. This is an empty class and is just a template for proposal classes derived from this
    class. The methods of Proposal should be overridden in the derived classes.
    """

    def draw(self, current_value):
        """
        Method to generate a proposal given the current value of the parameter.

        :param current_value: The current value of the parameter.
        """
        pass

    def logdensity(self, proposed_value, current_value):
        """
        Method to return the logarithm of the probability of going from the current value to the proposed value
        of the parameter.

        :param proposed_value: The proposed value of the parameter.
        :param current_value:  The current value of the parameter.
        """
        pass


class NormalProposal(Proposal):
    """
    Generate proposals from a univariate Normal distribution centered at the current value of the parameter.
    """

    def __init__(self, sigma):
        """
        Constructor for Normal Proposal object.

        :param sigma: The standard deviation of the proposed values.
        """
        self.sigma = sigma

    def draw(self, current_value):
        proposed_value = np.random.normal(current_value, self.sigma)
        return proposed_value

    def logdensity(self, proposed_value, current_value):
        return 0.0  # Symmetric proposal, so just return zero.


class MultiNormalProposal(Proposal):
    """
    Generate proposals from a multivariate normal distribution centered at the current value of the parameter.
    """

    def __init__(self, covar):
        """
        Constructor for multivariate normal proposal object.

        :param covar (array-like): The covariance matrix of the proposals.
        """
        self.covar = covar

    def draw(self, current_value):
        proposed_value = np.random.multivariate_normal(current_value, self.covar)
        return proposed_value

    def logdensity(self, proposed_value, current_value):
        return 0.0  # Symmetric proposal, so returned value doesn't matter.


class LogNormalProposal(Proposal):
    """
    Generate proposals from a log-normal distribution.
    """

    def __init__(self, scale):
        """
        Constructor for the Log-normal proposal object.

        :param scale: The standard deviation of log(parameter value)
        """
        self.scale = scale

    def draw(self, current_value):
        assert current_value > 0  # Make sure values are positive
        proposed_value = np.random.lognormal(math.log(current_value), self.scale)
        return proposed_value

    def logdensity(self, proposed_value, current_value):
        assert current_value > 0  # Make sure values are positive
        assert proposed_value > 0
        chi = (math.log(proposed_value) - math.log(current_value)) / self.scale
        log_density = -math.log(proposed_value) - chi * chi
        return log_density


class StudentProposal(Proposal):
    """
    Generate proposals from a Student's t-distribution.
    """

    def __init__(self, dof, scale):
        """
        Constructor for the Student's t-distribution proposal object.

        :param dof: The degrees of freedom for the t-distribution proposals.
        :param scale: The scale of the t-distribution.
        """
        self.dof = dof
        self.scale = scale

    def draw(self, current_value):
        proposed_value = current_value + self.scale * np.random.standard_t(self.dof)
        return proposed_value

    def logdensity(self, proposed_value, current_value):
        return 0.0  # Symmetric proposal, so returned value does not matter.


