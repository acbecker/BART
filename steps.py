"""
This file contains the class definitions and functions that form the base of an MCMC sampler.
These classes are the base Parameter class, various MCMC step classes (e.g., Gibbs update,
Metropolis-Hastings update), and various proposal distributions for the Metropolis-like
algorithms.
"""

#TODO: ADD IN ASSERTIONS TO CHECK TYPEDEFS OF INPUT CLASSES

__author__ = 'Brandon C. Kelly'

import numpy as np
import math
from misc import CholUpdateR1
from scipy.linalg import cholesky


class Parameter(object):
    """
    Base parameter class. The Parameter base class is abstract and should never be instantiated on its own.
    Instead, it is meant as a prototype for one to derive parameter classes from. As a result, it contains
    several methods that are expected to be overridden.
    """

    def __init__(self, name, track=True, temperature=1.0):
        """
        Constructor for the parameter class.

        :param name: A string containing the parameter name.
        :param track: A boolean variable indicating whether the parameter values should be saved.
        :param temperature: The parameter's temperature, used for tempered distributions. If you don't
                            know what this means than just use the default of 1.0, since this corresponds
                            to "normal" MCMC.
        """
        self.name = name
        self.track = track
        self._temperature = temperature
        self._log_posterior = 0.0
        self.SetStartingValue()

    def SetStartingValue(self):
        """
        Method to generate starting values for the parameter. This is empty and must be overridden by the
        derived class.
        """
        self.value = 0.0

    def LogDensity(self, value):
        """
        Method to compute the logarithm of the posterior, given the input parameter value. This
        method does not need to be defined when using the Gibbs sampler.

        :param value: The value of the parameter.
        """
        return 0.0

    def RandomPosterior(self):
        """
        Method to generate a random draw of the parameter from its posterior distribution. This
        method is used by the Gibbs step and must be defined when employing a Gibbs sampler.
        """
        return 0.0


class Step(object):
    """
    Base step class. This Step base class forms a prototype for all MCMC steps, and should never be instantiated
    on its own. Instead, the user should instantiate one of the derived classes (e.g., a Gibbs update or
    Metropolis-Hastings update), or create his/her own steps derived from Step. The methods of Step are empty
    and should be overridden in derived classes.
    """

    def __init__(self, parameter):
        """
        Constructor for the Step class.

        :param parameter: A Parameter object. The MCMC steps used by Step will be applied to parameter.
        """
        self._parameter = parameter

    def DoStep(self):
        """
        Method to perform the MCMC step.
        """
        pass


class GibbStep(Step):
    """
    The Gibbs update. This works by updating the parameter value with a random draw from the posterior
    distribution, conditional on the data and any other parameter objects. The user must have defined the
    method RandomPosterior for the parameter object in order to use this step.
    """

    def __init__(self, parameter):
        Step.__init__(self, parameter)

    def DoStep(self):
        self._parameter.value = self._parameter.RandomPosterior()


class MetroStep(Step):
    """
    The Metropolis-Hastings update. This works by generating a proposed parameter value from a proposal
    object, and computing the ratio of the posterior under the proposed value compared to the current
    parameter value. The proposal is accepted with probability equal to this ratio or one, whichever is
    smaller. MetroStep objects require the user to supply a proposal object, and that the LogDensity method
    of the parameter object to be defined.
    """

    def __init__(self, parameter, proposal, report_iter=-1):
        """
        Constructor for Metropolis-Hastings step object.


        :param parameter: The parameter object associated with this step.
        :param proposal: The proposal object used to generate parameter value proposals.
        :param report_iter: Report on the acceptance rate after this meany iterations of the MCMC sampler.
        """
        Step.__init__(self, parameter)
        self._proposal = proposal
        self.report_iter = report_iter
        self.naccept = 0
        self.niter = 0
        self._alpha = 0.0
        # Set starting value of log-posterior
        self._parameter._log_posterior = self._parameter.LogDensity(self._parameter.value)

    def Report(self):
        """
        Method to report the average acceptance rates since the beginning of the sampler.
        """
        arate = float(self.naccept) / self.niter
        print 'Average acceptance rate is:', arate

    def Accept(self, proposed_value, current_value):
        """
        Method to generate a boolean random variable describing whether the proposed parameter value
        is accepted.

        :param proposed_value: The proposed parameter value.
        :param current_value: The current parameter value.
        """

        ## Accept the proposed value with min(1.0, exp(alpha)).
        alpha = self._parameter.LogDensity(proposed_value) - self._proposal.LogDensity(current_value, proposed_value) \
            - (self._parameter._log_posterior - self._proposal.LogDensity(proposed_value, current_value))

        self._alpha = min(1.0, math.exp(alpha))

        unif = np.random.uniform()

        if not np.isfinite(alpha):
            # Make sure alpha is finite, otherwise reject the proposal
            self._alpha = alpha
            return False

        return unif < self._alpha

    def DoStep(self):
        proposed_value = self._proposal.Draw(self._parameter.value)

        if self.Accept(proposed_value, self._parameter.value):
            self._parameter.value = proposed_value
            self._parameter._log_posterior = self._parameter.LogDensity(proposed_value)
            self.naccept += 1

        self.niter += 1

        if self.niter == self.report_iter:
            self.Report()


class AdaptiveMetro(MetroStep):
    """
    Robust Adaptive Metropolis Algorithm (RAM).

    Reference: Robust Adaptive Metropolis Algorithm with Coerced Acceptance Rate,
	M. Vihola, 2012, Statistics & Computing, 22, 997-1008
    """

    def __init__(self, parameter, proposal, covar, target_rate, maxiter, report_iter=-1):
        """
        Constructor for Robust Adaptive Metropolis step object.

        :param parameter: The parameter object.
        :param proposal: The proposal object to generate unit (untransformed) proposals.
        :param covar: The initial covariance matrix of the proposals. This can be a scalar for scalar-valued parameters.
        :param target_rate: The target acceptance rate.
        :param maxiter: The maximum number of iterations in which to adapt the covariance of the proposals.
        :param report_iter: Report the average acceptance rate after this many iterations.
        """
        MetroStep.__init__(self, parameter, proposal, report_iter=report_iter)
        self._covar = covar
        self.target_rate = target_rate
        self._maxiter = maxiter
        # Default value of the decay rate. If you don't know what this is don't change it.
        self._gamma = 2.0 / 3.0
        # The cholesky factor for the covariance matrix of the proposals. The algorithm adapts the covariance
        # matrix self.covar through computations on the cholesky factor. If the parameter is scalar-valued then
        # this is ignored.
        if np.isscalar(self._covar):
            self._cholesky_factor = np.sqrt(self._covar)  # Cholesky factor is just standard deviation for scalars
        else:
            self._cholesky_factor = cholesky(covar)  # Cholesky factor is upper triangular, needed for the rank 1 update

    def UpdateCovar(self, proposed_value, unit_proposal, centered_proposal):
        """
        Method to update the covariance matrix (actually, its Cholesky decomposition), based on the
        proposed value and the value of the unit proposal.

        :param proposed_value (array-like): The proposed value of the parameter.
        :param unit_proposal (array-like):  The value generated by the proposal object.
        """
        # The step size sequence for the scale matrix update. This is eta_n in the notation
        # of Vihola (2012).
        step_size = min(1.0, proposed_value.size / (self.niter + 1.0) ** self._gamma)

        if np.isscalar(proposed_value):
            # Parameter is scalar-valued so the update is done analytically
            log_width = np.log(self._cholesky_factor) + 0.5 * np.log(1.0 + step_size * (self._alpha - self.target_rate))
            self._cholesky_factor = np.exp(log_width)
        else:
            # Parameter is vector-valued, so do rank-1 Cholesky update/downdate
            unit_norm = np.sqrt(unit_proposal.dot(unit_proposal))  # L2 norm of the vector

            # Rescale the proposal vector for updating the scale matrix cholesky factor
            scaled_proposal = np.sqrt(step_size * abs(self._alpha - self.target_rate)) / unit_norm * centered_proposal

            # Update or downdate?
            downdate = self._alpha < self.target_rate

            # Perform the rank-1 update (or downdate) of the scale matrix cholesky factor
            CholUpdateR1(self._cholesky_factor, scaled_proposal, downdate=downdate)

    def DoStep(self):
        # First draw the unit proposal
        if np.isscalar(self._parameter.value):
            # Parameter is scalar-valued, need to handle this case separately
            unit_proposal = self._proposal.Draw(0.0)
            centered_proposal = self._cholesky_factor * unit_proposal
            # Rescale the unit proposal to a proposed parameter value
            proposed_value = self._parameter.value + centered_proposal
        else:
            # Parameter is array (vector) valued, so use linear algebra operations
            unit_proposal = self._proposal.Draw(np.zeros(self._parameter.value.size))
            # Transform the unit proposal to a proposed value through scaling and translation
            centered_proposal = np.transpose(self._cholesky_factor).dot(unit_proposal)
            proposed_value = self._parameter.value + centered_proposal

        # Accept this proposal?
        if self.Accept(proposed_value, self._parameter.value):
            # Proposal is accepted, update the parameter value
            self._parameter.value = proposed_value
            self._parameter._log_posterior = self._parameter.LogDensity(proposed_value)
            self.naccept += 1

        if (self.niter < self._maxiter) & (np.isfinite(self._alpha)):
            # Update the scale matrix of the proposals
            self.UpdateCovar(proposed_value, unit_proposal, centered_proposal)

        self.niter += 1

        if self.niter == self.report_iter:
            # Report on progress
            self.Report()
