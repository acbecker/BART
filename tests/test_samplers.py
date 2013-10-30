"""
Test the classes in samplers.py. This will test the MCMC sampler by running various MCMC samplers on a Normal model.
"""

__author__ = 'Brandon C. Kelly'

import numpy as np
import tests.test_steps as tsteps
import steps, samplers, priors, proposals
from matplotlib import pyplot as plt
from matplotlib import mlab

# First create some data
mu = np.random.standard_cauchy()
var = np.random.chisquare(10) / 10.0

ndata = 1000
data = np.random.normal(mu, var, ndata)

# Construct a simple normal model. Subsequent tests are based on this object.

# First instantiate the prior and parameter objects
MuPrior = priors.Normal(0.0, 1000.0, 1.0)
VarPrior = priors.ScaledInvChiSqr(2.0, 1.0, 1.0)
NormMean = tsteps.NormalMean(data, MuPrior, "mu", True, 1.0)
NormVar = tsteps.NormalVariance(data, VarPrior, "sigsqr", True, 1.0)

# Make the two parameter objects aware of eachother
NormMean.SetVariance(NormVar)
NormVar.SetMean(NormMean)

# Set initial values
NormMean.set_starting_value()
NormVar.set_starting_value()

# Instantiate Gibbs step objects
MuGibbs = steps.GibbStep(NormMean)
VarGibbs = steps.GibbStep(NormVar)

# Now instantiate the MCMC samples and sampler object
nsamples = 20000
burnin = 20000  # Do 5000 iterations of burn-in
NormSampler = samplers.Sampler()

###### Construct a bivariate normal model. This will test vector versions of the sampler class methods.
# First create some data
mu2 = np.random.standard_cauchy(size=2)

zz = np.random.multivariate_normal(np.zeros(2), np.identity(2), 5)
covar = zz.T.dot(zz) / 5.0

data2 = np.random.multivariate_normal(mu2, covar, ndata)

# Instantiate the prior and parameter object
NormPar = tsteps.BivariateNormalMean(data2, covar, priors.Uninformative())

# Create proposal object for RAM proposals
covar = np.identity(2)
UnitProp = proposals.MultiNormalProposal(covar)

# Create Robust Adaptive Metropolis step object
target_rate = 0.4
BiNormRAM = steps.AdaptiveMetro(NormPar, UnitProp, covar, target_rate, burnin)

# Create bivariate normal samples and sampler objects
BiNormSampler = samplers.Sampler()


def test_addstep():
    """
    Test the addstep method of the Sampler class.
    """
    # First test step addition on scalar-valued parameter
    NormSampler.add_step(MuGibbs)
    assert MuGibbs in NormSampler._steps
    NormSampler.sample_size = nsamples
    NormSampler.start()
    NormSamples = NormSampler.mcmc_samples
    # Was this parameter added to the dictionary of samples in the NormSamples object?
    assert NormMean.name in NormSamples._samples
    if NormMean.name in NormSamples._samples:
        # NormMean is a scalar-value parameter. So check to make sure this is a single column array.
        assert NormSamples._samples[NormMean.name].shape == (nsamples,)

    # Now test step addition on vector-valued parameter
    BiNormSampler.add_step(BiNormRAM)
    assert BiNormRAM in BiNormSampler._steps
    BiNormSampler.sample_size = nsamples
    BiNormSampler.start()
    BiNormSamples = BiNormSampler.mcmc_samples

    # Was this parameter added to the dictionary of samples in the BiNormSamples object?
    assert NormPar.name in BiNormSamples._samples
    if NormPar.name in BiNormSamples._samples:
        # Values of NormPar are 2-element vectors, so make sure the samples are a np.array([nsamples,2]) object.
        assert BiNormSamples._samples[NormPar.name].shape == (nsamples, 2)

        # TODO: TEST THE addstep METHOD FOR MATRIX-VALUED PARAMETERS

    print 'Test of Sampler.add_step() was successful.'


def test_savevalues():
    """
    Test the method of the Sampler class that saves the sampled parameter values.
    """
    # First test that we have correctly save the value for a scalar valued parameter
    NormSampler.save_values()

    current_index = NormSampler._sampler_bar.currval
    NormSamples = NormSampler.mcmc_samples
    mu_samples = NormSamples.get_samples(NormMean.name)
    assert mu_samples[current_index] == NormMean.value

    # Now do same test for vector-valued parameter
    BiNormSampler.save_values()
    current_index = BiNormSampler._sampler_bar.currval
    BiNormSamples = BiNormSampler.mcmc_samples
    mu2d_samples = BiNormSamples.get_samples(NormPar.name)
    equal_array = (mu2d_samples[current_index, :] == NormPar.value)
    assert np.sum(equal_array) == NormPar.value.size

    print 'Test of Sampler.save_values() was successful.'


def test_generate_from_file():
    """
    Test the method of the MCMCSample class that construct a MCMCSample object from a asciifile.
    """
    pass


def test_normal_mean_mha():
    """
    Test the Metropolis-Hastings algorithm for a single parameter using the normal mean model.
    """
    NormVar.value = var
    NormProp = proposals.NormalProposal(np.sqrt(var / ndata))
    MuMHA = steps.MetroStep(NormMean, NormProp, report_iter=burnin)

    MuSampler = samplers.Sampler([MuMHA])

    MuSamples = MuSampler.run(burnin, nsamples)
    MuSamples.newaxis()

    # Get the parameter values
    trace = MuSamples.get_samples(NormMean.name)

    # Compare the estimated PDF with the true posterior
    counts, mugrid, patches = plt.hist(trace, bins=25, normed=True)
    pdf0 = 1.0 / np.sqrt(2.0 * 3.14 * var / ndata) * np.exp(-0.5 * ndata * (NormMean.data_mean - mugrid) ** 2 / var)
    plt.plot(mugrid, pdf0, 'r', lw=2)
    plt.title("Mean Value for Normal Model, MHA")
    plt.ylabel("Posterior PDF")
    plt.xlabel("$\mu$")
    plt.show()

    # Summarize and plot the posterior
    MuSamples.posterior_summaries(NormMean.name)
    MuSamples.plot_parameter(NormMean.name)

    # Test that the moments of the MCMC samples are within 3sigma and 5% of their theoretical values
    neffective = MuSamples.effective_samples(NormMean.name)  # Effective number of independent samples
    assert np.abs(np.mean(trace) - data.mean()) < 3.0 * np.std(trace)
    assert np.abs(np.std(trace) - np.sqrt(var / ndata)) * np.sqrt(ndata / var) < 0.05

    print 'Test of MHA algorithm for scalar-valued parameter was successful.'


def test_normal_mean_ram():
    """
    Test the Robust Adaptive Metropolis algorithm for single parameter using the normal mean model.
    """
    NormVar.value = var
    NormProp = proposals.StudentProposal(8.0, 1.0)
    target_rate = 0.45
    MuRAM = steps.AdaptiveMetro(NormMean, NormProp, np.sqrt(var / ndata), target_rate, burnin, report_iter=burnin)

    MuSamples = samplers.MCMCSample()
    MuSampler = samplers.Sampler(MuSamples, nsamples, burnin)

    MuSampler.add_step(MuRAM)

    MuSampler.run()

    # Make sure acceptance rate is within 2% of the target rate, since we did not test this for scalar-valued parameters
    # in test_AdaptiveMetro().
    acceptance_rate = MuRAM.naccept / float(MuRAM.niter)
    assert abs(acceptance_rate - target_rate) / abs(target_rate) < 0.02

    # Get the parameter values
    trace = MuSamples.get_samples(NormMean.name)

    # Compare the estimated PDF with the true posterior
    counts, mugrid, patches = plt.hist(trace, bins=25, normed=True)
    pdf0 = 1.0 / np.sqrt(2.0 * np.pi * var / ndata) * np.exp(-0.5 * ndata * (NormMean.data_mean - mugrid) ** 2 / var)
    plt.plot(mugrid, pdf0, 'r', lw=2)
    plt.title("Mean Value for Normal Model, RAM")
    plt.ylabel("Posterior PDF")
    plt.xlabel("$\mu$")
    plt.show()

    # Summarize and plot the posterior
    MuSamples.posterior_summaries(NormMean.name)
    MuSamples.plot_parameter(NormMean.name)

    # Test that the moments of the MCMC samples are within 3sigma and 5% of their theoretical values
    neffective = MuSamples.effective_samples(NormMean.name)  # Effective number of independent samples
    assert np.abs(np.mean(trace) - data.mean()) < 3.0 * np.std(trace)
    assert np.abs(np.std(trace) - np.sqrt(var / ndata)) * np.sqrt(ndata / var) < 0.05


def test_normal_model_ram():
    """
    Test the Robust Adaptive Metropolis algorithm for a vector-valued parameter using the bivariate normal meam model.
    """
    # Create proposal object for RAM proposals
    covar = np.identity(2)
    UnitProp = proposals.MultiNormalProposal(covar)

    # Create Robust Adaptive Metropolis step object
    target_rate = 0.4
    initial_covar = np.cov(data2, rowvar=0)
    BiNormRAM = steps.AdaptiveMetro(NormPar, UnitProp, initial_covar, target_rate, burnin)

    # Create bivariate normal samples and sampler objects
    BiNormSamples = samplers.MCMCSample()
    BiNormSampler = samplers.Sampler(BiNormSamples, nsamples, burnin, thin=5)

    # Add RAM step
    BiNormSampler.add_step(BiNormRAM)

    # Run the sampler
    BiNormSampler.run()

    # Get the parameter values
    trace = BiNormSamples.get_samples(NormPar.name)

    # Test that the moments of the MCMC samples are less than 3sigma from the true values
    neffective = BiNormSamples.effective_samples(NormPar.name)
    inv_covar = np.matrix(np.inv(NormPar.covar / neffective + NormPar.covar / ndata))
    mu_hat = np.mean(trace, axis=0)
    chisqr = np.matrix(mu_hat - mu2) * inv_covar * np.matrix(mu_hat - mu2).T
    assert chisqr < 9.21  # chisqr < 99th percentile of chi-square distribution with 2 degrees of freedom

    mcmc_covar = np.cov(trace, rowvar=0)
    mu_covar = NormPar.covar / ndata
    covar_diff = np.abs(mcmc_covar - mu_covar)
    # Approximate the sample covariance MCMC error with a Wishart distribution
    assert neffective * covar_diff[0, 0] < 3.0 * np.sqrt(2.0 * neffective * mu_covar[0, 0])
    assert neffective * covar_diff[1, 1] < 3.0 * np.sqrt(2.0 * neffective * mu_covar[1, 1])
    assert neffective * covar_diff[0, 1] < \
        3.0 * np.sqrt(neffective * (mu_covar[0, 1] ** 2 + mu_covar[0, 0] * mu_covar[1, 1]))

    # Compare the estimated PDF with the true posterior
    plt.subplot(221)
    counts, mugrid, patches = plt.hist(trace[:, 0], bins=25, normed=True)
    pdf0 = 1.0 / np.sqrt(2.0 * np.pi * NormPar.covar[0, 0] / ndata) * \
           np.exp(-0.5 * ndata * (NormPar.data_mean[0] - mugrid) ** 2 / NormPar.covar[0, 0])
    plt.plot(mugrid, pdf0, 'r', lw=2)
    plt.title("First element of Mean Value, RAM")
    plt.ylabel("Posterior PDF")
    plt.xlabel("$\mu$[0]")

    plt.subplot(222)
    xgrid = np.linspace(mu_hat[0] - 4.0 * mcmc_covar[0, 0], mu_hat[0] + 4.0 * mcmc_covar[0, 0], 100)
    ygrid = np.linspace(mu_hat[1] - 4.0 * mcmc_covar[1, 1], mu_hat[1] + 4.0 * mcmc_covar[1, 1], 100)

    X, Y = np.meshgrid(xgrid, ygrid)

    true_pdf = mlab.bivariate_normal(X, Y, np.sqrt(mu_covar[0, 0]), np.sqrt(mu_covar[1, 1]),
                                     mu2[0], mu2[1], mu_covar[0, 1])

    plt.plot(trace[:, 0], trace[:, 1], '.')
    plt.contour(X, Y, true_pdf)
    plt.title('Bivariate Mean, RAM')

    plt.subplot(224)
    counts, mugrid, patches = plt.hist(trace[:, 1], bins=25, normed=True)
    pdf0 = 1.0 / np.sqrt(2.0 * np.pi * NormPar.covar[1, 1] / ndata) * \
           np.exp(-0.5 * ndata * (NormPar.data_mean[1] - mugrid) ** 2 / NormPar.covar[1, 1])
    plt.plot(mugrid, pdf0, 'r')
    plt.title("Second element of Mean Value, RAM")
    plt.ylabel("Posterior PDF")
    plt.xlabel("$\mu$[1]")
    plt.show()

    # Summarize and plot the posterior
    BiNormSamples.posterior_summaries(NormPar.name)
    for i in xrange(NormPar.value.size):
        BiNormSamples.plot_parameter(NormPar.name, pindex=i)


def test_normal_model_gibbs():
    """
    Test the Gibbs sampler for the normal model with unknown mean and variance.
    """
    NormMean.SetVariance(NormVar)
    NormVar.SetMean(NormMean)

    MuGibbs = steps.GibbStep(NormMean)
    VarGibbs = steps.GibbStep(NormVar)

    NormSamples = samplers.MCMCSample()
    NormSampler = samplers.Sampler(NormSamples, nsamples, burnin)

    # Add the Gibbs steps
    NormSampler.add_step(MuGibbs)
    NormSampler.add_step(VarGibbs)

    NormSampler.run()

    # Get the parameter values
    mutrace = NormSamples.get_samples(NormMean.name)
    vartrace = NormSamples.get_samples(NormVar.name)

    # Test that the moments of the MCMC samples for the mean are less than 3sigma from the true values
    neffective = NormSamples.effective_samples(NormMean.name)  # Effective number of independent samples
    assert np.abs(np.mean(mutrace) - mu) < 3.0 * np.std(mutrace)
    assert np.abs(np.var(mutrace) - var / ndata) < 3.0 * np.sqrt(2.0 / neffective) * var / ndata

    neffective = NormSamples.effective_samples(NormVar.name)
    assert np.abs(np.mean(vartrace) - var) < 3.0 * np.std(vartrace)
    assert np.abs(np.var(vartrace) - 2.0 * var / ndata) < 3.0 * np.sqrt(2.0 / neffective) * 2.0 * var / ndata

    # Compare the estimated PDF with the true posterior
    counts, mugrid, patches = plt.hist(mutrace, bins=25, normed=True)
    pdf0 = 1.0 / np.sqrt(2.0 * np.pi * var / ndata) * np.exp(-0.5 * ndata * (NormMean.data_mean - mugrid) ** 2 / var)
    plt.plot(mugrid, pdf0, 'r', lw=2)
    plt.title("Mean Value for Normal Model, Gibbs Sampler")
    plt.ylabel("Posterior PDF")
    plt.xlabel("$\mu$")
    plt.show()

    counts, vgrid = plt.hist(vartrace, bins=25, normed=True)
    pdf0 = ndata / 2.0 * np.log(var * ndata / 2.0) - np.math.lgamma(ndata / 2.0) - \
           (1.0 + ndata / 2.0) * np.log(vgrid) - ndata * var / (2.0 * vgrid)
    plt.plot(vgrid, pdf0, 'r', lw=2)
    plt.title("Variance for Normal Model, Gibbs Sampler")
    plt.ylabel("Posterior PDF")
    plt.xlabel("$\sigma^2$")
    plt.show()

    # Summarize and plot the posterior
    NormSamples.posterior_summaries(NormMean.name)
    NormSamples.plot_parameter(NormMean.name)

    NormSamples.posterior_summaries(NormVar.name)
    NormSamples.plot_parameter(NormVar.name)


if __name__ == "__main__":
    test_addstep()
    test_savevalues()
    test_normal_mean_mha()