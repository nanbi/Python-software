from itertools import product
import numpy as np
import nose.tools as nt

from ..convenience import lasso, step, threshold
from ..glm import target as glm_target
from ...tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance)
from ...tests.flags import SMALL_SAMPLES
from ...tests.decorators import set_sampling_params_iftrue 

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_lasso_constructors(ndraw=1000, burnin=200):
    """
    Smoke tests for lasso convenience constructors
    """
    cls = lasso
    for const_info, rand in product(zip([gaussian_instance,
                                         logistic_instance,
                                         poisson_instance],
                                        [cls.gaussian,
                                         cls.logistic,
                                         cls.poisson]),
                              ['gaussian', 'logistic', 'laplace']):

        inst, const = const_info
        X, Y = inst()[:2]
        n, p = X.shape

        W = np.ones(X.shape[1]) * 20
        conv = const(X, Y, W, randomizer=rand)
        signs = conv.fit()

        marginalizing_groups = np.zeros(p, np.bool)
        marginalizing_groups[:int(p/2)] = True
        
        conditioning_groups = ~marginalizing_groups
        conditioning_groups[-int(p/4):] = False

        selected_features = np.zeros(p, np.bool)
        selected_features[:3] = True

        conv.summary(selected_features,
                     ndraw=ndraw,
                     burnin=burnin,
                     compute_intervals=True)

        conv.decompose_subgradient(marginalizing_groups=marginalizing_groups,
                                   conditioning_groups=conditioning_groups)

        conv.summary(selected_features,
                     ndraw=ndraw,
                     burnin=burnin)

        target_sampler, target_observed = glm_target(conv.loglike,
                                                     selected_features,
                                                     conv._queries,
                                                     bootstrap=False)

        S = target_sampler.sample(ndraw,
                                  burnin)


@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_step_constructors(ndraw=1000, burnin=200):
    """
    Smoke tests for greedy_step convenience constructors
    """

    cls = step
    for const_info, rand in product(zip([gaussian_instance,
                                         logistic_instance,
                                         poisson_instance],
                                        [cls.gaussian,
                                         cls.logistic,
                                         cls.poisson]),
                              ['gaussian', 'logistic', 'laplace']):

        inst, const = const_info
        X, Y = inst()[:2]
        W = np.ones(X.shape[1])
        conv = const(X, Y, W)
        conv.fit()

        n, p = X.shape
        active = np.zeros(p, np.bool)
        active[:int(p/2)] = True

        candidate = ~active
        candidate[-int(p/4):] = False

        conv1 = const(X, Y, W, active=active)
        conv1.fit()

        conv2 = const(X, Y, W, candidate=candidate)
        conv2.fit()
        
        conv3 = const(X, Y, W, candidate=candidate, active=active)
        conv3.fit()
        
        selected_features = np.zeros(p, np.bool)
        selected_features[:3] = True

        conv3.summary(selected_features,
                      ndraw=ndraw,
                      burnin=burnin,
                      compute_intervals=True)

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
def test_threshold_constructors(ndraw=1000, burnin=200):
    """
    Smoke tests for marginal threshold convenience constructors
    """

    cls = threshold
    for const_info, rand in product(zip([gaussian_instance,
                                         logistic_instance,
                                         poisson_instance],
                                        [cls.gaussian,
                                         cls.logistic,
                                         cls.poisson]),
                              ['gaussian', 'logistic', 'laplace']):

        inst, const = const_info
        X, Y = inst()[:2]
        W = np.ones(X.shape[1])

        n, p = X.shape
        active = np.zeros(p, np.bool)
        active[:int(p/2)] = True

        candidate = ~active
        candidate[-int(p/4):] = False

        conv1 = const(X, Y, W, active=active)
        conv1.fit()

        conv2 = const(X, Y, W, candidate=candidate)
        conv2.fit()
        
        conv3 = const(X, Y, W, candidate=candidate, active=active)
        conv3.fit()
        
        selected_features = np.zeros(p, np.bool)
        selected_features[:3] = True

        conv3.summary(selected_features,
                      ndraw=ndraw,
                      burnin=burnin)


