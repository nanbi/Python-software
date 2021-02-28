"""
Classes encapsulating some common workflows in randomized setting
"""

from copy import copy
import functools

import numpy as np
import regreg.api as rr
from .lasso import lasso
from .randomization import randomization
from .query import multiple_queries, optimization_intervals, affine_gaussian_sampler
#from .M_estimator import restricted_Mest
from .base import restricted_estimator
from ..constraints.affine import constraints
from scipy.optimize import bisect
from scipy.linalg import sqrtm, qr
from scipy import matrix, compress, transpose
from scipy.stats import norm, chi2, f
from scipy.special import gamma
from math import pi


# this class is for specication problem with weak IV
# computes the selective pvalue using CLR statistic
class weak_iv_clr(object):

    def __init__(self, Y, D, Z):
        self.Z = Z
        self.D = D
        self.Y = Y
        self.n = Z.shape[0]
        self.p = Z.shape[1]
        #C_0 is taken to be 10 for now
        self.C_0 = 10. 
        self.ZTZ_inv = np.linalg.inv(Z.T.dot(Z))
        self.ZTD = Z.T.dot(D)
        self.ZTY = self.Z.T.dot(self.Y)

    def pre_test(self):
        n, p = self.Z.shape
        #P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        #denom = self.D.dot(np.identity(n) - P_Z).dot(self.D) / (n-p)
        #num = self.D.dot(P_Z).dot(self.D) / p
        self.DPZD = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTD)
        self.DPZY = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTY)
        denom = (self.D.dot(self.D) - self.DPZD) / (n-p)
        num = self.DPZD / p
        f = num / denom
        self.ftest = f
        self.C = p * denom * self.C_0
        # C_0 is taken to be 10 for now
        #C_0 = 10.
        if f >= self.C_0:
            return True
        else:
            return False

    # take a null parameter value \beta_0 and compute the constants needed
    def setup(self, parameter=None):
        if parameter is None:
            parameter = 0.

        # here we will return the 2 by 2 matrix \hat{\Omega}
        #P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        n, p = self.Z.shape
        #X = np.vstack([self.Y, self.D])
        #cov_estim = X.dot(np.identity(n)-P_Z).dot(X.T) / (n-p)
        YPZY = self.ZTY.dot(self.ZTZ_inv).dot(self.ZTY)
        self.DPZD = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTD)
        self.DPZY = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTY)
        YRY = self.Y.dot(self.Y) - YPZY
        DRY = self.D.dot(self.Y) - self.DPZY
        DRD = self.D.dot(self.D) - self.DPZD
        cov_estim = np.zeros((2,2))
        cov_estim[0,0] = YRY 
        cov_estim[0,1] = DRY 
        cov_estim[1,0] = DRY
        cov_estim[1,1] = DRD
        self.Omega = cov_estim * 1. / (n-p)
        self.Omega_inv = np.linalg.inv(self.Omega)

        #self.beta_estim = (self.D.T.dot(P_Z).dot(self.Y)) / (self.D.T.dot(P_Z).dot(self.D))

        b0 = np.array((1., -parameter)).reshape((-1,1))
        a0 = np.array((parameter, 1.)).reshape((-1,1))
        sqrt_bomega = np.sqrt(b0.T.dot(self.Omega).dot(b0))
        sqrt_aomega = np.sqrt(a0.T.dot(self.Omega_inv).dot(a0))
        sqrt_omega = sqrtm(self.Omega)
        J0 = np.hstack([sqrt_omega.dot(b0) / sqrt_bomega, sqrtm(self.Omega_inv).dot(a0) / sqrt_aomega])

        cmat = J0.T.dot(sqrt_omega)
        self.d0 = cmat[0,1] **2
        self.d1 = 2 * cmat[0,1] * cmat[1,1]
        self.d2 = cmat[1,1] **2

        temp = sqrtm(self.ZTZ_inv).dot(np.vstack([self.ZTY, self.ZTD]).T)
        observed = np.hstack([temp.dot(b0) / sqrt_bomega, temp.dot(self.Omega_inv).dot(a0) / sqrt_aomega])
        Q_obs = observed.T.dot(observed)
        self.qT_obs = Q_obs[1,1]
        self.s2_obs = Q_obs[0,1] / np.sqrt(Q_obs[0,0] * Q_obs[1,1])

        self.observed_stat = .5*(Q_obs[0,0]-Q_obs[1,1]+np.sqrt((Q_obs[0,0]+Q_obs[1,1])**2-4.*(Q_obs[0,0]*Q_obs[1,1]-Q_obs[0,1]**2)))

        self.K4 = gamma(self.p/2.)/(np.sqrt(pi)*gamma((self.p-1)/2.))

        # compute tsls estimate
        self.two_stage_ls = self.DPZY / self.DPZD


    def clr_inverse(self, s2):
        return (self.qT_obs + self.observed_stat) / (1. + self.qT_obs*(s2**2)/self.observed_stat)

    def cond_chisq(self, s2):
        delta = (self.d1**2) * self.qT_obs * (s2**2) - 4.*self.d0*(self.d2*self.qT_obs-self.C)
        linterm = -self.d1*np.sqrt(self.qT_obs)*s2
        if delta <= 0:
            return 0.
        else:
            lroot = (linterm - np.sqrt(delta)) / (2.*self.d0)
            rroot = (linterm + np.sqrt(delta)) / (2.*self.d0)
            threshold = np.sqrt(self.clr_inverse(s2))
            if rroot <= 0:
                return 0.
            elif threshold >= rroot:
                return 0.
            elif threshold < lroot:
                return 1.
            elif threshold >= lroot and lroot >= 0.:
                return (chi2.cdf(rroot**2, self.p) - chi2.cdf(threshold**2, self.p)) / (chi2.cdf(rroot**2, self.p) - chi2.cdf(lroot**2, self.p))
            elif lroot < 0.:
                return (chi2.cdf(rroot**2, self.p) - chi2.cdf(threshold**2, self.p)) / (chi2.cdf(rroot**2, self.p))
        #if delta <= 0 or linterm+np.sqrt(delta) <= 0:
        #    return 1.- chi2.cdf(self.clr_inverse(s2), self.p)
        #elif linterm-np.sqrt(delta) <= 0:
        #    rroot = (linterm + np.sqrt(delta)) / (2.*self.d0)
        #    return min(1., (1.-chi2.cdf(self.clr_inverse(s2),self.p)) / (1.-chi2.cdf(rroot**2)))
        #else:
        #    lroot = (linterm - np.sqrt(delta)) / (2.*self.d0)
        #    rroot = (linterm + np.sqrt(delta)) / (2.*self.d0)
        #    threshold = self.clr_inverse(s2)
        #    if threshold >= rroot**2:
        #        return (1.-chi2.cdf(threshold, self.p)) / (1.-chi2.cdf(rroot**2, self.p)+chi2.cdf(lroot**2, self.p))
        #    else:
        #        num = 1.-chi2.cdf(threshold, self.p)-(chi2.cdf(rroot**2, self.p)-chi2.cdf(max(lroot**2, threshold), self.p))
        #        return num / (1.-chi2.cdf(rroot**2, self.p)+chi2.cdf(lroot**2, self.p))

    def density(self, s2):
        dens = self.K4 * (1.-s2**2)**((self.p-3.)/2.) * self.cond_chisq(s2)
        return dens

    # evaluate integral of self.density using Simpson's rule
    def evaluate(self, nsep=100):
        if nsep % 2 != 0:
            raise ValueError("nsep need to be an even number!")
        dx = 2. / nsep
        fns = np.zeros(nsep+1)
        for i in range(nsep+1):
            fns[i] = self.density(-1.+i*dx)
        nums = np.array([1.] + [4., 2.] * ((nsep-2)/2) + [4., 1.])
        return np.sum(fns * nums) * dx / 3.

    # this is not quite working yet, hard to find the initial points for bisect
    def confidence_interval(self, beta_reference=None, how_many=10., level=0.05, myrange=None):

        if beta_reference is None:
            beta_reference = self.two_stage_ls

        def _root(param):
            self.setup(parameter=param)
            return self.evaluate()-level

        if myrange is None:
            unit = np.std(self.Y)
            upper = bisect(_root, beta_reference, beta_reference+unit*how_many, xtol=1.e-4)
            lower = bisect(_root, beta_reference, beta_reference-unit*how_many, xtol=1.e-4) 
        else:
            [left, right] = myrange
            upper = bisect(_root, beta_reference, left, xtol=1.e-4)
            lower = bisect(_root, beta_reference, right, xtol=1.e-4)            

        return [lower, upper]     

    def naive_chisq(self, s2):
        return 1.-chi2.cdf(self.clr_inverse(s2), self.p)

    def naive_density(self, s2):
        dens = self.K4 * (1.-s2**2)**((self.p-3)/2.) * self.naive_chisq(s2)
        return dens

    def naive_evaluate(self, nsep=100):
        if nsep % 2 != 0:
            raise ValueError("nsep need to be an even number!")
        dx = 2. / nsep
        fns = np.zeros(nsep+1)
        for i in range(nsep+1):
            fns[i] = self.naive_density(-1.+i*dx)
        nums = np.array([1.] + [4., 2.] * ((nsep-2)/2) + [4., 1.])
        return np.sum(fns * nums) * dx / 3.

    def naive_confidence_interval(self, beta_reference=None, how_many=10., level=0.05, myrange=None):

        if beta_reference is None:
            beta_reference = self.two_stage_ls

        def _root(param):
            self.setup(parameter=param)
            return self.naive_evaluate()-level

        if myrange is None:
            unit = np.std(self.Y)
            upper = bisect(_root, beta_reference, beta_reference+unit*how_many, xtol=1.e-4)
            lower = bisect(_root, beta_reference, beta_reference-unit*how_many, xtol=1.e-4) 
        else:
            [left, right] = myrange
            upper = bisect(_root, beta_reference, left, xtol=1.e-4)
            lower = bisect(_root, beta_reference, right, xtol=1.e-4)            

        return [lower, upper] 



class group_lasso_iv(lasso):

    def __init__(self,
                 Y,
                 D,
                 Z,
                 penalty=None,
                 ridge_term=None,
                 randomizer_scale=None,
                 perturb=None,
                 C0 = None):

        n, p = Z.shape
        self.ZTZ_inv = np.linalg.inv(Z.T.dot(Z))
        sqrtQ = sqrtm(self.ZTZ_inv)
        self.ZTD = Z.T.dot(D)
        data_part = sqrtQ.dot(self.ZTD)
        loglike = rr.glm.gaussian(np.identity(p), data_part)
        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        if C0 is None:
            self.C_0 = 10.
            # C_0 is taken to be 10 for now
        else:
            self.C_0 = C0

        if penalty is None:
            #P_Z = Z.dot(np.linalg.pinv(Z))
            #penalty = np.sqrt(D.T.dot(np.identity(n)-P_Z).dot(D)*p*self.C_0/(n-p))
            penalty = np.sqrt((D.T.dot(D) - self.ZTD.T.dot(self.ZTZ_inv).dot(self.ZTD))*p*self.C_0/(n-p))
        penalty = rr.group_lasso(np.zeros(p), weights=dict({0: penalty}), lagrange=1.)
        self.penalty = penalty

        if ridge_term is None:
            ridge_term = 0.
        self.ridge_term = ridge_term

        mean_diag = 1. # X is identity matrix here
        if p > 1:
            if randomizer_scale is None:
                randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(data_part) * np.sqrt(n / (n - 1.))
            else: 
                randomizer_scale *= np.sqrt(mean_diag) * np.std(data_part * np.sqrt(n / (n - 1.)))
        elif randomizer_scale is None:
            randomizer_scale = 0.5
             
        self.randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        self._initial_omega = perturb

        self.Z = Z
        self.D = D
        self.Y = Y

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        p = self.nfeature
            
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 0, -self._initial_omega, 0)
        problem = rr.simple_problem(self.loglike, self.penalty)
        self.initial_soln = problem.solve(quad, **solve_args)

        if np.linalg.norm(self.initial_soln) < 1e-10:
            #print 'no active set, initial_soln: ', self.initial_soln
            return False

        ## initial state for opt variables
        initial_subgrad = -(self.loglike.smooth_objective(self.initial_soln, 'grad') + 
                            quad.objective(self.initial_soln, 'grad'))
        self.initial_subgrad = initial_subgrad

        initial_scalings = np.linalg.norm(self.initial_soln)
        self.observed_opt_state = np.atleast_1d(initial_scalings)

        # form linear part
        self.num_opt_var = self.observed_opt_state.shape[0]

        _opt_linear_term = np.zeros((p, self.num_opt_var))
        #_score_linear_term = None

        X, y = self.loglike.data
        # note the minus sign of the score
        self.observed_score_state = -y

        self._lagrange = self.penalty.weights[0]
        self.active_directions = initial_subgrad / self._lagrange

        _opt_linear_term = self.active_directions.reshape((-1,1))

        self.opt_transform = (_opt_linear_term, self.initial_subgrad)
        #self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # compute implied mean and covariance

        cov, prec = self.randomizer.cov_prec
        opt_linear, opt_offset = self.opt_transform

        cond_precision = opt_linear.T.dot(opt_linear) * prec
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(opt_linear.T) * prec

        cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

        # log_density should include Jacobian term
        # it is not used in truncated gaussian sampling but in importance weighting
        def log_density(logdens_linear, offset, cond_prec, score, opt):
            if score.ndim == 1:
                mean_term = logdens_linear.dot(score.T + offset).T
            else:
                mean_term = logdens_linear.dot(score.T + offset[:, None]).T
            arg = opt + mean_term
            jacobian_part = self.log_jacobian(opt)

            return (-0.5*np.sum(arg*cond_prec.dot(arg.T).T, 1)+jacobian_part)

        log_density = functools.partial(log_density, logdens_linear, opt_offset, cond_precision)

        # now make the constraints
        A_scaling = -np.identity(self.num_opt_var)
        b_scaling = np.zeros(self.num_opt_var)

        affine_con = constraints(A_scaling,
                                 b_scaling,
                                 mean=cond_mean,
                                 covariance=cond_cov)
        logdens_transform = (logdens_linear, opt_offset)
        self.sampler = affine_gaussian_sampler(affine_con,
                                               self.observed_opt_state,
                                               self.observed_score_state,
                                               log_density,
                                               logdens_transform)

        return True


    def log_jacobian(self, opt):
        n, p = self.Z.shape

        if p == 1:
            return 0.

        V = np.zeros((p, p-1))

        def null(A, eps=1e-12):
            u,s,vh = np.linalg.svd(A)
            padding = max(0, np.shape(A)[1] - np.shape(s)[0])
            null_mask = np.concatenate(((s <= eps), np.ones((padding,), dtype=bool)), axis=0)
            null_space = compress(null_mask, vh, axis=0)
            return transpose(null_space)

        V = null(matrix(self.active_directions))
        component = self._lagrange*V.T.dot(V)
        jacobs = np.array([np.linalg.det(item*np.identity(p-1)+component) for item in opt])

        return np.log(jacobs)


    def summary(self,
                parameter=None,
                Sigma_11 = 1.,
                Sigma_12 = .8,
                level=0.95,
                ndraw=1000,
                burnin=1000,
                compute_intervals=True):

        if parameter is None:
            parameter = np.zeros(1)
        parameter = np.atleast_1d(parameter)

        # compute tsls, i.e. the observed target
        #P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        #denom = self.D.T.dot(P_Z).dot(self.D)
        #two_stage_ls = (self.D.T.dot(P_Z).dot(self.Y)) / denom
        self.ZTY = self.Z.T.dot(self.Y)
        self.DPZD = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTD)
        self.DPZY = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTY)
        self.two_stage_ls = self.DPZY / self.DPZD
        two_stage_ls = np.atleast_1d(self.two_stage_ls)
        observed_target = two_stage_ls

        # compute cov_target, cov_target_score
        X, y = self.loglike.data
        cov_target = np.atleast_2d(Sigma_11/self.DPZD)
        cov_target_score = -1.*(Sigma_12/self.DPZD)*y
        cov_target_score = np.atleast_2d(cov_target_score)

        alternatives = ['twosided']

        opt_sample = self.sampler.sample(ndraw, burnin)

        pivots = self.sampler.coefficient_pvalues(observed_target,
                                                  cov_target,
                                                  cov_target_score,
                                                  parameter=parameter,
                                                  sample=opt_sample,
                                                  alternatives=alternatives)
        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(observed_target, 
                                                       cov_target, 
                                                       cov_target_score, 
                                                       parameter=np.zeros_like(parameter), 
                                                       sample=opt_sample,
                                                       alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.sampler.confidence_intervals(observed_target, 
                                                          cov_target, 
                                                          cov_target_score, 
                                                          sample=opt_sample,
                                                          level=level)

        return pivots, pvalues, intervals


    # here we will return the 2 by 2 matrix
    def estimate_covariance(self):
        #P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        #beta_estim = (self.D.T.dot(P_Z).dot(self.Y)) / (self.D.T.dot(P_Z).dot(self.D))

        n = self.Z.shape[0]
        #X = np.vstack([self.Y-self.D*beta_estim, self.D])
        #cov_estim = X.dot(np.identity(n)-P_Z).dot(X.T) / n

        self.ZTY = self.Z.T.dot(self.Y)
        YPZY = self.ZTY.dot(self.ZTZ_inv).dot(self.ZTY)
        self.DPZD = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTD)
        self.DPZY = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTY)
        YRY = self.Y.dot(self.Y) - YPZY
        DRY = self.D.dot(self.Y) - self.DPZY
        DRD = self.D.dot(self.D) - self.DPZD
        self.two_stage_ls = self.DPZY / self.DPZD
        cov_estim = np.zeros((2,2))
        cov_estim[0,0] = YRY + (self.two_stage_ls**2) * DRD - 2*self.two_stage_ls * DRY
        cov_estim[0,1] = DRY - self.two_stage_ls * DRD
        cov_estim[1,0] = DRY - self.two_stage_ls * DRD
        cov_estim[1,1] = DRD

        return (cov_estim * 1. / n)

    def naive_inference(self, 
                        parameter=None,
                        Sigma_11 = 1.,
                        compute_intervals=False,
                        level=0.95):
        if parameter is None:
            parameter = 0.
        if self.pre_test():
            #P_Z = self.Z.dot(np.linalg.pinv(self.Z))
            #denom = self.D.dot(P_Z).dot(self.D)
            #tsls = (self.D.dot(P_Z).dot(self.Y)) / denom
            #std = np.sqrt(Sigma_11 / denom)
            denom = self.ZTD.T.dot(self.ZTZ_inv).dot(self.ZTD)
            tsls = (self.ZTD.T.dot(self.ZTZ_inv).dot(self.ZTY)) / denom
            std = np.sqrt(Sigma_11 / denom)
            pval = norm.cdf(tsls, loc=parameter, scale=std)
            pval = 2. * min(pval, 1-pval)
            interval = None
            if compute_intervals:
                interval = [tsls - std * norm.ppf(q=(level+1.)/2.), tsls + std * norm.ppf(q=(level+1.)/2.)]
            return pval, interval
        else:
            #print 'did not pass pre test'
            return None, None

    # there is no invalid IV
    @staticmethod
    def bigaussian_instance(n=1000,p=10,
                            s=3,snr=7.,random_signs=False,
                            gsnr = 1., #true gamma parameter
                            beta = 1., #true beta parameter
                            Sigma = np.array([[1., 0.8], [0.8, 1.]]), #noise variance matrix
                            rho=0,scale=False,center=True): #Z matrix structure, note that scale=TRUE will simulate weak IV case!

        # Generate parameters
        # --> gamma coefficient
        gamma = np.repeat([gsnr],p)

        # Generate samples
        # Generate Z matrix 
        Z = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
            np.sqrt(rho) * np.random.standard_normal(n)[:,None])
        if center:
            Z -= Z.mean(0)[None,:]
        if scale:
            Z /= (Z.std(0)[None,:] * np.sqrt(n))
        #    Z /= np.sqrt(n)
        # Generate error term
        mean = [0, 0]
        errorTerm = np.random.multivariate_normal(mean,Sigma,n)
        # Generate D and Y
        D = Z.dot(gamma) + errorTerm[:,1]
        Y = D * beta + errorTerm[:,0]

        D = D - D.mean(0)
        Y = Y - Y.mean(0)
    
        return Z, D, Y, beta, gamma

    def pre_test(self):
        n, p = self.Z.shape
        #P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        #denom = self.D.dot(np.identity(n) - P_Z).dot(self.D) / (n-p)
        #num = self.D.dot(P_Z).dot(self.D) / p
        self.ZTY = self.Z.T.dot(self.Y)
        self.DPZD = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTD)
        self.DPZY = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTY)
        denom = (self.D.dot(self.D) - self.DPZD) / (n-p)
        num = self.DPZD / p
        f = num / denom
        self.ftest = f
        # C_0 is taken to be 10 for now
        #C_0 = 10.
        if f >= self.C_0:
            return True
        else:
            return False


class group_lasso_iv_ar(group_lasso_iv):

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        p = self.nfeature
            
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 0, -self._initial_omega, 0)
        problem = rr.simple_problem(self.loglike, self.penalty)
        self.initial_soln = problem.solve(quad, **solve_args)

        if np.linalg.norm(self.initial_soln) < 1e-10:
            #print 'no active set, initial_soln: ', self.initial_soln
            return False

        ## initial state for opt variables
        initial_subgrad = -(self.loglike.smooth_objective(self.initial_soln, 'grad') + 
                            quad.objective(self.initial_soln, 'grad'))
        self.initial_subgrad = initial_subgrad

        initial_scalings = np.linalg.norm(self.initial_soln)
        self.observed_opt_state = np.atleast_1d(initial_scalings)

        # form linear part
        self.num_opt_var = self.observed_opt_state.shape[0]

        _opt_linear_term = np.zeros((p, self.num_opt_var))
        #_score_linear_term = None

        X, y = self.loglike.data
        # note the minus sign of the score
        self.observed_score_state = -y

        self._lagrange = self.penalty.weights[0]
        self.active_directions = initial_subgrad / self._lagrange

        _opt_linear_term = self.active_directions.reshape((-1,1))

        self.opt_transform = (_opt_linear_term, self.initial_subgrad)
        #self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # compute implied mean and covariance

        cov, prec = self.randomizer.cov_prec
        opt_linear, opt_offset = self.opt_transform

        cond_precision = opt_linear.T.dot(opt_linear) * prec
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(opt_linear.T) * prec

        cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

        # log_density should include Jacobian term
        # it is not used in truncated gaussian sampling but in importance weighting
        def log_density(logdens_linear, offset, cond_prec, score, opt):
            if score.ndim == 1:
                mean_term = logdens_linear.dot(score.T + offset).T
            else:
                mean_term = logdens_linear.dot(score.T + offset[:, None]).T
            arg = opt + mean_term
            jacobian_part = self.log_jacobian(opt)

            return (-0.5*np.sum(arg*cond_prec.dot(arg.T).T, 1)+jacobian_part)

        log_density = functools.partial(log_density, logdens_linear, opt_offset, cond_precision)

        # now make the constraints
        A_scaling = -np.identity(self.num_opt_var)
        b_scaling = np.zeros(self.num_opt_var)

        affine_con = constraints(A_scaling,
                                 b_scaling,
                                 mean=cond_mean,
                                 covariance=cond_cov)
        logdens_transform = (logdens_linear, opt_offset)
        self.sampler = affine_gaussian_sampler_iv(affine_con,
                                               self.observed_opt_state,
                                               self.observed_score_state,
                                               log_density,
                                               logdens_transform)

        return True

    def summary(self,
                parameter=None,
                Sigma_11 = 1.,
                Sigma_12 = .8,
                level=0.95,
                ndraw=1000,
                burnin=1000,
                compute_intervals=False,
                compute_power=False,
                beta_alternative=None):

        if parameter is None: # this is for pivot -- could use true beta^*
            parameter = np.zeros(1)

        parameter = np.atleast_1d(parameter)

        # compute the observed_target for AR statistic
        # target = Z^T (Y-D \beta_0)

        self.ZTZ = self.Z.T.dot(self.Z)
        self.ZTY = self.Z.T.dot(self.Y)
        self.DPZD = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTD)
        self.DPZY = self.ZTD.dot(self.ZTZ_inv).dot(self.ZTY)

        self.K1 = self.ZTY
        self.K2 = self.ZTD
        observed_target = self.K1 - self.K2 * parameter

        # compute cov_target, cov_target_score

        cov_target = self.ZTZ * Sigma_11
        cov_target_score = sqrtm(self.ZTZ)
        cov_target_score *= - Sigma_12

        # tsls estimator as beta reference for confidence interval
        two_stage_ls = self.DPZY / self.DPZD
        observed_target_tsls = self.K1 - self.K2 * two_stage_ls

        # this is for Anderson-Rubin
        alternatives = ['greater']

        opt_sample = self.sampler.sample(ndraw, burnin)

        pivots = self.sampler.coefficient_pvalues(observed_target,
                                                  cov_target,
                                                  cov_target_score,
                                                  self.K2,
                                                  self.test_stat,
                                                  parameter=parameter, 
                                                  sample=opt_sample,
                                                  alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(observed_target, 
                                                       cov_target, 
                                                       cov_target_score, 
                                                       self.K2,
                                                       self.test_stat,
                                                       parameter=np.zeros_like(parameter), 
                                                       sample=opt_sample,
                                                       alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None

        coverage = self.sampler.confidence_interval_coverage(observed_target_tsls,
                                                  cov_target,
                                                  cov_target_score,
                                                  self.K2,
                                                  self.test_stat,
                                                  beta_reference=two_stage_ls,
                                                  beta_candidate=parameter,
                                                  sample=opt_sample,
                                                  level=level)

        if compute_intervals:
            intervals = self.sampler.confidence_intervals(observed_target_tsls, 
                                                          cov_target, 
                                                          cov_target_score, 
                                                          self.K2,
                                                          self.test_stat,
                                                          beta_reference=two_stage_ls,
                                                          sample=opt_sample,
                                                          level=level,
                                                          how_many_sd=4000)
            return pivots, pvalues, intervals

        if compute_power:
            if beta_alternative is None:
                beta_alternative = np.array([i*0.02+parameter[0] for i in range(-5,6) if i != 0])
            powers = []
            for beta in beta_alternative:
                detection = ~self.sampler.confidence_interval_coverage(observed_target_tsls,
                                                  cov_target,
                                                  cov_target_score,
                                                  self.K2,
                                                  self.test_stat,
                                                  beta_reference=two_stage_ls,
                                                  beta_candidate=beta,
                                                  sample=opt_sample,
                                                  level=level)
                powers.append(detection)

            return pivots, pvalues, coverage, powers

        return pivots, pvalues, coverage



    def test_stat(self, parameter, target):
        # target is of n by p
        target = np.atleast_2d(target)

        n, p = self.Z.shape
        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        R_Z = np.identity(n) - P_Z
        denom = (self.Y-self.D*parameter).dot(R_Z).dot(self.Y-self.D*parameter) / (n-p)
        ZTZ_inv = np.linalg.inv(self.Z.T.dot(self.Z))
        result = np.sum(target * (ZTZ_inv.dot(target.T).T), axis=1)
        result = result / p / denom

        return result

    # have not implemented C.I. yet
    def naive_inference(self, 
                        parameter=None,
                        Sigma_11 = 1.,
                        Sigma_12 = .8, 
                        compute_intervals=False,
                        level=0.95):
        if parameter is None:
            parameter = 0.
        if self.pre_test():
            self.ZTY = self.Z.T.dot(self.Y)
            target = self.ZTY - self.ZTD * parameter
            n, p = self.Z.shape
            P_Z = self.Z.dot(np.linalg.pinv(self.Z))
            R_Z = np.identity(n) - P_Z
            denom = (self.Y-self.D*parameter).dot(R_Z).dot(self.Y-self.D*parameter) / (n-p)
            ZTZ_inv = np.linalg.inv(self.Z.T.dot(self.Z))
            tar = target.dot(ZTZ_inv).dot(target) / p / denom
            pval = 1. - f.cdf(tar, dfn=p, dfd=n-p) 
            return pval
        else:
            #print 'did not pass pre test'
            return None




class lasso_iv(lasso):

    r"""
    A class for the LASSO with invalid instrumental variables for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\alpha, \beta} \frac{1}{2} \|P_Z (y-Z\alpha-D\beta)\|^2_2 + 
            \lambda \|\alpha\|_1 - \omega^T(\alpha \beta) + \frac{\epsilon}{2} \|(\alpha \beta)\|^2_2

    where $\lambda$ is `lam`, $\omega$ is a randomization generated below
    and the last term is a small ridge penalty.

    NOTE: use beta_tsls instead of the tsls test statistic itself, such to better fit the package structure

    """

    # add .Z field for lasso IV subclass
    def __init__(self,
                 Y, 
                 D,
                 Z, 
                 penalty=None, 
                 ridge_term=None,
                 randomizer_scale=None):
    
        # form the projected design and response
        P_Z = Z.dot(np.linalg.pinv(Z))
        X = np.hstack([Z, D.reshape((-1,1))])
        P_ZX = P_Z.dot(X)
        P_ZY = P_Z.dot(Y)
        loglike = rr.glm.gaussian(P_ZX, P_ZY)

        n, p = Z.shape

        if penalty is None:
            penalty = 2.01 * np.sqrt(n * np.log(n))
        penalty = np.ones(loglike.shape[0]) * penalty
        penalty[-1] = 0.

        mean_diag = np.mean((P_ZX**2).sum(0))
        if ridge_term is None:
            #ridge_term = 1. * np.sqrt(n)
            ridge_term = (np.std(P_ZY) * np.sqrt(mean_diag) / np.sqrt(n - 1.))

        if randomizer_scale is None:
            #randomizer_scale = 0.5*np.sqrt(n)
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(P_ZY) * np.sqrt(n / (n - 1.))
        else: 
            randomizer_scale *= np.sqrt(mean_diag) * np.std(P_ZY) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p+1,), randomizer_scale)

        lasso.__init__(self, loglike, penalty, ridge_term, randomizer)
        self.Z = Z
        self.D = D
        self.Y = Y


    # this is a direct modification of fit() 
    # to be able to Monte Carlo sample and marginalize inactive subgrads
    # user can call one of fit() or fit_for_marginalize()
    # to condition on or MC marginalize over the inactive subgradients
    def fit_for_marginalize(self, 
                            solve_args={'tol':1.e-12, 'min_its':50}, 
                            perturb=None):

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 0, -self._initial_omega, 0)
        problem = rr.simple_problem(self.loglike, self.penalty)
        self.initial_soln = problem.solve(quad, **solve_args)

        active_signs = np.sign(self.initial_soln)
        active = self._active = active_signs != 0

        self._lagrange = self.penalty.weights
        unpenalized = self._lagrange == 0

        active *= ~unpenalized

        self._overall = overall = (active + unpenalized) > 0
        self._inactive = inactive = ~self._overall
        self._unpenalized = unpenalized

        _active_signs = active_signs.copy()
        _active_signs[unpenalized] = np.nan # don't release sign of unpenalized variables
        self.selection_variable = {'sign':_active_signs,
                                   'variables':self._overall}

        # initial state for opt variables

        initial_subgrad = -(self.loglike.smooth_objective(self.initial_soln, 'grad') + 
                            quad.objective(self.initial_soln, 'grad')) 
        self.initial_subgrad = initial_subgrad

        initial_scalings = np.fabs(self.initial_soln[active])
        initial_unpenalized = self.initial_soln[self._unpenalized]

        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized,
                                                  self.initial_subgrad[self._inactive]], axis=0)

        _beta_unpenalized = restricted_estimator(self.loglike, self._overall, solve_args=solve_args)

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        # observed state for score in internal coordinates

        self.observed_internal_state = np.hstack([_beta_unpenalized,
                                                  -self.loglike.smooth_objective(beta_bar, 'grad')[inactive]])

        # form linear part

        self.num_opt_var = self.observed_opt_state.shape[0]

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, self.num_opt_var))
        _score_linear_term = np.zeros((p, self.num_opt_var))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        est_slice = slice(0, overall.sum())
        X, y = self.loglike.data
        W = self._W = self.loglike.saturated_loss.hessian(X.dot(beta_bar))
        _hessian_active = np.dot(X.T, X[:, active] * W[:, None])
        _hessian_unpen = np.dot(X.T, X[:, unpenalized] * W[:, None])

        _score_linear_term[:, est_slice] = -np.hstack([_hessian_active, _hessian_unpen])

        null_idx = np.arange(overall.sum(), p)
        inactive_idx = np.nonzero(inactive)[0]
        for _i, _n in zip(inactive_idx, null_idx):
            _score_linear_term[_i,_n] = -1

        # set the observed score (data dependent) state

        self.observed_score_state = _score_linear_term.dot(self.observed_internal_state)

        def signed_basis_vector(p, j, s):
            v = np.zeros(p)
            v[j] = s
            return v

        active_directions = np.array([signed_basis_vector(p, j, active_signs[j]) for j in np.nonzero(active)[0]]).T

        scaling_slice = slice(0, active.sum())
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = _hessian_active * active_signs[None, active] + self.ridge_term * active_directions
        _opt_linear_term[:, scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active.sum(), active.sum() + unpenalized.sum())
        unpenalized_directions = np.array([signed_basis_vector(p, j, 1) for j in np.nonzero(unpenalized)[0]]).T
        if unpenalized.sum():
            _opt_linear_term[:, unpenalized_slice] = (_hessian_unpen
                                                      + self.ridge_term * unpenalized_directions) 

        subgrad_idx = range(active.sum() + unpenalized.sum(), active.sum() + inactive.sum() + unpenalized.sum())
        subgrad_slice = slice(active.sum() + unpenalized.sum(), active.sum() + inactive.sum() + unpenalized.sum())
        for _i, _s in zip(inactive_idx, subgrad_idx):
            _opt_linear_term[_i,_s] = 1

        # two transforms that encode score and optimization
        # variable roles 

        _opt_affine_term = np.zeros(p)
        _opt_affine_term[active] = active_signs[active] * self._lagrange[active]

        self.opt_transform = (_opt_linear_term, _opt_affine_term)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self._setup = True
        self.scaling_slice = scaling_slice
        self.unpenalized_slice = unpenalized_slice
        self.ndim = self.loglike.shape[0]

        # compute implied mean and covariance

        cov, prec = self.randomizer.cov_prec
        opt_linear, opt_offset = self.opt_transform

        cond_precision = opt_linear.T.dot(opt_linear) * prec
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(opt_linear.T) * prec

        cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

        def log_density(logdens_linear, offset, cond_prec, score, opt):
            if score.ndim == 1:
                mean_term = logdens_linear.dot(score.T + offset).T
            else:
                mean_term = logdens_linear.dot(score.T + offset[:, None]).T
            arg = opt + mean_term
            return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)
        log_density = functools.partial(log_density, logdens_linear, opt_offset, cond_precision)

        # now make the constraints

        inactive_lagrange = self.penalty.weights[inactive]

        I = np.identity(cond_cov.shape[0])
        A_scaling = -I[self.scaling_slice]
        b_scaling = np.zeros(A_scaling.shape[0])
        A_subgrad = np.vstack([I[subgrad_slice],-I[subgrad_slice]])
        b_subgrad = np.hstack([inactive_lagrange,inactive_lagrange])

        linear_term = np.vstack([A_scaling, A_subgrad])
        offset = np.hstack([b_scaling, b_subgrad])

        affine_con = constraints(linear_term,
                                 offset,
                                 mean=cond_mean,
                                 covariance=cond_cov)

        logdens_transform = (logdens_linear, opt_offset)

        self.sampler = affine_gaussian_sampler(affine_con,
                                               self.observed_opt_state,
                                               self.observed_score_state,
                                               log_density,
                                               logdens_transform,
                                               selection_info=self.selection_variable) # should be signs and the subgradients we've conditioned on
        
        return active_signs


    def summary(self,
                parameter=None,
                Sigma_11=1.,
                level=0.95,
                ndraw=10000, 
                burnin=2000,
                compute_intervals=True,
                compute_power=False,
                beta_alternative=None):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        selected_features : np.bool
            Binary encoding of which features to use in final
            model and targets.

        parameter : np.array
            Hypothesized value for parameter beta_star -- defaults to 0.

        Sigma_11 : true Sigma_11, known for now

        level : float
            Confidence level.

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        """

        if parameter is None: # this is for pivot -- could use true beta^*
            parameter = np.zeros(1)

        parameter = np.atleast_1d(parameter)

        # compute tsls, i.e. the observed_target

        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        #P_ZX, P_ZY = self.loglike.data
        #P_ZD = P_ZX[:,-1]
        #two_stage_ls = (P_ZD.dot(P_Z-P_ZE).dot(self.P_ZY-P_ZD*parameter))/np.sqrt(Sigma_11*P_ZD.dot(P_Z-P_ZE).dot(P_ZD))
        #denom = P_ZD.dot(P_Z - P_ZE).dot(P_ZD)
        #two_stage_ls = (P_ZD.dot(P_Z - P_ZE).dot(P_ZY)) / denom

        denom = self.D.dot(P_Z - P_ZE).dot(self.D)
        two_stage_ls = self.D.dot(P_Z - P_ZE).dot(self.Y) / denom

        two_stage_ls = np.atleast_1d(two_stage_ls)
        observed_target = two_stage_ls

        # only has the parametric version right now
        # compute cov_target, cov_target_score

        cov_target = np.atleast_2d(Sigma_11/denom)
        #score_cov = -1.*np.sqrt(Sigma_11/P_ZD.dot(P_Z-P_ZE).dot(P_ZD))*np.hstack([self.Z.T.dot(P_Z-P_ZE).dot(P_ZD),P_ZD.dot(P_Z-P_ZE).dot(P_ZD)])
        #cov_target_score = -1.*(Sigma_11/denom)*np.hstack([self.Z.T.dot(P_Z-P_ZE).dot(P_ZD),P_ZD.dot(P_Z-P_ZE).dot(P_ZD)])
        cov_target_score = -1.*(Sigma_11/denom)*np.hstack([self.Z.T.dot(P_Z - P_ZE).dot(self.D), self.D.dot(P_Z - P_ZE).dot(self.D)])
        cov_target_score = np.atleast_2d(cov_target_score)

        alternatives = ['twosided']

        opt_sample = self.sampler.sample(ndraw, burnin)

        pivots = self.sampler.coefficient_pvalues(observed_target, 
                                                  cov_target, 
                                                  cov_target_score, 
                                                  parameter=parameter, 
                                                  sample=opt_sample,
                                                  alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(observed_target, 
                                                       cov_target, 
                                                       cov_target_score, 
                                                       parameter=np.zeros_like(parameter), 
                                                       sample=opt_sample,
                                                       alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.sampler.confidence_intervals(observed_target, 
                                                          cov_target, 
                                                          cov_target_score, 
                                                          sample=opt_sample,
                                                          level=level)

        if compute_power:
            if beta_alternative is None:
                beta_alternative = np.array([i*0.02+parameter[0] for i in range(-5,6) if i != 0])
            powers=[~(intervals[0][0]<=beta and beta<=intervals[0][1]) for beta in beta_alternative]

            return pivots, pvalues, intervals, powers

        return pivots, pvalues, intervals

    def estimate_covariance(self):
        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        #P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        ZE = self.Z[:,self._overall[:-1]]
        P_ZE = ZE.dot(np.linalg.inv(ZE.T.dot(ZE))).dot(ZE.T)
        P_diff = P_Z - P_ZE
        beta_estim = (self.D.T.dot(P_diff).dot(self.Y)) / (self.D.T.dot(P_diff).dot(self.D))

        n = self.Z.shape[0]
        cov_estim = (self.Y-self.D*beta_estim).dot(np.identity(n)-P_Z).dot(self.Y-self.D*beta_estim) / n

        return cov_estim


    def naive_inference(self, 
                        parameter=None,
                        Sigma_11 = 1.,
                        compute_intervals=False,
                        level=0.95):

        if parameter is None:
            parameter = 0.
            
        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))

        denom = self.D.dot(P_Z - P_ZE).dot(self.D)
        tsls = self.D.dot(P_Z - P_ZE).dot(self.Y) / denom
        std = np.sqrt(Sigma_11 / denom)
        pval = norm.cdf(tsls, loc=parameter, scale=std)
        pval = 2. * min(pval, 1-pval)
        interval = None
        if compute_intervals:
            interval = [tsls - std * norm.ppf(q=(level+1.)/2.), tsls + std * norm.ppf(q=(level+1.)/2.)]
        return pval, interval


    @staticmethod
    def bigaussian_instance(n=1000,p=10,
                            s=3,snr=7.,random_signs=False, #true alpha parameter
                            gsnr_invalid = 1., #true gamma parameter
                            gsnr_valid = 1.,
                            beta = 1., #true beta parameter
                            Sigma = np.array([[1., 0.8], [0.8, 1.]]), #noise variance matrix
                            rho=0,scale=False,center=True): #Z matrix structure, note that scale=TRUE will simulate weak IV case!

        # Generate parameters
        # --> Alpha coefficients
        alpha = np.zeros(p) 
        alpha[:s] = snr 
        if random_signs:
            alpha[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
        active = np.zeros(p, np.bool)
        active[:s] = True
        # --> gamma coefficient
        #gamma = np.repeat([gsnr],p)
        gamma = np.ones(p)
        gamma[:s] *= gsnr_invalid
        gamma[s:] *= gsnr_valid

        # Generate samples
        # Generate Z matrix 
        Z = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
            np.sqrt(rho) * np.random.standard_normal(n)[:,None])
        if center:
            Z -= Z.mean(0)[None,:]
        if scale:
            Z /= (Z.std(0)[None,:] * np.sqrt(n))
        #    Z /= np.sqrt(n)
        # Generate error term
        mean = [0, 0]
        errorTerm = np.random.multivariate_normal(mean,Sigma,n)
        # Generate D and Y
        D = Z.dot(gamma) + errorTerm[:,1]
        Y = Z.dot(alpha) + D * beta + errorTerm[:,0]
    
        return Z, D, Y, alpha, beta, gamma



class lasso_iv_ar(lasso_iv):

    r"""
    A class for the LASSO with invalid instrumental variables for post-selection inference.
    Specifically sampling for the Anderson-Rubin test statistic, note the sampling structure
    is different from the standard LASSO code.
    The problem solved is

    .. math::

        \text{minimize}_{\alpha, \beta} \frac{1}{2} \|P_Z (y-Z\alpha-D\beta)\|^2_2 + 
            \lambda \|\alpha\|_1 - \omega^T(\alpha \beta) + \frac{\epsilon}{2} \|(\alpha \beta)\|^2_2

    where $\lambda$ is `lam`, $\omega$ is a randomization generated below
    and the last term is a small ridge penalty.

    """

    # # the ONLY change is .sampler line, compared to lasso_iv
    def fit_for_marginalize(self, 
                            solve_args={'tol':1.e-12, 'min_its':50}, 
                            perturb=None):

        p = self.nfeature

        if perturb is None:
            perturb = self.randomizer.sample()
        self._initial_omega = perturb
        quad = rr.identity_quadratic(self.ridge_term, 0, -perturb, 0)
        problem = rr.simple_problem(self.loglike, self.penalty)
        self.initial_soln = problem.solve(quad, **solve_args)

        active_signs = np.sign(self.initial_soln)
        active = self._active = active_signs != 0

        self._lagrange = self.penalty.weights
        unpenalized = self._lagrange == 0

        active *= ~unpenalized

        self._overall = overall = (active + unpenalized) > 0
        self._inactive = inactive = ~self._overall
        self._unpenalized = unpenalized

        _active_signs = active_signs.copy()
        _active_signs[unpenalized] = np.nan # don't release sign of unpenalized variables
        self.selection_variable = {'sign':_active_signs,
                                   'variables':self._overall}

        # initial state for opt variables

        initial_subgrad = -(self.loglike.smooth_objective(self.initial_soln, 'grad') + 
                            quad.objective(self.initial_soln, 'grad')) 
        self.initial_subgrad = initial_subgrad

        initial_scalings = np.fabs(self.initial_soln[active])
        initial_unpenalized = self.initial_soln[self._unpenalized]

        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized,
                                                  self.initial_subgrad[self._inactive]], axis=0)

        _beta_unpenalized = restricted_estimator(self.loglike, self._overall, solve_args=solve_args)

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        # observed state for score in internal coordinates

        self.observed_internal_state = np.hstack([_beta_unpenalized,
                                                  -self.loglike.smooth_objective(beta_bar, 'grad')[inactive]])

        # form linear part

        self.num_opt_var = self.observed_opt_state.shape[0]

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, self.num_opt_var))
        _score_linear_term = np.zeros((p, self.num_opt_var))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        est_slice = slice(0, overall.sum())
        X, y = self.loglike.data
        W = self._W = self.loglike.saturated_loss.hessian(X.dot(beta_bar))
        _hessian_active = np.dot(X.T, X[:, active] * W[:, None])
        _hessian_unpen = np.dot(X.T, X[:, unpenalized] * W[:, None])

        _score_linear_term[:, est_slice] = -np.hstack([_hessian_active, _hessian_unpen])

        null_idx = np.arange(overall.sum(), p)
        inactive_idx = np.nonzero(inactive)[0]
        for _i, _n in zip(inactive_idx, null_idx):
            _score_linear_term[_i,_n] = -1

        # set the observed score (data dependent) state

        self.observed_score_state = _score_linear_term.dot(self.observed_internal_state)

        def signed_basis_vector(p, j, s):
            v = np.zeros(p)
            v[j] = s
            return v

        active_directions = np.array([signed_basis_vector(p, j, active_signs[j]) for j in np.nonzero(active)[0]]).T

        scaling_slice = slice(0, active.sum())
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = _hessian_active * active_signs[None, active] + self.ridge_term * active_directions
        _opt_linear_term[:, scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active.sum(), active.sum() + unpenalized.sum())
        unpenalized_directions = np.array([signed_basis_vector(p, j, 1) for j in np.nonzero(unpenalized)[0]]).T
        if unpenalized.sum():
            _opt_linear_term[:, unpenalized_slice] = (_hessian_unpen
                                                      + self.ridge_term * unpenalized_directions) 

        subgrad_idx = range(active.sum() + unpenalized.sum(), active.sum() + inactive.sum() + unpenalized.sum())
        subgrad_slice = slice(active.sum() + unpenalized.sum(), active.sum() + inactive.sum() + unpenalized.sum())
        for _i, _s in zip(inactive_idx, subgrad_idx):
            _opt_linear_term[_i,_s] = 1

        # two transforms that encode score and optimization
        # variable roles 

        _opt_affine_term = np.zeros(p)
        _opt_affine_term[active] = active_signs[active] * self._lagrange[active]

        self.opt_transform = (_opt_linear_term, _opt_affine_term)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self._setup = True
        self.scaling_slice = scaling_slice
        self.unpenalized_slice = unpenalized_slice
        self.ndim = self.loglike.shape[0]

        # compute implied mean and covariance

        cov, prec = self.randomizer.cov_prec
        opt_linear, opt_offset = self.opt_transform

        cond_precision = opt_linear.T.dot(opt_linear) * prec
        cond_cov = np.linalg.inv(cond_precision)
        logdens_linear = cond_cov.dot(opt_linear.T) * prec

        cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

        def log_density(logdens_linear, offset, cond_prec, score, opt):
            if score.ndim == 1:
                mean_term = logdens_linear.dot(score.T + offset).T
            else:
                mean_term = logdens_linear.dot(score.T + offset[:, None]).T
            arg = opt + mean_term
            return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)
        log_density = functools.partial(log_density, logdens_linear, opt_offset, cond_precision)

        # now make the constraints

        inactive_lagrange = self.penalty.weights[inactive]

        I = np.identity(cond_cov.shape[0])
        A_scaling = -I[self.scaling_slice]
        b_scaling = np.zeros(A_scaling.shape[0])
        A_subgrad = np.vstack([I[subgrad_slice],-I[subgrad_slice]])
        b_subgrad = np.hstack([inactive_lagrange,inactive_lagrange])

        linear_term = np.vstack([A_scaling, A_subgrad])
        offset = np.hstack([b_scaling, b_subgrad])

        affine_con = constraints(linear_term,
                                 offset,
                                 mean=cond_mean,
                                 covariance=cond_cov)

        logdens_transform = (logdens_linear, opt_offset)

        self.sampler = affine_gaussian_sampler_iv(affine_con,
                                               self.observed_opt_state,
                                               self.observed_score_state,
                                               log_density,
                                               logdens_transform,
                                               selection_info=self.selection_variable) # should be signs and the subgradients we've conditioned on
        
        return active_signs

    # can have infinite intervals, easier to compute coverage given beta star rather than the actual interval
    def summary(self,
                parameter=None,
                Sigma_11=1.,
                level=0.95,
                ndraw=10000, 
                burnin=2000,
                compute_intervals=False,
                compute_power=False,
                beta_alternative=None):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        selected_features : np.bool
            Binary encoding of which features to use in final
            model and targets.

        parameter : np.array
            Hypothesized value for parameter beta_star -- defaults to 0.

        Sigma_11 : true Sigma_11, known for now

        level : float
            Confidence level.

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        """

        if parameter is None: # this is for pivot -- could use true beta^*
            parameter = np.zeros(1)

        parameter = np.atleast_1d(parameter)

        # compute the observed_target for AR statistic
        # target = Z^T (I - P_{Z_E}) (Y-D \beta_0)

        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        #P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        ZE = self.Z[:,self._overall[:-1]]
        P_ZE = ZE.dot(np.linalg.inv(ZE.T.dot(ZE))).dot(ZE.T)
        n, _ = self.Z.shape
        R_ZE = np.identity(n) - P_ZE

        self.K1 = self.Z.T.dot(R_ZE).dot(self.Y)
        self.K2 = self.Z.T.dot(R_ZE).dot(self.D)
        #observed_target = self.Z.T.dot(R_ZE).dot(self.Y - self.D * parameter)
        observed_target = self.K1 - self.K2 * parameter

        # only has the parametric version right now
        # compute cov_target, cov_target_score

        cov_target = self.Z.T.dot(R_ZE).dot(self.Z) * Sigma_11
        cov_target_score = np.hstack([self.Z, self.D.reshape((-1,1))]).T.dot(R_ZE).dot(self.Z)
        cov_target_score *= - Sigma_11

        # tsls estimator as beta reference for confidence interval
        denom = self.D.dot(P_Z - P_ZE).dot(self.D)
        two_stage_ls = self.D.dot(P_Z - P_ZE).dot(self.Y) / denom
        observed_target_tsls = self.K1 - self.K2 * two_stage_ls

        # this is for Anderson-Rubin
        alternatives = ['greater']

        opt_sample = self.sampler.sample(ndraw, burnin)

        pivots = self.sampler.coefficient_pvalues(observed_target,
                                                  cov_target,
                                                  cov_target_score,
                                                  self.K2,
                                                  self.test_stat,
                                                  parameter=parameter, 
                                                  sample=opt_sample,
                                                  alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(observed_target, 
                                                       cov_target, 
                                                       cov_target_score, 
                                                       self.K2,
                                                       self.test_stat,
                                                       parameter=np.zeros_like(parameter), 
                                                       sample=opt_sample,
                                                       alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.sampler.confidence_intervals(observed_target_tsls, 
                                                          cov_target, 
                                                          cov_target_score, 
                                                          self.K2,
                                                          self.test_stat,
                                                          beta_reference=two_stage_ls,
                                                          sample=opt_sample,
                                                          level=level,
                                                          how_many_sd=4000)

        coverage = self.sampler.confidence_interval_coverage(observed_target_tsls,
                                                  cov_target,
                                                  cov_target_score,
                                                  self.K2,
                                                  self.test_stat,
                                                  beta_reference=two_stage_ls,
                                                  beta_candidate=parameter,
                                                  sample=opt_sample,
                                                  level=level)

        if compute_power:
            if beta_alternative is None:
                beta_alternative = np.array([i*0.02+parameter[0] for i in range(-5,6) if i != 0])
            powers = []
            for beta in beta_alternative:
                detection = ~self.sampler.confidence_interval_coverage(observed_target_tsls,
                                                  cov_target,
                                                  cov_target_score,
                                                  self.K2,
                                                  self.test_stat,
                                                  beta_reference=two_stage_ls,
                                                  beta_candidate=beta,
                                                  sample=opt_sample,
                                                  level=level)
                powers.append(detection)

            return pivots, pvalues, coverage, powers

        return pivots, pvalues, coverage


    def test_stat(self, parameter, target):
        # target is of n by p
        target = np.atleast_2d(target)

        n, p = self.Z.shape
        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        R_Z = np.identity(n) - P_Z
        denom = (self.Y-self.D*parameter).dot(R_Z).dot(self.Y-self.D*parameter) / (n-p)
        ZTZ_inv = np.linalg.inv(self.Z.T.dot(self.Z))
        result = np.sum(target * (ZTZ_inv.dot(target.T).T), axis=1)
        result = result / (p-np.sum(self._overall[:-1])) / denom

        return result

    def naive_inference(self,
                        parameter=None,
                        Sigma_11 = 1.,
                        compute_intervals=False,
                        level=0.95):

        if parameter is None:
            parameter = 0.

        P_Z = self.Z.dot(np.linalg.pinv(self.Z))
        #P_ZE = self.Z[:,self._overall[:-1]].dot(np.linalg.pinv(self.Z[:,self._overall[:-1]]))
        ZE = self.Z[:,self._overall[:-1]]
        P_ZE = ZE.dot(np.linalg.inv(ZE.T.dot(ZE))).dot(ZE.T)
        n, p = self.Z.shape
        P_Zperp = np.identity(n) - P_Z
        nactive = sum(self._overall) - sum(self._unpenalized)
        denom = (self.Y-self.D*parameter).dot(P_Zperp).dot(self.Y-self.D*parameter) / (n-p)
        numerator = (self.Y-self.D*parameter).dot(P_Z - P_ZE).dot(self.Y-self.D*parameter) / (p-nactive)
        tar_obs = numerator / denom
        pval = 1. - f.cdf(tar_obs, p-nactive, n-p)
        return pval


class affine_gaussian_sampler_iv(affine_gaussian_sampler):

    def confidence_interval_coverage(self,
                                     observed_target,
                                     target_cov,
                                     score_cov,
                                     linear_func,
                                     test_stat,
                                     beta_reference,
                                     beta_candidate,
                                     sample_args=(),
                                     sample=None,
                                     level=0.95):

        if sample is None:
            sample = self.sample(*sample_args)
        else:
            ndraw = sample.shape[0]

        _intervals = optimization_intervals_iv([(self, sample, target_cov, score_cov)],
                                            observed_target, ndraw)


        pvalue = _intervals.pivot(linear_func,beta_reference,beta_candidate-beta_reference,test_stat,alternative='greater')
        coverage = pvalue >= (1.-level)

        return coverage

    def confidence_intervals(self,
                             observed_target,
                             target_cov,
                             score_cov,
                             linear_func,
                             test_stat,
                             beta_reference,
                             sample_args=(),
                             sample=None,
                             level=0.95,
                             how_many_sd=20,
                             how_many_steps=100):
        '''

        Parameters
        ----------

        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.

        sample_args : sequence
           Arguments to `self.sample` if sample is None.

        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc.

        level : float (optional)
            Specify the
            confidence level.

        Notes
        -----

        Construct selective confidence intervals
        for each parameter of the target.

        Returns
        -------

        intervals : [(float, float)]
            List of confidence intervals.
        '''

        if sample is None:
            sample = self.sample(*sample_args)
        else:
            ndraw = sample.shape[0]

        _intervals = optimization_intervals_iv([(self, sample, target_cov, score_cov)],
                                            observed_target, ndraw)

        limits = []

        limits.append(_intervals.confidence_interval(linear_func, test_stat, 
            beta_reference, level=level, how_many_sd=how_many_sd, how_many_steps=how_many_steps))

        return np.array(limits)

    def coefficient_pvalues(self,
                            observed_target,
                            target_cov,
                            score_cov,
                            linear_func,
                            test_stat,
                            parameter=None,
                            sample_args=(),
                            sample=None,
                            alternatives=None):
        '''
        Construct selective p-values
        for each parameter of the target.

        Parameters
        ----------

        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.

        parameter : np.float (optional)
            A vector of parameters with shape `self.shape`
            at which to evaluate p-values. Defaults
            to `np.zeros(self.shape)`.

        sample_args : sequence
           Arguments to `self.sample` if sample is None.

        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc.

        alternatives : list of ['greater', 'less', 'twosided']
            What alternative to use.

        Returns
        -------
        pvalues : np.float

        '''

        if alternatives is None:
            #alternatives = ['twosided'] * observed_target.shape[0]
            alternatives = ['greater']

        if sample is None:
            sample = self.sample(*sample_args)
        else:
            ndraw = sample.shape[0]

        if parameter is None:
            #parameter = np.zeros(observed_target.shape[0])
            parameter = np.zeros(1)

        _intervals = optimization_intervals_iv([(self, sample, target_cov, score_cov)],
                                            observed_target, ndraw)
        pvals = []

        pvals.append(_intervals.pivot(linear_func, parameter, 0., test_stat, 'greater'))

        return np.array(pvals)


class optimization_intervals_iv(optimization_intervals):

    def pivot(self,
              linear_func,
              parameter_reference, 
              candidate,
              test_stat,
              alternative='twosided'):
        '''
        alternative : ['greater', 'less', 'twosided']
            What alternative to use.
        Returns
        -------
        pvalue : np.float
        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        # observed target at beta_reference
        observed_target = self.observed
        sample_target = self._normal_sample

        target_cov = self.target_cov

        nuisance = []
        translate_dirs = []
        translate_dirs_candidate = []
        for opt_sampler, opt_sample, _, score_cov in self.opt_sampling_info:
            #cur_score_cov = linear_func.dot(score_cov)
            cur_score_cov = score_cov

            # cur_nuisance is in the view's score coordinates
            cur_nuisance = opt_sampler.observed_score_state - cur_score_cov.dot(np.linalg.inv(target_cov)).dot(observed_target)
            nuisance.append(cur_nuisance)
            translate_dirs.append(cur_score_cov.dot(np.linalg.inv(target_cov)))
            translate_dirs_candidate.append(cur_score_cov.dot(np.linalg.inv(target_cov)).dot(linear_func))


        weights = self._weights(sample_target,  # normal sample under zero
                                candidate, # the difference beta-beta_reference
                                nuisance,                 # nuisance sufficient stats for each view
                                translate_dirs,           # points will be moved like sample * score_cov
                                translate_dirs_candidate)               
        
        
        #pivot = np.mean((sample_stat + candidate <= observed_stat) * weights) / np.mean(weights)
        beta = parameter_reference + candidate
        observed_target_beta = observed_target - linear_func*candidate
        observed_stat = test_stat(beta, observed_target_beta)
        sample_stat = test_stat(beta, sample_target)
        pivot = np.mean((sample_stat <= observed_stat) * weights) / np.mean(weights)

        if alternative == 'twosided':
            return 2 * min(pivot, 1 - pivot)
        elif alternative == 'less':
            return pivot
        else:
            return 1 - pivot

    def confidence_interval(self, linear_func, test_stat, beta_reference, level=0.95, how_many_sd=20, how_many_steps=10):

        sample_stat = test_stat(beta_reference, self._normal_sample)
        #observed_stat = test_stat(self.observed.dot(linear_func))
        
        #_norm = np.linalg.norm(linear_func)
        grid_min, grid_max = -how_many_sd * np.std(sample_stat), how_many_sd * np.std(sample_stat)

        def _root(gamma):
            return self.pivot(linear_func,
                              beta_reference,
                              gamma,
                              test_stat,
                              alternative='greater') - (1 - level)

        # debugging...
        #upper = bisect(_root, 0., grid_max, xtol=1.e-5*(grid_max - grid_min))
        #lower = bisect(_root, grid_min, 0., xtol=1.e-5*(grid_max - grid_min))
        betas = np.linspace(grid_min, grid_max, how_many_steps)
        pivots = [_root(beta)+(1-level) for beta in betas]

        #return lower + beta_reference, upper + beta_reference
        return pivots

    # Private methods

    def _weights(self, 
                 sample_target,
                 candidate,
                 nuisance,
                 translate_dirs,
                 translate_dirs_candidate):

        # Here we should loop through the views
        # and move the score of each view 
        # for each projected (through linear_func) normal sample
        # using the linear decomposition

        # We need access to the map that takes observed_score for each view
        # and constructs the full randomization -- this is the reconstruction map
        # for each view

        # The data state for each view will be set to be N_i + A_i \hat{\theta}_i
        # where N_i is the nuisance sufficient stat for the i-th view's
        # data with respect to \hat{\theta} and N_i  will not change because
        # it depends on the observed \hat{\theta} and observed score of i-th view

        # In this function, \hat{\theta}_i will change with the Monte Carlo sample

        score_sample = []
        _lognum = 0
        for i, opt_info in enumerate(self.opt_sampling_info):
            opt_sampler, opt_sample = opt_info[:2]
            score_sample = translate_dirs[i].dot(sample_target.T).T + nuisance[i][None, :] # these are now score coordinates
            score_sample += (translate_dirs_candidate[i]*candidate)[None, :]
            _lognum += opt_sampler.log_density(score_sample, opt_sample)

        _logratio = _lognum - self._logden
        _logratio -= _logratio.max()

        return np.exp(_logratio)



# using only summary statistics
# summary data:
# Ghat: Y ~ Z
# ghat: D ~ Z
# Gse, gse
# set $Z_1^T Z_1 = 1$ since the factor does not matter for the inference result
class lasso_iv_summary(lasso_iv):

    r"""
    A class for the LASSO with invalid instrumental variables for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\alpha, \beta} \frac{1}{2} \|P_Z (y-Z\alpha-D\beta)\|^2_2 + 
            \lambda \|\alpha\|_1 - \omega^T(\alpha \beta) + \frac{\epsilon}{2} \|(\alpha \beta)\|^2_2

    where $\lambda$ is `lam`, $\omega$ is a randomization generated below
    and the last term is a small ridge penalty.

    NOTE: use beta_tsls instead of the tsls test statistic itself, such to better fit the package structure
    this works specifically for summary data and is a direct modification of lasso_iv class

    """

    def __init__(self,
                 Ghat, 
                 ghat,
                 Gse,
                 gse,
                 n,
                 penalty=None, 
                 ridge_term=None,
                 randomizer_scale=None):
    
        # form the projected design and response
        p = Ghat.shape[0]
        self.n = n
        self.p = p

        Gks = (n - 1.) * Gse**2 + Ghat**2
        self.ZTZ_diag = Gks[0] / Gks
        sqrt_ZTZ_diag = np.sqrt(self.ZTZ_diag)
        self.ZTY = Ghat * self.ZTZ_diag
        self.ZTD = ghat * self.ZTZ_diag

        #X = np.hstack([np.identity(p), ghat.reshape((-1,1))])
        #Y = Ghat
        X = np.hstack([np.diag(sqrt_ZTZ_diag), (self.ZTD / sqrt_ZTZ_diag).reshape((-1, 1))])
        Y = self.ZTY / sqrt_ZTZ_diag
        loglike = rr.glm.gaussian(X, Y)

        if penalty is None:
            penalty = 2.01 #* np.sqrt(n * np.log(n))
        penalty = np.ones(loglike.shape[0]) * penalty
        penalty[-1] = 0.

        mean_diag = np.mean((X**2).sum(0))
        if ridge_term is None:
            #ridge_term = 1. * np.sqrt(n)
            ridge_term = (np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1.))

        if randomizer_scale is None:
            #randomizer_scale = 0.5*np.sqrt(n)
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))
        #else: 
            #randomizer_scale *= np.sqrt(mean_diag) * np.std(Y) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p+1,), randomizer_scale)

        lasso.__init__(self, loglike, penalty, ridge_term, randomizer)

        self.Ghat = Ghat
        self.ghat = ghat
        self.Gse = Gse
        self.gse = gse

    def summary(self,
                parameter=None,
                Sigma_11=1.,
                level=0.95,
                ndraw=10000, 
                burnin=2000,
                compute_intervals=True,
                compute_power=False,
                beta_alternative=None):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        selected_features : np.bool
            Binary encoding of which features to use in final
            model and targets.

        parameter : np.array
            Hypothesized value for parameter beta_star -- defaults to 0.

        Sigma_11 : true Sigma_11, known for now

        level : float
            Confidence level.

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        """

        p = self.Ghat.shape[0]

        if parameter is None: # this is for pivot -- could use true beta^*
            parameter = np.zeros(1)

        parameter = np.atleast_1d(parameter)

        # compute tsls, i.e. the observed_target

        #self.GhatE = self.Ghat[self._overall[:-1]]
        self.ghatE = self.ghat[self._overall[:-1]]
        #denom = self.ghat.dot(self.ghat)-self.ghatE.dot(self.ghatE)
        #two_stage_ls = (self.ghat.dot(self.Ghat)-self.ghatE.dot(self.GhatE))/denom
        self.ZTYE = self.ZTY[self._overall[:-1]]
        self.ZTDE = self.ZTD[self._overall[:-1]]
        self.ZTZE = self.ZTZ_diag[self._overall[:-1]]
        self.ZTZ_inv_diag = 1. / self.ZTZ_diag
        self.ZTZE_inv_diag = self.ZTZ_inv_diag[self._overall[:-1]]
        denom = np.sum(self.ZTD**2 * self.ZTZ_inv_diag) - np.sum(self.ZTDE**2 * self.ZTZE_inv_diag)
        num = np.sum(self.ZTD * self.ZTZ_inv_diag * self.ZTY) - np.sum(self.ZTDE * self.ZTZE_inv_diag * self.ZTYE)
        self.two_stage_ls = num / denom
        two_stage_ls = np.atleast_1d(self.two_stage_ls)
        observed_target = two_stage_ls

        # only has the parametric version right now
        # compute cov_target, cov_target_score

        cov_target = np.atleast_2d(Sigma_11/denom)
        tmp = np.zeros(p)
        #tmp[self._overall[:-1]] = self.ghatE
        #cov_target_score = -1.*(Sigma_11/denom)*np.hstack([self.ghat - tmp, denom])\
        tmp[self._overall[:-1]] = self.ZTZE * self.ghatE
        cov_target_score = -1.*(Sigma_11/denom)*np.hstack([self.ZTD - tmp, denom])
        cov_target_score = np.atleast_2d(cov_target_score)

        alternatives = ['twosided']

        opt_sample = self.sampler.sample(ndraw, burnin)

        pivots = self.sampler.coefficient_pvalues(observed_target, 
                                                  cov_target, 
                                                  cov_target_score, 
                                                  parameter=parameter, 
                                                  sample=opt_sample,
                                                  alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(observed_target, 
                                                       cov_target, 
                                                       cov_target_score, 
                                                       parameter=np.zeros_like(parameter), 
                                                       sample=opt_sample,
                                                       alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.sampler.confidence_intervals(observed_target, 
                                                          cov_target, 
                                                          cov_target_score, 
                                                          sample=opt_sample,
                                                          level=level)

        if compute_power:
            if beta_alternative is None:
                beta_alternative = np.array([i*0.02+parameter[0] for i in range(-5,6) if i != 0])
            powers=[~(intervals[0][0]<=beta and beta<=intervals[0][1]) for beta in beta_alternative]

            return pivots, pvalues, intervals, powers

        return pivots, pvalues, intervals

    # seG, seg are se(Ghat), se(ghat)
    def estimate_covariance(self):

        self.ZTYE = self.ZTY[self._overall[:-1]]
        self.ZTDE = self.ZTD[self._overall[:-1]]
        self.ZTZE = self.ZTZ_diag[self._overall[:-1]]
        self.ZTZ_inv_diag = 1. / self.ZTZ_diag
        self.ZTZE_inv_diag = self.ZTZ_inv_diag[self._overall[:-1]]
        denom = np.sum(self.ZTD**2 * self.ZTZ_inv_diag) - np.sum(self.ZTDE**2 * self.ZTZE_inv_diag)
        num = np.sum(self.ZTD * self.ZTZ_inv_diag * self.ZTY) - np.sum(self.ZTDE * self.ZTZE_inv_diag * self.ZTYE)
        beta_estim = num / denom

        #self.GhatE = self.Ghat[self._overall[:-1]]
        #self.ghatE = self.ghat[self._overall[:-1]]
        #denom = self.ghat.dot(self.ghat)-self.ghatE.dot(self.ghatE)
        #beta_estim = (self.ghat.dot(self.Ghat)-self.ghatE.dot(self.GhatE))/denom

        YTY = (self.n-1.)*self.Gse[0]**2+self.Ghat[0]**2
        DTD = (self.n-1.)*self.gse[0]**2+self.ghat[0]**2
        Omega_11 = (YTY - np.sum(self.ZTY**2/self.ZTZ_diag)) / (self.n - self.p + 1.)
        Omega_22 = (DTD - np.sum(self.ZTD**2/self.ZTZ_diag)) / (self.n - self.p + 1.)
        Omega = np.diag([Omega_11, Omega_22])
        coef_mat = np.identity(2)
        coef_mat[0,1] = -beta_estim
        Sigma = coef_mat.dot(Omega).dot(coef_mat.T)
        self.Sigma = Sigma
        cov_estim = Sigma[0,0]

        return cov_estim

    def naive_inference(self, 
                        parameter=None,
                        Sigma_11 = 1.,
                        compute_intervals=True,
                        level=0.95):

        if parameter is None:
            parameter = 0.

        #self.GhatE = self.Ghat[self._overall[:-1]]
        #self.ghatE = self.ghat[self._overall[:-1]]
        #denom = self.ghat.dot(self.ghat)-self.ghatE.dot(self.ghatE)
        #tsls = (self.ghat.dot(self.Ghat)-self.ghatE.dot(self.GhatE))/denom 

        self.ZTYE = self.ZTY[self._overall[:-1]]
        self.ZTDE = self.ZTD[self._overall[:-1]]
        self.ZTZE = self.ZTZ_diag[self._overall[:-1]]
        self.ZTZ_inv_diag = 1. / self.ZTZ_diag
        self.ZTZE_inv_diag = self.ZTZ_inv_diag[self._overall[:-1]]
        denom = np.sum(self.ZTD**2 * self.ZTZ_inv_diag) - np.sum(self.ZTDE**2 * self.ZTZE_inv_diag)
        num = np.sum(self.ZTD * self.ZTZ_inv_diag * self.ZTY) - np.sum(self.ZTDE * self.ZTZE_inv_diag * self.ZTYE)
        tsls = num / denom        
        std = np.sqrt(Sigma_11 / denom)
        pval = norm.cdf(tsls, loc=parameter, scale=std)
        pval = 2. * min(pval, 1-pval)
        interval = None
        if compute_intervals:
            interval = [tsls - std * norm.ppf(q=(level+1.)/2.), tsls + std * norm.ppf(q=(level+1.)/2.)]
        return pval, interval

