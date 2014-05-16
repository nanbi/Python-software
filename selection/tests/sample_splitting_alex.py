from rpy2.robjects.packages import importr
import os
import numpy as np
from scipy.stats import norm as ndist
from selection.covtest import reduced_covtest, covtest
from selection.affine import constraints, gibbs_test
from selection.forward_step import forward_stepwise
n, p, sigma = 50, 80, 1.0

from multi_forward_step import forward_step

def sample_split(X, Y, sigma=None,
                 nstep=10,
                 burnin=1000,
                 ndraw=5000,
                 reduced=True):

    n, p = X.shape
    half_n = int(n/2)
    X1, Y1 = X[:half_n,:]*1., Y[:half_n]*1.
    X1 -= X1.mean(0)[None,:]
    Y1 -= Y1.mean()

    X2, Y2 = X[half_n:], Y[half_n:]
    X2 -= X2.mean(0)[None,:]
    Y2 -= Y2.mean()

    FS_half = forward_stepwise(X1, Y1) # sample splitting model
    FS_full = forward_stepwise(X.copy(), Y.copy()) # full data model
    
    spacings_P = []
    split_P = []
    reduced_Pknown = []
    reduced_Punknown = []
    covtest_P = []

    for i in range(nstep):

        FS_half.next()

        if FS_half.P[i] is not None:
            RX = FS_half.X - FS_half.P[i](FS_half.X)
            RY = FS_half.Y - FS_half.P[i](FS_half.Y)
            covariance = centering(FS_half.Y.shape[0]) - np.dot(FS_half.P[i].U, FS_half.P[i].U.T)
        else:
            RX = FS_half.X
            RY = FS_half.Y
            covariance = centering(FS_half.Y.shape[0])

        RX -= RX.mean(0)[None,:]
        RX /= (RX.std(0)[None,:] * np.sqrt(RX.shape[0]))

        # covtest on half -- not saved

        con, pval, idx, sign = covtest(RX, RY, sigma=sigma,
                                       covariance=covariance,
                                       exact=True)

        # spacings on half -- not saved

        eta1 = RX[:,idx] * sign
        Acon = constraints(FS_half.A, np.zeros(FS_half.A.shape[0]),
                           covariance=centering(FS_half.Y.shape[0]))
        Acon.covariance *= sigma**2
        Acon.pivot(eta1, FS_half.Y)

        # sample split

        eta2 = np.linalg.pinv(X2[:,FS_half.variables])[-1]
        eta_sigma = np.linalg.norm(eta2) * sigma
        split_P.append(2*ndist.sf(np.fabs((eta2*Y2).sum() / eta_sigma)))

        # inference on full mu using split model, this \beta^+_s.

        zero_block = np.zeros((Acon.linear_part.shape[0], (n-half_n)))
        linear_part = np.hstack([Acon.linear_part, zero_block])
        Fcon = constraints(linear_part, Acon.offset,
                           covariance=centering(n))
        Fcon.covariance *= sigma**2

        if i > 0:
            U = np.linalg.pinv(X[:,FS_half.variables[:-1]])
            Uy = np.dot(U, Y)
            Fcon = Fcon.conditional(U, Uy)
        else:
            Fcon = Fcon

        eta_full = np.linalg.pinv(X[:,FS_half.variables])[-1]

        if reduced:
            reduced_pval = gibbs_test(Fcon, Y, eta_full,
                                      ndraw=ndraw,
                                      burnin=burnin,
                                      sigma_known=sigma is not None,
                                      alternative='twosided')[0]
            reduced_Pknown.append(reduced_pval)

            reduced_pval = gibbs_test(Fcon, Y, eta_full,
                                      ndraw=ndraw,
                                      burnin=burnin,
                                      sigma_known=False,
                                      alternative='twosided')[0]
            reduced_Punknown.append(reduced_pval)


        # now use all the data

        FS_full.next()
        if FS_full.P[i] is not None:
            RX = X - FS_full.P[i](X)
            RY = Y - FS_full.P[i](Y)
            covariance = centering(RY.shape[0]) - np.dot(FS_full.P[i].U, FS_full.P[i].U.T)
        else:
            RX = X
            RY = Y.copy()
            covariance = centering(RY.shape[0])
        RX -= RX.mean(0)[None,:]
        RX /= RX.std(0)[None,:]

        con, pval, idx, sign = covtest(RX, RY, sigma=sigma,
                                       covariance=covariance,
                                       exact=False)
        covtest_P.append(pval)

        # spacings on full data

        eta1 = RX[:,idx] * sign
        Acon = constraints(FS_full.A, np.zeros(FS_full.A.shape[0]),
                           centering(RY.shape[0]))
        Acon.covariance *= sigma**2
        spacings_P.append(Acon.pivot(eta1, Y))

    return split_P, reduced_Pknown, reduced_Punknown, spacings_P, covtest_P, FS_half.variables

def centering(n):
    return np.identity(n) - np.ones((n,n)) / n

import rpy2.robjects as rpy
from rpy2.robjects.numpy2ri import numpy2ri
rpy.conversion.py2ri = numpy2ri

def instance():

    rpy.r('''
n <- 200  # number of observations
p <- 100  # number of variables
k <- 5   # number of signal variables
sigma <- 1 # noise standard deviation

MAX_STEPS <- 25  # set to p in actual simulations

# Parameter vector
beta_factor <- 0.35 # signal factor (others are 0.25, 0.175)
scale_factor <- sqrt(2)*gamma((n+1)/2)/gamma(n/2)
beta <- beta_factor*scale_factor*c(seq(2,sqrt(2*log(p)), length.out=k), rep(0,p-k))

# Generate data
X <- matrix(rnorm(p*n), n, p)  # iid design
# Standardize X
mx<-colMeans(X); sx<-sqrt(apply(X,2,var)); X<-scale(X,mx,sx)/sqrt(n-1)

y <- X%*%beta + sigma*rnorm(n)
''')

    X = np.array(rpy.r('X'))
    y = np.array(rpy.r('y')).reshape(-1)
    k = np.array(rpy.r('k')).reshape(-1)
    print X.shape, y.shape
    return X, y, int(k[0])

def simulation(n, p, sigma, nsim=500, label=0,
               reduced=True,
               reduced_full=True): # nnz = number nonzero

    splitP = []
    covtestP = []
    spacings = []
    reduced_known = []
    reduced_unknown = []
    reduced_known_full = []
    reduced_unknown_full = []
    hypotheses = []
    hypotheses_full = []

    for i in range(nsim):
        X, Y, nnz = instance()
        Y -= Y.mean()

        split = sample_split(X.copy(),
                             Y.copy(),
                             sigma=sigma,
                             burnin=1000,
                             ndraw=2000,
                             nstep=10,
                             reduced=reduced)
        splitP.append(split[0])
        reduced_known.append(split[1])
        reduced_unknown.append(split[2])
        spacings.append(split[3])
        covtestP.append(split[4])
        hypotheses.append([var in range(nnz) for var in split[5]])

        if reduced_full:
            fs = forward_step(X, Y,
                              sigma=sigma,
                              burnin=1000,
                              ndraw=2000,
                              nstep=10)
            reduced_known_full.append(fs[1])
            reduced_unknown_full.append(fs[2])
            hypotheses_full.append([var in range(nnz) for var in fs[4]])


        for D, name in zip([splitP, spacings, covtestP], ['split', 'spacings', 'covtest']):
            means = map(lambda x: x[~np.isnan(x)].mean(), np.array(D).T)[:(nnz+3)]
            SDs = map(lambda x: x[~np.isnan(x)].std(), np.array(D).T)[:(nnz+3)]
            print means, SDs, name

        if reduced:
            print (np.mean(np.array(reduced_known)[:,:(nnz+3)],0), 
                   np.std(np.array(reduced_known)[:,:(nnz+3)],0), 'reduced known split')
            print (np.mean(np.array(reduced_unknown)[:,:(nnz+3)],0), 
                   np.std(np.array(reduced_unknown)[:,:(nnz+3)],0), 'reduced unknown split'), i

        if reduced_full:
            print (np.mean(np.array(reduced_unknown_full)[:,:(nnz+3)],0), 
                   np.std(np.array(reduced_unknown_full)[:,:(nnz+3)],0), 'reduced unknown full'), i
            print (np.mean(np.array(reduced_known_full)[:,:(nnz+3)],0), 
                   np.std(np.array(reduced_known_full)[:,:(nnz+3)],0), 'reduced known full'), i

        if reduced:
            np.save('reduced_split_known_alex%d.npy' % (label,), np.array(reduced_known))
            np.save('reduced_split_unknown_alex%d.npy' % (label,), np.array(reduced_unknown))

        np.save('split_alex%d.npy' % (label,), np.array(splitP))
        np.save('spacings_split_alex%d.npy' % (label,), np.array(spacings))
        np.save('covtest_split_alex%d.npy' % (label,), np.array(covtestP))
        np.save('hypotheses_split__alex%d.npy' % (label,), np.array(hypotheses))

        if reduced_full:
            np.save('hypotheses_splitfull__alex%d.npy' % (label,), np.array(hypotheses_full))
            np.save('reduced_splitfull_known_alex%d.npy' % (label,), np.array(reduced_known_full))
            np.save('reduced_splitfull_unknown_alex%d.npy' % (label,), np.array(reduced_unknown_full))
        #os.system('cp *split*npy ~/Dropbox/sample_split')

if __name__ == "__main__":

    import sys
    if len(sys.argv) > 1:
        simulation(n, p, sigma, label=int(sys.argv[1]), reduced_full=True, reduced=False)
