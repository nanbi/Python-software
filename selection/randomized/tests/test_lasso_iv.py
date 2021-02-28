import numpy as np

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso_iv import lasso_iv, lasso_iv_ar, group_lasso_iv, group_lasso_iv_ar, weak_iv_clr, lasso_iv_summary
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

################################################################################

#  test file for lasso_iv class

# include the screening in here

################################################################################


############################################### Weak IV #########################################################


def test_weak_iv_clr(nsim=500, n=1000, p=10, Sigma_12=0.8, gsnr=1., beta_star=1., nsep=100):
    P0 = []
    #coverages = []
    #lengths = []
    for i in range(nsim):
        Z, D, Y, beta, gamma = group_lasso_iv.bigaussian_instance(n=n, p=p, beta=beta_star, gsnr=gsnr, 
            Sigma = np.array([[1., Sigma_12], [Sigma_12, 1.]]))
        conv = weak_iv_clr(Y, D, Z)
        if not conv.pre_test():
            conv.setup(parameter=beta_star)
            p0 = conv.evaluate(nsep=nsep)
            P0.append(p0)
            #interval = conv.confidence_interval(beta_star, how_many=1.)
            #coverages.extend([(interval[0] < beta_star) * (interval[1] > beta_star)])
            #lengths.extend([interval[1] - interval[0]])

    print len(P0), ' instances not passing pre-test out of ', nsim, ' total instances'
    print 'pivots: ', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05)
    #print('confidence intervals', np.mean(coverages), np.mean(lengths))

    return P0


def test_weak_iv_clr_naive(nsim=500, n=1000, p=10, Sigma_12=0.8, gsnr=1., beta_star=1., nsep=100):
    P0 = []
    for i in range(nsim):
        Z, D, Y, beta, gamma = group_lasso_iv.bigaussian_instance(n=n, p=p, beta=beta_star, gsnr=gsnr, 
            Sigma = np.array([[1., Sigma_12], [Sigma_12, 1.]]))
        conv = weak_iv_clr(Y, D, Z)
        if not conv.pre_test():
            conv.setup(parameter=beta_star)
            p0 = conv.naive_evaluate(nsep=nsep)
            P0.append(p0)
    print len(P0), ' instances not passing pre-test out of ', nsim, ' total instances'
    print 'pivots: ', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05)

    return P0


# if true_model is True, Sigma_12 is the true Sigma_{12}
# otherwise Sigma_12 will be the consistent estimator
def test_group_lasso_iv_ar_instance(n=1000, p=10, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1., marginalize=False, randomizer_scale=None):

    Z, D, Y, beta, gamma = group_lasso_iv.bigaussian_instance(beta=beta_star, gsnr=gsnr,Sigma = np.array([[1., Sigma_12], [Sigma_12, 1.]]))
    conv = group_lasso_iv_ar(Y,D,Z, randomizer_scale=randomizer_scale)
    passed = conv.fit()
    if not passed:
        return None, None

    if true_model is True:
        sigma_11 = 1.
        sigma_12 = Sigma_12
    else:
        Sigma_matrix = conv.estimate_covariance()
        sigma_11 = Sigma_matrix[0,0]
        sigma_12 = Sigma_matrix[0,1]

    pivot = None
    coverage = None

    pivot, _, coverage = conv.summary(parameter=beta_star, Sigma_11=sigma_11, Sigma_12=sigma_12)

    return pivot, coverage


def test_pivots_group_lasso_ar(nsim=500, n=1000, p=10, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1., marginalize=False, randomizer_scale=None):
    P0 = []
    #intervals = []
    coverages = []
    #lengths = []
    for i in range(nsim):
        p0, coverage = test_group_lasso_iv_ar_instance(n=n, p=p, true_model=true_model, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star, marginalize=marginalize, randomizer_scale=randomizer_scale)
        if p0 is not None:
            P0.extend(p0)
            #intervals.extend(interval)
            coverages.append(coverage)
            #lengths.extend([interval[0][1] - interval[0][0]])

    print len(P0), ' instances passing pre-test out of ', nsim, ' total instances'
    print 'pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05)
    print 'coverages', np.mean(coverages)
    #print 'confidence intervals', np.mean(coverages), np.mean(lengths)

    return P0

def naive_pre_test_ar_instance(n=1000, p=10, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1.):

    Z, D, Y, beta, gamma = group_lasso_iv.bigaussian_instance(n=n, p=p, beta=beta_star, gsnr=gsnr,Sigma = np.array([[1., Sigma_12], [Sigma_12, 1.]]))
    conv = group_lasso_iv_ar(Y,D,Z)

    if true_model is True:
        sigma_11 = 1.
        sigma_12 = Sigma_12
    else:
        Sigma_matrix = conv.estimate_covariance()
        sigma_11 = Sigma_matrix[0,0]
        sigma_12 = Sigma_matrix[0,1]

    pval = conv.naive_inference(parameter=beta_star, Sigma_11 = sigma_11, Sigma_12 = sigma_12)
    
    return pval

def test_pivots_naive_pre_test_ar(nsim=500, n=1000, p=10, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1.):
    P0 = []
    #intervals = []
    #coverages = []
    #lengths = []
    for i in range(nsim):
        p0 = naive_pre_test_ar_instance(n=n, p=p, true_model=true_model, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star)
        if p0 is not None:
            P0.extend([p0])
            #intervals.extend(interval)

    print len(P0), ' instances passing pre-test out of ', nsim, ' total instances' 
    print 'pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05)
    #print 'confidence intervals', np.mean(coverages), np.mean(lengths)

    return P0


# if true_model is True, Sigma_12 is the true Sigma_{12}
# otherwise Sigma_12 will be the consistent estimator
def test_group_lasso_iv_instance(n=1000, p=10, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1., marginalize=False, randomizer_scale=None):

    Z, D, Y, beta, gamma = group_lasso_iv.bigaussian_instance(beta=beta_star, gsnr=gsnr,Sigma = np.array([[1., Sigma_12], [Sigma_12, 1.]]))
    conv = group_lasso_iv(Y,D,Z, randomizer_scale=randomizer_scale)
    passed = conv.fit()
    if not passed:
        return None, None

    if true_model is True:
        sigma_11 = 1.
        sigma_12 = Sigma_12
    else:
        Sigma_matrix = conv.estimate_covariance()
        sigma_11 = Sigma_matrix[0,0]
        sigma_12 = Sigma_matrix[0,1]

    pivot = None
    interval = None

    pivot, _, interval = conv.summary(parameter=beta_star, Sigma_11=sigma_11, Sigma_12=sigma_12)

    return pivot, interval


def test_pivots_group_lasso(nsim=500, n=1000, p=10, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1., marginalize=False, randomizer_scale=None):
    P0 = []
    #intervals = []
    coverages = []
    lengths = []
    for i in range(nsim):
        p0, interval= test_group_lasso_iv_instance(n=n, p=p, true_model=true_model, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star, marginalize=marginalize, randomizer_scale=randomizer_scale)
        if p0 is not None:
            P0.extend(p0)
            #intervals.extend(interval)
            coverages.extend([(interval[0][0] < beta_star) * (interval[0][1] > beta_star)])
            lengths.extend([interval[0][1] - interval[0][0]])

    print len(P0), ' instances passing pre-test out of ', nsim, ' total instances'
    print 'pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05)
    print 'confidence intervals', np.mean(coverages), np.mean(lengths)

    return P0


def naive_pre_test_instance(n=1000, p=10, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1.):

    Z, D, Y, beta, gamma = group_lasso_iv.bigaussian_instance(n=n, p=p, beta=beta_star, gsnr=gsnr,Sigma = np.array([[1., Sigma_12], [Sigma_12, 1.]]))
    conv = group_lasso_iv(Y,D,Z)

    if true_model is True:
        sigma_11 = 1.
    else:
        Sigma_matrix = conv.estimate_covariance()
        sigma_11 = Sigma_matrix[0,0]

    pval, interval = conv.naive_inference(parameter=beta_star, Sigma_11 = sigma_11, compute_intervals=True)
    
    return pval, interval


def test_pivots_naive_pre_test(nsim=500, n=1000, p=10, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1.):
    P0 = []
    #intervals = []
    coverages = []
    lengths = []
    for i in range(nsim):
        p0, interval= naive_pre_test_instance(n=n, p=p, true_model=true_model, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star)
        if p0 is not None:
            P0.extend([p0])
            #intervals.extend(interval)
            coverages.extend([(interval[0] < beta_star) * (interval[1] > beta_star)])
            lengths.extend([interval[1] - interval[0]])

    print len(P0), ' instances passing pre-test out of ', nsim, ' total instances' 
    print 'pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05)
    print 'confidence intervals', np.mean(coverages), np.mean(lengths)

    return P0


def plain_tsls_instance(n=1000, p=10, s=3, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1.):

    Z, D, Y, beta, gamma = group_lasso_iv.bigaussian_instance(beta=beta_star, gsnr=gsnr,Sigma = np.array([[1., Sigma_12], [Sigma_12, 1.]]))
    conv = group_lasso_iv(Y,D,Z)

    if true_model is True:
        sigma_11 = 1.
    else:
        Sigma_matrix = conv.estimate_covariance()
        sigma_11 = Sigma_matrix[0,0]

    pval, interval = conv.plain_inference(parameter=beta_star, Sigma_11 = sigma_11, compute_intervals=True)
    
    return pval, interval


def test_pivots_plain_tsls(nsim=500, n=1000, p=10, s=3, true_model=True, Sigma_12=0.8, gsnr=1., beta_star=1.):
    P0 = []
    #intervals = []
    coverages = []
    lengths = []
    for i in range(nsim):
        p0, interval= plain_tsls_instance(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star)
        if p0 is not None:
            P0.extend([p0])
            #intervals.extend(interval)
            coverages.extend([(interval[0] < beta_star) * (interval[1] > beta_star)])
            lengths.extend([interval[1] - interval[0]])

    print('pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))
    print('confidence intervals', np.mean(coverages), np.mean(lengths))

    return P0


############################################### Invalid IV #########################################################




# if true_model is True, Sigma_12 is the true Sigma_{12}
# otherwise Sigma_12 will be the consistent estimator
def test_lasso_iv_instance(n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., marginalize=False):

    Z, D, Y, alpha, beta_star, gamma = lasso_iv.bigaussian_instance(n=n,p=p,s=s, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    conv = lasso_iv(Y, D, Z)
    if marginalize:
        conv.fit_for_marginalize()
    else:
        conv.fit()

    if true_model is True:
        sigma_11 = 1.
    else:
        sigma_11 = conv.estimate_covariance()

    pivot = None
    interval = None
    power = None
    if set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0]) and conv._inactive.sum()>0:
        pivot, _, interval, power = conv.summary(parameter=beta_star, Sigma_11=sigma_11, ndraw=ndraw, burnin=burnin, compute_power=True)
    return pivot, interval, power

def test_pivots(nsim=500, n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., marginalize=False):
    P0 = []
    coverages = []
    lengths = []
    powers = []
    for i in range(nsim):
        p0, interval, power = test_lasso_iv_instance(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta_star=beta_star, marginalize=marginalize)
        if p0 is not None and interval is not None:
            P0.extend(p0)
            coverages.extend([(interval[0][0] < beta_star) * (interval[0][1] > beta_star)])
            lengths.extend([interval[0][1] - interval[0][0]])
            powers.append(power)

    print('pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))
    print('confidence intervals', np.mean(coverages), np.mean(lengths), np.std(lengths))
    print('powers', np.mean(np.array(powers), axis=0))

    return P0

def test_lasso_iv_instance_without_correct_selection(n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., snr=7., marginalize=False, penalty=None):

    Z, D, Y, alpha, beta_star, gamma = lasso_iv.bigaussian_instance(n=n,p=p,s=s, snr=snr, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    conv = lasso_iv(Y, D, Z, penalty=penalty)
    if marginalize:
        conv.fit_for_marginalize()
    else:
        conv.fit()

    if true_model is True:
        sigma_11 = 1.
    else:
        sigma_11 = conv.estimate_covariance()

    pivot = None
    interval = None
    if conv._overall[:-1].sum()>0 and (not set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0])) and conv._inactive.sum()>0:
        pivot, _, interval = conv.summary(parameter=beta_star, Sigma_11=sigma_11)
    return pivot, interval

def test_pivots_without_correct_selection(nsim=500, n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., snr=7., marginalize=False, penalty=None):
    P0 = []
    coverages = []
    lengths = []
    for i in range(nsim):
        p0, interval = test_lasso_iv_instance_without_correct_selection(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta_star=beta_star, snr=snr, marginalize=marginalize, penalty=penalty)
        if p0 is not None and interval is not None:
            P0.extend(p0)
            coverages.extend([(interval[0][0] < beta_star) * (interval[0][1] > beta_star)])
            lengths.extend([interval[0][1] - interval[0][0]])

    print len(P0), ' instances of ', nsim, ' total instances' 
    print('pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))
    print('confidence intervals', np.mean(coverages), np.mean(lengths), np.std(lengths))

    return P0

def test_lasso_iv_instance_naive_inference_without_correct_selection(n=1000, p=10, s=3, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., snr=7., penalty=1.01):

    Z, D, Y, alpha, beta_star, gamma = lasso_iv.bigaussian_instance(n=n,p=p,s=s, snr=snr, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    conv = lasso_iv(Y, D, Z, penalty=penalty*np.sqrt(n * np.log(n)), ridge_term=0., randomizer_scale=0.)
    conv.fit_for_marginalize()

    if true_model is True:
        sigma_11 = 1.
    else:
        sigma_11 = conv.estimate_covariance()

    pivot = None
    interval = None
    if conv._overall[:-1].sum()>0 and (not set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0])) and conv._inactive.sum()>0:
        pivot, interval = conv.naive_inference(parameter=beta_star, Sigma_11=sigma_11, compute_intervals=True)
    return pivot, interval

def test_pivots_naive_inference_without_correct_selection(nsim=500, n=1000, p=10, s=3, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., snr=7., penalty=1.01):
    P0 = []
    coverages = []
    lengths = []
    for i in range(nsim):
        p0, interval= test_lasso_iv_instance_naive_inference_without_correct_selection(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta_star=beta_star, snr=snr, penalty=penalty)
        if p0 is not None and interval is not None:
            P0.extend([p0])
            coverages.extend([(interval[0] < beta_star) * (interval[1] > beta_star)])
            lengths.extend([interval[1] - interval[0]])

    print len(P0), ' instances of ', nsim, ' total instances' 
    print 'pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05)
    print 'confidence intervals', np.mean(coverages), np.mean(lengths)

    return P0

# if true_model is True, Sigma_12 is the true Sigma_{12}
# otherwise Sigma_12 will be the consistent estimator
def test_lasso_iv_ar_instance(n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1.):

    Z, D, Y, alpha, beta_star, gamma = lasso_iv_ar.bigaussian_instance(n=n,p=p,s=s, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    conv = lasso_iv_ar(Y, D, Z)
    conv.fit_for_marginalize()

    if true_model is True:
        sigma_11 = 1.
    else:
        sigma_11 = conv.estimate_covariance()

    pivot = None
    coverage = None
    power = None
    if set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0]) and conv._inactive.sum()>0:
        pivot, _, coverage, power = conv.summary(parameter=beta_star, Sigma_11=sigma_11, compute_power=True)
    return pivot, coverage, power

def test_pivots_ar(nsim=500, n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1.):
    P0 = []
    #intervals = []
    coverages = []
    #lengths = []
    powers = []
    for i in range(nsim):
        p0, coverage, power = test_lasso_iv_ar_instance(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta_star=beta_star)
        #if p0 is not None and interval is not None:
        if p0 is not None:
            P0.extend(p0)
            coverages.extend([coverage])
            powers.append(power)
            #intervals.extend(interval)
            #coverages.extend([(interval[0][0] < beta_star) * (interval[0][1] > beta_star)])
            #lengths.extend([interval[0][1] - interval[0][0]])

    print('pivots', np.mean(P0), np.std(P0), 1.-np.mean(np.array(P0) < 0.05))
    print('coverage', np.mean(coverages))
    print('powers', np.mean(np.array(powers), axis=0))
    #print('confidence intervals', np.mean(coverages), np.mean(lengths))

    return P0

def test_lasso_iv_ar_instance_without_correct_selection(n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., snr=7., marginalize=False, penalty=None):

    Z, D, Y, alpha, beta_star, gamma = lasso_iv_ar.bigaussian_instance(n=n,p=p,s=s, snr=snr, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    conv = lasso_iv_ar(Y, D, Z, penalty=penalty)
    if marginalize:
        conv.fit_for_marginalize()
    else:
        conv.fit()

    if true_model is True:
        sigma_11 = 1.
    else:
        sigma_11 = conv.estimate_covariance()

    pivot = None
    coverage = None
    if conv._overall[:-1].sum()>0 and (not set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0])) and conv._inactive.sum()>0:
        pivot, _, coverage = conv.summary(parameter=beta_star, Sigma_11=sigma_11)
    return pivot, coverage

def test_pivots_ar_without_correct_selection(nsim=500, n=1000, p=10, s=3, ndraw=10000, burnin=2000, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., snr=7., marginalize=False, penalty=None):
    P0 = []
    coverages = []
    for i in range(nsim):
        p0, coverage = test_lasso_iv_ar_instance_without_correct_selection(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta_star=beta_star, snr=snr, marginalize=marginalize, penalty=penalty)
        if p0 is not None and coverage is not None:
            P0.extend(p0)
            coverages.extend([coverage])

    print len(P0), ' instances of ', nsim, ' total instances' 
    print('pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))
    print('mean coverage rate', np.mean(coverages))

    return P0

def test_lasso_iv_ar_instance_naive_inference_without_correct_selection(n=1000, p=10, s=3, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., snr=7., penalty=1.01):

    Z, D, Y, alpha, beta_star, gamma = lasso_iv_ar.bigaussian_instance(n=n,p=p,s=s, snr=snr, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta=beta_star,Sigma=np.array([[1., Sigma_12],[Sigma_12, 1.]]))

    conv = lasso_iv_ar(Y, D, Z, penalty=penalty*np.sqrt(n * np.log(n)), ridge_term=0., randomizer_scale=0.)
    conv.fit_for_marginalize()

    if true_model is True:
        sigma_11 = 1.
    else:
        sigma_11 = conv.estimate_covariance()

    pivot = None
    #interval = None
    if conv._overall[:-1].sum()>0 and (not set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0])) and conv._inactive.sum()>0:
        pivot = conv.naive_inference(parameter=beta_star, Sigma_11=sigma_11)
    return pivot

def test_pivots_ar_naive_inference_without_correct_selection(nsim=500, n=1000, p=10, s=3, true_model=True, Sigma_12=0.8, gsnr_invalid=1., gsnr_valid=1., beta_star=1., snr=7., penalty=1.01):
    P0 = []
    coverages = []
    for i in range(nsim):
        p0 = test_lasso_iv_ar_instance_naive_inference_without_correct_selection(n=n, p=p, s=s, true_model=true_model, Sigma_12=Sigma_12, gsnr_invalid=gsnr_invalid, gsnr_valid=gsnr_valid, beta_star=beta_star, snr=snr, penalty=penalty)
        if p0 is not None:
            P0.extend([p0])
            coverages.extend([(p0 < 0.05)])

    print len(P0), ' instances of ', nsim, ' total instances' 
    print 'pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05)
    print 'confidence intervals', np.mean(coverages)

    return P0

# ad hoc simulation function for generating data close to observed MR dataset
# added for AoAS revision
# s is number of invalid alpha as sensitivity parameter
# ZTZ, Sigma are estimated based on formulae in the manuscript, up to the constant number 
# betaD, betaY are observed MR data, gamma_se hand tuned as roughly np.std(betaD)
def lasso_iv_summary_bigaussian_instance(n, p, s, beta, betaD, betaY, ZTZ, Sigma, gamma_se, alpha_se):

    gamma = np.random.multivariate_normal(betaD, np.diag(np.ones(p)*(gamma_se**2)))
    alpha_raw = np.random.multivariate_normal(betaY, np.diag(np.ones(p)*(alpha_se**2)))
    thres = np.sort(np.abs(alpha_raw))[-s]
    alpha = (np.abs(alpha_raw)>=thres)*alpha_raw

    Z = np.random.multivariate_normal(mean=np.zeros(p), cov=ZTZ/(n-1.), size=n)
    Z -= Z.mean(0)[None,:]
    mean = [0, 0]
    errorTerm = np.random.multivariate_normal(mean,Sigma,n)

    D = Z.dot(gamma) + errorTerm[:,1]
    Y = Z.dot(alpha) + D * beta + errorTerm[:,0]

    D -= D.mean()
    Y -= Y.mean()

    return Z, D, Y, alpha, beta, gamma

def test_lasso_iv_summary_instance(n, p, s, beta_star, betaD, betaY, ZTZ, Sigma, gamma_se, alpha_se, penalty, 
                                        ndraw=2000, burnin=2000, true_model=True):

    Z, D, Y, alpha, beta_star, gamma = lasso_iv_summary_bigaussian_instance(n, p, s, beta_star, betaD, betaY, ZTZ, Sigma, gamma_se, alpha_se)

    D -= D.mean()
    Y -= Y.mean()

    ghat = np.array(Z.T.dot(D) / np.diag(Z.T.dot(Z)))
    gse = np.array([np.sqrt(np.sum((D[i] - Z[:,i]*ghat[i])**2)/(n-1.)) for i in range(p)])

    Ghat = np.array(Z.T.dot(Y) / np.diag(Z.T.dot(Z)))
    Gse = np.array([np.sqrt(np.sum((Y[i] - Z[:,i]*Ghat[i])**2)/(n-1.)) for i in range(p)])

    conv = lasso_iv_summary(Ghat=Ghat, ghat=ghat, Gse=Gse, gse=gse, n=n, penalty=penalty)
    conv.fit_for_marginalize()

    if true_model is True:
        sigma_11 = Sigma[0,0]
    else:
        sigma_11 = conv.estimate_covariance()

    pivot = None
    interval = None
    # print(np.nonzero(alpha)[0])
    # print(np.nonzero(conv._overall)[0])
    if set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0]) and conv._inactive.sum()>0:
        pivot, _, interval = conv.summary(parameter=beta_star, Sigma_11=sigma_11, ndraw=ndraw, burnin=burnin)
    return pivot, interval

def test_pivots_summary(n, p, s, beta_star, betaD, betaY, ZTZ, Sigma, gamma_se, alpha_se, penalty, 
                            nsim=500, ndraw=2000, burnin=2000, true_model=True):
    P0 = []
    coverages = []
    lengths = []
    for i in range(nsim):
        p0, interval = test_lasso_iv_summary_instance(n, p, s, beta_star, betaD, betaY, ZTZ, Sigma, gamma_se, alpha_se, penalty,
                                        ndraw, burnin, true_model)
        if p0 is not None and interval is not None:
            P0.extend(p0)
            coverages.extend([(interval[0][0] < beta_star) * (interval[0][1] > beta_star)])
            lengths.extend([interval[0][1] - interval[0][0]])

    print len(P0), ' instances of ', nsim, ' total instances'
    print('pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))
    print('confidence intervals', np.mean(coverages), np.mean(lengths), np.std(lengths))

    return P0

def test_lasso_iv_summary_instance_naive_inference(n, p, s, beta_star, betaD, betaY, ZTZ, Sigma, gamma_se, alpha_se, penalty, true_model=True):

    Z, D, Y, alpha, beta_star, gamma = lasso_iv_summary_bigaussian_instance(n, p, s, beta_star, betaD, betaY, ZTZ, Sigma, gamma_se, alpha_se)

    D -= D.mean()
    Y -= Y.mean()

    ghat = np.array(Z.T.dot(D) / np.diag(Z.T.dot(Z)))
    gse = np.array([np.sqrt(np.sum((D[i] - Z[:,i]*ghat[i])**2)/(n-1.)) for i in range(p)])

    Ghat = np.array(Z.T.dot(Y) / np.diag(Z.T.dot(Z)))
    Gse = np.array([np.sqrt(np.sum((Y[i] - Z[:,i]*Ghat[i])**2)/(n-1.)) for i in range(p)])

    conv = lasso_iv_summary(Ghat=Ghat, ghat=ghat, Gse=Gse, gse=gse, n=n, penalty=penalty)
    conv.fit_for_marginalize()

    if true_model is True:
        sigma_11 = Sigma[0,0]
    else:
        sigma_11 = conv.estimate_covariance()

    pivot = None
    interval = None
    # print(np.nonzero(alpha)[0])
    # print(np.nonzero(conv._overall)[0])
    if set(np.nonzero(alpha)[0]).issubset(np.nonzero(conv._overall)[0]) and conv._inactive.sum()>0:
        pivot, interval = conv.naive_inference(parameter=beta_star, Sigma_11=sigma_11)
    return pivot, interval

def test_pivots_summary_naive_inference(n, p, s, beta_star, betaD, betaY, ZTZ, Sigma, gamma_se, alpha_se, penalty, nsim=500, true_model=True):
    P0 = []
    coverages = []
    lengths = []
    for i in range(nsim):
        p0, interval = test_lasso_iv_summary_instance_naive_inference(n, p, s, beta_star, betaD, betaY, ZTZ, Sigma, gamma_se, alpha_se, penalty, true_model)
        if p0 is not None and interval is not None:
            P0.extend([p0])
            coverages.extend([(interval[0] < beta_star) * (interval[1] > beta_star)])
            lengths.extend([interval[1] - interval[0]])

    print len(P0), ' instances of ', nsim, ' total instances'
    print('pivots', np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))
    print('confidence intervals', np.mean(coverages), np.mean(lengths), np.std(lengths))

    return P0



def main(nsim=500):

    P0 = []
    from statsmodels.distributions import ECDF

    n, p, s = 1000, 10, 3
    Sigma_12 = 0.8
    gsnr = 1.
    beta_star = 1.

    for i in range(nsim):
        try:
            p0 = test_lasso_iv_instance(n=n, p=p, s=s, Sigma_12=Sigma_12, gsnr=gsnr, beta_star=beta_star)
        except:
            p0 = []
        P0.extend(p0)

    print(np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.05))

    U = np.linspace(0, 1, 101)
    #plt.clf()
    plt.plot(U, ECDF(P0)(U))
    plt.plot(U, U, 'r--')
    #plt.savefig("plot.pdf")
    plt.show()


if __name__ == "__main__":
    main()
