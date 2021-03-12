import os
import sys
import numpy as np
import bayesiancoresets as bc

# make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import model_linreg
import time

nm = sys.argv[1]
tr = sys.argv[2]

# experiment params
M = 300
opt_itrs = 100
proj_dim = 100
pihat_noise = 0.75
n_bases_per_scale = 50
N_subsample = 1000

# load data and compute true posterior
# each row of x is [lat, lon, price]
print('Loading data')

# trial num as seed for loading data
np.random.seed(int(tr) + 2020)

x = np.load('../data/prices2018.npy')

print('Taking a random subsample')
# get a random subsample of it
idcs = np.arange(x.shape[0])
np.random.shuffle(idcs)
x = x[idcs[:N_subsample], :]

# log transform
x[:, 2] = np.log10(x[:, 2])

# get empirical mean/std
datastd = x[:, 2].std()
datamn = x[:, 2].mean()

# bases of increasing size; the last one is effectively a constant
basis_unique_scales = np.array([.2, .4, .8, 1.2, 1.6, 2., 100])
basis_unique_counts = np.hstack((n_bases_per_scale * np.ones(6, dtype=np.int64), 1))

# the dimension of the scaling vector for the above bases
d = basis_unique_counts.sum()
print('Basis dimension: ' + str(d))

# model params
mu0 = datamn * np.ones(d)
Sig0 = (datastd ** 2 + datamn ** 2) * np.eye(d)
Sig0inv = np.linalg.inv(Sig0)

# generate basis functions by uniformly randomly picking locations in the dataset
print('Trial ' + tr)
print('Creating bases')
basis_scales = np.array([])
basis_locs = np.zeros((0, 2))
for i in range(basis_unique_scales.shape[0]):
    basis_scales = np.hstack((basis_scales, basis_unique_scales[i] * np.ones(basis_unique_counts[i])))
    idcs = np.random.choice(np.arange(x.shape[0]), replace=False, size=basis_unique_counts[i])
    basis_locs = np.vstack((basis_locs, x[idcs, :2]))

print('Converting bases and observations into X/Y matrices')
# convert basis functions + observed data locations into a big X matrix
X = np.zeros((x.shape[0], basis_scales.shape[0]))
for i in range(basis_scales.shape[0]):
    X[:, i] = np.exp(-((x[:, :2] - basis_locs[i, :]) ** 2).sum(axis=1) / (2 * basis_scales[i] ** 2))
Y = x[:, 2]

# get true posterior
print('Computing true posterior')
mup, Sigp = model_linreg.weighted_post(mu0, Sig0inv, datastd ** 2, X, Y, np.ones(X.shape[0]))
Sigpinv = np.linalg.inv(Sigp)

# create function to output log_likelihood given param samples
print('Creating log-likelihood function')
log_likelihood = lambda samples: model_linreg.potentials(datastd ** 2, X, Y, samples)

# create tangent space for well-tuned Hilbert coreset alg
print('Creating tuned tangent space for Hilbert coreset construction')
sampler_optimal = lambda n, w, ids: np.random.multivariate_normal(mup, Sigp, n)
tsf_optimal = bc.BayesianTangentSpaceFactory(log_likelihood, sampler_optimal, proj_dim)

# create tangent space for poorly-tuned Hilbert coreset alg
print('Creating untuned tangent space for Hilbert coreset construction')
U = np.random.rand()
muhat = U * mup + (1. - U) * mu0
Sighat = U * Sigp + (1. - U) * Sig0
# now corrupt the smoothed pihat
muhat += pihat_noise * np.sqrt((muhat ** 2).sum()) * np.random.randn(muhat.shape[0])
Sighat *= np.exp(-2 * pihat_noise * np.fabs(np.random.randn()))

sampler_realistic = lambda n, w, ids: np.random.multivariate_normal(muhat, Sighat, n)
tsf_realistic = bc.BayesianTangentSpaceFactory(log_likelihood, sampler_realistic, proj_dim)


##############################
###Exact projection in SparseVI for gradient computation
# for this model we can do the tangent space projection exactly
def tsf_exact_w(wts, idcs):
    w = np.zeros(X.shape[0])
    w[idcs] = wts
    muw, Sigw = model_linreg.weighted_post(mu0, Sig0inv, datastd ** 2, X, Y, w)
    lmb, V = np.linalg.eigh(Sigw)
    beta = X.dot(V * np.sqrt(np.maximum(lmb, 0.)))
    nu = Y - X.dot(muw)

    # project the matrix term down to 20*20 = 400 dimensions
    lmb, V = np.linalg.eigh(beta.T.dot(beta))
    n_dim = 20
    beta_proj = beta.dot(V[:, -n_dim:])

    return np.hstack((nu[:, np.newaxis] * beta,
                      1. / np.sqrt(2.) * (beta_proj[:, :, np.newaxis] * beta_proj[:, np.newaxis, :]).reshape(
                          beta.shape[0], n_dim ** 2))) / datastd ** 2


tsf_exact_optimal = lambda: tsf_exact_w(np.ones(x.shape[0]), np.arange(x.shape[0]))
# rlst_idcs = np.arange(x.shape[0])
# np.random.shuffle(rlst_idcs)
# rlst_idcs = rlst_idcs[:int(0.1 * rlst_idcs.shape[0])]
# rlst_w = np.zeros(x.shape[0])
# rlst_w[rlst_idcs] = 2. * x.shape[0] / rlst_idcs.shape[0] * np.random.rand(rlst_idcs.shape[0])
# tsf_exact_realistic = lambda: tsf_exact_w(2. * np.random.rand(x.shape[0]), np.arange(x.shape[0]))

##############################


# create coreset construction objects
print('Creating coreset construction objects')
sparsevi = bc.SparseVICoreset(tsf_exact_w, opt_itrs=opt_itrs)
giga_optimal = bc.HilbertCoreset(tsf_optimal)
giga_optimal_exact = bc.HilbertCoreset(tsf_exact_optimal)
# giga_realistic = bc.HilbertCoreset(tsf_realistic)
# giga_realistic_exact = bc.HilbertCoreset(tsf_exact_realistic)
unif = bc.UniformSamplingCoreset(x.shape[0])
iht = bc.IHTCoreset(tsf_exact_optimal, proj_dim, 'IHT')
iht_ii = bc.IHTCoreset(tsf_exact_optimal, proj_dim, 'IHT-2')

algs = {'SVI': sparsevi,
        'GIGAOE': giga_optimal_exact,
        'IHT': iht,
        'IHT-2': iht_ii,
        'RAND': unif}
alg = algs[nm]

print('Building coreset')
# build coresets
w = np.zeros((M + 1, x.shape[0]))
cputs = np.zeros(M + 1)
t0 = time.perf_counter()
if nm == 'IHT' or nm == 'IHT-2':
    for m in range(2, M + 1, 1):
        print('trial: ' + tr + ' alg: ' + nm + ' ' + str(m) + '/' + str(M))
        t0 = time.perf_counter()
        alg.build(1, m)  # building from scratch
        # record time and weights
        cputs[m] = time.perf_counter() - t0
        wts, idcs = alg.weights()
        w[m, idcs] = wts
else:
    for m in range(2, M + 1):
        print('trial: ' + tr + ' alg: ' + nm + ' ' + str(m) + '/' + str(M))
        alg.build(1, m)
        # record time and weights
        cputs[m] = time.perf_counter() - t0  # starting from the previous result
        # store weights
        wts, idcs = alg.weights()
        w[m, idcs] = wts

muw = np.zeros((M + 1, mu0.shape[0]))
Sigw = np.zeros((M + 1, mu0.shape[0], mu0.shape[0]))
rklw = np.zeros(M + 1)
fklw = np.zeros(M + 1)
for m in range(M + 1):
    print('KL divergence computation for trial: ' + tr + ' alg: ' + nm + ' ' + str(m) + '/' + str(M))
    muw[m, :], Sigw[m, :, :] = model_linreg.weighted_post(mu0, Sig0inv, datastd ** 2, X, Y, w[m, :])
    rklw[m] = model_linreg.weighted_post_KL(mu0, Sig0inv, datastd ** 2, X, Y, w[m, :], reverse=True)
    fklw[m] = model_linreg.weighted_post_KL(mu0, Sig0inv, datastd ** 2, X, Y, w[m, :], reverse=False)

if not os.path.exists('results/'):
    os.mkdir('results')
print('Saving result for trial: ' + tr + ' alg: ' + nm)
np.savez('results/results_' + nm + '_' + tr + '.npz', x=x, mu0=mu0, Sig0=Sig0, mup=mup, Sigp=Sigp, w=w,
         muw=muw, Sigw=Sigw, rklw=rklw, fklw=fklw,
         basis_scales=basis_scales, basis_locs=basis_locs, datastd=datastd, cputs=cputs)
