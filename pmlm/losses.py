from autograd import numpy as np
import utils

EPSILON_CONSTANT = 1e-20

# Point-wise losses

def point_scalar_loss(preds, truth):
    """
    -<preds, truth>
    """
    return -np.multiply(preds, truth).sum(axis=1).mean()

def point_l2_loss(preds, truth):
    """
    ||preds - truth||^2

    For binay truth this is Renyi's quadratic entropy on preds

    <p - t, p - t> = <p,p> - 2<p,t> + <t,t> 
    """
    return ((preds - truth) ** 2).sum(axis=1).mean()

def point_entropy_loss(preds, truth):
    """
    -<truth, log(preds)>

    For binary truth this is the same as entropy_loss
    """
    return -np.log(np.maximum(EPSILON_CONSTANT, np.multiply(preds, truth).sum(axis=1))).mean()



# Density based losses

def cip(gmm1, gmm2):
    """
    Cross information potential between two GMMs, defined as

    cip(GMM1, GMM2) = INT GMM1 GMM2 [x] dx
    """
    distances = gmm1[:, 1].reshape(-1, 1) - gmm2[:, 1].reshape(1, -1)
    variances = np.maximum(EPSILON_CONSTANT, gmm1[:, 2].reshape(-1, 1) + gmm2[:, 2].reshape(1, -1))
    coefs = gmm1[:, 0].reshape(-1, 1) * gmm2[:, 0].reshape(1, -1)
    
    integrals = np.multiply(1. / np.sqrt(2 * np.pi * variances),
                            np.multiply(coefs, np.exp( -(distances ** 2) / (2 * variances))))

    return np.sum(integrals) / (len(gmm1) * len(gmm2))



def D_ED(gmm1, gmm2):
    """
    L2 distance between two GMMs

    l2(GMM1, GMM2) = INT (GMM1 - GMM2)^2[x] dx
    """
    return cip(gmm1, gmm1) + cip(gmm2, gmm2) - 2 * cip(gmm1, gmm2)    

def D_CS(gmm1, gmm2):
    """
    Cauchy-Schwarz divergence

    CS(GMM1, GMM2) = INT GMM1^2[x] dx + INT GMM2^2[x] dx - 2 INT GMM1 GMM2 [x] dx
    """
    return np.log(cip(gmm1, gmm1)) + np.log(cip(gmm2, gmm2)) - 2 * np.log(cip(gmm1, gmm2)) 

def D_KL(gmm1, gmm2):
    """
    Symmetrized version of Kulback-Leibler divergence
    """
    if len(gmm1) > 1 or len(gmm2) > 1:
        raise Exception('KL divergence can be computed only for single Gaussians!')

    def Dkl(m1, var1, m2, var2):
        return (m1-m2)**2 / (2*var2) + 0.5 * (var1 / var2 - 1 - np.log(var1 / var2))
    
    a, b = gmm1[0], gmm2[0]
    return 0.5 * (Dkl(a[1], a[2], b[1], b[2]) + Dkl(b[1], b[2], a[1], a[2]))

_MEANS = {
    'arithmetic': np.mean,
    'geometric': utils.geometric_mean,
    'harmonic': utils.harmonic_mean
}

def densities_loss(gmm, divergence, method='geometric'):
    """
    Creates loss from a given divergence by returning the negation of its multi-distribution generalization.
    """

    if method in _MEANS:
        method = _MEANS[method]
    else:
        raise Exception('{} is not supported'.format(method))

    values = []
    for i, gmm1 in enumerate(gmm):
        for gmm2 in gmm[i+1:]:
            values.append(divergence(gmm1, gmm2))
    return -method(values)
