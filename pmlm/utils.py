from autograd import numpy as np
import warnings

def sample_from_unit_sphere(d):
    """ Returns a vecot runiformly sampled from d-dimensional unit sphere. """
    tmp = np.random.normal(size=(1, d))
    return tmp / norm(tmp)

def norm(x):
    """
    Helper function for differentiable ||x||
    """
    return np.sqrt((x ** 2).sum())

def geometric_mean(container):
    """ Returns geometrical mean of values in container. """
    result = 1.
    for element in container:
        result *= element
    return result ** (1./len(container))

def harmonic_mean(container, epsilon=1e-20):
    """ Returns harmonic mean of values in container. """
    result = 0.
    for element in container:
        result += 1. / np.maximum(epsilon, element)
    return 1./result

def check_matrix(X):
    """ Checks whether given matrix can be converted to dense, if so - converts. """
    if hasattr(X, 'toarray') and callable(X.toarray):
        warnings.warn('Currently only dense matrices are supported due to autograd limitations, casting to dense')
    return X.toarray()

def BAC(truth, predicted):
    """ Returns balanced accuracy. """
    return np.mean([(predicted[truth == label] == truth[truth == label]).mean() for label in set(truth)])

def ACC(truth, predicted):
    """ Returns accuracy. """
    return (predicted == truth).mean()