from autograd import numpy as np
from sklearn.covariance import LedoitWolf
import warnings

class DensityEstimator:

    def init(self, X):
        pass

    def fit(self, v, X):
        """
        Fits density estimator to <v, X>
        """
        raise Exception('DensityEstimator is an abstract class')
    
    def predict(self, v, X):
        """
        Returns estimates on <v, X>
        """
        raise Exception('DensityEstimator is an abstract class')

    def fit_predict(self, v, Xfit, X):
        """
        Fits density estimator to <v, Xfit> and returns estimates on <v, X>
        """
        self.fit(v, Xfit)
        return self.predict(v, X)

    def toGMM(self):
        raise Exception('DensityEstimator is an abstract class')


class KDEEstimator1D(DensityEstimator):
    """
    This will be simple 1D KDE estimator based on Silverman's rule of thumb. 
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def fit(self, v, X):
        self.means = np.dot(X, v)
        mean = np.mean(self.means)
        self.var = ((self.means - mean) ** 2).sum()
        self.N = X.shape[0]
        self.h = np.sqrt(self.var) * 1.06 * self.N ** (-0.2) * self.gamma

    def K(self, u):
        return 1. / np.sqrt(2. * np.pi) * np.exp(-0.2 * u ** 2)

    def predict(self, v, X):
        X = np.dot(X, v)
        pred = 1. / (self.N * self.h) * self.K((X.reshape(-1, 1) - self.means.reshape(1, -1)) / self.h).sum(axis=1)
        return pred

    def toGMM(self):
        weights = np.array([1. / self.N] * self.N)
        variances = np.array([self.h] * self.N)
        gmm = np.vstack((weights, self.means, variances)).T # weight, mean, var
        return gmm

class NormalEstimator1D(DensityEstimator):
    """
    Implements regularized maximum likelihood estimator of 1D normal density.
    For efficiency reasons, it uses LedoitWolf estimator in the input space, in order to compute
    estimators in the projected space (thus fitting estimator in the projected space is independent
    on the size of training set)
    """

    def __init__(self, gamma=1.0, cov_estimator=LedoitWolf):
        self.gamma = gamma
        self.cov_estimator = cov_estimator

    def init(self, X):
        self.mean = X.mean(axis=0)
        self.cov = self.cov_estimator(store_precision = False).fit(X).covariance_ * self.gamma ** 2 

    def fit(self, v, X):
        self.mu = np.dot(v, self.mean)
        self.var = np.dot(np.dot(v, self.cov ), v.T)

    def predict(self, v, X):
        return 1. / np.sqrt(2 * np.pi * self.var) * np.exp(-(np.dot(X, v) - self.mu)**2 / (2*self.var))

    def toGMM(self):
        return np.array([[1.0, self.mu, self.var]]) # weight, mean, var
