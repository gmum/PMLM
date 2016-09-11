import sklearn
import utils

from functools import partial
from scipy.optimize import minimize
from autograd import numpy as np
from autograd import grad
from densities import NormalEstimator1D, KDEEstimator1D
from losses import point_entropy_loss, point_scalar_loss, point_l2_loss
from losses import D_CS, D_ED, D_KL, densities_loss


class PMLM(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """ Probabilistic Multithreshold Linear Model 

    In general, looking for a linear projection v which minimizes following loss:

       L(v) = (1 - alpha) SUM_i l_p(y_i, f(x_i, [[ <v, X> ]])) 
                        + alpha l_d([[ <v, X_1> ]], ..., [[ <v, X_K> ]])

    where l_p is "pointwise_loss", l_d is "density_loss" and [[ . ]] is a particular "density_estimator".
    """

    def __init__(self, 
                 density_estimator='normal',
                 pointwise_loss='mle',
                 density_loss=None,
                 density_averaging='geometric',
                 balanced=False,
                 alpha=0.,
                 gamma=1.0,
                 initializations=30,
                 iterations=None,
                 optimizer='L-BFGS-B',
                 random_state=None):
        """
        Keyword arguments:
            density_estimator - estimator for P(<v, x>|y)
                                supported values: "normal", "kde".
                                default: "normal"
            pointwise_loss - loss applied in point-wise part of the total loss
                             supported values: "mle", "me", "l2", None
                             default: "mle"
            density_loss - loss applied to density based part of the total loss
                           supported values: "dcs", "ded", "dkl", None
                              default: None
            density_averaging - type of average used to generalize divergences
                                supported values: "arithmetic", "geometric", "harmonic"
                                default: "geometric"
            balanced - whether to maximize balanced accuracy
                       default: False
            alpha - mixing coefficient of pointwise_loss and density_loss
                    default: 0.
            gamma - density estimator regularization, which rescales variance estimation by gamma^2.
                    default: 1.0
            initializations - how many times to run optimization from random vector.
                              default: 30
            iterations - how many iterations of optimizer to run (None means no limit).
                         default: None
            optimizer - optimizer to use.
                        supported values: all optimizeres supported by scipy.optimize.minimize
                        default: L-BFGS-B
            random_state - random state seed which will be set before each .fit call
                           default: None

        """

        self.density_estimator = density_estimator
        self.density_averaging = density_averaging
        self.pointwise_loss = pointwise_loss
        self.density_loss = density_loss
        self.alpha = alpha
        self.gamma = gamma
        self.initializations = initializations
        self.iterations = iterations
        self.balanced = balanced
        self.optimizer = optimizer
        self.random_state = random_state

        self._str_fields = ['alpha', 'density_estimator', 'density_loss', 'pointwise_loss', 'gamma',
                            'initializations', 'iterations', 'balanced', 'optimizer', 'density_averaging',
                            'random_state']

    def fit(self, X, y):

        self._build_objects()

        np.random.seed(self.random_state)

        X = utils.check_matrix(X)

        self.inv_labels = np.array(list(sorted(set(y))))
        self.labels = {label : index for index, label in enumerate(self.inv_labels)}

        # empirical priors
        self.priors = [(y == l).sum() / float(X.shape[0]) for l in self.inv_labels]

        # estimators initialization
        self.estimators = {}
        for label in self.labels:
            self.estimators[label] = self._density_estimator(**self._density_kwargs)
            self.estimators[label].init(X[y == label])

        self._X = X
        tmp = [self.labels[label] for label in y]
        self._raw_y = y

        self._y = np.zeros((X.shape[0], len(self.labels)))
        for i, e in enumerate(tmp):        
            self._y[i, e] = 1

        results = {}
        for i in xrange(self.initializations):
            results[i] = minimize(self._loss, jac=grad(self._loss), method=self.optimizer,
                                  x0=utils.sample_from_unit_sphere(X.shape[1]),
                                  options={'maxiter': self.iterations})

        best = None
        for i in results:
            if best is None or results[i]['fun'] < best:
                best = results[i]['fun']
                self.v = results[i]['x']

        for label in self.labels:
            self.estimators[label].fit(self.v, X[y == label])

        self._y = None

    def predict(self, X):
        X = utils.check_matrix(X)
        return np.array([self.inv_labels[label] for label in np.argmax(self._forward(self.v, X), axis=1)])

    def predict_proba(self, X):
        X = utils.check_matrix(X)
        probabilities = self._forward(self.v, X)
        if probabilities.shape[1] == 2: # for binary classification sklearn expects probability of positive class only
            return probabilities[:, 0]
        else:
            return probabilities

    def decision_function(self, X):
        return self.predict_proba(X)

    def get_params(self, deep=False):
        return {field: getattr(self, field) for field in self._str_fields}

    def _build_objects(self):
    
        if self.density_estimator in ['normal', 'n', 'N']:
            self._density_estimator = NormalEstimator1D
            self._density_kwargs = {'gamma' : self.gamma}
        elif self.density_estimator in ['kde', 'KDE']:
            self._density_estimator = KDEEstimator1D
            self._density_kwargs = {'gamma' : self.gamma}
        else:
            raise Exception('{} is not a valid density estimator'.format(self.density_estimator))

        if self.pointwise_loss in ['entropy', 'mle', 'MLE']:
            self._pointwise_loss = point_entropy_loss
        elif self.pointwise_loss in ['scalar', 'me']:
            self._pointwise_loss = point_scalar_loss
        elif self.pointwise_loss in ['ed', 'l2', 'L2']:
            self._pointwise_loss = point_l2_loss
        elif self.pointwise_loss is None:
            if alpha == 1.0:
                self._pointwise_loss = None
            else:
                raise Exception('You cannot specify alpha={} and no pointwise loss'.format(self.alpha))
        else:
            raise Exception('{} is not a valid pointwise loss'.format(self.pointwise_loss))


        if self.density_loss in ['melc', 'dcs', 'DCS', 'D_CS']:
            self._density_loss = partial(densities_loss, divergence=D_CS, method=self.density_averaging)
        elif self.density_loss in ['ed', 'ded', 'DED', 'D_ED']:
            self._density_loss = partial(densities_loss, divergence=D_ED, method=self.density_averaging)
        elif self.density_loss in ['kl', 'dkl', 'DKL', 'D_KL']:
            if not self._density_estimator is NormalEstimator1D:
                raise Exception('KL divergence cannot be used with other density estimators than "normal"')
            self._density_loss = partial(densities_loss, divergence=D_KL, method=self.density_averaging)
        elif self.density_loss is None:
            if self.alpha == 0.0:
                self._density_loss = None
            else:
                raise Exception('You cannot specify alpha={} and no density loss'.format(self.alpha))
        else:
            raise Exception('{} is not a valid density loss'.format(self.density_loss))        

    def _loss(self, v):
        
        v /= utils.norm(v)

        if self.alpha == 1.0:
            return self._density_loss(self._forward_densities(v))
        elif self.alpha == 0.0:
            return self._pointwise_loss(self._forward(v, self._X), self._y)
        else:
            return (1-self.alpha) * self._pointwise_loss(self._forward(v, self._X), self._y) + \
                   self.alpha * self._density_loss(self._forward_densities(v)) 

    def _forward(self, v, X):
        """
        Computes P(y|v^TX)
        """
        answers = []
        prior_array = []
        for l in xrange(len(self.inv_labels)):
            est = self.estimators[self.inv_labels[l]]
            est.fit(v, self._X[self._raw_y == self.inv_labels[l]])
            col = est.predict(v, X)
            answers.append(col)
            prior_array.append(self._priors(l))
        answers = np.array(answers).T
        answers /= answers.sum(axis=1).reshape(X.shape[0], -1)
        prior_array = np.array(prior_array).reshape(1, -1)
        
        return np.multiply(answers, prior_array)

    def _forward_densities(self, v):
        gmms = []
        for l in xrange(len(self.inv_labels)):
            est = self.estimators[self.inv_labels[l]]
            est.fit(v, self._X[self._raw_y == self.inv_labels[l]])
            gmms.append(est.toGMM())
        return gmms

    def _priors(self, l):
        if self.balanced: # balancing means P(y) = 1/K
            return 1. / len(self.labels)
        else:
            return self.priors[l]

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__,
                               ', '.join(['{}={}'.format(field, getattr(self, field))
                                         for field in self._str_fields]))

    def __repr__(self):
        return str(self)