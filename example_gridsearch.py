import sys
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search

from pmlm import PMLM
from pmlm.utils import ACC, BAC

from pprint import pprint

if __name__ == "__main__":

    X, y = load_svmlight_file(sys.argv[1])

    parameters = {
        'density_estimator': ('normal', 'kde'),
        'gamma': (0.25, 0.5, 1.0)
    }

    pmlm = PMLM(random_state=1)

    clf = grid_search.GridSearchCV(pmlm, parameters)
    clf.fit(X, y)

    pprint(clf.grid_scores_)
    