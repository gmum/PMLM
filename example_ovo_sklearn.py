import sys
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import StratifiedKFold
from sklearn.multiclass import OneVsOneClassifier

from pmlm import MLMM, MEMM
from pmlm.utils import ACC, BAC

if __name__ == "__main__":

    X, y = load_svmlight_file(sys.argv[1])
    folds = StratifiedKFold(y, n_folds=5, random_state=1)

    for balanced in [True, False]:
        for pmlm in [OneVsOneClassifier(MLMM(balanced = balanced, gamma=1.5, random_state=1)), 
                     OneVsOneClassifier(MEMM(balanced = balanced, random_state=1))]:
        
            print pmlm
            for train, test in folds:
                pmlm.fit(X[train], y[train])
                print 'acc', ACC(y[test], pmlm.predict(X[test])) 
                print 'bac', BAC(y[test], pmlm.predict(X[test]))

