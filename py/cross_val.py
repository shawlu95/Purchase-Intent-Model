import analysis
import random
from sklearn.base import clone
from random import seed
from random import randrange
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np

def cross_validation_split(dataset, labl, folds=5):
    dataset_split = []
    dataset_copy = dataset
    fold_size = int(len(dataset) / folds)
    
    imp = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
    
    for i in range(folds):
        fold = list()
        
        tst_data = dataset_copy[i * fold_size : (i + 1) * fold_size]
        tst_labl = labl[i * fold_size : (i + 1) * fold_size]

        trn_data_1 = dataset_copy[0 : i * fold_size ]
        trn_labl_1 = labl[0 : i * fold_size ]
        
        trn_data_2 = dataset_copy[(i + 1) * fold_size : ]
        trn_labl_2 = labl[(i + 1) * fold_size : ]

        trn_data = pd.concat([trn_data_1, trn_data_2])
        trn_labl = pd.concat([trn_labl_1, trn_labl_2])
        
        imp.fit(trn_data)
        trn_data = imp.transform(trn_data)
        tst_data = imp.transform(tst_data)
        
        dataset_split.append([trn_data, trn_labl, tst_data, tst_labl])
    return dataset_split

def folds_decompose(folds):
    i = 1
    for fold in folds:
        analysis.log("Fold %i" % (i))
        analysis.log(fold[1].value_counts())
        i = i + 1

def cross_eval(model, dataset, labl, folds=5, random_state=0,  verbose = False):
    # test cross validation split
    random.seed(random_state)

    folds = cross_validation_split(dataset, labl, folds)
    i = 1
    evals = {}
    for fold in folds:
        analysis.log("Fold %i" % (i))

        clone_clf = clone(model)
        trn_data_fold, trn_labl_fold, tst_data_fold, tst_labl_fold = fold[0], fold[1], fold[2], fold[3]
        clone_clf.fit(trn_data_fold, trn_labl_fold)

        y_probas = clone_clf.predict_proba(tst_data_fold) [:, 1]
        score, stat = analysis.evaluate_model(tst_labl_fold, y_probas, verbose = verbose)
        tprs = stat[0]
        evals[str(i)] = [score, tprs]

        analysis.log("Score = %.10f\n"%(score))

        i = i + 1
        
    tot = np.zeros((1, 3))
    for key in evals:
        tot = tot + np.array(evals[key][1])
    tot = tot / len(evals)
    score = tot[0][0] * 0.4 + (tot[0][1] + tot[0][2]) * 0.3
    analysis.log("K-fold score: %.10f"%(score))
    return evals, folds