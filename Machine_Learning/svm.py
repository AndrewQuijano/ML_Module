from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import dump
import time
import numpy as np
from .misc import plot_grid_search


def get_svm(train_x, train_y, n_fold=10, slow=False):
    start_time = time.time()
    best_svm = tune_svm(train_x, train_y, n_fold, slow)
    print("--- Best Parameter SVM parameters time to complete: %s seconds ---" % (time.time() - start_time))
    print("Best SVM Parameters: " + str(best_svm.best_params_))
    print("[SVM] Training Mean Test Score: " + str(best_svm.score(train_x, train_y)))
    with open("results.txt", "a+") as my_file:
        my_file.write("[SVM_Radial] Best Parameters: " + str(best_svm.best_params_) + '\n')
        my_file.write("[SVM Radial] Training Mean Test Score: " + str(best_svm.score(train_x, train_y)) + '\n')
    return best_svm


def tune_svm(train_x, train_y, n_folds=10, slow=False):
    c = np.arange(0.1, 1, 0.1)
    gammas = np.arange(0.1, 1, 0.1)
    random_grid = {
        'C': c,
        'gamma': gammas,
        'kernel': ["rbf", "linear", "poly", "sigmoid"]
    }
    model = svm.SVC(probability=True)
    if slow:
        svm_search = GridSearchCV(model, param_grid=random_grid, cv=n_folds,
                                  n_jobs=-1, error_score='raise', verbose=2)
    else:
        svm_search = RandomizedSearchCV(model, param_distributions=random_grid,
                                        cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    svm_search.fit(train_x, train_y)
    plot_grid_search(svm_search, 'C', 'SVM_RBF')
    plot_grid_search(svm_search, 'gamma', 'SVM_RBF')
    dump(svm_search, "./Classifiers/" + type(model).__name__ + ".joblib")
    return svm_search
