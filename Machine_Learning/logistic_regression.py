from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from joblib import dump
import numpy as np
from .misc import plot_grid_search


def get_logistic(train_x, train_y, n_fold=10, slow=False):
    start = time.time()
    n = np.logspace(-3, 3)
    param_grid = {'C': n}
    clf = LogisticRegression(warm_start=False, max_iter=6000, multi_class='ovr', solver='lbfgs')
    if slow:
        log_model = GridSearchCV(clf, param_grid, n_jobs=-1, cv=n_fold, verbose=2)
    else:
        log_model = RandomizedSearchCV(clf, param_grid, n_jobs=-1, cv=n_fold, verbose=2)
    log_model.fit(train_x, train_y)

    plot_grid_search(log_model, 'C', 'Logistic_Regression')

    print("[INFO] Logistic Regression-Best Parameters: " + str(log_model.best_params_))
    print("[INFO] randomized search took {:.2f} seconds".format(time.time() - start))
    print("[Logistic] Training Score is: " + str(log_model.score(train_x, train_y)))
    with open("results.txt", "a+") as my_file:
        my_file.write("[Logistic Regression] Best Parameters: " + str(log_model.best_params_) + '\n')
        my_file.write("[Logistic Regression] Training Mean Test Score: " +
                      str(log_model.score(train_x, train_y)) + '\n')
    dump(log_model, "./Classifiers/" + type(clf).__name__ + ".joblib")
    return log_model
