from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from joblib import dump
import time
import numpy as np
from .misc import plot_grid_search


# http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py
# http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
def get_brain(train_x, train_y, n_fold=10, slow=False):
    start_time = time.time()
    clf = tune_brain(train_x, train_y, n_fold, slow)
    print("--- Best Parameter NN Generation: %s seconds ---" % (time.time() - start_time))
    print("[NN] Training Mean Test Score: " + str(clf.score(train_x, train_y)))

    with open("results.txt", "a") as my_file:
        my_file.write("[Neural_Network] Best Parameters: " + str(clf.best_params_) + '\n')
        my_file.write("[Neural_Network] Training Mean Test Score: " + str(clf.score(train_x, train_y)) + '\n')
    return clf


# Note alpha needs to grow exponentially!
def tune_brain(train_x, train_y, n_fold=10, slow=False):
    # Want to go from 0.001 to 1, but on exponential scale!
    alphas = np.logspace(start=-5, stop=0, endpoint=True, num=5)
    hidden_layer = np.arange(3, 10, 1)
    solvers = ['lbfgs', 'adam']
    param_grid = {'alpha': alphas, 'hidden_layer_sizes': hidden_layer, 'solver': solvers}

    clf = MLPClassifier(warm_start=False)
    if slow:
        best_clf = GridSearchCV(clf, param_grid, n_jobs=-1, cv=n_fold)
    else:
        best_clf = RandomizedSearchCV(clf, param_grid, n_jobs=-1, cv=n_fold)

    best_clf.fit(train_x, train_y)
    plot_grid_search(best_clf, 'alpha', 'NN')
    plot_grid_search(best_clf, 'hidden_layer_sizes', 'NN')
    plot_grid_search(best_clf, 'solver', 'NN')
    dump(best_clf, "./Classifiers/" + type(clf).__name__ + ".joblib")
    return best_clf
