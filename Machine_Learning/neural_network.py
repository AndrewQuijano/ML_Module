from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from joblib import dump
import time
import numpy as np


# http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py
# http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
def get_brain(train_x, train_y, n_fold=10, slow=False):
    start_time = time.time()
    clf = tune_brain(train_x, train_y, n_fold, slow)
    print("--- Best Parameter NN Generation: %s seconds ---" % (time.time() - start_time))
    print("[NN] Training Mean Test Score: " + str(clf.score(train_x, train_y)))
    dump(clf, "./Classifiers/NN.joblib")

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

    if slow:
        clf = GridSearchCV(MLPClassifier(warm_start=False), param_grid, n_jobs=-1, cv=n_fold)
    else:
        clf = RandomizedSearchCV(MLPClassifier(warm_start=False), param_grid, n_jobs=-1, cv=n_fold)

    clf.fit(train_x, train_y)
    plot_grid_search(clf, 'alpha', 'NN')
    plot_grid_search(clf, 'hidden_layer_sizes', 'NN')
    plot_grid_search(clf, 'solver', 'NN')
    return clf


def plot_grid_search(clf, name_param, clf_name, directory="./Cross_Validation/"):
    from collections import OrderedDict
    from matplotlib import pyplot as plt

    # Get Test Scores Mean
    # Get the specific parameter to compare with CV
    coordinates = dict()
    scores_mean = clf.cv_results_['mean_test_score']
    parameters = clf.cv_results_['param_' + name_param]
    scores_mean = np.array(scores_mean).reshape(len(parameters), 1)

    # Step 1- Build dictionary
    for x, y in zip(parameters, scores_mean):
        if x not in coordinates:
            coordinates[x] = y
        else:
            if coordinates[x] > y:
                coordinates[x] = y

    # Step 2- Make into ordered set, sort by key!
    coordinates = OrderedDict(sorted(coordinates.items().__iter__()))

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    _, ax = plt.subplots(1, 1)
    ax.plot(coordinates.keys(), coordinates.values(), label="CV-Curve")
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid(True)
    plt.savefig(str(directory + 'CV_Plot_' + clf_name + '_' + name_param + '.png'))
    plt.close()
