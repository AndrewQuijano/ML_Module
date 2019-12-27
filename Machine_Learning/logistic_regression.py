from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from joblib import dump
import numpy as np


def get_logistic(train_x, train_y, n_fold=10, slow=False):
    start = time.time()
    n = np.logspace(-3, 3)
    param_grid = {'C': n}
    log = LogisticRegression(warm_start=False, max_iter=6000, multi_class='ovr', solver='lbfgs')
    if slow:
        log_model = GridSearchCV(log, param_grid, n_jobs=-1, cv=n_fold, verbose=2)
    else:
        log_model = RandomizedSearchCV(log, param_grid, n_jobs=-1, cv=n_fold, verbose=2)
    log_model.fit(train_x, train_y)

    plot_grid_search(log_model, 'C', 'Logistic_Regression')

    print("[INFO] Logistic Regression-Best Parameters: " + str(log_model.best_params_))
    print("[INFO] randomized search took {:.2f} seconds".format(time.time() - start))
    print("[Logistic] Training Score is: " + str(log_model.score(train_x, train_y)))
    with open("results.txt", "a+") as my_file:
        my_file.write("[Logistic Regression] Best Parameters: " + str(log_model.best_params_) + '\n')
        my_file.write("[Logistic Regression] Training Mean Test Score: " +
                      str(log_model.score(train_x, train_y)) + '\n')
    dump(log_model, "./Classifiers/Logistic_Regression.joblib")
    return log_model


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
