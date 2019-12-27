from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import time
from joblib import dump
import numpy as np


def get_forest(train_x, train_y, n_fold=10, slow=False):
    start_time = time.time()
    best_forest = tune_forest(train_x, train_y, n_fold, slow)
    print("--- Best Parameter Random Forest Time: %s seconds ---" % (time.time() - start_time))
    print("Best Random Forest Parameters: " + str(best_forest.best_params_))
    print("[Random_Forest] Training Mean Test Score: " + str(best_forest.score(train_x, train_y)))
    with open("results.txt", "a+") as my_file:
        my_file.write("[Random_Forest] Best Parameters: " + str(best_forest.best_params_) + '\n')
        my_file.write("[Random_Forest] Training Mean Test Score: " + str(best_forest.score(train_x, train_y)) + '\n')
    dump(best_forest, "./Classifiers/Random_Forest.joblib")
    return best_forest


# Citation:
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
def tune_forest(train_features, train_labels, n_fold=10, slow=False):
    # Number of trees in random forest
    n_estimators = np.arange(10, 510, 10)
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = np.arange(3, 20, 1)
    # Minimum number of samples required to split a node
    min_samples_split = np.arange(5, 20, 1)
    # Minimum number of samples required at each leaf node
    min_samples_leaf = np.arange(5, 20, 1)

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
    }

    # Step 1: Use the random grid to search for best hyper parameters
    # First create the base model to tune
    rf = RandomForestClassifier(warm_start=False, n_estimators=100)
    if slow:
        tune_rf = GridSearchCV(estimator=rf, param_grid=random_grid, cv=n_fold, n_jobs=-1, verbose=2)
    else:
        tune_rf = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                     cv=n_fold, n_jobs=-1, verbose=2)
    tune_rf.fit(train_features, train_labels)

    plot_grid_search(tune_rf, 'n_estimators', 'Random_Forest')
    plot_grid_search(tune_rf, 'max_features', 'Random_Forest')
    plot_grid_search(tune_rf, 'max_depth', 'Random_Forest')
    plot_grid_search(tune_rf, 'min_samples_split', 'Random_Forest')
    plot_grid_search(tune_rf, 'min_samples_leaf', 'Random_Forest')
    return tune_rf


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
