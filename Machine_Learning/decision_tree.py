import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from joblib import dump


def tune_tree(train_x, train_y, n_fold=10, slow=False, n_iter_search=10):
    # Minimum number of samples required to split a node
    min_samples_split = np.arange(5, 20, 1)
    # Minimum number of samples required at each leaf node
    min_samples_leaf = np.arange(5, 20, 1)
    # Maximum number of levels in tree
    max_depth = np.arange(3, 20, 1)

    random_grid = {
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_depth': max_depth
    }

    if slow:
        tree = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=random_grid,
                            cv=n_fold, verbose=2, n_jobs=-1)
    else:
        tree = RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions=random_grid,
                                  cv=n_fold, verbose=2, n_iter=n_iter_search, n_jobs=-1)
    tree.fit(train_x, train_y)
    plot_grid_search(tree, 'min_samples_split', 'Decision_Tree')
    plot_grid_search(tree, 'min_samples_leaf', 'Decision_Tree')
    plot_grid_search(tree, 'max_depth', 'Decision_Tree')
    return tree


def get_tree(train_x, train_y, n_fold=10, slow=False):
    start_time = time.time()
    tree = tune_tree(train_x, train_y, n_fold, slow)
    print("--- Best Parameter Decision Tree Time: %s seconds ---" % (time.time() - start_time))
    print("Best Decision Tree Parameters: " + str(tree.best_params_))
    print("[Decision_Tree] Training Mean Test Score: " + str(tree.score(train_x, train_y)))

    with open("results.txt", "a+") as my_file:
        my_file.write("[Decision Tree] Best Parameters: " + str(tree.best_params_) + '\n')
        my_file.write("[Decision Tree] Training Mean Test Score: " + str(tree.score(train_x, train_y)) + '\n')
    dump(tree, "./Classifiers/Decision_Tree.joblib")
    return tree


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
