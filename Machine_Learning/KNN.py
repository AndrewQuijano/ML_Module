import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
import numpy as np


# https://www.pyimagesearch.com/2016/08/15/how-to-tune-hyperparameters-with-python-and-scikit-learn/
def get_knn(train_x, train_y, n_fold=10, slow=False):
    n = np.arange(3, 22, 2)
    start = time.time()
    # tune the hyper parameters via a randomized search
    if slow:
        best_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': n},
                                n_jobs=-1, cv=n_fold, verbose=2)
    else:
        best_knn = RandomizedSearchCV(estimator=KNeighborsClassifier(), param_distributions={'n_neighbors': n},
                                      n_jobs=-1, cv=n_fold, verbose=2)
    best_knn.fit(train_x, train_y)
    # Plot the CV-Curve
    plot_grid_search(best_knn, 'n_neighbors', 'KNN')
    print("[INFO] KNN-Best Parameters: " + str(best_knn.best_params_))
    print("[INFO] Tuning took {:.2f} seconds".format(time.time() - start))
    print("[KNN] Training Score is: " + str(best_knn.score(train_x, train_y)))

    with open("results.txt", "a+") as my_file:
        my_file.write("[KNN] KNN-Best Parameters: " + str(best_knn.best_params_) + '\n')
        my_file.write("[KNN] Training Mean Test Score: " + str(best_knn.score(train_x, train_y)) + '\n')
    dump(best_knn, "./Classifiers/KNN.joblib")
    return best_knn


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
