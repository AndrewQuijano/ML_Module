import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
import numpy as np
from .misc import plot_grid_search

# https://www.pyimagesearch.com/2016/08/15/how-to-tune-hyperparameters-with-python-and-scikit-learn/
def get_knn(train_x, train_y, n_fold=10, slow=False):
    n = np.arange(3, 22, 2)
    start = time.time()
    # tune the hyper parameters via a randomized search
    clf = KNeighborsClassifier()
    if slow:
        best_knn = GridSearchCV(estimator=clf, param_grid={'n_neighbors': n},
                                n_jobs=-1, cv=n_fold, verbose=2)
    else:
        best_knn = RandomizedSearchCV(estimator=clf, param_distributions={'n_neighbors': n},
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
    dump(best_knn, "./Classifiers/" + type(clf).__name__ + ".joblib")
    return best_knn
