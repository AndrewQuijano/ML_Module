from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import dump
import time
import numpy as np


def svm_rbf(train_x, train_y, n_fold=10, slow=False):
    start_time = time.time()
    svm_radial = svc_rbf_param_selection(train_x, train_y, n_fold, slow)
    print("--- Best Parameter RBF Time to complete: %s seconds ---" % (time.time() - start_time))
    print("Best RBF Parameters: " + str(svm_radial.best_params_))
    print("[SVM_Radial] Training Mean Test Score: " + str(svm_radial.score(train_x, train_y)))
    with open("results.txt", "a+") as my_file:
        my_file.write("[SVM_Radial] Best Parameters: " + str(svm_radial.best_params_) + '\n')
        my_file.write("[SVM Radial] Training Mean Test Score: " + str(svm_radial.score(train_x, train_y)) + '\n')
    dump(svm_radial, "./Classifiers/SVM_Radial.joblib")
    return svm_radial


def svc_rbf_param_selection(x, y, n_folds=10, slow=False):
    c = np.arange(0.1, 1, 0.1)
    gammas = np.arange(0.1, 1, 0.1)
    random_grid = {
        'C': c,
        'gamma': gammas
    }
    model = svm.SVC(kernel='rbf', probability=True)
    if slow:
        rbf_search = GridSearchCV(model, param_grid=random_grid, cv=n_folds,
                                  n_jobs=-1, error_score='raise', verbose=2)
    else:
        rbf_search = RandomizedSearchCV(model, param_distributions=random_grid,
                                        cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    rbf_search.fit(x, y)
    plot_grid_search(rbf_search, 'C', 'SVM_RBF')
    plot_grid_search(rbf_search, 'gamma', 'SVM_RBF')
    return rbf_search


# http://scikit-learn.org/stable/modules/model_evaluation.html
def svm_linear(train_x, train_y, n_fold=10, slow=False):
    start_time = time.time()
    svm_line = svc_linear_param_selection(train_x, train_y, n_fold, slow)
    print("--- Best Parameter Linear SVM: %s seconds ---" % (time.time() - start_time))
    print("Best Linear Parameters: " + str(svm_line.best_params_))
    print("[SVM_Linear] Training Mean Test Score: " + str(svm_line.score(train_x, train_y)))
    with open("results.txt", "a+") as my_file:
        my_file.write("[SVM_Linear] Best Parameters: " + str(svm_line.best_params_) + '\n')
        my_file.write("[SVM_Linear] Training Mean Test Score: " + str(svm_line.score(train_x, train_y)) + '\n')
    dump(svm_line, "./Classifiers/SVM_Linear.joblib")
    return svm_line


def svc_linear_param_selection(x, y, n_folds=10, slow=False):
    c = np.arange(0.1, 1.1, 0.1)
    param_grid = {'C': c}
    model = svm.SVC(kernel='linear', probability=True)
    if slow:
        svm_line = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        svm_line = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    svm_line.fit(x, y)
    plot_grid_search(svm_line, 'C', 'SVM_Linear')
    return svm_line


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
