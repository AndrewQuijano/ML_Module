from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
import time
from joblib import dump


# http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py
def naive_bayes(train_x, train_y, n_fold=10):
    # Gaussian Naive-Bayes with no calibration
    clf = GaussianNB()
    # Gaussian Naive-Bayes with isotonic calibration
    clf_isotonic = CalibratedClassifierCV(clf, cv=n_fold, method='isotonic')
    # Gaussian Naive-Bayes with sigmoid calibration
    clf_sigmoid = CalibratedClassifierCV(clf, cv=n_fold, method='sigmoid')

    start_time = time.time()
    clf.fit(train_x, train_y)
    clf_isotonic.fit(train_x, train_y)
    clf_sigmoid.fit(train_x, train_y)
    print("--- Time to fit 3 Bayes Classifiers: %s seconds ---" % (time.time() - start_time))

    with open("results.txt", "a+") as my_file:
        my_file.write("[NB] Training Mean Test Score: " + str(clf.score(train_x, train_y)) + '\n')
        my_file.write("[NB Isotonic] Training Mean Test Score: " + str(clf_isotonic.score(train_x, train_y)) + '\n')
        my_file.write("[NB Sigmoid] Training Mean Test Score: " + str(clf_sigmoid.score(train_x, train_y)) + '\n')
    dump(clf, "./Classifiers/NB.joblib")
    dump(clf_sigmoid, "./Classifiers/NB_Sigmoid.joblib")
    dump(clf_isotonic, "./Classifiers/NB_Isotonic.joblib")
    return clf, clf_isotonic, clf_sigmoid


def plot_grid_search(clf, name_param, clf_name, directory="./Cross_Validation/"):
    import numpy as np
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
