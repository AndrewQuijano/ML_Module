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

    print()
    with open("results.txt", "a+") as my_file:
        my_file.write("[NB] Training Mean Test Score: " + str(clf.score(train_x, train_y)) + '\n')
        my_file.write("[NB Isotonic] Training Mean Test Score: " + str(clf_isotonic.score(train_x, train_y)) + '\n')
        my_file.write("[NB Sigmoid] Training Mean Test Score: " + str(clf_sigmoid.score(train_x, train_y)) + '\n')
    dump(clf, "./Classifiers/" + type(clf).__name__ + ".joblib")
    dump(clf_sigmoid, "./Classifiers/" + type(clf_isotonic).__name__ + "_Isotonic.joblib")
    dump(clf_isotonic, "./Classifiers/" + type(clf_sigmoid).__name__ + "_Sigmoid.joblib")
    return clf, clf_isotonic, clf_sigmoid
