from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import time
from joblib import dump


def discriminant_line(train_x, train_y):
    clf = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    start_time = time.time()
    clf.fit(train_x, train_y)
    print("--- Time to fit LDA: %s seconds ---" % (time.time() - start_time))
    print("Training Score (LDA): " + str(clf.score(train_x, train_y)))

    with open("results.txt", "a+") as my_file:
        my_file.write("[LDA] Training Mean Test Score: " + str(clf.score(train_x, train_y)) + '\n')
    dump(clf, "./Classifiers/" + type(clf).__name__  + ".joblib")
    return clf


def discriminant_quad(train_x, train_y):
    clf = QuadraticDiscriminantAnalysis(store_covariance=True)
    start_time = time.time()
    clf.fit(train_x, train_y)
    print("--- Time to fit QDA: %s seconds ---" % (time.time() - start_time))
    print("Training Score is (QDA): " + str(clf.score(train_x, train_y)))

    with open("results.txt", "a+") as my_file:
        my_file.write("[QDA] Training Mean Test Score: " + str(clf.score(train_x, train_y)) + '\n')
    dump(clf, "./Classifiers/" + type(clf).__name__  +".joblib")
    return clf
