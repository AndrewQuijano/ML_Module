from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import time
from joblib import dump


def discriminant_line(train_x, train_y):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    start_time = time.time()
    lda.fit(train_x, train_y)
    print("--- Time to fit LDA: %s seconds ---" % (time.time() - start_time))
    print("Training Score (LDA): " + str(lda.score(train_x, train_y)))

    with open("results.txt", "a+") as my_file:
        my_file.write("[LDA] Training Mean Test Score: " + str(lda.score(train_x, train_y)) + '\n')
    dump(lda, "./Classifiers/LDA.joblib")
    return lda


def discriminant_quad(train_x, train_y):
    qda = QuadraticDiscriminantAnalysis(store_covariance=False)
    start_time = time.time()
    qda.fit(train_x, train_y)
    print("--- Time to fit QDA: %s seconds ---" % (time.time() - start_time))
    print("Training Score is (QDA): " + str(qda.score(train_x, train_y)))

    with open("results.txt", "a+") as my_file:
        my_file.write("[QDA] Training Mean Test Score: " + str(qda.score(train_x, train_y)) + '\n')
    dump(qda, "./Classifiers/QDA.joblib")
    return qda


def plot_grid_search(clf, name_param, clf_name, directory="./Cross_Validation/"):
    from collections import OrderedDict
    from matplotlib import pyplot as plt
    import numpy as np

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
