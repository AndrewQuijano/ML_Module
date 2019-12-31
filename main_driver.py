#!/usr/bin/env python3
from Machine_Learning.bayes import *
from Machine_Learning.discriminant import *
from Machine_Learning.KNN import *
from Machine_Learning.logistic_regression import *
from Machine_Learning.random_forest import *
from Machine_Learning.svm import *
from Machine_Learning.decision_tree import *
from Machine_Learning.neural_network import *
from Machine_Learning.misc import *

from sys import argv, exit
from sklearn.model_selection import KFold
from os.path import basename


# Just test functionality of script!
def main():
    train_x = None
    train_y = None
    test_x = None
    test_y = None

    # Check if both sets are available
    if len(argv) == 2:
        # Read the training and testing data-set
        # This assumes the class variable is on the first column!
        # It also assumes all data is numeric!
        if is_valid_file_type(argv[1]):
            train_x, train_y = read_data(argv[1])
        else:
            print("Training Set Not Found or invalid file extension!")
            exit(0)

        # Now make a split between training and testing set from the input data
        train_x, train_y, test_x, test_y = get_cv_set(train_x, train_y)
        b = basename(argv[1])

        # Format columns to be 1-D shape
        train_y = train_y.reshape(-1, 1)
        test_y = test_y.reshape(-1, 1)
        train = np.concatenate((train_y, train_x), axis=1)
        test = np.concatenate((test_y, test_x), axis=1)

        np.savetxt("./train_" + b, train, fmt="%s", delimiter=",")
        np.savetxt("./test_" + b, test,  fmt="%s", delimiter=",")
        exit(0)

    elif len(argv) == 4:
        # Read the training and testing data-set
        # This assumes the class variable is on the first column!
        # It also assumes all data is numeric!
        if is_valid_file_type(argv[1]):
            train_x, train_y = read_data(argv[1])
        else:
            print("Training Set Not Found or invalid file extension!")
            exit(0)

        if is_valid_file_type(argv[2]):
            test_x, test_y = read_data(argv[2])
        else:
            print("Testing Set Not Found or invalid file extension!")
            exit(0)
    else:
        print("Usage: python3 main_driver.py <train-set> <test-set> <True/False Speed>")
        exit(0)

    # First thing, Check if there was a previous run or not!
    # Then the user chooses to delete and run the script!
    start()

    # Now train ALL classifiers and dump the classifiers!
    if argv[3] == '0':
        clf_list(train_x, train_y, False)
    else:
        clf_list(train_x, train_y, True)

    # Run Testing Now
    load_and_test(test_x, test_y)


def clf_list(train_x, train_y, speed):
    kf = KFold(n_splits=5, shuffle=False)
    names = ["SVM_Linear",
             "SVM_Radial",
             "Random_Forest",
             "Logistic_Regression",
             "KNN",
             "LDA",
             "QDA",
             "Decision_tree",
             "NB",
             "NB_Isotonic",
             "NB_Sigmoid"
             ]

    # 1- SVM
    start_time = time.time()
    svm_line_clf = svm_linear(train_x, train_y, kf, speed)
    svm_rbf_clf = svm_rbf(train_x, train_y, kf, speed)

    # 2- Random Forest
    forest_clf = get_forest(train_x, train_y, kf, speed)

    # 3- Logistic Regression
    logistic_clf = get_logistic(train_x, train_y, kf, speed)

    # 4- KNN
    knn_clf = get_knn(train_x, train_y, kf, speed)

    # 5- LDA/QDA
    lda_clf = discriminant_line(train_x, train_y)
    qda_clf = discriminant_quad(train_x, train_y)

    # 6- Bayes
    bayes, bayes_isotonic, bayes_sigmoid = naive_bayes(train_x, train_y)

    # 7- Decision Tree
    tree = get_tree(train_x, train_y, kf, speed)

    # 8- Neural Networks
    # brain_clf = get_brain(train_x, train_y, kf, speed)
    classifiers = [svm_line_clf, svm_rbf_clf, forest_clf, logistic_clf, knn_clf,
                   lda_clf, qda_clf, tree, bayes, bayes_isotonic, bayes_sigmoid
                   ]
    print("---Time to complete training everything: %s seconds---" % (time.time() - start_time))
    return names, classifiers


if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python3 main_driver.py test <test-set>")
        print("Usage: python3 main_driver.py <train-set> <test-set> <True/False Speed>")
        exit(0)

    if argv[1] == 'test':
        # Just run some test sets!
        test_set_x, test_set_y = read_data(argv[2])
        load_and_test(test_set_x, test_set_y)
    else:
        # Run full script with training and everything!
        main()
