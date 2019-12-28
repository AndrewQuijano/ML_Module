#!/usr/bin/env python3

# first of all import the socket library
# main issue: http://scikit-learn.org/stable/modules/scaling_strategies.html
import socket
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load

# from tuning.py
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor

# from misc.py
import itertools
import random
from os import mkdir, path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from collections import Counter
from csv import reader


# ORIGINAL PARTS WHICH ALREADY ARE INCREMENTAL!
def init_classifiers(train_x, train_y):
    if train_x is None or train_y is None:
        bayes = MultinomialNB()
        percep = Perceptron(warm_start=True, max_iter=10, tol=1e-3)
        sgd_class = SGDClassifier(warm_start=True, max_iter=10, tol=1e-3)
        pa_classifier = PassiveAggressiveClassifier(warm_start=True, max_iter=10, tol=1e-3)
        sgd_regress = SGDRegressor(warm_start=True, max_iter=10, tol=1e-3)
        pa_regress = PassiveAggressiveRegressor(warm_start=True, max_iter=10, tol=1e-3)
    else:
        kf = KFold(n_splits=10)
        bayes = tune_bayes(train_x, train_y, kf, False)
        percep = tune_perceptron(train_x, train_y, kf, False)
        sgd_class = tune_sgd_clf(train_x, train_y, kf, False)
        sgd_regress = tune_passive_aggressive_reg(train_x, train_y, kf, False)
        pa_classifier = tune_passive_aggressive_clf(train_x, train_y, kf, False)
        pa_regress = tune_passive_aggressive_reg(train_x, train_y, kf, False)
        # Get Parameters now
        with open("results.txt", "w+") as fd:
            fd.write("[bayes] Best Parameters: " + str(bayes.best_params_) + '\n')
            fd.write("[percep] Best Parameters: " + str(percep.best_params_) + '\n')
            fd.write("[sgd_class] Best Parameters: " + str(sgd_class.best_params_) + '\n')
            fd.write("[pa_classifier] Best Parameters: " + str(pa_classifier.best_params_) + '\n')
            fd.write("[sgd_regress] Best Parameters: " + str(sgd_regress.best_params_) + '\n')
            fd.write("[pa_regress] Best Parameters: " + str(pa_regress.best_params_) + '\n')

            fd.write("[bayes] Training Score: " + str(bayes.score(train_x, train_y)) + '\n')
            fd.write("[percep] Training Score: " + str(percep.score(train_x, train_y)) + '\n')
            fd.write("[sgd_class] Training Score: " + str(sgd_class.score(train_x, train_y)) + '\n')
            fd.write("[pa_classifier] Training Score: " + str(pa_classifier.score(train_x, train_y)) + '\n')
            fd.write("[sgd_regress] Training Score: " + str(sgd_regress.score(train_x, train_y)) + '\n')
            fd.write("[pa_regress] Training Score: " + str(pa_regress.score(train_x, train_y)) + '\n')
        # If trained, should just dump now...
        dump(bayes, "i_bayes.joblib")
        dump(sgd_class, "sgd_class.joblib")
        dump(sgd_regress, "sgd_regress.joblib")
        dump(pa_classifier, "PA_class.joblib")
        dump(pa_regress, "PA_regress.joblib")
        dump(percep, "percep.joblib")
    return [bayes, percep, sgd_class, pa_classifier, sgd_regress, pa_regress]


# Return X, Y for training, or just X to be used for classifiers
# This will ONLY WORK FOR ONE LINE!
def parse_string_to_numpy(data, training_phase=True):
    try:
        if training_phase:
            x = np.fromstring(data, dtype='float32', sep=',')
            y = x[0]
            x = x[1:]
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            return x, y
        else:
            x = np.fromstring(data, dtype='float32', sep=',')
            x = x[1:]
            x = x.reshape(1, -1)
            return x, None
    except ValueError:
        return None, None


def server():
    server_socket = create_server_socket(12345)
    if server_socket is None:
        exit("Failed to make Server Socket!")

    # Once server socket is ready get all classifiers up!
    bayes = MultinomialNB(class_prior=None, fit_prior=True)
    perceptron = Perceptron()
    sgd_class = SGDClassifier()
    pa_classifier = PassiveAggressiveClassifier()
    sgd_regressor = SGDRegressor()
    pa_regressor = PassiveAggressiveRegressor()

    # For Partial fit to work, I need to know all classes ahead of time!
    while True:
        try:

            # Establish connection with client.
            connection, address = server_socket.accept()
            print('Got connection from: ', address)

            # When starting to run server you have the following options
            # 1- Train model
            # Write to .csv file for tuning, update all models
            # 2- Test Model
            # input the features, get results and send back to client
            # 3- exit

            # input example, first column is the command, second column is label
            # To keep things simple, assume a comma separated string!
            # "train", 1, 0.25, 0.6, 0.8
            # "test", 0.5, 20, 2,
            # "exit"

            # ---PLEASE NOTE CURRENTLY THIS IS BUILD WITH ONE THING AT A TIME!---
            data = connection.recv(1024).decode()
            print("Input is: " + data)
            args = data.split(",")

            if args[0] == "train":
                x, y = parse_string_to_numpy(data, True)
                # Error occurred in converting string to numpy!
                if x is None:
                    connection.close()
                    continue

                # 1- Write the data to a CSV file
                with open("data_set.csv") as file:
                    file.write(data + '\n')

                # 2- Check if it is time to tune classifier?

                # 3- Update Classifiers

                bayes.partial_fit(x, y)
                connection.close()

            elif args[0] == "test":
                x, y = parse_string_to_numpy(data, False)
                # Error occurred in converting string to numpy!
                if x is None:
                    connection.close()
                    continue
                # Run the prediction and send the results back!
                else:
                    y_pred = bayes.predict(x)
                    connection.send(np.array_str(np.arange(1)).encode())
                connection.close()

            elif args[0] == "exit":
                connection.close()
                break

            else:
                connection.close()
                continue

        except KeyboardInterrupt:
            print('CTRL-C received, Exit!')
            break

    server_socket.close()


# test driver only on local host with ML model, see main_driver in ML python library
# Test with ZIP code data set
def main(train_data):
    # Once server socket is ready get all classifiers up!
    # For Partial fit to work, I need to know all classes ahead of time!
    # classes = [3.0, 5.0, 6.0, 8.0]

    train_x, train_y = read_data(train_data)
    class_names = ["bayes", "percep", "sgd_class", "pa_classifier", "sgd_regress", "pa_regress"]
    # classes = np.arange(0, 23, 1, dtype=float)
    # [0, 23) or [0, 22]
    classifiers = init_classifiers(train_x, train_y)

    # Train it
    # TODO: Read say 100 lines, make to Numpy THEN FIT
    # with open(train_data, "r") as file:
    #    for line in file:
    #        x, y = parse_string_to_numpy(line.rstrip())
    #        for clf in classifiers:
    #            clf.partial_fit(x, y, classes=classes)

    # Ideally figure out how to tune after partial fit ONCE in...
    # I guess would I technically keep a record and try to refit?

    while True:
        try:
            arg = input("Input: ")
            args = arg.split()
            if args[0] == "exit":
                break

            elif args[0] == "detect":  # args: csv type
                test_x, test_y = read_data(args[1])
                for idx in range(len(classifiers)):
                    incremental_test(classifiers[idx], test_x, test_y, class_names[idx])
        except KeyboardInterrupt:
            break
        except EOFError:
            break


def incremental_test(clf, test_x, test_y, clf_name):
    y_hat = clf.predict(test_x)

    print("Testing Mean Test Score " + str(accuracy_score(test_y, y_hat)))
    make_confusion_matrix(y_true=test_y, y_pred=y_hat, clf=clf, clf_name=clf_name)

    with open("results.txt", "a") as my_file:
        my_file.write("[" + clf_name + "] Testing Mean Test Score: " + str(accuracy_score(test_y, y_hat)) + '\n')

    with open("classification_reports.txt", "a") as my_file:
        my_file.write("---[" + clf_name + "]---\n")
        my_file.write(classification_report(y_true=test_y, y_pred=y_hat, labels=clf.classes_,
                                            target_names=[str(i) for i in clf.classes_]))
        my_file.write('\n')


def load_test(file_path):
    test_x, test_y = read_data(file_path)
    bayes = load("i_bayes.joblib")
    sgd = load("sgd_class.joblib")
    pa = load("PA_class.joblib")
    percep = load("percep.joblib")
    incremental_test(bayes, test_x, test_y, "Bayes")
    incremental_test(sgd, test_x, test_y, "SGD")
    incremental_test(pa, test_x, test_y, "Passive_Aggressive")
    incremental_test(percep, test_x, test_y, "Perceptron")


def create_server_socket(port_number, max_connections=5):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind(('', port_number))
    except socket.error:
        if server_socket:
            server_socket.close()
        return None
    server_socket.listen(max_connections)  # Now wait for client connection.
    return server_socket


if __name__ == "__main__":
    load_test("./shit.csv")


# -------------------------Tune Classifier-----------------------------------------------------------
def tune_bayes(x, y, n_folds=10, slow=True):
    print("Tuning Multinomial Bayes...")
    c = np.arange(0.01, 1.3, 0.01)
    param_grid = {'alpha': c}
    model = MultinomialNB()
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, 'Multinomial_Bayes')
    print("Finished Tuning Multinomial Bayes...")
    return true_model


def tune_perceptron(x, y, n_folds=10, slow=True):
    print("Tuning Perceptron...")
    c = np.arange(0.01, 1.3, 0.01)
    param_grid = {'alpha': c}
    model = Perceptron(tol=1e-3, warm_start=True)
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, "Perceptron")
    print("Finished Tuning Perceptron...")
    return true_model


def tune_sgd_clf(x, y, n_folds=10, slow=True):
    print("Tuning SGD Classifier...")
    c = np.arange(0.0001, 0.01, 0.00001)
    param_grid = {'alpha': c}
    model = SGDClassifier(warm_start=True, tol=1e-3)
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, 'SGD_Classifier')
    print("Finished Tuning SGD Classifier...")
    return true_model


def tune_sgd_reg(x, y, n_folds=10, slow=True):
    print("Tuning SGD Regressor...")
    c = np.arange(0.0001, 0.01, 0.00001)
    param_grid = {'alpha': c}
    model = SGDRegressor(warm_start=True, tol=1e-3)
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, 'SGD_Regression')
    print("Finished Tuning SGD Regressor...")
    return true_model


def tune_passive_aggressive_clf(x, y, n_folds=10, slow=True):
    print("Tuning Passive Aggressive Classifier...")
    c = np.arange(0.01, 1.6, 0.01)
    param_grid = {'C': c}
    model = PassiveAggressiveClassifier(warm_start=True, tol=1e-3)
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, 'Passive_Aggressive_CLF')
    print("Finished Tuning Passive Aggressive Classifier...")
    return true_model


def tune_passive_aggressive_reg(x, y, n_folds=10, slow=True):
    print("Tuning Passive Aggressive Regression Classifier...")
    c = np.arange(0.01, 1.6, 0.01)
    param_grid = {'C': c}
    model = PassiveAggressiveRegressor(warm_start=True, tol=1e-3)
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, 'Passive_Aggressive_Regression')
    print("Finished Tuning Passive Aggressive Regression...")
    return true_model


# --------------------------------------misc.py--------------------------------------------------------------
def read_data(file, skip_head=True):
    if skip_head:
        features = np.genfromtxt(file, delimiter=',', skip_header=1, dtype=float, autostrip=True, converters=None)
    else:
        features = np.genfromtxt(file, delimiter=',', skip_header=0, dtype=float, autostrip=True, converters=None)

    if np.isnan(features).any():
        if skip_head:
            features = np.genfromtxt(file, delimiter=',', skip_header=1, dtype=str, autostrip=True, converters=None)
        else:
            features = np.genfromtxt(file, delimiter=',', skip_header=0, dtype=str, autostrip=True, converters=None)
        classes = features[:, 0]
        features = features[:, 1:]
        # Now you have NaN in your features, ok now you have issues!
        if np.isnan(features).any():
            print("There are NaNs found in your features at: " + str(list(map(tuple, np.where(np.isnan(features))))))
            exit(0)
        else:
            features.astype(float)
    else:
        classes = features[:, 0]
        features = features[:, 1:]

    return features, classes


def summation(elements):
    answer = 0
    for i in range(len(elements)):
        answer += elements[i]
    return answer


def is_valid_file_type(file):
    if not path.exists(file):
        return False
    if not path.isfile(file):
        return False
    return file.lower().endswith(('.csv', '.txt'))


def mean(elements):
    numerator = summation(elements)
    return numerator/len(elements)


def std_dev(elements):
    miu = mean(elements)
    variance = 0
    for i in range(len(elements)):
        variance += (elements[i] - miu) * (elements[i] - miu)
    variance = variance/len(elements)
    return variance


# Input: A file with numbers with frequencies:
# For example a List of Exam Scores:
# 80, 90, 100, 90, 75, ...
# Get <90, 2> <80, 1>, <90, 1>, in a dictionary to be used
def frequency_count(filename):
    # Read the input file into one long list
    objects = []
    with open(filename, 'r') as file:
        read_row = reader(file)
        for row in read_row:
            objects.append(row)
    counter = Counter(objects)
    return dict(counter)


# Input: A Hash Map <K, V> Key is item, Value is Frequency
# Plot a Histogram!
def frequency_histogram(hash_map):
    plt.bar(list(hash_map.keys()), hash_map.values(), color='g')
    plt.xlabel('elements')
    plt.ylabel('count')
    plt.title('Frequency histogram')
    plt.savefig(str('./histogram.png'))
    plt.show()
    plt.close()


def get_cv_set(training_set, test_set, percentile=0.2):
    row = np.shape(training_set)[0]
    col = np.shape(training_set)[1]
    sample_idx = random.sample(range(row), int(percentile * row))

    # Get your CV data
    cv_train = training_set[sample_idx[:], 0:col]
    cv_test = test_set[sample_idx[:]]

    # Remove CV data from original
    set_diff = np.setdiff1d(np.arange(row), sample_idx)
    training_set = training_set[set_diff[:], 0:col]
    test_set = test_set[set_diff[:]]
    return training_set, test_set, cv_train, cv_test


# Technically setting the extra attempts = 1 should be equivalent to getting you the test score
def top(clf, test_x, test_y, classifier, extra_attempts=1):
    # Get your list of classes
    # Sort it such that highest probabilities come first...
    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    # To print highest first, set reverse=True
    probability_dict = []
    for i in range(len(test_y)):
        if hasattr(clf, 'decision_function'):
            probability_dict.append(dict(zip(clf.classes_, clf.decision_function(test_x)[i])))
        else:
            probability_dict.append(dict(zip(clf.classes_, clf.predict_proba(test_x)[i])))
        probability_dict[i] = sorted([(v, k) for k, v in probability_dict[i].items()], reverse=True)

    success = 0
    # Let us say test the first 3 rooms? See if it matches!
    for i in range(len(test_y)):
        # print(probability_dict[i])
        for j in range(extra_attempts):
            if probability_dict[i][j][1] == test_y[i]:
                success = success + 1
                break

    # Print Results
    score = success/len(test_y)
    with open("results.txt", "a") as my_file:
        my_file.write("[" + classifier + "] Testing Mean Test Score with " + str(extra_attempts)
                      + ": " + str(score))
    # print("Test Error for " + str(extra_rooms) + " Rooms: " + str(success/len(test_y)))


def scale(train_x, test_x):
    scalar = StandardScaler()
    # Don't cheat - fit only on training data
    scalar.fit(train_x)
    x_train = scalar.transform(train_x)
    # apply same transformation to test data
    x_test = scalar.transform(test_x)
    return x_train, x_test


# If the data is co-linear you must use PCA
# Hopefully this function should get the PCA the explains up to 90% variance minimum
def scale_and_pca(train_x, test_x):
    scaled_train_x, scaled_test_x = scale(train_x, test_x)
    pr_comp = PCA(n_components=0.99, svd_solver='full')
    pr_comp.fit(scaled_train_x)
    return pr_comp.transform(scaled_train_x), pr_comp.transform(scaled_test_x)


def plot_grid_search(cv_results, grid_param, name_param, directory="Cross_Validation"):
    # Create target Directory if don't exist
    if not path.exists(directory):
        mkdir(directory)
    #    print("Directory ", directory, " Created! ")
    # else:
        # print("Directory ", "Cross_Validation", " already exists")

    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param))

    # Plot Grid search scores
    _, ax = plt.subplots(1, 1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    ax.plot(grid_param, scores_mean, label="CV-Curve")
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid(True)
    plt.savefig(str('./' + directory + '/CV_Plot_' + name_param + '.png'))
    plt.close()


# METHOD IS NOT USED AT THE MOMENT!
def plot_validation_curve(x, y, param_range, param_name, clf, clf_name):
    train_scores, test_scores = validation_curve(
        clf, x, y, param_name=param_name, param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with " + clf_name)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")


# Source code from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print("Normalized confusion matrix")
    # else:
    #    print('Confusion matrix, without normalization')
    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def make_confusion_matrix(y_true, y_pred, clf, clf_name, directory="Confusion_Matrix"):
    # Create target Directory if don't exist
    if not path.exists(directory):
        mkdir("Confusion_Matrix")
    #    print("Directory ", directory, " Created ")
    # else:
    #    print("Directory ", directory, " already exists")

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[str(i) for i in clf.classes_], normalize=False,
                          title='Confusion matrix, without normalization: ')
    plt.savefig(str('./' + directory + '/Normalized_Confusion_Matrix_' + clf_name + '.png'))
    # plt.show()
    plt.close()

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[str(i) for i in clf.classes_], normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(str('./' + directory + '/Normalized_Confusion_Matrix_' + clf_name + '.png'))
    # plt.show()
    plt.close()
