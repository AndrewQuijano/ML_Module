#!/usr/bin/env python3

# Needed for main test driver (main method)
from sys import argv
from os.path import basename

# first of all import the socket library
# main issue: http://scikit-learn.org/stable/modules/scaling_strategies.html
import socket
# To delete?
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load

from ..Machine_Learning.misc import *
from ..Machine_Learning.incremental_learners import incremental_clf_list
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor

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
            data = read_from_socket(connection).decode()
            print("Input is: " + data)
            # Check the first part! for 'test' or 'tran' -> 'train'
            cmd = data[:4]
            data = data[4:]

            if cmd == "tran":
                x, y = parse_string_to_numpy(data, True)
                # Error occurred in converting string to numpy!
                if x is None:
                    connection.close()
                    continue

                # 1- Write the data to a CSV file
                with open("./data_set.csv") as file:
                    file.write(data + '\n')

                # 2- Check if it is time to tune classifier?

                # 3- Update Classifiers
                bayes.partial_fit(x, y, classes=None)
                connection.close()

            elif cmd == "test":
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

            elif cmd == "exit":
                connection.close()
                break

            else:
                connection.close()
                continue

        except KeyboardInterrupt:
            print('CTRL-C received, Exit!')
            break

    server_socket.close()


# test driver only on local host with ML model, see main_driver.py in ML python library for basic structure/logic
# Test with ZIP code data set
def main():
    # Once server socket is ready get all classifiers up!
    # For Partial fit to work, I need to know all classes ahead of time!
    # classes = [3.0, 5.0, 6.0, 8.0]

    class_names = ["bayes", "percep", "sgd_class", "pa_classifier", "sgd_regress", "pa_regress"]
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
        incremental_clf_list(train_x, train_y, False)
    else:
        incremental_clf_list(train_x, train_y, True)

    # Run Testing Now
    load_and_test(test_x, test_y)


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


def load_and_test(test_x, test_y):
    clf_list = ["i_bayes.joblib", "sgd_class.joblib", "PA_class.joblib", "percep.joblib"]
    for clf_name in clf_list:
        try:
            clf = load("./Classifiers/" + clf_name)
        except FileNotFoundError:
            continue
        classifier_test(clf, clf_name.split('.')[0], test_x, test_y)


def create_server_socket(port_number, max_connections=5):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind(('', port_number))
    except socket.error:
        if server_socket:
            server_socket.close()
        return None
    server_socket.listen(max_connections)
    return server_socket


def read_from_socket(sock, buffer_size=1024):
    result = bytearray()
    while True:
        data = sock.recv(buffer_size)
        result.extend(data)
        if len(data) < buffer_size:
            break
    return result


if __name__ == "__main__":
    # Test basic running of incremental classifier
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

