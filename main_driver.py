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
import collections
from collections import OrderedDict
from operator import itemgetter
from math import sqrt
from os.path import basename, dirname, abspath
from os import name


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
        p = dirname(abspath(argv[1]))
        b = basename(argv[1])

        # Format columns to be 1-D shape
        train_y = train_y.reshape(-1, 1)
        test_y = test_y.reshape(-1, 1)
        train = np.concatenate((train_y, train_x), axis=1)
        test = np.concatenate((test_y, test_x), axis=1)

        if name == 'nt':
            np.savetxt(p + '\\train_' + b, train, fmt="%s", delimiter=",")
            np.savetxt(p + "\\test_" + b, test, fmt="%s", delimiter=",")
        else:
            np.savetxt(p + "/train_" + b, train, fmt="%s", delimiter=",")
            np.savetxt(p + "/test_" + b, test,  fmt="%s", delimiter=",")
        exit(0)

    elif len(argv) == 3:
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
        print("Usage: python3 main_driver <train-set> <test-set>")
        exit(0)

    # First thing, Check if there was a previous run or not!
    # Then the user chooses to delete and run or not
    start_and_clean_up(test_x, test_y)

    # Now train ALL classifiers!
    clf_list(train_x, train_y, False)

    # Run Testing Now
    load_and_test(test_x, test_y)


def clf_list(train_x, train_y, speed):
    kf = KFold(n_splits=10, shuffle=False)
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


# Use this to run IDS using Classifier!
def ids():
    # To run the program I need either
    # 1- the raw PCAP with labels?
    # 2- just the raw CSV pre-processed using Stolfo's KDD Data mining techniques
    if len(argv) != 2:
        exit("usage: python3 main_ids <training data set>")
    # PCAP File not accepted for training!
    if not is_valid_file_type(argv[1]):
        exit("Invalid file type! only accept.txt or .csv file extensions!")

    print("Please wait...Reading the Training Data ML...")
    train_x, train_y = read_data(argv[1])
    print("Please wait...Training Data read! Setting up ML Models!")

    # Now make a split between training and testing set from the input data
    start_time = time.time()
    kf = KFold(n_splits=5, shuffle=False)

    # 1- Bayes
    print("Fitting Bayes Classifiers...")
    bayes, bayes_isotonic, bayes_sigmoid = naive_bayes(train_x, train_y, n_fold=kf)
    print("Bayes classifier ready!")

    # 2- LDA/QDA
    print("Fitting LDA and QDA...")
    lda_clf = discriminant_line(train_x, train_y)
    print("LDA ready!")

    qda_clf = discriminant_quad(train_x, train_y)
    print("QDA ready!")

    # 3- SVM
    print("Fitting Linear SVM...")
    svm_line_clf = svm_linear(train_x, train_y, n_fold=kf, slow=False)

    # print("SVM Linear Model Ready!")
    print("Fitting RBF SVM...")
    svm_rbf_clf = svm_rbf(train_x, train_y, n_fold=kf, slow=False)
    print("SVM RBF Kernel Ready!")

    # 4- Random Forest
    print("Fitting Random Forest...")
    forest_clf = get_forest(train_x, train_y, n_fold=kf, slow=False)
    print("Random Forest Ready!")

    # 5- Logistic Regression
    print("Fitting Logistic Regression...")
    logistic_clf = get_logistic(train_x, train_y, n_fold=kf, slow=False)
    print("Logistic Regression Ready!")

    # 6- KNN
    print("Fitting KNN...")
    knn_clf = get_knn(train_x, train_y, n_fold=kf, slow=False)
    print("KNN ready!")

    # 7- Decision Tree
    print("Fitting Decision tree...")
    tree = get_tree(train_x, train_y, n_fold=kf, slow=False)
    print("Decision tree ready!")

    print("--- Model Training Time: %s seconds ---" % (time.time() - start_time))
    print("All models are trained... They are dumped as well")


# THIS IS FOR LABELING TEST DATA GENERATED BY THE FUZZER!
# THE OUTPUT OF KDDPROCESSOR 99 -E Spits last 5 extra columns...
# SRC IP, SRC PORT, DEST IP, DEST PORT, TIME STAMP
# SINCE U KNOW THE ATTACKS ARE BY SPECIFIC IP, USE THAT TO LABEL
# PLAY WITH COLUMN 28-32
# GOAL: LABEL IS ON FIRST COLUMN
def label_testing_set(file_path, output):
    # From fuzzer I know the mapping of IP and attack
    # 192.168.147.152 is IP of Client running Kali Linux
    attack_map = {"192.168.147.150": "back.", "192.168.147.151": "neptune.",
                  "192.168.147.152": "satan.", "192.168.147.153": "teardrop.", "192.168.147.154": "pod.",
                  "192.168.147.160": "ipsweep.", "192.168.147.161": "portsweep.", "192.168.147.162": "portsweep."}
    # Pulled from NSL-KDD Labels
    label_map = {"normal.": 11, "back.": 0, "ipsweep.": 5, "land.": 6, "neptune.": 9, "pod.": 14,
                 "portsweep.": 15, "satan.": 17, "smurf.": 18, "teardrop.": 20}

    # DON'T FORGET TO LABEL THE FEATURES. See labels.txt in NSL data set folder
    # Protocol
    label_protocol = {
        'tcp': 1, 'udp': 2, 'icmp': 0
    }
    # Flag
    label_flag = {
        'SF': 9, 'S2': 7, 'S1': 6, 'S3': 8,
        'OTH': 0, 'REJ': 1, 'RSTO': 2, 'S0': 5, 'RSTR': 4,
        'RSTOS0': 3, 'SH': 10
    }
    # Service
    label_service = {
        'http': 24, 'smtp': 54, 'domain_u': 12, 'auth': 4,
        'finger': 18, 'telnet': 60, 'eco_i': 14, 'ftp': 19, 'ntp_u': 43,
        'ecr_i': 15, 'other': 44, 'urp_i': 65, 'private': 49, 'pop_3': 47,
        'ftp_data': 20, 'netstat': 40, 'daytime': 9, 'ssh': 56, 'echo': 13,
        'time': 63, 'name': 36, 'whois': 69, 'domain': 11, 'mtp': 35,
        'gopher': 21, 'remote_job': 51, 'rje': 52, 'ctf': 8, 'supdup': 58,
        'link': 33, 'systat': 59, 'discard': 10, 'X11': 1, 'shell': 53,
        'login': 34, 'imap4': 28, 'nntp': 42, 'uucp': 66, 'pm_dump': 45,
        'IRC': 0, 'Z39_50': 2, 'netbios_dgm': 37, 'ldap': 32, 'sunrpc': 57,
        'courier': 6, 'exec': 17, 'bgp': 5, 'csnet_ns': 7, 'http_443': 26,
        'klogin': 30, 'printer': 48, 'netbios_ssn': 39, 'pop_2': 46, 'nnsp': 41,
        'efs': 16, 'hostnames': 23, 'uucp_path': 67, 'sql_net': 55, 'vmnet': 68,
        'iso_tsap': 29, 'netbios_ns': 38, 'kshell': 31, 'urh_i': 64, 'http_2784': 25,
        'harvest': 22, 'aol': 3, 'tftp_u': 61, 'http_8001': 27, 'tim_i': 62,
        'red_i': 50, 'oth_i': 70
    }

    # Features are on Columns 1, 2, 3
    with open(file_path, "r") as read, open(output, "w+") as write:
        for line in read:
            ln = line.rstrip()
            parts = ln.split(',')

            # DON'T FORGOT TO ENCODE NOW!
            parts[1] = str(label_protocol[parts[1]])
            parts[2] = str(label_service[parts[2]])
            parts[3] = str(label_flag[parts[3]])

            # signature of land
            if parts[28] == parts[30]:
                parts.insert(0, str(label_map["land."]))
            elif parts[28] in attack_map:
                lab = attack_map[parts[28]]
                parts.insert(0, str(label_map[lab]))
            elif parts[30] in attack_map:
                lab = attack_map[parts[30]]
                parts.insert(0, str(label_map[lab]))
            else:
                parts.insert(0, str(label_map["normal."]))

            # drop the columns and write
            parts = parts[:29]
            new_line = ','.join(parts)
            write.write(new_line + '\n')
            write.flush()


def stat_column(data_set, label, column_number=2):
    freq_n = {}
    freq_a = {}
    with open(data_set, "r") as f:
        for line in f:
            try:
                # Get the right column
                row = line.split(",")
                key = row[column_number]
                if row[41] != label:
                    if key in freq_a:
                        freq_a[key] = freq_a[key] + 1
                    else:
                        freq_a[key] = 1
                else:
                    if key in freq_n:
                        freq_n[key] = freq_n[key] + 1
                    else:
                        freq_n[key] = 1

            except ValueError:
                exit("NAN FOUND!")

    # Using frequency map compute mean and std dev
    # print(mean_freq(freq_n))
    # print(std_dev_freq(freq_n))

    order_freq_n = collections.OrderedDict(sorted(freq_n.items().__iter__()))
    order_freq_a = collections.OrderedDict(sorted(freq_a.items().__iter__()))
    if len(freq_a) == 0:
        frequency_histogram(order_freq_n)
    else:
        dual_frequency_histogram(order_freq_n, order_freq_a)


# Purpose: Just get the stats. NO HISTOGRAM
def stat_one_column(data_set, label, column_number=2):
    freq_n = {}
    with open(data_set, "r") as f:
        for line in f:
            try:
                # Get the right column
                row = line.split(",")
                key = row[column_number]
                if row[0] == label:
                    if key in freq_n:
                        freq_n[key] = freq_n[key] + 1
                    else:
                        freq_n[key] = 1
                else:
                    continue
            except ValueError:
                exit("NAN FOUND!")
    # print contents
    u = mean_freq(freq_n)
    s = std_dev_freq(freq_n, u)

    # To make it easier to figure out most frequent feature value
    # sort the map by value!
    sorted_freq = OrderedDict(sorted(freq_n.items().__iter__(), key=itemgetter(1)))

    with open("stat_result_" + label + ".txt", "a+") as fd:
        fd.write("-----for Column " + str(column_number) + "-----\n")
        fd.write(print_map(sorted_freq) + '\n')
        fd.write("The mean is: " + str(u) + '\n')
        fd.write("The standard deviation is: " + str(s) + '\n')


def mean_freq(freq):
    n = sum(list(freq.values()))
    miu = 0
    for key, value in freq.items():
        miu = miu + float(key) * value
    miu = miu/n
    return miu


def print_map(hash_map, per_row=5):
    line_counter = 1
    answer = "{\n"
    for k, v in hash_map.items():
        if line_counter % per_row == 0:
            answer = answer + '\n'
        line = str(k) + ":" + str(v) + " "
        answer = answer + line
        line_counter += 1
    answer = answer + "\n}"
    return answer


def std_dev_freq(freq, miu=None):
    if miu is None:
        miu = mean_freq(freq)
    n = sum(list(freq.values()))
    sigma = 0
    for val, f in freq.items():
        sigma += f * (float(val) - miu) * (float(val) - miu)
    sigma = sigma/n
    sigma = sqrt(sigma)
    return sigma


def stats_columns(file_name, label):
    for col in range(0, 29, 1):
        stat_one_column(file_name, label, column_number=col)


if __name__ == "__main__":
    main()
