#!/usr/bin/python3
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from os import remove, rename
from os.path import basename, isfile
import numpy as np


# The purpose of this class is mostly to modify data sets
# for machine learning purposes. This includes functions such as follows:

# NOTE: When using this script it ASSUMES class is the first column, aka column 0
# NOTE: PLEASE MAKE COPIES OF FILES BEING TAMPERED WITH! IT WILL BE DELETED IF IN SAME WORKING DIRECTORY!
# NOTE: All new files will be written to the current working directory of the shell script!
# 1- Drop Columns (INPUT: COLUMNS TO KEEP)
# 2- Merge/Split CSVs, this is because Github doesn't permit easy storage of big data sets!
# 3- Seen in NSL-KDD data set, remove ALL duplicate rows from data set!
# 4- Encode Data in specified column using regular label encoding (uses helper function 'filter_duplicate_features'
# 5- Encode Data in specified column using Hot Label Encoding
# 6- Uses Drop Rows, but given a column, if a certain value is found in that column, delete the row!
# 7- Shift Column (This is for moving the class column from last column to first column)
# 8- Drop row if at column X the feature is Y
# 9- Filter duplicate rows (Used by NSL-KDD from original KDD)


def load_label_encoder(loader):
    categories = []
    with open(loader, "r") as f:
        for line in f:
            line = line.rstrip()
            # If , there is <Key, Value>, Just get the key!
            if ',' in line:
                categories.append(line.split()[0])
    le = LabelEncoder()
    le.fit(categories)
    return le


def print_label_encoder(le, col):
    with open("./labels.txt", "a+") as f:
        f.write("For Column " + str(col) + '\n')
        features = le.classes_
        label = le.transform(features)
        label_map = dict(zip(features, label))
        for k, v in label_map.items():
            f.write(str(k) + "," + str(v) + '\n')
        f.write('\n')
    return label_map


# Makes a guess if the file has a header or not
# Assumes header will be always non-numeric names.
def has_header(file_name):
    with open(file_name) as read_data:
        # Read the first line and see if you fail to convert to int for first row
        for line in read_data:
            first_row = line.rstrip().split(',')
            num_col = len(first_row)
            counter = 0
            for element in first_row:
                try:
                    int(element)
                except ValueError:
                    counter += 1
            break
    return num_col == counter


# This is because the damn kdd does NOT have the headers pre-made...
def append_header(file_name='./kdd.csv', header_name="./header.txt"):
    b = basename(file_name)
    header_names = []
    # Get the header from "./header.txt"
    with open(header_name) as read_header:
        for line in read_header:
            header_names.append(line.rstrip())
    write_headers = ','.join(header_names)

    with open(file_name) as read_data, open('./prep_' + b, 'w+') as write_data:
        write_data.write(write_headers + '\n')
        for line in read_data:
            write_data.write(line)

    # Now you have the completed file, create the file in cwd!
    # You can over-write if the file exists in CWD!
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, b)
    print("Finished appending the header!")


def n_col(file_name):
    with open(file_name) as read_data:
        for line in read_data:
            line = line.rstrip()
            parts = line.split(",")
            if len(parts) > 0:
                break


def n_row(file_name):
    size = 0
    with open(file_name) as data:
        for line in data:
            if line is not None:
                size += 1
    return size


# For kdd, we skip columns 10 - 22
# Input: file is the CSV file to be modified
# Input: ranges is a list of tuples containing columns to NOT be dropped!
def drop_columns(file_name, keep_col_ranges):
    b = basename(file_name)
    use_cols = []
    for tup in keep_col_ranges:
        use_cols = use_cols + [i for i in range(tup[0], tup[1])]
    df = pd.read_csv(file_name, usecols=use_cols)
    df.to_csv("./prep_" + b, index=False)
    # Now you have the completed file, create the file in cwd!
    # You can over-write if the file exists in CWD!
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, b)
    print("Column Dropping complete, only kept columns: " + str(use_cols))


# Remember Github's Limit is 100 MB
# So to store large data sets online, I just split them up by 500,000 row chunks
# so <blah>/kdd.csv will become in cwd:
# 1_kdd.csv, 2_kdd.csv, etc.
def split_csv(file_name, size=500000):
    b = basename(file_name)
    line_number = 1
    file_part = 1
    lines = []
    with open(file_name, 'r') as big_file:
        for line in big_file:
            if line_number % size == 0:
                with open('./' + str(file_part) + '_' + b, 'w+') as chunk:
                    chunk.writelines(lines)
                lines = []
                file_part += 1
            else:
                lines.append(line)
            line_number += 1
    # Get the last chunk that is NOT 50,000 lines long!
    if len(lines) > 0:
        with open('./' + str(file_part) + '_' + b, 'w+') as chunk:
            chunk.writelines(lines)
    print("CSV split complete!")


# This is the counter-part function to split_csv
# THIS EXPECTS THE MERGING CSVs to be in same working directory!
# e. g. If I see ./1_kdd.csv, ./2_kdd.csv, the input should be 'kdd.csv'
# You should end up with a './kdd.csv'
def merge_csv(file_name):
    b = basename(file_name)
    file_part = 1
    lines = []
    while True:
        current_file_part = './' + str(file_part) + '_' + b
        if isfile(current_file_part):
            # Get the lines
            with open(current_file_part, 'r') as file_chunk:
                for line in file_chunk:
                    lines.append(line)
            # Append to bigger file!
            with open(file_name, 'a+') as big_file:
                big_file.writelines(lines)
            del lines[:]
            file_part += 1
        else:
            break
    print("Completed Merging the CSVs!")


# Given a file with CSV, use regular Label encoding!
def encode_data(file_name, col_to_encode):
    b = basename(file_name)
    if has_header(file_name):
        data = pd.read_csv(file_name)
    else:
        data = pd.read_csv(file_name, header=None)

    # In each column, get the unique columns
    for col in col_to_encode:
        le = LabelEncoder()
        data.iloc[:, col] = le.fit_transform(data.iloc[:, col])
        # Save encoder mapping!
        print_label_encoder(le, col)
    data.to_csv('./prep_' + b, index=False)

    # Now you have the completed file, create the file in cwd!
    # You can over-write if the file exists in CWD!
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, './' + b)
    print("Finished Data encoding!")


# It needs to be returned as list to be used by label encoder!
def filter_duplicate_features(file_name, col_number):
    header = has_header(file_name)
    s = set()
    with open(file_name, "r") as read:
        # If there is a header, don't include in encoding!
        if header:
            read.readline()

        for line in read:
            args = line.rstrip().split(",")
            s.add(args[col_number])
    return list(s)


# IT IS FUCKING CRITICAL THAT CLASS COLUMN IS ALWAYS ON FIRST COLUMN
def hot_encoder(file_name, encode_columns):
    b = basename(file_name)
    use_header = has_header(file_name)
    header_name = []
    new_headers = []
    if use_header:
        data = pd.read_csv(file_name)
        header_name = list(data.columns)
    else:
        data = pd.read_csv(file_name, header=None)

    # Label encode the whole column(s)!
    for col in encode_columns:
        le = LabelEncoder()
        data.iloc[:, col] = le.fit_transform(data.iloc[:, col])
        # Save encoder mapping!
        le_map = print_label_encoder(le, col)
        if use_header:
            del header_name[col]
            new_headers.extend(le_map.keys())

    # Now use the hot encoder! BE SURE TO NOT TOUCH THE CLASS LABEL!
    classes = data.iloc[:, 0]
    features = data.iloc[:, 1:]

    # NOW I MUST SHIFT THE COLUMNS ENCODED BY - 1
    for i in range(len(encode_columns)):
        if encode_columns[i] != 0:
            encode_columns[i] = encode_columns[i] - 1
        else:
            del encode_columns[i]
    one_hot_encoder = OneHotEncoder(categorical_features=encode_columns)
    data = one_hot_encoder.fit_transform(features).toarray()
    classes = classes.values.reshape(-1, 1)
    updated_data = np.concatenate((classes, data), axis=1)
    # Now you have the completed file, create the file in cwd!
    # final header
    if use_header:
        head = header_name[0] + ',' + ','.join(new_headers) + ',' + ','.join(header_name[1:])
        print(header_name[0])
    else:
        head = None
    np.savetxt('./prep_' + b, updated_data, fmt="%s", delimiter=",", header=head, comments='')
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, b)
    print("Hot Encoding complete!")


# A decent chunk of data sets prefer the class to be the last column
# Personally, I prefer it to be the first column. This method will adjust that for me
def shift_column(file_name):
    b = basename(file_name)

    with open(file_name) as read_data, open('./prep_' + b, "w+") as write_kdd:
        for line in read_data:
            # Swap using encoder
            line = line.rstrip()
            parts = line.split(",")
            # As my ML stuff excepts class on first column
            last_column = parts[len(parts) - 1]
            parts.remove(parts[len(parts) - 1])
            parts.insert(0, last_column)
            # Write the result
            new_line = ','.join(parts)
            write_kdd.write(new_line + '\n')
            write_kdd.flush()
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, b)
    print("Shifting Column complete!")


def filter_rows_by_feature(file_name, column_number, target_feature):
    b = basename(file_name)
    with open(file_name) as read_data, open('./prep_' + b, 'w+') as write_data:
        for line in read_data:
            lines = line.rstrip().split(',')
            if lines[column_number] != target_feature:
                write_data.write(','.join(lines) + '\n')

    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, b)
    print("All rows that at Column: " + str(column_number) + " has the feature: " + target_feature + " is now removed!")


# Use this method to ensure ALL rows in the data set are unique
# Write it out to new file
def filter_duplicate_rows(file_name):
    b = basename(file_name)
    s = set()
    with open(file_name, "r") as read:
        for line in read:
            s.add(line)
    with open('./prep_' + b, "w") as wr:
        for line in s:
            wr.write(line)
    # Now you have the completed file, create the file in cwd!
    # You can over-write if the file exists in CWD!
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, './' + b)


# Main shell function to do all data-set manipulation
def data_shell():
    while True:
        commands = input("data_manipulation> ")
        arg_vector = commands.split()
        if arg_vector[0] == 'exit':
            print("EXITING!")
            break
        else:
            call_functions(arg_vector)


def call_functions(arg_vector):
    if len(arg_vector) <= 1:
        print("Invalid Number of Arguments!")
        return

    command = arg_vector[0]
    file_name = arg_vector[1]
    # Stop any errors here
    if not isfile(file_name) and command != 'merge':
        print("File not found: " + file_name)
        return

    # USE TUPLE RANGES TO INDICATE COLUMNS TO KEEP e. g. (0, 9)
    # ex: keep_columns <data-set> 0 9 21 41
    # Keeps only columns 0 - 9 and 21 - 41
    if command == 'keep_columns':
        cols = []
        for i in range(2, len(arg_vector), 2):
            start = int(arg_vector[i])
            # The +1 is critical to make it inclusive both ends!
            end = int(arg_vector[i + 1]) + 1
            cols.append((start, end))
        drop_columns(file_name, cols)
    elif command == 'split':
        split_csv(file_name)
    elif command == 'merge':
        merge_csv(file_name)
    elif command == 'encode':
        cols = []
        for i in range(2, len(arg_vector)):
            cols.append(int(arg_vector[i]))
        encode_data(file_name, cols)
    elif command == 'hot_encode':
        cols = []
        for i in range(2, len(arg_vector)):
            # Will need a -1 because the class is always the first column
            # So, when fitting, it will be out of scope if it is the last feature to hot encode!
            col = int(arg_vector[i])
            cols.append(col)
        hot_encoder(file_name, cols)
    # This is just in case you do have classes in first column
    elif command == 'shift':
        shift_column(file_name)
    elif command == 'filter':
        filter_duplicate_rows(file_name)
    elif command == 'filter_feature':
        col_number = int(arg_vector[2])
        target_feature = arg_vector[3]
        filter_rows_by_feature(file_name, col_number, target_feature)


def main():
    # Read all commands from a batch file!
    if isfile('./batch.txt'):
        append_header()
        print("Batch file with commands found! Running it now!")
        with open("./batch.txt", "r") as rd:
            for line in rd:
                call_functions(line.split())
        print("Batch file complete!")
    else:
        print("No batch file found, starting shell!")
    data_shell()


# To convert KDD
# 1- First Swap columns
# 2- Encode Columns 2, 3, 4
# 3- Drop Columns that are content related, So keep columns: (0, 9) and (21, 41)
if __name__ == "__main__":
    main()
