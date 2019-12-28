#!/usr/bin/python3
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from os import name, remove, rename
from os.path import basename, dirname, abspath, isfile

# The purpose of this class is mostly to modify data sets
# for machine learning purposes. This includes functions such
# as follows:

# NOTE: When using this script it ASSUMES class is the first column, aka column 0
# NOTE: PLEASE MAKE COPIES OF FILES BEING TAMPERED WITH! IT WILL BE DELETED IF IN SAME WORKING DIRECTORY!
# NOTE: All new files will be written to the current working directory of the shell script!
# 1- Drop Columns (INPUT: COLUMNS TO KEEP)
# 2- Drop Rows (INPUT: ROWS TO KEEP)
# 3- Merge/Split CSVs, this is because Github doesn't permit easy storage of big data sets!
# 4- Seen in NSL-KDD data set, remove ALL duplicate rows from data set!
# 5- Encode Data in specified column using regular label encoding (uses helper function 'filter_duplicate_features'
# 6- Encode Data in specified column using Hot Label Encoding
# 7- Uses Drop Rows, but given a column, if a certain value is found in that column, delete the row!


def n_col(file_name):
    with open(file_name) as read_kdd_data:
        for line in read_kdd_data:
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
def drop_columns(file_name, keep_col_ranges, head=False):
    b = basename(file_name)
    use_cols = []
    for tup in keep_col_ranges:
        use_cols = use_cols + [i for i in range(tup[0], tup[1])]
    df = pd.read_csv(file_name, usecols=use_cols)
    df.to_csv("./prep_" + b, index=False, header=head)
    # Now you have the completed file, create the file in cwd!
    # You can over-write if the file exists in CWD!
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, b)


# Use the method below
def drop_rows(file_name, keep_row_ranges, head=False):
    b = basename(file_name)
    use_rows = []
    for tup in keep_row_ranges:
        use_rows = use_rows + [i for i in range(tup[0], tup[1])]
    df = pd.read_csv(file_name, userows=use_rows)
    df.to_csv("./prep_" + b, index=False, header=head)
    # Now you have the completed file, create the file in cwd!
    # You can over-write if the file exists in CWD!
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, b)


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


# Use this method to ensure ALL rows in the data set are unique
# Write it out to new file
def filter_duplicate_rows(file_name):
    b = basename(file_name)
    s = set()
    with open(file_name, "r") as read:
        for line in read:
            s.add(line)
    with open('./prep' + b, "w") as wr:
        for line in s:
            wr.write(line)
    # Now you have the completed file, create the file in cwd!
    # You can over-write if the file exists in CWD!
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, './' + b)


# Given a file with CSV, use regular Label encoding!
def encode_data(file_name, col_to_encode):
    # Get the file information
    p = dirname(abspath(file_name))
    b = basename(file_name)
    label_map = {}

    # In each column, get the unique columns
    for col in col_to_encode:
        features = filter_duplicate_features(file_name, col)
        lab = LabelEncoder()
        label = lab.fit_transform(features)
        label_map[col] = lab
        if name == 'nt':
            with open(p + "\\labels.txt", "a+") as f:
                f.write("For Column " + str(col) + '\n')
                for k, v in zip(features, label):
                    f.write(k + "," + str(v) + '\n')
                f.write('\n')
        else:
            with open(p + "./labels.txt", "a+") as f:
                f.write("For Column " + str(col) + '\n')
                for k, v in zip(features, label):
                    f.write(k + "," + str(v) + '\n')
                f.write('\n')

    with open(file_name) as read_data, open("./prep_" + b, "w+") as write_data:
        for line in read_data:
            parts = line.rstrip().split(',')

            # Now that I built the encoder, iterate columns
            for col in col_to_encode:
                encoded = label_map[col]
                updated = encoded.transform([parts[col]])
                parts[col] = str(updated[0])

            # Write the result
            new_line = ','.join(parts)
            write_data.write(new_line + '\n')
            write_data.flush()
    # Now you have the completed file, create the file in cwd!
    # You can over-write if the file exists in CWD!
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, './' + b)


def filter_duplicate_features(file_name, col_number):
    s = set()
    with open(file_name, "r") as read:
        for line in read:
            args = line.rstrip().split(",")
            s.add(args[col_number])
    return s


def hot_encoder(file_name, encode_columns, head=False):
    b = basename(file_name)
    data = pd.read_csv(file_name)
    one_hot_encoder = OneHotEncoder(categorical_features=encode_columns)
    data = one_hot_encoder.fit_transform(data).toarray()
    # Now you have the completed file, create the file in cwd!
    data.to_csv('./prep_' + b, index=False, header=head)
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, b)


# A decent chunk of data sets prefer the class to be the last column
# Personally, I prefer it to be the first column. This method will adjust that for me
def shift_column(file_name):
    p = dirname(abspath(file_name))
    b = basename(file_name)
    if name == 'nt':
        output = p + "\\shifted_" + b
    else:
        output = p + "//shifted_" + b

    with open(file_name) as read_kdd_data, open(output, "w+") as write_kdd:
        for line in read_kdd_data:
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


# Main shell function to do all data-set manipulation
def data_shell():
    while True:
        commands = input()
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

    if command == 'drop_columns':
        cols = []
        for i in range(2, len(arg_vector) - 1):
            start = int(arg_vector[i])
            end = int(arg_vector[i + 1])
            cols.append((start, end))
        drop_columns(file_name, cols)
    elif command == 'drop_rows':
        rows = []
        for i in range(2, len(arg_vector) - 1):
            start = int(arg_vector[i])
            end = int(arg_vector[i + 1])
            rows.append((start, end))
        drop_rows(file_name, rows)
    elif command == 'split':
        split_csv(file_name)
    elif command == 'merge':
        merge_csv(file_name)
    elif command == 'filter':
        filter_duplicate_rows(file_name)
    elif command == 'encode':
        cols = []
        for i in range(2, len(arg_vector)):
            cols.append(int(arg_vector[i]))
        encode_data(file_name, cols)
    elif command == 'hot_encode':
        cols = []
        for i in range(2, len(arg_vector)):
            cols.append(int(arg_vector[i]))
        hot_encoder(file_name, cols)
    # This is just in case you do have classes in first column
    elif command == 'shift':
        shift_column(file_name)


def main():
    # Read all commands from a batch file!
    if isfile('./batch.txt'):
        with open("./batch.txt", "r") as rd:
            for line in rd:
                call_functions(line)
    data_shell()


# To convert KDD
# 1- First Swap columns AND Encode it
# 2- Drop Columns that are content related
# 3- Split into parts -- FOR SAVING IT ONLY
# **To use it, just merge it, use raw file name w/o extension!**
if __name__ == "__main__":
    main()
