#!/usr/bin/python3
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from os import name
from os.path import basename, dirname, abspath

# The purpose of this class is mostly to modify data sets
# for machine learning purposes. This includes functions such
# as follows:

# NOTE: When using this script it ASSUMES class is the first column, aka column 0
# NOTE: All new files will have pre-pended the operation completed on it!
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
    p = dirname(abspath(file_name))
    b = basename(file_name)
    use_cols = []
    for tup in keep_col_ranges:
        use_cols = use_cols + [i for i in range(tup[0], tup[1])]
    df = pd.read_csv(file_name, usecols=use_cols)
    # Detect if Windows
    if name == 'nt':
        df.to_csv(p + "\\col_" + b, index=False, header=head)
    else:
        df.to_csv(p + "/col_" + b, index=False, header=head)


# Use the method below
def drop_rows(file_name, keep_row_ranges, head=False):
    p = dirname(abspath(file_name))
    b = basename(file_name)
    use_rows = []
    for tup in keep_row_ranges:
        use_rows = use_rows + [i for i in range(tup[0], tup[1])]
    df = pd.read_csv(file_name, userows=use_rows)
    # Detect if Windows
    if name == 'nt':
        df.to_csv(p + "\\row_" + b, index=False, header=head)
    else:
        df.to_csv(p + "/row_" + b, index=False, header=head)


# Remember Github's Limit is 100 MB
# So to store large data sets online, I just split them up by 500,000 row chunks
def split_csv(file_name, size=500000):
    b = basename(file_name)
    p = dirname(abspath(file_name))
    file_part = 1
    idx = 1
    lines = []
    with open(file_name, 'r') as big_file:
        for line in big_file:
            if file_part % size == 0:
                # Write to Windows
                if name == 'nt':
                    with open(p + "\\" + b + "_part_" + str(idx), 'w+') as chunk:
                        chunk.writelines(lines)
                # Write to Linux
                else:
                    with open(p + "\\" + b + "_part" + str(idx), 'w+') as chunk:
                        chunk.writelines(lines)
                lines = []
                idx += 1
            else:
                lines.append(line)
            file_part += 1


# This is the counter-part function
def merge_csv(file_name, n_parts=9):
    b = basename(file_name)
    p = dirname(abspath(file_name))
    # Remove extension!
    b_part = str(b.split(".")[0])
    file_part = 1
    # if n_parts is 9
    # j goes from 0 - 8
    # Remember files parts is goes 1 - 9!

    for j in range(n_parts):
        if name == 'nt':
            with open(p + "\\" + b_part + "_part" + str(j + 1) + ".csv", 'r') as chunk:
                with open(file_name, 'a+') as big_file:
                    for line in chunk:
                        big_file.write(line)
        else:
            with open(p + "/" + b_part + "_part" + str(j + 1) + ".csv", 'r') as chunk:
                with open(file_name, 'a+') as big_file:
                    for line in chunk:
                        big_file.write(line)
        file_part += 1


# Use this method to ensure ALL rows in the data set are unique
# Write it out to new file
def filter_duplicate_rows(file_name):
    s = set()
    with open(file_name, "r") as read:
        for line in read:
            s.add(line)
    with open(file_name, "w") as wr:
        for line in s:
            wr.write(line)


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

    if name == 'nt':
        output_file = p + "\\prep_" + b
    else:
        output_file = p + "/prep_" + b

    with open(file_name) as read_kdd_data, open(output_file, "w+") as write_kdd:
        for line in read_kdd_data:
            parts = line.rstrip().split(',')

            # Now that I built the encoder, iterate columns
            for col in col_to_encode:
                encoded = label_map[col]
                updated = encoded.transform([parts[col]])
                parts[col] = str(updated[0])

            # Write the result
            new_line = ','.join(parts)
            write_kdd.write(new_line + '\n')
            write_kdd.flush()


def filter_duplicate_features(file_name, col_number):
    s = set()
    with open(file_name, "r") as read:
        for line in read:
            args = line.rstrip().split(",")
            s.add(args[col_number])
    return list(s)


def hot_encoder(train_x):
    oh = OneHotEncoder()
    oh.fit_transform(train_x)


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
    command = arg_vector[0]
    file_name = arg_vector[1]
    if command == 'drop_columns':
        cols = []
        drop_columns(file_name, [(0, 9), (21, 41)])
    elif command == 'drop_rows':
        drop_rows(file_name, [0])
    elif command == 'split':
        split_csv(file_name)
    elif command == 'merge':
        merge_csv(file_name)
    elif command == 'filter':
        filter_duplicate_rows(file_name)
    elif command == 'encode':
        cols = []
        cols.add(int(arg_vector[2]))
        encode_data(file_name, cols)
    elif command == 'hot_encode':
        cols = []
        cols.add(int(arg_vector[2]))
        hot_encoder(file_name)
    # This is just in case you do have classes in first column
    elif command == 'shift':
        shift_column(file_name)


def main():
    # Read all commands from a batch file!
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
