#!/usr/bin/python3
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from os import remove, rename
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
# 8- Shift Column (This is for moving the class column from last column to first column)
# 9- Drop row if at column X the feature is Y


# This is because the mini hand-writing data-set doesn't have the label prepended!
def append_column(file_name='./train_3.txt', label='3'):
    b = basename(file_name)
    with open(file_name) as read_data, open('./prep_' + b, 'w+') as write_data:
        for line in read_data:
            write_data.write(label + ',' + line)

    # Now you have the completed file, create the file in cwd!
    # You can over-write if the file exists in CWD!
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, b)
    print("Finished appending the column!")


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


# Use the method below to keep all rows you want to keep!
def drop_rows(file_name, keep_row_ranges):
    b = basename(file_name)
    use_rows = []
    for tup in keep_row_ranges:
        use_rows = use_rows + [i for i in range(tup[0], tup[1])]
    # GET ALL POSSIBLE ROWS THEN GET KEEP ROWS
    all_rows = [i for i in range(0, n_row(file_name) - 1)]
    to_drop_rows = list(set(all_rows) - set(keep_row_ranges))
    df = pd.read_csv(file_name)
    df.drop(df.index[to_drop_rows])
    df.to_csv("./prep_" + b, index=False)
    # Now you have the completed file, create the file in cwd!
    # You can over-write if the file exists in CWD!
    if isfile('./' + b):
        remove('./' + b)
    rename('./prep_' + b, b)
    print("Row Dropping complete, only kept columns: " + str(use_rows))


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
    print("Finished Data encoding!")


# It needs to be returned as list to be used by label encoder!
def filter_duplicate_features(file_name, col_number):
    s = set()
    with open(file_name, "r") as read:
        for line in read:
            args = line.rstrip().split(",")
            s.add(args[col_number])
    return list(s)


def hot_encoder(file_name, encode_columns):
    b = basename(file_name)
    data = pd.read_csv(file_name)
    one_hot_encoder = OneHotEncoder(categorical_features=encode_columns)
    # You do need to use Label encoder!
    # encode labels with value between 0 and n_classes - 1.
    le = LabelEncoder()
    new_data = data.apply(le.fit_transform)
    # Now use the hot encoder!
    data = one_hot_encoder.fit_transform(new_data).toarray()
    # Now you have the completed file, create the file in cwd!
    data.to_csv('./prep_' + b, index=False)
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
    print("Dropping row by feature complete!")


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
            end = int(arg_vector[i + 1]) + 1
            cols.append((start, end))
        drop_columns(file_name, cols)
    # USE TUPLE RANGES TO INDICATE COLUMNS TO KEEP e. g. (0, 10)
    # ex: keep_columns <data-set> 0 10
    # keep rows 0 - 10 ONLY
    elif command == 'keep_rows':
        rows = []
        for i in range(2, len(arg_vector), 2):
            start = int(arg_vector[i])
            end = int(arg_vector[i + 1]) + 1
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
