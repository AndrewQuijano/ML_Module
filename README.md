# ML_Module
I will be using sci-kit learn for various projects such as my IDS project, Indoor Localization Project and potentially others. To avoid confusing myself, I will place the modules here so I can just import into those projects.

With regards to the data_manipulation.py script

NOTE: When using this script it ASSUMES class is the first column, aka column 0. If not, use the shift column method!
NOTE: PLEASE MAKE COPIES OF FILES BEING TAMPERED WITH! IF THE FILE BEING MODIFIED IS IN CURRENT WORKING DIRECTORY, IT WILL BE TAMPERED WITH! This is for easier automation of pre-processing with multiple steps!
NOTE: All new files will be written to the current working directory of the shell script!
1- Drop Columns (INPUT: COLUMNS TO KEEP in tuples!)
2- Drop Rows (INPUT: ROWS TO KEEP in tuples!)
3- Merge/Split CSVs, this is because Github doesn't permit easy storage of big data sets!
4- As seen in NSL-KDD data set, remove ALL duplicate rows from data set!
5- Encode Data in specified column using regular label encoding (uses helper function 'filter_duplicate_features'
6- Encode Data in specified column using Hot Label Encoding
7- Uses Drop Rows, but given a column, if a certain value is found in that column, delete the row!

With regards to main_driver.py, it uses all the sci-kit learn classifiers and generates a text file with scores, Confusion Matricies, Classification Reports, etc.


With regards to incremental learning, it has just using server.py. 
client.py was to test getting training data from a remote source!

TODO:
1- Finish testing Incremental learning to process data from an external client to train an incremental model.
2- Build some other Classifiers from the Weka Library that is not in scikit-learn (e. g. Weka has 