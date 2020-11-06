# ML_Module
I will be using sci-kit learn for various projects such as my IDS project, Indoor Localization Project and potentially others. To avoid confusing myself, I will place the modules here so I can just import into those projects.
Please feel free to use this. If you run the main_driver.py script it will generate the following
- A folder containing confusion matricies
- A classification report
- Cross Validation plots of paramters vs score
- A textfile containing the training/testing score of all models tested
- printing the classifiers into a .joblib format so you can easily load/migrate the completed classifier.

I have provided the handwriting dataset used in my Statiscal Machine Learning class to provide examples how to use this.

## Installation
If you are on a Linux environment run the **install.sh** script to install all dependancies for you.

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

## Usage
**data manipulation**

**main driver**
```python

```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Authors and acknowledgment
Code Author: Andrew Quijano
This was started from my final project from Statistical Machine Learning. 

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Project status
1- Build some other Classifiers from the Weka Library that is not in scikit-learn and provide similar functionality as with Python.
