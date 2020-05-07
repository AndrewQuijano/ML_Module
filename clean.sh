#!/bin/bash

# Clean main driver results
rm -rf ./Classifiers
rm -rf ./Confusion_Matrix
rm -rf ./Cross_Validation
rm -rf ./ROC
rm -rf ./Machine_Learning/__pycache__/
rm classification_reports.txt
rm results.txt
rm train_*
rm test_*

# Clean incremental ML results
