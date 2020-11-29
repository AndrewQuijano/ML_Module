#!/bin/bash

# Install basics
sudo apt-get install vim

# Install Python and packages
sudo apt-get install python3-pip

# Install Python and packages used by IDS and Fuzzer
sudo -H pip3 install sklearn
sudo -H pip3 install pandas
sudo -H pip3 install joblib
sudo -H pip3 install scikit-plot

# For the script to work as intended, you need the correct version of matplotlib
# In due time, I will try to find out the highest version this would still work!
sudo -H pip3 install matplotlib
