#!/bin/bash
# Run this script using ./hw5.sh runtype
# Example:  ./hw5.sh rmlr or ./hw5.sh svm or ./hw5.sh (without runtype, it runs feature engineer)

echo "Running feature prediction using method " $1
python main.py $1
