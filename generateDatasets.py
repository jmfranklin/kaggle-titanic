#!/usr/bin/python3

import pandas as pd

# Matplotlib and seaborn used for correlation heatmap
from matplotlib import pyplot as plt
import seaborn as sns


# This imports the prepareData.py file from our "bin" folder
from bin import prepareData
from bin.helperFunctions import storePickledData

# print out more information
print_debug = 1

train_file_in   = 'data/train.csv'
test_file_in    = 'data/test.csv'

pickled_datasets_file = 'data/pickled_datasets.bin'

pd.set_option('display.max_colwidth', 30)
pd.set_option('expand_frame_repr', False)

train = pd.read_csv(train_file_in)
test  = pd.read_csv(test_file_in)
dataSets = [train, test]

# look at where the missing values are whilst data is still raw
if print_debug:
    #print ('\n',train['Fare'].value_counts())
    print('\nMissing values before imputation:\n', train.isnull().sum())
    print('\nMissing values before imputation:\n', train.isnull().sum())


# prepareData contains all the functions for adding features to the datasets
dataSets = prepareData.prepare(dataSets)

if print_debug:
    print("\nmissing value counts after imputation:\n", train.isnull().sum())
    print('\n', train.describe())
    print("\n", train[['Gender', 'Survived']].groupby('Gender', as_index=False).mean())
    print("\n", train[['AgeGroup', 'Survived']].groupby('AgeGroup', as_index=False).mean())
    print("\n", train[['Deck', 'Survived']].groupby('Deck', as_index=False).mean())
    print("\n", train[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean())
    print("\n", train[['Port', 'Survived']].groupby('Port', as_index=False).mean())
    print("\n", train[['Title', 'Survived']].groupby('Title', as_index=False).mean())
    print("\n", train[['Class', 'Survived']].groupby('Class', as_index=False).mean())


## drop unneeded features
if print_debug:
    print('Before trimming unneeded features:\n', train.head(1))

for feature in ['Pclass', 'Name', 'Sex', 'Age', 'Ticket', 'Fare', 'Cabin', 'Embarked']:
        train = train.drop(feature, axis=1)
        test  = test.drop(feature, axis=1)

# needed in test DS to generate the submission file.
train = train.drop('PassengerId', axis=1)

if print_debug:
    print('\nAfter trimming unneeded features:\n', train.head(1))
    # trying a correlation heatmap for data to see which variables most closely relate to each other
    corr = train.corr()
    sns.heatmap(corr,
                cmap="RdBu", center=0,vmin=-1, vmax=1)
    plt.show()

# prepare data sets for the model(s)
x_train = train.drop('Survived', axis=1)
y_train = train['Survived']
x_test  = test.drop('PassengerId', axis=1).copy()

if print_debug:
    print("\n\ndataSets shapes:", x_train.shape, y_train.shape, x_test.shape)
    print("x_train.head()\n", x_train.head())
    print("x_test.head()\n", x_test.head())


pickled_datasets = (x_train, y_train, x_test)

if storePickledData(pickled_datasets, pickled_datasets_file):
    print('Datasets pickled into {}'.format(pickled_datasets_file))
else:
    print('Failed pickling Datasets to {}'.format(pickled_datasets_file))
