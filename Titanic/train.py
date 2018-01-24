# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:24:16 2018

Cleans the data and trains the algorithm on an ensemble of SVM

@author: Eric Cotner
"""

import numpy as np
import pandas as pd
from process_data import process_data1, scale_data
from sklearn import svm

# Set up important macros
TRAIN_DATA_PATH = '../Datasets/Titanic/train.csv'
TEST_DATA_PATH = '../Datasets/Titanic/test.csv'
VAL_FRACTION = 0.25

# Import the data into a DataFrame
data = pd.read_csv(TRAIN_DATA_PATH)

# Clean the data
X_train, Y_train = process_data1(data)

# Scale the data
X_train, mean, std = scale_data(X_train, sparse_feature_idxs=[1,3,4])

# Split into training and validation
m_val = int(VAL_FRACTION * X_train.shape[0])
X_val = X_train[:m_val,:]
Y_val = Y_train[:m_val]
X_train = X_train[m_val:,:]
Y_train = Y_train[m_val:]

# Save mean and variance to file
with open('./Titanic.log', 'w+') as fo:
    fo.write('Mean: {}\n'.format(mean))
    fo.write('Std: {}\n'.format(std))

# Run training
model = {}
model[0] = svm.SVC(kernel='rbf', C=.4, gamma=.16, probability=False)
model[1] = svm.SVC(kernel='linear', C=1, gamma='auto', probability=False)
model[2] = svm.SVC(kernel='poly', degree=3, C=1, gamma=.1, probability=False)
model[3] = svm.SVC(kernel='poly', degree=4, C=13, gamma=.06, probability=False)
model[4] = svm.SVC(kernel='sigmoid', C=3, gamma=.01, coef0=0, probability=False)
for key in model:
    model[key].fit(X_train, Y_train)

# Evaluate prediction
for key in model:
    train_accuracy = np.mean(np.equal(model[key].predict(X_train), Y_train))
    val_accuracy = np.mean(np.equal(model[key].predict(X_val), Y_val))
    print('Train accuracy: {}, validation accuracy: {}'.format(train_accuracy, val_accuracy))

# Do some ensembling
predictions = np.stack([model[i].predict(X_val) for i in model]).T
ens_predictions = np.mean(predictions, axis=1) >= 0.5
ens_accuracy = np.mean(np.equal(ens_predictions, Y_val))
print('Ensemble accuracy: {}'.format(ens_accuracy))

# Load test set
data = pd.read_csv(TEST_DATA_PATH)

# Prepare test set
X_test, PassengerId = process_data1(data, test=True)
X_test = (X_test - np.expand_dims(mean, axis=0)) / np.expand_dims(std, axis=0)
predictions = np.stack([model[i].predict(X_test) for i in range(5)]).T
ens_predictions = (np.mean(predictions, axis=1) >= 0.5).astype(int)
#print(ens_predictions)
output_df = pd.DataFrame(np.stack([PassengerId, ens_predictions]).T, columns=['PassengerId','Survived'], dtype=int)
output_df.to_csv('./TitanicPredictions.csv', index=False)





