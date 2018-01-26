# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:24:16 2018

Cleans the data and trains the algorithm on an ensemble of SVM

@author: Eric Cotner
"""

import numpy as np
import pandas as pd
import utils as u
from sklearn import svm
from sklearn.neural_network import MLPClassifier as MLP
#from mlp import MLP

# Set up important macros
TRAIN_DATA_PATH = '../Datasets/Titanic/train.csv'
TEST_DATA_PATH = '../Datasets/Titanic/test.csv'
#VAL_FRACTION = 0.25

# Import the data into a DataFrame
train_data = pd.read_csv(TRAIN_DATA_PATH)
m_train = train_data.shape[0]
test_data = pd.read_csv(TEST_DATA_PATH)
all_data = pd.concat([train_data, test_data])

# Clean the data, shuffle it randomly, scale it, and split into train/test
X, Y = u.process_data2(all_data)
X, mean, std = u.scale_data(X)
X_train, Y_train = X[:m_train], Y[:m_train]
np.random.seed(1)
X_train = np.random.permutation(X_train)
np.random.seed(1)
Y_train = np.random.permutation(Y_train)
X_test = X[m_train:]
PassengerId = np.arange(m_train+1, all_data.shape[0]+1)
#print(X.shape, Y.shape, X_train.shape, Y_train.shape, X_test.shape, PassengerId.shape)

# Split into training and validation
#m_val = int(VAL_FRACTION * X_train.shape[0])
#X_val = X_train[:m_val,:]
#Y_val = Y_train[:m_val]
#X_train = X_train[m_val:,:]
#Y_train = Y_train[m_val:]

# Save mean and variance to file
with open('./Titanic.log', 'w+') as fo:
    fo.write('Mean: {}\n'.format(mean))
    fo.write('Std: {}\n'.format(std))

# Build models using k-fold cross-validation
model = {}
N_MODELS = 2
K_FOLDS = 4
for k in range(K_FOLDS):
    model[N_MODELS*k+0] = svm.SVC(kernel='rbf', C=1e2, gamma=9e-4, probability=False)
    model[N_MODELS*k+1] = svm.SVC(kernel='linear', C=9e-2, gamma=.001, probability=False)
#    model[N_MODELS*k+2] = svm.SVC(kernel='poly', degree=2, C=1e3, gamma=.001, probability=False)
#    model[N_MODELS*k+3] = svm.SVC(kernel='poly', degree=3, C=3e5, gamma=.0001, probability=False)
#    model[N_MODELS*k+4] = svm.SVC(kernel='sigmoid', C=2e0, gamma=.001, coef0=0, probability=False)
#    model[N_MODELS*k+5] = MLP(solver='adam', learning_rate='constant', learning_rate_init=3e-3, alpha=1.5e1, hidden_layer_sizes=(50,20), activation='relu')

# Train and validate models
model_train_acc_lists = {i:[] for i in range(N_MODELS)}
model_val_acc_lists = {i:[] for i in range(N_MODELS)}
for k in range(K_FOLDS):
    # Split into train and validation set
    m_val = m_train//K_FOLDS + 1
    val_idx_start = k*m_val
    val_idx_end = min((k+1)*m_val, m_train)
    X_val = X_train[val_idx_start:val_idx_end].copy()
    Y_val = Y_train[val_idx_start:val_idx_end].copy()
    X_train_ = np.concatenate([X_train[:val_idx_start], X_train[val_idx_end:]], axis=0)
    Y_train_ = np.concatenate([Y_train[:val_idx_start], Y_train[val_idx_end:]], axis=0)
#    print('Dimensions:\nX_train: {}, X_val: {}, X_train_: {}'.format(X_train.shape, X_val.shape, X_train_.shape))
    for idx in range(N_MODELS):
        model[N_MODELS*k+idx].fit(X_train_, Y_train_)
        train_accuracy = np.mean(np.equal(model[N_MODELS*k+idx].predict(X_train_)>0.5, Y_train_))
        val_accuracy = np.mean(np.equal(model[N_MODELS*k+idx].predict(X_val)>0.5, Y_val))
        model_train_acc_lists[idx].append(train_accuracy)
        model_val_acc_lists[idx].append(val_accuracy)
        print('Model {}, fold {}: train accuracy: {:.3f}, validation accuracy: {:.3f}'.format(idx, k, train_accuracy, val_accuracy))
for idx in range(N_MODELS):
    print('Model {}: train accuracy: {:.3f}±{:.3f}, validation accuracy: {:.3f}±{:.3f}'.format(idx, np.mean(model_train_acc_lists[idx]), np.std(model_train_acc_lists[idx]), np.mean(model_val_acc_lists[idx]), np.std(model_val_acc_lists[idx])))

## Do some ensembling
#predictions = np.stack([model[i].predict(X_train).squeeze() for i in model]).T
#ens_predictions = np.mean(predictions, axis=1) >= 0.5
#ens_accuracy = np.mean(np.equal(ens_predictions, Y_train))
#print('Ensemble accuracy: {}'.format(ens_accuracy))

# Rerun training on the ENTIRE training set w/o changing parameters to get as much out of it as possible
for idx in model:
    model[idx].fit(X_train, Y_train)

# Run predictions on the test set
predictions = np.stack([model[i].predict(X_test).squeeze() for i in model]).T
ens_predictions = (np.mean(predictions, axis=1) >= 0.5).astype(int)

# Output predictions to CSV
output_df = pd.DataFrame(np.stack([PassengerId, ens_predictions]).T, columns=['PassengerId','Survived'], dtype=int)
output_df.to_csv('./TitanicPredictions.csv', index=False)





