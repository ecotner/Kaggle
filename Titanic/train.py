# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:24:16 2018

Cleans the data and trains the algorithm on an ensemble of SVM

@author: Eric Cotner
"""

import numpy as np
import pandas as pd
import utils as u
import time
from sklearn import svm
from sklearn.neural_network import MLPClassifier as MLP
#from mlp import MLP

# Load the pre-processed data
X = np.load('./X.npy')
Y = np.load('./Y.npy')
m_train = 891
print('X.shape = {}'.format(X.shape))
# Split into train and test sets
X_train, Y_train = X[:m_train], Y[:m_train]
seed = int(time.time())
np.random.seed(seed)
X_train = np.random.permutation(X_train)
np.random.seed(seed)
Y_train = np.random.permutation(Y_train)
X_test = X[m_train:]
PassengerId = np.arange(m_train+1, X.shape[0]+1)

def SVM_hyperparameter_search(C_max, C_min, gamma_max, gamma_min, kernel='rbf', degree=2, K_FOLDS=10, search_time=300):
    acc_best = 0
    tic = time.time()
    toc = time.time()
    while toc-tic < search_time:
        C = np.power(10, np.log10(C_max/C_min)*np.random.rand() + np.log10(C_min))
        gamma = np.power(10, np.log10(gamma_max/gamma_min)*np.random.rand() + np.log10(gamma_min))
        model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree) # <<< This is the model being searched on
        model_train_acc_list = []
        model_val_acc_list = []
        for k in range(K_FOLDS):
            # Split into train and validation set
            m_val = m_train//K_FOLDS + 1
            val_idx_start = k*m_val
            val_idx_end = min((k+1)*m_val, m_train)
            X_val = X_train[val_idx_start:val_idx_end].copy()
            Y_val = Y_train[val_idx_start:val_idx_end].copy()
            X_train_ = np.concatenate([X_train[:val_idx_start], X_train[val_idx_end:]], axis=0)
            Y_train_ = np.concatenate([Y_train[:val_idx_start], Y_train[val_idx_end:]], axis=0)
            # Train on single fold
            model.fit(X_train_, Y_train_)
            train_accuracy = np.mean(np.equal(model.predict(X_train_)>0.5, Y_train_))
            val_accuracy = np.mean(np.equal(model.predict(X_val)>0.5, Y_val))
            model_train_acc_list.append(train_accuracy)
            model_val_acc_list.append(val_accuracy)
        # Calculate average validation accuracy and record parameters if best
        mean_val_accuracy = np.mean(model_val_acc_list)
        if mean_val_accuracy > acc_best:
            C_best = C
            gamma_best = gamma
            acc_best = mean_val_accuracy
            print('Improved parameters: C={}, gamma={}, val. acc.={}'.format(C, gamma, acc_best))
        toc = time.time()
    with open('./SVC_hyperparameter tuning.log', 'a+') as fo:
        if kernel == 'poly':
            fo.write('SVC kernel {}, degree {}: C_best={}, gamma_best={}, acc_best={}\n'.format(kernel, degree, C_best, gamma_best, acc_best))
        else:
            fo.write('SVC kernel {}: C_best={}, gamma_best={}, acc_best={}\n'.format(kernel, C_best, gamma_best, acc_best))
    print('Hyperparameter training complete')

def MLP_hyperparameter_search(hidden_layer_sizes, learning_rate_max, learning_rate_min, alpha_max, alpha_min, optimizer='adam', K_FOLDS=10, search_time=300):
    acc_best = 0
    tic = time.time()
    toc = time.time()
    while toc-tic < search_time:
        learning_rate = np.power(10, np.log10(learning_rate_max/learning_rate_min)*np.random.rand() + np.log10(learning_rate_min))
        alpha = np.power(10, np.log10(alpha_max/alpha_min)*np.random.rand() + np.log10(alpha_min))
        model = MLP(hidden_layer_sizes, learning_rate_init=learning_rate, alpha=alpha, solver=optimizer) # <<< This is the model being searched on
        model_train_acc_list = []
        model_val_acc_list = []
        for k in range(K_FOLDS):
            # Split into train and validation set
            m_val = m_train//K_FOLDS + 1
            val_idx_start = k*m_val
            val_idx_end = min((k+1)*m_val, m_train)
            X_val = X_train[val_idx_start:val_idx_end].copy()
            Y_val = Y_train[val_idx_start:val_idx_end].copy()
            X_train_ = np.concatenate([X_train[:val_idx_start], X_train[val_idx_end:]], axis=0)
            Y_train_ = np.concatenate([Y_train[:val_idx_start], Y_train[val_idx_end:]], axis=0)
            # Train on single fold
            model.fit(X_train_, Y_train_)
            train_accuracy = np.mean(np.equal(model.predict(X_train_)>0.5, Y_train_))
            val_accuracy = np.mean(np.equal(model.predict(X_val)>0.5, Y_val))
            model_train_acc_list.append(train_accuracy)
            model_val_acc_list.append(val_accuracy)
        # Calculate average validation accuracy and record parameters if best
        mean_val_accuracy = np.mean(model_val_acc_list)
        if mean_val_accuracy > acc_best:
            learning_rate_best = learning_rate
            alpha_best = alpha
            acc_best = mean_val_accuracy
            print('Improved parameters: learning_rate={}, alpha={}, val. acc.={}'.format(learning_rate, alpha, acc_best))
        toc = time.time()
    with open('./SVC_hyperparameter tuning.log', 'a+') as fo:
            fo.write('MLP {} solver {}: learning_rate_best={}, alpha_best={}, acc_best={}\n'.format(hidden_layer_sizes, optimizer, learning_rate_best, alpha_best, acc_best))
    print('Hyperparameter training complete')

# Build models using k-fold cross-validation
def validate_models(X_train, Y_train, save_test_predictions=False):
    model = {}
    N_MODELS = 6
    K_FOLDS = 10
    for k in range(K_FOLDS):
        model[N_MODELS*k+0] = svm.SVC(kernel='rbf', C=1.065056820926003, gamma=0.0009649926839202406)
        model[N_MODELS*k+1] = svm.SVC(kernel='linear', C=0.002166166639470253, gamma=0.0003881519436299208)
        model[N_MODELS*k+2] = svm.SVC(kernel='poly', degree=2, C=0.6034908850231115, gamma=3.4225826148291048)
        model[N_MODELS*k+3] = svm.SVC(kernel='poly', degree=3, C=0.07600817839752295, gamma=0.4394395870011307)
#        model[N_MODELS*k+4] = svm.SVC(kernel='sigmoid', C=3e4, gamma=.6e-5, coef0=0) # C=3e4, gamma=.6e-5 -> .840
        model[N_MODELS*k+4] = MLP(solver='lbfgs', learning_rate='constant', learning_rate_init=0.8612728621291262, alpha=4.107458336192167, hidden_layer_sizes=[100], activation='relu')
        model[N_MODELS*k+5] = MLP(solver='lbfgs', learning_rate='constant', learning_rate_init=0.00016724330316360546, alpha=8.89303935500149, hidden_layer_sizes=[20, 20], activation='relu')
    
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

    if save_test_predictions:
        # Rerun training on the ENTIRE training set w/o changing parameters to get as much out of it as possible
        for idx in range(N_MODELS):
            model[idx].fit(X_train, Y_train)
        
        # Run predictions on the test set
        predictions = np.stack([model[idx].predict(X_test).squeeze() for idx in range(N_MODELS)]).T
        ens_predictions = (np.mean(predictions, axis=1) >= 0.5).astype(int)
        
        # Output predictions to CSV
        output_df = pd.DataFrame(np.stack([PassengerId, ens_predictions]).T, columns=['PassengerId','Survived'], dtype=int)
        output_df.to_csv('./TitanicPredictions.csv', index=False)

#SVM_hyperparameter_search(C_max=1e-1, C_min=1e-3, gamma_max=1e0, gamma_min=1e-1, kernel='poly', degree=3, search_time=5*60)

#MLP_hyperparameter_search(hidden_layer_sizes=[20,20], learning_rate_max=1e-1, learning_rate_min=1e-5, alpha_max=1e1, alpha_min=1e-1, optimizer='lbfgs', K_FOLDS=10, search_time=5*60)

validate_models(X_train, Y_train, True)