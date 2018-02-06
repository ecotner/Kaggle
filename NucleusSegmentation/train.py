#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:23:32 2018

@author: ecotner
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from UNet import UNet
import utils as u

MODEL_PATH = Path('./models/1/UNet1')
TRAIN_DATA_PATH = Path('../Datasets/NucleusSegmentation/stage1_train')
#K_FOLDS = 5
VAL_BATCH_SIZE = 16
SEED = 0

# Construct computational graph
#models = {k:UNet() for k in range(K_FOLDS)}
#for k in models:
print('Constructing graphs...')
model = UNet() # models[k]
model.convolution(f=3, s=1, n_out=32, activation='relu')
model.dropout(1)
model.convolution(f=3, s=1, n_out=32, activation='relu')
model.dropout(1)
model.squeeze_convolution(f=2, s=2, n_out=32, activation='relu')
model.convolution(f=3, s=1, n_out=64, activation='relu')
model.dropout(1)
model.convolution(f=3, s=1, n_out=64, activation='relu')
model.dropout(1)
model.squeeze_convolution(f=2, s=2, n_out=64, activation='relu')
model.convolution(f=3, s=1, n_out=128, activation='relu')
model.dropout(1)
model.convolution(f=3, s=1, n_out=128, activation='relu')
model.dropout(1)
model.squeeze_convolution(f=2, s=2, n_out=128, activation='relu')
model.convolution(f=3, s=1, n_out=256, activation='relu')
model.dropout(2)
model.convolution(f=3, s=1, n_out=256, activation='relu')
model.dropout(2)
model.squeeze_convolution(f=2, s=2, n_out=256, activation='relu')
model.convolution(f=3, s=1, n_out=512, activation='relu')
model.dropout(2)
model.convolution(f=3, s=1, n_out=512, activation='relu')
model.dropout(2)
model.convolution(f=3, s=1, n_out=512, activation='relu')
model.stretch_transpose_convolution(f=2, s=2, n_out=256, activation='relu')
model.convolution(f=3, s=1, n_out=256, activation='relu')
model.convolution(f=3, s=1, n_out=256, activation='relu')
model.stretch_transpose_convolution(f=2, s=2, n_out=128, activation='relu')
model.convolution(f=3, s=1, n_out=128, activation='relu')
model.convolution(f=3, s=1, n_out=128, activation='relu')
model.stretch_transpose_convolution(f=2, s=2, n_out=64, activation='relu')
model.convolution(f=3, s=1, n_out=64, activation='relu')
model.convolution(f=3, s=1, n_out=64, activation='relu')
model.stretch_transpose_convolution(f=2, s=2, n_out=32, activation='relu')
model.convolution(f=3, s=1, n_out=32, activation='relu')
model.convolution(f=3, s=1, n_out=32, activation='relu')
model.convolution(f=1, s=1, n_out=1, activation='identity')
model.add_loss(loss_type='xentropy', reg_type='L2')
model.add_optimizer(opt_type='adam')
model.save_graph(MODEL_PATH)

# Load the data, shuffle, split into train/validation sets
print('Loading data...')
X_train = np.load(TRAIN_DATA_PATH/'X_train.npy')
Y_train = np.load(TRAIN_DATA_PATH/'Y_train.npy')

np.random.seed(SEED)
X_train = np.random.permutation(X_train)
np.random.seed(SEED)
Y_train = np.random.permutation(Y_train)

m = X_train.shape[0]
m_val = VAL_BATCH_SIZE
X_val = X_train[:m_val]
Y_val = Y_train[:m_val]
X_train = X_train[m_val:]
Y_train = Y_train[m_val:]
print('Beginning training...')
model.train(X_train, Y_train, X_val, Y_val, max_epochs=int(1e9), batch_size=16, learning_rate_init=2e-4, reg_param=0, learning_rate_decay_type='constant', learning_rate_decay_parameter=1, keep_prob=[0.9, 0.9], early_stopping=True, save_path=MODEL_PATH, reset_parameters=True, val_checks_per_epoch=10, seed=SEED, data_on_GPU=False)
    