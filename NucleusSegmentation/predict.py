#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 16:48:02 2018

Runs prediction on the test set data and outputs segmentation predictions to CSV. Images are loaded one-by-one and prediction is run (so we don't have to worry about the images being of different size).

@author: ecotner
"""

import tensorflow as tf
import numpy as np
import utils as u
import math
import time
from UNet import UNet
from pathlib import Path
from PIL import Image
from pandas import DataFrame
import matplotlib.pyplot as plt
import re

MODEL_PATH = Path('./models/1/UNet1')
TEST_DATA_PATH = Path('../Datasets/NucleusSegmentation/stage1_test')

def run_length_encoding(M, threshold):
    # Unroll matrix into vector such that pixels are indexed from top to bottom, left to right (Fortran)
    v = M.reshape(-1, order='F')
    v = v > threshold
    s = ''
    e_prev = False
    for idx in range(len(v)):
        e = v[idx]
        if e and (not e_prev):
            s += '{} '.format(idx+1)
            run_len = 1
        elif e and e_prev:
            run_len += 1
        elif (not e) and e_prev:
            s += '{} '.format(run_len)
        e_prev = e
    if e_prev:
        s += '{} '.format(run_len)
    # Chop off trailing whitespace
    s = s[:-1]
    return s

# Load model graph
model = UNet()
saver = model.load_graph(str(MODEL_PATH))

# Get mean and std from log file
p_mean = re.compile('X_mean')
p_std = re.compile('X_std')
p_float = re.compile(r'\d+\.\d+')
with open(str(MODEL_PATH)+'.log', 'r') as fo:
    for line in fo:
        if re.match(p_mean, line) is not None:
            m = re.search(p_float, line)
            X_mean = float(m.group())
        elif re.match(p_std, line) is not None:
            m = re.search(p_float, line)
            X_std = float(m.group())
# Apply normalization

# Add probability tensor to graph
with model.G.as_default():
    prob = tf.sigmoid(model.output)
    
    # Restore session
    with tf.Session() as sess:
        saver.restore(sess, str(MODEL_PATH))
        
        # Iterate through images
        n_folders = len(list(TEST_DATA_PATH.iterdir()))
        submission = np.zeros([n_folders, 2], dtype=np.object)
        for i, folder in enumerate(TEST_DATA_PATH.iterdir()):
            print('Image {}/{}'.format(i+1, n_folders))
            name = str(folder.parts[-1])
            fo = list(folder.glob('images/*.png'))[0]
            x = (np.expand_dims(np.array(Image.open(fo)), axis=0)-X_mean)/X_std
            if x.shape[-1] != 4:
                x = np.concatenate([x, np.zeros([1, x.shape[1], x.shape[2], 4-x.shape[-1]])], axis=-1)
            
            # Run prediction
            feed_dict = {model.input:x}
            Y = sess.run(prob, feed_dict=feed_dict).squeeze()
            if i == 40:
                plt.ioff()
                fig, ax = plt.subplots(1,2)
                [a.cla() for a in ax]
                ax[0].imshow((x.squeeze()-np.min(x))/(np.max(x)-np.min(x)))
                ax[1].imshow(Y>0.5)
                fig.savefig(str(MODEL_PATH)+'_prediction_ex.png')
            
            # Format for submission
            y = run_length_encoding(Y, 0.5)
            submission[i] = (name, y)
            
            # Make DataFrame and write to CSV
            df = DataFrame(submission, columns=['ImageId', 'EncodedPixels'])
            df.to_csv(str(MODEL_PATH)+'_test_predictions.csv', index=False)


        
    























