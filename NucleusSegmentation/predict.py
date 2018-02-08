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

MODEL_PATH = Path('./models/2/UNet2')
TEST_DATA_PATH = Path('../Datasets/NucleusSegmentation/stage1_test')

def gen_instance_maps(Y, class_threshold=0.5, boundary_threshold=0.1):
    """ Generates the masks for each instance of a nuclei from the segmentation map by assigning a unique nonzero integer to every positively identified pixel, then taking the max over adjacent pixels until convergence. SHOULD PROBABLY WRITE THIS IN C - IT'S PRETTY TIME-CONSUMING."""
    # Assign unique nonzero integer to each pixel over threshold
    y = (Y[:,:,0] > class_threshold).astype(int)
    b = (Y[:,:,1] > boundary_threshold).astype(int)
    height, width = y.shape
    for row in range(height):
        y[row,:] *= 1+np.arange(row*width,(row+1)*width, dtype=int)
    y_ = np.zeros((height, width), dtype=int)
    # Repeat until convergence
    while np.any(np.not_equal(y, y_)):
        y_ = np.copy(y)
        # print(np.sum(y))
        # print(np.sum(y_))
        # Scan over elements of array
        for h in range(height):
            if np.all(np.equal(y[h,:], 0)):
                pass
            else:
                for w in range(width):
                    # Assign max value of neighbors to current element, unless that neighbor is boundary pixel
                    if y[h,w] != 0:
                        y[h,w] = np.max(y[max(h-1,0):min(h+2,height), max(w-1,0):min(w+2,width)] * (1-b[max(h-1,0):min(h+2,height), max(w-1,0):min(w+2,width)]))
                        # print(np.sum(y))
        # print(np.sum(y))
    # Split clusters of unique integers into separate segmentation masks
    # Get list of unique integers, except zero
    int_list = []
    for h in range(height):
        for w in range(width):
            if (y[h,w] not in int_list) and (y[h,w] != 0):
                int_list.append(y[h,w])
    # Iterate over unique integers
    mask_list = []
    for n in int_list:
        # Construct segmentation mask and add to list of masks
        mask = np.equal(y, n).astype(int)
        if np.sum(mask) > 5:
            mask_list.append(mask)
    # Return list of masks
    print("Found {} nuclei!".format(len(mask_list)))
    return mask_list

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

tic = time.time()

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
        submission = []
        for i, folder in enumerate(TEST_DATA_PATH.iterdir()):
            print('Image {}/{}: {}'.format(i+1, n_folders, str(folder.parts[-1])))
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
                ax[1].imshow(Y[:,:,0]>0.5)
                ax[1].imshow(Y[:,:,1]>0.1, alpha=0.35, cmap='Greys')
                fig.savefig(str(MODEL_PATH)+'_prediction_ex.png')
            
            # Format for submission
            instance_maps = gen_instance_maps(Y, class_threshold=0.5, boundary_threshold=0.1)
            for mask in instance_maps:
                y = run_length_encoding(mask, threshold=0.5)
                submission.append((name, y))
            
            # Make DataFrame and write to CSV
            df = DataFrame(submission, columns=['ImageId', 'EncodedPixels'])
            df.to_csv(str(MODEL_PATH)+'_test_predictions.csv', index=False)

toc = time.time()
print("Total time: {:.2f} minutes".format((toc-tic)/60))