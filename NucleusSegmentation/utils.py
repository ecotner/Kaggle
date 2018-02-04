# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:49:03 2018

Utility functions

@author: Eric Cotner
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import math

def stack_masks(save_path):
    ''' Stacks the nucleus image masks on top of each other into a single mask, then saves it to a single image. '''
    dir_len = len(list(save_path.iterdir()))
    for i, x in enumerate(save_path.iterdir()):
        masks_glob = x.glob('masks/*.png')
        stacked_img = None
        print('Image {}/{}'.format(i, dir_len))
        # Stack images on each other (by simply adding)
        for fo in masks_glob:
            img = np.array(Image.open(fo))
            if stacked_img is None:
                stacked_img = img
            else:
                stacked_img += img
        # Save as both .npy file and .png
        np.save(x/'stacked_img.npy', stacked_img)
        stacked_img = Image.fromarray(stacked_img)
        stacked_img.save(x/'stacked_img.png')

def upscale_and_crop(img):
    ''' Dimensions of largest image are 1040x1388, so we'll upscale to just beyond that size (1200x1400) and then crop. '''
    max_height = 1200
    max_width = 1400
    height, width = img.shape[:2]
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    n_channels = img.shape[-1]
    n_height_refl = max_height//height
    n_width_refl = max_width//width
    # Make blank image to fit all reflections
    new_img = np.zeros([(n_height_refl+1)*height, (n_width_refl+1)*width, n_channels])
    # Iterate over the reflections (fortunately reflections around edges are commutative)
    for nx in range(1+n_width_refl):
        for ny in range(1+n_height_refl):
            refl_img = img.copy()
            if nx % 2 != 0:
                refl_img = np.flip(refl_img, axis=1)
            if ny % 2 != 0:
                refl_img =  np.flip(refl_img, axis=0)
            new_img[ny*height:(ny+1)*height, nx*width:(nx+1)*width] = refl_img
    # Crop to desired size
    new_img = new_img[:max_height,:max_width,:]
    del refl_img, img
    return new_img

def preprocess_data(save_path):
    ''' Loads the data, preprocesses it (by reflecting smaller images around their edges and then cropping to uniform size), then stacks them all into a single pair of (X,Y) arrays and saves as an .npy file so we don't have to do this every time. '''
    
    # Iterate over images and apply upscale_and_crop
    X_list = []
    Y_list = []
    n_imgs = len(list(save_path.iterdir()))
    for i, folder in enumerate(save_path.iterdir()):
        print('Image {}/{}'.format(i+1, n_imgs))
        img = np.array(Image.open(list(folder.glob('images/*.png'))[0]))
        img = upscale_and_crop(img)
        X_list.append(img.astype(np.uint8)) # Convert to 8-bit integer to save space!
        mask = np.array(Image.open(folder/'stacked_img.png'))
        mask = upscale_and_crop(mask)
        Y_list.append(mask.astype(bool))
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    print('Saving X...')
    np.save(save_path/'X_train.npy', X)
    print('Saving Y...')
    np.save(save_path/'Y_train.npy', Y)

def plot_metrics(data_path, plot_path=None):
    '''
    Plots the training/validation metrics collected in log files.
    '''
    # Import necessary modules
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    import numpy as np
    import re
    
    if plot_path is None:
        plot_path = data_path
    
    # Get training size, batch size, logging frequency
    with open(data_path+'.log', 'r') as fo:
        p_train = re.compile('Training data')
        p_val_batch = re.compile('Validation set size')
        p_train_batch = re.compile('Batch size')
        p_log = re.compile('Logging frequency')
        p_num = re.compile(r'\d+')
        for line in fo:
            if re.match(p_train, line) is not None:
                m = re.search(p_num, line)
                m_train = int(m.group())
            elif re.match(p_train_batch, line) is not None:
                m = re.search(p_num, line)
                b_train = int(m.group())
            elif re.match(p_val_batch, line) is not None:
                m = re.search(p_num, line)
                b_val = int(m.group())
            elif re.match(p_log, line) is not None:
                m = re.search(p_num, line)
                log_freq = int(m.group())
    
    # Plot the loss
    loss_dict = [('Training','_train_loss.log'), ('Validation','_val_loss.log')]
    plt.figure(num='Loss')
    plt.clf()
    ax = plt.gca()
    for name, file in loss_dict:
        loss_list = []
        with open(data_path+file, 'r') as fo:
            for line in fo:
                loss_list.append(float(line))
        x = (log_freq*b_train/(m_train-b_val))*np.arange(len(loss_list))
        if name == 'Training':
            x_train = len(x)
        else:
            x = (x_train/len(x))*x
        plt.plot(x, loss_list, 'o', label=name, alpha=0.25)
    del loss_list
    plt.title('Average batch loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()
    plt.savefig(plot_path+'_loss.png')
    
#    # Plot the error
#    acc_dict = [('Training','_train_accuracy.log'), ('Validation','_val_accuracy.log')]
#    plt.figure(num='Error')
#    plt.clf()
#    ax = plt.gca()
#    for name, file in acc_dict:
#        acc_list = []
#        with open(data_path+file, 'r') as fo:
#            for line in fo:
#                acc_list.append(float(line))
#        x = (log_freq*b_train/(m_train-b_val))*np.arange(len(acc_list))
#        if name == 'Training':
#            x_train = len(x)
#        else:
#            x = (x_train/len(x))*x
#        plt.plot(x, 100*(1-np.array(acc_list)), 'o' , label=name, alpha=0.25)
#    del acc_list
#    plt.title('Average batch error')
#    plt.xlabel('Epoch')
#    plt.ylabel('Error (%)')
#    ax.set_xscale('log')
#    ax.set_yscale('log')
#    plt.legend()
#    plt.savefig(plot_path+'_error.png')
#    
#    # Plot the confidence
#    conf_dict = [('Training','_train_confidence.log'), ('Validation','_val_confidence.log')]
#    plt.figure(num='Uncertainty')
#    plt.clf()
#    ax = plt.gca()
#    for name, file in conf_dict:
#        conf_list = []
#        with open(data_path+file, 'r') as fo:
#            for line in fo:
#                conf_list.append(float(line))
#        x = (log_freq*b_train/(m_train-b_val))*np.arange(len(conf_list))
#        if name == 'Training':
#            x_train = len(x)
#        else:
#            x = (x_train/len(x))*x
#        plt.plot(x, (1-np.array(conf_list)), 'o' , label=name, alpha=0.25)
#    del conf_list
#    plt.title('Average batch uncertainty')
#    plt.xlabel('Epoch')
#    plt.ylabel('Uncertainty')
#    ax.set_xscale('log')
#    ax.set_yscale('log')
#    plt.legend()
#    plt.savefig(plot_path+'_uncertainty.png')

if __name__ == '__main__':
    TRAIN_PATH = Path('../Datasets/NucleusSegmentation/stage1_train')
    
#    a = np.eye(3)
#    b = upscale_and_crop(a)
#    print(a)
#    print(b[:,:,0])
    
#    stack_masks(TRAIN_PATH)
    
    preprocess_data(TRAIN_PATH)

































