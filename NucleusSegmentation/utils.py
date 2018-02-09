# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:49:03 2018

Utility functions

@author: Eric Cotner
"""

import numpy as np
from pathlib import Path
from PIL import Image

def stack_masks(save_path):
    """ Stacks the nucleus image masks on top of each other into a single mask, then saves it to a single image. """
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

def generate_boundary_layer(img):
    """ Takes in a binary map in numpy array form, and finds the boundary pixels for each mask. """
    boundary_array = np.zeros(shape=img.shape, dtype=np.uint8)
    height, width = img.shape[:2]
    for y in range(height):
        if not np.all(np.equal(img[y,:], 0)):
            for x in range(width):
                if img[y,x] == 0:
                    pass
                else:
                    neighbor_pix = [img[y,max(0,x-1)], img[y,min(width-1,x+1)], img[max(0,y-1),x], img[min(height-1,y+1),x]]
                    if np.any(np.equal(neighbor_pix, 0)):
                        boundary_array[y,x] = 1
    return boundary_array

def stack_masks_and_gen_boundary(save_path):
    """ Stacks the nucleus image masks on top of each other into a single mask, then saves it to a single image. Also generates an additional layer which is used as an interpolation between the center and boundaries of the nucleus by calculating the centroid of the nucleus, setting this point to have a value of 1, then solving Laplace's equaiton in the interior. """
    dir_len = len(list(save_path.iterdir()))
    for i, x in enumerate(save_path.iterdir()):
        masks_glob = x.glob('masks/*.png')
        stacked_img = None
        print('Image {}/{}'.format(i, dir_len))
        # Stack images on each other (by simply adding)
        if x.is_dir():
            for fo in masks_glob:
                img = np.array(Image.open(fo))
                boundary = generate_boundary_layer(img)
                if stacked_img is None:
                    stacked_img = np.stack([img, boundary], axis=-1)
                else:
                    stacked_img += np.stack([img, boundary], axis=-1)
        # Save as both .npy file and .png
            np.save(x/'stacked_and_interpolated_img.npy', stacked_img)
#        stacked_img = Image.fromarray(stacked_img)
#        stacked_img.save(x/'stacked_img.png')

def upscale_and_crop(img, max_height, max_width):
    ''' Dimensions of largest image are 1040x1388, so we'll upscale to just beyond that size (1200x1400) and then crop. '''
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

def upscale_crop_and_split(img, max_size):
    ''' Upscales the image through reflection/rotations, then crops to the next mulitple of max_size. '''
    height, width = img.shape[:2]
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    n_channels = img.shape[-1]
    if height > max_size:
        n_height_refl = 1
    else:
        n_height_refl = max_size//height
    if width > max_size:
        n_width_refl = 1
    else:
        n_width_refl = max_size//width
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
    height, width = new_img.shape[:2]
    new_img = new_img[:max_size*(height//max_size), :max_size*(width//max_size), :]
    del refl_img, img
    # new_img is now the image upscaled and cropped to fit in a multiple of max_size; now we split it into these multiples
    height, width = new_img.shape[:2]
    nx_splits = width//max_size
    ny_splits = height//max_size
    img_list = []
    for nx in range(nx_splits):
        for ny in range(ny_splits):
            img_list.append(new_img[ny*max_size:(ny+1)*max_size, nx*max_size:(nx+1)*max_size])
    del new_img
    return img_list

def split_image(img, n):
    ''' Splits a square image n times along each axis. '''
    stride = img.shape[0]//n
    img_list = []
    for nx in range(n):
        for ny in range(n):
            img_slice = img[ny*stride:(ny+1)*stride, nx*stride:(nx+1)*stride, :]
            img_list.append(img_slice)
    return img_list

def preprocess_data(save_path):
    ''' Loads the data, preprocesses it (by reflecting smaller images around their edges and then cropping to uniform size), then stacks them all into a single pair of (X,Y) arrays and saves as an .npy file so we don't have to do this every time. '''
    
    # Iterate over images and apply upscale_and_crop and split_image
    X_list = []
    Y_list = []
    n_imgs = len(list(save_path.iterdir()))
    for i, folder in enumerate(save_path.iterdir()):
        if folder.is_dir():
            print('Image {}/{}'.format(i+1, n_imgs))
            img = np.array(Image.open(list(folder.glob('images/*.png'))[0]))
            img = upscale_and_crop(img, max_height=256*6, max_width=256*6)
            img = split_image(img, n=6)
            for im in img:
                X_list.append(im.astype(np.uint8)) # Convert to 8-bit integer to save space!
            mask = np.array(Image.open(folder/'stacked_img.png'))
            mask = upscale_and_crop(mask, max_height=256*6, max_width=256*6)
            mask = split_image(mask, 6)
            for m in mask:
                Y_list.append(m.astype(bool))
    X = np.stack(X_list, axis=0)
    del X_list
    Y = np.stack(Y_list, axis=0)
    del Y_list
    print('Saving X...')
    np.save(save_path/'X_train.npy', X)
    del X
    print('Saving Y...')
    np.save(save_path/'Y_train.npy', Y)
    del Y

def preprocess_data2(save_path):
    ''' Loads the data, preprocesses it (by reflecting smaller images around their edges and then cropping to uniform size), then stacks them all into a single pair of (X,Y) arrays and saves as an .npy file so we don't have to do this every time. Also generates an additional output layer which is a measure of the interpolation between the center and boundary of the nucleus.'''

    # Iterate over images and apply upscale_and_crop and split_image
    X_list = []
    Y_list = []
    n_imgs = len(list(save_path.iterdir()))
    for i, folder in enumerate(save_path.iterdir()):
        if folder.is_dir():
            print('Image {}/{}'.format(i+1, n_imgs))
            img = np.array(Image.open(list(folder.glob('images/*.png'))[0]))
            img = upscale_crop_and_split(img, max_size=256)
            for im in img:
                X_list.append(im.astype(np.uint8)) # Convert to 8-bit integer to save space!
            mask = np.load(folder/'stacked_and_interpolated_img.npy')
            mask = upscale_crop_and_split(mask, max_size=256)
            for m in mask:
                Y_list.append(m.astype(bool))
    X = np.stack(X_list, axis=0)
    del X_list
    Y = np.stack(Y_list, axis=0)
    del Y_list
    print('Saving X...')
    np.save(save_path/'X_train.npy', X)
    del X
    print('Saving Y...')
    np.save(save_path/'Y_train.npy', Y)
    del Y

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
    
    # Plot the error
    acc_dict = [('Training','_train_accuracy.log'), ('Validation','_val_accuracy.log')]
    plt.figure(num='Error')
    plt.clf()
    ax = plt.gca()
    for name, file in acc_dict:
        acc_list = []
        with open(data_path+file, 'r') as fo:
            for line in fo:
                acc_list.append(float(line))
        x = (log_freq*b_train/(m_train-b_val))*np.arange(len(acc_list))
        if name == 'Training':
            x_train = len(x)
        else:
            x = (x_train/len(x))*x
        plt.plot(x, 100*(1-np.array(acc_list)), 'o' , label=name, alpha=0.25)
    del acc_list
    plt.title('Average batch error')
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()
    plt.savefig(plot_path+'_error.png')
    
    # Plot the uncertainty
    conf_dict = [('Training','_train_confidence.log'), ('Validation','_val_confidence.log')]
    plt.figure(num='Uncertainty')
    plt.clf()
    ax = plt.gca()
    for name, file in conf_dict:
        conf_list = []
        with open(data_path+file, 'r') as fo:
            for line in fo:
                conf_list.append(float(line))
        x = (log_freq*b_train/(m_train-b_val))*np.arange(len(conf_list))
        if name == 'Training':
            x_train = len(x)
        else:
            x = (x_train/len(x))*x
        plt.plot(x, (1-np.array(conf_list)), 'o' , label=name, alpha=0.25)
    del conf_list
    plt.title('Average batch uncertainty')
    plt.xlabel('Epoch')
    plt.ylabel('Uncertainty')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()
    plt.savefig(plot_path+'_uncertainty.png')
    
    # Plot the IOU
    IOU_dict = [('Training','_train_IOU.log'), ('Validation','_val_IOU.log')]
    plt.figure(num='IOU')
    plt.clf()
    ax = plt.gca()
    for name, file in IOU_dict:
        IOU_list = []
        with open(data_path+file, 'r') as fo:
            for line in fo:
                IOU_list.append(float(line))
        x = (log_freq*b_train/(m_train-b_val))*np.arange(len(IOU_list))
        if name == 'Training':
            x_train = len(x)
        else:
            x = (x_train/len(x))*x
        plt.plot(x, IOU_list, 'o' , label=name, alpha=0.25)
    del IOU_list
    plt.title('Intersection over union (IOU)')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    plt.legend()
    plt.savefig(plot_path+'_IOU.png')

def var(X, mean):
    ''' Memory-efficient calculation of variance (since numpy runs out of memory computing variance of dataset). '''
    sum_of_squares = 0
    for i in range(X.shape[0]):
        sum_of_squares += np.sum(np.square(X[i]-mean))
    return sum_of_squares/np.prod(X.shape)
    

if __name__ == '__main__':
    TRAIN_PATH = Path('../Datasets/NucleusSegmentation/stage1_train')
    
#    a = np.eye(3)
#    b = upscale_and_crop(a)
#    print(a)
#    print(b[:,:,0])
    
#    stack_masks_and_gen_boundary(TRAIN_PATH)
    
#    preprocess_data2(TRAIN_PATH)
    
    plot_metrics('./models/3/UNet3')


