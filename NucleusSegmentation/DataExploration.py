# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:19:45 2018

Script for exploring the nucleus segmentation data. I will be using Pillow and numpy here for ease of use, but for actually handling/preprocessing the images during the training, I should probably use a library with a fast C implementation like OpenCV to reduce the preprocessing overhead and avoid bottleneck.

@author: Eric Cotner
"""

import numpy as np
from PIL import Image
from pathlib import Path
import time
import matplotlib.pyplot as plt

#print(os.getcwd())

DATA_DIR = Path('../Datasets/NucleusSegmentation')
TRAIN_DIR = DATA_DIR / 'stage1_train'
CWD = Path('.')

def plot_images():
    plt.figure('Raw image')
    # Iterate over subdirectories
    for x in TRAIN_DIR.iterdir():
        # Get the image file
        fg = x.glob('images/*.png')
        for fimg in fg:
            # Open image
            im = np.array(Image.open(fimg))
            plt.clf()
            # Plot image
            plt.imshow(im)
            plt.draw()
            plt.pause(1e-9)
        time.sleep(3)

def plot_masked_images():
    plt.figure('Masked image')
    # Iterate over subdirectories
    for x in TRAIN_DIR.iterdir():
        # Get the image file
        fg = x.glob('images/*.png')
        for fimg in fg:
            im = np.array(Image.open(fimg))
        # Get masks
        fg = x.glob('masks/*.png')
        masks = []
        for fimg in fg:
            masks.append(np.array(Image.open(fimg)))
        # Plot image together with masks overlaid
        plt.clf()
        plt.imshow(im)
        for mask in masks:
            mask = np.ma.masked_equal(mask,0)
            plt.imshow(mask, alpha=0.25, cmap='Greens_r')
        plt.show()
        plt.pause(1e-9)
        time.sleep(3)

def summarize_data():
    ''' Iterates through each image and summarizes statistics on image size, color channels, aspect ratios, number of nuclei/image, pixelwise size/volume of nuclei, mask center of mass, pixel values in raw image covered by mask, THINK OF MORE THINGS! '''
    
    def print_and_write(string, fo):
        print(string)
        fo.write(string+'\n')
    
    # Initialize empty lists of things
    L_image_dims = []
    L_mask_volumes = []
    L_mask_CM = []
#    L_mask_pixels_raw = []
    # Iterate over subdirectories
    print('Gathering data')
    dir_len = len(list(TRAIN_DIR.iterdir()))
    for i, x in enumerate(TRAIN_DIR.iterdir()):
        print('Image {}/{}'.format(i, dir_len))
        # Get image and masks data
        im_glob = x.glob('images/*.png')
        mask_glob = x.glob('masks/*.png')
        for fo in im_glob:
            im = Image.open(fo)
        dims = np.array(im).shape
        L_image_dims.append(dims)
        L_mask_volumes.append([])
        L_mask_CM.append([])
#        L_mask_pixels_raw.append([])
        grid_x, grid_y = np.meshgrid(range(dims[1]), range(dims[0]))
        # Iterate over masks
        for fo in mask_glob:
            mask = np.array(Image.open(fo))
            volume = np.sum(mask)/256
            L_mask_volumes[-1].append(volume)
            CM_x = np.sum(mask*grid_x)/volume
            CM_y = np.sum(mask*grid_y)/volume
            L_mask_CM[-1].append((CM_x, CM_y))
    
    print('Generating summary data')
    # Print summaries
    SUMMARY_PATH = CWD/'DataSummary'
    SUMMARY_PATH.mkdir(exist_ok=True)
    with (SUMMARY_PATH/'DataSummary.log').open('w+') as fo:
        # Get min/max image size
        min_size = L_image_dims[np.argmin(np.prod(L_image_dims, axis=1))]
        max_size = L_image_dims[np.argmax(np.prod(L_image_dims, axis=1))]
        print_and_write('Min image size: {}, max image size: {}'.format(min_size, max_size), fo)
        # Count nuclei/image, volumes of nuclei
        nucleus_count = []
        flattened_volumes = []
        flattened_norm_volumes = []
        for i, volumes in enumerate(L_mask_volumes):
            nucleus_count.append(len(volumes))
            for volume in volumes:
                flattened_volumes.append(volume)
                flattened_norm_volumes.append(volume/np.prod(L_image_dims[i][:-1]))
        print_and_write('Mean nucleus volume (pixels): {:.1f}±{:.1f}, mean fractional nucleus volume: {:.2e}±{:.2e}'.format(np.mean(flattened_volumes), np.std(flattened_volumes), np.mean(flattened_norm_volumes), np.std(flattened_norm_volumes)), fo)
        print_and_write('Number of nuclei per image: {:.1f}±{:.1f}'.format(np.mean(nucleus_count), np.std(nucleus_count)), fo)
        
    # Plot histograms
    plt.figure('Histogram')
    
    plt.clf()
    plt.hist(nucleus_count)
    plt.title('Number of nuclei per image')
    plt.xlabel('Num. nuclei')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.savefig(str(SUMMARY_PATH/'NucleiCountHistogram'))
    
    plt.clf()
    plt.hist(flattened_volumes)
    plt.title('Nuclei volumes in pixels')
    plt.xlabel('Volume (pix)')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.savefig(str(SUMMARY_PATH/'VolumeHistogram.png'))
    
    plt.clf()
    plt.hist(np.sqrt(flattened_volumes))
    plt.title('Geometric nucleus length in pixels')
    plt.xlabel('Length (pix)')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.savefig(str(SUMMARY_PATH/'LengthHistogram.png'))
    
    plt.clf()
    plt.hist(flattened_norm_volumes)
    plt.title('Nuclei volumes as fraction of image volume')
    plt.xlabel('Cell/image ratio')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.savefig(str(SUMMARY_PATH/'FractionalVolumeHistogram.png'))
    
    plt.clf()
    dims = np.array(L_image_dims)
    plt.hist(dims[:,0], label='Height', alpha=0.5)
    plt.hist(dims[:,1], label='Width', alpha=0.5)
    plt.title('Image height/width')
    plt.xlabel('Pixels')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(str(SUMMARY_PATH/'ImgDimensionsHistogram'))


''' ======================================================================== '''
#plot_images()
#plot_masked_images()
summarize_data()












