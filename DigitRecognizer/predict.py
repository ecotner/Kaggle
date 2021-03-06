# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:17:47 2018

Runs inference on the prediction of the classification network.

@author: Eric Cotner
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils as u
import time
import re
import pandas as pd

# Define parameters
DATA_PATH = '../Datasets/MNIST/test.csv'
SAVE_PATH = './checkpoints/{0}/DigitRecognizer_{0}'.format(2)
N_DROPOUT_GROUPS = 3
PREDICTION_PERIOD = 10 # Number of second between displaying successive predictions
BATCH_SIZE = 256
USE_GPU = True
GPU_MEM_FRACTION = 0.3
USE_UNLABELED_DATA = True
RUN_INDIVIDUAL_INFERENCE = False
SHOW_MISCLASSIFICATIONS = False
label_names = ['0','1','2','3','4','5','6','7','8','9']

# Load dataset
print('Loading test data...')
data_raw = pd.read_csv(DATA_PATH)

# (Pre-)process data
print('Processing data...')
if USE_UNLABELED_DATA:
    X_test = np.reshape(np.array(data_raw.iloc[:,:], dtype=int), [-1,28,28,1])
else:
    X_test = np.reshape(np.array(data_raw.iloc[:,1:], dtype=int), [-1,28,28,1])
    Y_test = np.array(data_raw.iloc[:,0], dtype=int)
del data_raw

# Get mean and std from log file
p_mean = re.compile('X_mean')
p_std = re.compile('X_std')
p_float = re.compile(r'\d+\.\d+')
with open(SAVE_PATH+'.log', 'r') as fo:
    for line in fo:
        if re.match(p_mean, line) is not None:
            m = re.search(p_float, line)
            X_mean = float(m.group())
        elif re.match(p_std, line) is not None:
            m = re.search(p_float, line)
            X_std = float(m.group())
# Apply normalization and permutation
X_test = (X_test - X_mean)/X_std
if not USE_UNLABELED_DATA:
    seed = int(time.time())
    np.random.seed(seed)
    X_test = np.random.permutation(X_test)
    np.random.seed(seed)
    Y_test = np.random.permutation(Y_test)

# Load tensorflow graph
G = tf.Graph()
with G.as_default():
    print('Loading graph...')
    if USE_GPU:
        saver = tf.train.import_meta_graph(SAVE_PATH+'.meta', clear_devices=True)
        sess_config = tf.ConfigProto(device_count={'CPU':4, 'GPU':1})
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    else:
        tf.device('/cpu:0')
        saver = tf.train.import_meta_graph(SAVE_PATH+'.meta', clear_devices=True)
        tf.device('/cpu:0')
        sess_config = tf.ConfigProto(device_count={'CPU':4, 'GPU':0})
    
    # Get important tensors from the graph
    X = G.get_tensor_by_name('input:0')
    Y = G.get_tensor_by_name('output:0')
    labels = G.get_tensor_by_name('labels:0')
    J = G.get_tensor_by_name('loss:0')
    regularization_parameter = G.get_tensor_by_name('regularization_parameter:0')
    keep_prob = {n:G.get_tensor_by_name('keep_prob_'+str(n)+':0') for n in range(1,N_DROPOUT_GROUPS+1)}
    is_training = G.get_tensor_by_name('is_training:0')
    
    # Make any new necessary tensors:
    prob = tf.nn.softmax(Y) # Predicted probability distribution of classes
    
    # Begin tensorflow session
    with tf.Session(config=sess_config) as sess:
        print('Restoring parameters...')
        saver.restore(sess, SAVE_PATH)
        
        plt.figure('MNIST Prediction')
        
        print('Running prediction...')
        if USE_UNLABELED_DATA:
            predicted_labels = []
        else:
            classification_score = {'right':0, 'wrong':0}
        for b in range(X_test.shape[0]//BATCH_SIZE+1):
            # Run prediction on images (do a large batch at a time so batch norm can get statistics, since it looks like the inference mode is broken)
            print('Running batch {}/{}...'.format(b, X_test.shape[0]//BATCH_SIZE))
            slice_lower = b*BATCH_SIZE
            slice_upper = min(X_test.shape[0], (b+1)*BATCH_SIZE)
            if USE_UNLABELED_DATA:
                feed_dict = {**{X:X_test[slice_lower:slice_upper,:,:,:], is_training:True}, **{keep_prob[n]:1 for n in range(1,N_DROPOUT_GROUPS+1)}}
                class_prob = sess.run(prob, feed_dict=feed_dict)
            else:
                feed_dict = {**{X:X_test[slice_lower:slice_upper,:,:,:], labels:Y_test[slice_lower:slice_upper], is_training:True}, **{keep_prob[n]:1 for n in range(1,N_DROPOUT_GROUPS+1)}}
                class_prob, loss = sess.run([prob, J], feed_dict=feed_dict)
            class_pred = np.argmax(class_prob, axis=-1).squeeze()
            if SHOW_MISCLASSIFICATIONS and (not RUN_INDIVIDUAL_INFERENCE):
                misclassification_idx = np.not_equal(class_pred, Y_test[slice_lower:slice_upper])
            
            if RUN_INDIVIDUAL_INFERENCE:
                # Plot each individual image in the batch
                for idx, lbl in enumerate(class_pred):
                    if (not SHOW_MISCLASSIFICATIONS) or (misclassification_idx[idx]):
                        tic = time.time()
                        
                        # Plot the results
                        plt.clf()
                        if X_test.shape[-1] != 1:
                            img = X_test[slice_lower+idx,:,:,:]
                        else:
                            img = X_test[slice_lower+idx,:,:,0]
                        img = (img - np.min(img))/(np.max(img) - np.min(img))
                        plt.imshow(img)
                        if USE_UNLABELED_DATA:
                            print('Class prob. dist: {}\nClass_pred: {}\n\n'.format(class_prob[idx], label_names[lbl]))
                            plt.title('Prediction: {} ({:.1f}%)'.format(label_names[lbl], 100*class_prob[idx,lbl]))
                        else:
                            if lbl == Y_test[slice_lower+idx]:
                                right_or_wrong = '\u2714'
                            else:
                                right_or_wrong = 'X'
                            print('Class prob. dist: {}\nClass_pred: {}, loss: {:.2e}\n\n'.format(class_prob[idx], label_names[lbl], loss))
                            plt.title('Class: {}, prediction: {} ({:.1f}%) {}'.format(label_names[Y_test[slice_lower+idx]], label_names[lbl], 100*class_prob[idx,lbl], right_or_wrong))
                        plt.xticks([])
                        plt.yticks([])
                        plt.draw()
                        plt.pause(1e-9)
                        
                        # Time delay between plotting
                        toc = time.time()
                        time.sleep(max(0, PREDICTION_PERIOD - (toc-tic)))
            else:
                if USE_UNLABELED_DATA:
                    # Gather predicted labels
                    predicted_labels = np.append(predicted_labels, class_pred, axis=0)
                else:
                    # Gather summary statistics
                    n_right = np.sum(np.equal(class_pred, Y_test[slice_lower:slice_upper]))
                    classification_score['right'] += n_right
                    classification_score['wrong'] += (len(class_pred) - n_right)
        
        if USE_UNLABELED_DATA:
            # Write predictions to a csv
            output = pd.DataFrame(data=np.stack([np.arange(1,len(predicted_labels)+1, dtype=int), predicted_labels.astype(int)], axis=-1), columns=['ImageId','Label'])
            output.to_csv(SAVE_PATH+'_predictions.csv', index=False)
        else:
            # Print summary stats
            if not RUN_INDIVIDUAL_INFERENCE:
                error = classification_score['wrong']/(classification_score['right'] + classification_score['wrong'])
                print('Test set error: {}\nNum correctly classified: {}, incorrectly classified: {}'.format(error, classification_score['right'], classification_score['wrong']))