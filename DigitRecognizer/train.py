# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:11:01 2018

@author: Eric Cotner
"""

# Import necessary modules
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import numpy as np
import utils as u
import time
import pandas as pd

# Define hyperparameters, file paths, and control flow variables
RESET_PARAMETERS = True
BATCH_SIZE = 256
VAL_BATCH_SIZE = 256
LEARNING_RATE = 7e-2
LEARNING_RATE_ANNEAL_RATE = 50     # Number of epochs after which learning rate is annealed by
LEARNING_RATE_ANNEAL_TYPE = '1/x'
STEPPED_ANNEAL = True # Whether or not the learning rate is annealed all at once, or slowly over time
REGULARIZATION_TYPE = 'L2'  # Regularization type is already determined in ResNet.py
REGULARIZATION_PARAMETER = 1e-2
INPUT_NOISE_MAGNITUDE = np.sqrt(0.1)
WEIGHT_NOISE_MAGNITUDE = np.sqrt(0.1)
KEEP_PROB = {1: .5, 2: 0.6, 3: 0.7}
SAVE_PATH = './checkpoints/{0}/DigitRecognizer_{0}'.format(2)
MAX_EPOCHS = int(1e10)
DATA_PATH = '../Datasets/MNIST/train.csv'
LOG_EVERY_N_STEPS = 50
GPU_MEM_FRACTION = 0.6

# Load image data. Returns a pandas Dataframe object
print('Loading data...')
data_raw = pd.read_csv(DATA_PATH)

# Extract data from dict
print('Processing data...')
X = np.reshape(np.array(data_raw.iloc[:,1:], dtype=int), [-1,28,28,1])
Y = np.array(data_raw.iloc[:,0], dtype=int)
del data_raw

# Visualize just to make sure it looks right
#u.visualize_dataset(X,Y)

# Preprocess data by normalizing to zero mean and unit variance
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean)/X_std

# Shuffle data and split into training and validation sets
RANDOM_SEED = int(time.time())
np.random.seed(RANDOM_SEED)
X_train = np.random.permutation(X)
np.random.seed(RANDOM_SEED)
Y_train = np.random.permutation(Y)
tf.set_random_seed(RANDOM_SEED)
del X, Y
m_train = len(Y_train) - VAL_BATCH_SIZE

# Log some info about the data for future use
with open(SAVE_PATH+'.log', 'w+') as fo:
    fo.write('Training log\n\n')
    fo.write('Dataset metrics:\n')
    fo.write('Training data shape: {}\n'.format(X_train.shape))
    fo.write('Validation set size: {}\n'.format(VAL_BATCH_SIZE))
    fo.write('X_mean: {}\n'.format(X_mean))
    fo.write('X_std: {}\n\n'.format(X_std))
    fo.write('Hyperparameters:\n')
    fo.write('Batch size: {}\n'.format(BATCH_SIZE))
    fo.write('Learning rate: {}\n'.format(LEARNING_RATE))
    fo.write('Learning rate annealed every N epochs: {}\n'.format(LEARNING_RATE_ANNEAL_RATE))
    fo.write('Learning rate anneal type: {}\n'.format(LEARNING_RATE_ANNEAL_TYPE))
    fo.write('Stepped anneal: {}\n'.format(STEPPED_ANNEAL))
    fo.write('Regularization type: {}\n'.format(REGULARIZATION_TYPE))
    fo.write('Regularization parameter: {}\n'.format(REGULARIZATION_PARAMETER))
    fo.write('Input noise variance: {:.2f}\n'.format(INPUT_NOISE_MAGNITUDE**2))
    fo.write('Weight noise variance: {:.2f}\n'.format(WEIGHT_NOISE_MAGNITUDE**2))
    for n in range(1,len(KEEP_PROB)+1):
        fo.write('Dropout keep prob. group {}: {:.2f}\n'.format(n, KEEP_PROB[n]))
    fo.write('Logging frequency: {} global steps\n'.format(LOG_EVERY_N_STEPS))
    fo.write('Random seed: {}\n'.format(RANDOM_SEED))
    fo.write('\nNotes:\n')

# Load network architecture/parameters
G = u.load_graph(SAVE_PATH)
with G.as_default():
    tf.set_random_seed(RANDOM_SEED)
    tf.device('/gpu:0')
    saver = tf.train.Saver(var_list=tf.global_variables())
    
    # Get important tensors/operations from graph
    X = G.get_tensor_by_name('input:0')
    Y = G.get_tensor_by_name('output:0')
    labels = G.get_tensor_by_name('labels:0')
    J = G.get_tensor_by_name('loss:0')
    training_op = G.get_operation_by_name('training_op')
    learning_rate = G.get_tensor_by_name('learning_rate:0')
    regularization_parameter = G.get_tensor_by_name('regularization_parameter:0')
    keep_prob = {n:G.get_tensor_by_name('keep_prob_'+str(n)+':0') for n in range(1,len(KEEP_PROB)+1)}
    weight_noise = G.get_tensor_by_name('weight_noise:0')
    
    # Load the preprocessed training and validation data in TensorFlow constants if possible so that there is no bottleneck sending things from CPU to GPU
    print('Loading dataset to GPU')
    X_train_ = tf.constant(X_train, dtype=tf.float32)
    Y_train_ = tf.constant(Y_train, dtype=tf.int32)
    del X_train, Y_train
    X_train = X_train_
    Y_train = Y_train_
    
    # Reroute tensors to the location of the data on the GPU, add noise and image augmentation (random crops)
    train_idx = tf.placeholder(tf.int32, shape=[None], name='train_idx')
    tf.add_to_collection('placeholders', train_idx)
    input_noise_magnitude = tf.placeholder_with_default(0.0, shape=[], name='input_noise_magnitude')
    tf.add_to_collection('placeholders', input_noise_magnitude)
    X_train = tf.gather(X_train, train_idx)
    X_train = tf.pad(X_train, paddings=[[0,0],[3,3],[3,3],[0,0]])
    X_train = u.random_crop(X_train, [28,28,1])
    X_train += input_noise_magnitude*tf.random_normal(tf.shape(X_train), dtype=tf.float32)
    Y_train = tf.gather(Y_train, train_idx)
    ge.reroute_ts([X_train, Y_train], [X, labels])
    
    # Define tensor to compute accuracy and confidence metrics
    Y_pred = tf.argmax(Y, axis=-1, output_type=tf.int32)
    prob = tf.nn.softmax(Y, dim=-1)
    Y_pred_prob = tf.reduce_max(prob, axis=-1)
    A = tf.cast(tf.equal(Y_train, Y_pred), tf.float32)
    acc = tf.reduce_mean(A)
    conf = tf.reduce_mean(Y_pred_prob*A)
    
    # Start the TF session and load variables
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    with tf.Session(config=config) as sess:
        if RESET_PARAMETERS:
            print('Resetting parameters...')
            sess.run(tf.global_variables_initializer())
        else:
            print('Loading parameters...')
            saver.restore(sess, SAVE_PATH)
        
        # Initialize control flow variables and logs
        max_val_accuracy = -1
        max_val_conf = -1
        min_val_loss = np.inf
        global_steps = 0
        with open(SAVE_PATH+'_val_accuracy.log', 'w+') as fo:
                fo.write('')
        with open(SAVE_PATH+'_val_loss.log', 'w+') as fo:
            fo.write('')
        with open(SAVE_PATH+'_val_confidence.log', 'w+') as fo:
            fo.write('')
        with open(SAVE_PATH+'_train_accuracy.log', 'w+') as fo:
            fo.write('')
        with open(SAVE_PATH+'_train_loss.log', 'w+') as fo:
            fo.write('')
        with open(SAVE_PATH+'_train_confidence.log', 'w+') as fo:
            fo.write('')
        with open(SAVE_PATH+'_learning_rate.log', 'w+') as fo:
            fo.write('')
        
        # Iterate over epochs
        print('Beginning training...')
        for epoch in range(MAX_EPOCHS):
            if STEPPED_ANNEAL:
                x = (epoch//LEARNING_RATE_ANNEAL_RATE)
            else:
                x = (epoch/LEARNING_RATE_ANNEAL_RATE)
            if LEARNING_RATE_ANNEAL_TYPE in ['exponential','exp','1/e','exp(-x)']:
                lr = LEARNING_RATE*np.exp(-x)
            elif LEARNING_RATE_ANNEAL_TYPE in ['inverse','1/x']:
                lr = LEARNING_RATE/(1+x)
            elif LEARNING_RATE_ANNEAL_TYPE in ['none','None',None]:
                lr = LEARNING_RATE
            # Iterate over batches
            for b in range(m_train//BATCH_SIZE+1):
                
                # Perform forward/backward pass
                slice_lower = b*BATCH_SIZE
                slice_upper = min((b+1)*BATCH_SIZE, m_train)
                feed_dict = {**{learning_rate:lr, train_idx:range(slice_lower, slice_upper), regularization_parameter:REGULARIZATION_PARAMETER, input_noise_magnitude:INPUT_NOISE_MAGNITUDE, weight_noise:WEIGHT_NOISE_MAGNITUDE}, **{keep_prob[n]:KEEP_PROB[n] for n in range(1,len(KEEP_PROB)+1)}}
                train_loss, train_accuracy, train_conf, _ = sess.run([J, acc, conf, training_op], feed_dict=feed_dict)
                if (train_loss in [np.nan, np.inf]) or (train_loss > 1e3):
                    print('Detected numerical instability in training, exiting')
                    exit()
                
                # Compute metrics, add to logs
                if (global_steps % LOG_EVERY_N_STEPS == 0) and (global_steps != 0):
                    slice_lower = m_train
                    slice_upper = m_train + VAL_BATCH_SIZE
                    feed_dict = {train_idx:range(slice_lower, slice_upper), regularization_parameter:REGULARIZATION_PARAMETER, input_noise_magnitude:0}
                    val_loss, val_accuracy, val_conf = sess.run([J, acc, conf], feed_dict=feed_dict)
                    print('Validation loss: {:.2e}, validation accuracy: {:.3f}'.format(val_loss, val_accuracy))
                    with open(SAVE_PATH+'_train_loss.log', 'a') as fo:
                        fo.write(str(train_loss)+'\n')
                    with open(SAVE_PATH+'_train_accuracy.log', 'a') as fo:
                        fo.write(str(train_accuracy)+'\n')
                    with open(SAVE_PATH+'_train_confidence.log', 'a') as fo:
                        fo.write(str(train_conf)+'\n')
                    with open(SAVE_PATH+'_val_accuracy.log', 'a') as fo:
                        fo.write(str(val_accuracy)+'\n')
                    with open(SAVE_PATH+'_val_loss.log', 'a') as fo:
                        fo.write(str(val_loss)+'\n')
                    with open(SAVE_PATH+'_val_confidence.log', 'a') as fo:
                        fo.write(str(val_conf)+'\n')
                    with open(SAVE_PATH+'_learning_rate.log', 'a') as fo:
                        fo.write(str(lr)+'\n')
                    
                    # Save if improvement
#                    if (val_accuracy > max_val_accuracy) or ((val_accuracy >= max_val_accuracy) and (val_loss < min_val_loss)):
#                        max_val_accuracy = val_accuracy
#                        min_val_loss = val_loss
                    if val_conf > max_val_conf:
                        max_val_conf = val_conf
                        print('Saving variables...')
                        saver.save(sess, SAVE_PATH, write_meta_graph=False)
                    
                print('Epoch: {}, batch: {}/{}, loss: {:.2e}, acc: {:.3f}, conf: {:.3f}, lr: {:.2e}'.format(epoch, b, m_train//BATCH_SIZE, train_loss, train_accuracy, train_conf, lr))
                
                # Iterate global step
                global_steps += 1
        
        # Print stuff once done
        print('Done!')
























