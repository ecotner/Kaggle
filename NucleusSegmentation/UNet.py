# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:49:17 2018

U-net architecture based on arXiv:1505.04597

This sort of generalizes the architecture by making the input image size dynamic, and uses "same" padding rather than the valid used in the paper since it avoids the need for cropping operations when concatenating the feature maps from the contracting phase to the expanding phase.

I'm also planning on replacing the vanilla convolutions with Inception blocks so that the network gets its choice of filters.

@author: Eric Cotner
"""

import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import numpy as np
import math
import utils as u

class UNet(object):
    '''
    U-net object. Provides methods for constructing, training, and predicting image segmentation tasks.
    
    Using the construction methods (convolution, inception_block), one may easily add successive layers to the network. Then, using the squeeze methods, the network is automatically branched - one branch reduces the dimensionality of the feature maps from the previous layer, while the other is simply an identity mapping. The reduced-dimensionality branch is said to be "reduced by one level". The identity is not touched until the other branch is brought back up to its original level. This is accomplished through transpose convolutions (stretch methods) to increase the dimensionality of the feature maps from lower levels back to their dimensionality before the branch. The identity maps are then concatenated with the features from the transpose convolutions. This results in a U-shaped architecture which has the same dimensionality in the output as the input (except for number of channels). Below is an example diagram of what the architecture looks like
    
     input->||||----------||||->output           | = convolutional layer
               V          Λ                      - = identity
               |||------|||                      V = downsample
      squeeze    V      Λ    stretch             Λ = upsample
         |       |||--|||       Λ
         V         V  Λ         |
                   ||||
    
    This architecture is supposedly well-suited to image segmentation tasks.
    '''
    def __init__(self, input_size=[None, None, 4], use_bn=True):
        self.G = tf.Graph()
        with self.G.as_default():
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, input_size[0], input_size[1], input_size[2]], name='input')
            self.output = self.input
            self.current_level = 0
            self.tensors_above = []
            self.loss = None
            self.labels = None
            self.reg_param = None
            self.learning_rate = None
            self.optimizer_placeholders = None
            self.train_op = None
    
    def f_act(self, x, activation):
        if activation == 'relu':
            return tf.nn.relu(x)
        elif activation == 'softmax':
            return tf.nn.softmax(x)
        elif activation == 'identity':
            return x
        elif activation == 'sigmoid':
            return tf.nn.sigmoid(x)
        elif activation == 'lrelu':
            return tf.maximum(0.2*x, x)
        elif activation == 'arelu': # Adaptive relu: learns the best slopes
            a1 = tf.Variable(0)
            a2 = tf.Variable(1)
            return tf.maximum(a1*x, a2*x)
        else:
            raise Exception('Unknown activation function')
    
    def convolution(self, f, s, n_out, activation='relu', padding='SAME'):
        with self.G.as_default():
            n_in = self.output.shape.as_list()[-1]
            F = tf.Variable(tf.random_normal([f, f, n_in, n_out])*tf.sqrt(2./tf.cast(n_in*f**2 + n_out, tf.float32)))
            tf.add_to_collection('weights', F)
            b = tf.Variable(0.01*tf.ones([n_out]))
            tf.add_to_collection('biases', b)
            self.output = self.f_act(tf.nn.conv2d(self.output, F, [1,s,s,1], padding=padding) + b, activation)
    
    def transpose_convolution(self, f, s, n_out, output_shape=None, activation='relu', padding='SAME'):
        with self.G.as_default():
            n_in = self.output.shape.as_list()[-1]
            F = tf.Variable(tf.random_normal([f, f, n_out, n_in])*tf.sqrt(2./tf.cast(n_in + n_out*f**2, tf.float32)))
            tf.add_to_collection('weights', F)
            b = tf.Variable(0.01*tf.ones([n_out]))
            tf.add_to_collection('biases', b)
            if output_shape is None:
                output_shape = tf.concat([tf.shape(self.output)[:-1], [n_out]], axis=0)
            self.output = self.f_act(tf.nn.conv2d_transpose(self.output, F, output_shape, [1,s,s,1], padding=padding) + b, activation)
    
    def max_pool(self, f, s, padding='SAME'):
        with self.G.as_default():
            self.output = tf.nn.max_pool(self.output, [1,f,f,1], [1,s,s,1], padding)
    
    def squeeze_convolution(self, f, s, n_out, activation='relu', padding='SAME'):
        with self.G.as_default():
            self.tensors_above.append(self.output)
            self.convolution(f, s, n_out, activation, padding)
            self.current_level += -1
    
    def stretch_transpose_convolution(self, f, s, n_out, activation='relu', padding='SAME'):
        with self.G.as_default():
            tensor_above = self.tensors_above.pop()
            output_shape = tf.concat([tf.shape(tensor_above)[:-1], [n_out]], axis=0)
            self.transpose_convolution(f, s, n_out, output_shape=output_shape, activation=activation, padding=padding)
            self.output = tf.concat([self.output, tensor_above], axis=-1)
            self.current_level += +1
    
    def inception_block(self, f_sizes, f_channels, s, use_shield=True):
        pass
    
    def squeeze_inception_block(self):
        pass
    
    def stretch_inception_block(self):
        pass
    
    def dropout(self):
        pass
    
    def add_loss(self, loss_type='xentropy', reg_type='L2'):
        with self.G.as_default():
            assert ((self.loss == None) and (self.labels == None)), 'Already have loss tensor'
            # Make placeholder for labels, feed into training loss
            self.labels = tf.placeholder(dtype=tf.float32, shape=self.output.shape.as_list(), name='labels')
            tf.add_to_collection('placeholders', self.labels)
            self.output = tf.identity(self.output, name='output')
            if loss_type == 'xentropy':
                if self.output.shape.as_list()[-1] > 1:
                    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels, dim=-1))
                else:
                    self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.labels))
            elif loss_type == 'L2':
                self.loss = tf.reduce_mean(tf.square(self.output - self.labels))
            else:
                raise Exception('Unknown loss function')
            
            # Add regularization loss
            self.reg_param = tf.placeholder_with_default(0., shape=[], name='reg_param')
            tf.add_to_collection('placeholders', self.reg_param)
            if reg_type == 'L2':
                reg_loss = tf.constant(0, dtype=tf.float32)
                param_count = 0
                for W in tf.get_collection('weights'):
                    param_count += np.prod(W.shape.as_list())
                    reg_loss += tf.nn.l2_loss(W)
                self.loss = tf.add(self.loss, (self.reg_param/param_count)*reg_loss, name='loss')
            else:
                raise Exception('Unknown regularizer')
    
    def add_optimizer(self, opt_type='adam'):
        assert (self.train_op is None) and (self.learning_rate is None), 'Already have training operation'
        with self.G.as_default():
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
            tf.add_to_collection('placeholders', self.learning_rate)
            if opt_type == 'adam':
                beta1 = tf.placeholder_with_default(0.9, shape=[], name='beta1')
                beta2 = tf.placeholder_with_default(0.999, shape=[], name='beta2')
                tf.add_to_collection('optimizer_placeholders', beta1)
                tf.add_to_collection('optimizer_placeholders', beta2)
                self.optimizer_placeholders = [self.learning_rate, beta1, beta2]
                optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2)
            elif opt_type == 'momentum':
                momentum = tf.placeholder_with_default(0.99, shape=[], name='momentum')
                tf.add_to_collection('optimizer_placeholders', momentum)
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum)
            else:
                raise Exception('Unknown optimizer')
            self.train_op = optimizer.minimize(self.loss, name='train_op')
    
    def metrics(self):
        pass
    
    def save_graph(self, save_path):
        with self.G.as_default():
            tf.train.export_meta_graph(filename=str(save_path)+'.meta')
    
    def load_graph(self, save_path):
        self.G = tf.Graph()
        with self.G.as_default():
            # Actually load the graph
            saver = tf.train.import_meta_graph(save_path+'.meta')
            
            # Rebind attributes
            self.input = self.G.get_tensor_by_name('input:0')
            self.output = self.G.get_tensor_by_name('output:0')
            self.current_level = 0
            self.tensors_above = []
            self.loss = self.G.get_tensor_by_name('loss:0')
            self.labels = self.G.get_tensor_by_name('labels:0')
            self.reg_param = self.G.get_tensor_by_name('reg_param:0')
            try:
                self.learning_rate = self.G.get_tensor_by_name('learning_rate:0')
                self.optimizer_placeholders = tf.get_collection('optimizer_placeholders')
                self.train_op = self.G.get_operation_by_name('train_op')
            except:
                print('Warning: no optimizer found')
        return saver
    
    def data_augmentation(self, X, Y):
        ''' Performs data augmentation on input batches. Just returns identity for now. '''
        return X, Y
    
    def train(self, X_train, Y_train, X_val, Y_val, max_epochs, batch_size, learning_rate_init, reg_param=0, learning_rate_decay_type='inverse', learning_rate_decay_parameter=10, early_stopping=True, save_path='./UNet', reset_parameters=False, check_val_every_n_batches=5, seed=0, data_on_GPU=True):
        '''
        Trains the network on the given data, provided as numpy arrays. It is assumed all preprocessing has already been done, including shuffling and splitting of the data into training/validation sets.
        '''
        
        # (Re)load base graph from file
        print('Loading graph...')
        saver = self.load_graph(save_path)
        
        with self.G.as_default():
            
            print('Inserting data augmentation operations...')
            # Load data onto GPU, replace the input placeholder with an index into the data on the GPU (if applicable)
            m_train, height, width, n_channels = X_train.shape
            m_val = X_val.shape[0]
            m = m_train + m_val
            n_classes = Y_train.shape[-1]
            if data_on_GPU:
                X_train_t = tf.constant(X_train, dtype=tf.uint8)
                X_val_t = tf.constant(X_val, dtype=tf.uint8)
                Y_train_t = tf.constant(Y_train, dtype=tf.bool)
                Y_val_t = tf.constant(Y_val, dtype=tf.bool)
                del X_train, X_val, Y_train, Y_val
                train_idx = tf.placeholder_with_default([0], shape=[None])
                X_train_t = tf.gather(X_train_t, train_idx, axis=0)
                Y_train_t = tf.gather(Y_train_t, train_idx, axis=0)
            else:
                X_train_t = tf.placeholder_with_default(np.zeros([0,height,width,n_channels], dtype=np.uint8), shape=self.input.shape, name='X_train_input')
                X_val_t = tf.placeholder_with_default(np.zeros([0,height,width,n_channels], dtype=np.uint8), shape=self.input.shape, name='X_val_input')
                Y_train_t = tf.placeholder_with_default(np.zeros([0,height,width,n_classes], dtype=np.bool), shape=self.labels.shape, name='Y_train_input')
                Y_val_t = tf.placeholder_with_default(np.zeros([0,height,width,n_classes], dtype=np.bool), shape=self.labels.shape, name='Y_val_input')
            # Insert data augmentation steps to graph
            train_or_val_idx = tf.placeholder(dtype=tf.int32, shape=[None])
            X_train_aug, Y_train_aug = self.data_augmentation(X_train_t, Y_train_t)
            X = tf.cast(tf.gather(tf.concat([X_train_aug, X_val_t], axis=0), train_or_val_idx), tf.float32)
            Y = tf.cast(tf.gather(tf.concat([Y_train_aug, Y_val_t], axis=0), train_or_val_idx), tf.float32)
            ge.swap_ts([X, Y], [self.input, self.labels]) # Use X and Y from now on!
            
            # Write to log file
            with open(str(save_path)+'.log', 'w+') as fo:
                fo.write('Training log\n\n')
                fo.write('Dataset metrics:\n')
                fo.write('Training data shape: {}\n'.format(X_train.shape))
                fo.write('Validation set size: {}\n'.format(m_val))
#                fo.write('X_mean: {}\n'.format(X_mean))
#                fo.write('X_std: {}\n\n'.format(X_std))
                fo.write('Hyperparameters:\n')
                fo.write('Batch size: {}\n'.format(batch_size))
                fo.write('Learning rate: {}\n'.format(learning_rate_init))
                fo.write('Learning rate annealed every N epochs: {}\n'.format(learning_rate_decay_parameter))
                fo.write('Learning rate anneal type: {}\n'.format(learning_rate_decay_type))
#                fo.write('Stepped anneal: {}\n'.format(STEPPED_ANNEAL))
#                fo.write('Regularization type: {}\n'.format(REGULARIZATION_TYPE))
                fo.write('Regularization parameter: {}\n'.format(reg_param))
#                fo.write('Input noise variance: {:.2f}\n'.format(INPUT_NOISE_MAGNITUDE**2))
#                fo.write('Weight noise variance: {:.2f}\n'.format(WEIGHT_NOISE_MAGNITUDE**2))
#                for n in range(1,len(KEEP_PROB)+1):
#                    fo.write('Dropout keep prob. group {}: {:.2f}\n'.format(n, KEEP_PROB[n]))
                fo.write('Logging frequency: {} global steps\n'.format(check_val_every_n_batches))
                fo.write('Random seed: {}\n'.format(seed))
                fo.write('\nNotes:\n')
            
            # Initialize control flow variables and logs
#            max_val_accuracy = -1
#            max_val_conf = -1
            best_val_loss = np.inf
            global_step = 0
#            with open(str(save_path)+'_val_accuracy.log', 'w+') as fo:
#                fo.write('')
            with open(str(save_path)+'_val_loss.log', 'w+') as fo:
                fo.write('')
#            with open(str(save_path)+'_val_confidence.log', 'w+') as fo:
#                fo.write('')
#            with open(str(save_path)+'_train_accuracy.log', 'w+') as fo:
#                fo.write('')
            with open(str(save_path)+'_train_loss.log', 'w+') as fo:
                fo.write('')
#            with open(str(save_path)+'_train_confidence.log', 'w+') as fo:
#                fo.write('')
            with open(str(save_path)+'_learning_rate.log', 'w+') as fo:
                fo.write('')
            
            # Start tensorflow session, reset_parameters or reload checkpoint
            print('Starting tensorflow session...')
            with tf.Session() as sess:
                if reset_parameters:
                    saver = tf.train.Saver()
                    sess.run(tf.global_variables_initializer())
                else:
                    saver.restore(sess, save_path)
                
                uninitialized_vars = []
                for var in tf.global_variables():
                    try:
                        sess.run(var)
                    except tf.errors.FailedPreconditionError:
                        uninitialized_vars.append(var)
                
                sess.run(tf.variables_initializer(uninitialized_vars))
                
                # Iterate over training epochs
                best_val_loss = np.inf
                for epoch in range(max_epochs):
                    if learning_rate_decay_type == 'inverse':
                        learning_rate = learning_rate_init/(1+epoch/learning_rate_decay_parameter)
                    elif learning_rate_decay_type == 'constant':
                        learning_rate = learning_rate_init
                    elif learning_rate_decay_type == 'exponential':
                        learning_rate = learning_rate_init*np.exp(-epoch/learning_rate_decay_parameter)
                    else:
                        raise Exception('Unknown learning rate decay function')
                    
                    # Iterate over batches:
                    n_batches = math.ceil(m_train/batch_size)
                    for b in range(n_batches):
                        train_idx_i = b*batch_size
                        train_idx_f = min((b+1)*batch_size, m_train)
                        
                        if data_on_GPU:
                            feed_dict={train_idx:range(train_idx_i, train_idx_f+1), train_or_val_idx:range(train_idx_f-train_idx_i), self.learning_rate:learning_rate, self.reg_param:reg_param}
                        else:
                            feed_dict={X_train_t:X_train[train_idx_i:train_idx_f], Y_train_t:Y_train[train_idx_i:train_idx_f], train_or_val_idx:range(train_idx_f-train_idx_i), self.learning_rate:learning_rate, self.reg_param:reg_param}
                        train_loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                        print('Epoch {}, batch {}/{}: loss={:.3e}'.format(epoch+1, b, n_batches, train_loss))
                        if np.isnan(train_loss) or np.isinf(train_loss):
                            print('Detected nan, exiting training')
                            quit()
                            exit()
                            break
                        
                        if (global_step % check_val_every_n_batches) == 0:
                            if data_on_GPU:
                                feed_dict = {train_or_val_idx:range(1,m_val+1), self.reg_param:reg_param}
                            else:
                                feed_dict = {X_val_t:X_val, Y_val_t:Y_val, train_or_val_idx:range(m_val), self.reg_param:reg_param}
                            val_loss = sess.run(self.loss, feed_dict=feed_dict)
                            if early_stopping and (val_loss < best_val_loss):
                                best_val_loss = val_loss
                                print('New best validation loss: {:.3e}! Saving...'.format(val_loss))
                                saver.save(sess, save_path, write_meta_graph=False)
                            
                            # Write to logs everytime validation set run
                            with open(str(save_path)+'_train_loss.log', 'a') as fo:
                                fo.write(str(train_loss)+'\n')
#                                with open(str(save_path)+'_train_accuracy.log', 'a') as fo:
#                                    fo.write(str(train_accuracy)+'\n')
#                                with open(str(save_path)+'_train_confidence.log', 'a') as fo:
#                                    fo.write(str(train_conf)+'\n')
#                                with open(str(save_path)+'_val_accuracy.log', 'a') as fo:
#                                    fo.write(str(val_accuracy)+'\n')
                            with open(str(save_path)+'_val_loss.log', 'a') as fo:
                                fo.write(str(val_loss)+'\n')
#                                with open(str(save_path)+'_val_confidence.log', 'a') as fo:
#                                    fo.write(str(val_conf)+'\n')
                            with open(str(save_path)+'_learning_rate.log', 'a') as fo:
                                fo.write(str(learning_rate)+'\n')
                            u.plot_metrics(str(save_path))
                                
                        global_step += 1
    
    def predict(self):
        pass


# Testing to make sure everything runs smoothly
if __name__ == '__main__':
    un = UNet()
    un.convolution(3,1,5, activation='relu')
    un.squeeze_convolution(3,2,4)
    un.convolution(3,1,7)
    un.squeeze_convolution(3,2,19)
    un.convolution(3,1,31)
    un.stretch_transpose_convolution(3,2,40)
    un.convolution(3,1,2)
    un.stretch_transpose_convolution(3,2,15)
    un.convolution(1,1,1, activation='identity')
    un.add_loss('xentropy')
    un.add_optimizer('adam')
    un.save_graph('./UNet')
    
    X_train = np.random.randn(100,25,25,4)
    X_val = np.random.randn(5,25,25,4)
    Y_train = (np.random.randn(100,25,25,1)>0).astype(int)
    Y_val = (np.random.randn(5,25,25,1)).astype(int)
    un.train(X_train, Y_train, X_val, Y_val, max_epochs=100, batch_size=10, learning_rate_init=1e-3, learning_rate_decay_type='inverse', data_on_GPU=False, reset_parameters=True, early_stopping=True, save_path='./UNet')















