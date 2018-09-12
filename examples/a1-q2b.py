""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = '/home/donald/Documents/work/cs20/stanford-tensorflow-tutorials/examples/data/nomnist/notMNIST-to-MNIST/data'#'data/mnist'
#utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.batch(batch_size)

#############################
########## TO DO ############
#############################


# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y

d_input = tf.cast(train_data.output_shapes[0].dims[1], 'int32')
d_output = tf.cast(train_data.output_shapes[1].dims[1], 'int32')

d_intermediate_1 = 256
d_intermediate_2 = 256
d_intermediate_3 = 256

w_1 = tf.get_variable(name='w_1', initializer=tf.random_normal((d_input,d_intermediate_1), 0,0.01))
b_1 = tf.get_variable(name='b_1',initializer=tf.zeros((d_intermediate_1,)))

w_2 = tf.get_variable(name='w_2', initializer=tf.random_normal((d_intermediate_1, d_intermediate_2), 0,0.01))
b_2 = tf.get_variable(name='b_2',initializer=tf.zeros((d_intermediate_2,)))

w_3 = tf.get_variable(name='w_3', initializer=tf.random_normal((d_intermediate_2, d_intermediate_3), 0,0.01))
b_3 = tf.get_variable(name='b_3',initializer=tf.zeros((d_intermediate_3,)))

w = tf.get_variable(name='w', initializer=tf.random_normal((d_intermediate_3,d_output), 0,0.01))
b = tf.get_variable(name='b',initializer=tf.zeros((d_output,)))


#############################
########## TO DO ############
#############################


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
z_1 = tf.matmul(img,w_1) + b_1
a_1 = tf.layers.dropout(tf.nn.relu(z_1))

z_2 = tf.matmul(a_1,w_2) + b_2 + a_1
a_2 = tf.layers.dropout(tf.nn.relu(z_2))

z_3 = tf.matmul(a_2,w_3) + b_3 + a_2
a_3 = tf.layers.dropout(tf.nn.relu(z_3))

logits = tf.matmul(a_3, w) + b
#############################
########## TO DO ############
#############################


#print(tf.shape(logits))
# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
loss = tf.losses.softmax_cross_entropy(label, logits)
#tf.losses.softmax_cross_entropy(tf.one_hot(label, train[1].shape[0]), logits)
#############################
########## TO DO ############
#############################


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer().minimize(loss)
#############################
########## TO DO ############
#############################


# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
   
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs): 	
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))
    #print(total_correct_preds, n_test)
writer.close()
