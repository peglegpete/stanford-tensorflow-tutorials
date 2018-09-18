""" Using convolutional net on MNIST dataset of handwritten digits
MNIST dataset: http://yann.lecun.com/exdb/mnist/
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 07
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 

import tensorflow as tf

import utils

def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    '''
    A method that does convolution + relu on inputs
    '''
    biases = filters[1]
    
    with tf.name_scope(scope_name):
        conv = tf.nn.conv2d(
            inputs,
            filters[0],
            stride,
            padding
            )
        conv_out = conv + biases
        relu_out = tf.nn.relu(conv_out)
    return relu_out

def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    '''A method that does max pooling on inputs'''
    with tf.name_scope(scope_name):
        max_out = tf.nn.max_pool(
            inputs,
            ksize,
            stride,
            padding
            )
    return max_out

def fully_connected(inputs, in_dim, out_dim, scope_name='fc'):
    '''
    A fully connected linear layer on inputs
    '''
    dim = tf.reduce_prod(tf.shape(inputs)[1:])
    inputs = tf.reshape(inputs, [-1, dim])
    print('fc inputs shape', tf.shape(inputs))
    with tf.name_scope(scope_name):
        w = tf.get_variable(name = scope_name+'_weights', initializer = tf.random_normal([in_dim, out_dim], 0,0.01))
        b = tf.get_variable(name = scope_name+'_biases', initializer = tf.zeros((out_dim,)))
    return tf.matmul(inputs, w) + b

class ConvNet(object):
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = 10000

    def get_data(self):
        with tf.name_scope('data'):
            train_data, test_data = utils.get_mnist_dataset(self.batch_size)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                                   train_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            # reshape the image to make it work with tf.nn.conv2d

            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)    # initializer for train_data

    def inference(self):
        '''
        Build the model according to the description we've shown in class
        '''
        #############################
        ########## TO DO ############
        #############################
        #first conv + relu layer
        scope_name = 'conv1'
        with tf.name_scope(scope_name):
            filters1 = [tf.get_variable(name = scope_name+'_filters1', initializer = tf.random_normal([5,5,1,32], 0,0.01)), tf.get_variable(name = scope_name+'_biases', initializer = tf.zeros((32,)))]
            conv1_out = conv_relu(self.img, filters = filters1, k_size=5, stride = [1,1,1,1], padding='SAME', scope_name = scope_name)
        
        #first maxpool layer
        scope_name = 'pool1'
        max1_out = maxpool(conv1_out, ksize=[1,2,2,1], stride=[1,2,2,1], padding='VALID', scope_name=scope_name)
        
        #second conv + relu layer
        scope_name = 'conv2'
        with tf.name_scope(scope_name):
            filters2 = [tf.get_variable(name = scope_name+'_filters2', initializer = tf.random_normal([5,5,32,64], 0,0.01)), tf.get_variable(name = scope_name+'_biases', initializer = tf.zeros((64,)))]
            conv2_out = conv_relu(max1_out, filters = filters2, k_size=5, stride = [1,1,1,1], padding='SAME', scope_name = scope_name)
        
        #second maxpool layer
        scope_name = 'pool2'
        max2_out = maxpool(conv2_out, ksize=[1,2,2,1], stride=[1,2,2,1], padding='VALID', scope_name=scope_name)
        
        #first fc
        scope_name = 'fully1'
        fc1_out = fully_connected(max2_out, 7*7*64, 1024, scope_name=scope_name)
        fc1_out = tf.nn.relu(fc1_out)
        
        #second fc
        scope_name = 'fully2'
        self.logits = fully_connected(fc1_out, 1024, 10, scope_name=scope_name)

    def loss(self):
        '''
        define loss function
        use softmax cross entropy with logits as the loss function
        tf.nn.softmax_cross_entropy_with_logits
        softmax is applied internally
        don't forget to compute mean cross all sample in a batch
        '''
        #############################
        ########## TO DO ############
        #############################
        loss = tf.nn.softmax_cross_entropy_with_logits(
            _sentinel=None,
            labels=self.label,
            logits=self.logits,
            dim=-1,
            name=None
            )
        self.loss = tf.reduce_mean(loss)
    
    def optimize(self):
        '''
        Define training op
        using Adam Gradient Descent to minimize cost
        Don't forget to use global step
        '''
        #############################
        ########## TO DO ############
        #############################
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(loss = self.loss, global_step = self.gstep)

    def summary(self):
        '''
        Create summaries to write on TensorBoard
        Remember to track both training loss and test accuracy
        '''
        #############################
        ########## TO DO ############
        #############################
        tf.summary.scalar("training-loss", self.loss)
        tf.summary.scalar("test-acc", self.accuracy)
        self.summary_op = tf.summary.merge_all()
        
    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        '''
        Build the computation graph
        '''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init) 
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, 'checkpoints/convnet_starter/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        '''
        The train function alternates between training one epoch and evaluating
        '''
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_starter')
        writer = tf.summary.FileWriter('./graphs/convnet_starter', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_starter/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=15)
