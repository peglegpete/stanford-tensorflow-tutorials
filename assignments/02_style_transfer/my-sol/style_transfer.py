""" Implementation in TensorFlow of the paper 
A Neural Algorithm of Artistic Style (Gatys et al., 2016) 

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

For more details, please read the assignment handout:
https://docs.google.com/document/d/1FpueD-3mScnD0SJQDtwmOb1FrSwo1NGowkXzMwPoLH4/edit?usp=sharing
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import tensorflow as tf

import load_vgg
import utils

def setup():
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('outputs')

class StyleTransfer(object):
    def __init__(self, content_img, style_img, img_width, img_height):
        '''
        img_width and img_height are the dimensions we expect from the generated image.
        We will resize input content image and input style image to match this dimension.
        Feel free to alter any hyperparameter here and see how it affects your training.
        '''
        self.img_width = img_width
        self.img_height = img_height
        self.content_img = utils.get_resized_image(content_img, img_width, img_height)
        self.style_img = utils.get_resized_image(style_img, img_width, img_height)
        self.initial_img = utils.generate_noise_image(self.content_img, img_width, img_height)

        ###############################
        ## 
        ## create global step (gstep) and hyperparameters for the model
        self.content_layer = 'conv4_2'
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        # content_w, style_w: corresponding weights for content loss and style loss
        self.content_w = 1 #alpha
        self.style_w = 1 #beta
        # style_layer_w: weights for different style layers. deep layers have more weights
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0] #[4., 3., 1.5, 1., .5]
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step') # global step
        self.lr = 0.1
        ###############################

    def create_input(self):
        '''
        We will use one input_img as a placeholder for the content image, 
        style image, and generated image, because:
            1. they have the same dimension
            2. we have to extract the same set of features from them
        We use a variable instead of a placeholder because we're, at the same time, 
        training the generated image to get the desirable result.

        Note: image height corresponds to number of rows, not columns.
        '''
        with tf.variable_scope('input') as scope:
            self.input_img = tf.get_variable('in_img', 
                                        shape=([1, self.img_height, self.img_width, 3]),
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
    def load_vgg(self):
        '''
        Load the saved model parameters of VGG-19, using the input_img
        as the input to compute the output at each layer of vgg.

        During training, VGG-19 mean-centered all images and found the mean pixels
        to be [123.68, 116.779, 103.939] along RGB dimensions. We have to subtract
        this mean from our images.

        '''
        self.vgg = load_vgg.VGG(self.input_img)
        self.vgg.load()
        self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

    def _content_loss(self, P, F):
        ''' Calculate the loss between the feature representation of the
        content image and the generated image.
        
        Inputs: 
            P: content representation of the content image
            F: content representation of the generated image
            Read the assignment handout for more details

            Note: Don't use the coefficient 0.5 as defined in the paper.
            Use the coefficient defined in the assignment handout.
        '''
        ###############################
        ## 
        #print(tf.reduce_sum((F-P)).dtype)
        #print(tf.reduce_prod(tf.shape(P)).dtype)
        factor = tf.cast(1/(4*tf.reduce_prod(tf.shape(P))), tf.float32)
        self.content_loss = factor*tf.reduce_sum(tf.square(F-P))
        ###############################
        
    def _gram_matrix(self, F, N, M):
        """ Create and return the gram matrix for tensor F
            Hint: you'll first have to reshape F
        """
        ###############################
        F_reshaped = tf.reshape(F, [M, N])
        return tf.transpose(F_reshaped) @ F_reshaped
        
        #tf.tensordot(F_reshaped, tf.transpose(F_reshaped), axes = 1)
        
        #F_reshaped @ tf.transpose(F_reshaped)
        ###############################

    def _single_style_loss(self, a, g):
        """ Calculate the style loss at a certain layer
        Inputs:
            a is the feature representation of the style image at that layer
            g is the feature representation of the generated image at that layer
        Output:
            the style loss at a certain layer (which is E_l in the paper)

        Hint: 1. you'll have to use the function _gram_matrix()
            2. we'll use the same coefficient for style loss as in the paper
            3. a and g are feature representation, not gram matrices
        """
        ###############################
        ## 
        dims = tf.shape(a)
        #self.dims = tf.shape(a)
        N = dims[-1]
        #self.N = tf.cast(dims[-1], tf.float64)
        M = tf.reduce_prod(dims[:-1])
        #self.M = tf.cast(tf.reduce_prod(dims[:-1]), tf.float64)
        A = self._gram_matrix(a, N, M)
        G = self._gram_matrix(g, N, M)
        N = tf.cast(N, tf.float64)
        M = tf.cast(M, tf.float64)
        factor = tf.cast((1./(4.*(N*M)**2)), tf.float32)
        #self.factor = tf.cast(1./(4.*(self.N*self.M)**2), tf.float32)#tf.cast((1/(4*(N*M)**2)), tf.float32)
        return factor*tf.reduce_sum(tf.square(G-A))
        ###############################

    def _style_loss(self, A, G):
        """ Calculate the total style loss as a weighted sum 
        of style losses at all style layers
        Hint: you'll have to use _single_style_loss()
        """
        ###############################
        ##
        self.style_loss = 0
        for a, g, w in zip(A, G, self.style_layer_w):
        	self.style_loss += w*self._single_style_loss(a, g)
        ###############################

    def losses(self):
        with tf.variable_scope('losses') as scope:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                # assign content image to the input variable
                sess.run(self.input_img.assign(self.content_img)) 
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            self._content_loss(content_img_content, gen_img_content)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(self.input_img.assign(self.style_img))
                
                gen_img_styles = [getattr(self.vgg, layer) for layer in self.style_layers]
                style_layers = sess.run(gen_img_styles)                              
            self._style_loss(style_layers, gen_img_styles)

            ##########################################
            ## TO DO: create total loss. 
            ## Hint: don't forget the weights for the content loss and style loss
            self.total_loss = self.content_w*self.content_loss + self.style_w*self.style_loss
            ##########################################

    def optimize(self):
        ###############################
        ## create optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.total_loss)
        ###############################

    def create_summary(self):
        ###############################
        ## TO DO: create summaries for all the losses
        ## Hint: don't forget to merge them
        with tf.name_scope("summaries"):
        	tf.summary.scalar("content loss", self.content_loss)
        	tf.summary.scalar("style loss", self.style_loss)
        	tf.summary.scalar("total loss", self.total_loss)
        	self.summary_op = tf.summary.merge_all()
        ###############################


    def build(self):
        self.create_input()
        print('machine.build(): create input done')
        self.load_vgg()
        print('machine.build(): load vgg done')
        self.losses()
        print('machine.build(): build losses done')
        self.optimize()
        print('machine.build(): build optimizer done')
        self.create_summary()
        print('machine.build(): build summary done')
        

    def train(self, n_iters):
        skip_step = 1
        with tf.Session() as sess:
            
            ###############################
            ## TO DO: 
            ## 1. initialize your variables
            sess.run(tf.global_variables_initializer())
            ## 2. create writer to write your grapp
            utils.safe_mkdir('graphs')
            writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
            ###############################
            
            sess.run(self.input_img.assign(self.initial_img))

            ###############################
            ## TO DO: 
            ## 1. create a saver object
            saver = tf.train.Saver()
            ## 2. check if a checkpoint exists, restore the variables
            utils.safe_mkdir('checkpoints')
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            ##############################

            initial_step = self.gstep.eval()
            
            start_time = time.time()
            for index in range(initial_step, n_iters):
                if index >= 5 and index < 20:
                    skip_step = 10
                elif index >= 20:
                    skip_step = 20
                    
                print(index) #TODO: debugg print
                sess.run(self.opt)
                
                if (index + 1) % skip_step == 0:
                    ###############################
                    ## TO DO: obtain generated image, loss, and summary
                    #print('A') #TODO: debugg print
                    gen_image, total_loss, summary = sess.run([self.input_img, self.total_loss, self.summary_op])
                    #gen_image = sess.run(self.input_img)
                    style_loss = sess.run(self.style_loss)
                    content_loss = sess.run(self.content_loss)
                    #total_loss = sess.run(self.total_loss)
                    
                    #N, M, dims, factor = sess.run([self.N, self.M, self.dims, self.factor])
                    #summary = sess.run(self.summary_op)
                    
                    
                    
                    ###############################
                    #print('B') #TODO: debugg print
                    # add back the mean pixels we subtracted before
                    gen_image = gen_image + self.vgg.mean_pixels 
                    writer.add_summary(summary, global_step=index)
                    print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                    #print('N:'+str(N)+' M:'+str(M)+' dims:'+str(dims)+' factor:'+str(factor))
                    print('   Loss: {:5.1f}'.format(total_loss))
                    print('   Style loss: {:5.1f}'.format(style_loss))
                    print('   Content loss: {:5.1f}'.format(content_loss))
                    print('   Took: {} seconds'.format(time.time() - start_time))
                    start_time = time.time()

                    filename = 'outputs/%d.png' % (index)
                    utils.save_image(filename, gen_image)

                    if (index + 1) % 20 == 0:
                        ###############################
                        ## save the variables into a checkpoint
                        saver.save(sess, 'checkpoints/', index)
                        ###############################
                 
            writer.close()

if __name__ == '__main__':
    setup()
    machine = StyleTransfer('content/portrait-content.jpg', 'styles/portrait-style.jpg', 301, 401)
    print('machine init done')
    machine.build()
    print('machine build done')
    machine.train(300)
    print('machine train done')
