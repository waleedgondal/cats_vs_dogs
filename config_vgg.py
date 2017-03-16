import tensorflow as tf
import numpy as np
import cPickle

class VGG16():
    
    """The class defines VGG16 model. The class initializes the network with pretrained weights for VGG16. 
    
    Parameters
    -----------
    n_labels: int
        An integer representing the number of output classes    
    weight_file_path: list of numpy arrays
        List of arrays that contain pretrained network parameters. The layers are parsed with respect
        to the respective layer name.
    
    Yields
    --------
    output : numpy array of float32
        The corresponding output scores. Shape (batchsize, num_labels)"""     
    
    
    
    def __init__(self, n_labels, weight_file_path = None):
        self.mean = tf.constant([108.64628601, 75.86886597, 54.34005737], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        self.n_labels = n_labels
        self.image_mean = [103.939, 116.779, 123.68]        
        self.epsilon = 1e-3
        
        assert (weight_file_path is not None), 'No pretrained weight file found'
        #if weight_file_path is not None:
        self.w = np.load(weight_file_path)


    def batch_norm_wrapper(self, inputs, is_training, decay = 0.999):
        
    """Implementation of batch normalization for training.
    The wrapper is taken from http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    which is a simpler version of Tensorflow's 'official' version. See:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
    
    inputs: tensor
        Tensor of layer output of size (Batchsize, height, width, channels)
    is_training: bool
        Indicator for training"""
    
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2]) # [0,1,2] for global normalization as mentioned in documentation
            
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, self.epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, self.epsilon)

    

     

    def inference( self, image, is_training, dropout=0.5):
        
        """ Defines the Standard VGG16 Network, proposed in Very Deep Convolutional Networks for 
        Large-Scale Image Recognition Simonyan et al. https://arxiv.org/abs/1409.1556
        
        Parameters
        ----------
        image: Tensor
            A tensor of shape (batchsize, height, width, channels)
        is_training: bool
            Indicator for training
        dropout: int
            Regularization factor in fully connected layers of the network.
            
        Yields
        --------
        output : numpy array of float32
            The corresponding output scores. Shape (batchsize, num_labels)"""
        
        with tf.variable_scope('conv1_1') as scope:
            
            w = self.w['conv1_1_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv1_1 = tf.nn.relu(bn, name='conv1_1')          
            print 'conv1_1w', self.conv1_1.get_shape()
      
    
        with tf.variable_scope('conv1_2') as scope:
            
            w = self.w['conv1_2_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv1_2 = tf.nn.relu(bn, name='conv1_2')                     
            print 'conv1_2w', self.conv1_2.get_shape()            

        pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        with tf.variable_scope('conv2_1') as scope:
            
            w = self.w['conv2_1_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv2_1 = tf.nn.relu(bn, name='conv2_1')
            print 'conv2_1w', self.conv2_1.get_shape()
                    

        with tf.variable_scope('conv2_2') as scope:
            
            w = self.w['conv2_2_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv2_2 = tf.nn.relu(bn, name='conv2_2')
            print 'conv2_2w', self.conv2_2.get_shape()
            

        pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        
        
        with tf.variable_scope('conv3_1') as scope:
            w = self.w['conv3_1_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv3_1 = tf.nn.relu(bn, name='conv3_1')
            print 'conv3_1w', self.conv3_1.get_shape()        

        with tf.variable_scope('conv3_2') as scope:
            
            w = self.w['conv3_2_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv3_2 = tf.nn.relu(bn, name='conv3_2')
            print 'conv3_2w', self.conv3_2.get_shape()  
            
        with tf.variable_scope('conv3_3') as scope:
            w = self.w['conv3_3_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv3_3 = tf.nn.relu(bn, name='conv3_3')
            print 'conv3_3w', self.conv3_3.get_shape() 

        pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        with tf.variable_scope('conv4_1') as scope:
            w = self.w['conv4_1_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv4_1 = tf.nn.relu(bn, name='conv4_1')
            print 'conv4_1w', self.conv4_1.get_shape()          

        with tf.variable_scope('conv4_2') as scope:
            w = self.w['conv4_2_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv4_2 = tf.nn.relu(bn, name='conv4_2')
            print 'conv4_2w', self.conv4_2.get_shape() 
            
        with tf.variable_scope('conv4_3') as scope:
            w = self.w['conv4_3_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv4_3 = tf.nn.relu(bn, name='conv4_3')
            print 'conv4_3w', self.conv4_3.get_shape() 

        pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')        
        
        
        with tf.variable_scope('conv5_1') as scope:
            w = self.w['conv5_1_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv5_1 = tf.nn.relu(bn, name='conv5_1')
            print 'conv5_1w', self.conv5_1.get_shape() 
            
        with tf.variable_scope('conv5_2') as scope:
            w = self.w['conv5_2_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv5_2 = tf.nn.relu(bn, name='conv5_2')
            print 'conv5_2w', self.conv5_2.get_shape()

        with tf.variable_scope('conv5_3') as scope:
            w = self.w['conv5_3_W']
            kernel = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            bn = self.batch_norm_wrapper(conv, is_training)
            self.conv5_3 = tf.nn.relu(bn, name='conv5_3')
            print 'conv5_3w', self.conv5_3.get_shape()
            
        pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5') 
            
        with tf.variable_scope('fc1') as scope:           
            
            w = self.w['fc6_W']
            fcw = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            b = self.w['fc6_b']
            fcb = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b))
            
            shape = int(np.prod(pool5.get_shape()[1:]))
            pool5_flat = tf.reshape(pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fcw), fcb)
            self.fc1 = tf.nn.relu(fc1l)
            self.fc1 = tf.nn.dropout(self.fc1, dropout)
            print 'fc1', self.fc1.get_shape()          

        with tf.variable_scope('fc2') as scope:
            
            w = self.w['fc7_W']
            fcw = tf.get_variable("W", shape=w.shape, initializer=tf.constant_initializer(w))
            b = self.w['fc7_b']
            fcb = tf.get_variable("b", shape=b.shape, initializer=tf.constant_initializer(b))            
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fcw), fcb)
            self.fc2 = tf.nn.relu(fc2l)
            self.fc2 = tf.nn.dropout(self.fc2, dropout)
            print 'fc2', self.fc2.get_shape()            

        with tf.variable_scope('fc3') as scope:
            
            w = tf.get_variable(
                    "W",
                    shape=[4096, self.n_labels],
                    initializer=tf.contrib.layers.xavier_initializer())            

            output = tf.matmul(self.fc2, w)            
        return output






