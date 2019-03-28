"""A neural network implementation for classifciation."""
import numpy as np
import tensorflow as tf
from functools import partial
import math

tf.reset_default_graph()

class NN:
    """A neural network implementation for classification in TensorFlow.
    
    To operate use methods fit(X, Y) and test(X, Y).
    """
    def __init__(self,
                 data_dim = 200,
                 hidden_layer_size = [300],
                 init_weight_func_hidden=partial(tf.truncated_normal, stddev=math.sqrt(2.0), seed=0),
                 init_bias_func_hidden=partial(tf.truncated_normal, stddev=math.sqrt(2.0), seed=0),
                 init_weight_func_output=partial(tf.truncated_normal, stddev=math.sqrt(2.0), seed=0),
                 init_bias_func_output=partial(tf.truncated_normal, stddev=math.sqrt(2.0), seed=0),
                 activation_func_hidden=tf.nn.sigmoid,
                 dropout_keep_rate = 1.0,            
                 optimizer = 'GradientDescent',
                 initial_learning_rate = 0.001,
                 decay_rate = 0.99,
                 decay_freq = 1000,
                 training_logging_interval = 100,
                 use_tensorboard = False):
        
        if verbosity_level > 3: print('[NN] Initialising')
        self.data_dim = data_dim         # data_dim refers to the number of features
        self.hidden_layer_size = hidden_layer_size
        self.training_logging_interval = training_logging_interval
        self.use_tensorboard = use_tensorboard
        self.dropout_keep_rate = dropout_keep_rate
        self.initial_learning_rate = initial_learning_rate

######################################
        #nn specifications
        self.init_weight_func_hidden = init_weight_func_hidden
        self.init_bias_func_hidden = init_bias_func_hidden
        self.init_weight_func_output = init_weight_func_output
        self.init_bias_func_output = init_bias_func_output
        self.activation_func_hidden = activation_func_hidden 
######################################

        # path to write event files
        self.logs_path = '/tmp/tensorflow_logs/eeg'

        ###---SETUP---###
        self.sess = tf.Session()

        # initialize Placeholders
        self.x = tf.placeholder(tf.float32, shape=(None, self.data_dim), name='input')
        self.label = tf.placeholder(tf.float32, shape=(None, 3), name='label')
        self.keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob')

	#save of the net
        self.var_list = []

        # build tensor graph
        with tf.name_scope('hidden_layers'):
            hiddens = []
            dropped = []
            input_size = self.data_dim 
            input_data = self.x
            for h in self.hidden_layer_size:
                hiddens.append(self._nn_layer(input_data, input_size, h))
                dropped.append(tf.nn.dropout(hiddens[-1], self.keep_prob_pl))
                input_data = dropped[-1]
                input_size = h

        with tf.name_scope('output'):
            self.output = self._output_layer(dropped[-1]) # calulates the absolute predicted output (without probabilties)

        # loss  operand
        with tf.name_scope('optimizer_and_loss'): 
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.label)
            self.loss = tf.reduce_mean(cross_entropy)
            # optimizer
            if optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(self.initial_learning_rate).minimize(self.loss)
            elif optimizer == 'GradientDescent':
                global_step = tf.Variable(0)
                learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_freq, decay_rate, staircase=True)
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
            else:
                raise ValueError('Can be either Adam or GradientDescent optimizer.')
            
            # accuracy
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # TensorBoard summaries
        if self.use_tensorboard:
            self.loss_summary = tf.summary.scalar("loss", self.loss)
            self.misscl_summary = tf.summary.scalar("accuracy", self.accuracy)
            self.merged_summary_op = tf.summary.merge_all()
        
        #self.saver = tf.train.Saver(self.var_list)
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        if self.use_tensorboard:
            self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())

    ###---NN-BUILDING-BLOCKS---###

    def _nn_layer(self, input_tensor, input_size, hidden_layer_size):
        weights = tf.Variable(self.init_weight_func_hidden(shape=[input_size, hidden_layer_size]))
        biases = tf.Variable(self.init_bias_func_hidden(shape=[hidden_layer_size]))
        preactivate = tf.add(tf.matmul(input_tensor, weights), biases)
        activation = self.activation_func_hidden(preactivate)

        self.var_list.append(weights)
        self.var_list.append(biases)
        return activation
    
    def _output_layer(self, input_tensor):
        # produce output that represents 3 categories
        weights = tf.Variable(self.init_weight_func_output(shape=[self.hidden_layer_size[-1], 3]))   
        biases = tf.Variable(self.init_bias_func_output(shape=[3])) 
        activation = tf.add(tf.matmul(input_tensor, weights), biases)

        self.var_list.append(weights)
        self.var_list.append(biases)
        return activation

    ###---FUNCTIONS---###
    def _run_training(self, X, label):
        if verbosity_level > 3: print('[NN] Training starts')
        feed_dict = {self.x: X, self.label: label, self.keep_prob_pl: self.dropout_keep_rate}

        if self.use_tensorboard:
            summaries = self.sess.run(self.merged_summary_op, feed_dict=feed_dict)
            self.summary_writer.add_summary(summaries, epoch)
            tf.train.SummaryWriter.flush(self.summary_writer)                  
        
        self.sess.run(self.optimizer, feed_dict=feed_dict)
        return #cost

    def _run_testing(self, X, label):
        if verbosity_level > 3: print('[NN] Testing starts')
        feed_dict = {self.x: X, self.label: label, self.keep_prob_pl:1.0}
        cost, self.accuracy_out = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        if verbosity_level > 3: print('[NN] Testing finished; error=%.9f\n' % cost)
        return cost
        
    ###---INTERFACE---###
    def fit(self, X, label, verbose=False):
        #train
        self._run_training(X, label)
        
        #training accuracy
        feed_dict = {self.x: X, self.label: label, self.keep_prob_pl: 1.0}
        self.accuracy_out = self.accuracy.eval(session=self.sess, feed_dict=feed_dict) 
        return 
        
    def test(self, X, label):
        return self._run_testing(X, label)

    def save(self, path):
        self.saver.save(self.sess, path)
        return

    def load(self, path): 
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('tf_saves/')

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess,  path)
        return
	
    def close(self):
        return self.sess.close()
