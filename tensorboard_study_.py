#! usr/bin/python
'''
Coded by luke on 30th June 2017
Aiming to study the tensorflow tensorboard for the visualization

'''

from __future__ import print_function
import tensorflow as tf


# Define the add_layer for the NN
def add_layer(inputs,input_size,output_size,activation_function=None):
	'Add one layer and return the output of this layer'
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([input_size,output_size]),name='W')
		with tf.name_scope('biase'):
			Biases = tf.Variable(tf.zeros([1,output_size])+0.1,name='b')
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs,Weights)+Biases
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)

		return outputs

# Define placeholder for inputs to network
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32,[None,1],name='x_input')
	ys = tf.placeholder(tf.float32,[None,1],name='y_input')

# Add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# Add the output layer
prediction = add_layer(l1,10,1,activation_function=None)

# The error between prediction and real data
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

# The train process
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

# tf.train.SummaryWriter soon be deprecated using following
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0])<1: # tensorflow version <0.12
	writer = tf.train.SummaryWriter('log/',sess.graph)
else: # tensorflow version > =0.12
	writer = tf.summary.FileWriter('log/',sess.graph)

init = tf.global_variables_initializer()

sess.run(init)

