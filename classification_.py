#! usr/bin/python
'''
Coded by luke on 30th June 2017
Aiming to get familiar with classification in tensorflow
'''

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10
#from add_layer_ import add_layer

mnist = input_data.read_data_sets('MNIST_data',one_hot = True)


def add_layer(inputs,input_size,output_size,activation_function=None):
	# Add one more layer and return the output of this layer
	Weights = tf.Variable(tf.random_normal([input_size,output_size]),name='W')
	Biases = tf.Variable(tf.zeros([1,output_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs,Weights) + Biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)

	return outputs

# Define the placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

# Add output layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

# The error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

# Train step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre = sess.run(prediction,feed_dict={xs:v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
	return result


sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch_x,batch_y = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={xs:batch_x,ys:batch_y})
	if i%50 ==0:
		print(compute_accuracy(mnist.test.images,mnist.test.labels))
	
