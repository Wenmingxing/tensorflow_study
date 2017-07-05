#! usr/bin/python

'''
Coded by luke on 3rd July 2017
Aiming to familiarize myself with the RNN
RNN for classification
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1) # Set the random seed

# Prepare for the data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)



# hyperparameters
lr = 0.001 # learning rate
#training_iters = 100000 # train step upper limit
batch_size = 128 
n_inputs = 28 # MNIST data input (img shape 28*28) 28 columns
n_steps = 28  # time steps, the number of rows for one picture
n_hidden_units = 128 # neurons in hidden layer
n_classes = 10 # MNIST classed (0-9 digits)

# Define the placeholder for this model
xs = tf.placeholder(tf.float32,[None,n_steps,n_inputs],name="inputs")
ys = tf.placeholder(tf.float32,[None,n_classes],name="outputs")

# define the initialization values for the weights and biases
Weights = {'in':tf.Variable(tf.random_uniform([n_inputs,n_hidden_units],-1.0,1.0),name="in_w"),
	   'out':tf.Variable(tf.random_uniform([n_hidden_units,n_classes],-1.0,1.0),name="out_w")}

Biases = {
	'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units]),name="in_b"),
	'out':tf.Variable(tf.constant(0.1,shape=[n_classes]),name="out_b")

}

# Define the RNN main function
def RNN(x,weights,bias):
	# hidden_layer for the input
	# x:(batch_size,28,28)
	with tf.name_scope('inlayer'):
		x = tf.reshape(x,[-1,n_inputs])
		x_in = tf.matmul(x,weights['in']) + bias['in']
		x_in = tf.reshape(x_in,[-1,n_steps,n_hidden_units])

	# RNN cell
	with tf.name_scope("RNN_CELL"):
        	lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
        # _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # ouputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state)
        	outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, dtype=tf.float32)

		

	# Out layer
	with tf.name_scope('outlayer'):
		results = tf.matmul(states[1],weights['out']) + bias['out']

	return results

pred = RNN(xs,Weights,Biases)

# Cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# accuracy
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# Run 
import time
init = tf.global_variables_initializer()
epochs = 30

st = time.time()

with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter('logs/',sess.graph)
	

	batch = mnist.train.num_examples / batch_size
	for epoch in range(epochs):
		for i in range(int(batch)):
			batch_x,batch_y = mnist.train.next_batch(batch_size)
			batch_x = batch_x.reshape([batch_size,n_inputs,n_steps])
			sess.run(train_op,feed_dict={xs:batch_x,ys:batch_y})

		print ('epoch:',epoch+1, 'accuracy:', sess.run(accuracy,feed_dict={xs: mnist.test.images.reshape([-1, n_steps, n_inputs]),ys:mnist.test.labels}))
	end = time.time()
	print ('*' * 30)
	print ('training finish.\ncost time:',int(end-st), 'seconds\naccuracy:', sess.run(accuracy, feed_dict={xs: mnist.test.images.reshape([-1, n_steps, n_inputs]), ys: mnist.test.labels}))

