#! usr/bin/pyhon
'''
Coded by luke on 30th June 2017
Aiming to get to know the tensorboard a bit more

'''
import tensorflow as tf
import numpy as np

## Create the real data
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5 + noise

# define the add_layer
def add_layer(inputs,input_size,output_size,n_layer,activation_function=None):
	# Add one more layer and return the output of this layer
	layer_name ='layer%s'%n_layer ## define a new var
	## and so on..
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([input_size,output_size]),name='W')
		# tf.histogram_summary (layer_name+'/weights',Weights)
			tf.summary.histogram(layer_name+'weights',Weights)
		with tf.name_scope('biases'):
			Biases = tf.Variable(tf.zeros([1,output_size]),name='b')
			tf.summary.histogram(layer_name+'/biases',Biases)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs,Weights),Biases)
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)

		tf.summary.histogram(layer_name+'/outputs',outputs)
	return outputs

# Define the placeholder for the layer
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32,[None,1],name='x_input')
	ys = tf.placeholder(tf.float32,[None,1],name='y_input')
# Define the hidden layer
l1 = add_layer(xs,1,10,1,activation_function=tf.nn.relu)

# define the output layer
prediction = add_layer(l1,10,1,2,activation_function=None)


with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
	tf.summary.scalar('loss',loss)
with tf.name_scope('train'):	
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('log/',sess.graph)
	sess.run(init)
	for i in range(1000):
		#train 
		sess.run(train_step,feed_dict={xs: x_data,ys: y_data})
		if i%50 == 0:
			rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
			writer.add_summary(rs,i)
		
