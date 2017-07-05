#! usr/bin/python
'''
Coded by luke on 30th June 2017 (finished on 3rd July)
Aiming to get familiar with drop_out

'''

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
#from add_layer import add_layer

#Prepare for the digits
digits = load_digits()
x = digits.data
y = digits.target

y = LabelBinarizer().fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3)


def add_layer(inputs,input_size,output_size,layer_name,activation_function=None):
	# Add one more layer and return the output for this layer
	Weights = tf.Variable(tf.random_normal([input_size,output_size]))
	Biases = tf.Variable(tf.zeros([1,output_size]) + 0.1)


	Wx_plus_b = tf.matmul(inputs,Weights) + Biases
	# the dropout 
	Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)

	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)

	tf.summary.histogram(layer_name + '/output',outputs)
	return outputs


# Define the placeholders for inputs to network
keep_prob = tf.placeholder(tf.float32)

xs = tf.placeholder(tf.float32,[None,64])
ys = tf.placeholder(tf.float32,[None,10])


# add output layer
l1 = add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)
prediction = add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax)

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))

tf.summary.scalar('loss',cross_entropy)
# Train method
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

# summary writer goes in here
train_writer = tf.summary.FileWriter('logs/train',sess.graph)
test_writer = tf.summary.FileWriter('logs/test',sess.graph)

# initialize all the variables
sess.run(tf.global_variables_initializer())

for i in range(500):
	# to determine the keeping probability
	sess.run(train_step,feed_dict={xs: x_train,ys:y_train,keep_prob:0.5})
	if i%50 ==0:
		# record loss
		train_result = sess.run(merged,feed_dict={xs:x_train,ys:y_train,keep_prob:1})
		test_result = sess.run(merged,feed_dict={xs:x_test,ys:y_test,keep_prob:1})
		train_writer.add_summary(train_result,i)
		test_writer.add_summary(test_result,i)




