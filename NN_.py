#! usr/bin/python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	Biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	Wx_plus_B = tf.matmul(inputs,Weights) + Biases
	if activation_function is None:
		outputs = Wx_plus_B
	else:
		outputs = activation_function(Wx_plus_B)

	return outputs

x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

h1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
prediction = add_layer(h1,10,1)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.show()
for i in range(1000):
	sess.run(train_step,feed_dict={xs: x_data,ys:y_data})
	if i % 50 == 0:
		print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

