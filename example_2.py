#! usr/bin/python
'''
Coded by Luke on 29th, June, 2017

This program is aiming to familirizz myself to the tensorflow basic sturcture through estimating a linear function parameters. 
'''
import tensorflow as tf
import numpy as np

# Create the data 
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

# Create the weight and bias for the model
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
Biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + Biases

# Calculate the error between the prediction and the expected value
loss = tf.reduce_mean(tf.square(y_data - y))

# Update the weights and biases with the backpropagation method
optimizer = tf.train.GradientDescentOptimizer(0.5) # the learning rate is 0.5
train = optimizer.minimize(loss)

# We just finish the graph for system, in order to use it we need to initialize the variables defined
init = tf.global_variables_initializer() 

# We also need to set a session for the system
sess = tf.Session()
sess.run(init) # This is very important

for step in range(201):
	sess.run(train)
	if step%10 == 0:
		print(step,sess.run(Weights),sess.run(Biases))
 
