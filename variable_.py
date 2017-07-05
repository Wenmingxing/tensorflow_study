#! user/bin/python
'''
Coded by luke on 29th June 2017
Aiming to familiarize myself with tf.Variable
'''

import tensorflow as tf

# Define a varible 
state = tf.Variable(0,name='counter')

# Define the constant
one = tf.constant(1)

# Define an add 
new_value = tf.add(state,one)

# Assign the state value with the new_value
update = tf.assign(state,new_value)


# Initialize the variables involved
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))
