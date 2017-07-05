#! usr/bin/python
'''
Coded by luke on 29th June 2017
Aiming to learn the tf.placeholder
'''

import tensorflow as tf

# Define the placeholder whose default type is float32
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# tf.mul is the matrix multiplication
output = tf.multiply(input1,input2)

with tf.Session() as sess:
	print(sess.run(output,feed_dict={input1: [7.],input2: [2.]}))
