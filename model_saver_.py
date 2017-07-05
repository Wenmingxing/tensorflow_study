#! usr/bin/python

'''

Coded by luke on 3rd July 2017 
Aiming to get familiar with the model parameter saver

'''

import tensorflow as tf
import numpy as np

## Save to file 
# Remember to define the same dtype and shape iwhen restore
W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

# initialize the variables
init = tf.global_variables_initializer()

# Define a subject for the saving
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	save_path = saver.save(sess,"my_net/save_net.ckpt")
	print("Save to path",save_path)


