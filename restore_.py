#! usr/bin/python

'''
Coded by luke on 3rd July 2017
aiming to restore the model parameters from the saver
'''
import tensorflow as tf
import numpy as np

# First we need to defien the container for the parameters which you want to restore

W = tf.Variable(np.arange(6).reshape(2,3),dtype=tf.float32,name="weights")
b = tf.Variable(np.arange(3).reshape(1,3),dtype=tf.float32,name='biases')

# There is no need for the tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
	# restore the parameters
	saver.restore(sess,'my_net/save_net.ckpt')
	print('weights:',sess.run(W))
	print('biases:',sess.run(b))

