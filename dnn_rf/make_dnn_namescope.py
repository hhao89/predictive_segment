# Creat a dnn model and save it
# no input needed
from numpy import array
from numpy import argmax
import tensorflow as tf
import numpy as np
import os


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main():

	d = 574 # nbr of features
	ll = 2 # dimension of output

	# Create the model
	nbr_features = tf.constant(d,dtype = tf.int32, name = 'nbr_features')
	nbr_of_layers = 3
	nbr_layer1 = 350
	nbr_layer2 = 350
	epsilon = 1e-3
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, d], name = 'x')
		y_ = tf.placeholder(tf.float32, [None, ll], name = 'y_')
		keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

	# x_drop = x # no dropout on input layer
	with tf.name_scope('layer1'):
		x_drop = tf.nn.dropout(x, keep_prob) # adding dropout in the input layer
		W1 = weight_variable([d, nbr_layer1])
		b1 = bias_variable([nbr_layer1])
		z1 = tf.matmul(x_drop, W1) + b1
		batch_mean1, batch_var1 = tf.nn.moments(z1, [0])
		z1_hat = (z1 - batch_mean1)/tf.sqrt(batch_var1 + epsilon)
		scale1 = tf.Variable(tf.ones([nbr_layer1]))
		beta1 = tf.Variable(tf.zeros([nbr_layer1]))
		h1 = tf.nn.relu(scale1*z1_hat + beta1)
		h1_drop = tf.nn.dropout(h1, keep_prob)
	with tf.name_scope('layer2'):
		W2 = weight_variable([nbr_layer1, nbr_layer2])
		b2 = bias_variable([nbr_layer2])
		z2 = tf.matmul(h1_drop,W2) + b2
		batch_mean2, batch_var2 = tf.nn.moments(z2, [0])
		z2_hat = (z2 - batch_mean2)/tf.sqrt(batch_var2 + epsilon)
		scale2 = tf.Variable(tf.ones([nbr_layer2]))
		beta2 = tf.Variable(tf.zeros([nbr_layer2]))
		h2 = tf.nn.relu(scale2*z2_hat + beta2)
		h2_drop = tf.nn.dropout(h2, keep_prob)
	with tf.name_scope('output'):
		W3 = weight_variable([nbr_layer2, ll])
		b3 = bias_variable([ll])
		y = tf.matmul(h2_drop, W3) + b3

	# Define loss and optimizer
	with tf.name_scope('xcent'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name = 'cross_entropy')
	
	with tf.name_scope('train'):
		starter_learning_rate = 0.05
		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step , decay_steps = 5000, decay_rate = 0.95, staircase=True, name=None)	
		train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy, global_step = global_step, name = 'loss_optimizer')
	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name = 'correct_prediction')
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')	
		softmaxed_logits = tf.nn.softmax(y, name = 'softmaxed_logits')
	# init = tf.global_variables_initializer()
	init_op = tf.global_variables_initializer()
	# start session and save model
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	tf.train.write_graph(sess.graph_def,'.','dnn_only.pb',as_text=False)

	## if save as .meta file ########
	# saver = tf.train.Saver()
	# saver.save(sess, './dnn_graph')
	#################################
	train_writer = tf.summary.FileWriter('./TFlogs_graph', sess.graph)
	sess.close()

	return 1

if __name__ == "__main__":
	result = main()

