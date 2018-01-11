# Creat a dnn model and save it
# no input needed
from numpy import array
from numpy import argmax
import tensorflow as tf

from tensorflow.contrib.tensor_forest.client import *
from tensorflow.contrib.tensor_forest.python import *
from tensorflow.python.ops import resources

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main():
	d = 574 # nbr of features
	ll = 1 # dimension of output
         
	# Create the model
	x = tf.placeholder(tf.float32, [None, d], name = 'x')
	y = tf.placeholder(tf.int32, shape=[None],name ='y_')
	keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
	num_classes  = 2
	num_features = d
	num_trees    = 100
	max_nodes    = 1000
	split_after_samples = 50

	# Random Forest Parameters
	hparams = tensor_forest.ForestHParams(num_classes=num_classes,
	                                      num_features=num_features,
	                                      num_trees=num_trees,
	                                      max_nodes=max_nodes,
	                                      split_after_samples=split_after_samples).fill()
	
	# Build the Random Forest
	#classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(hparams, model_dir="./")
	forest_graph = tensor_forest.RandomForestGraphs(hparams)
	# Get training graph and loss
	train_op = forest_graph.training_graph(x, y)
	loss_op = forest_graph.training_loss(x, y,name = 'loss_optimizer')

	# Measure the accuracy
	infer_op, _, _ = forest_graph.inference_graph(x)
	correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(y, tf.int64),name = 'correct_prediction')
	accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'acc')
	

	# Initialize the variables (i.e. assign their default value) and forest resources
	init_vars = tf.group(tf.global_variables_initializer(),resources.initialize_resources(resources.shared_resources()))

	# Start TensorFlow session
	sess = tf.Session()
	# Run the initializer
	sess.run(init_vars)
	saver = tf.train.Saver()
	saver.save(sess, './rf_graph')
	sess.close()

	return 1

if __name__ == "__main__":
	result = main()

