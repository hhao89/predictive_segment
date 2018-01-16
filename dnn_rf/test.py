from numpy import array
from numpy import argmax
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import time
from tensorflow.python.platform import gfile
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing

from tensorflow.contrib.tensor_forest.client import *
from tensorflow.contrib.tensor_forest.python import *
from tensorflow.python.ops import resources


class Data:
	def __init__(self, X, labels, labels_sca):
		self.X = X
		self.labels = labels
		self.labels_sca = labels_sca

def get_data():
    	# data = load_svmlight_file("./train-0.svm")
    	data = load_svmlight_file("../../ps_data/train-0.svm")
    	return data[0], data[1]


def main():

	train_percentage = 0.67
	# training_data = load_svmlight_file("./train-0.svm")
	training_data = load_svmlight_file("../../ps_data/train-0.svm")
	Xr = training_data[0].todense()
	lb = preprocessing.LabelBinarizer()
	yr = lb.fit_transform(training_data[1])
	testing_data = training_data # load_svmlight_file("./testing.svm") 
	n = training_data[0].shape[0]

	sess = tf.Session()

	# load the graph and variables
	# saver = tf.train.import_meta_graph('rf/rf_graph.meta')
	# saver = tf.train.import_meta_graph('dnn/dnn_graph.meta')
	# saver = tf.train.import_meta_graph('dnn_only.graph')
	with gfile.FastGFile("dnn_only.graph",'rb') as f:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(f.read())
	    sess.graph.as_default()
	    tf.import_graph_def(graph_def, name='')

	accuracy = sess.graph.get_tensor_by_name('accuracy:0')
	nbr_features_graph = sess.graph.get_tensor_by_name('nbr_features:0')
	# print sess.run(nbr_features_graph)
	#softmaxed_logits = graph.get_tensor_by_name('softmaxed_logits:0')
	x = sess.graph.get_tensor_by_name('x:0')
	y_ = sess.graph.get_tensor_by_name('y_:0')
	keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
	train_step = sess.graph.get_tensor_by_name('loss_optimizer:0')

	sess.run(tf.global_variables_initializer())
		
	batch_size = 40
	
	for i in range(30):
		
		# train the whole epoch (first shuffle the data)
		idx = np.arange(0, n)
		#np.random.shuffle(idx)
		#X_shuffle = [Xr[k] for k in idx]
		#labels_shuffle = [yr[k] for k in idx]

		for j in range(int(n/batch_size)):
			batch_xs = Xr[j*batch_size: (j+1)*batch_size-1]
			batch_ys = yr[j*batch_size: (j+1)*batch_size-1]	
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

		# finish training, try on testing data
		if i % 10 is 0:     
			print ('epoch: ' + str(i))
	
	save_path = saver.save(sess, "dnn/dnn_model.ckpt")		




	sess.close()
if __name__ == "__main__":
	result = main()
