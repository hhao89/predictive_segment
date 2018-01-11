# Load a dnn model from myM.meta file
# Train and test model using data from file training_set

# Input: 
# 1. files generated by ps_save_graph.py 
# 2. training_set (may need to modify the inputfile path)

from numpy import array
from numpy import argmax
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import time

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
	st_time = time.time()
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
	saver = tf.train.import_meta_graph('rf/rf_graph.meta')
	# saver = tf.train.import_meta_graph('dnn/dnn_graph.meta')
	sess.run(tf.global_variables_initializer())
	graph = tf.get_default_graph()
	acc = graph.get_tensor_by_name('acc:0')
	#softmaxed_logits = graph.get_tensor_by_name('softmaxed_logits:0')
	x = graph.get_tensor_by_name('x:0')
	y_ = graph.get_tensor_by_name('y_:0')
	keep_prob = graph.get_tensor_by_name('keep_prob:0')
	train_step = graph.get_tensor_by_name('loss_optimizer:0')

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
	# calculate acc using test data
	#acc_test, soft_logits_test = sess.run([acc, softmaxed_logits], feed_dict={x: testing_data[0].todense(), y_: testing_data[1], keep_prob: 1.0})
	#sk_auc_test = metrics.roc_auc_score(y_true = np.array(data_test.labels), y_score = np.array(soft_logits_test))
	
	#print ('test accuracy: ' + str(acc_test))
	#print('test sk auc: ' + str(sk_auc_test))
	
	sess.close()

	end_time = time.time()
	print('run time: '+ str(round(end_time-st_time)) + ' seconds')
	return 1

if __name__ == "__main__":
	result = main()

