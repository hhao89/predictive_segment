# Load a dnn model from myM.meta file 
# Train and test model using data from file training_set
# Read data without using pyspark (only difference compared with ps_load_train_test.py)

# Input: 
# 1. files generated by ps_save_graph.py 
# 2. training_set (may need to modify the inputfile path)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import array
from numpy import argmax
import tensorflow as tf
import numpy as np
import os

import sklearn.metrics as metrics
import time

class Data:
	def __init__(self, X, labels, labels_sca):
		self.X = X
		self.labels = labels
		self.labels_sca = labels_sca


def fSplitTrainAndTest(X, l, l_sca, train_percentage):
	nbr_total = len(X)
	# first shuffle the data
	idx = np.arange(0, nbr_total)
	np.random.shuffle(idx)
	X = [X[k] for k in idx]
	l = [l[k] for k in idx]
	l_sca = [l_sca[k] for k in idx]

	train_nbr = int (round(train_percentage * nbr_total))
	test_nbr = nbr_total - train_nbr
	data_train = Data(X[0:train_nbr - 1], l[0:train_nbr - 1], l_sca[0:train_nbr - 1]) # 75 percent for training 
	data_test = Data(X[train_nbr: nbr_total - 1], l[train_nbr: nbr_total - 1], l_sca[train_nbr: nbr_total - 1])
	return data_train, data_test


def main():
	st_time = time.time()
	filename = '../../ps_data/ps_oct/training_set'
	
	# read input data 
	X = []
	label = []
	with open(filename) as fin:
	    for line in fin:
	        line2 = map( float, line.split(',') )
	        label.append(line2[len(line2) - 1])
	        X.append(line2[0:len(line2) - 1])

	l = np.column_stack([np.array(label), 1-np.array(label)])
	X = np.array(X)


	train_percentage = 0.67
	data_train, data_test = fSplitTrainAndTest(X, l, label, train_percentage)
	n = len(data_train.X)

	sess = tf.InteractiveSession()

	# load the graph and variables
	saver = tf.train.import_meta_graph('myM.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./'))
	graph = tf.get_default_graph()
	acc = graph.get_tensor_by_name('acc:0')
	softmaxed_logits = graph.get_tensor_by_name('softmaxed_logits:0')
	x = graph.get_tensor_by_name('x:0')
	y_ = graph.get_tensor_by_name('y_:0')
	keep_prob = graph.get_tensor_by_name('keep_prob:0')
	train_step = graph.get_tensor_by_name('train_step:0')

	batch_size = 100
	
	for i in range(100):
		
		# train the whole epoch (first shuffle the data)
		idx = np.arange(0, n)
		np.random.shuffle(idx)
		X_shuffle = [data_train.X[k] for k in idx]
		labels_shuffle = [data_train.labels[k] for k in idx]

		for j in range(int(n/batch_size)):
			batch_xs = X_shuffle[j*batch_size: (j+1)*batch_size-1]
			batch_ys = labels_shuffle[j*batch_size: (j+1)*batch_size-1]	
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

		# finish training, try on testing data
		if i % 10 is 0:     
			print ('epoch: ' + str(i))
			
	# calculate acc using test data
	acc_test, soft_logits_test = sess.run([acc, softmaxed_logits], feed_dict={x: data_test.X, y_: data_test.labels, keep_prob: 1.0})
	sk_auc_test = metrics.roc_auc_score(y_true = np.array(data_test.labels), y_score = np.array(soft_logits_test))
	
	print ('test accuracy: ' + str(acc_test))
	print('test sk auc: ' + str(sk_auc_test))
	
	sess.close()

	end_time = time.time()
	print('run time: '+ str(round(end_time-st_time)) + ' seconds')
	return 1

if __name__ == "__main__":
	result = main()

