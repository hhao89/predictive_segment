# Based on clickPredict_v9.py 
# remove onehot encoded features with too many or too less zeros
# on top of v8: 
# 1. adding dropout of input layer, 
# 2. learning rate decay
# 3. ploting loss, accuracy, auc for both training and testing, for comparison purpose 
# on top of v9: 
# plot for every 500 iterations
# based on ps_grid_v2.py, but use different testing set
# based on ps_grid_v3.py, but try to seperate learning and testing 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import array
from numpy import argmax
import tensorflow as tf
import numpy as np
import os
from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import SparseVector, DenseVector
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
#from tensorflow.examples.tutorials.mnist import input_data
# from pyspark.context import SparkContext
# from pyspark.conf import SparkConf
import time

FLAGS = None
W1 = None
W2 = None
class Data:
	"""docstring for Data"""
	def __init__(self, X, labels, labels_sca):
		self.X = X
		self.labels = labels
		self.labels_sca = labels_sca

#def fVisualizefeatures()


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
	global W1, W2
	cid = 000000 # representing ps data
	# filename = 'ps_train.svm'
	# sc = SparkContext("local", "Simple App")
	# filename = 'hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/predseg/models/2017-09-29/ps.51/training_set'
	filename = 'ps_oct/training_set'
	# sc = SparkContext(conf=SparkConf().setAppName("ps_spark_grid")
	# conf = (SparkConf().set('spark.yarn.executor.memoryOverhead', '4096').set('spark.kryoserializer.buffer.max.mb', '2047').set('spark.driver.maxResultSize','2g'))
	conf = (SparkConf().setMaster('local[*]').set('spark.executor.memory', '4G').set('spark.driver.memory', '45G').set('spark.driver.maxResultSize', '10G'))
	sc = SparkContext(conf=conf)
	data = sc.textFile(filename)
	# labels_sca = data.map(lambda x: int(x[0])) # int type
	labels_sca = data.map(lambda line: line.split(',')).map(lambda y:float(y[len(y)-1])) 
	nbr_samples = data.count() 
	# nbr_samples = 10000
	l_sca = np.array(labels_sca.take(nbr_samples))
	#l, _ = fOnehot_encode(labels_sca.take(nbr_samples))
	l = np.column_stack([np.array(l_sca), 1-np.array(l_sca)])

	# features = data.map(lambda x: x.split(' ')).map(lambda y: [int(y[i][-1]) for i in range(902)])
	features = data.map(lambda line: line.split(',')).map(lambda y: [float(y[i]) for i in range( len(y)-1) ])
	X = np.array(features.take(nbr_samples))
	

	nbr_feature = len(X[0])
	print ('nbr of features: ' + str(nbr_feature))


	train_percentage = 0.67
	# data_train, _ = fSplitTrainAndTest(X, l, l_sca, train_percentage)
	data_train, data_test = fSplitTrainAndTest(X, l, l_sca, train_percentage)
	
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

	# calculate acc using test data
	acc_test, soft_logits_test = sess.run([acc, softmaxed_logits], feed_dict={x: data_test.X, y_: data_test.labels, keep_prob: 1.0})
	sk_auc_test = metrics.roc_auc_score(y_true = np.array(data_test.labels), y_score = np.array(soft_logits_test))
	
	print ('test accuracy: ' + str(acc_test))
	print('test sk auc: ' + str(sk_auc_test))
	
	sess.close()
	sc.stop()

	end_time = time.time()
	print('run time: '+ str(round(end_time-st_time)) + ' seconds')
	return 1

if __name__ == "__main__":
	result = main()

