# use sparsetensor to handle sparse vector input
# 
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
import time
from scipy.sparse import coo_matrix

class Data:
	"""docstring for Data"""
	def __init__(self, X, labels, labels_sca):
		self.X = X
		self.labels = labels

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  # initial = tf.ones(shape, dtype = tf.float32)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


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

def libsvm_2_coo(libsvm_data, shape):
    coo_rows = []
    coo_cols = []
    coo_data = []
    n = 0
    for x, d in libsvm_data:
        coo_rows.extend([n] * len(x))
        coo_cols.extend(x)
        coo_data.extend(d)
        n += 1
    coo_rows = np.array(coo_rows)
    coo_cols = np.array(coo_cols)
    coo_data = np.array(coo_data)
    return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)



def fReadData(new_data_test, nbr_features):

	nbr_samples_test = new_data_test.count()
	# nbr_samples_test = 10000
	data2 = new_data_test.map(lambda line:line.split('\t')).map(lambda x:x[1])
	labels = np.array(data2.map(lambda x: float(x[0])).take(nbr_samples_test))
	labels = np.column_stack([labels, 1 - labels]) # labels as np array with shape (None, 2)
	feature_str = data2.map(lambda x: x[2:])
	t2 = feature_str.map(lambda lines: lines.split(' '))

	col = t2.map(lambda x: [int(i.split(':')[0]) for i in x] )
	X = col.take(nbr_samples_test) 
	d = t2.map(lambda x: [float(i.split(':')[1]) for i in x] )
	D = d.take(nbr_samples_test)

	fea = libsvm_2_coo(zip(X, D), (len(X), nbr_features)).tocsr()
	return fea, labels


# def fReadTrainingData(new_data_test, nbr_features):

# 	nbr_samples_test = new_data_test.count()
# 	data2 = new_data_test.map(lambda line:line.split('\t')).map(lambda x:x[1])
# 	labels = np.array(data2.map(lambda x: float(x[0])).take(nbr_samples_test))
# 	labels = np.column_stack([labels, 1 - labels]) # labels as np array with shape (None, 2)
# 	feature_str = data2.map(lambda x: x[2:])
# 	t2 = feature_str.map(lambda lines: lines.split(' '))

# 	col = t2.map(lambda x: [int(i.split(':')[0]) for i in x] )
# 	X = col.take(nbr_samples_test) 
# 	d = t2.map(lambda x: [float(i.split(':')[1]) for i in x] )
# 	D = d.take(nbr_samples_test)

# 	fea = libsvm_2_coo(zip(X, D), (len(X), nbr_features)).tocsr()
# 	return fea, labels

def csr_2_input(csr_mat):
    if not isinstance(csr_mat, list):
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
        values = csr_mat.data
        shape = csr_mat.shape
        return indices, values, shape
    else:
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr_2_input(csr_i))
        return inputs


def slice(csr_data, start=0, size=-1):
    if not isinstance(csr_data[0], list):
        if size == -1 or start + size >= csr_data[0].shape[0]:
            slc_data = csr_data[0][start:]
            slc_labels = csr_data[1][start:]
        else:
            slc_data = csr_data[0][start:start + size]
            slc_labels = csr_data[1][start:start + size]
    else:
        if size == -1 or start + size >= csr_data[0][0].shape[0]:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:])
            slc_labels = csr_data[1][start:]
        else:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:start + size])
            slc_labels = csr_data[1][start:start + size]
    return csr_2_input(slc_data), slc_labels


def fSplit(data, train_percentage):
	nbr_total = data[0].shape[0]
	train_nbr = int (round(train_percentage * nbr_total))
	test_nbr = nbr_total - train_nbr

	data_train = data[0][0:train_nbr], data[1][0:train_nbr]
	data_test = data[0][train_nbr: nbr_total], data[1][train_nbr: nbr_total]

	return data_train, data_test

def shuffle(data):
    X, y = data
    ind = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(ind)
    return X[ind], y[ind]


def main():
	st_time = time.time()

	# filename = 'ps_train.svm'
	# sc = SparkContext("local", "Simple App")
	# filename = 'hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/predseg/models/2017-09-29/ps.51/training_set'
	conf = (SparkConf().setMaster('local[*]').set('spark.executor.memory', '4G').set('spark.driver.memory', '45G').set('spark.driver.maxResultSize', '10G'))
	sc = SparkContext(conf=conf)
	if False:
		filename = '../../ps_data/ps_oct/training_set'
		# sc = SparkContext(conf=SparkConf().setAppName("ps_spark_grid")
		# conf = (SparkConf().set('spark.yarn.executor.memoryOverhead', '4096').set('spark.kryoserializer.buffer.max.mb', '2047').set('spark.driver.maxResultSize','2g'))
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




	##### uncomment this if try using another testing set

	# filename_test_new = 'hdfs://jetblue-nn1.blue.ygrid.yahoo.com:8020/projects/predseg/xg/test_data/2017-09-20/ps.51/part-r-01088'
	filename = '../../ps_data/part-r-01088'
	# filename = '../ps_data/ps_oct/training_set'
	nbr_features = 600
	new_data_test = sc.textFile(filename)
	data = fReadData(new_data_test, nbr_features) # data is a tuple where data[0] is the features, and data[1] is the labels
	# data = fReadTraininngData(new_data_test, nbr_features)
	train_percentage = 0.67
	data_train, data_test = fSplit(data, train_percentage) # data_train, data_test are tuples where [0] is the features, and [1] is the labels

	n = data_train[0].shape[0] # total number of training samples
	d = nbr_features # number of features 
	ll = 2 #output dimension

	# Create the model
	x = tf.sparse_placeholder(tf.float32, [None, d])
	keep_prob = tf.placeholder(tf.float32)


	nbr_of_layers = 2
	nbr_layer1 = 250
	nbr_layer2 = 50
	epsilon = 1e-3

	# x_drop = tf.nn.dropout(x, keep_prob) # adding dropout in the input layer
	x_drop = x # no dropout on input layer
	W1 = weight_variable([d, nbr_layer1])
	b1 = bias_variable([nbr_layer1])
	z1 = tf.sparse_tensor_dense_matmul(x_drop, W1) + b1
	batch_mean1, batch_var1 = tf.nn.moments(z1, [0])
	z1_hat = (z1 - batch_mean1)/tf.sqrt(batch_var1 + epsilon)
	scale1 = tf.Variable(tf.ones([nbr_layer1]))
	beta1 = tf.Variable(tf.zeros([nbr_layer1]))
	#b1 = bias_variable([nbr_layer1])
	h1 = tf.nn.relu(scale1*z1_hat + beta1)
	h1_drop = tf.nn.dropout(h1, keep_prob)
	if nbr_of_layers == 2:
		
		W2 = weight_variable([nbr_layer1, ll])
		b2 = bias_variable([ll])
		y = tf.matmul(h1_drop,W2) + b2
	#h1 = tf.nn.sigmoid(scale1*z1_hat + beta1)
	else:
		W2 = weight_variable([nbr_layer1, nbr_layer2])
		b2 = bias_variable([nbr_layer2])
		z2 = tf.matmul(h1_drop,W2) + b2
		batch_mean2, batch_var2 = tf.nn.moments(z2, [0])
		z2_hat = (z2 - batch_mean2)/tf.sqrt(batch_var2 + epsilon)
		scale2 = tf.Variable(tf.ones([nbr_layer2]))
		beta2 = tf.Variable(tf.zeros([nbr_layer2]))
		h2 = tf.nn.relu(scale2*z2_hat + beta2)
		h2_drop = tf.nn.dropout(h2, keep_prob)
		#h2 = tf.nn.sigmoid(scale2*z2_hat + beta2)

		W3 = weight_variable([nbr_layer2, ll])
		b3 = bias_variable([ll])

		y = tf.matmul(h2_drop, W3) + b3

	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, ll])

	tf.summary.histogram('W1',W1)
	tf.summary.histogram('W2',W2)
	cross_entropy = tf.reduce_mean(
	  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	starter_learning_rate = 0.01
	global_step = tf.Variable(0, trainable=False)
	# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step , decay_steps = 50000, decay_rate = 0.98, staircase=True, name=None)
	# train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy, global_step = global_step)
	
	train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy, global_step = global_step)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	
	auc_ftrain = tf.metrics.auc(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(tf.argmax(y_, 1), tf.float32))
	auc_ftest = tf.metrics.auc(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(tf.argmax(y_, 1), tf.float32))
	softmaxed_logits = tf.nn.softmax(y)



	tf.local_variables_initializer().run()
	sess.run(tf.initialize_local_variables())
	tf.summary.scalar('cross_entropy', cross_entropy)
	tf.summary.scalar('accuracy', accuracy)
	tf.summary.scalar('auc_ftrain', auc_ftrain[0])
	tf.summary.scalar('auc_ftest', auc_ftest[0])

	train_writer = tf.summary.FileWriter("/tmp/histogram_example/train", sess.graph)
	test_writer = tf.summary.FileWriter("/tmp/histogram_example/test")

	# writer = tf.summary.FileWriter("/tmp/histogram_example")
	summaries = tf.summary.merge_all()
	# save 
	st = np.array([])
	
	ac_train = np.array([])
	ca_train = np.array([])
	auc_train = np.array([])

	ac_test = np.array([])
	ca_test = np.array([])
	auc_test = np.array([])

	batch_size = 100
	data_test_input_features = csr_2_input(data_test[0]) # transform csr to input form that is a tuple of (index, value, shape)
	data_train_input_features = csr_2_input(data_train[0])
	for i in range(100):
		data_train = shuffle(data_train)

		for j in range(int(n/batch_size)):

			miniBatch = slice(data_train, j * batch_size, batch_size)
			sess.run(train_step, feed_dict={x: tf.SparseTensorValue(miniBatch[0][0], miniBatch[0][1],miniBatch[0][2]), y_ : miniBatch[1], keep_prob: 0.5})
			# batch_xs = X_shuffle[j*batch_size: (j+1)*batch_size-1]
			# batch_ys = labels_shuffle[j*batch_size: (j+1)*batch_size-1]	
			# sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

		# finish training, try on testing data
		if i % 10 is 0:     
			print (i)
			
			soft_logits_train, summary_train, ca_train_i, ac_train_i, auc_train_i = sess.run(
				[softmaxed_logits, summaries, cross_entropy, accuracy, auc_ftrain],
				feed_dict={x: tf.SparseTensorValue(data_train_input_features[0], data_train_input_features[1],data_train_input_features[2]), y_ : data_train[1], keep_prob: 1.0})

			soft_logits_test, summary_test, ca_test_i, ac_test_i,auc_test_i = sess.run(
				[softmaxed_logits, summaries, cross_entropy, accuracy, auc_ftest],
				feed_dict={x: tf.SparseTensorValue(data_test_input_features[0], data_test_input_features[1],data_test_input_features[2]), y_ : data_test[1], keep_prob: 1.0})

			#train_writer.add_summary(summary_train, i)
			#test_writer.add_summary(summary_test, i)
			
			# print (soft_logits_train)
			# print (data_train.labels)
			sk_auc_train = metrics.roc_auc_score(y_true = data_train[1], y_score = np.array(soft_logits_train))
			sk_auc_test = metrics.roc_auc_score(y_true = data_test[1], y_score = np.array(soft_logits_test))													
			print ('learning rate: ' + str(sess.run(learning_rate)))

			print ('train cross entropy: ' + str(ca_train_i))
			print ('test cross entropy: ' + str(ca_test_i))
			
			print ('train accuracy: ' + str(ac_train_i))
			print ('test accuracy: ' + str(ac_test_i))

			# print ('train auc: ' + str(auc_train_i[0]))
			# print ('test auc: '+ str(auc_test_i[0]))


			print('train sk auc: ' + str(sk_auc_train))
			print('test sk auc: ' + str(sk_auc_test))
			# print ('train auc sk' + str(auc_sk_train))
			# print ('test auc sk' + str(auc_sk_test))

		# ca_test, ac_test, auc_test = sess.run([cross_entropy, accuracy, auc], feed_dict={x: data_test.X, y_: data_test.labels, keep_prob: 1.0})
		# print ('test cross entropy: ' + str(ca_test))
		# print ('test accuracy: ' + str(ac_test))
		# print ('test auc: '+ str(auc_test[0]))
	sess.close()
	sc.stop()

	end_time = time.time()
	print('run time: '+ str(round(end_time-st_time)) + ' seconds')
	print('tensorboard --logdir=/tmp/histogram_example')
	return 1

if __name__ == "__main__":
	result = main()

