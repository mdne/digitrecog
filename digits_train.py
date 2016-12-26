# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class Dataset():
	def __init__(self):
		self.labelsTr = None
		self.dataTr = None
		self.labelsTes = None
		self.dataTes = None
		self.indexEpoch = 0
		self.epochsCompleted = 0
		self.numExamples = 0
		self.batch_size = 128
		self.test_size = 256

	def labelsToOneHot(self, labels):
		result = np.zeros((labels.shape[0], 10), dtype=np.int)
		result[np.arange(labels.shape[0]), labels] = 1
		return result

	def read(self):
		data = np.loadtxt("pendigits.tra", dtype=int, delimiter=",", unpack=False)
		num_rows = data.shape[0]
		num_cols = data.shape[1]
		self.numExamples = num_rows
		self.labelsTr = data[:, num_cols-1]
		self.labelsTr = self.labelsToOneHot(self.labelsTr)
		self.dataTr = np.loadtxt("pendigits.tra", dtype=int, delimiter=",", usecols = range(0, num_cols-1), unpack=False)
		
		data = np.loadtxt("pendigits.tes", dtype=int, delimiter=",", unpack=False)
		num_rows = data.shape[0]
		num_cols = data.shape[1]
		self.labelsTes = data[:, num_cols-1]
		self.labelsTes = self.labelsToOneHot(self.labelsTes)
		self.dataTes = np.loadtxt("pendigits.tes", dtype=int, delimiter=",", usecols = range(0, num_cols-1), unpack=False)

	def nextBatch(self, batch_size):
		start = self.indexEpoch
		self.indexEpoch += batch_size
		if self.indexEpoch > self.numExamples:
			self.epochsCompleted += 1
			perm = np.arange(self.numExamples)
			np.random.shuffle(perm)
			self.dataTr = self.dataTr[perm]
			self.labelsTr = self.labelsTr[perm]
			start = 0
			self.indexEpoch = batch_size
			assert batch_size <= self.numExamples
		end = self.indexEpoch
		return self.dataTr[start:end], self.labelsTr[start:end]

	def init_weights(self, shape):
		return tf.Variable(tf.random_normal(shape, stddev=0.01))

	def model(self, X, w):
		return tf.matmul(X, w)

	def train(self, testNum):
		x = tf.placeholder(tf.float32, shape=[None, 16])
		y_ = tf.placeholder(tf.float32, shape=[None, 10])
		W = tf.Variable(tf.zeros([16,10]))
		b = tf.Variable(tf.zeros([10]))

		y = tf.matmul(x, W) + b
		y_ = tf.placeholder(tf.float32, [None, 10])
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
		train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		for _ in range(100):
			batch_xs, batch_ys = self.nextBatch(100)
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
		#test
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print(sess.run(accuracy, feed_dict={x: self.dataTes, y_: self.labelsTes}))

		prediction = tf.argmax(y, 1)
		testNum = np.reshape(testNum, (-1, 16))
		print testNum
		print "predictions", prediction.eval(feed_dict={x: testNum}, session=sess)

	def model(self, X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
		l1a = tf.nn.relu(tf.nn.conv2d(X, w,strides=[1, 1, 1, 1], padding='SAME'))
		l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
		l1 = tf.nn.dropout(l1, p_keep_conv)
		l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,strides=[1, 1, 1, 1], padding='SAME'))
		l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
		l2 = tf.nn.dropout(l2, p_keep_conv)
		l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,strides=[1, 1, 1, 1], padding='SAME'))
		l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
		l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
		l3 = tf.nn.dropout(l3, p_keep_conv)

		l4 = tf.nn.relu(tf.matmul(l3, w4))
		l4 = tf.nn.dropout(l4, p_keep_hidden)

		pyx = tf.matmul(l4, w_o)
		return pyx

	def train(self):
		trX = self.dataTr.reshape(-1, 1, 16, 1)
		teX = self.dataTes.reshape(-1, 1, 16, 1)
		trY = self.labelsTr;
		teY = self.labelsTes
		X = tf.placeholder("float", [None, 1, 16, 1])
		Y = tf.placeholder("float", [None, 10])

		w = self.init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
		w2 = self.init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
		w3 = self.init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
		w4 = self.init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
		w_o = self.init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

		p_keep_conv = tf.placeholder("float")
		p_keep_hidden = tf.placeholder("float")
		py_x = self.model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
		train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
		predict_op = tf.argmax(py_x, 1)

		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		for i in range(100):
			training_batch = zip(range(0, len(trX), self.batch_size),
				range(self.batch_size, len(trX)+1, self.batch_size))
			for start, end in training_batch:
				sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
						p_keep_conv: 0.8, p_keep_hidden: 0.5})

			test_indices = np.arange(len(teX))
			np.random.shuffle(test_indices)
			test_indices = test_indices[0:test_size]

			print(i, np.mean(np.argmax(teY[test_indices], axis=1) == 
				sess.run(predict_op,feed_dict={X: teX[test_indices],
					p_keep_conv: 1.0,p_keep_hidden: 1.0})))
	def model1(self, X, w_h, w_o):
		h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
		return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us
	
	def train1(self, testNum):
		trX = self.dataTr
		teX = self.dataTes
		trY = self.labelsTr;
		teY = self.labelsTes
		X = tf.placeholder("float", [None, 16])
		Y = tf.placeholder("float", [None, 10])
		w_h = self.init_weights([16, 625]) # create symbolic variables
		w_o = self.init_weights([625, 10])
		py_x = self.model1(X, w_h, w_o)
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
		train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
		predict_op = tf.argmax(py_x, 1)
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		for i in range(100):
			for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
				sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
			print(i, np.mean(np.argmax(teY, axis=1) ==
					sess.run(predict_op, feed_dict={X: teX})))
		prediction = predict_op
		testNum = np.reshape(testNum, (-1, 16))
		print testNum
		print "predictions", prediction.eval(feed_dict={X: testNum}, session=sess)

if __name__ == '__main__':
	d = Dataset()
	d.read()
	d.train1()