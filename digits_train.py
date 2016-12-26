# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class Dataset():

	def __init__(self):
		self.labels = None
		self.train_data = None
		self.indexEpoch = 0
		self.epochsCompleted = 0
		self.numExamples = 0

	def labelsToOneHot(self, labels):
		result = np.zeros((labels.shape[0], 10), dtype=np.int)
		result[np.arange(labels.shape[0]), labels] = 1
		return result

	def read(self):
		data = np.loadtxt("pendigits.tra", dtype=int, delimiter=",", unpack=False)
		num_rows = data.shape[0]
		num_cols = data.shape[1]
		self.numExamples = num_rows
		self.labels = data[:, num_cols-1]
		self.labels = self.labelsToOneHot(self.labels)
		self.train_data = np.loadtxt("pendigits.tra", dtype=int, delimiter=",", usecols = range(0, num_cols-1), unpack=False)
		
	def nextBatch(self, batch_size):
		start = self.indexEpoch
		self.indexEpoch += batch_size
		if self.indexEpoch > self.numExamples:
			self.epochsCompleted += 1
			perm = np.arange(self.numExamples)
			np.random.shuffle(perm)
			self.train_data = self.train_data[perm]
			self.labels = self.labels[perm]
			start = 0
			self.indexEpoch = batch_size
			assert batch_size <= self.numExamples
		end = self.indexEpoch
		return self.train_data[start:end], self.labels[start:end]

	def train(self):
		x = tf.placeholder(tf.float32, shape=[None, 16])
		y_ = tf.placeholder(tf.float32, shape=[None, 10])
		W = tf.Variable(tf.zeros([16,10]))
		b = tf.Variable(tf.zeros([10]))

		y = tf.nn.softmax(tf.matmul(x, W) + b)
		y_ = tf.placeholder(tf.float32, [None, 10])
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
		train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

		with tf.Session() as sess:
			tf.initialize_all_variables().run()
			for i in range(1000):
				batch_xs, batch_ys = self.nextBatch(100)
				sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
				correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	    		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	    		print(sess.run(accuracy, feed_dict={x: self.train_data, y_: self.labels}))

	def prediction(self, testNum):

	def load(self):

if __name__ == '__main__':
	d = Dataset()
	d.read()
	d.train()