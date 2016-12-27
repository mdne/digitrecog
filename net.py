# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import os

class NeuralNetwork():
	def __init__(self):
		self.ckpt_dir = "./ckpt_dir"
		if not os.path.exists(self.ckpt_dir):
			os.makedirs(self.ckpt_dir)
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		self.saver = tf.train.Saver()
		self.X = tf.placeholder("float", [None, 16])
		self.Y = tf.placeholder("float", [None, 10])
		self.w_h = self.init_weights([16, 625]) # create symbolic variables
		self.w_o = self.init_weights([625, 10])
		self.py_x = self.model(self.X, self.w_h, self.w_o)
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.py_x, self.Y)) # compute costs
		self.train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(self.cost)
		self.predict_op = tf.argmax(self.py_x, 1)
		self.sess = None

	def labelsToOneHot(self, labels):
		result = np.zeros((labels.shape[0], 10), dtype=np.int)
		result[np.arange(labels.shape[0]), labels] = 1
		return result

	def read(self):
		data = np.loadtxt("pendigits.tra", dtype=int, delimiter=",", unpack=False)
		num_rows = data.shape[0]
		num_cols = data.shape[1]
		
		labelsTr = data[:, num_cols-1]
		labelsTr = self.labelsToOneHot(labelsTr)
		dataTr = np.loadtxt("pendigits.tra", dtype=int, delimiter=",", usecols = range(0, num_cols-1), unpack=False)
		
		data = np.loadtxt("pendigits.tes", dtype=int, delimiter=",", unpack=False)
		num_rows = data.shape[0]
		num_cols = data.shape[1]
		labelsTes = data[:, num_cols-1]
		labelsTes = self.labelsToOneHot(labelsTes)
		dataTes = np.loadtxt("pendigits.tes", dtype=int, delimiter=",", usecols = range(0, num_cols-1), unpack=False)
		return dataTr, dataTes, labelsTr, labelsTes

	def init_weights(self, shape):
			return tf.Variable(tf.random_normal(shape, stddev=0.01))

	def model(self, X, w_h, w_o):
		h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
		return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us
	
	def train(self):
		trX, teX, trY, teY = self.read()

		self.sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		for i in range(100):
			for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
				self.sess.run(self.train_op, feed_dict={self.X: trX[start:end], self.Y: trY[start:end]})
			self.global_step.assign(i).eval() # set and update(eval) global_step with index, i
			self.saver.save(self.sess, self.ckpt_dir + "/model.ckpt", global_step=self.global_step)
			print(i, np.mean(np.argmax(teY, axis=1) ==
					self.sess.run(self.predict_op, feed_dict={self.X: teX})))
	def load(self):
		self.sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			print "load success"

	def predict(self, digit):
		digit = np.reshape(digit, (-1, 16))
		return self.predict_op.eval(feed_dict={self.X: digit}, session=self.sess)

if __name__ == '__main__':
	nnet = NeuralNetwork()
	nnet.train()