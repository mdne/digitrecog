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
		self.X = tf.placeholder("float", [None, 16])
		self.Y = tf.placeholder("float", [None, 10])
		self.w_h = self.init_weights([16, 625]) # create symbolic variables
		self.w_h2 = self.init_weights([625, 625])
		self.w_o = self.init_weights([625, 10])
		self.p_keep_input = tf.placeholder("float")
		self.p_keep_hidden = tf.placeholder("float")
		self.py_x = self.model(self.X, self.w_h, self.w_h2, self.w_o, self.p_keep_input, self.p_keep_hidden)
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.py_x, self.Y)) # compute costs
		self.train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(self.cost)
		self.predict_op = tf.argmax(self.py_x, 1)
		self.sess = None
		self.saver = None

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

	def model(self, X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
		X = tf.nn.dropout(X, p_keep_input)
		h = tf.nn.relu(tf.matmul(X, w_h))
		h = tf.nn.dropout(h, p_keep_hidden)
		h2 = tf.nn.relu(tf.matmul(h, w_h2))
		h2 = tf.nn.dropout(h2, p_keep_hidden)
		return tf.matmul(h2, w_o)
	
	def train(self):
		trX, teX, trY, teY = self.read()
		self.saver = tf.train.Saver()
		self.sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		for i in range(100):
			for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
				self.sess.run(self.train_op, feed_dict={self.X: trX[start:end], self.Y: trY[start:end],
					self.p_keep_input: 0.8, self.p_keep_hidden: 0.5})
			self.global_step.assign(i).eval() # set and update(eval) global_step with index, i
			self.saver.save(self.sess, self.ckpt_dir + "/model.ckpt", global_step=self.global_step)
			print(i, np.mean(np.argmax(teY, axis=1) ==
					self.sess.run(self.predict_op, feed_dict={self.X: teX,
												self.p_keep_input: 1.0,
												self.p_keep_hidden: 1.0})))
	def load(self):
		self.saver = tf.train.Saver()
		self.sess = tf.InteractiveSession()
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			print "load success"
		tf.global_variables_initializer().run()

	def predict(self, digit):
		digit = np.reshape(digit, (-1, 16))
		return self.predict_op.eval(feed_dict={self.X: digit, self.p_keep_input: 1.0,
												self.p_keep_hidden: 1.0}, session=self.sess)[0]

if __name__ == '__main__':
	nnet = NeuralNetwork()
	nnet.train()