#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-03-06 11:28
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input
from DeepFM.layers import FM_layer, DNN_layer
import numpy as np


class DeepFM(Model):
	def __init__(self, feature_columns, embedding_size, w_reg, v_reg, hidden_units, output_dim, activation):
		super(DeepFM, self).__init__()

		self.dense_feature_columns, self.sparse_feature_columns = feature_columns

		self.embed_layers = {
			'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
			                             input_length=1,
			                             output_dim=feat['embed_dim'],
			                             embeddings_initializer='random_normal')
			for i, feat in enumerate(self.sparse_feature_columns)
		}

		self.index_mapping = []
		self.feature_length = 0
		for feat in self.sparse_feature_columns:
			self.index_mapping.append(self.feature_length)
			self.feature_length += feat['feat_num']
		self.embed_dim = self.sparse_feature_columns[0]['embed_dim']  # all sparse features have the same embed_dim

		self.FM = FM_layer(embedding_size, w_reg, v_reg)
		self.DNN = DNN_layer(hidden_units, output_dim, activation)
		self.Dense = Dense(1, activation=None)

	def call(self, inputs):
		dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
		# embedding
		sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
		                          for i in range(sparse_inputs.shape[1])], axis=1)

		# fm_input = tf.concat([dense_inputs, sparse_embed], axis=-1)
		fm_input = tf.concat([tf.cast(dense_inputs, dtype=tf.float32), sparse_embed], axis=-1)
		# fm
		fm_output = self.FM(fm_input)
		# deep
		deep_output = self.DNN(sparse_embed)
		deep_output = self.Dense(deep_output)
		output = tf.nn.sigmoid(0.5 * (tf.add(fm_output, deep_output)))
		return output



	#
	#
	# def build_model(self):
	#
	# 	self.feat_index = tf.Variable(dtype=tf.int32, shape=[None, self.field_size], name='feature_index')
	# 	self.feat_value = tf.Variable(dtype=tf.float32, shape=[None, None], name='feature_value')
	# 	self.label = tf.Variable(dtype=tf.float32, shape=[None, 1], name='label')
	#
	# 	# Embedding权值定义，即One-hot编码后的sparse feature到Dense embeddings层的全连接权重，Dense embeddings层的神经元个数由field_size和决定
	# 	self.weight['feature_weight'] = tf.Variable(tf.random_normal(
	# 		[self.feature_sizes, self.embedding_size], mean=0, stddev=0.1), name='feature_weight', dtype=tf.float32)
	#
	# 	# first order
	# 	# FM部分中一次项的权值定义
	# 	self.weight['feature_bias'] = tf.Variable(tf.random_normal(
	# 		[self.feature_sizes, 1], minval=0.0, maxval=1.0), name='feature_bias', dtype=tf.float32)
	#
	# 	# deep部分的weight定义
	# 	num_layer = len(self.deep_layers)
	# 	input_size = self.field_size * self.embedding_size
	# 	init_method = np.sqrt(2.0 / (input_size + self.deep_layers[0]))       # normal: stddev = sqrt(2/(fan_in + fan_out))
	#
	# 	self.weight['layer_0'] = tf.Variable(initial_value=
	# 	                                     tf.random_normal(shape=[input_size, self.deep_layers[0]],
	# 	                                                      mean=0, stddev=init_method), dtype=tf.float32)
	# 	self.weights['bias_0'] = tf.Variable(initial_value=tf.random_normal(shape=[1, self.deep_layers[0]],
	# 	                                                                    mean=0, stddev=init_method), dtype=tf.float32)
	# 	# deep网络中的每层weight 和 bias定义
	# 	for i in range(1, num_layer):
	# 		init_method = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
	# 		self.weights['layer_%d' % i] = tf.Variable(
	# 			initial_value=tf.random_normal(shape=[self.deep_layers[i - 1], self.deep_layers[i]], mean=0, stddev=init_method),
	# 			dtype=tf.float32)
	# 		self.weights['bias_%d' % i] = tf.Variable(
	# 			initial_value=tf.random_normal(shape=[1, self.deep_layers[i]], mean=0, stddev=init_method),
	# 			dtype=tf.float32)
	# 	# deep部分的output_size + 一次项output_size + 二次项output_size
	# 	deep_size = self.deep_layers[-1]
	# 	fm_size = self.field_size + self.embedding_size
	# 	last_layer_size = fm_size + deep_size
	# 	init_method = np.sqrt(np.sqrt(2.0 / (last_layer_size + 1)))
	# 	# 生成最后一层的结果
	# 	self.weights['last_layer'] = tf.Variable(
	# 		initial_value=tf.random_normal(shape=[last_layer_size, 1], mean=0, stddev=init_method),
	# 		dtype=tf.float32)
	# 	self.weights['last_bias'] = tf.Variable(tf.constant(value=0.01), dtype=tf.float32)
	#
	#
	#
	# 	# embedding_part, Sparse Features -> Dense Embedding
	# 	self.embedding_index = tf.nn.embedding_lookup(self.weight['feature_weight'],
	# 	                                              ids=self.feat_index)  # Batch*F*K, [None, field_size, embedding_size]
	# 	# self.feat_value = tf.reshape(tensor=self.feat_value, shape=[-1, self.field_size, 1])  # -1 * field_size * 1
	#
	# 	# 一阶特征，求和 ∑W1i * Xi
	# 	self.first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], ids=self.feat_index)  # [None, field_size, 1]
	# 	self.w_mul_x = tf.multiply(self.y_first_order, tf.reshape(tensor=self.feat_value, shape=[-1, self.field_size, 1]))  # [None, field_size, 1]  Wi * Xi
	# 	self.first_order = tf.reduce_sum(input_tensor=self.w_mul_x, axis=2)  # [None, field_size]
	#
	# 	# 二阶特征
	# 	self.embedding_part = tf.multiply(self.embedding_index,
	# 	                         tf.reshape(tensor=self.feat_value, shape=[-1, self.field_size, 1]))  # [None, field_size, embedding_size] multiply不是矩阵相乘，而是矩阵对应位置相乘。这里应用了broadcast机制。
	# 	self.sum_second_order = tf.reduce_sum(self.embedding_part, 1)
	# 	self.sum_second_order_square = tf.square(self.sum_second_order)
	#
	# 	self.square_second_order = tf.square(self.embedding_part)
	# 	self.square_second_order_sum = tf.reduce_sum(self.square_second_order, 1)
	#
	# 	# 1/2*((a+b)^2 - a^2 - b^2)=ab
	# 	self.second_order = 0.5 * tf.subtract(self.sum_second_order_square, self.square_second_order_sum)
	#
	# 	# ----------- Deep Component ------------
	# 	self.y_deep = tf.reshape(self.embedding_index, shape=[-1, self.field_size * self.embedding_size])  # [None, field_size * embedding_size]
	# 	for i in range(0, len(self.deep_layers)):
	# 		y_deep = tf.add(tf.matmul(self.y_deep, self.weights['layer_%d' % i]), self.weights['bias_%d' % i])
	# 		y_deep = self.deep_activation(y_deep)
	#
	# 	# FM部分的输出
	# 	self.fm_part = tf.concat([self.first_order, self.second_order], axis=1)
	#
	# 	# FM输出与DNN输出拼接
	# 	din_all = tf.concat([self.fm_part, self.y_deep], axis=1)
	# 	self.out = tf.add(tf.matmul(din_all, self.weights['last_layer']), self.weight['last_bias'])
	# 	# print('output:', self.out)
	# 	self.out = tf.nn.sigmoid(self.out)
	#
	# 	# self.loss = -tf.reduce_mean(
	# 	# 	self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))
	#
	# 	self.loss = tf.losses.log_loss(self.label, self.out)
	# 	# 正则：sum(w^2)/2*l2_reg_rate
	# 	# 这边只加了weight，有需要的可以加上bias部分
	# 	self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["last_layer"])
	# 	for i in range(len(self.deep_layers)):
	# 		self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["layer_%d" % i])
	#
	# 	# optimizer
	# 	optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
	# 	                                   epsilon=1e-8).minimize(self.loss)

























































