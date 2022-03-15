#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-03-06 11:28
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Dense

class FM_layer(Model):
	def __init__(self, embedding_size, w_reg, v_reg):
		super(FM_layer, self).__init__()
		self.embedding_size = embedding_size
		self.w_reg = w_reg
		self.v_reg = v_reg


	def build(self, input_shape):
		self.w0 = self.add_weight(name='w0', shape=(1,),
		                          initializer=tf.zeros_initializer(),
		                          trainable=True, )
		# FM部分中一次项的权值定义
		self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
		                         initializer=tf.random_normal_initializer(),
		                         trainable=True,
		                         regularizer=tf.keras.regularizers.l2(self.w_reg))
		# FM部分中二次项的权值定义
		self.v = self.add_weight(name='v', shape=(input_shape[-1], self.embedding_size),
		                         initializer=tf.random_normal_initializer(),
		                         trainable=True,
		                         regularizer=tf.keras.regularizers.l2(self.v_reg))
	def call(self, inputs, **kwargs):
		# # first order
		# y_first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, ids=self.feat_index), axis=1)
		# w_mul_x = tf.multiply(y_first_order, tf.reshape(tensor=self.feat_value,
		#                                                      shape=[-1, self.field_size, 1]))
		# first_order = tf.reduce_sum(input_tensor=w_mul_x, axis=2)  # [None, field_size]
		#
		# first_order = tf.matmul(inputs, self.w) + self.w0  # shape:(batchsize, 1)
		#
		# square_sum = tf.square(tf.reduce_sum(inputs, self.v))  # shape:(batchsize, self.k)
		# sum_square = tf.reduce_sum()  # shape:(batchsize, self.k)
		# second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2, keepdims=True)  # shape:(batchsize, 1)
		#
		# output = first_order + second_order
		# return output

		# first order
		first_order = tf.matmul(inputs, self.w) + self.w0  # shape:(batchsize, 1)

		# second order
		square_sum = tf.pow(tf.matmul(inputs, self.v), 2)   # (batch_size, 1, embed_dim)
		sum_square = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))  # (batch_size, 1, embed_dim)
		second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=-1, keepdims=True)  # (batch_size, 1)

		output = first_order + second_order
		return output

class DNN_layer(Layer):
	def __init__(self, hidden_units, output_dim, activation):
		super().__init__()
		self.hidden_units = hidden_units
		self.output_dim = output_dim
		self.activation = activation

		self.hidden_layer = [Dense(i, activation=self.activation) for i in self.hidden_units]
		self.output_layer = Dense(self.output_dim, activation=None)


	def call(self, inputs):
		x = inputs
		for layer in self.hidden_layer:
			x = layer(x)
		output = self.output_layer(x)
		return output
