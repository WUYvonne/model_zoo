#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-02-15 16:48
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class Wide_layer(Layer):
	def __init__(self):
		super(Wide_layer, self).__init__()

	def build(self, input_shape):

		self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
		                         initializer=tf.random_normal_initializer(),
		                         trainable=True,
		                         regularizer=tf.keras.regularizers.l2(1e-4))
		self.w0 = self.add_weight(name='w0', shape=(1,),
		                          initializer=tf.zeros_initializer(),
		                          trainable=True)

	def call(self, inputs, **kwargs):  # dense_inputs + onehot_inputs
		x = tf.matmul(inputs, self.w) + self.w0  # shape: (batchsize, 1)
		return x

class Deep_layer(Layer):
	def __init__(self, hidden_units, output_dim, activation):
		super(Deep_layer,self).__init__()
		self.hidden_layer = [Dense(i, activation=activation) for i in hidden_units]
		self.output_layer = Dense(output_dim, activation=None)

	def call(self, inputs, **kwargs):
		x = inputs
		for layer in self.hidden_layer:
			x = layer(x)
		output = self.output_layer(x)
		return output


