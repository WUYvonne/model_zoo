#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-02-03 23:24
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class Res_layer(Layer):
	def __init__(self, hidden_unit, embed_layers_len):
		super(Res_layer, self).__init__()
		self.dense_layer = Dense(hidden_unit, activation='relu')
		self.output_layer = Dense(embed_layers_len, activation=None)

	def call(self, inputs, **kwargs):
		x1 = self.dense_layer(inputs)
		x2 = self.output_layer(x1)
		outputs = tf.nn.relu(x2 + inputs)
		return outputs


