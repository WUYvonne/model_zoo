#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-02-15 16:45
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.regularizers import l2
from Wide_Deep.layers import Wide_layer, Deep_layer

class Wide_Deep(Model):
	def __init__(self, feature_columns, hidden_units, output_dim, activation, embed_reg=1e-6):
		super(Wide_Deep, self).__init__()
		self.dense_feature_columns, self.sparse_feature_columns = feature_columns
		self.embed_layers = {
			'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
			                             output_dim=feat['embed_dim'],
			                             input_length=1,
			                             embeddings_initializer='random_uniform',
			                             embeddings_regularizer=l2(embed_reg),
			                             mask_zero=False)
			for i, feat in enumerate(self.sparse_feature_columns)
		}
		self.wide_layer = Wide_layer()
		self.deep_layer = Deep_layer(hidden_units, output_dim, activation)


	def call(self, inputs):
		# onehot_inputs为wide部分的输入
		dense_inputs, sparse_inputs, onehot_inputs = inputs[:, :13], inputs[:, 13:39], inputs[:, 39:]
		"""
		    Wide部分使用了规范化后的连续特征、离散特征（也可以交叉特征）
		"""
		# wide_input = tf.concat([dense_inputs, onehot_inputs], axis=-1)
		wide_input = tf.concat([tf.cast(dense_inputs, dtype=tf.float32), tf.cast(onehot_inputs, dtype=tf.float32)], axis=-1)
		wide_output = self.wide_layer(wide_input)

		"""
			deep部分使用embedding后的特征
		"""
		sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
							for i in range(sparse_inputs.shape[1])], axis=-1)
		deep_output = self.deep_layer(sparse_embed)

		output = tf.nn.sigmoid(0.5 * (wide_output + deep_output))
		return output




