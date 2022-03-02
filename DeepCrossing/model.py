#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-01-30 21:24
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input
from DeepCrossing.layers import Res_layer

class Deep_Crossing(Model):
	def __init__(self, feature_columns, hidden_units, embed_reg=1e-6):
		super(Deep_Crossing, self).__init__()
		self.dense_feature_columns, self.sparse_feature_columns = feature_columns

		# embedding层， 这里需要一个列表的形式， 因为每个类别特征都需要embedding
		self.embed_layers = {
			'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
										output_dim=feat['embed_dim'],
										input_length=1,
										embeddings_initializer='random_uniform',
										embeddings_regularizer=l2(embed_reg),
										mask_zero=False)
			for i, feat in enumerate(self.sparse_feature_columns)
		}

		embed_layers_len = sum([feat['embed_dim'] for feat in self.sparse_feature_columns]) + len(self.dense_feature_columns)
		self.res_layer = [Res_layer(unit, embed_layers_len) for unit in hidden_units]
		self.output_layer = Dense(1, activation='sigmoid')

	def call(self, inputs):
		dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
		emb = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
							for i in range(sparse_inputs.shape[1])], axis=-1)
		# emb = self.embed_layers(sparse_inputs)
		x = tf.concat([tf.cast(dense_inputs, dtype=tf.float32), emb], axis=-1)

		# x = tf.cast(dense_inputs, dtype=tf.float32)
		# for i in range(sparse_inputs.shape[1]):
		# 	embed_i = self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
		# 	x = tf.concat([x, embed_i], axis=-1)

		for res in self.res_layer:
			r = res(x)
		output = self.output_layer(r)

		return output




