#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-01-30 15:30
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# criteo_sampled_data数据集,单机操作为了速度截取该数据集的3000条样本作为测试，kaggle上下载完成数据集并查看对应介绍
def create_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=1000, test_size=0.2):
	if read_part:
		data_df = pd.read_csv(file, iterator=True)
		data_df = data_df.get_chunk(sample_num)
	else:
		data_df = pd.read_csv(file)

	# dense特征为以'I'开头的特征名字
	# sparse特征为以'C'开头的特征名字
	cols = data_df.columns.to_list()
	dense_feats = ['I' + str(i) for i in range(1, 14)]
	sparse_feats = ['C' + str(i) for i in range(1, 27)]
	feats = dense_feats + sparse_feats


	def process_dense_feats(data, feats):
		d = data.copy()
		d = d[feats].fillna(0.0)
		for feat in feats:
			d[feat] = d[feat].apply(lambda x: np.log(x+1) if x > -1 else -1)

		return d


	def process_spares_feats(data, feats):
		d = data.copy()
		d = d[feats].fillna('-1')
		for feat in feats:
			le = LabelEncoder()
			d[feat] = le.fit_transform(d[feat])

		return d

	data_dense = process_dense_feats(data_df, dense_feats)
	data_sparse = process_spares_feats(data_df, sparse_feats)

	total_data = pd.concat([data_dense, data_sparse], axis=1)
	total_data['label'] = data_df['label']

	def sparseFeature(feat, feat_num, embed_dim=8):
		"""
		create dictionary for sparse feature
		:param feat: feature name
		:param feat_num: the total number of sparse features that do not repeat
		:param embed_dim: embedding dimension
		:return:
		"""
		return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

	def denseFeature(feat):
		"""
		create dictionary for dense feature
		:param feat: dense feature name
		:return:
		"""
		return {'feat_name': feat}

	feature_columns = [[denseFeature(feat) for feat in dense_feats]] + \
	                  [[sparseFeature(feat, int(total_data[feat].max()) + 1, embed_dim=embed_dim) for feat in sparse_feats]]

	train, test = train_test_split(total_data, test_size=test_size)

	train_X = train[feats].values.astype('int32')
	train_y = train['label'].values.astype('int32')
	test_X = test[feats].values.astype('int32')
	test_y = test['label'].values.astype('int32')

	return feature_columns, (train_X, train_y), (test_X, test_y)


