#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-01-30 21:06
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com

from data_process import create_criteo_dataset
from model import Deep_Crossing
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import AUC
# from sklearn.metrics import accuracy_score
import os

if __name__ == '__main__':
	file_path = os.getcwd() + '/../public_dataset/criteo_sampled_data_mini.csv'
	hidden_units = [256, 128, 64]
	embed_dim = 32
	learning_rate = 0.01
	batch_size = 32
	epochs = 10

	# feature_columns, train, test = create_criteo_dataset(file=file_path, test_size=0.2, embed_dim=embed_dim)
	#
	# train_X, train_y = train
	# test_X, test_y = test

	feature_columns, (train_X, train_y), (test_X, test_y) = create_criteo_dataset(file_path, test_size=0.2, embed_dim=embed_dim)
	# print(feature_columns)
	model = Deep_Crossing(feature_columns, hidden_units)
	optimizer = optimizers.SGD(0.01)

	# train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
	# train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
	# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	# model.fit(train_dataset, epochs=epochs)
	# logloss, auc = model.evaluate(X_test, y_test)
	# print('logloss {}\nAUC {}'.format(round(logloss, 2), round(auc, 2)))
	#

	mirrored_strategy = tf.distribute.MirroredStrategy() # 分布式训练, 可指定gpu
	with mirrored_strategy.scope():
		model = Deep_Crossing(feature_columns, hidden_units)
		model.build(input_shape=(train_X.shape[0], train_X.shape[1]))
		model.summary()
		model.compile(loss=binary_crossentropy, optimizer=optimizers.SGD(learning_rate=learning_rate),
			              metrics=['accuracy'])


	model.fit(
		train_X,
		train_y,
		epochs=epochs,
		callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
		batch_size=batch_size,
		validation_split=0.2
	)

	print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])





