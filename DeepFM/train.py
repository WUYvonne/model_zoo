#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022-03-06 11:28
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com


from public_dataset.data_process import create_criteo_dataset
from DeepFM.model import DeepFM
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
    embedding_size = 10
    w_reg = 1e-4
    v_reg = 1e-4
    hidden_units = [256, 128, 64]
    output_dim = 1
    activation = 'relu'
    learning_rate = 0.01
    embed_dim = 10
    epochs = 5
    batch_size = 32
    feature_columns, (train_X, train_y), (test_X, test_y) = create_criteo_dataset(file_path, test_size=0.2, embed_dim=embed_dim)

    mirrored_strategy = tf.distribute.MirroredStrategy() # 分布式训练, 可指定gpu
    with mirrored_strategy.scope():
        model = DeepFM(feature_columns, embedding_size, w_reg, v_reg, hidden_units, output_dim, activation)
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




