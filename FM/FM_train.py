#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-10-26 11:39
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com

import random
import numpy as np
import pandas as pd
import os
import random
import math
from datetime import datetime
from sklearn import preprocessing

global project_path
project_path = os.getcwd()


# 数据集划分
def loadData(fileName, ratio):
    trainingData = []
    testData = []
    with open(fileName) as txtData:
        lines = txtData.readlines()
        for line in lines:
            lineData = line.strip().split(",")
            if random.random() < ratio:
                trainingData.append(lineData)
            else:
                testData.append(lineData)
            np.savetxt(project_path + "/data/diabetes_train.txt", trainingData, delimiter=',', fmt='%s')
            np.savetxt(project_path + "/data/diabetes_test.txt", testData, delimiter=',', fmt='%s')

    return trainingData, testData

# 预处理数据

def preprocessData(data):
    feature = np.array(data.iloc[:, :-1])
    label = np.array(data.iloc[:, -1].map(lambda x: 1 if x == 1 else -1))
    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(feature), label


def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


# FM模型
def FM(dataMatrix, classLabels, k, iter, alpha):
    '''
    :param dataMatrix: 特征矩阵
    :param classLabels: 标签矩阵
    :param k: 隐向量大小
    :param iter: 迭代次数
    :param alpha: 梯度下降的学习率
    :return: 常数项w_0，一阶特征系数w，二阶交叉特征系数v
    '''
    if isinstance(dataMatrix, np.ndarray):
        pass
    else:
        try:
            dataMatrix = np.array(dataMatrix)
        except:
            raise TypeError("numpy.ndarray required for dataMatrix")

    m, n = dataMatrix.shape

    # 初始化参数
    w = np.zeros((n, 1))   #一阶特征的系数
    w_0 = 0            # 常数项
    v = random.normalvariate(0, 0.2) * np.ones((n, k))

    for it in range(iter):
        for x in range(m):
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
            interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2
            # 预测值 p
            p = w_0 + dataMatrix[x] * w + interaction
            # 计算sigmoid(y*p)
            loss = 1 - (sigmoid(p[0,0] * classLabels[x]))
            # 更新参数
            w_0 = w_0 + alpha * loss * classLabels[x]
            # 更新w和v的参数
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] + alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] + alpha * loss * classLabels[x] * (dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

            if it % 10 == 0:
                loss = getLoss(getPrediction(np.mat(dataMatrix), w_0, w, v), classLabels)
                print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v

# 损失
def getLoss(predict, classLabels):
    m = len(predict)
    loss = 0.0
    for i in range(m):
        loss -= math.log(sigmoid(predict[i] * classLabels[i]))
    return loss

# 预测
def getPrediction(dataMatrix, w_0, w, v):
    if isinstance(dataMatrix, np.ndarray):
        pass
    else:
        try:
            dataMatrix = np.array(dataMatrix)
        except:
            raise TypeError("numpy.ndarray required for dataMatrix")
    m = dataMatrix.shape[0]
    result = []
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
        interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2
        p = w_0 + dataMatrix[x] * w + interaction
        pre = sigmoid(p[0, 0])
        result.append(pre)

    return result

# 评估预测的准确性
def getAccuracy(predict, classLabels):
    m = len(predict)
    allItem = 0
    error = 0
    for i in range(m):
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue

    return float(error/allItem)


if __name__ == '__main__':
    fileName = project_path + '/data/diabetes.txt'
    loadData(fileName, 0.2)

    trainData = project_path + '/data/diabetes_train.txt'
    testData = project_path + '/data/diabetes_test.txt'
    train = pd.read_csv(trainData)
    test = pd.read_csv(testData)
    dataTrain, labelTrain = preprocessData(train)
    dataTest, labelTest = preprocessData(test)
    date_startTrain = datetime.now()

    print("开始训练")
    w_0, w, v = FM(np.mat(dataTrain), labelTrain, 4, 10, 0.01)
    print("w_0:", w_0)
    print("w:", w)
    print("v:", v)
    predict_train_result = getPrediction(np.mat(dataTrain), w_0, w, v)
    print("训练准确性为：%f" % (1 - getAccuracy(predict_train_result, labelTrain)))
    date_endTrain = datetime.now()
    print("训练用时为：%s" % (date_endTrain - date_startTrain))

    print("开始测试")
    predict_test_result = getPrediction(np.mat(dataTest), w_0, w, v)
    print("测试准确性为：%s" % (1 - getAccuracy(predict_test_result, labelTest)))

