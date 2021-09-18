#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-09-08 19:53
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com

import pandas as pd
import numpy as np
import operator


def loadData():
    items={"item_1": {"user_1": 3, "user_2": 4, "user_3": 3, "user_4": 1, "user_5": 5},
           "item_2": {"user_1": 1, "user_2": 3, "user_3": 3, "user_4": 5, "user_5": 3},
           "item_3": {"user_1": 2, "user_2": 4, "user_3": 1, "user_4": 5, "user_5": 4},
           "item_4": {"user_1": 3, "user_2": 3, "user_3": 5, "user_4": 1},
           "item_5": {"user_1": 3, "user_2": 5, "user_3": 4, "user_4": 1, "user_5": 4},
           "item_6": {"user_1": 3, "user_2": 2, "user_3": 1, "user_4": 5}
          }

    users = {"user_1": {"item_1": 3, "item_2": 1, "item_3": 2, "item_4": 3, "item_5": 3, "item_6": 3},
             "user_2": {"item_1": 4, "item_2": 3, "item_3": 4, "item_4": 3, "item_5": 5, "item_6": 2},
             "user_3": {"item_1": 3, "item_2": 3, "item_3": 1, "item_4": 5, "item_5": 4, "item_6": 1},
             "user_4": {"item_1": 1, "item_2": 5, "item_3": 5, "item_4": 2, "item_5": 1, "item_6": 5},
             "user_5": {"item_1": 5, "item_2": 3, "item_3": 4, "item_5": 4}
             }

    return items, users


def cal_similar(df, cal_type):
    """
    对共现矩阵，计算其余弦相似度
    :param df: 待计算的共现矩阵
    :param cal_type: 待计算的type, 可选对于user或者item计算其向量相似度
    :return: dict形式保存的所有用户/物品相似度字典，{('item_1', 'item_2'): 0.7389884275845571, ('item_1', 'item_3'): 0.7476671794188403, .....}
    """
    if cal_type == 'user':
        df_x = df.loc[:, ~df.isnull().any()]
    if cal_type == 'item':
        df_x = df.dropna(axis=0, how='any').T    # 矩阵按照行来计算各行之间的向量相似度，因此如果计算itemCF这儿需要转置
    m, n = df_x.shape
    all_sim_dict = {}
    for i in range(m):
        for j in range(i+1, m):
            x = df_x.iloc[i].values.tolist()
            y = df_x.iloc[j].values.tolist()
            cos_val = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))  # 计算相似度
            all_sim_dict[cal_type + '_' + str(i+1), cal_type + '_' + str(j+1)] = round(cos_val,3)   # 对应的矩阵中的index,col的输出的, 因此需要+1

    return all_sim_dict


def sort_mat(sim_dict, top_n, target_id_pred):
    """
    根据相似度字典，排序后取出与目标用户/物品相似的top_n个用户或物品
    :param sim_dict: 相似度字典
    :param top_n: 定义取前n个用户/物品相似度
    :param target_id_pred: 有待计算的目标用户/物品
    :return: target_sim_dict: 过滤出与目标用户/物品有关的相似度字典,
             sorted_top_sim_dict: top_n个过滤出与目标用户/物品有关的相似度字典,
             sorted_top_sim_val
    """
    target_sim_dict = {}
    for key, value in sim_dict.items():
        if target_id_pred == key[0]:
            target_sim_dict[key[1]] = value   # 之前计算相似度时是交叉计算的，这边调整顺序
        if target_id_pred == key[1]:
            target_sim_dict[key[0]] = value
    # sorted_vec = sorted(all_pred_vec.items(), key=lambda x: x[1], reverse=True)[:top_n]
    sorted_top_sim_dict = dict(sorted(target_sim_dict.items(), key=operator.itemgetter(1), reverse=True)[:top_n])
    # sorted_top_sim_val = sorted(list(target_sim_dict.values()), reverse=True)[:top_n]

    return target_sim_dict, sorted_top_sim_dict


def sort_mats(df, sim_dict, top_n=2):
    """
    计算目标用户/物品相似的top_n个用户或物品的相似度，并以字典形式返回
    :param df: 共现矩阵
    :param sim_dict: 所有用户/物品相似度字典
    :param top_n: 定义取前n个用户/物品相似度
    :return: target_list: 目标用户/物品的列表, 如['item_4', 'item_6']
             target_sim_dict: 目标用户/物品的top_n个相似的用户/物品的相似度字典, 如{'item_4': {'item_5': 0.94, 'item_1': 0.937}, 'item_6': {'item_3': 0.944, 'item_2': 0.893}}
    """

    # 得到共现矩阵中未评分的物品的id
    target_list = df.isnull().any()
    target_list = target_list[target_list].index.to_list()

    # 输入所有用户/物品相似度字典sim_dict, top_n, 以及目标用户/物品的id，得到前n个相似物品的相似度
    # 这里得到与item_4最相近的两个物品为item_5和item_1,
    # {'item_4': {'item_5': 0.94, 'item_1': 0.937}, 'item_6': {'item_3': 0.944, 'item_2': 0.893}}
    target_sim_dict = {}
    for i in target_list:
        id_sim_dict, sorted_top_sim_dict = sort_mat(sim_dict, top_n, i)
        target_sim_dict[i] = sorted_top_sim_dict

    return target_list, target_sim_dict


def cal_pred_target_value(df, sim_dict, user_id):
    """
    得到最终评分（注意这里区别于userCF中的计算方式, 具体参照文档中的公式5,
    考虑到用户的评分标准不一的情况, 这里针对该物品的评分与此用户的所有评分的差值进行加权平均做了处理）
    :param df: 共现矩阵
    :param sim_dict: 目标用户/物品的top_n个相似的用户/物品字典的相似度字典
    :param user_id: 目标用户id
    :return:
    """
    R = {}
    for key_i, value_i in sim_dict.items():
        avg_R_p = np.nanmean(df[key_i])
        sum_S = 0
        sum_w = 0
        for key_j, value_j in value_i.items():
            sum_S += value_j * (df.loc[user_id, key_j] - np.nanmean(df[key_j]))
            sum_w += value_j
        R[key_i] = round(avg_R_p + sum_S / sum_w, 2)

    return R


if __name__ == "__main__":
    items, users = loadData()
    item_df = pd.DataFrame(items).T
    # 构建以用户为行坐标，物品为列坐标的共现矩阵
    user_df = pd.DataFrame(users).T
    print(user_df)
    # 根据共现矩阵，计算物品之间的相似度
    all_sim_dict = cal_similar(user_df, 'item')
    print(all_sim_dict)
    # 得到与目标物品相似的top_n个物品的相似度
    pred_target_item_list, pred_target_sim_dict = sort_mats(user_df, all_sim_dict, 2)
    # 计算user_5对物品的评分
    R = cal_pred_target_value(user_df, pred_target_sim_dict, 'user_5')
    print(R)

