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


def sort_mats(sim_dict, top_n, target_id_list):
    """
    计算目标用户/物品相似的top_n个用户或物品的相似度，并以字典形式返回
    :param sim_dict: 所有用户/物品相似度字典
    :param top_n: 定义取前n个用户/物品相似度
    :param target_id_list: 目标用户的列表
    :return: target_sim_dict: 目标用户的top_n个相似的用户的相似度字典, 如{'user_5': {'user_2': 0.985, 'user_1': 0.975}}
    """
    target_sim_dict = {}
    for i in target_id_list:
        id_sim_dict, sorted_top_sim_dict = sort_mat(sim_dict, top_n, i)
        target_sim_dict[i] = sorted_top_sim_dict

    return target_sim_dict


def cal_pred_target_value(df, sim_dict, user_id, item_id_list):
    """
    得到最终评分（参照文档中的公式3）
    :param df: 共现矩阵
    :param sim_dict: 目标用户/物品的top_n个相似的用户/物品字典的相似度字典
    :param user_id: 目标用户id
    :return: R: 最终得分
    """
    R = {}
    for item in item_id_list:
        sum_S = 0
        sum_w = 0
        for user_key, user_value in sim_dict[user_id].items():
            sum_S += user_value * df.loc[user_key, item]
            sum_w += user_value
        R[item] = round(sum_S/sum_w,3)

    return R


if __name__ == "__main__":
    items, users = loadData()
    item_df = pd.DataFrame(items).T
    # 构建以用户为行坐标，物品为列坐标的共现矩阵
    user_df = pd.DataFrame(users).T
    print(user_df)
    # 根据共现矩阵，计算用户之间的相似度
    all_sim_dict = cal_similar(user_df, 'user')
    print(all_sim_dict)
    # 得到与目标用户相似的top_n个用户的相似度
    pred_target_sim_dict = sort_mats(all_sim_dict, 2, ['user_5'])
    print(pred_target_sim_dict)
    # # 计算user_5对物品的评分
    R = cal_pred_target_value(user_df, pred_target_sim_dict, 'user_5', ['item_4', 'item_6'])
    print(R)

