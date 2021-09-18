#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021-09-08 19:53
# @Author  : wuyingwen
# @Contact : wuyingwen66@163.com

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

items, users = loadData()
item_df = pd.DataFrame(items).T
user_df = pd.DataFrame(users).T
print(user_df)

# 计算相似用户的矩阵
def cal_similar(df):
    df_x = df.dropna(axis=1, how='any')
    similar_mat = cosine_similarity(df_x)  # 使用余弦相似度计算
    return similar_mat

# 取top_n = 2
top_n = 2
similar_mat = cal_similar(user_df)
print(similar_mat)

# 排序得到前n个相似用户的相似度
def sort_mat(similar_mat, top_n, id_pred):
    user_dict = {}
    m, n = similar_mat.shape
    for i in range(m-1):
        for j in range(i+1, n):
            # 只输出与输入的user_id相关的相似度
            if i == id_pred:
                index_xy = j
                user_dict[index_xy] = similar_mat[i, j]
            if j == id_pred:
                index_xy = i
                user_dict[index_xy] = similar_mat[i, j]
    # sorted_vec = sorted(user_dict.values(), reverse=True)
    sorted_vec = sorted(user_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_vec


# 输入similar_mat, top_n, 以及预测用户user_5的id=4，得到前n个相似用户与目标用户的相似度得分
similarity_users_list = sort_mat(similar_mat, 2, 4)
print(similarity_users_list)

# 根据相似用户计算目标用户对物品的最终得分
# 将top_n相似用户对于item_4,itme_6的评分取出来存进similar_user_score矩阵
# 取出user_5未评分的物品为['item_4', 'item_6']
# pred_item_list = ['item_4', 'item_6']
pred_item_list = user_df.isnull().any()
pred_item_list = pred_item_list[pred_item_list].index.to_list()

user_similarity = [similarity_users_list[i][1] for i in range(len(similarity_users_list))]
similarity_users = [user_df.index.to_list()[x] for x in [similarity_users_list[i][0] for i in range(len(similarity_users_list))]]
similar_user_score = user_df.loc[similarity_users, pred_item_list]

print(user_similarity)
print(similar_user_score)

def cal_pred_value(user_similarity, similar_user_score):
    m, n = similar_user_score.shape
    pred_score = []
    for j in range(n):
        pred_score_i = 0
        sum_user_similarity = 0
        for i in range(m):
            pred_score_i += similar_user_score.iloc[i][j] * user_similarity[i]
            sum_user_similarity += user_similarity[i]
        pred_score_x = pred_score_i / sum_user_similarity
        pred_score.append(np.round(pred_score_x, 3))

    return pred_score

print(cal_pred_value(user_similarity, similar_user_score))











