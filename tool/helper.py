#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @version: V-17-4-13
    @author: Linlifang
    @file: lstm_api.py
    @time: 17-4-13.下午3:01
"""
import os
import re
import csv
import numpy as np
import pandas as pd

from config.config import *
from tool.logger import logger


def get_embedding(file_path):
    char_id, id_char = load_map(CHAR_ID_PATH)
    with open(file_path, "r") as fp:
        emb_matrix = dict()
        lines = fp.readlines()
        for index, line in enumerate(lines):
            line = line.strip()
            if line:
                seg = line.split('\t')
                if index == 0:
                    emb_dim = int(seg[1])
                    emb_matrix = np.zeros((len(char_id), emb_dim))
                else:
                    char = seg[0]
                    emb_vec = [float(val) for val in seg[1:]]
                    if char in char_id:
                        a = int(char_id[char])
                        emb_matrix[a] = emb_vec
    return emb_matrix


def extract_entity(sentence, labels):
    """
    函数说明: 获取标签对应的字符值
    :param sentence:
    :param labels:
    :return:
    """
    entitys = []
    pattern = re.compile(r'B*M*E|S')
    for kk in pattern.finditer(labels):
        start, end = kk.span()
        entity = sentence[start: end]
        entitys.append(entity)
    return entitys


def next_batch(x, y, start_index, batch_size=128):
    """
    函数说明: 截取句子片段
    :param x: 字符
    :param y:  标签
    :param start_index: 开始索引
    :param batch_size: 大小
    :return: x[start_index: start_index+batch_size]
    """
    last_index = start_index + batch_size
    x_batch = list(x[start_index: min(last_index, len(x))])
    y_batch = list(y[start_index: min(last_index, len(x))])
    if last_index > len(x):
        left_size = last_index - (len(x))
        for i in range(left_size):
            index = np.random.randint(len(x))
            x_batch.append(x[index])
            y_batch.append(y[index])
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


def next_random_batch(x, y, batch_size=128):
    """
    函数说明: 随机选取验证集
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    x_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(x))
        x_batch.append(x[index])
        y_batch.append(y[index])
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


def next_test_batch(x, y, start_index, num_steps=200, batch_size=128):
    """
    函数说明: 截取句子片段
    :param x: 字符
    :param y:  标签
    :param start_index: 开始索引
    :param num_steps: 句子长度
    :param batch_size: 大小
    :return: x[start_index: start_index+batch_size]
    """
    last_index = start_index + batch_size
    x_batch = list(x[start_index: min(last_index, len(x))])
    y_batch = list(y[start_index: min(last_index, len(x))])
    if last_index > len(x):
        left_size = last_index - (len(x))
        for i in range(left_size):
            x_batch.append([0] * num_steps)
            y_batch.append(['x'] * num_steps)
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


def padding(sample, seq_max_len):
    """
    函数说明: add数组长度为seq_max_len
    :param sample:
    :param seq_max_len: 句子长度
    :return:
    """
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
    return sample


def prepare(chars, labels, seq_max_len, is_padding=True):
    """
    函数说明: -1分离句子，取句子长度为200字,转化为数组
    :param chars: 字符-id
    :param labels: 标签-id
    :param seq_max_len: 句子长度,默认200
    :param is_padding: 不足200补0
    :return: 数组
    """
    x = []
    y = []
    tmp_x = []
    tmp_y = []
    for c, l in zip(chars, labels):
        if l == -1:
            if len(tmp_x) <= seq_max_len and tmp_x and tmp_y:
                x.append(tmp_x), y.append(tmp_y)
            tmp_x = []
            tmp_y = []
        else:
            tmp_x.append(c), tmp_y.append(l)
    if is_padding:
        x = np.array(padding(x, seq_max_len))
    else:
        x = np.array(x)
    y = np.array(padding(y, seq_max_len))
    return x, y


def load_map(file_path):
    """
    函数说明: 生成id-字符, 字符-id字典
    :param file_path: 文件路径
    :return: 
    """
    if not os.path.isfile(file_path):
        logger.error("id_char文件不存在,不能生成id_char的map")
        exit()

    token_id = {}
    id_token = {}
    with open(file_path, 'r', encoding='UTF-8') as fp:
        line = fp.readlines()
        for data in line:
            a = data.replace('\n', '').split('\t')
            # print(a)
            token, key_id = [i for i in a[0:2]]
            token_id[token] = key_id
            id_token[key_id] = token
    return token_id, id_token


def save_map(id_char, id_label):
    """
    函数说明: 保存数字-字符、标签
    :param id_char: id, char对应关系
    :param id_label: id, label对应关系
    :return:
    """

    num = 0
    with open(CHAR_ID_PATH, "w", encoding='utf8') as fp:
        for idx, char in id_char.items():
            # if not char.strip():
            #     num = num + 1
            #     continue
            fp.writelines(char + "\t" + str(idx - num) + "\n")
    with open(LABEL_ID_PATH, "w", encoding='utf8') as fp:
        for idx, label in id_label.items():
            fp.writelines(label + "\t" + str(idx) + "\n")


def build_map(train_path):
    """
    函数说明: 字符、标签数字化
    :param train_path: 训练数据路径
    :return: 字典id-char, id-label
    """
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                           names=["char", "label"])

    # 提取字符及标签
    chars = list(set(df_train["char"][df_train["char"].notnull()]))
    labels = list(set(df_train["label"][df_train["label"].notnull()]))

    # 字符标签数字化{'O': 1, 'I':2, 'B': 3, 'E': 4}
    char_id = dict(zip(chars, range(1, len(chars) + 1)))
    label_id = dict(zip(labels, range(1, len(labels) + 1)))

    # 数字-->字符标签
    id_char = dict(zip(range(1, len(chars) + 1), chars))
    id_label = dict(zip(range(1, len(labels) + 1), labels))

    # 开始字符<PAD>
    id_char[0] = "<PAD>"
    id_label[0] = "<PAD>"
    char_id["<PAD>"] = 0
    label_id["<PAD>"] = 0

    # 新字符<NEW>
    id_char[len(chars) + 1] = "<NEW>"
    char_id["<NEW>"] = len(chars) + 1
    save_map(id_char, id_label)
    return char_id, id_char, label_id, id_label


def get_train(train_path, train_val_ratio=0.9, seq_max_len=200):
    """
    函数说明: 获取训练数据
    :param train_path: 训练数据路径
    :param train_val_ratio: 训练集,验证集比例
    :param seq_max_len: 序列长度
    :return:
    """
    char_id, id_char, label_id, id_label = build_map(train_path)
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                           names=["char", "label"])

    # 字符、标签转化为数字, 空字符(句子)转化为-1
    df_train["char_id"] = df_train.char.map(lambda xx: -1 if str(xx) == str(np.nan) else char_id[xx])
    df_train["label_id"] = df_train.label.map(lambda xx: -1 if str(xx) == str(np.nan) else label_id[xx])

    # 转换为数组n*200
    x, y = prepare(df_train["char_id"], df_train["label_id"], seq_max_len)

    # 随机排列句子顺序
    num_samples = len(x)
    index = np.arange(num_samples)
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

    # 训练数据集和验证数据集

    val_num = int(num_samples * train_val_ratio)
    x_train = x[:val_num]
    y_train = y[:val_num]
    x_val = x[val_num:]
    y_val = y[val_num:]
    logger.info('训练集大小:%s, 验证集大小:%s' % (str(len(x_train)), str(len(y_val))))
    num_chars = len(id_char)
    num_labels = len(id_label)
    data = {'train': [x_train, y_train, x_val, y_val], 'token': [char_id, id_char, label_id, id_label],
            'number': [num_chars, num_labels]}
    return data


def get_test(test_path, seq_max_len=200, token_path='token/'):
    """
    函数说明: 获取测试数据
    :param test_path: 测试数据路径
    :param seq_max_len:
    :param token_path:
    :return:
    """
    char_id, id_char = load_map(CHAR_ID_PATH)
    label_id, id_label = load_map(LABEL_ID_PATH)

    def map_func(chars):
        char_list = []
        id_list = []
        if not chars:
            char_list.append(-1)
            id_list.append(-1)
        else:
            char_list.append(chars)
            new_id = char_id['<NEW>']
            id_list.append(char_id.get(chars, new_id))
        return char_list, id_list

    # 字符标签转换为数字
    test_char = []
    test_id = []
    with open(test_path, 'r', encoding='utf-8') as fp:
        fline = fp.readlines()
        for line in fline:
            char_, id_ = map_func(line.strip())
            test_char.extend(char_)
            test_id.extend(id_)
    test_char.append(-1)
    test_id.append(-1)

    # 数字数组化200一句,不足200补0
    x_test_str, x_test = prepare(test_char, test_id, seq_max_len, is_padding=False)
    print("测试数据大小", len(x_test))
    num_chars = len(id_char)
    num_labels = len(id_label)
    data = {'test': [x_test, x_test_str], 'token': [char_id, id_char, label_id, id_label],
            'number': [num_chars, num_labels]}
    return data


def get_transition(y_batch):
    """
    函数说明:
    :param y_batch:
    :return:
    """
    transition_batch = []
    for m in range(len(y_batch)):
        y = [5] + list(y_batch[m]) + [0]
        for t in range(len(y)):
            if y[t] == 0:
                break
            transition_batch.append(y[t] * 6 + y[t + 1])
    transition_batch = np.array(transition_batch)
    return transition_batch
